"""Frozen-neighbor mining and coverage-aware mixed sampling (NumPy only)."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def add_balanced_sampler_arguments(parser):
    parser.add_argument("--rarity-power", "--rarity_power", type=float, default=0.5,
                        help="oversampling strength: 0 preserves counts; 1 equalizes PID quotas")
    parser.add_argument("--sampler-exposure", choices=["coverage", "oversample"],
                        default="coverage", help="cover all rows once, or add rare-PID repeats")
    parser.add_argument("--sampler-mining-cache", default="",
                        help="frozen negative-neighbor NPZ exported by sampler_feature_audit.py")
    parser.add_argument("--hard-negative-fraction", type=float, default=0.25)
    parser.add_argument("--semi-hard-negative-fraction", type=float, default=0.25)
    parser.add_argument("--hard-top-k", type=int, default=10,
                        help="hard rank band; subsequent cached ranks are semi-hard")


def balanced_sampler_kwargs(args, negative_index=None):
    return dict(rarity_power=getattr(args, "rarity_power", 0.5),
                exposure_mode=getattr(args, "sampler_exposure", "coverage"),
                negative_index=negative_index,
                hard_fraction=getattr(args, "hard_negative_fraction", 0.25),
                semi_hard_fraction=getattr(args, "semi_hard_negative_fraction", 0.25),
                hard_top_k=getattr(args, "hard_top_k", 10),
                seed=getattr(args, "seed", 1))


def row_fingerprint(rows):
    digest = hashlib.sha256()
    for pid, image_id, image_path, caption in rows:
        filename = str(image_path).replace("\\", "/").split("/")[-1]
        payload = json.dumps([int(pid), int(image_id), filename, caption],
                             ensure_ascii=False).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


class NegativeNeighborIndex:
    """Row-index neighbors for image-to-text, text-to-image, and text-to-text."""

    def __init__(self, rows, neighbors, checkpoint=""):
        values = np.asarray(neighbors)
        if (values.ndim != 3 or values.shape[:2] != (len(rows), 3)
                or values.shape[2] < 2 or values.dtype.kind not in "iu"):
            raise ValueError("neighbors must be integer [rows, 3, ranks>=2]")
        if np.any(values < -1) or np.any(values >= len(rows)):
            raise ValueError("negative neighbor index is outside the dataset")
        self.neighbors = values.astype(np.int32, copy=True)
        self.fingerprint = row_fingerprint(rows)
        self.checkpoint = str(checkpoint)
        pids = np.asarray([row[0] for row in rows])
        for direction in range(3):
            candidates = self.neighbors[:, direction]
            valid = candidates >= 0
            if np.any(valid & (pids[np.maximum(candidates, 0)] == pids[:, None])):
                raise ValueError("negative neighbors must have a different PID")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez_compressed(handle, neighbors=self.neighbors,
                                fingerprint=np.asarray(self.fingerprint),
                                checkpoint=np.asarray(self.checkpoint), version=np.asarray(1))

    @classmethod
    def load(cls, path, rows):
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != 1:
                raise ValueError("unsupported mining cache version")
            if str(data["fingerprint"]) != row_fingerprint(rows):
                raise ValueError("mining cache does not match dataset rows/order")
            return cls(rows, data["neighbors"], str(data["checkpoint"]))


def negative_index_from_topk(rows, global_neighbors, row_image_positions, checkpoint):
    width = min(indices.shape[1] for indices, _ in global_neighbors)
    if width < 2:
        raise ValueError("at least two global negative ranks are required for mining")
    image_to_row = {}
    for row, image in enumerate(row_image_positions):
        image_to_row.setdefault(int(image), row)
    directions = []
    for direction, (indices, scores) in enumerate(global_neighbors):
        selected = indices[:, :width].copy()
        valid = np.isfinite(scores[:, :width])
        if direction == 1:
            selected = np.asarray([
                [image_to_row[int(image)] if ok else -1
                 for image, ok in zip(candidates, mask)]
                for candidates, mask in zip(selected, valid)
            ], dtype=np.int32)
        selected[~valid] = -1
        directions.append(selected)
    return NegativeNeighborIndex(rows, np.stack(directions, axis=1), checkpoint)


class BalancedMixedIndexSampler:
    """Keep all rows; optionally oversample rare PIDs and mine frozen neighbors.

    Coverage mode uses every source row exactly once. Oversample mode sets
    PID quota ceil(max_count**alpha * count**(1-alpha)), never discarding
    majority rows. Alpha=1 equalizes PID quotas; alpha=0 preserves counts.
    Batch constraints are relaxed only when remaining rows cannot satisfy
    them; diagnostics expose those cases instead of dropping the tail.
    """

    def __init__(self, data_source, batch_size, positive_pairs_per_batch,
                 rarity_power=0.5, exposure_mode="coverage", negative_index=None,
                 hard_fraction=0.25, semi_hard_fraction=0.25, hard_top_k=10, seed=1):
        if batch_size < 3 or positive_pairs_per_batch < 1:
            raise ValueError("batch_size >= 3 and positive_pairs_per_batch >= 1 required")
        if 2 * positive_pairs_per_batch >= batch_size:
            raise ValueError("positive pairs must leave singleton slots")
        if not 0 <= rarity_power <= 1 or exposure_mode not in ("coverage", "oversample"):
            raise ValueError("rarity_power must be in [0,1]; invalid exposure_mode")
        if (not np.isfinite(hard_fraction) or not np.isfinite(semi_hard_fraction)
                or min(hard_fraction, semi_hard_fraction) < 0
                or hard_fraction + semi_hard_fraction > 1):
            raise ValueError("mining fractions must be nonnegative and sum to <= 1")
        if hard_top_k < 1:
            raise ValueError("hard_top_k must be positive")
        if not data_source:
            raise ValueError("empty dataset")
        self.rows = data_source
        self.batch_size = batch_size
        self.positive_pairs_per_batch = positive_pairs_per_batch
        self.rarity_power = rarity_power
        self.exposure_mode = exposure_mode
        self.hard_fraction = hard_fraction
        self.semi_hard_fraction = semi_hard_fraction
        self.hard_top_k = hard_top_k
        self.seed, self.epoch = seed, 0
        self.negative_index = negative_index
        if negative_index is not None:
            if negative_index.fingerprint != row_fingerprint(data_source):
                raise ValueError("negative index does not match dataset")
            if hard_top_k >= negative_index.neighbors.shape[2]:
                raise ValueError("hard_top_k must leave a semi-hard rank band")
        self.pids = list(dict.fromkeys(row[0] for row in data_source))
        if len(self.pids) < 2:
            raise ValueError("at least two PIDs are required")
        self.pid_to_pos = {pid: pos for pos, pid in enumerate(self.pids)}
        self.row_pids = np.asarray([self.pid_to_pos[row[0]] for row in data_source])
        self.row_images = [row[1] for row in data_source]
        self.image_rows = [defaultdict(list) for _ in self.pids]
        for row, pid in enumerate(self.row_pids):
            self.image_rows[pid][self.row_images[row]].append(row)
        self.source_counts = np.bincount(self.row_pids, minlength=len(self.pids))
        self.target_counts = self.source_counts.copy()
        if exposure_mode == "oversample":
            maximum = int(self.source_counts.max())
            self.target_counts = np.maximum(self.source_counts, np.ceil(
                self.source_counts * (maximum / self.source_counts) ** rarity_power
            ).astype(np.int64))
        self.length = int(self.target_counts.sum())
        self.diagnostics = {}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.length

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        remaining = np.ones(len(self.rows), dtype=np.int64)
        for pid, target in enumerate(self.target_counts):
            row_ids = np.flatnonzero(self.row_pids == pid)
            extra = int(target - len(row_ids))
            while extra:
                chosen = rng.permutation(row_ids)[:extra]
                remaining[chosen] += 1
                extra -= len(chosen)
        pid_remaining = self.target_counts.copy()
        image_remaining = [{image: int(remaining[indices].sum())
                            for image, indices in groups.items()}
                           for groups in self.image_rows]
        available_image_counts = np.asarray([len(groups) for groups in self.image_rows])
        stats = {"version": 2, "exposure_mode": self.exposure_mode,
                 "rarity_power": self.rarity_power,
                 "mining_enabled": self.negative_index is not None,
                 "hard_requested": 0, "hard_draws": 0,
                 "semi_hard_requested": 0, "semi_hard_draws": 0,
                 "mining_fallback_draws": 0, "relaxed_batches": 0,
                 "pair_shortfall_batches": 0, "same_image_fallback_draws": 0,
                 "tail_repair_swaps": 0, "unresolved_same_image_positive_pairs": 0,
                 "batches_without_negative_pid": 0}
        stats["planned_extra_draws"] = self.length - len(self.rows)
        stats["target_pid_exposure_min"] = int(self.target_counts.min())
        stats["target_pid_exposure_max"] = int(self.target_counts.max())
        output = []
        image_fallback_batches = []

        def images_available(pid):
            return [image for image, count in image_remaining[pid].items() if count > 0]

        def pick_pid(pool):
            weights = pid_remaining[pool].astype(np.float64)
            return int(rng.choice(pool, p=weights / weights.sum()))

        def draw(pid, image=None):
            if image is None:
                available_images = images_available(pid)
                image = available_images[rng.integers(len(available_images))]
            indices = self.image_rows[pid][image]
            available = [index for index in indices if remaining[index] > 0]
            return int(rng.choice(available))

        def consume(index, batch):
            remaining[index] -= 1
            pid, image = self.row_pids[index], self.row_images[index]
            pid_remaining[pid] -= 1
            image_remaining[pid][image] -= 1
            if image_remaining[pid][image] == 0:
                available_image_counts[pid] -= 1
            batch.append(index)

        def mine(batch, used_pids, mode):
            shuffled = rng.permutation(batch).tolist()
            queries = ([query for query in shuffled if query not in mined_queries]
                       + [query for query in shuffled if query in mined_queries])
            # Rotate directions and queries, avoiding a single dominant modality.
            for query in queries:
                for direction in rng.permutation(3):
                    ranked = self.negative_index.neighbors[query, direction]
                    ranked = (ranked[:self.hard_top_k] if mode == "hard"
                              else ranked[self.hard_top_k:])
                    for candidate in rng.permutation(ranked):
                        if candidate < 0 or self.row_pids[candidate] in used_pids:
                            continue
                        if remaining[candidate] > 0:
                            mined_queries.add(query)
                            return int(candidate)
                        # Text-to-image neighbors index one representative caption.
                        if direction == 1:
                            siblings = self.image_rows[self.row_pids[candidate]][
                                self.row_images[candidate]]
                            available = [i for i in siblings if remaining[i] > 0]
                            if available:
                                mined_queries.add(query)
                                return int(rng.choice(available))
            return None

        while len(output) < self.length:
            fallback_before = stats["same_image_fallback_draws"]
            capacity = min(self.batch_size, self.length - len(output))
            batch, used_pids = [], set()
            mined_queries = set()
            used_mask = np.zeros(len(self.pids), dtype=bool)
            pair_count = min(self.positive_pairs_per_batch, max(0, (capacity - 1) // 2))
            actual_pairs = 0
            for _ in range(pair_count):
                pool = np.flatnonzero((available_image_counts >= 2) & ~used_mask)
                if not len(pool):
                    break
                pid = pick_pid(pool)
                image_ids = images_available(pid)
                for image_pos in rng.choice(len(image_ids), 2, replace=False):
                    consume(draw(pid, image_ids[image_pos]), batch)
                used_pids.add(pid)
                used_mask[pid] = True
                actual_pairs += 1
            if actual_pairs < pair_count:
                stats["pair_shortfall_batches"] += 1
            singleton_slots = capacity - len(batch)
            hard_slots = (int(singleton_slots * self.hard_fraction)
                          if self.negative_index is not None else 0)
            semi_slots = (int(singleton_slots * self.semi_hard_fraction)
                          if self.negative_index is not None else 0)
            stats["hard_requested"] += hard_slots
            stats["semi_hard_requested"] += semi_slots
            modes = (["hard"] * hard_slots + ["semi_hard"] * semi_slots
                     + ["random"] * (singleton_slots - hard_slots - semi_slots))
            relaxed = False
            for mode in modes:
                pool = np.flatnonzero((pid_remaining > 0) & ~used_mask)
                if not len(pool):
                    relaxed = True
                    # Prefer new images before allowing the unavoidable tail duplicates.
                    used_images = {(self.row_pids[i], self.row_images[i]) for i in batch}
                    available = np.flatnonzero(remaining > 0)
                    diverse = [i for i in available
                               if (self.row_pids[i], self.row_images[i]) not in used_images]
                    candidate = int(rng.choice(diverse if diverse else available))
                    if not diverse:
                        stats["same_image_fallback_draws"] += 1
                else:
                    candidate = mine(batch, used_pids, mode) if mode != "random" else None
                    if candidate is not None:
                        stats[f"{mode}_draws"] += 1
                    else:
                        if mode != "random":
                            stats["mining_fallback_draws"] += 1
                        candidate = draw(pick_pid(pool))
                consume(candidate, batch)
                used_pids.add(int(self.row_pids[candidate]))
                used_mask[self.row_pids[candidate]] = True
            if relaxed or actual_pairs < pair_count:
                stats["relaxed_batches"] += 1
            if len(used_pids) < 2:
                stats["batches_without_negative_pid"] += 1
            if stats["same_image_fallback_draws"] > fallback_before:
                image_fallback_batches.append(len(output))
            output.extend(rng.permutation(batch).tolist())
        # Swap a conflicting tail row with an earlier singleton when feasible.
        # Existing cross-image positive pairs and exact PID quotas stay intact.
        for start in image_fallback_batches:
            end = min(start + self.batch_size, len(output))
            seen = set()
            for position in range(start, end):
                index = output[position]
                key = (self.row_pids[index], self.row_images[index])
                if key in seen:
                    tail_pids = {self.row_pids[i] for i in output[start:end]}
                    repaired = False
                    for earlier in range(0, start, self.batch_size):
                        earlier_rows = output[earlier:earlier + self.batch_size]
                        counts = Counter(self.row_pids[i] for i in earlier_rows)
                        if self.row_pids[index] in counts:
                            continue
                        candidates = [j for j, row in enumerate(earlier_rows)
                                      if counts[self.row_pids[row]] == 1
                                      and self.row_pids[row] not in tail_pids]
                        if candidates:
                            source = earlier + int(rng.choice(candidates))
                            output[position], output[source] = output[source], index
                            stats["tail_repair_swaps"] += 1
                            repaired = True
                            break
                    if repaired:
                        index = output[position]
                        key = (self.row_pids[index], self.row_images[index])
                seen.add(key)
            counts = Counter((self.row_pids[i], self.row_images[i]) for i in output[start:end])
            stats["unresolved_same_image_positive_pairs"] += sum(
                count * (count - 1) // 2 for count in counts.values())
        self.diagnostics = stats
        return iter(output)
