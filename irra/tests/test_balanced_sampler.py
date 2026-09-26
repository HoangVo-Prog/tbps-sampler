"""Run with python -m unittest discover -s irra/tests -v (NumPy only)."""

from collections import Counter
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "sampler_mining", Path(__file__).parents[1] / "datasets" / "sampler_mining.py")
mining = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mining)
Sampler = mining.BalancedMixedIndexSampler
NeighborIndex = mining.NegativeNeighborIndex


def rows_for_counts(counts):
    return [(pid, pid * 100 + pos % 3, f"images/{pid}_{pos % 3}.jpg", f"caption {pos}")
            for pid, count in enumerate(counts) for pos in range(count)]


def ring_index(rows, width=8):
    pids = list(dict.fromkeys(row[0] for row in rows))
    first_rows = {pid: next(i for i, row in enumerate(rows) if row[0] == pid)
                  for pid in pids}
    neighbors = np.empty((len(rows), 3, width), dtype=np.int32)
    for i, row in enumerate(rows):
        nearest = [(row[0] + distance) % len(pids) for distance in range(1, width + 1)]
        neighbors[i] = [first_rows[pid] for pid in nearest]
    return NeighborIndex(rows, neighbors, "frozen-ring-encoder")


def neighbor_hit_rate(rows, indices, batch_size, pid_count):
    hits = 0
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        pids = {rows[index][0] for index in batch}
        hits += sum((rows[index][0] + 1) % pid_count in pids for index in batch)
    return hits / len(indices)


class BalancedSamplerTests(unittest.TestCase):
    def test_coverage_visits_every_row_once_and_keeps_partial_tail(self):
        rows = rows_for_counts([6] * 40 + [1])
        sampler = Sampler(rows, 32, 4)
        indices = list(sampler)
        self.assertEqual(len(indices), len(sampler))
        self.assertEqual(Counter(indices), Counter(range(len(rows))))
        self.assertEqual(len(indices) % 32, len(rows) % 32)

    def test_regular_batch_has_cross_image_pairs_and_many_singletons(self):
        rows = rows_for_counts([12] * 80)
        sampler = Sampler(rows, 64, 4, seed=17)
        batch = [rows[index] for index in list(sampler)[:64]]
        counts = Counter(row[0] for row in batch)
        self.assertEqual(len(counts), 60)
        self.assertEqual(sum(count == 2 for count in counts.values()), 4)
        for pid, count in counts.items():
            if count == 2:
                self.assertEqual(len({row[1] for row in batch if row[0] == pid}), 2)

    def test_uneven_and_low_diversity_tail_is_retained_and_reported(self):
        rows = [(0, 0, "0.jpg", "same") for _ in range(37)]
        rows += rows_for_counts([2, 3, 4])[2:]
        sampler = Sampler(rows, 16, 2)
        self.assertEqual(Counter(list(sampler)), Counter(range(len(rows))))
        self.assertGreater(sampler.diagnostics["relaxed_batches"], 0)
        self.assertGreater(sampler.diagnostics["same_image_fallback_draws"], 0)

    def test_oversampling_equalizes_pid_quota_and_covers_majority(self):
        rows = rows_for_counts([2, 8, 18, 32])
        sampler = Sampler(rows, 8, 1, rarity_power=1, exposure_mode="oversample")
        indices = list(sampler)
        self.assertEqual(Counter(rows[i][0] for i in indices), {pid: 32 for pid in range(4)})
        self.assertEqual(set(indices), set(range(len(rows))))
        self.assertEqual(len(sampler), len(indices))
        self.assertEqual(sampler.diagnostics["planned_extra_draws"], 128 - len(rows))

    def test_rarity_counts_rows_not_images(self):
        rows = [(pid, pid * 10 + pos % 2, "img.jpg", str(pos))
                for pid, count in enumerate([2, 18, 8]) for pos in range(count)]
        sampler = Sampler(rows, 8, 1, rarity_power=0.5, exposure_mode="oversample")
        self.assertEqual(sampler.target_counts.tolist(), [6, 18, 12])
        self.assertEqual(Counter(rows[i][0] for i in sampler), {0: 6, 1: 18, 2: 12})

    def test_zero_rarity_oversampling_has_no_repeats(self):
        rows = rows_for_counts([2, 18, 8])
        sampler = Sampler(rows, 8, 1, rarity_power=0, exposure_mode="oversample")
        self.assertEqual(Counter(list(sampler)), Counter(range(len(rows))))

    def test_balanced_source_does_not_get_unnecessary_oversampling(self):
        rows = rows_for_counts([10] * 16)
        sampler = Sampler(rows, 16, 2, rarity_power=0.5, exposure_mode="oversample")
        self.assertEqual(len(sampler), len(rows))
        self.assertEqual(Counter(list(sampler)), Counter(range(len(rows))))

    def test_seed_and_epoch_are_reproducible(self):
        rows = rows_for_counts([6] * 24)
        sampler = Sampler(rows, 16, 2, seed=7)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        second = list(sampler)
        self.assertNotEqual(first, second)
        other = Sampler(rows, 16, 2, seed=7)
        other.set_epoch(1)
        self.assertEqual(second, list(other))

    def test_mining_improves_known_global_neighbor_coverage(self):
        rows = rows_for_counts([6] * 80)
        index = ring_index(rows)
        random_hits, mined_hits = [], []
        for seed in range(5):
            baseline = Sampler(rows, 16, 2, seed=seed)
            guided = Sampler(rows, 16, 2, seed=seed, negative_index=index,
                             hard_top_k=2, hard_fraction=0.5, semi_hard_fraction=0.25)
            baseline_indices, guided_indices = list(baseline), list(guided)
            random_hits.append(neighbor_hit_rate(rows, baseline_indices, 16, 80))
            mined_hits.append(neighbor_hit_rate(rows, guided_indices, 16, 80))
            self.assertEqual(Counter(guided_indices), Counter(range(len(rows))))
            self.assertGreater(guided.diagnostics["hard_draws"], 0)
            self.assertGreater(guided.diagnostics["semi_hard_draws"], 0)
        self.assertGreater(np.mean(mined_hits), np.mean(random_hits) + 0.1)

    def test_depleted_neighbor_pool_falls_back_without_losing_rows(self):
        rows = rows_for_counts([3] * 20)
        index = NeighborIndex(rows, np.full((len(rows), 3, 4), -1, dtype=np.int32))
        sampler = Sampler(rows, 16, 2, negative_index=index, hard_top_k=1)
        self.assertEqual(Counter(list(sampler)), Counter(range(len(rows))))
        self.assertGreater(sampler.diagnostics["mining_fallback_draws"], 0)
        self.assertEqual(sampler.diagnostics["hard_draws"], 0)

    def test_cosine_mining_raises_top10_hit_rate_across_seeds(self):
        pid_count = 256
        rows = rows_for_counts([6] * pid_count)
        rng = np.random.default_rng(42)
        features = rng.normal(size=(pid_count, 16))
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        similarities = features @ features.T
        np.fill_diagonal(similarities, -np.inf)
        ranked_pids = np.argsort(-similarities, axis=1)[:, :24]
        first_rows = np.arange(pid_count) * 6
        row_rankings = first_rows[ranked_pids[np.arange(len(rows)) // 6]]
        index = NeighborIndex(rows, np.repeat(row_rankings[:, None, :], 3, axis=1))

        def evaluate(indices):
            hits = 0
            for start in range(0, len(indices), 16):
                batch = indices[start:start + 16]
                pids = {rows[i][0] for i in batch}
                hits += sum(bool(set(ranked_pids[rows[i][0], :10]) & pids) for i in batch)
            return hits / len(indices)

        baseline, guided = [], []
        for seed in range(5):
            baseline.append(evaluate(list(Sampler(rows, 16, 2, seed=seed))))
            guided.append(evaluate(list(Sampler(rows, 16, 2, seed=seed,
                                               negative_index=index, hard_top_k=10))))
        self.assertGreater(np.mean(guided), np.mean(baseline) + 0.05)

    def test_image_neighbors_use_image_positions_not_caption_row_positions(self):
        rows = rows_for_counts([2] * 4)
        image_positions = np.asarray([2, 2, 0, 0, 1, 1, 3, 3])
        representative = {0: 2, 1: 4, 2: 0, 3: 6}
        candidates = np.asarray([
            [(image + 1) % 4, (image + 2) % 4] for image in image_positions])
        row_candidates = np.asarray([[representative[int(i)] for i in row]
                                     for row in candidates])
        scores = np.ones_like(candidates, dtype=np.float32)
        index = mining.negative_index_from_topk(
            rows, [(row_candidates, scores), (candidates, scores), (row_candidates, scores)],
            image_positions, "encoder")
        np.testing.assert_array_equal(index.neighbors[:, 1], row_candidates)

    def test_cache_roundtrip_and_dataset_order_validation(self):
        rows = rows_for_counts([3] * 12)
        index = ring_index(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neighbors.npz"
            index.save(path)
            loaded = NeighborIndex.load(path, rows)
            np.testing.assert_array_equal(loaded.neighbors, index.neighbors)
            self.assertEqual(loaded.checkpoint, index.checkpoint)
            with self.assertRaisesRegex(ValueError, "rows/order"):
                NeighborIndex.load(path, list(reversed(rows)))

    def test_same_pid_cannot_be_loaded_as_negative(self):
        rows = rows_for_counts([3] * 12)
        invalid = np.tile(np.arange(len(rows))[:, None, None], (1, 3, 4))
        with self.assertRaisesRegex(ValueError, "different PID"):
            NeighborIndex(rows, invalid)

    def test_image_neighbor_row_mapping_and_invalid_mask(self):
        rows = rows_for_counts([2] * 4)
        top = np.asarray([[(i + 2) % len(rows), (i + 4) % len(rows)]
                          for i in range(len(rows))])
        scores = np.ones_like(top, dtype=np.float32)
        scores[:, 1] = -np.inf
        index = mining.negative_index_from_topk(
            rows, [(top, scores)] * 3, np.arange(len(rows)), "encoder")
        np.testing.assert_array_equal(index.neighbors[:, :, 1], -1)

    def test_parameter_validation(self):
        rows = rows_for_counts([3] * 12)
        for options in (dict(rarity_power=-0.1), dict(rarity_power=1.1),
                        dict(hard_fraction=0.9, semi_hard_fraction=0.2),
                        dict(hard_fraction=float("nan")), dict(exposure_mode="typo")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Sampler(rows, 16, 2, **options)


if __name__ == "__main__":
    unittest.main()
