"""Measure batch coverage of global hard negatives and caption neighbors."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.bases import ImageDataset, TextDataset
from datasets.build import build_transforms
from datasets.cuhkpedes import CUHKPEDES
from datasets.icfgpedes import ICFGPEDES
from datasets.rstpreid import RSTPReid
from datasets.sampler import (
    RandomIdentityImageSampler,
    RandomIdentitySampler,
    RandomPositiveMixedSampler,
)
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs


def encode_features(rows, model, args, device, batch_size, workers):
    image_by_id = {}
    for pid, image_id, image_path, _ in rows:
        image_by_id.setdefault(image_id, (pid, image_path))
    image_ids = list(image_by_id)
    image_id_to_pos = {image_id: pos for pos, image_id in enumerate(image_ids)}

    transform = build_transforms(img_size=args.img_size, is_train=False)
    image_set = ImageDataset(
        [image_by_id[i][0] for i in image_ids],
        [image_by_id[i][1] for i in image_ids],
        transform,
    )
    text_set = TextDataset(
        [row[0] for row in rows], [row[3] for row in rows],
        text_length=args.text_length,
    )
    image_loader = DataLoader(
        image_set, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda",
    )
    text_loader = DataLoader(
        text_set, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda",
    )

    image_features = []
    with torch.inference_mode():
        for _, images in image_loader:
            image_features.append(
                F.normalize(model.encode_image(images.to(device)).float(), dim=1).cpu()
            )
        text_features = []
        for _, tokens in text_loader:
            text_features.append(
                F.normalize(model.encode_text(tokens.to(device)).float(), dim=1).cpu()
            )

    image_features = torch.cat(image_features)
    text_features = torch.cat(text_features)
    row_image_positions = np.asarray(
        [image_id_to_pos[row[1]] for row in rows], dtype=np.int64
    )
    pids = np.asarray([row[0] for row in rows], dtype=np.int64)
    image_pids = np.asarray([image_by_id[i][0] for i in image_ids], dtype=np.int64)
    return image_features, text_features, row_image_positions, pids, image_pids


def global_topk(query, candidates, query_pids, candidate_pids, k, chunk_size, device):
    query = query.to(device)
    candidates = candidates.to(device)
    candidate_pids = torch.as_tensor(candidate_pids, device=device)
    query_pids = torch.as_tensor(query_pids, device=device)
    k = min(k, candidates.shape[0] - 1)
    all_indices, all_scores = [], []
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        scores = query[start:end] @ candidates.T
        scores.masked_fill_(
            query_pids[start:end, None] == candidate_pids[None, :], -torch.inf
        )
        values, indices = scores.topk(k, dim=1, largest=True, sorted=True)
        all_indices.append(indices.cpu().numpy())
        all_scores.append(values.cpu().numpy())
    return np.concatenate(all_indices), np.concatenate(all_scores)


def sample_indices(rows, sampler_name, batch_size, k, mixed_pairs, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if sampler_name == "random":
        return torch.randperm(len(rows)).tolist()
    if sampler_name == "identity":
        sampler = RandomIdentitySampler(rows, batch_size, k)
    elif sampler_name == "identity_image":
        sampler = RandomIdentityImageSampler(rows, batch_size, k)
    else:
        sampler = RandomPositiveMixedSampler(rows, batch_size, mixed_pairs)
    return list(iter(sampler))


def batch_metrics(indices, image_features, text_features, row_image_positions,
                  pids, image_pids, global_results, batch_number, sampler,
                  seed, k_values, device):
    index = np.asarray(indices, dtype=np.int64)
    batch_pids = pids[index]
    batch_images = row_image_positions[index]
    unique_images = np.unique(batch_images)
    image_pos = torch.as_tensor(batch_images, device=device)
    text_pos = torch.as_tensor(index, device=device)
    unique_image_pos = torch.as_tensor(unique_images, device=device)
    batch_pid_tensor = torch.as_tensor(batch_pids, device=device)
    unique_image_pids = torch.as_tensor(image_pids[unique_images], device=device)

    image_batch_feats = image_features.to(device)[image_pos]
    text_batch_feats = text_features.to(device)[text_pos]
    unique_image_feats = image_features.to(device)[unique_image_pos]

    i2t = image_batch_feats @ text_batch_feats.T
    i2t.masked_fill_(batch_pid_tensor[:, None] == batch_pid_tensor[None, :], -torch.inf)
    t2i = text_batch_feats @ unique_image_feats.T
    t2i.masked_fill_(batch_pid_tensor[:, None] == unique_image_pids[None, :], -torch.inf)
    t2t = text_batch_feats @ text_batch_feats.T
    t2t.masked_fill_(batch_pid_tensor[:, None] == batch_pid_tensor[None, :], -torch.inf)
    batch_hardest = [
        i2t.max(dim=1).values.cpu().numpy(),
        t2i.max(dim=1).values.cpu().numpy(),
        t2t.max(dim=1).values.cpu().numpy(),
    ]

    candidate_sets = [set(index.tolist()), set(unique_images.tolist()), set(index.tolist())]
    neighbor_indices = [global_results[0][0][index], global_results[1][0][index],
                        global_results[2][0][index]]
    global_max = [global_results[0][1][index, 0], global_results[1][1][index, 0],
                  global_results[2][1][index, 0]]
    names = ["image_to_text", "text_to_image", "caption_to_caption"]
    result = {
        "sampler": sampler, "seed": seed, "batch": batch_number,
        "batch_size": len(index), "unique_pids": int(len(np.unique(batch_pids))),
    }
    for direction, candidates, neighbors, top_scores, local_max in zip(
        names, candidate_sets, neighbor_indices, global_max, batch_hardest
    ):
        valid = np.isfinite(local_max)
        result[f"{direction}_anchors_with_batch_negatives"] = int(valid.sum())
        result[f"{direction}_batch_hardest_cosine"] = (
            float(local_max[valid].mean()) if valid.any() else None
        )
        result[f"{direction}_global_hardest_cosine"] = float(top_scores.mean())
        result[f"{direction}_global_minus_batch_hardest_gap"] = (
            float((top_scores[valid] - local_max[valid]).mean()) if valid.any() else None
        )
        for top_k in k_values:
            width = min(top_k, neighbors.shape[1])
            hits = np.asarray([
                len(set(row[:width].tolist()) & candidates) / width
                for row in neighbors
            ])
            result[f"{direction}_global_top{top_k}_coverage"] = float(hits.mean())
            result[f"{direction}_global_top{top_k}_hit_rate"] = float((hits > 0).mean())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--dataset-name", default="RSTPReid",
                        choices=["RSTPReid", "CUHK-PEDES", "ICFG-PEDES"])
    parser.add_argument("--model-run", required=True,
                        help="IRRA run folder containing configs.yaml")
    parser.add_argument("--checkpoint", default="best.pth",
                        help="checkpoint filename or full path")
    parser.add_argument("--output-dir", default="results/sampler_feature_audit")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--num-instances", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--positive-pairs-per-batch", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_train_configs(args.model_run)
    config.root_dir = args.root_dir
    config.dataset_name = args.dataset_name
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset_cls = {
        "RSTPReid": RSTPReid,
        "CUHK-PEDES": CUHKPEDES,
        "ICFG-PEDES": ICFGPEDES,
    }[args.dataset_name]
    dataset = dataset_cls(root=args.root_dir, verbose=False)
    model = build_model(config, num_classes=len(dataset.train_id_container)).to(device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(args.model_run) / checkpoint_path
    Checkpointer(model).load(str(checkpoint_path))
    model.eval()

    rows = dataset.train
    image_features, text_features, row_image_positions, pids, image_pids = encode_features(
        rows, model, config, device, args.feature_batch_size, args.num_workers
    )
    image_features = image_features.to(device)
    text_features = text_features.to(device)
    row_image_features = image_features[
        torch.as_tensor(row_image_positions, dtype=torch.long, device=device)
    ]
    top_k = max(args.top_k)
    global_neighbors = [
        global_topk(row_image_features, text_features, pids, pids, top_k,
                    args.chunk_size, device),
        global_topk(text_features, image_features, pids, image_pids, top_k,
                    args.chunk_size, device),
        global_topk(text_features, text_features, pids, pids, top_k,
                    args.chunk_size, device),
    ]

    configurations = [("random", None)]
    configurations.extend(("identity", k) for k in args.num_instances)
    configurations.extend(("identity_image", k) for k in args.num_instances)
    configurations.extend(("mixed", args.positive_pairs_per_batch) for _ in [0])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sampler, value in configurations:
        for seed in args.seeds:
            indices = sample_indices(rows, sampler, args.batch_size,
                                     value or args.num_instances[0],
                                     args.positive_pairs_per_batch, seed)
            records = []
            for batch_number, start in enumerate(range(0, len(indices), args.batch_size), 1):
                batch = indices[start:start + args.batch_size]
                records.append(batch_metrics(
                    batch, image_features, text_features,
                    row_image_positions, pids, image_pids, global_neighbors, batch_number,
                    sampler, seed, args.top_k, device
                ))
            summary = {"sampler": sampler, "value": value, "seed": seed,
                       "checkpoint": str(checkpoint_path), "similarity": "cosine",
                       "batches": len(records)}
            for key in records[0]:
                if key in {"sampler", "seed", "batch", "batch_size", "unique_pids"}:
                    continue
                values = [row[key] for row in records if row[key] is not None]
                summary[f"mean_{key}"] = float(np.mean(values)) if values else None
                summary[f"std_{key}"] = float(np.std(values)) if values else None
            stem = f"{sampler}_{value or 'na'}_seed{seed}"
            (output_dir / f"{stem}.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            with (output_dir / f"{stem}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
            print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
