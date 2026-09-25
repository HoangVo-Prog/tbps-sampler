"""Audit batch composition produced by IRRA's random and identity samplers."""

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from datasets.cuhkpedes import CUHKPEDES
from datasets.icfgpedes import ICFGPEDES
from datasets.rstpreid import RSTPReid
from datasets.sampler import (
    RandomIdentityImageSampler,
    RandomIdentitySampler,
    RandomPositiveMixedSampler,
    BalancedMixedSampler,
)


def mean(values):
    return statistics.fmean(values) if values else 0.0


def std(values):
    return statistics.pstdev(values) if values else 0.0


def make_indices(rows, sampler_name, batch_size, num_instances, seed):
    if sampler_name == "random":
        generator = torch.Generator().manual_seed(seed)
        return torch.randperm(len(rows), generator=generator).tolist(), len(rows)

    random.seed(seed)
    np.random.seed(seed)
    sampler = RandomIdentitySampler(rows, batch_size, num_instances)
    return list(iter(sampler)), len(sampler)


def audit_batches(rows, indices, sampler_name, batch_size, num_instances, seed,
                  reported_length):
    pid_by_index = [row[0] for row in rows]
    image_by_index = [row[1] for row in rows]
    exposure = Counter(indices)
    batch_records = []

    for batch_number, start in enumerate(range(0, len(indices), batch_size), 1):
        batch_indices = indices[start:start + batch_size]
        pid_counts = Counter(pid_by_index[index] for index in batch_indices)
        image_counts = Counter(
            (pid_by_index[index], image_by_index[index])
            for index in batch_indices
        )

        positive_pairs = sum(count * (count - 1) // 2
                             for count in pid_counts.values())
        same_image_positive_pairs = sum(count * (count - 1) // 2
                                        for count in image_counts.values())
        anchors_with_positive = sum(count for count in pid_counts.values()
                                    if count > 1)
        unique_images_per_pid = [
            len({image_by_index[index] for index in batch_indices
                 if pid_by_index[index] == pid})
            for pid in pid_counts
        ]

        batch_records.append({
            "sampler": sampler_name,
            "num_instances": num_instances if sampler_name.startswith("identity") else "",
            "positive_pairs_per_batch": num_instances if sampler_name == "mixed" else "",
            "seed": seed,
            "batch": batch_number,
            "batch_size": len(batch_indices),
            "unique_pids": len(pid_counts),
            "negative_pids_per_anchor": max(0, len(pid_counts) - 1),
            "positive_pid_pairs": positive_pairs,
            "anchors_with_positive": anchors_with_positive,
            "anchor_positive_fraction": anchors_with_positive / len(batch_indices),
            "same_image_positive_pairs": same_image_positive_pairs,
            "same_image_fraction_of_positive_pairs": (
                same_image_positive_pairs / positive_pairs if positive_pairs else 0.0
            ),
            "mean_unique_images_per_pid": mean(unique_images_per_pid),
        })

    exposure_by_pid = Counter()
    for index, count in exposure.items():
        exposure_by_pid[pid_by_index[index]] += count
    pid_exposures = [exposure_by_pid[pid] for pid in set(pid_by_index)]
    source_counts = Counter(pid_by_index)
    exposure_ratios = [
        exposure_by_pid[pid] / source_counts[pid] for pid in source_counts
    ]
    actual_length = len(indices)

    summary = {
        "sampler": sampler_name,
        "num_instances": num_instances if sampler_name.startswith("identity") else None,
        "positive_pairs_per_batch": num_instances if sampler_name == "mixed" else None,
        "seed": seed,
        "source_samples": len(rows),
        "source_pids": len(source_counts),
        "source_samples_per_pid_min": min(source_counts.values()),
        "source_samples_per_pid_max": max(source_counts.values()),
        "source_samples_per_pid_cv": std(list(source_counts.values())) / mean(list(source_counts.values())),
        "sampler_reported_length": reported_length,
        "yielded_samples": actual_length,
        "yielded_fraction": actual_length / len(rows),
        "unique_rows_yielded": len(exposure),
        "rows_not_yielded": len(rows) - len(exposure),
        "duplicate_row_draws": sum(max(0, count - 1) for count in exposure.values()),
        "batches": len(batch_records),
        "batches_without_positive_pairs": sum(
            record["positive_pid_pairs"] == 0 for record in batch_records
        ),
        "fraction_batches_without_positive_pairs": mean([
            float(record["positive_pid_pairs"] == 0) for record in batch_records
        ]),
        "mean_unique_pids_per_batch": mean([r["unique_pids"] for r in batch_records]),
        "std_unique_pids_per_batch": std([r["unique_pids"] for r in batch_records]),
        "mean_positive_pid_pairs_per_batch": mean([r["positive_pid_pairs"] for r in batch_records]),
        "std_positive_pid_pairs_per_batch": std([r["positive_pid_pairs"] for r in batch_records]),
        "mean_anchor_positive_fraction": mean([r["anchor_positive_fraction"] for r in batch_records]),
        "std_anchor_positive_fraction": std([r["anchor_positive_fraction"] for r in batch_records]),
        "mean_negative_pids_per_anchor": mean([r["negative_pids_per_anchor"] for r in batch_records]),
        "mean_same_image_positive_fraction": mean([
            r["same_image_fraction_of_positive_pairs"] for r in batch_records
        ]),
        "mean_unique_images_per_pid_in_batch": mean([
            r["mean_unique_images_per_pid"] for r in batch_records
        ]),
        "pid_exposure_min": min(pid_exposures),
        "pid_exposure_max": max(pid_exposures),
        "pid_exposure_std": std(pid_exposures),
        "pid_exposure_cv": std(pid_exposures) / mean(pid_exposures) if mean(pid_exposures) else 0.0,
        "exposure_per_source_row_min": min(exposure_ratios),
        "exposure_per_source_row_max": max(exposure_ratios),
        "exposure_per_source_row_std": std(exposure_ratios),
    }
    return summary, batch_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True,
                        help="Dataset root containing the selected dataset directory")
    parser.add_argument("--dataset-name", default="RSTPReid",
                        choices=["RSTPReid", "CUHK-PEDES", "ICFG-PEDES"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-instances", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--positive-pairs-per-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="results/sampler_audit")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if any(k <= 0 or args.batch_size % k for k in args.num_instances):
        parser.error("each --num-instances value must be positive and divide batch-size")

    dataset_cls = {
        "RSTPReid": RSTPReid,
        "CUHK-PEDES": CUHKPEDES,
        "ICFG-PEDES": ICFGPEDES,
    }[args.dataset_name]
    dataset = dataset_cls(root=args.root_dir)
    rows = dataset.train
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    batch_rows = []
    configs = [("random", None)]
    configs.extend(("identity", k) for k in args.num_instances)
    configs.extend(("identity_image", k) for k in args.num_instances)
    configs.append(("mixed", args.positive_pairs_per_batch))
    configs.append(("balanced_mixed", args.positive_pairs_per_batch))
    for sampler_name, num_instances in configs:
        if sampler_name == "identity_image":
            random.seed(args.seed)
            sampler = RandomIdentityImageSampler(rows, args.batch_size, num_instances)
            indices, reported_length = list(iter(sampler)), len(sampler)
        elif sampler_name == "mixed":
            random.seed(args.seed)
            sampler = RandomPositiveMixedSampler(
                rows, args.batch_size, args.positive_pairs_per_batch
            )
            indices, reported_length = list(iter(sampler)), len(sampler)
        elif sampler_name == "balanced_mixed":
            random.seed(args.seed)
            np.random.seed(args.seed)
            sampler = BalancedMixedSampler(
                rows, args.batch_size, args.positive_pairs_per_batch
            )
            indices, reported_length = list(iter(sampler)), len(sampler)
        else:
            indices, reported_length = make_indices(
                rows, sampler_name, args.batch_size, num_instances, args.seed
            )
        summary, per_batch = audit_batches(
            rows, indices, sampler_name, args.batch_size, num_instances,
            args.seed, reported_length
        )
        summaries.append(summary)
        batch_rows.extend(per_batch)

    suffix = (
        f"b{args.batch_size}_m{args.positive_pairs_per_batch}_seed{args.seed}"
    )
    summary_path = output_dir / f"summary_{suffix}.json"
    batches_path = output_dir / f"batches_{suffix}.csv"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with batches_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=batch_rows[0].keys())
        writer.writeheader()
        writer.writerows(batch_rows)

    for summary in summaries:
        print(json.dumps(summary, ensure_ascii=False))
    print(f"Wrote {summary_path} and {batches_path}")


if __name__ == "__main__":
    main()
