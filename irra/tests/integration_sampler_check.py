"""CPU integration checks: python irra/tests/integration_sampler_check.py -v."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1]))
from datasets.sampler import BalancedMixedSampler
from datasets.sampler_mining import NegativeNeighborIndex, negative_index_from_topk
from sampler_audit import audit_batches, pid_exposure_records
from sampler_feature_audit import batch_metrics, global_topk, sample_indices


class SamplerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [(pid, pid * 3 + image, f"{pid}_{image}.jpg", f"caption {caption}")
                     for pid in range(64) for image in range(3) for caption in range(2)]
        self.device = torch.device("cpu")

    def test_torch_dataloader_length_tail_and_epoch(self):
        sampler = BalancedMixedSampler(self.rows, 32, 4, seed=3)
        loader = torch.utils.data.DataLoader(torch.arange(len(self.rows)),
                                            batch_size=32, sampler=sampler)
        first = torch.cat(list(loader)).tolist()
        self.assertEqual(sorted(first), list(range(len(self.rows))))
        self.assertEqual(len(sampler), len(first))
        sampler.set_epoch(1)
        second = torch.cat(list(loader)).tolist()
        self.assertNotEqual(first, second)
        self.assertEqual(sorted(second), sorted(first))
        partial = BalancedMixedSampler(self.rows[:-1], 32, 4)
        batches = list(torch.utils.data.DataLoader(torch.arange(len(self.rows) - 1),
                                                  batch_size=32, sampler=partial))
        self.assertEqual(len(batches), 12)
        self.assertEqual(len(batches[-1]), 31)

    def test_real_cosine_cache_sampler_and_audit_pipeline(self):
        generator = torch.Generator().manual_seed(7)
        images = F.normalize(torch.randn(192, 24, generator=generator), dim=1)
        texts = F.normalize(torch.randn(len(self.rows), 24, generator=generator), dim=1)
        positions = np.asarray([row[1] for row in self.rows])
        pids = np.asarray([row[0] for row in self.rows])
        image_pids = np.repeat(np.arange(64), 3)
        neighbors = [
            global_topk(images[positions], texts, pids, pids, 32, 61, self.device),
            global_topk(texts, images, pids, image_pids, 32, 61, self.device),
            global_topk(texts, texts, pids, pids, 32, 61, self.device),
        ]
        index = negative_index_from_topk(self.rows, neighbors, positions, "synthetic-cosine")
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "neighbors.npz"
            index.save(cache)
            index = NegativeNeighborIndex.load(cache, self.rows)
        for name in ("random", "identity", "identity_image", "mixed",
                     "balanced_mixed_metadata", "balanced_mixed"):
            indices, length, diagnostics = sample_indices(
                self.rows, name, 32, 2, 4, 1, negative_index=index)
            summary, records = audit_batches(self.rows, indices, name, 32, 2, 1, length)
            self.assertEqual(summary["yielded_samples"], len(indices))
            exposures = pid_exposure_records(self.rows, indices)
            self.assertEqual(sum(row["draws"] for row in exposures), len(indices))
            if name.startswith("balanced_mixed"):
                self.assertEqual(summary["rows_not_yielded"], 0)
                self.assertEqual(summary["duplicate_row_draws"], 0)
                self.assertEqual(diagnostics["mining_enabled"], name == "balanced_mixed")
            for start in range(0, len(indices), 32):
                metrics = batch_metrics(indices[start:start + 32], images, texts,
                                        positions, pids, image_pids, neighbors,
                                        start // 32 + 1, name, 1, [1, 5, 10], self.device)
                for direction in ("image_to_text", "text_to_image", "caption_to_caption"):
                    self.assertGreaterEqual(
                        metrics[f"{direction}_global_minus_batch_hardest_gap"], -1e-6)
                    self.assertTrue(0 <= metrics[f"{direction}_global_top10_hit_rate"] <= 1)

    def test_same_pid_mask_is_not_mined_on_small_dataset(self):
        rows = self.rows[:12]
        pids = np.asarray([row[0] for row in rows])
        features = F.normalize(torch.randn(len(rows), 8), dim=1)
        neighbors = global_topk(features, features, pids, pids, 11, 5, self.device)
        self.assertTrue(np.isneginf(neighbors[1]).any())
        index = negative_index_from_topk(rows, [neighbors] * 3,
                                        np.arange(len(rows)), "masked-test")
        valid = index.neighbors >= 0
        self.assertTrue(np.all(pids[np.maximum(index.neighbors, 0)][valid]
                               != np.broadcast_to(pids[:, None, None], valid.shape)[valid]))


if __name__ == "__main__":
    unittest.main()
