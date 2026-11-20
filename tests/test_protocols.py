import unittest

import numpy as np
import torch

from softmaskgat.data import quality_weights
from softmaskgat.shrink import ShrinkAdapter
from softmaskgat.statistics import compare_predictions, draw_counts
from softmaskgat.tables import table_rows
from softmaskgat.training import make_model, splits_for


class StudyChecks(unittest.TestCase):
    def test_branches_and_readout(self):
        for name, count in (("GAP", 31169), ("Attn", 64641)):
            model = make_model(name)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            mask = torch.tensor([[True, False]])
            quality = torch.tensor([[0.7, 0.0]])
            h = torch.tensor([[[2.0], [10000.0]]])
            result = model.backend.readout(h, mask, quality, use_quality=True)
            torch.testing.assert_close(result, h[:, 0])

    def test_shrink_endpoints_and_attention_modes(self):
        q = torch.tensor([[0.25, 0.75, 0.0]])
        mask = q > 0
        for exponent, expected in ((0, mask.float()), (1, q)):
            model = ShrinkAdapter(make_model("GAP"), fixed_lambda=exponent)
            torch.testing.assert_close(model.effective_quality(q, mask), expected, rtol=0, atol=0)
        hard = make_model("GAP", "Binary_attention").backend.gat_blocks[0].gat
        soft = make_model("GAP", "No_q_attention").backend.gat_blocks[0].gat
        self.assertIsNone(hard._build_log_q(mask, mask.float()))
        torch.testing.assert_close(soft._build_log_q(mask, mask.float())[0, 2], torch.tensor(0.001).log())

    def test_quality_fitting_and_participant_splits(self):
        ratio = np.array([[1.0, 3.0], [2.0, 6.0], [5.0, 8.0]])
        mask = np.ones_like(ratio, dtype=bool)
        q = quality_weights(ratio, mask, [0, 1])
        ratio[2] *= 1000
        np.testing.assert_array_equal(q[:2], quality_weights(ratio, mask, [0, 1])[:2])
        labels = np.tile([0, 1], 20)
        fixed1 = splits_for(labels, "core", 42, 10)
        fixed2 = splits_for(labels, "core", 123, 10)
        repeat2 = splits_for(labels, "quality", 123, 10)
        for (tr, te), (_, other) in zip(fixed1, fixed2):
            self.assertFalse(np.intersect1d(tr, te).size)
            np.testing.assert_array_equal(te, other)
        self.assertTrue(any(not np.array_equal(a[1], b[1]) for a, b in zip(fixed1, repeat2)))

    def test_aggregate_tables(self):
        rows = table_rows()
        self.assertEqual(rows[2][-1][1], "0.825 [0.760, 0.885]")
        self.assertEqual(len(rows[11]) - 1, 7)
        self.assertEqual(rows[13][1][1], "+0.0176")

    def test_single_class_bootstrap_draws(self):
        labels = np.array([0, 0, 0, 1])
        branches = np.array([[[[0.7, 0.2, 0.6, 0.8], [0.4, 0.5, 0.9, 0.1]]]])
        for protocol, name, valid_auc in (("primary", "Fusion", 128), ("handling", "FULL", 96)):
            rows = compare_predictions(branches, ["FULL"], labels, protocol, draws=128)
            totals = {row["metric"]: row["valid_draws"] for row in rows if row["comparison"] == name}
            self.assertEqual(totals, {"AUC": valid_auc, "BAcc": 128, "MCC": 128, "Brier": 128})

    def test_targeted_draws_across_blocks(self):
        labels = np.arange(197) % 2
        indices = np.random.default_rng(20260713).integers(0, 197, size=(650, 197), dtype=np.int16)
        expected = np.stack([np.bincount(row, minlength=197) for row in indices])
        np.testing.assert_array_equal(np.concatenate(list(draw_counts(labels, "targeted", 650))), expected)


if __name__ == "__main__":
    unittest.main()
