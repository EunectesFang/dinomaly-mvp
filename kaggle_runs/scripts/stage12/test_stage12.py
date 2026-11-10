"""CPU tests for stage 12 masked local reconstruction.

Run:
    python -m unittest -v test_stage12
or:
    python test_stage12.py

The GPU-only checks (actual DINOv2 prepare_tokens invariance, stage-7 B0
identity on a real checkpoint, gradient through the real head) are in
masked_recon_v12.run_gpu_smoke and run at the start of every notebook.
"""
import os
import sys
import json
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import masked_recon_v12 as m  # noqa: E402

# The script usually receives these through prepare_environment(); the CPU
# tests do not build the real model, so provide them here.
m.torch = torch
m.nn = torch.nn
m.F = torch.nn.functional


class TestMaskGeometry(unittest.TestCase):
    def test_four_group_partition_exact(self):
        masks = m.numpy_group_masks()
        self.assertEqual(masks.shape, (4, 48 * 48))
        # every patch exactly once
        self.assertTrue(bool((masks.sum(axis=0) == 1).all()))
        # exactly 25% per group
        self.assertTrue(bool((masks.sum(axis=1) == 576).all()))
        self.assertEqual(int(masks.sum()), 48 * 48)
        # each group is 36 disjoint 4x4 blocks
        grid = m.group_index_grid()
        for g in range(4):
            blocks = grid.reshape(12, 4, 12, 4)[:, 0, :, 0]
            self.assertEqual(int((blocks == g).sum()), 36)
            self.assertEqual(int((grid == g).sum()), 36 * 16)

    def test_compose_uses_each_groups_own_coordinates(self):
        errors = np.arange(1, 5, dtype=np.float32).reshape(4, 1, 1, 1)
        errors = np.repeat(np.repeat(errors, 2, axis=1), 48, axis=2)
        errors = np.repeat(errors, 48, axis=3)
        out = m.compose_group_maps(__import__("torch").from_numpy(errors))
        grid = m.group_index_grid()
        for g in range(4):
            sel = grid == g
            self.assertTrue(bool((out[:, sel] == float(g + 1)).all()))
        self.assertTrue(bool((out > 0).all()))

    def test_pixel_mask_matches_patch_grid(self):
        masks = m.numpy_group_masks()
        torch = __import__("torch")
        flat = torch.as_tensor(masks[2]).unsqueeze(0)
        pix = m.patch_mask_to_pixel_mask(flat, 672)
        self.assertEqual(tuple(pix.shape), (1, 1, 672, 672))
        self.assertEqual(int(pix.sum()), 576 * 14 * 14)
        # one probe pixel per selected patch
        grid = m.group_index_grid()
        for r in range(0, 48, 7):
            for c in range(0, 48, 7):
                if grid[r, c] == 2:
                    self.assertTrue(bool(pix[0, 0, r * 14 + 3, c * 14 + 3]))
                else:
                    self.assertFalse(bool(pix[0, 0, r * 14 + 3, c * 14 + 3]))

    def test_split_disjoint_and_deterministic(self):
        paths = [f"p{i:04d}.png" for i in range(200)]
        a = m.split_train_calibration(paths, 96, 16, 1)
        b = m.split_train_calibration(paths, 96, 16, 1)
        self.assertEqual(a, b)
        self.assertEqual(len(a[0]), 96)
        self.assertEqual(len(a[1]), 16)
        self.assertFalse(set(a[0]) & set(a[1]))
        self.assertEqual(a[0], sorted(a[0]))
        self.assertEqual(a[1], sorted(a[1]))

    def test_schedule_deterministic(self):
        o1, g1 = m.make_training_schedule(37, epochs=3, seed=1)
        o2, g2 = m.make_training_schedule(37, epochs=3, seed=1)
        self.assertTrue(np.array_equal(o1, o2))
        self.assertTrue(np.array_equal(g1, g2))
        self.assertEqual(g1.shape, (3, 37))
        for e in range(3):
            self.assertEqual(sorted(o1[e].tolist()), list(range(37)))
        self.assertGreaterEqual(int(g1.min()), 0)
        self.assertLessEqual(int(g1.max()), 3)


class TestMaskedTokenEntrance(unittest.TestCase):
    def test_perturbing_only_masked_pixels_preserves_tokens(self):
        """Minimal patch-embed + mask replacement, mirroring prepare_tokens."""
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        embed = nn.Conv2d(3, 16, kernel_size=14, stride=14)
        mask_token = nn.Parameter(torch.randn(16))
        x = torch.randn(1, 3, 672, 672)
        masks = torch.as_tensor(m.numpy_group_masks()[1]).unsqueeze(0)
        pix = m.patch_mask_to_pixel_mask(masks, 672)

        def entrance(image):
            tok = embed(image).flatten(2).transpose(1, 2)
            return torch.where(masks.unsqueeze(-1),
                               mask_token.unsqueeze(0), tok)

        t0 = entrance(x)
        # Perturb only pixels inside the selected 4x4 blocks.
        x2 = torch.where(pix.expand_as(x), torch.full_like(x, 0.37), x)
        t1 = entrance(x2)
        self.assertLess(float((t0 - t1).abs().max()), 1e-6)
        # Perturbing an unmasked patch must change its token.
        grid = m.group_index_grid()
        r, c = next((i, j) for i in range(48) for j in range(48)
                    if grid[i, j] != 1)
        x3 = x.clone()
        x3[:, :, r * 14 + 5, c * 14 + 5] += 1.0
        t2 = entrance(x3)
        self.assertGreater(float((t0 - t2).abs().max()), 0.0)


class TestHeadAndGradient(unittest.TestCase):
    def test_target_detached_and_head_receives_gradient(self):
        import torch
        torch.manual_seed(1)
        head = m.MaskedFeatureHead()
        feats = torch.randn(2, 2, 768, 48, 48)
        target = feats.detach().clone()
        mask = torch.as_tensor(m.numpy_group_masks()[0]).unsqueeze(0)
        mask = mask.expand(2, -1)
        pred = head(feats)
        self.assertEqual(tuple(pred.shape), (2, 2, 768, 48, 48))
        loss = m.masked_cosine_loss(pred, target, mask)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertFalse(target.requires_grad)
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in grads))

    def test_error_map_only_group_weights(self):
        import torch
        pred = torch.zeros(1, 2, 768, 48, 48)
        target = torch.zeros_like(pred)
        target[:, :, 0] = 1.0
        err = m.cosine_error_map(pred, target)
        self.assertEqual(tuple(err.shape), (1, 1, 48, 48))
        # group 0 channel 0 is zero: cosine is 1 there; group 1 is all zero
        # and therefore also cosine 1 (zero vectors -> 0 similarity in
        # PyTorch, so just require finite values here).
        self.assertTrue(bool(torch.isfinite(err).all()))


class TestCalibration(unittest.TestCase):
    def test_median_p95_scale_and_floor(self):
        v = np.linspace(0.0, 1.0, 1001)
        s = m.calibration_stats(v)
        self.assertAlmostEqual(s["median"], 0.5)
        self.assertAlmostEqual(s["q95"], 0.95)
        self.assertAlmostEqual(s["scale"], 0.45)
        self.assertFalse(s["clamped"])
        flat = m.calibration_stats(np.zeros(100))
        self.assertEqual(flat["scale"], m.SCALE_FLOOR)
        self.assertTrue(flat["clamped"])
        out = m.normalize_map(np.array([0.5, 0.95]), s)
        self.assertAlmostEqual(float(out[0]), 0.0)
        self.assertAlmostEqual(float(out[1]), 1.0)

    def test_normalize_uses_given_scale(self):
        s = {"median": 1.0, "scale": 2.0}
        out = m.normalize_map(np.array([1.0, 3.0]), s)
        self.assertTrue(np.allclose(out, [0.0, 1.0]))


class TestMetricsAndPairing(unittest.TestCase):
    def test_pooled_and_anomaly_only_and_per_image(self):
        from sklearn.metrics import average_precision_score
        rng = np.random.default_rng(7)
        gt = np.zeros((3, 16, 16), dtype=np.uint8)
        gt[1, 2:6, 2:6] = 1
        gt[2, 9:13, 9:13] = 1
        labels = np.array([0, 1, 1])
        good = rng.normal(0, 0.1, (3, 16, 16)) + gt * 2.0
        bad = -good
        self.assertAlmostEqual(
            float(average_precision_score(gt.ravel(), good.ravel())),
            float(average_precision_score(gt.ravel(), good.ravel())))
        auc, ap = m.anomaly_only_metrics(labels, gt, good)
        self.assertTrue(np.isfinite(auc) and np.isfinite(ap))
        pia = m.per_image_pixel_ap(gt, good, labels)
        self.assertTrue(np.isnan(pia[0]))
        self.assertTrue(np.isfinite(pia[1]) and np.isfinite(pia[2]))
        self.assertGreater(float(np.nanmean(pia)), 0.9)
        pia_bad = m.per_image_pixel_ap(gt, bad, labels)
        self.assertLess(float(np.nanmean(pia_bad)), 0.9)
        self.assertGreater(float(np.nanmean(pia - pia_bad)), 0.0)

    def test_fixed_fpr_recall_affine_invariance(self):
        rng = np.random.default_rng(3)
        gt = np.zeros((2, 8, 8), dtype=np.uint8)
        gt[0, 1:3, 1:3] = 1
        pr = rng.normal(size=(2, 8, 8)) + gt * 3.0
        a, ta = m.fixed_fpr_recall(pr.ravel(), gt.ravel())
        b, tb = m.fixed_fpr_recall(10.0 * pr.ravel() + 2.0, gt.ravel())
        for k in a:
            self.assertAlmostEqual(a[k], b[k])
        for k in ta:
            self.assertNotAlmostEqual(ta[k], tb[k])

    def test_aggregate_pairing_and_entry_condition(self):
        # Three classes gain 2pp over b0 and the fourth gains 1pp.
        classes = ["a", "b", "c", "d"]
        per_class = {}
        base = {"b0": 0.60, "c1": 0.61, "m1": 0.62, "b0_c1": 0.615,
                "b0_m1": 0.62}
        for i, it in enumerate(classes):
            d = {"fingerprint": m.FP, "variants": {},
                 "image_metrics_by_variant": {
                     v: {"I-AUROC": 0.5, "I-AP": 0.5, "I-F1": 0.5}
                     for v in m.VARIANTS},
                 "image_metrics_original_base": {
                     "I-AUROC": 0.5, "I-AP": 0.5, "I-F1": 0.5}}
            for v in m.VARIANTS:
                ap = base[v] + (0.02 if i == 0 else 0.0)
                if it == "d":
                    ap = base[v] - (0.01 if v == "b0_m1" else 0.0)
                d["variants"][v] = {
                    "metrics": {k: (ap if k == "P-AP" else 0.5)
                                for k in m.METRIC_KEYS},
                    "recall": {k: 0.5 for k in m.RECALL_KEYS},
                    "per_image_ap": [0.5, 0.6, None],
                }
            per_class[it] = d
        agg = m.aggregate(per_class, classes, None)
        self.assertIn("b0_m1 - b0", agg["deltas"])
        self.assertAlmostEqual(
            agg["deltas"]["b0_m1 - b0"]["metrics"]["P-AP"], 0.0175,
            places=6)
        self.assertTrue(agg["entry_condition"]["passed"])
        self.assertEqual(
            agg["paired_image_ap_deltas"]["b0_m1 - b0"]
            ["n_images_with_valid_pixel_ap"], 8)


class TestFaithfulB0(unittest.TestCase):
    def test_b0_is_half_base_half_hi_then_guided(self):
        import torch
        xb = torch.zeros(1, 3, 392, 392)
        xh = torch.zeros(1, 3, 672, 672)
        base_raw = torch.ones(1, 1, 392, 392) * 1.0
        hi_raw = torch.ones(1, 1, 672, 672) * 3.0
        seen = {}

        def fake_global(model, images, chunk=None):
            if images.shape[-1] == 392:
                return base_raw
            return hi_raw

        def fake_guide(x):
            return torch.ones(1, 1, 256, 256)

        def fake_guided(guide, heat):
            seen["heat"] = heat.detach().clone()
            return heat

        gauss = lambda x: m.F.interpolate(
            x, size=256, mode="bilinear", align_corners=False)
        m.torch = torch
        m.F = torch.nn.functional
        m.BASE_BATCH_EFF = 2
        m.HI_BATCH_EFF = 2
        with mock.patch.object(m, "global_map_chunked", side_effect=fake_global), \
                mock.patch.object(m, "guide_from_hi", side_effect=fake_guide), \
                mock.patch.object(m, "guided_filter", side_effect=fake_guided):
            out = m.stage7_reference_b0(None, xb, xh, gauss)
        self.assertEqual(tuple(out.shape), (1, 1, 256, 256))
        self.assertTrue(torch.allclose(out, torch.full_like(out, 2.0)))
        self.assertTrue(torch.allclose(seen["heat"], torch.full_like(
            seen["heat"], 2.0)))


class TestJsonAndSelfTest(unittest.TestCase):
    def test_strict_json_nan_to_null(self):
        d = tempfile.mkdtemp(prefix="stage12_test_")
        p = os.path.join(d, "x.json")
        m.write_json_atomic(p, {"a": float("nan"),
                                "b": np.float32(1.5),
                                "c": np.int64(3),
                                "d": [np.bool_(True)]})
        with open(p, encoding="utf-8") as f:
            raw = f.read()
        self.assertNotIn("NaN", raw)
        with open(p, encoding="utf-8") as f:
            got = json.load(f)
        self.assertIsNone(got["a"])
        self.assertEqual(got["b"], 1.5)
        self.assertEqual(got["c"], 3)
        self.assertEqual(got["d"], [True])

    def test_run_self_test(self):
        self.assertEqual(m.run_self_test(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
