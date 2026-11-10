# Dinomaly local refine v3 — stage 3

Slug: `eunectesqaq/dinomaly-local-refine-v3`  
Generated: 2026-11-05 11:16:33  
Checkpoints: `/kaggle/input/notebooks/eunectesqaq/dinomaly-base-10k/model_10000.pt` + `/kaggle/input/notebooks/eunectesqaq/dinomaly-local-adapt/local_adapt.pt`  
Test images: 1725 across 15 classes  
Calibration: 24 deterministic train/good images per class, shrink weight 0.500

## Bug fixes

- **Bug 1 (line 323):** every fusion uses the original baseline global map. No adapted-global forward is computed. `adapted_local_fuse` is exactly `0.5 * baseline_global + 0.5 * adapted_local`.
- **Bug 2 (lines 358-379):** fixed-FPR thresholds are fitted on each variant's own per-class negative pixels. Thresholds are stored per variant in each per-class JSON.

## Variants

| variant | definition |
|---|---|
| `base_global` | baseline global map, no calibration |
| `base_local_fuse` | 0.5 * baseline global + 0.5 * baseline local |
| `adapted_local_fuse` | 0.5 * baseline global + 0.5 * adapted local (bug 1 fixed) |
| `cal_scalar` | per-branch median/q95 alignment, then 0.5 fusion |
| `cal_spatial` | per-branch shrunk spatial background correction, no scalar alignment, then 0.5 fusion |
| `cal_fused` | per-branch scalar alignment + shrunk spatial background correction, then 0.5 fusion (primary calibrated variant) |

## 15-class means (%)

| variant | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|---|---|---|
| `base_global` | 99.62 | 99.81 | 99.15 | 98.33 | 68.82 | 69.04 | 94.67 |
| `base_local_fuse` | 99.62 | 99.81 | 99.15 | 97.82 | 71.29 | 70.06 | 94.34 |
| `adapted_local_fuse` | 99.62 | 99.81 | 99.15 | 98.11 | 71.87 | 70.53 | 94.82 |
| `cal_scalar` | 99.62 | 99.81 | 99.15 | 98.26 | 71.63 | 70.56 | 95.14 |
| `cal_spatial` | 99.62 | 99.81 | 99.15 | 98.08 | 71.54 | 70.44 | 94.91 |
| `cal_fused` | 99.62 | 99.81 | 99.15 | 98.23 | 71.27 | 70.40 | 95.00 |

Image metrics are identical by construction: every variant reuses the same original baseline image score. Only P-* columns differ.

## Recall at fixed pixel FPR (class mean, %)

| variant | R@FPR0.001 | R@FPR0.01 | R@FPR0.05 |
|---|---|---|---|
| `base_global` | 18.12 | 65.48 | 94.50 |
| `base_local_fuse` | 23.04 | 68.97 | 91.55 |
| `adapted_local_fuse` | 22.51 | 69.47 | 93.07 |
| `cal_scalar` | 21.64 | 69.02 | 93.90 |
| `cal_spatial` | 21.88 | 69.03 | 93.30 |
| `cal_fused` | 21.10 | 68.51 | 93.98 |

## Primary mean deltas (percentage points)

| comparison | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|
| base_local_fuse - base_global | -0.51 | +2.48 | +1.02 | -0.33 |
| adapted_local_fuse - base_global | -0.22 | +3.05 | +1.49 | +0.15 |
| adapted_local_fuse - base_local_fuse | +0.29 | +0.58 | +0.47 | +0.48 |
| cal_scalar - adapted_local_fuse | +0.15 | -0.24 | +0.02 | +0.32 |
| cal_spatial - adapted_local_fuse | -0.03 | -0.33 | -0.09 | +0.09 |
| cal_fused - adapted_local_fuse | +0.12 | -0.60 | -0.13 | +0.18 |
| cal_fused - cal_scalar | -0.03 | -0.36 | -0.15 | -0.14 |
| cal_fused - cal_spatial | +0.15 | -0.26 | -0.04 | +0.09 |
| cal_fused - base_global | -0.09 | +2.45 | +1.36 | +0.33 |

## Per-class pixel AP (%)

| class | `base_global` | `base_local_fuse` | `adapted_local_fuse` | `cal_scalar` | `cal_spatial` | `cal_fused` |
|---|---|---|---|---|---|---|
| bottle | 88.01 | 87.16 | 89.25 | 89.14 | 89.15 | 89.02 |
| cable | 67.80 | 59.72 | 61.65 | 64.36 | 60.30 | 63.09 |
| capsule | 60.95 | 60.09 | 62.04 | 62.13 | 63.10 | 63.04 |
| carpet | 69.46 | 75.48 | 75.72 | 74.91 | 75.76 | 74.92 |
| grid | 55.27 | 67.43 | 64.51 | 63.63 | 64.56 | 63.57 |
| hazelnut | 82.06 | 85.96 | 85.45 | 84.71 | 85.36 | 84.63 |
| leather | 50.43 | 59.78 | 58.65 | 57.40 | 58.71 | 57.43 |
| metal_nut | 79.63 | 82.18 | 82.38 | 81.97 | 82.26 | 81.79 |
| pill | 74.50 | 73.04 | 77.57 | 77.33 | 77.11 | 76.80 |
| screw | 61.27 | 62.63 | 64.64 | 64.29 | 64.15 | 63.89 |
| tile | 79.10 | 81.75 | 81.14 | 80.94 | 81.29 | 80.96 |
| toothbrush | 51.68 | 62.09 | 60.94 | 58.28 | 58.43 | 56.04 |
| transistor | 60.31 | 47.54 | 52.10 | 55.06 | 52.06 | 54.78 |
| wood | 72.69 | 78.50 | 77.74 | 76.72 | 77.88 | 76.70 |
| zipper | 79.12 | 86.08 | 84.30 | 83.61 | 82.97 | 82.45 |

## Identity and audit

- Image scores are computed once from the original baseline global map and reused for all variants (max |Δ| across variants = 0.000e+00).
- Recomputed baseline image scores vs stage2 original `results.base.global_scores`: 0.000e+00.
- Recomputed I-AUROC/I-AP/I-F1 vs stage2 reported baseline: n/a.
- Adapted-global branch computed: **no** (bug 1 fixed).
- Fixed-FPR negatives: **per variant, per class** (bug 2 fixed). Threshold values are in each per-class JSON.

## Calibration scalars

| class | g median | g q95 | g scale | l median | l q95 | l scale |
|---|---|---|---|---|---|---|
| bottle | 0.02994 | 0.04765 | 0.01771 | 0.04633 | 0.06924 | 0.02292 |
| cable | 0.04737 | 0.05918 | 0.01181 | 0.06097 | 0.08080 | 0.01984 |
| capsule | 0.02971 | 0.04447 | 0.01475 | 0.04414 | 0.06315 | 0.01900 |
| carpet | 0.04396 | 0.05226 | 0.00830 | 0.04963 | 0.06047 | 0.01084 |
| grid | 0.03284 | 0.04391 | 0.01107 | 0.04721 | 0.06069 | 0.01348 |
| hazelnut | 0.03794 | 0.05500 | 0.01707 | 0.04893 | 0.07715 | 0.02822 |
| leather | 0.04172 | 0.04925 | 0.00752 | 0.05303 | 0.06302 | 0.00998 |
| metal_nut | 0.04573 | 0.05843 | 0.01270 | 0.06378 | 0.08194 | 0.01815 |
| pill | 0.03716 | 0.04904 | 0.01188 | 0.04689 | 0.06282 | 0.01593 |
| screw | 0.03327 | 0.05093 | 0.01766 | 0.04189 | 0.07262 | 0.03074 |
| tile | 0.05034 | 0.05756 | 0.00722 | 0.05480 | 0.06476 | 0.00996 |
| toothbrush | 0.03680 | 0.05987 | 0.02307 | 0.04766 | 0.08682 | 0.03915 |
| transistor | 0.04101 | 0.05467 | 0.01367 | 0.05493 | 0.07787 | 0.02294 |
| wood | 0.04669 | 0.05380 | 0.00712 | 0.05450 | 0.06564 | 0.01114 |
| zipper | 0.03572 | 0.04495 | 0.00923 | 0.05364 | 0.06620 | 0.01256 |

Exact 24 selected train/good paths per class and the full background arrays are in `calibration/<class>.json` and `calibration/<class>.npz`. No test image is used for fitting.

Full per-class metrics, thresholds and deltas: `refine_report.json`.
