# Dinomaly conservative reference v6 (stage 6)

Slug: `eunectesqaq/dinomaly-aligned-reference-v6`  
Generated: 2026-11-05 13:00:44  
Checkpoint: `/kaggle/input/notebooks/eunectesqaq/dinomaly-base-10k/model_10000.pt` (sha256 `f5689b76aafa5c9d...`)  
Test images: 1725 across 15 classes  
Primary variant (pre-set): `ref_conservative`  

## What is new

A class-specific normal-reference branch reuses the two encoder feature groups already returned by ViTill at 392 (two 768-channel 28x28 maps). It asks two questions: does a patch have a similar normal exemplar anywhere (appearance, `ref_free`), and does a similar exemplar occur within a small spatial neighborhood (structure, `ref_spatial`)? The structure branch is gated per class by how repeatable the spatial constraint is on held-out calibration normals. Stage 6 adds a conservative fusion: the calibrated 392+672 base is only increased where the reference exceeds its calibrated q99 and both maps agree locally; the positive residual is capped and nothing is subtracted. Pose-aligned matching was deferred to protect the deadline, so every stage-6 variant is `unaligned`.

## Normal-reference protocol

- 40 memory + 16 disjoint calibration images per class, sampled deterministically and evenly from sorted `train/good` paths. No test image is used for fitting.
- Two encoder groups are z-scored separately using memory-bank statistics, concatenated to 1536 dimensions, and L2-normalised. No dimensionality reduction; matching is chunked batched GEMM.
- `ref_free`: 1 - mean top-3 cosine similarity over all 40 memory images x 784 patches.
- `ref_spatial`: the same top-3 statistic over bank patches within +/-1 patch position, across all memory images. Out-of-image positions are masked, never padded into the candidate set.
- Per-branch normal calibration from the 16 calibration normals: `(x - median) / max(q95 - median, 0.01)`.
- Adaptive gate from calibration normals only: `rel_extra = median((d_spatial - d_free) / max(d_free, 0.02))`; g = 0.15 + 0.35 * exp(-max(rel_extra, 0) / 0.25), clipped to [0.15, 0.50]; rel_extra = median((d_spatial - d_free) / max(d_free, 0.02)) over calibration normals. Higher relative extra distance lowers spatial weight. The formula and constants are fixed; no class weights are hard-coded and no test labels are used.

## Variants (5, fixed before evaluation)

| variant | definition |
|---|---|
| `base_fuse672` | archived stage-4 392+672 baseline: 0.5 * base_global + 0.5 * hi672 |
| `ref_adaptive` | (1-g) * calib(ref_free) + g * calib(ref_spatial); stage-5 normal-reference map (no base score) |
| `ref_plus_base` | stage-5 reproduction: 0.5 * calib(base_fuse672) + 0.5 * ref_adaptive |
| `ref_conservative` | calib(base_fuse672) + 0.25 * capped positive ref residual above calibrated q99, only where both maps agree in a 3x3 neighbourhood (PRIMARY, unaligned) **PRIMARY** |
| `ref_conservative_strong` | same gate with base must exceed calibrated q95, residual cap 1.0 and weight 0.5 (unaligned) |

All variants use the same official 256 grid, the exact original 448->392 mask transform, and the original baseline global top-1% image score. Conservative formula: `score = calib(base_fuse672) + w * clamp(calib(ref_spatial) - calib_q99, 0, cap) * agree; agree = 3x3 max-pool of [calib(base)>base_thr]; nothing is subtracted from base; pose alignment intentionally not applied in this run`.

## 15-class means (%)

| variant | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|---|---|---|
| `base_fuse672` | 99.62 | 99.81 | 99.15 | 98.39 | 72.47 | 71.37 | 95.27 |
| `ref_adaptive` | 99.62 | 99.81 | 99.15 | 94.93 | 61.97 | 62.55 | 89.00 |
| `ref_plus_base` | 99.62 | 99.81 | 99.15 | 98.22 | 72.58 | 71.17 | 95.28 |
| `ref_conservative` **PRIMARY** | 99.62 | 99.81 | 99.15 | 98.46 | 72.94 | 71.77 | 95.44 |
| `ref_conservative_strong` | 99.62 | 99.81 | 99.15 | 98.44 | 72.90 | 71.77 | 95.38 |

Image metrics are identical by construction. Only P-* columns and recall differ.

## Recall at fixed pixel FPR (class mean, %)

| variant | R@FPR0.001 | R@FPR0.01 | R@FPR0.05 |
|---|---|---|---|
| `base_fuse672` | 22.30 | 70.45 | 94.32 |
| `ref_adaptive` | 19.70 | 57.45 | 83.85 |
| `ref_plus_base` | 22.51 | 70.02 | 94.12 |
| `ref_conservative` | 22.40 | 71.06 | 94.63 |
| `ref_conservative_strong` | 22.33 | 70.98 | 94.70 |

Thresholds use each variant's own per-class test negative pixels, exactly as stages 3/4; they are metric thresholds, not fitting.

## Per-class calibration and adaptive gate

| class | mem | calib | rel_extra | g_spatial | free med | free q95 | free scale | spatial med | spatial q95 | spatial scale | base672 med | base672 q95 | base672 scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bottle | 40 | 16 | 0.0000 | 0.5000 | 0.15430 | 0.28144 | 0.12714 | 0.15502 | 0.28647 | 0.13145 | 0.04435 | 0.06150 | 0.01715 |
| cable | 40 | 16 | 0.0036 | 0.4951 | 0.32669 | 0.43815 | 0.11146 | 0.33635 | 0.45970 | 0.12335 | 0.05584 | 0.07599 | 0.02015 |
| capsule | 40 | 16 | 0.0000 | 0.5000 | 0.17346 | 0.27062 | 0.09717 | 0.17548 | 0.27774 | 0.10226 | 0.03943 | 0.06173 | 0.02231 |
| carpet | 40 | 16 | 0.0861 | 0.3980 | 0.43812 | 0.50415 | 0.06603 | 0.48357 | 0.55171 | 0.06814 | 0.07066 | 0.07914 | 0.01000 |
| grid | 40 | 16 | 0.3624 | 0.2321 | 0.28016 | 0.35977 | 0.07961 | 0.38797 | 0.47589 | 0.08793 | 0.05049 | 0.06190 | 0.01141 |
| hazelnut | 40 | 16 | 0.0365 | 0.4525 | 0.31591 | 0.46827 | 0.15236 | 0.33194 | 0.52825 | 0.19630 | 0.04665 | 0.06516 | 0.01851 |
| leather | 40 | 16 | 0.0684 | 0.4163 | 0.41089 | 0.48658 | 0.07568 | 0.44961 | 0.52714 | 0.07753 | 0.05641 | 0.06283 | 0.01000 |
| metal_nut | 40 | 16 | 0.0000 | 0.5000 | 0.27675 | 0.46100 | 0.18425 | 0.28886 | 0.48239 | 0.19353 | 0.05826 | 0.07169 | 0.01342 |
| pill | 40 | 16 | 0.0000 | 0.5000 | 0.26920 | 0.39071 | 0.12151 | 0.27827 | 0.41160 | 0.13333 | 0.04575 | 0.05969 | 0.01394 |
| screw | 40 | 16 | 0.0661 | 0.4187 | 0.30799 | 0.42183 | 0.11384 | 0.34172 | 0.47850 | 0.13678 | 0.03915 | 0.06802 | 0.02887 |
| tile | 40 | 16 | 0.0999 | 0.3847 | 0.49892 | 0.56617 | 0.06725 | 0.55618 | 0.62529 | 0.06911 | 0.05839 | 0.06586 | 0.01000 |
| toothbrush | 40 | 16 | 0.0000 | 0.5000 | 0.22069 | 0.33796 | 0.11727 | 0.22559 | 0.35024 | 0.12466 | 0.04353 | 0.07903 | 0.03549 |
| transistor | 40 | 16 | 0.0000 | 0.5000 | 0.23695 | 0.34615 | 0.10920 | 0.24332 | 0.37307 | 0.12975 | 0.04969 | 0.06852 | 0.01884 |
| wood | 40 | 16 | 0.0932 | 0.3911 | 0.51080 | 0.58337 | 0.07258 | 0.56967 | 0.64463 | 0.07496 | 0.05248 | 0.06035 | 0.01000 |
| zipper | 40 | 16 | 0.0482 | 0.4386 | 0.27807 | 0.38539 | 0.10732 | 0.29932 | 0.41695 | 0.11762 | 0.05949 | 0.08930 | 0.02981 |

## Mean pixel deltas (percentage points)

| comparison | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|
| ref_adaptive - base_fuse672 | -3.46 | -10.50 | -8.82 | -6.28 |
| ref_plus_base - base_fuse672 | -0.17 | +0.11 | -0.20 | +0.01 |
| ref_conservative - base_fuse672 | +0.07 | +0.47 | +0.40 | +0.17 |
| ref_conservative_strong - base_fuse672 | +0.04 | +0.42 | +0.41 | +0.10 |
| ref_plus_base - ref_adaptive | +3.29 | +10.61 | +8.62 | +6.29 |
| ref_conservative - ref_plus_base | +0.24 | +0.36 | +0.60 | +0.16 |
| ref_conservative_strong - ref_conservative | -0.03 | -0.05 | +0.00 | -0.06 |

## Per-class pixel AP (%)

| class | `base_fuse672` | `ref_adaptive` | `ref_plus_base` | `ref_conservative` | `ref_conservative_strong` |
|---|---|---|---|---|---|
| bottle | 89.60 | 89.12 | 90.35 | 89.77 | 89.64 |
| cable | 75.29 | 72.90 | 77.01 | 76.66 | 76.61 |
| capsule | 62.98 | 50.77 | 61.94 | 63.16 | 63.03 |
| carpet | 72.38 | 63.40 | 72.59 | 72.50 | 72.47 |
| grid | 60.93 | 61.28 | 62.27 | 61.05 | 60.94 |
| hazelnut | 85.58 | 84.46 | 86.53 | 85.86 | 86.09 |
| leather | 53.88 | 33.27 | 53.07 | 53.90 | 53.89 |
| metal_nut | 81.62 | 76.57 | 81.66 | 81.73 | 81.79 |
| pill | 74.25 | 54.06 | 70.88 | 73.94 | 73.76 |
| screw | 65.93 | 42.25 | 67.96 | 66.90 | 67.07 |
| tile | 82.13 | 62.66 | 80.81 | 81.92 | 81.98 |
| toothbrush | 60.81 | 65.26 | 63.76 | 61.81 | 61.68 |
| transistor | 58.95 | 70.15 | 68.48 | 62.51 | 62.50 |
| wood | 76.02 | 64.99 | 75.86 | 75.98 | 75.93 |
| zipper | 86.72 | 38.45 | 75.50 | 86.44 | 86.05 |

## Archived baseline check

- `base_fuse672_vs_stage4_fuse672`: max |delta| 0.0000 pp; P-AP +0.0000 pp, P-AUROC +0.0000 pp, P-AUPRO +0.0000 pp
- Image means vs stage 4: I-AUROC +0.0000 pp, I-AP +0.0000 pp, I-F1 +0.0000 pp
- The archived stage-4 392+672 baseline is `fuse672` P-AP = 72.47% on the same official protocol.

## Audit and runtime

- Image score is the original baseline global top-1% and is reused by all variants (max |delta| = 0.000e+00).
- No training; the unchanged frozen 10k checkpoint is reused.
- The normal-reference bank and calibration/gate statistics come only from `train/good`. Test labels enter only through the standard metric definitions (AUROC/AP/F1/PRO and fixed-FPR recall).
- High-res 672 maps are computed on the full native image and reduced to 256 for metrics; no raw high-res maps are saved.
- Stage 1 metrics unavailable; identity check skipped.

- Device: `cuda:0`; torch `2.11.0+cu130`
- Classes: 15; test images: 1725
- Forward passes per test image: 2 (1 baseline + 1 hi672)
- Effective batches: base 4, hi672 2, reference chunk 2
- Evaluation seconds (sum of per-class): 1605.5

Full per-class metrics, thresholds, calibration and gate values: `normal_reference_report.json`.
