# Dinomaly normal-reference v5 (stage 5)

Slug: `eunectesqaq/dinomaly-normal-reference-v5`  
Generated: 2026-11-05 12:18:52  
Checkpoint: `/kaggle/input/notebooks/eunectesqaq/dinomaly-base-10k/model_10000.pt` (sha256 `f5689b76aafa5c9d...`)  
Test images: 1725 across 15 classes  
Primary variant (pre-set): `ref_adaptive`  

## What is new

A class-specific normal-reference branch reuses the two encoder feature groups already returned by ViTill at 392 (two 768-channel 28x28 maps). It asks two questions: does a patch have a similar normal exemplar anywhere (appearance, `ref_free`), and does a similar exemplar occur within a small spatial neighborhood (structure, `ref_spatial`)? The structure branch is gated per class by how repeatable the spatial constraint is on held-out calibration normals.

## Normal-reference protocol

- 40 memory + 16 disjoint calibration images per class, sampled deterministically and evenly from sorted `train/good` paths. No test image is used for fitting.
- Two encoder groups are z-scored separately using memory-bank statistics, concatenated to 1536 dimensions, and L2-normalised. No dimensionality reduction; matching is chunked batched GEMM.
- `ref_free`: 1 - mean top-3 cosine similarity over all 40 memory images x 784 patches.
- `ref_spatial`: the same top-3 statistic over bank patches within +/-1 patch position, across all memory images. Out-of-image positions are masked, never padded into the candidate set.
- Per-branch normal calibration from the 16 calibration normals: `(x - median) / max(q95 - median, 0.01)`.
- Adaptive gate from calibration normals only: `rel_extra = median((d_spatial - d_free) / max(d_free, 0.02))`; g = 0.15 + 0.35 * exp(-max(rel_extra, 0) / 0.25), clipped to [0.15, 0.50]; rel_extra = median((d_spatial - d_free) / max(d_free, 0.02)) over calibration normals. Higher relative extra distance lowers spatial weight. The formula and constants are fixed; no class weights are hard-coded and no test labels are used.

## Variants (7, fixed before evaluation)

| variant | definition |
|---|---|
| `base_global` | original 392 baseline map (448->392), unchanged |
| `ref_free` | 1 - mean top-3 cosine sim over all normal-bank patches (appearance) |
| `ref_spatial` | 1 - mean top-3 cosine sim over normal-bank patches within +/-1 patch position across all memory images (structure) |
| `ref_fuse_fixed` | 0.5 * calib(ref_free) + 0.5 * calib(ref_spatial) |
| `ref_adaptive` | (1-g) * calib(ref_free) + g * calib(ref_spatial); per-class g from calibration-normal repeatability (PRIMARY) **PRIMARY** |
| `base_fuse672` | archived stage-4 392+672 baseline: 0.5 * base_global + 0.5 * hi672 |
| `ref_plus_base` | 0.5 * calib(base_fuse672) + 0.5 * ref_adaptive |

All variants use the same official 256 grid, the exact original 448->392 mask transform, and the original baseline global top-1% image score.

## 15-class means (%)

| variant | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|---|---|---|
| `base_global` | 99.62 | 99.81 | 99.15 | 98.33 | 68.82 | 69.04 | 94.67 |
| `ref_free` | 99.62 | 99.81 | 99.15 | 94.54 | 60.27 | 61.06 | 87.01 |
| `ref_spatial` | 99.62 | 99.81 | 99.15 | 94.80 | 59.54 | 60.84 | 89.18 |
| `ref_fuse_fixed` | 99.62 | 99.81 | 99.15 | 94.94 | 62.03 | 62.59 | 89.17 |
| `ref_adaptive` **PRIMARY** | 99.62 | 99.81 | 99.15 | 94.93 | 61.97 | 62.55 | 89.00 |
| `base_fuse672` | 99.62 | 99.81 | 99.15 | 98.39 | 72.47 | 71.37 | 95.27 |
| `ref_plus_base` | 99.62 | 99.81 | 99.15 | 98.22 | 72.58 | 71.17 | 95.28 |

Image metrics are identical by construction. Only P-* columns and recall differ.

## Recall at fixed pixel FPR (class mean, %)

| variant | R@FPR0.001 | R@FPR0.01 | R@FPR0.05 |
|---|---|---|---|
| `base_global` | 18.12 | 65.48 | 94.50 |
| `ref_free` | 19.71 | 55.05 | 81.60 |
| `ref_spatial` | 16.66 | 56.21 | 83.50 |
| `ref_fuse_fixed` | 19.54 | 57.66 | 83.92 |
| `ref_adaptive` | 19.70 | 57.45 | 83.85 |
| `base_fuse672` | 22.30 | 70.45 | 94.32 |
| `ref_plus_base` | 22.51 | 70.02 | 94.12 |

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
| ref_free - base_global | -3.79 | -8.55 | -7.98 | -7.67 |
| ref_spatial - base_global | -3.52 | -9.28 | -8.20 | -5.50 |
| ref_fuse_fixed - base_global | -3.38 | -6.79 | -6.46 | -5.51 |
| ref_adaptive - base_global | -3.39 | -6.85 | -6.49 | -5.68 |
| base_fuse672 - base_global | +0.07 | +3.65 | +2.33 | +0.60 |
| ref_plus_base - base_global | -0.10 | +3.76 | +2.13 | +0.61 |
| ref_spatial - ref_free | +0.27 | -0.72 | -0.22 | +2.17 |
| ref_adaptive - ref_fuse_fixed | -0.01 | -0.06 | -0.04 | -0.17 |
| ref_plus_base - base_fuse672 | -0.17 | +0.11 | -0.20 | +0.01 |
| ref_plus_base - ref_adaptive | +3.29 | +10.61 | +8.62 | +6.29 |

## Per-class pixel AP (%)

| class | `base_global` | `ref_free` | `ref_spatial` | `ref_fuse_fixed` | `ref_adaptive` | `base_fuse672` | `ref_plus_base` |
|---|---|---|---|---|---|---|---|
| bottle | 88.01 | 89.67 | 87.80 | 89.12 | 89.12 | 89.60 | 90.35 |
| cable | 67.80 | 70.23 | 73.75 | 72.92 | 72.90 | 75.29 | 77.01 |
| capsule | 60.95 | 52.23 | 47.09 | 50.77 | 50.77 | 62.98 | 61.94 |
| carpet | 69.46 | 52.07 | 69.06 | 65.28 | 63.40 | 72.38 | 72.59 |
| grid | 55.27 | 60.76 | 57.57 | 61.08 | 61.28 | 60.93 | 62.27 |
| hazelnut | 82.06 | 84.41 | 81.81 | 84.40 | 84.46 | 85.58 | 86.53 |
| leather | 50.43 | 26.93 | 39.07 | 34.40 | 33.27 | 53.88 | 53.07 |
| metal_nut | 79.63 | 75.85 | 76.00 | 76.57 | 76.57 | 81.62 | 81.66 |
| pill | 74.50 | 56.34 | 50.84 | 54.06 | 54.06 | 74.25 | 70.88 |
| screw | 61.27 | 44.21 | 12.84 | 40.23 | 42.25 | 65.93 | 67.96 |
| tile | 79.10 | 61.50 | 63.69 | 62.94 | 62.66 | 82.13 | 80.81 |
| toothbrush | 51.68 | 69.43 | 61.52 | 65.26 | 65.26 | 60.81 | 63.76 |
| transistor | 60.31 | 58.16 | 74.37 | 70.15 | 70.15 | 58.95 | 68.48 |
| wood | 72.69 | 63.69 | 61.07 | 64.92 | 64.99 | 76.02 | 75.86 |
| zipper | 79.12 | 38.54 | 36.66 | 38.36 | 38.45 | 86.72 | 75.50 |

## Archived baseline check

- `base_global_vs_stage4_base_global`: max |delta| 0.0000 pp; P-AP +0.0000 pp, P-AUROC +0.0000 pp, P-AUPRO -0.0000 pp
- `base_fuse672_vs_stage4_fuse672`: max |delta| 0.0000 pp; P-AP +0.0000 pp, P-AUROC +0.0000 pp, P-AUPRO +0.0000 pp
- Image means vs stage 4: I-AUROC +0.0000 pp, I-AP +0.0000 pp, I-F1 +0.0000 pp
- The archived stage-4 392+672 baseline is `fuse672` P-AP = 72.47% on the same official protocol.

## Audit and runtime

- Image score is the original baseline global top-1% and is reused by all variants (max |delta| = 0.000e+00).
- No training; the unchanged frozen 10k checkpoint is reused.
- The normal-reference bank and calibration/gate statistics come only from `train/good`. Test labels enter only through the standard metric definitions (AUROC/AP/F1/PRO and fixed-FPR recall).
- High-res 672 maps are computed on the full native image and reduced to 256 for metrics; no raw high-res maps are saved.
- Recomputed base_global vs stage 1: I-AUROC +0.0000 pp, I-AP -0.0000 pp, I-F1 -0.0000 pp, P-AUROC -0.0000 pp, P-AP -0.0000 pp, P-F1 -0.0000 pp, P-AUPRO +0.0000 pp

- Device: `cuda:0`; torch `2.11.0+cu130`
- Classes: 15; test images: 1725
- Forward passes per test image: 2 (1 baseline + 1 hi672)
- Effective batches: base 4, hi672 2, reference chunk 2
- Evaluation seconds (sum of per-class): 1976.8

Full per-class metrics, thresholds, calibration and gate values: `normal_reference_report.json`.
