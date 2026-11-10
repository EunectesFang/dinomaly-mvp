# Dinomaly aligned-reference v8 (targeted diagnostic)

Slug: `eunectesqaq/dinomaly-aligned-diagnostic-v8`  
Generated: 2026-11-05 13:07:30  
Checkpoint: `/kaggle/input/dinomaly-base-10k/model_10000.pt` (sha256 `f5689b76aafa5c9d...`)  
Test images: 578 across 4 classes  
Primary variant (pre-set): `aligned_plus_base`  

## TARGETED DIAGNOSTIC - NOT A FULL BENCHMARK

Only the four official classes selected after the full 15-class test set was initialised are evaluated: `transistor`, `zipper`, `pill`, `screw` (578 official test images).  These are v5 win/loss/pose cases, chosen before this run.  Do not read the class means as a 15-class benchmark result.

## Core independent variable

The stage-5 same-coordinate spatial search is replaced by a per-(test image, memory image) coherent shift alignment. Each pair selects one shift `(sr, sc)` in `[-2, 2]^2` by maximising the mean cosine similarity over in-image query positions, minus a 0.005 * (|sr| + |sc|) Manhattan prior. Candidates are the +/-1 neighbourhood around the shifted coordinate, preserving local tolerance. Out-of-image positions are masked. A query row with fewer than 3 finite aligned candidates falls back to the original zero-shift neighbourhood. The free appearance search is unchanged.

## Normal-reference protocol

- 40 memory + 16 disjoint calibration images per class, sampled deterministically and evenly from sorted `train/good` paths. No test image is used for fitting.
- Two encoder groups are z-scored separately using memory-bank statistics, concatenated to 1536 dimensions, and L2-normalised. No dimensionality reduction; matching is chunked batched GEMM.
- `ref_free`: 1 - mean top-3 cosine similarity over all 40 memory images x 784 patches (unchanged stage 5).
- `ref_spatial_aligned`: 1 - mean top-k cosine similarity over the shifted +/-1 neighbourhood, as described above.
- Per-branch normal calibration from the 16 calibration normals: `(x - median) / max(q95 - median, 0.01)`; the aligned spatial branch is recalibrated on normals.
- Adaptive gate from calibration normals only: `rel_extra = median((d_aligned - d_free) / max(d_free, 0.02))`; g = 0.15 + 0.35 * exp(-max(rel_extra, 0) / 0.25), clipped to [0.15, 0.50]; rel_extra = median((d_aligned - d_free) / max(d_free, 0.02)) over calibration normals. Higher relative extra distance lowers spatial weight. The formula and constants are fixed; no class weights are hard-coded and no test labels are used.

## Variants (4, fixed before evaluation)

| variant | definition |
|---|---|
| `base_fuse672` | archived stage-4 392+672 baseline: 0.5 * base_global + 0.5 * hi672 (control) |
| `ref_spatial_aligned` | raw aligned spatial normal-reference map: 1 - mean top-3 cosine sim around the per-(test image, memory image) coherent shift selected in [-2,2]^2, +/-1 local tolerance |
| `ref_adaptive_aligned` | (1-g) * calib(ref_free) + g * calib(ref_spatial_aligned); g from aligned calibration normals with the exact stage-5 gate |
| `aligned_plus_base` | 0.5 * calib(base_fuse672) + 0.5 * ref_adaptive_aligned (stage-5 ref_plus_base formula with the aligned branch; PRIMARY) **PRIMARY** |

All variants use the same official 256 grid, the exact original 448->392 mask transform, and the original baseline global top-1% image score.

## 4-class means (diagnostic subset) (%)

| variant | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|---|---|---|
| `base_fuse672` | 99.17 | 99.42 | 98.09 | 97.26 | 71.46 | 68.44 | 92.56 |
| `ref_spatial_aligned` | 99.17 | 99.42 | 98.09 | 89.13 | 43.94 | 47.21 | 81.24 |
| `ref_adaptive_aligned` | 99.17 | 99.42 | 98.09 | 89.59 | 51.30 | 53.26 | 81.43 |
| `aligned_plus_base` **PRIMARY** | 99.17 | 99.42 | 98.09 | 97.04 | 70.84 | 67.54 | 93.57 |

Image metrics are identical by construction. Only P-* columns and recall differ.

## Recall at fixed pixel FPR (class mean, %)

| variant | R@FPR0.001 | R@FPR0.01 | R@FPR0.05 |
|---|---|---|---|
| `base_fuse672` | 32.79 | 72.15 | 87.06 |
| `ref_spatial_aligned` | 15.52 | 45.13 | 66.65 |
| `ref_adaptive_aligned` | 21.18 | 49.05 | 68.05 |
| `aligned_plus_base` | 31.11 | 68.91 | 87.80 |

Thresholds use each variant's own per-class test negative pixels, exactly as stages 3/4; they are metric thresholds, not fitting.

## Shift-alignment diagnostics

| class | calib zero frac | calib fallback frac | test zero frac | test fallback frac | test mean |sr| | test mean |sc| | invalid shift pairs |
|---|---|---|---|---|---|---|---|
| transistor | 0.5328 | 0.000000 | 0.6623 | 0.000000 | 0.3310 | 0.0798 | 0 |
| zipper | 0.6859 | 0.000000 | 0.6962 | 0.000000 | 0.2781 | 0.0418 | 0 |
| pill | 1.0000 | 0.000000 | 1.0000 | 0.000000 | 0.0000 | 0.0000 | 0 |
| screw | 0.9516 | 0.000000 | 0.9523 | 0.000000 | 0.0430 | 0.0330 | 0 |

A nonzero fallback fraction is the explicit edge case: a shifted +/-1 neighbourhood left fewer than top-3 finite candidates for that query row, so the zero-shift neighbourhood was used instead.

## Per-class calibration and adaptive gate

| class | mem | calib | rel_extra (aligned) | g_spatial | free med | free q95 | free scale | aligned med | aligned q95 | aligned scale | base672 med | base672 q95 | base672 scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| transistor | 40 | 16 | 0.0000 | 0.5000 | 0.23695 | 0.34615 | 0.10920 | 0.24205 | 0.36351 | 0.12146 | 0.04969 | 0.06852 | 0.01884 |
| zipper | 40 | 16 | 0.0554 | 0.4304 | 0.27807 | 0.38539 | 0.10732 | 0.30205 | 0.42026 | 0.11821 | 0.05949 | 0.08930 | 0.02981 |
| pill | 40 | 16 | 0.0000 | 0.5000 | 0.26920 | 0.39071 | 0.12151 | 0.27827 | 0.41160 | 0.13333 | 0.04575 | 0.05969 | 0.01394 |
| screw | 40 | 16 | 0.0625 | 0.4226 | 0.30799 | 0.42183 | 0.11384 | 0.33948 | 0.47605 | 0.13657 | 0.03915 | 0.06802 | 0.02887 |

## Mean pixel deltas (percentage points)

| comparison | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|
| aligned_plus_base - base_fuse672 | -0.22 | -0.62 | -0.90 | +1.01 |
| aligned_plus_base - ref_adaptive_aligned | +7.45 | +19.54 | +14.28 | +12.14 |
| ref_adaptive_aligned - base_fuse672 | -7.67 | -20.16 | -15.17 | -11.13 |
| ref_spatial_aligned - base_fuse672 | -8.12 | -27.52 | -21.23 | -11.32 |

## Per-class pixel AP (%)

| class | `base_fuse672` | `ref_spatial_aligned` | `ref_adaptive_aligned` | `aligned_plus_base` |
|---|---|---|---|---|
| transistor | 58.95 | 75.55 | 70.85 | 69.00 |
| zipper | 86.72 | 36.08 | 38.24 | 75.47 |
| pill | 74.25 | 50.84 | 54.06 | 70.88 |
| screw | 65.93 | 13.28 | 42.05 | 68.02 |

## Downloaded v5 control (unchanged)

v5 class means on the same four classes (%):

| v5 variant | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|
| `base_fuse672` | 97.26 | 71.46 | 68.44 | 92.56 |
| `ref_spatial` | 89.15 | 43.68 | 46.78 | 80.86 |
| `ref_plus_base` | 97.02 | 70.70 | 67.37 | 93.48 |

This-run minus v5 (percentage points):

| comparison | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|
| base_fuse672 - v5_base_fuse672 | +0.00 | +0.00 | +0.00 | +0.00 |
| ref_spatial_aligned - v5_ref_spatial | -0.01 | +0.26 | +0.44 | +0.38 |
| aligned_plus_base - v5_ref_plus_base | +0.01 | +0.14 | +0.17 | +0.10 |

## Archived baseline check

- `base_fuse672_vs_stage4_fuse672`: max |delta| 2.9275 pp; P-AP -1.0090 pp, P-AUROC -1.1366 pp, P-AUPRO -2.7098 pp
- Image means vs stage 4: I-AUROC -0.4545 pp, I-AP -0.3875 pp, I-F1 -1.0627 pp

## Audit and runtime

- Image score is the original baseline global top-1% and is reused by all variants (max |delta| = 0.000e+00).
- No training; the unchanged frozen 10k checkpoint is reused.
- Only `train/good` images form the memory bank, and the calibration set is disjoint from it. Test labels enter only through standard metric definitions.
- High-res 672 maps are computed on the full native image and reduced to 256 for metrics; no raw high-res maps are saved.
- The stage-5 control is a downloaded JSON and was not refitted or altered.

- Device: `cuda:0`; torch `2.11.0+cu130`
- Classes: 4; test images: 578
- Forward passes per test image: 2 (1 baseline + 1 hi672)
- Effective batches: base 4, hi672 2, reference chunk 2
- Evaluation seconds (sum of per-class): 484.5

Full per-class metrics, thresholds, shift diagnostics and calibration values: `aligned_diagnostic_report.json`.
