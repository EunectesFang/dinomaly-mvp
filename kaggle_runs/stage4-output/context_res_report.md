# Dinomaly context resolution v4 (stage 4)

Slug: `eunectesqaq/dinomaly-context-res-v4`  
Generated: 2026-11-05 11:26:24  
Checkpoint: `/kaggle/input/notebooks/eunectesqaq/dinomaly-base-10k/model_10000.pt` (sha256 `f5689b76aafa5c9d...`)  
Test images: 1725 across 15 classes  
Enabled high-res scales: [{'name': 'hi560', 'crop': 560, 'resize': 640, 'fov': 0.875, 'batch': 4}, {'name': 'hi672', 'crop': 672, 'resize': 768, 'fov': 0.875, 'batch': 2}]  

## What is compared

All branches use the same frozen 10k checkpoint. The baseline is exactly stage 3's global branch. High-res branches independently resize the native PIL image to 640/768, centre-crop 560/672 (same 7/8 FOV), and run the full image through ViTill; DINOv2 interpolates its position embeddings and ViTill derives the square token side from the actual token count.

| variant | definition |
|---|---|
| `base_global` | stage 3 baseline: full 448->392 global map |
| `hi560` | full-context 560 map alone |
| `fuse560` | 0.5 * base_global + 0.5 * hi560 |
| `hi672` | full-context 672 map alone |
| `fuse672` | 0.5 * base_global + 0.5 * hi672 |
| `fuse3` | 0.5 * base_global + 0.25 * hi560 + 0.25 * hi672 |

Every variant is evaluated on the same 256 grid against masks produced by the exact original 448->392 ground-truth transform. The image score is always the original baseline global top-1% score and is reused by every variant.

## 15-class means (%)

| variant | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|---|---|---|
| `base_global` | 99.62 | 99.81 | 99.15 | 98.33 | 68.82 | 69.04 | 94.67 |
| `hi560` | 99.62 | 99.81 | 99.15 | 98.24 | 72.25 | 71.12 | 95.05 |
| `fuse560` | 99.62 | 99.81 | 99.15 | 98.43 | 71.87 | 71.01 | 95.11 |
| `hi672` | 99.62 | 99.81 | 99.15 | 97.93 | 71.76 | 70.43 | 94.41 |
| `fuse672` | 99.62 | 99.81 | 99.15 | 98.39 | 72.47 | 71.37 | 95.27 |
| `fuse3` | 99.62 | 99.81 | 99.15 | 98.43 | 72.34 | 71.32 | 95.30 |

Image metrics are identical by construction. Only P-* columns and recall differ.

## Recall at fixed pixel FPR (class mean, %)

| variant | R@FPR0.001 | R@FPR0.01 | R@FPR0.05 |
|---|---|---|---|
| `base_global` | 18.12 | 65.48 | 94.50 |
| `hi560` | 22.81 | 70.19 | 93.46 |
| `fuse560` | 21.45 | 69.50 | 94.66 |
| `hi672` | 23.44 | 69.64 | 91.84 |
| `fuse672` | 22.30 | 70.45 | 94.32 |
| `fuse3` | 22.04 | 70.17 | 94.56 |

Thresholds are computed from each variant's own per-class test negative pixels, matching stage 3's corrected recall; they are metric thresholds, not model fitting.

## Mean pixel deltas (percentage points)

| comparison | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|
| hi560 - base_global | -0.09 | +3.43 | +2.08 | +0.38 |
| fuse560 - base_global | +0.10 | +3.05 | +1.97 | +0.43 |
| hi672 - base_global | -0.39 | +2.94 | +1.39 | -0.27 |
| fuse672 - base_global | +0.07 | +3.65 | +2.33 | +0.60 |
| fuse3 - base_global | +0.11 | +3.52 | +2.28 | +0.63 |
| fuse560 - hi560 | +0.20 | -0.38 | -0.11 | +0.06 |
| fuse672 - hi672 | +0.46 | +0.71 | +0.93 | +0.87 |
| fuse3 - fuse560 | +0.00 | +0.48 | +0.31 | +0.20 |

## Per-class pixel AP (%)

| class | `base_global` | `hi560` | `fuse560` | `hi672` | `fuse672` | `fuse3` |
|---|---|---|---|---|---|---|
| bottle | 88.01 | 89.80 | 89.44 | 89.59 | 89.60 | 89.59 |
| cable | 67.80 | 76.60 | 73.61 | 77.77 | 75.29 | 74.64 |
| capsule | 60.95 | 61.99 | 63.29 | 58.33 | 62.98 | 63.39 |
| carpet | 69.46 | 74.10 | 72.38 | 73.56 | 72.38 | 72.48 |
| grid | 55.27 | 61.98 | 60.12 | 62.96 | 60.93 | 60.75 |
| hazelnut | 82.06 | 85.90 | 84.93 | 86.36 | 85.58 | 85.38 |
| leather | 50.43 | 54.45 | 53.10 | 55.62 | 53.88 | 53.58 |
| metal_nut | 79.63 | 81.86 | 81.07 | 82.48 | 81.62 | 81.39 |
| pill | 74.50 | 72.60 | 74.83 | 69.73 | 74.25 | 74.72 |
| screw | 61.27 | 63.58 | 65.46 | 61.83 | 65.93 | 66.18 |
| tile | 79.10 | 82.61 | 81.36 | 83.81 | 82.13 | 81.80 |
| toothbrush | 51.68 | 61.75 | 57.76 | 66.66 | 60.81 | 59.43 |
| transistor | 60.31 | 55.51 | 59.48 | 52.69 | 58.95 | 59.46 |
| wood | 72.69 | 76.64 | 75.37 | 77.47 | 76.02 | 75.78 |
| zipper | 79.12 | 84.34 | 85.79 | 77.59 | 86.72 | 86.59 |

## Identity and audit

- Image score is the original baseline global top-1% and is reused by all variants (max |delta| across variants = 0.000e+00).
- No training was performed; the unchanged 10k checkpoint was reused.
- The high-res scales were fixed a priori and no model or calibration parameter was fitted. Test labels enter only through the standard metric definitions (AUROC/AP/F1/PRO and per-variant fixed-FPR recall), exactly as stage 3.
- High-res maps are computed on the full high-res image, then bilinearly reduced to 256 for metrics; raw high-res maps are not saved.
- Recomputed baseline vs stage 1 reported metrics: I-AUROC +0.0000 pp, I-AP -0.0000 pp, I-F1 -0.0000 pp, P-AUROC -0.0000 pp, P-AP -0.0000 pp, P-F1 -0.0000 pp, P-AUPRO +0.0000 pp
- Max per-class |delta| vs stage 1: I-AUROC 0.0000 pp, I-AP 0.0000 pp, I-F1 0.0000 pp, P-AUROC 0.0000 pp, P-AP 0.0000 pp, P-F1 0.0000 pp, P-AUPRO 0.0000 pp

## Runtime

- Device: `cuda:0`; torch `2.11.0+cu130`
- Classes: 15; test images: 1725
- Forward passes per test image: 3 (1 baseline + 2 high-res)
- Effective batches: baseline 4, scales {'560': 4, '672': 2}
- Probe: {"560": {"batch": 4, "crop": 560, "enabled": true, "peak_mib": 1992.6259765625, "resize": 640, "seconds": 0.7297420501708984, "seconds_per_image": 0.1824355125427246}, "672": {"batch": 2, "crop": 672, "enabled": true, "peak_mib": 1958.419921875, "resize": 768, "seconds": 0.6341798305511475, "seconds_per_image": 0.31708991527557373}, "base": {"batch": 4, "crop": 392, "enabled": true, "peak_mib": 1154.81640625, "resize": 448, "seconds": 3.986924409866333, "seconds_per_image": 0.9967311024665833}}
- Evaluation seconds (sum of per-class): 2187.9

Full per-class metrics, thresholds and deltas: `context_res_report.json`.
