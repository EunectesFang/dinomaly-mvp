### Stage 1 — official-protocol 10k baseline (I-AUROC/I-AP/I-F1/P-AUROC/P-AP/P-F1/P-AUPRO, %)

| iter | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|---|---|---|
| 10000 | 99.62 | 99.81 | 99.15 | 98.33 | 68.82 | 69.04 | 94.67 |

### Stage 1 per class @ 10000

| class | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|---|---|---|
| bottle | 100.00 | 100.00 | 100.00 | 99.13 | 88.01 | 84.00 | 96.86 |
| cable | 100.00 | 100.00 | 100.00 | 98.33 | 67.80 | 70.99 | 93.99 |
| capsule | 98.05 | 99.58 | 97.72 | 98.68 | 60.95 | 60.51 | 97.42 |
| carpet | 99.88 | 99.96 | 98.89 | 99.34 | 69.46 | 71.41 | 97.54 |
| grid | 99.83 | 99.94 | 99.11 | 99.39 | 55.27 | 57.83 | 97.44 |
| hazelnut | 100.00 | 100.00 | 100.00 | 99.45 | 82.06 | 77.05 | 97.14 |
| leather | 100.00 | 100.00 | 100.00 | 99.29 | 50.43 | 53.14 | 97.17 |
| metal_nut | 100.00 | 100.00 | 100.00 | 96.92 | 79.63 | 86.91 | 94.86 |
| pill | 99.21 | 99.86 | 98.58 | 97.67 | 74.50 | 70.87 | 97.45 |
| screw | 98.34 | 99.47 | 96.27 | 99.64 | 61.27 | 60.93 | 98.37 |
| tile | 100.00 | 100.00 | 100.00 | 98.13 | 79.10 | 76.03 | 90.42 |
| toothbrush | 100.00 | 100.00 | 100.00 | 98.91 | 51.68 | 63.14 | 95.09 |
| transistor | 99.12 | 98.36 | 97.50 | 93.33 | 60.31 | 58.77 | 75.87 |
| wood | 99.91 | 99.97 | 99.17 | 97.52 | 72.69 | 68.57 | 93.19 |
| zipper | 100.00 | 100.00 | 100.00 | 99.18 | 79.12 | 75.43 | 97.27 |

### Stage 2 — 15-class means

| variant | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|---|---|---|
| Baseline global (paper) | 99.62 | 99.81 | 99.15 | 98.33 | 68.82 | 69.04 | 94.67 |
| Baseline + local fusion (no adapt) | 96.14 | 98.00 | 95.45 | 95.28 | 64.89 | 63.91 | 90.37 |
| Baseline + 0.5 fusion (no adapt) | 99.23 | 99.59 | 98.53 | 97.82 | 71.29 | 70.06 | 94.34 |
| Adapted local fusion | 98.29 | 98.76 | 97.41 | 96.33 | 70.71 | 69.24 | 93.09 |
| Adapted 0.5 fusion | 98.63 | 99.20 | 97.96 | 96.51 | 69.06 | 67.99 | 93.32 |
| Adapted global | 98.24 | 99.17 | 97.47 | 95.82 | 61.35 | 62.16 | 91.76 |
| Full-image extra global | 99.60 | 99.76 | 99.05 | 98.30 | 69.17 | 68.95 | 94.52 |
| Full-image extra local fusion | 95.95 | 97.94 | 95.27 | 95.12 | 65.22 | 64.03 | 90.21 |
| Full-image extra 0.5 fusion | 99.15 | 99.54 | 98.40 | 97.70 | 71.51 | 69.98 | 94.12 |

### Recall at fixed pixel FPR (class mean, %)

| variant | R@FPR0.1% | R@FPR1% | R@FPR5% |
|---|---|---|---|
| Baseline global (paper) | 18.12 | 65.48 | 94.50 |
| Baseline + local fusion (no adapt) | 34.94 | 84.84 | 99.03 |
| Baseline + 0.5 fusion (no adapt) | 26.41 | 76.83 | 98.40 |
| Adapted local fusion | 4.33 | 28.86 | 66.50 |
| Adapted 0.5 fusion | 8.01 | 43.13 | 79.20 |
| Full-image extra local fusion | 37.77 | 86.19 | 99.07 |
| Full-image extra 0.5 fusion | 28.25 | 78.99 | 98.31 |

### Mean anomaly score by region (class mean)

| variant | interior | inner_edge | outer_neighborhood | far_normal |
|---|---|---|---|---|
| Baseline global (paper) | 0.1506 | 0.1367 | 0.1158 | 0.0471 |
| Baseline + local fusion (no adapt) | 0.1729 | 0.1629 | 0.1360 | 0.0868 |
| Baseline + 0.5 fusion (no adapt) | 0.1618 | 0.1498 | 0.1259 | 0.0669 |
| Adapted local fusion | 0.1202 | 0.1122 | 0.0923 | 0.0537 |
| Adapted 0.5 fusion | 0.1286 | 0.1212 | 0.1057 | 0.0618 |
| Full-image extra local fusion | 0.1688 | 0.1591 | 0.1350 | 0.0879 |
| Full-image extra 0.5 fusion | 0.1572 | 0.1458 | 0.1242 | 0.0682 |

### Headline deltas (percentage points, class-mean)

| comparison | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|---|---|---|
| adapted_local - base_global | -1.33 | -1.05 | -1.74 | -2.00 | +1.89 | +0.20 | -1.58 |
| adapted_local - base_local | +2.15 | +0.77 | +1.97 | +1.05 | +5.82 | +5.33 | +2.72 |
| adapted_local - extra_local | +2.34 | +0.83 | +2.15 | +1.21 | +5.49 | +5.21 | +2.88 |
| adapted_fuse - base_global | -1.00 | -0.61 | -1.19 | -1.82 | +0.24 | -1.05 | -1.35 |
| adapted_fuse - base_fuse | -0.60 | -0.39 | -0.57 | -1.31 | -2.24 | -2.07 | -1.02 |
| adapted_fuse - extra_fuse | -0.52 | -0.35 | -0.44 | -1.19 | -2.45 | -1.99 | -0.80 |
| extra_local - base_local | -0.18 | -0.06 | -0.18 | -0.16 | +0.34 | +0.13 | -0.16 |
| extra_fuse - base_fuse | -0.08 | -0.04 | -0.14 | -0.12 | +0.21 | -0.08 | -0.23 |
| extra_global - base_global | -0.03 | -0.04 | -0.10 | -0.03 | +0.35 | -0.09 | -0.15 |

### Per-class pixel AP delta: adapted local minus baseline global (pp)

| class | P-AP Δ | P-F1 Δ | P-AUROC Δ | P-AUPRO Δ |
|---|---|---|---|---|
| bottle | +0.08 | -1.25 | -0.38 | -0.53 |
| cable | -24.16 | -22.87 | -14.02 | -13.43 |
| capsule | -4.88 | -3.29 | -1.95 | -1.45 |
| carpet | +9.60 | +6.13 | -0.38 | +0.13 |
| grid | +15.91 | +12.74 | +0.27 | +0.17 |
| hazelnut | +3.34 | +2.97 | -0.04 | +0.46 |
| leather | +16.87 | +11.17 | +0.32 | +1.97 |
| metal_nut | +4.25 | -2.75 | -0.23 | +0.64 |
| pill | +1.97 | +1.21 | -0.59 | +0.02 |
| screw | -3.55 | -0.74 | -0.79 | -3.10 |
| tile | +2.53 | +5.12 | +0.17 | +1.70 |
| toothbrush | +17.16 | +6.99 | +0.35 | +1.99 |
| transistor | -26.62 | -24.15 | -13.22 | -15.23 |
| wood | +8.85 | +6.32 | +0.33 | +2.39 |
| zipper | +7.03 | +5.41 | +0.20 | +0.56 |

Identity check (global image scores, base vs adapted): max |Δ| = 1.202e-01

Train stats: {
  "local_adapt": {
    "kind": "local_adapt",
    "steps": 1000,
    "batch": 6,
    "lr": 0.0002,
    "seed": 1,
    "loss_first100": 0.08418540485203266,
    "loss_last100": 0.06732105728238821,
    "minutes": 17.869694300492604,
    "peak_mib": 4341.39794921875,
    "checkpoint": "/kaggle/working/local_adapt.pt"
  },
  "full_extra": {
    "kind": "full_extra",
    "steps": 1000,
    "batch": 6,
    "lr": 0.0002,
    "seed": 1,
    "loss_first100": 0.052726220525801185,
    "loss_last100": 0.05100517887622118,
    "minutes": 17.85774667263031,
    "peak_mib": 4345.00537109375,
    "checkpoint": "/kaggle/working/full_extra.pt"
  }
}