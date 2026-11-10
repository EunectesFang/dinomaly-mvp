# Kaggle 云端运行归档 — Dinomaly / MVTec-AD

归档时间：**2026-11-05 17:30 (CST)**
来源：Kaggle 账号 `eunectesqaq`，四个 kernel 全部 `COMPLETE`，无失败、无未完成运行。
拉取方式：`kaggle kernels output eunectesqaq/<slug> -p <dir>`（含 kernel log 与 notebook 产物）。

本目录是 **云端 15 类官方协议** 的结果；与 `../experiment/` 的本地 3 类/单类 8 GB 显存探索是两条独立线，不要混用数字。

---

## 1. Kernel 清单

| slug | 运行窗口 (UTC) | 说明 | 产物 |
|---|---|---|---|
| `dinomaly-smoke` | 2026-11-04 06:01 | 环境探针：GPU/显存/数据布局 | log |
| `dinomaly-probe` | 2026-11-04 06:01 | 环境探针：输入路径与依赖 | log |
| `dinomaly-base-10k` | 2026-11-04 06:04 → 13:24（440 min） | **Stage 1**：论文协议 10k 迭代统一基线 | `model_5000.pt`、`model_10000.pt`、`metrics.json`、`train.log` |
| `dinomaly-local-adapt` | 2026-11-05 03:32 → 06:04（151 min） | **Stage 2**：局部裁切适配 + 同算力对照 | `local_adapt.pt`、`full_extra.pt`、`adapt_report.json`、`stage2.log` |

Stage 2 的 `kernel_sources` 依赖 Stage 1，输入数据为 `ipythonx/mvtec-ad` + `eunectesaqq/dinomaly-code`。
Stage 1 相对官方 `dinomaly_mvtec_uni.py` 的改动只有三处：`cuda:0`（官方写死 `cuda:1`）、加 checkpoint 保存、指标写 JSON。

---

## 2. 关键结果

### Stage 1 — 10k 官方基线（15 类均值，%）

| I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|---|---|
| 99.62 | 99.81 | 99.15 | 98.33 | 68.82 | 69.04 | 94.67 |

### Stage 2 — 局部适配 vs 同算力对照

| 配置 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|---|---|---|---|---|
| 基线 global（论文） | **98.33** | 68.82 | 69.04 | **94.67** |
| 基线 + 0.5 融合（不适配） | 97.82 | **71.29** | **70.06** | 94.34 |
| 适配后 global | 95.82 | 61.35 | 62.16 | 91.76 |
| 适配后 local | 96.33 | 70.71 | 69.24 | 93.09 |
| 适配后 0.5 融合 | 96.51 | 69.06 | 67.99 | 93.32 |
| 对照：整幅多训 global | 98.30 | 69.17 | 68.95 | 94.52 |

### 一句话结论

- **同算力对照是零效应**（±0.35）⇒ 所有变化都来自「局部裁切」本身。
- 局部视图变好（P-AP +5.82），但**全局视图被打崩（−7.47）**，**融合净值 −2.24**。
- 损失集中在两类：`transistor −26.62`、`cable −24.16`（P-AP，pp）；其余 10 类全为正。
- 适配后低 FPR 召回塌方（local 视图 R@FPR1% 84.84 → 28.86）⇒ **排序变好但标定坏掉**，按阈值卡的产品线用法不能直接上。
- 可部署最优仍是 **基线 + 0.5 融合、不做适配**。

完整表格见 `reports/report.md`（由 `scripts/make_report.py` 从两份 JSON 生成）。

---

## 3. 目录结构

```
kaggle_runs/
├── README.md                      ← 本文件
├── SHA256SUMS                     ← 全部文件校验和
├── dinomaly-smoke/                ← kernel log
├── dinomaly-probe/
├── dinomaly-base-10k/             ← log + metrics.json + 两个 ckpt (各 234 MB)
├── dinomaly-local-adapt/          ← log + adapt_report.json + 两个 ckpt (各 234 MB)
├── reports/
│   ├── report.md                  ← 自动生成的完整指标表
│   ├── metrics.json               ← Stage 1（副本）
│   ├── adapt_report.json          ← Stage 2（副本，含逐类/逐区域）
│   └── watch_*.log                ← 轮询 kernel 状态的原始记录
└── scripts/
    ├── make_report.py
    └── {smoke,probe,stage1,stage2}/  ← 提交到 Kaggle 的源码 + kernel-metadata.json
```

## 4. 校验与重跑

```bash
sha256sum -c SHA256SUMS          # 校验归档完整性
kaggle kernels output eunectesqaq/dinomaly-local-adapt -p ./retry   # 重新拉取
```

已知哈希（与本地 `~/dinomaly-kaggle/_out_*` 逐字节一致）：

- `model_10000.pt`  `f5689b76aafa5c9d2b36a0df8e87ef7bdc82075a042c76bcf7933e0edd637218`
- `local_adapt.pt`  `45bc75f70ebeb0bb4187ea7cc1494baf01f88f0dd7ea1f6400567290a5713eae`
- `full_extra.pt`   `4a5d39290a3ac4dd604f8773dbedfe5402309e265b3d5cfbe007cf8d5f8d02a2`

## 5. 注意

- 四个 `.pt` 各 234.4 MB，合计约 938 MB；只有 `model_10000.pt` 是正式基线，`model_5000.pt` 是中途快照。
- `adapt_report.json` 里的 `identity.global_score_max_abs_diff` 是 **base vs adapted 的全局分数差**（0.120），不是同一权重的复现误差 —— 别当成等价性检查读。
- `watch_stage2.log` 中 11:32:02 那条 `ERROR` 是改脚本重推前的旧状态，最终运行为 `COMPLETE`。
- MVTec-AD 许可 CC BY-NC-SA 4.0，仅非商业用途。
