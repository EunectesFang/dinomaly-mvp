# dinomaly-mvp

本地 Dinomaly（CVPR 2025）实验工作区。上游代码在 [`guojiajeremy/Dinomaly`](https://github.com/guojiajeremy/Dinomaly)，许可证 Apache-2.0；本仓库里的 `Dinomaly/` 是一份可修改副本。

| 目录 | 内容 |
|---|---|
| `Dinomaly/` | 上游训练与评测代码 |
| `experiment/` | 本地实验脚本与记录（单类 / 小显存） |
| `kaggle_runs/` | Kaggle 上 15 类协议的脚本、日志和指标 |

下面这些留在本地，没有放进 Git：

- `data/`：MVTec-AD 图像和压缩包
- `*.pt` / `*.pth`：训练 checkpoint 和 DINOv2 backbone 权重
- `kaggle_runs/**/deps/`：从 Kaggle 拉下来的第三方包
- `Dinomaly/.git.upstream/`：原来的上游 clone 元数据。要恢复成独立仓库时，把它改回 `Dinomaly/.git`
