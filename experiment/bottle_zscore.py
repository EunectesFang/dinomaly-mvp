"""Compare official, global-z, and positional-z heatmaps for the trained bottle model.

Statistics are estimated only on bottle train/good with the model in eval mode.
The test set is used only for scoring. Formulas are fixed:

    A_g(p) = 1 - cosine_g(p)          # 28x28
    Z_g(p) = (A_g(p) - mu_{g,p}) / max(s_{g,p}, 1e-3)
    Z = (Z_1 + Z_2) / 2

Global mu, s are scalars per group. Positional mu, s are 28x28 per group.
s is the population standard deviation (ddof=0).
Z uses the same 28 -> 392 (align_corners=True) -> 256 (align_corners=False)
-> 5x5 sigma=4 Gaussian path as the official map.
"""

import gc
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

REPO = Path(r"E:\dinomaly-mvp\Dinomaly")
DATA = Path(r"E:\dinomaly-mvp\data\mvtec_anomaly_detection")
WEIGHT = Path(r"E:\dinomaly-mvp\experiment\runs\bottle_sep_b6\dinomaly_bottle.pt")
SEP_PATH = Path(r"E:\dinomaly-mvp\experiment\bottle_sep_b6.py")
RUN = Path(r"E:\dinomaly-mvp\experiment\runs\bottle_zscore")
CLASS_NAME = "bottle"
IMAGE_SIZE = 448
CROP_SIZE = 392
SIDE = 28
S_FLOOR = 1e-3
FPRS = (0.001, 0.01, 0.05)
REFERENCE_PIXEL_AP = 0.8842199527833942
BATCH_CANDIDATES = (6, 4)


def load_sep():
    spec = importlib.util.spec_from_file_location("bottle_sep_b6_reuse", SEP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SEP = load_sep()
sys.path.insert(0, str(REPO))
from dataset import MVTecDataset, get_data_transforms  # noqa: E402


def log(message):
    print(message, flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def group_distance(encoder_feats, decoder_feats):
    maps = []
    for encoder_feat, decoder_feat in zip(encoder_feats, decoder_feats):
        anomaly = (1 - F.cosine_similarity(encoder_feat, decoder_feat)).unsqueeze(1)
        maps.append(anomaly)
    distance = torch.cat(maps, dim=1)
    if distance.shape[1] != 2 or distance.shape[-2:] != (SIDE, SIDE):
        raise RuntimeError(f"expected two 28x28 groups, got {tuple(distance.shape)}")
    return distance


def postprocess(anomaly_map, gaussian):
    return gaussian(SEP.upsample_to_256(anomaly_map, already_392=False))


def f1_max(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-7)
    return float(f1_scores[:-1].max())


def recall_at_fpr(gt, score, fpr):
    normal = score[gt == 0]
    defect = score[gt == 1]
    if normal.size == 0 or defect.size == 0:
        raise RuntimeError("normal or defect pixels are empty")
    threshold = float(np.quantile(normal, 1.0 - fpr, method="linear"))
    realized = float(np.mean(normal > threshold))
    recall = float(np.mean(defect > threshold))
    return {
        "fpr": fpr,
        "threshold": threshold,
        "recall": recall,
        "realized_normal_exceed_rate": realized,
        "rule": "threshold = linear quantile of gt==0 scores at probability 1-fpr; recall = fraction of gt==1 scores strictly above it",
    }


def region_stats(scores, gts):
    structure = np.ones((9, 9), dtype=int)
    sums = {"interior": 0.0, "boundary": 0.0, "far_normal": 0.0}
    counts = {"interior": 0, "boundary": 0, "far_normal": 0}
    empty = 0
    for score, gt in zip(scores, gts):
        defect = gt == 1
        eroded = ndimage.binary_erosion(defect, structure=structure)
        dilated = ndimage.binary_dilation(defect, structure=structure)
        boundary = np.logical_and(dilated, np.logical_not(eroded))
        far = np.logical_and(np.logical_not(dilated), gt == 0)
        if not np.any(eroded):
            empty += 1
        for key, mask in (("interior", eroded), ("boundary", boundary), ("far_normal", far)):
            count = int(mask.sum())
            if count:
                sums[key] += float(score[mask].sum(dtype=np.float64))
                counts[key] += count
    means = {
        key: (None if counts[key] == 0 else sums[key] / counts[key])
        for key in sums
    }
    return {
        "interior_mean": means["interior"],
        "boundary_mean": means["boundary"],
        "far_normal_mean": means["far_normal"],
        "interior_pixels": counts["interior"],
        "boundary_pixels": counts["boundary"],
        "far_normal_pixels": counts["far_normal"],
        "empty_interior_images": int(empty),
        "structure": "9x9 ones, one binary_erosion and one binary_dilation",
        "definition": "interior=eroded defect; boundary=dilated minus eroded; far_normal=outside dilation and gt==0; means are pixel micro-averages",
    }


def pixel_bundle(gt, score):
    gt_flat = gt.ravel().astype(np.uint8)
    pred_flat = score.ravel()
    bundle = {
        "pixel_auroc": float(roc_auc_score(gt_flat, pred_flat)),
        "pixel_ap": float(average_precision_score(gt_flat, pred_flat)),
        "pixel_f1_max": f1_max(gt_flat, pred_flat),
        "pixel_f1_max_threshold_on_test": True,
        "pixel_f1_max_optimistic": True,
        "recall_at_fpr": {str(fpr): recall_at_fpr(gt, score, fpr) for fpr in FPRS},
        "regions": region_stats(score, gt),
    }
    return bundle


def assert_train_good(dataset):
    for path, label in zip(dataset.img_paths, dataset.labels):
        norm = str(path).replace("\\", "/").lower()
        if "/train/good/" not in norm or int(label) != 0:
            raise RuntimeError(f"statistics sample is not train/good: {path} label={label}")


def load_model(device):
    model = SEP.build_model(device)
    probe = model.decoder[0].norm1.weight.detach().clone()
    try:
        state = torch.load(WEIGHT, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(WEIGHT, map_location=device)
    if "bottleneck" not in state or "decoder" not in state:
        raise RuntimeError(f"weight keys are {list(state.keys())}")
    model.bottleneck.load_state_dict(state["bottleneck"], strict=True)
    model.decoder.load_state_dict(state["decoder"], strict=True)
    if torch.allclose(probe, model.decoder[0].norm1.weight.detach()):
        raise RuntimeError("decoder weights did not change after load")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    drops = [module for module in model.modules() if isinstance(module, nn.Dropout)]
    if not drops or any(module.training for module in drops):
        raise RuntimeError("Dropout is missing or still in train mode")
    del state, probe
    return model


def collect_train_distances(model, loader, device):
    chunks = []
    seen = 0
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        paths = batch[3]
        labels = batch[2]
        for path, label in zip(paths, labels):
            norm = str(path).replace("\\", "/").lower()
            if "/train/good/" not in norm or int(label) != 0:
                raise RuntimeError(f"refusing non-good training sample {path}")
        encoder_feats, decoder_feats = model(images)
        distance = group_distance(encoder_feats, decoder_feats).detach().to(dtype=torch.float32).cpu()
        chunks.append(distance)
        seen += distance.shape[0]
    stacked = torch.cat(chunks, dim=0).double().numpy()
    if stacked.shape != (seen, 2, SIDE, SIDE):
        raise RuntimeError(f"unexpected train distance shape {stacked.shape}")
    mu_global = stacked.mean(axis=(0, 2, 3))
    s_global = stacked.std(axis=(0, 2, 3), ddof=0)
    mu_pos = stacked.mean(axis=0)
    s_pos = stacked.std(axis=0, ddof=0)
    flat = stacked[:, 0].reshape(-1)
    if abs(float(flat.mean()) - float(mu_global[0])) > 1e-8:
        raise RuntimeError("global mean reduction mismatch")
    if abs(float(flat.std(ddof=0)) - float(s_global[0])) > 1e-8:
        raise RuntimeError("global std reduction mismatch")
    return stacked.shape[0], mu_global, s_global, mu_pos, s_pos


def standardize(distance, mu, scale):
    mu_t = torch.as_tensor(mu, device=distance.device, dtype=distance.dtype)
    scale_t = torch.as_tensor(scale, device=distance.device, dtype=distance.dtype)
    if mu_t.ndim == 1:
        mu_t = mu_t.view(1, -1, 1, 1)
        scale_t = scale_t.view(1, -1, 1, 1)
    elif mu_t.ndim == 3:
        mu_t = mu_t.unsqueeze(0)
        scale_t = scale_t.unsqueeze(0)
    else:
        raise RuntimeError(f"unexpected statistic rank {mu_t.ndim}")
    return (distance - mu_t) / scale_t


@torch.no_grad()
def collect_test(model, loader, device, gaussian, stats):
    mu_global, s_global, mu_pos, s_pos = stats
    scale_global = np.maximum(s_global, S_FLOOR)
    scale_pos = np.maximum(s_pos, S_FLOOR)
    official_chunks, global_chunks, pos_chunks = [], [], []
    mask_chunks, label_chunks, score_chunks = [], [], []
    checked = False
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        target = batch[1].to(device, non_blocking=True)
        encoder_feats, decoder_feats = model(images)
        official_392, _ = SEP.cal_anomaly_maps(encoder_feats, decoder_feats, images.shape[-1])
        official_256 = gaussian(SEP.upsample_to_256(official_392, already_392=True))
        distance = group_distance(encoder_feats, decoder_feats)
        if not checked:
            averaged = distance.mean(dim=1, keepdim=True)
            recon = postprocess(averaged, gaussian)
            delta = float((recon - official_256).abs().max())
            log(f"official vs average-then-upsample max_abs_diff={delta:.3e}")
            if delta > 1e-3:
                raise RuntimeError(f"official postprocess diverged from 28x28 average: {delta}")
            checked = True
        z_global = standardize(distance, mu_global, scale_global).mean(dim=1, keepdim=True)
        z_pos = standardize(distance, mu_pos, scale_pos).mean(dim=1, keepdim=True)
        official_chunks.append(official_256.cpu())
        global_chunks.append(postprocess(z_global, gaussian).cpu())
        pos_chunks.append(postprocess(z_pos, gaussian).cpu())
        score_chunks.append(SEP.image_score(official_256).cpu())
        mask_256 = F.interpolate(target, size=256, mode="nearest").bool().float().cpu()
        mask_chunks.append(mask_256[:, :1])
        label_chunks.append(batch[2].reshape(-1).cpu())
    maps = {
        "official": torch.cat(official_chunks, dim=0)[:, 0].numpy(),
        "z_global": torch.cat(global_chunks, dim=0)[:, 0].numpy(),
        "z_positional": torch.cat(pos_chunks, dim=0)[:, 0].numpy(),
    }
    gt = torch.cat(mask_chunks, dim=0)[:, 0].numpy()
    labels = torch.cat(label_chunks, dim=0).numpy().astype(np.int64)
    image_scores = torch.cat(score_chunks, dim=0).numpy()
    return maps, gt, labels, image_scores


def stat_summary(mu_global, s_global, mu_pos, s_pos, n_train):
    return {
        "train_good_images": int(n_train),
        "positions": SIDE * SIDE,
        "ddof": 0,
        "s_floor": S_FLOOR,
        "source": "bottle/train/good only; model.eval(); Dropout off",
        "global": {
            "mu": [float(v) for v in mu_global],
            "s": [float(v) for v in s_global],
            "scale": [float(v) for v in np.maximum(s_global, S_FLOOR)],
        },
        "positional": {
            "mu_min": [float(v) for v in mu_pos.reshape(2, -1).min(axis=1)],
            "mu_max": [float(v) for v in mu_pos.reshape(2, -1).max(axis=1)],
            "s_min": [float(v) for v in s_pos.reshape(2, -1).min(axis=1)],
            "s_max": [float(v) for v in s_pos.reshape(2, -1).max(axis=1)],
            "positions_below_floor": [int(np.sum(s_pos[g] < S_FLOOR)) for g in range(2)],
        },
    }


def execute(batch_size):
    started = time.time()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log(f"device={torch.cuda.get_device_name(device)} batch={batch_size}")
    model = load_model(device)
    log("loaded bottleneck and decoder; encoder is dinov2reg_vit_base_14; eval Dropout off")

    data_transform, gt_transform = get_data_transforms(IMAGE_SIZE, CROP_SIZE)
    train_set = MVTecDataset(
        root=str(DATA / CLASS_NAME),
        transform=data_transform,
        gt_transform=gt_transform,
        phase="train",
    )
    assert_train_good(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    n_train, mu_global, s_global, mu_pos, s_pos = collect_train_distances(model, train_loader, device)
    log(
        f"train_good={n_train} mu_global={mu_global.tolist()} s_global={s_global.tolist()} "
        f"s_pos_min={s_pos.min(axis=(1, 2)).tolist()} s_pos_max={s_pos.max(axis=(1, 2)).tolist()}"
    )
    del train_loader, train_set

    test_set = MVTecDataset(
        root=str(DATA / CLASS_NAME),
        transform=data_transform,
        gt_transform=gt_transform,
        phase="test",
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    gaussian = SEP.get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    maps, gt, labels, image_scores = collect_test(
        model, test_loader, device, gaussian, (mu_global, s_global, mu_pos, s_pos)
    )
    if gt.shape[0] != len(test_set) or set(np.unique(gt).tolist()) - {0.0, 1.0}:
        raise RuntimeError(f"unexpected test masks: shape={gt.shape} unique={np.unique(gt)[:8]}")
    log(f"test={len(test_set)} anomaly_images={int(labels.sum())} good_images={int((labels == 0).sum())}")

    rows = {}
    for name in ("official", "z_global", "z_positional"):
        log(f"scoring {name}")
        rows[name] = pixel_bundle(gt, maps[name])
        row = rows[name]
        recalls = ", ".join(
            f"{fpr}:{row['recall_at_fpr'][str(fpr)]['recall']:.4f}" for fpr in FPRS
        )
        regions = row["regions"]
        log(
            f"{name}: P-AP={row['pixel_ap']:.4f} P-AUROC={row['pixel_auroc']:.4f} "
            f"P-F1max={row['pixel_f1_max']:.4f} recall_at_fpr[{recalls}] "
            f"interior={regions['interior_mean']:.6f} boundary={regions['boundary_mean']:.6f} "
            f"far={regions['far_normal_mean']:.6f} empty_interior={regions['empty_interior_images']}"
        )

    image_auroc = float(roc_auc_score(labels, image_scores))
    official_ap = rows["official"]["pixel_ap"]
    in_range = bool(0.86 <= official_ap <= 0.91)
    empty_counts = [rows[name]["regions"]["empty_interior_images"] for name in rows]
    if len(set(empty_counts)) != 1:
        raise RuntimeError(f"empty interior counts differ across methods: {empty_counts}")
    peak_mib = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
    report = {
        "class_name": CLASS_NAME,
        "weight": str(WEIGHT),
        "device": "cuda:0",
        "batch_size": batch_size,
        "counts": {
            "train_good": int(n_train),
            "test": int(len(test_set)),
            "test_anomaly": int(labels.sum()),
            "test_good": int((labels == 0).sum()),
        },
        "peak_mib": peak_mib,
        "seconds": round(time.time() - started, 1),
        "statistics": stat_summary(mu_global, s_global, mu_pos, s_pos, n_train),
        "image_auroc_official_top1pct_mean": image_auroc,
        "image_score_definition": "mean of the highest 1 percent pixels of the official Gaussian heatmap; z-scores are not used",
        "reference_official_pixel_ap": REFERENCE_PIXEL_AP,
        "official_pixel_ap": official_ap,
        "official_pixel_ap_minus_reference": official_ap - REFERENCE_PIXEL_AP,
        "official_pixel_ap_in_0.86_0.91": in_range,
        "methods": rows,
        "notes": {
            "pixel_f1_max": "阈值在测试集上选取，偏乐观",
            "recall_at_fpr": "三个固定误报率都报告。阈值是测试集 gt==0 像素分数的 1-fpr 线性分位数，召回是缺陷像素中分数严格超过该阈值的比例。",
            "statistics": "均值和总体标准差只用 bottle train/good。s 下限 1e-3。没有用测试集估计均值或方差。",
            "postprocess": "28x28 上平均后，双线性到 392（align_corners=True），再到 256（align_corners=False），再 5x5 sigma=4 高斯。official 走 bottle_sep_b6 的原评估路径。",
        },
    }
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"wrote {RUN / 'report.json'} peak_mib={peak_mib} official_ap_in_range={in_range} image_auroc={image_auroc:.4f}")
    if not in_range:
        raise RuntimeError(
            f"official pixel AP {official_ap:.4f} is outside [0.86, 0.91]; "
            "baseline is not usable, z-score rows in the report must not be treated as the result"
        )
    return report


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("cuda is not available")
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "run.log").write_text("", encoding="utf-8")
    error_path = RUN / "error.txt"
    if error_path.exists():
        error_path.unlink()
    log("bottle z-score comparison; no decoder retraining")
    for index, batch_size in enumerate(BATCH_CANDIDATES):
        try:
            execute(batch_size)
            return
        except RuntimeError as exc:
            oom = "out of memory" in str(exc).lower()
            if not oom or index == len(BATCH_CANDIDATES) - 1:
                raise
            log(f"batch {batch_size} OOM, retry at {BATCH_CANDIDATES[index + 1]}")
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        text = traceback.format_exc()
        RUN.mkdir(parents=True, exist_ok=True)
        (RUN / "error.txt").write_text(text, encoding="utf-8")
        print(text, flush=True)
        try:
            log(text)
        except Exception:
            pass
        raise
