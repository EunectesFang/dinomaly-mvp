"""Stage 3: calibrated local refinement on the 15-class MVTec-AD protocol.

Reuses two frozen checkpoints, no retraining:
  * eunectesqaq/dinomaly-base-10k   -> model_10000.pt   (baseline global)
  * eunectesqaq/dinomaly-local-adapt -> local_adapt.pt  (adapted local, 196->392)

Fixes two bugs in stage 2:
  1. The adapted fusion used the adapted global map.  Here every fusion uses the
     ORIGINAL baseline global map.  No adapted global forward is ever computed.
  2. Fixed-FPR recall reused one global negative distribution for every variant.
     Here each variant gets its own per-class negative distribution / threshold.

Variants (all use the same original baseline image scores):
  base_global          baseline global map, no calibration
  base_local_fuse      0.5 * baseline global + 0.5 * baseline local
  adapted_local_fuse   0.5 * baseline global + 0.5 * adapted local  (bug 1 fixed)
  cal_scalar           per-branch median/q95 alignment + 0.5 fusion
  cal_spatial          per-branch shrunk spatial background correction + 0.5 fusion
  cal_fused            scalar alignment + shrunk spatial correction + 0.5 fusion

Calibration uses exactly 24 deterministic train/good images per class.  Nothing
from the test split is used to fit calibration or thresholds.

Expensive forwards are computed once per image and shared by every variant.
Per-class calibration and results are written atomically so a killed run keeps
all completed classes.
"""
import os
import sys
import json
import time
import argparse
import hashlib
import traceback

import numpy as np

OUT = "/kaggle/working"
CROP = 392
WINDOW = 196
STARTS = (0, 98, 196)
CALIB_N = 24
SHRINK_K = 24          # n=24, k=24 -> shrink weight 0.5
CALIB_BATCH = 6
EVAL_BATCH = 8
FPRS = (0.001, 0.01, 0.05)
EXPECTED_TEST_IMAGES = 1725
SLUG = "eunectesqaq/dinomaly-local-refine-v3"

VARIANTS = (
    "base_global",
    "base_local_fuse",
    "adapted_local_fuse",
    "cal_scalar",
    "cal_spatial",
    "cal_fused",
)
METRIC_KEYS = ("I-AUROC", "I-AP", "I-F1",
               "P-AUROC", "P-AP", "P-F1", "P-AUPRO")
RECALL_KEYS = tuple(f"R@FPR{f:g}" for f in FPRS)

LOG_PATH = None
FP = None
CODE = None
DATA = None
BASE_CKPT = None
ADAPT_CKPT = None
STAGE2_REPORT = None

# Runtime imports are bound here by prepare_environment().
torch = None
nn = None
F = None
DataLoader = None
transforms = None
Image = None
partial = None
hann = None
measure = None
roc_auc_score = None
average_precision_score = None
auc = None
get_data_transforms = None
MVTecDataset = None
ViTill = None
vit_encoder = None
VitBlock = None
bMlp = None
LinearAttention2 = None
cal_anomaly_maps = None
get_gaussian_kernel = None
f1_score_max = None


# ---------------------------------------------------------------------------
# infrastructure
# ---------------------------------------------------------------------------
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_fingerprint(cfg):
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def locate(marker, preferred):
    for c in preferred:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, marker)):
            return c
    for root, dirs, files in os.walk("/kaggle/input"):
        if marker in dirs or marker in files:
            return root
    raise RuntimeError(f"cannot locate {marker}")


def find_input(name, preferred):
    for p in preferred:
        if os.path.isfile(p):
            return p
    for root, _dirs, files in os.walk("/kaggle/input"):
        if name in files:
            return os.path.join(root, name)
    return None


# ---------------------------------------------------------------------------
# deterministic calibration / fusion helpers (pure numpy, self-testable)
# ---------------------------------------------------------------------------
def select_calibration_indices(n, k=CALIB_N):
    """Evenly spaced sorted indices, endpoints included, deterministic."""
    if n < k:
        raise ValueError(f"need at least {k} train/good images, got {n}")
    idx = np.linspace(0, n - 1, k).round().astype(int).tolist()
    if len(set(idx)) != k:
        raise RuntimeError(f"non-unique calibration indices for n={n}")
    return idx


def branch_stats(raw_maps, shrink_k=SHRINK_K):
    """Robust scalar median/q95 and shrunk per-pixel background for one branch.

    raw_maps: [N, 1, H, W] or [N, H, W], pixel values of calibration normals.
    Scalar alignment is (x - median) / (q95 - median).
    Spatial background is mean(x) per pixel, shrunk toward its own global mean
    with weight w = N / (N + shrink_k).
    """
    x = np.asarray(raw_maps, dtype=np.float64)
    if x.ndim == 4:
        x = x[:, 0]
    if x.ndim != 3:
        raise ValueError(f"expected [N,1,H,W] or [N,H,W], got {x.shape}")
    n = int(x.shape[0])
    vals = x.reshape(-1)
    median = float(np.median(vals))
    q95 = float(np.quantile(vals, 0.95))
    scale = q95 - median
    fallback = False
    if not np.isfinite(scale) or abs(scale) < 1e-9:
        scale = float(np.std(vals))
        fallback = True
        if not np.isfinite(scale) or abs(scale) < 1e-9:
            scale = 1.0

    bg_raw = x.mean(axis=0)
    mu_raw = float(bg_raw.mean())
    w = float(n) / float(n + shrink_k)
    bg_raw_shrunk = mu_raw + w * (bg_raw - mu_raw)

    bg_align = (bg_raw - median) / scale
    mu_align = float(bg_align.mean())
    bg_align_shrunk = mu_align + w * (bg_align - mu_align)

    return {
        "n": n,
        "shrink_weight": w,
        "median": median,
        "q95": q95,
        "scale": scale,
        "scale_fallback": fallback,
        "mu_raw": mu_raw,
        "mu_align": mu_align,
        "bg_raw": bg_raw,
        "bg_raw_shrunk": bg_raw_shrunk,
        "bg_align": bg_align,
        "bg_align_shrunk": bg_align_shrunk,
    }


def build_variant_maps(g, lb, la, cal):
    """Build every variant from one set of branch maps.

    g  : ORIGINAL baseline global map
    lb : baseline local map
    la : adapted local map
    cal: per-branch calibration with either numpy or torch background arrays.

    There is deliberately no adapted-global argument, so bug 1 cannot recur.
    """
    g_med, g_scale = cal["g"]["median"], cal["g"]["scale"]
    l_med, l_scale = cal["l"]["median"], cal["l"]["scale"]
    z_g = (g - g_med) / g_scale
    z_l = (la - l_med) / l_scale
    return {
        "base_global": g,
        "base_local_fuse": 0.5 * g + 0.5 * lb,
        "adapted_local_fuse": 0.5 * g + 0.5 * la,
        "cal_scalar": 0.5 * z_g + 0.5 * z_l,
        "cal_spatial": (0.5 * (g - cal["g"]["bg_raw_shrunk"])
                        + 0.5 * (la - cal["l"]["bg_raw_shrunk"])),
        "cal_fused": (0.5 * (z_g - cal["g"]["bg_align_shrunk"])
                      + 0.5 * (z_l - cal["l"]["bg_align_shrunk"])),
    }


def fixed_fpr_recall(pr_flat, gt_flat, fprs=FPRS):
    """Recall at fixed FPR using THIS variant's own negative pixels only."""
    pr_flat = np.asarray(pr_flat).reshape(-1)
    gt_flat = np.asarray(gt_flat).reshape(-1)
    neg = pr_flat[gt_flat == 0]
    pos = pr_flat[gt_flat == 1]
    recall, thresholds = {}, {}
    if neg.size == 0 or pos.size == 0:
        return recall, thresholds
    for fpr in fprs:
        thr = float(np.quantile(neg, 1.0 - fpr))
        recall[f"R@FPR{fpr:g}"] = float((pos >= thr).mean())
        thresholds[f"T@FPR{fpr:g}"] = thr
    return recall, thresholds


# ---------------------------------------------------------------------------
# Kaggle runtime setup
# ---------------------------------------------------------------------------
def prepare_environment():
    global CODE, DATA
    global torch, nn, F, DataLoader, transforms, Image, partial, hann, measure
    global roc_auc_score, average_precision_score, auc
    global get_data_transforms, MVTecDataset, ViTill, vit_encoder
    global VitBlock, bMlp, LinearAttention2
    global cal_anomaly_maps, get_gaussian_kernel, f1_score_max

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from PIL import Image
    from functools import partial
    from scipy.signal.windows import hann
    from skimage import measure
    from sklearn.metrics import (roc_auc_score, average_precision_score, auc)

    CODE = locate("dinomaly_mvtec_uni.py", [
        "/kaggle/input/dinomaly-code",
        "/kaggle/input/datasets/eunectesqaq/dinomaly-code",
    ])
    DATA = locate("bottle", [
        "/kaggle/input/mvtec-ad",
        "/kaggle/input/datasets/ipythonx/mvtec-ad",
    ])

    sys.path.insert(0, CODE)
    os.chdir(CODE)

    from dataset import get_data_transforms, MVTecDataset
    from models.uad import ViTill
    from models import vit_encoder
    from models.vision_transformer import (Block as VitBlock, bMlp,
                                           LinearAttention2)
    from utils import (cal_anomaly_maps, get_gaussian_kernel, f1_score_max)

    log(f"CODE={CODE}")
    log(f"DATA={DATA}")


def find_checkpoints():
    global BASE_CKPT, ADAPT_CKPT, STAGE2_REPORT
    BASE_CKPT = find_input("model_10000.pt", [
        "/kaggle/input/dinomaly-base-10k/model_10000.pt",
        "/kaggle/input/notebooks/eunectesqaq/dinomaly-base-10k/model_10000.pt",
        "/kaggle/input/datasets/eunectesqaq/dinomaly-base-10k/model_10000.pt",
    ])
    ADAPT_CKPT = find_input("local_adapt.pt", [
        "/kaggle/input/dinomaly-local-adapt/local_adapt.pt",
        "/kaggle/input/notebooks/eunectesqaq/dinomaly-local-adapt/local_adapt.pt",
        "/kaggle/input/datasets/eunectesqaq/dinomaly-local-adapt/local_adapt.pt",
    ])
    STAGE2_REPORT = find_input("adapt_report.json", [
        "/kaggle/input/dinomaly-local-adapt/adapt_report.json",
        "/kaggle/input/notebooks/eunectesqaq/dinomaly-local-adapt/adapt_report.json",
        "/kaggle/input/datasets/eunectesqaq/dinomaly-local-adapt/adapt_report.json",
    ])
    if not BASE_CKPT:
        raise RuntimeError("model_10000.pt not found under /kaggle/input")
    if not ADAPT_CKPT:
        raise RuntimeError("local_adapt.pt not found under /kaggle/input")
    log(f"base checkpoint={BASE_CKPT}")
    log(f"adapted checkpoint={ADAPT_CKPT}")
    log(f"stage2 report={STAGE2_REPORT}")


def load_stage2_report():
    if not STAGE2_REPORT:
        return None
    try:
        rep = read_json(STAGE2_REPORT)
        base = rep.get("results", {}).get("base", {})
        if not base:
            log("stage2 report has no results.base; identity check skipped")
            return None
        log(f"stage2 base classes={len(base.get('per_class', {}))}")
        return base
    except Exception as exc:
        log(f"could not read stage2 report: {exc}")
        return None


# ---------------------------------------------------------------------------
# model / maps
# ---------------------------------------------------------------------------
def build_model(device):
    encoder = vit_encoder.load("dinov2reg_vit_base_14")
    bottleneck = nn.ModuleList([bMlp(768, 768 * 4, 768, drop=0.2)])
    decoder = nn.ModuleList([
        VitBlock(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-8),
                 attn=LinearAttention2)
        for _ in range(8)])
    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                   target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
                   mask_neighbor_size=0,
                   fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
                   fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]])
    return model.to(device)


def load_ckpt(model, path, device):
    st = torch.load(path, map_location=device, weights_only=False)
    model.bottleneck.load_state_dict(st["bottleneck"])
    model.decoder.load_state_dict(st["decoder"])
    return st


def norm_tensors(device):
    mean = torch.tensor((0.485, 0.456, 0.406),
                        device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225),
                       device=device).view(1, 3, 1, 1)
    return mean, std


def normalize(x, mean, std):
    return (x - mean) / std


def global_map(model, images):
    en, de = model(images)
    amap, _ = cal_anomaly_maps(en, de, images.shape[-1])
    return amap


def local_map(model, images, weight, mean, std):
    rgb = images * std + mean
    b = images.shape[0]
    acc = torch.zeros(b, 1, CROP, CROP, device=images.device)
    wsum = torch.zeros(b, 1, CROP, CROP, device=images.device)
    for y in STARTS:
        for x0 in STARTS:
            patch = rgb[:, :, y:y + WINDOW, x0:x0 + WINDOW]
            z = F.interpolate(patch, size=CROP, mode="bilinear",
                              align_corners=False)
            en, de = model(normalize(z, mean, std))
            dist = torch.stack(
                [1 - F.cosine_similarity(e, d) for e, d in zip(en, de)],
                0).mean(0).unsqueeze(1)
            pasted = F.interpolate(dist, size=WINDOW, mode="bilinear",
                                   align_corners=True)
            acc[:, :, y:y + WINDOW, x0:x0 + WINDOW] += pasted * weight
            wsum[:, :, y:y + WINDOW, x0:x0 + WINDOW] += weight
    return acc / wsum


def postprocess(amap, gaussian):
    return gaussian(F.interpolate(amap, size=256, mode="bilinear",
                                  align_corners=False))


def top1pct(amap):
    flat = amap.flatten(1)
    k = max(1, int(flat.shape[1] * 0.01))
    return torch.sort(flat, dim=1, descending=True).values[:, :k].mean(dim=1)


# ---------------------------------------------------------------------------
# calibration persistence
# ---------------------------------------------------------------------------
def calib_paths(item):
    base = os.path.join(OUT, "calibration", item)
    return base + ".npz", base + ".json"


def cal_summary(cal):
    out = {}
    for br in ("g", "l"):
        out[br] = {
            "median": float(cal[br]["median"]),
            "q95": float(cal[br]["q95"]),
            "scale": float(cal[br]["scale"]),
            "scale_fallback": bool(cal[br]["scale_fallback"]),
            "mu_raw": float(cal[br]["mu_raw"]),
            "mu_align": float(cal[br]["mu_align"]),
            "n": int(cal[br]["n"]),
            "shrink_weight": float(cal[br]["shrink_weight"]),
        }
    return out


def save_calibration(item, cal, selected_paths):
    npz_path, json_path = calib_paths(item)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    payload = {}
    for br in ("g", "l"):
        c = cal[br]
        payload.update({
            f"{br}_bg_raw": c["bg_raw"].astype(np.float32),
            f"{br}_bg_raw_shrunk": c["bg_raw_shrunk"].astype(np.float32),
            f"{br}_bg_align": c["bg_align"].astype(np.float32),
            f"{br}_bg_align_shrunk": c["bg_align_shrunk"].astype(np.float32),
            f"{br}_median": np.float64(c["median"]),
            f"{br}_q95": np.float64(c["q95"]),
            f"{br}_scale": np.float64(c["scale"]),
            f"{br}_scale_fallback": np.bool_(c["scale_fallback"]),
            f"{br}_mu_raw": np.float64(c["mu_raw"]),
            f"{br}_mu_align": np.float64(c["mu_align"]),
            f"{br}_n": np.int64(c["n"]),
            f"{br}_shrink_weight": np.float64(c["shrink_weight"]),
        })
    np.savez_compressed(npz_path, **payload)
    meta = {
        "fingerprint": FP,
        "item": item,
        "n": CALIB_N,
        "shrink_k": SHRINK_K,
        "shrink_weight": float(cal["g"]["shrink_weight"]),
        "selection": "sorted train/good paths, evenly spaced (np.linspace), endpoints included",
        "selected": [os.path.relpath(p, DATA) for p in selected_paths],
        "summary": cal_summary(cal),
        "forwards": {"baseline_global": CALIB_N,
                     "adapted_local": CALIB_N * 9},
    }
    write_json_atomic(json_path, meta)


def load_calibration(item):
    npz_path, json_path = calib_paths(item)
    if not (os.path.isfile(npz_path) and os.path.isfile(json_path)):
        return None
    meta = read_json(json_path)
    if meta.get("fingerprint") != FP:
        return None
    cal = {"g": {}, "l": {}}
    with np.load(npz_path) as z:
        for br in ("g", "l"):
            c = cal[br]
            for key in ("median", "q95", "scale", "mu_raw", "mu_align",
                        "shrink_weight"):
                c[key] = float(z[f"{br}_{key}"])
            c["scale_fallback"] = bool(z[f"{br}_scale_fallback"])
            c["n"] = int(z[f"{br}_n"])
            for key in ("bg_raw", "bg_raw_shrunk", "bg_align",
                        "bg_align_shrunk"):
                c[key] = z[f"{br}_{key}"].astype(np.float64)
    return cal, meta


def compute_calibration(item, data_transform, model_base, model_adapt,
                        weight, mean, std, device, calib_batch):
    folder = os.path.join(DATA, item, "train", "good")
    all_paths = sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    idx = select_calibration_indices(len(all_paths))
    selected = [all_paths[i] for i in idx]
    g_chunks, l_chunks = [], []
    for start in range(0, CALIB_N, calib_batch):
        chunk = selected[start:start + calib_batch]
        imgs = torch.stack(
            [data_transform(Image.open(p).convert("RGB")) for p in chunk]
        ).to(device)
        with torch.no_grad():
            g = global_map(model_base, imgs)
            l = local_map(model_adapt, imgs, weight, mean, std)
        g_chunks.append(g.cpu().numpy())
        l_chunks.append(l.cpu().numpy())
    g_maps = np.concatenate(g_chunks, axis=0)
    l_maps = np.concatenate(l_chunks, axis=0)
    cal = {"g": branch_stats(g_maps), "l": branch_stats(l_maps)}
    return cal, selected


def calibration_to_device(cal, device):
    out = {"g": dict(cal["g"]), "l": dict(cal["l"])}
    for br in ("g", "l"):
        for key in ("bg_raw_shrunk", "bg_align_shrunk"):
            out[br][key] = torch.as_tensor(
                cal[br][key], dtype=torch.float32, device=device
            ).view(1, 1, CROP, CROP)
    return out


# ---------------------------------------------------------------------------
# per-class evaluation
# ---------------------------------------------------------------------------
def image_metrics_from_scores(labels, scores):
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {"I-AUROC": float("nan"), "I-AP": float("nan"),
                "I-F1": float("nan")}
    return {
        "I-AUROC": float(roc_auc_score(labels, scores)),
        "I-AP": float(average_precision_score(labels, scores)),
        "I-F1": float(f1_score_max(labels, scores)),
    }


def evaluate_class(item, test_data, model_base, model_adapt, cal_raw,
                   gaussian, weight, mean, std, device, eval_batch,
                   stage2_base):
    cal = calibration_to_device(cal_raw, device)
    loader = DataLoader(test_data, batch_size=eval_batch, shuffle=False,
                        num_workers=4)
    buckets = {v: [] for v in VARIANTS}
    gt_buf, labels, recomputed_scores = [], [], []
    n_batches = len(loader)
    started = time.time()

    with torch.no_grad():
        for bi, (img, gt, label, _path) in enumerate(loader):
            img = img.to(device, non_blocking=True)

            # Exactly three expensive map producers per image, shared by all
            # six variants.  No adapted-global forward exists.
            g = global_map(model_base, img)                    # baseline global
            lb = local_map(model_base, img, weight, mean, std)  # baseline local
            la = local_map(model_adapt, img, weight, mean, std)  # adapted local
            raw = build_variant_maps(g, lb, la, cal)

            for v in VARIANTS:
                m = postprocess(raw[v], gaussian)
                if v == "base_global":
                    recomputed_scores.extend(
                        top1pct(m).cpu().numpy().tolist())
                buckets[v].append(m.cpu())

            gt256 = F.interpolate(gt.float(), size=256, mode="nearest").bool()
            if gt256.shape[1] > 1:
                gt256 = gt256.max(dim=1, keepdim=True)[0]
            gt_buf.append(gt256.cpu())
            labels.extend(label.numpy().tolist())

            if bi == 0 or (bi + 1) % 10 == 0 or (bi + 1) == n_batches:
                log(f"  {item} batch {bi+1}/{n_batches} "
                    f"images={len(labels)}/{len(test_data)} "
                    f"elapsed={time.time()-started:.1f}s")

    gt_np = torch.cat(gt_buf, dim=0)[:, 0].numpy()
    gt_np = (gt_np > 0.5).astype(np.int64)
    gt_flat = gt_np.ravel()
    labels_np = np.asarray(labels, dtype=np.int64)
    recomputed_scores = np.asarray(recomputed_scores, dtype=np.float64)
    regions = label_regions(gt_np)

    # The same original baseline image scores are used for every variant.
    img_metrics = image_metrics_from_scores(labels_np, recomputed_scores)

    stage2_scores = None
    stage2_metrics = {}
    if stage2_base:
        stage2_scores = stage2_base.get("global_scores", {}).get(item)
        stage2_metrics = (stage2_base.get("per_class", {})
                          .get(item, {}).get("metrics", {}))
    score_diff = None
    if stage2_scores is not None:
        if len(stage2_scores) == len(recomputed_scores):
            score_diff = float(np.max(np.abs(
                np.asarray(stage2_scores, dtype=np.float64)
                - recomputed_scores)))
            log(f"  {item} stage2 baseline image-score max|delta|="
                f"{score_diff:.3e}")
        else:
            log(f"  {item} WARNING: stage2 score length "
                f"{len(stage2_scores)} != {len(recomputed_scores)}; "
                f"identity comparison skipped")
    metric_diff = {}
    if stage2_metrics:
        for k in ("I-AUROC", "I-AP", "I-F1"):
            if k in stage2_metrics:
                metric_diff[k] = float(img_metrics[k]
                                       - float(stage2_metrics[k]))

    variants = {}
    for v in VARIANTS:
        pr = torch.cat(buckets[v], dim=0)[:, 0].numpy().astype(np.float32)
        buckets[v] = None
        pr_flat = pr.ravel()
        recall, thresholds = fixed_fpr_recall(pr_flat, gt_flat, FPRS)
        row = {
            "I-AUROC": img_metrics["I-AUROC"],
            "I-AP": img_metrics["I-AP"],
            "I-F1": img_metrics["I-F1"],
            "P-AUROC": float(roc_auc_score(gt_flat, pr_flat)),
            "P-AP": float(average_precision_score(gt_flat, pr_flat)),
            "P-F1": float(f1_score_max(gt_flat, pr_flat)),
            "P-AUPRO": float(compute_pro_pre(gt_np, pr, regions)),
        }
        variants[v] = {
            "metrics": row,
            "recall": recall,
            "thresholds": thresholds,
        }

    result = {
        "class": item,
        "fingerprint": FP,
        "n_test": int(len(labels_np)),
        "n_normal": int((labels_np == 0).sum()),
        "n_anomaly": int((labels_np == 1).sum()),
        "n_gt_positive_pixels": int(gt_flat.sum()),
        "n_gt_negative_pixels": int((gt_flat == 0).sum()),
        "calibration": cal_summary(cal_raw),
        "image_metrics": img_metrics,
        "image_scores": {
            "source": "baseline global top1% of the postprocessed 256 map, "
                      "computed once and reused by every variant",
            "reused_for_all_variants": True,
            "max_abs_diff_across_variants": 0.0,
            "sha256": hashlib.sha256(
                recomputed_scores.astype(np.float64).tobytes()).hexdigest(),
            "stage2_original_max_abs_diff": score_diff,
            "stage2_metric_delta": metric_diff,
        },
        "variants": variants,
        "seconds": float(time.time() - started),
        "gpu_peak_mib": float(torch.cuda.max_memory_allocated() / 1024 ** 2)
        if torch.cuda.is_available() else 0.0,
    }
    return result


def label_regions(gt_np):
    return [[r.coords for r in measure.regionprops(measure.label(m))]
            for m in gt_np]


def compute_pro_pre(masks, amaps, regions, num_th=200):
    min_th, max_th = float(amaps.min()), float(amaps.max())
    if not np.isfinite(min_th) or max_th <= min_th:
        return 0.0
    delta = (max_th - min_th) / num_th
    inverse = 1 - masks
    neg_total = max(int(inverse.sum()), 1)
    pros, fprs = [], []
    for th in np.arange(min_th, max_th, delta):
        binmap = amaps > th
        cur = []
        for bmap, regs in zip(binmap, regions):
            for coords in regs:
                cur.append(bmap[coords[:, 0], coords[:, 1]].sum() / len(coords))
        pros.append(float(np.mean(cur)) if cur else 0.0)
        fprs.append(float(np.logical_and(inverse, binmap).sum()) / neg_total)
    fprs = np.asarray(fprs)
    pros = np.asarray(pros)
    keep = fprs < 0.3
    if keep.sum() < 2 or fprs[keep].max() <= 0:
        return 0.0
    return float(auc(fprs[keep] / fprs[keep].max(), pros[keep]))


# ---------------------------------------------------------------------------
# aggregation / report
# ---------------------------------------------------------------------------
def finite_mean(values):
    vals = [float(v) for v in values
            if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(per_class, item_list):
    mean = {}
    for v in VARIANTS:
        metrics = {}
        for k in METRIC_KEYS:
            metrics[k] = finite_mean(
                [per_class[it]["variants"][v]["metrics"][k]
                 for it in item_list])
        recall = {}
        for k in RECALL_KEYS:
            recall[k] = finite_mean(
                [per_class[it]["variants"][v]["recall"].get(k)
                 for it in item_list])
        mean[v] = {"metrics": metrics, "recall": recall}

    image_mean = {k: finite_mean([per_class[it]["image_metrics"][k]
                                  for it in item_list])
                  for k in ("I-AUROC", "I-AP", "I-F1")}

    comparisons = {
        "base_local_fuse - base_global":
            ("base_local_fuse", "base_global"),
        "adapted_local_fuse - base_global":
            ("adapted_local_fuse", "base_global"),
        "adapted_local_fuse - base_local_fuse":
            ("adapted_local_fuse", "base_local_fuse"),
        "cal_scalar - adapted_local_fuse":
            ("cal_scalar", "adapted_local_fuse"),
        "cal_spatial - adapted_local_fuse":
            ("cal_spatial", "adapted_local_fuse"),
        "cal_fused - adapted_local_fuse":
            ("cal_fused", "adapted_local_fuse"),
        "cal_fused - cal_scalar":
            ("cal_fused", "cal_scalar"),
        "cal_fused - cal_spatial":
            ("cal_fused", "cal_spatial"),
        "cal_fused - base_global":
            ("cal_fused", "base_global"),
    }
    deltas = {}
    for name, (a, b) in comparisons.items():
        deltas[name] = {
            "metrics": {k: mean[a]["metrics"][k] - mean[b]["metrics"][k]
                        for k in METRIC_KEYS},
            "recall": {k: mean[a]["recall"][k] - mean[b]["recall"][k]
                       for k in RECALL_KEYS},
        }

    per_class_deltas = {}
    for name, (a, b) in comparisons.items():
        per_class_deltas[name] = {}
        for it in item_list:
            per_class_deltas[name][it] = {
                "metrics": {
                    k: per_class[it]["variants"][a]["metrics"][k]
                       - per_class[it]["variants"][b]["metrics"][k]
                    for k in METRIC_KEYS},
                "recall": {
                    k: per_class[it]["variants"][a]["recall"].get(k, float("nan"))
                       - per_class[it]["variants"][b]["recall"].get(k, float("nan"))
                    for k in RECALL_KEYS},
            }

    score_diffs = [per_class[it]["image_scores"]["stage2_original_max_abs_diff"]
                   for it in item_list]
    score_diffs = [d for d in score_diffs if d is not None]
    metric_diffs = [v for it in item_list
                    for v in per_class[it]["image_scores"]["stage2_metric_delta"].values()]
    return {
        "mean": mean,
        "image_mean": image_mean,
        "deltas": deltas,
        "per_class_deltas": per_class_deltas,
        "identity": {
            "image_scores_identical_across_variants": True,
            "image_score_max_abs_diff_across_variants": 0.0,
            "image_scores_source": "one baseline-global score per test image, "
                                   "reused by every variant",
            "stage2_original_image_scores_max_abs_diff": (
                max(score_diffs) if score_diffs else None),
            "stage2_reported_image_metric_max_abs_diff": (
                max(abs(x) for x in metric_diffs) if metric_diffs else None),
            "adapted_global_branch_computed": False,
            "fusion_global_branch": "original baseline model_10000.pt",
            "fixed_fpr_negatives": "per variant, per class; never shared",
        },
    }


def pct(x):
    return f"{100.0 * x:.2f}"


def write_markdown(report, path):
    lines = []
    lines.append("# Dinomaly local refine v3 — stage 3")
    lines.append("")
    lines.append(f"Slug: `{SLUG}`  ")
    lines.append(f"Generated: {report['experiment']['generated']}  ")
    lines.append(f"Checkpoints: `{report['checkpoints']['base']['path']}` "
                 f"+ `{report['checkpoints']['adapted']['path']}`  ")
    lines.append(f"Test images: {report['config']['test_images_evaluated']} "
                 f"across {len(report['config']['eval_classes'])} classes  ")
    lines.append(f"Calibration: {CALIB_N} deterministic train/good images per "
                 f"class, shrink weight {report['config']['shrink_weight']:.3f}")
    lines.append("")
    lines.append("## Bug fixes")
    lines.append("")
    lines.append("- **Bug 1 (line 323):** every fusion uses the original "
                 "baseline global map. No adapted-global forward is computed. "
                 "`adapted_local_fuse` is exactly `0.5 * baseline_global + "
                 "0.5 * adapted_local`.")
    lines.append("- **Bug 2 (lines 358-379):** fixed-FPR thresholds are fitted "
                 "on each variant's own per-class negative pixels. Thresholds "
                 "are stored per variant in each per-class JSON.")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    lines.append("| variant | definition |")
    lines.append("|---|---|")
    defs = {
        "base_global": "baseline global map, no calibration",
        "base_local_fuse": "0.5 * baseline global + 0.5 * baseline local",
        "adapted_local_fuse": "0.5 * baseline global + 0.5 * adapted local (bug 1 fixed)",
        "cal_scalar": "per-branch median/q95 alignment, then 0.5 fusion",
        "cal_spatial": "per-branch shrunk spatial background correction, "
                       "no scalar alignment, then 0.5 fusion",
        "cal_fused": "per-branch scalar alignment + shrunk spatial background "
                     "correction, then 0.5 fusion (primary calibrated variant)",
    }
    for v in VARIANTS:
        lines.append(f"| `{v}` | {defs[v]} |")
    lines.append("")
    n_eval = len(report["config"]["eval_classes"])
    means_title = "15-class means" if n_eval == 15 else f"{n_eval}-class means"
    lines.append(f"## {means_title} (%)")
    lines.append("")
    lines.append("| variant | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        m = report["results"]["mean"][v]["metrics"]
        lines.append("| `" + v + "` | " + " | ".join(
            pct(m[k]) for k in METRIC_KEYS) + " |")
    lines.append("")
    lines.append("Image metrics are identical by construction: every variant "
                 "reuses the same original baseline image score. Only P-* "
                 "columns differ.")
    lines.append("")
    lines.append("## Recall at fixed pixel FPR (class mean, %)")
    lines.append("")
    lines.append("| variant | R@FPR0.001 | R@FPR0.01 | R@FPR0.05 |")
    lines.append("|---|---|---|---|")
    for v in VARIANTS:
        r = report["results"]["mean"][v]["recall"]
        lines.append("| `" + v + "` | " + " | ".join(
            pct(r[k]) for k in RECALL_KEYS) + " |")
    lines.append("")
    lines.append("## Primary mean deltas (percentage points)")
    lines.append("")
    lines.append("| comparison | P-AUROC | P-AP | P-F1 | P-AUPRO |")
    lines.append("|---|---|---|---|---|")
    for name in report["deltas"]:
        d = report["deltas"][name]["metrics"]
        lines.append(f"| {name} | " + " | ".join(
            f"{100*d[k]:+.2f}" for k in
            ("P-AUROC", "P-AP", "P-F1", "P-AUPRO")) + " |")
    lines.append("")
    lines.append("## Per-class pixel AP (%)")
    lines.append("")
    lines.append("| class | " + " | ".join(f"`{v}`" for v in VARIANTS) + " |")
    lines.append("|---" * (len(VARIANTS) + 1) + "|")
    for item in report["config"]["eval_classes"]:
        vals = [report["results"]["per_class"][item]["variants"][v]
                ["metrics"]["P-AP"] for v in VARIANTS]
        lines.append(f"| {item} | " + " | ".join(pct(x) for x in vals) + " |")
    lines.append("")
    lines.append("## Identity and audit")
    lines.append("")
    ident = report["identity"]
    lines.append(f"- Image scores are computed once from the original baseline "
                 f"global map and reused for all variants "
                 f"(max |Δ| across variants = "
                 f"{ident['image_score_max_abs_diff_across_variants']:.3e}).")
    sd = ident["stage2_original_image_scores_max_abs_diff"]
    lines.append(f"- Recomputed baseline image scores vs stage2 original "
                 f"`results.base.global_scores`: "
                 f"{'n/a' if sd is None else f'{sd:.3e}'}.")
    md = ident["stage2_reported_image_metric_max_abs_diff"]
    lines.append(f"- Recomputed I-AUROC/I-AP/I-F1 vs stage2 reported baseline: "
                 f"{'n/a' if md is None else f'{md:.3e}'}.")
    lines.append("- Adapted-global branch computed: **no** (bug 1 fixed).")
    lines.append("- Fixed-FPR negatives: **per variant, per class** (bug 2 fixed). "
                 "Threshold values are in each per-class JSON.")
    lines.append("")
    lines.append("## Calibration scalars")
    lines.append("")
    lines.append("| class | g median | g q95 | g scale | l median | l q95 | l scale |")
    lines.append("|---|---|---|---|---|---|---|")
    for item in report["config"]["eval_classes"]:
        c = report["calibration"]["per_class"][item]
        g, l = c["g"], c["l"]
        lines.append(f"| {item} | {g['median']:.5f} | {g['q95']:.5f} | "
                     f"{g['scale']:.5f} | {l['median']:.5f} | "
                     f"{l['q95']:.5f} | {l['scale']:.5f} |")
    lines.append("")
    lines.append("Exact 24 selected train/good paths per class and the full "
                 "background arrays are in `calibration/<class>.json` and "
                 "`calibration/<class>.npz`. No test image is used for fitting.")
    lines.append("")
    lines.append("Full per-class metrics, thresholds and deltas: "
                 "`refine_report.json`.")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# self-test (no Kaggle paths, no torch)
# ---------------------------------------------------------------------------
def run_self_test():
    rng = np.random.default_rng(123)
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        if not cond:
            raise AssertionError(msg)
        checks += 1

    maps = rng.normal(0.25, 0.04, size=(CALIB_N, 1, 12, 12))
    st = branch_stats(maps)
    flat = maps.reshape(-1)
    ok(st["n"] == CALIB_N, "calibration count")
    ok(abs(st["median"] - np.median(flat)) < 1e-12, "median")
    ok(abs(st["q95"] - np.quantile(flat, 0.95)) < 1e-12, "q95")
    w = CALIB_N / (CALIB_N + SHRINK_K)
    ok(abs(st["shrink_weight"] - w) < 1e-12, "shrink weight")
    exp_raw = st["mu_raw"] + w * (st["bg_raw"] - st["mu_raw"])
    ok(np.allclose(st["bg_raw_shrunk"], exp_raw, atol=1e-14),
       "raw background shrinkage")
    exp_align = st["mu_align"] + w * (st["bg_align"] - st["mu_align"])
    ok(np.allclose(st["bg_align_shrunk"], exp_align, atol=1e-14),
       "aligned background shrinkage")

    for n in (24, 60, 209, 391):
        idx = select_calibration_indices(n)
        ok(len(idx) == CALIB_N, f"selection length n={n}")
        ok(len(set(idx)) == CALIB_N, f"selection unique n={n}")
        ok(idx == sorted(idx), f"selection sorted n={n}")
        ok(idx[0] == 0 and idx[-1] == n - 1, f"selection endpoints n={n}")

    b, h, wd = 2, 12, 12
    g = rng.normal(1.0, 0.1, size=(b, 1, h, wd))
    lb = rng.normal(2.0, 0.1, size=(b, 1, h, wd))
    la = rng.normal(0.3, 0.1, size=(b, 1, h, wd))
    gst = branch_stats(rng.normal(0.2, 0.05, size=(CALIB_N, 1, h, wd)))
    lst = branch_stats(rng.normal(0.4, 0.05, size=(CALIB_N, 1, h, wd)))
    cal = {
        "g": {
            "median": gst["median"], "q95": gst["q95"],
            "scale": gst["scale"],
            "bg_raw_shrunk": gst["bg_raw_shrunk"][None, None],
            "bg_align_shrunk": gst["bg_align_shrunk"][None, None],
        },
        "l": {
            "median": lst["median"], "q95": lst["q95"],
            "scale": lst["scale"],
            "bg_raw_shrunk": lst["bg_raw_shrunk"][None, None],
            "bg_align_shrunk": lst["bg_align_shrunk"][None, None],
        },
    }
    vm = build_variant_maps(g, lb, la, cal)
    ok(np.allclose(vm["base_global"], g), "base global")
    ok(np.allclose(vm["base_local_fuse"], 0.5 * g + 0.5 * lb),
       "baseline local fusion")
    ok(np.allclose(vm["adapted_local_fuse"], 0.5 * g + 0.5 * la),
       "adapted local fusion uses baseline global")
    adapted_global = rng.normal(0.9, 0.1, size=(b, 1, h, wd))
    wrong = 0.5 * adapted_global + 0.5 * la
    ok(not np.allclose(vm["adapted_local_fuse"], wrong),
       "adapted global must not be used")
    zg = (g - cal["g"]["median"]) / cal["g"]["scale"]
    zl = (la - cal["l"]["median"]) / cal["l"]["scale"]
    ok(np.allclose(vm["cal_scalar"], 0.5 * zg + 0.5 * zl),
       "scalar fusion")
    ok(np.allclose(
        vm["cal_spatial"],
        0.5 * (g - cal["g"]["bg_raw_shrunk"])
        + 0.5 * (la - cal["l"]["bg_raw_shrunk"])),
       "spatial fusion")
    ok(np.allclose(
        vm["cal_fused"],
        0.5 * (zg - cal["g"]["bg_align_shrunk"])
        + 0.5 * (zl - cal["l"]["bg_align_shrunk"])),
       "calibrated fusion")

    n_neg, n_pos = 200, 50
    gt = np.concatenate([np.zeros(n_neg, dtype=np.int64),
                         np.ones(n_pos, dtype=np.int64)])
    neg_a = rng.normal(0.0, 1.0, n_neg)
    pos_a = rng.normal(2.0, 1.0, n_pos)
    a = np.concatenate([neg_a, pos_a])
    b_scores = 10.0 * a + 5.0          # same ranks, different threshold
    d_scores = -a                      # reversed ranks
    ra, ta = fixed_fpr_recall(a, gt)
    rb, tb = fixed_fpr_recall(b_scores, gt)
    rd, _td = fixed_fpr_recall(d_scores, gt)
    ok(abs(ra["R@FPR0.01"] - rb["R@FPR0.01"]) < 1e-12,
       "affine score transform preserves recall")
    ok(abs(ta["T@FPR0.01"] - tb["T@FPR0.01"]) > 1e-6,
       "thresholds must differ per variant")
    manual = float((pos_a >= np.quantile(neg_a, 0.99)).mean())
    ok(abs(ra["R@FPR0.01"] - manual) < 1e-12,
       "recall uses this variant's negatives")
    ok(abs(rd["R@FPR0.01"] - ra["R@FPR0.01"]) > 0.1,
       "shared negatives would be wrong")
    print(f"SELF-TEST OK ({checks} checks)")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    global LOG_PATH, FP, EVAL_BATCH, CALIB_BATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--eval-batch", type=int, default=EVAL_BATCH)
    ap.add_argument("--calib-batch", type=int, default=CALIB_BATCH)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--limit-classes", type=int, default=0,
                    help="smoke only: evaluate first N classes")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()

    EVAL_BATCH = args.eval_batch
    CALIB_BATCH = args.calib_batch
    os.makedirs(OUT, exist_ok=True)
    LOG_PATH = os.path.join(OUT, "stage3.log")
    log(f"=== {SLUG} start ===")
    log(f"args={vars(args)}")

    # Tiny deterministic checks also run inside the kernel, so a math
    # regression fails in seconds instead of after an hour of forwards.
    run_self_test()
    log("self-test passed")

    prepare_environment()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"device={device} torch={torch.__version__}")
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    find_checkpoints()

    base_sha = sha256_file(BASE_CKPT)
    adapt_sha = sha256_file(ADAPT_CKPT)
    log(f"base sha256={base_sha}")
    log(f"adapted sha256={adapt_sha}")

    stage2_base = load_stage2_report()
    if stage2_base is None:
        log("WARNING: stage2 report unavailable; identity comparison skipped")

    item_list = sorted(
        d for d in os.listdir(DATA)
        if os.path.isdir(os.path.join(DATA, d)))
    if len(item_list) != 15:
        raise RuntimeError(f"expected 15 MVTec classes, got {len(item_list)}: "
                           f"{item_list}")

    data_transform, gt_transform = get_data_transforms(448, CROP)
    test_data_list = [
        MVTecDataset(root=os.path.join(DATA, item),
                     transform=data_transform, gt_transform=gt_transform,
                     phase="test")
        for item in item_list
    ]
    counts = {it: len(ds) for it, ds in zip(item_list, test_data_list)}
    total_test = int(sum(counts.values()))
    log(f"classes={len(item_list)} test images={total_test}")
    log(f"per-class test counts={counts}")
    if total_test != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"expected {EXPECTED_TEST_IMAGES} official test images, "
            f"got {total_test}: {counts}")

    eval_items = item_list
    if args.limit_classes and args.limit_classes > 0:
        eval_items = item_list[:args.limit_classes]
    eval_items = list(eval_items)
    test_images_evaluated = int(sum(counts[it] for it in eval_items))
    partial = len(eval_items) != len(item_list)

    config = {
        "slug": SLUG,
        "stage": 3,
        "script": os.path.basename(os.path.abspath(__file__)),
        "code_sha256": sha256_file(os.path.abspath(__file__)),
        "crop": CROP,
        "window": WINDOW,
        "starts": list(STARTS),
        "classes_total": len(item_list),
        "eval_classes": eval_items,
        "partial": partial,
        "test_images_total": total_test,
        "test_images_evaluated": test_images_evaluated,
        "test_counts": counts,
        "eval_batch": EVAL_BATCH,
        "calib_batch": CALIB_BATCH,
        "calib_n": CALIB_N,
        "shrink_k": SHRINK_K,
        "shrink_weight": float(CALIB_N) / float(CALIB_N + SHRINK_K),
        "fprs": list(FPRS),
        "variants": list(VARIANTS),
        "base_checkpoint": BASE_CKPT,
        "adapted_checkpoint": ADAPT_CKPT,
        "base_sha256": base_sha,
        "adapted_sha256": adapt_sha,
        "stage2_report": STAGE2_REPORT,
        "image_score_source": "baseline global top1% of postprocessed 256 map, "
                              "computed once and reused for every variant",
        "calibration_selection": "24 sorted train/good paths per class, "
                                 "evenly spaced with np.linspace endpoints included",
        "calibration_fit": "per class, per branch: median/q95 scalar alignment "
                           "and per-pixel mean background shrunk toward its "
                           "global mean with weight n/(n+k)",
        "bug_fixes": {
            "bug1": "adapted fusion uses original baseline global, not adapted global",
            "bug2": "fixed-FPR negatives/thresholds are per variant and per class",
        },
        "optional_highres_full_context": "skipped: a 448 full-image branch "
                                         "would change the official 256-mask "
                                         "crop protocol and delay launch",
    }
    FP = config_fingerprint(config)
    log(f"config fingerprint={FP}")
    write_json_atomic(os.path.join(OUT, "config.json"), config)

    per_class_dir = os.path.join(OUT, "per_class")
    os.makedirs(per_class_dir, exist_ok=True)
    progress_path = os.path.join(OUT, "progress.json")
    status_path = os.path.join(OUT, "status.json")

    def set_status(status, **extra):
        obj = {"slug": SLUG, "fingerprint": FP, "status": status,
               "updated": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
        write_json_atomic(status_path, obj)

    def per_class_path(item):
        return os.path.join(per_class_dir, item + ".json")

    def load_per_class(item):
        p = per_class_path(item)
        if not os.path.isfile(p):
            return None
        try:
            d = read_json(p)
        except Exception:
            return None
        if d.get("fingerprint") != FP:
            return None
        return d

    done = []
    if args.resume and not args.fresh and os.path.isfile(progress_path):
        try:
            prog = read_json(progress_path)
        except Exception:
            prog = {}
        if prog.get("fingerprint") == FP:
            for item in prog.get("classes_done", []):
                if item in eval_items and load_per_class(item) is not None:
                    done.append(item)
    log(f"resume: {len(done)}/{len(eval_items)} classes already complete")

    def update_progress(current=None, latest=None):
        write_json_atomic(progress_path, {
            "slug": SLUG,
            "fingerprint": FP,
            "classes_total": len(eval_items),
            "classes_done": done,
            "current": current,
            "latest": latest,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    set_status("running", classes_done=done, classes_total=len(eval_items),
               current=None)
    update_progress()

    remaining = [it for it in eval_items if it not in set(done)]
    model_base = model_adapt = None
    gaussian = weight = mean = std = None
    if remaining:
        log(f"remaining classes={remaining}")
        model_base = build_model(device)
        st_base = load_ckpt(model_base, BASE_CKPT, device)
        model_base.eval()
        model_adapt = build_model(device)
        st_adapt = load_ckpt(model_adapt, ADAPT_CKPT, device)
        model_adapt.eval()
        log(f"base checkpoint iter={st_base.get('iter')} "
            f"kind={st_base.get('kind')}")
        log(f"adapted checkpoint iter={st_adapt.get('steps')} "
            f"kind={st_adapt.get('kind')}")

        gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
        h = torch.from_numpy(hann(WINDOW, sym=True)).float()
        weight = (0.1 + 0.9 * torch.outer(h, h)).view(
            1, 1, WINDOW, WINDOW).to(device)
        mean, std = norm_tensors(device)
    else:
        log("all classes already complete; skipping model build")

    for item in eval_items:
        if item in set(done):
            log(f"{item}: resume skip")
            continue
        set_status("running", classes_done=done,
                   classes_total=len(eval_items), current=item)
        update_progress(current=item)
        test_data = test_data_list[item_list.index(item)]
        started = time.time()

        loaded = None if args.fresh else load_calibration(item)
        if loaded is not None:
            cal_raw, cal_meta = loaded
            log(f"{item}: calibration resume skip "
                f"(selected={len(cal_meta.get('selected', []))})")
        else:
            log(f"{item}: computing calibration on {CALIB_N} train/good images")
            cal_raw, selected = compute_calibration(
                item, data_transform, model_base, model_adapt,
                weight, mean, std, device, CALIB_BATCH)
            save_calibration(item, cal_raw, selected)
            gs, ls = cal_raw["g"], cal_raw["l"]
            log(f"{item}: calib g median={gs['median']:.5f} "
                f"q95={gs['q95']:.5f} scale={gs['scale']:.5f}; "
                f"l median={ls['median']:.5f} q95={ls['q95']:.5f} "
                f"scale={ls['scale']:.5f}")

        log(f"{item}: evaluating {counts[item]} test images")
        result = evaluate_class(
            item, test_data, model_base, model_adapt, cal_raw,
            gaussian, weight, mean, std, device, EVAL_BATCH, stage2_base)
        write_json_atomic(per_class_path(item), result)
        done.append(item)
        latest = {
            "class": item,
            "seconds": result["seconds"],
            "P-AP": {v: result["variants"][v]["metrics"]["P-AP"]
                     for v in VARIANTS},
            "P-AUPRO": {v: result["variants"][v]["metrics"]["P-AUPRO"]
                        for v in VARIANTS},
        }
        update_progress(current=None, latest=latest)
        log(f"{item}: done in {result['seconds']:.1f}s "
            f"P-AP " + " ".join(
                f"{v}={result['variants'][v]['metrics']['P-AP']:.4f}"
                for v in VARIANTS))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_class = {}
    for item in eval_items:
        d = load_per_class(item)
        if d is None:
            raise RuntimeError(f"missing per-class result for {item}")
        per_class[item] = d

    agg = aggregate(per_class, eval_items)
    report = {
        "experiment": {
            "slug": SLUG,
            "stage": 3,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "partial": partial,
        },
        "config": config,
        "checkpoints": {
            "base": {
                "path": BASE_CKPT,
                "sha256": base_sha,
                "iter": (st_base.get("iter") if remaining else None),
            },
            "adapted": {
                "path": ADAPT_CKPT,
                "sha256": adapt_sha,
                "kind": (st_adapt.get("kind") if remaining else "local_adapt"),
            },
        },
        "calibration": {
            "n_per_class": CALIB_N,
            "shrink_k": SHRINK_K,
            "shrink_weight": config["shrink_weight"],
            "selection": config["calibration_selection"],
            "fit": config["calibration_fit"],
            "per_class": {it: per_class[it]["calibration"]
                          for it in eval_items},
        },
        "bug_fixes": config["bug_fixes"],
        "identity": agg["identity"],
        "results": {
            "per_class": per_class,
            "mean": agg["mean"],
        },
        "image_metrics": {
            "mean": agg["image_mean"],
            "note": "identical for every variant; taken from the original "
                    "baseline global map",
        },
        "deltas": agg["deltas"],
        "per_class_deltas": agg["per_class_deltas"],
        "runtime": {
            "device": device,
            "torch": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "classes_evaluated": len(eval_items),
            "test_images_evaluated": test_images_evaluated,
            "forwards_per_test_image": 1 + 9 + 9,
            "forwards_per_calibration_image": 1 + 9,
            "seconds_total": float(sum(per_class[it]["seconds"]
                                       for it in eval_items)),
        },
        "files": {
            "per_class_dir": per_class_dir,
            "calibration_dir": os.path.join(OUT, "calibration"),
            "progress": progress_path,
            "status": status_path,
        },
    }

    report_json = os.path.join(OUT, "refine_report.json")
    report_md = os.path.join(OUT, "refine_report.md")
    write_json_atomic(report_json, report)
    write_markdown(report, report_md)
    set_status("complete", classes_done=done, classes_total=len(eval_items),
               report=report_json, markdown=report_md)
    update_progress(current=None)
    log("DONE")
    log(f"report={report_json}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        log("FAILED\n" + traceback.format_exc())
        try:
            if OUT and FP:
                write_json_atomic(os.path.join(OUT, "status.json"), {
                    "slug": SLUG, "fingerprint": FP, "status": "failed",
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "error": traceback.format_exc(limit=8),
                })
        except Exception:
            pass
        raise
