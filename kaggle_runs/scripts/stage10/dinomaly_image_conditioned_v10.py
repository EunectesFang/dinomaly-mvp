"""Stage 10: image-conditioned calibration of the stage-7 guided pixel map.

Frozen 10k checkpoint, no training, no test-label fitting.

Hypothesis: the original 392 top-1% image score is already excellent at image
level (I-AUROC 0.9962), but heatmaps from normal test images still compete with
anomaly heatmaps in pixel ranking.  Calibrating that original 392 score against
deterministic train/good images and softly multiplying each pixel map by an
image-confidence gate should suppress normal-image false positives without
hurting anomaly localization.

Protocol is identical to stage 7 for the two forwards used:
    original image -> Resize(448) -> CenterCrop(392) -> Normalize -> 392 map
    original image -> Resize(768) -> CenterCrop(672) -> Normalize -> 672 map
    both maps -> bilinear 256 -> gaussian(5,4)
    exact fusion = 0.5 * base_global + 0.5 * hi672
    exact prior-guide reference = stage-7 grayscale guided filter on fusion
The original baseline top-1% image score is computed once per test image and
reused unchanged; it is never gated or replaced.  Masks are the exact original
448->392 ground-truth transform nearest-resized to 256.  All 15 classes /
1725 official test images are evaluated.

Calibration uses ONLY train/good images, never test labels:
    deterministic 48 images per class (sorted paths, evenly spaced)
    original 392 map -> top1% -> q95, q99, median
    mild gate   g = .10 + .90 * sigmoid((s-q95)/max(q95-median,.005) * 2)
    sharp gate  g = .03 + .97 * sigmoid((s-q99)/max(q99-median,.005) * 2)
where s is the test image's original 392 top1% score.  Pixel maps are then
scaled by g (a per-image non-negative factor), so within-image ranking is
untouched and only cross-image competition is reweighted.

Four pixel variants on the same 256 grid:
    guide_baseline          exact stage-7 guide_fuse672, no gate
    guide_gate_mild         mild-gated guide map
    guide_gate_sharp        sharp-gated guide map
    guide_gate_mild_medsub  mild gate * (guide - per-image median)
The fourth is cheap; it tests whether removing each image's median background
level further separates normal from anomalous maps.

No test labels are used to select, tune, fit, or gate anything.  Gate floors,
midpoints, quantiles, width scale and calibration count are fixed a priori.
Results are atomically written per class.  Markdown is deliberately omitted;
the report is strict (NaN-free) JSON.
"""
import os
import sys
import json
import math
import time
import argparse
import hashlib
import glob
import traceback

import numpy as np

OUT = "/kaggle/working"
SLUG = "eunectesqaq/dinomaly-image-conditioned-v10"
BASE_RESIZE = 448
BASE_CROP = 392
SCALE_SPECS = ((672, 768),)              # only the 48x48 token branch
EXPECTED_TEST_IMAGES = 1725
EVAL_BATCH = 4
FPRS = (0.001, 0.01, 0.05)
METRIC_KEYS = ("I-AUROC", "I-AP", "I-F1",
               "P-AUROC", "P-AP", "P-F1", "P-AUPRO",
               "P-AUROC-anom", "P-AP-anom")
RECALL_KEYS = tuple(f"R@FPR{f:g}" for f in FPRS)
GUIDE_RADIUS = 7
GUIDE_EPS = 0.0025
DETAIL_STRENGTH = 0.5
DETAIL_CAP_FRAC = 0.10
CALIB_IMAGES = 48
CALIB_MIN_DENOM = 0.005
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GATE_MILD = {"floor": 0.10, "mid_key": "q95", "width_scale": 2.0,
             "min_denom": CALIB_MIN_DENOM}
GATE_SHARP = {"floor": 0.03, "mid_key": "q99", "width_scale": 2.0,
              "min_denom": CALIB_MIN_DENOM}
FINAL_VARIANTS = ("guide_baseline", "guide_gate_mild",
                  "guide_gate_sharp", "guide_gate_mild_medsub")

LOG_PATH = None
FP = None
CODE = None
DATA = None
BASE_CKPT = None
STAGE1_METRICS = None
CALIBRATION = {}      # class -> q95/q99/median/denominators/score stats
BASE_BATCH = EVAL_BATCH
ENABLED = []          # list of (crop, resize)
SCALE_BATCH = {}      # crop -> effective CUDA batch
VARIANTS = FINAL_VARIANTS

# Runtime imports are bound by prepare_environment().
torch = None
nn = None
F = None
DataLoader = None
Image = None
partial = None
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


def _json_safe(obj):
    """Convert non-finite / NumPy scalars so the report is strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=2, ensure_ascii=False,
                  allow_nan=False)
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


def reset_peak():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_mib():
    if torch is not None and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / 1024 ** 2)
    return 0.0


def sync():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# pure-numpy metric / fusion helpers (self-testable)
# ---------------------------------------------------------------------------
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


def variant_names(crops):
    names = ["base_global"]
    for c in crops:
        names.extend([f"hi{c}", f"fuse{c}"])
    if len(crops) >= 2:
        names.append("fuse3")
    return tuple(names)


def fov(crop, resize):
    return float(crop) / float(resize)


def finite_mean(values):
    vals = [float(v) for v in values
            if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _box_mean(x, radius):
    """Exact box mean; count_include_pad=False keeps borders unbiased."""
    radius = int(radius)
    k = 2 * radius + 1
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=radius,
                        count_include_pad=False)


def guided_filter(guide, heatmap, radius=GUIDE_RADIUS, eps=GUIDE_EPS):
    """Stable single-channel local-linear guided filter.

    guide and heatmap are [B, 1, H, W].  The variance is clamped non-negative
    before dividing, which avoids NaNs on flat guide regions.
    """
    guide = guide.clamp(0.0, 1.0).float()
    heatmap = heatmap.float()
    mi = _box_mean(guide, radius)
    mp = _box_mean(heatmap, radius)
    corr_i = _box_mean(guide * guide, radius)
    corr_ip = _box_mean(guide * heatmap, radius)
    var_i = (corr_i - mi * mi).clamp_min(0.0)
    cov_ip = corr_ip - mi * mp
    a = cov_ip / (var_i + float(eps))
    b = mp - a * mi
    return _box_mean(a, radius) * guide + _box_mean(b, radius)


def guide_from_672_image(x_hi):
    """De-normalize the exact 672 crop to RGB 0..1, resize 256, grayscale."""
    mean = x_hi.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x_hi.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    rgb = (x_hi * std + mean).clamp(0.0, 1.0)
    rgb = F.interpolate(rgb, size=256, mode="bilinear",
                        align_corners=False)
    gray = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2]
            + 0.114 * rgb[:, 2:3])
    return gray.clamp(0.0, 1.0)


def refine_heatmap(heatmap, guide, radius=GUIDE_RADIUS, eps=GUIDE_EPS,
                   strength=DETAIL_STRENGTH, cap_frac=DETAIL_CAP_FRAC):
    """Return box2 control, full guided output, capped detail residual."""
    box2 = _box_mean(_box_mean(heatmap, radius), radius)
    guided = guided_filter(guide, heatmap, radius, eps)
    flat = heatmap.flatten(1)
    scale = (torch.quantile(flat, 0.99, dim=1)
             - torch.quantile(flat, 0.01, dim=1)).clamp_min(1e-4)
    cap = (float(cap_frac) * scale).view(-1, 1, 1, 1)
    hp = guided - box2
    clipped = hp.clamp(-cap, cap)
    detail = heatmap + float(strength) * clipped
    clip_frac = (hp.abs() > cap).float().mean()
    return box2, guided, detail, cap, clip_frac


def calibration_quantiles(scores):
    """Class-level quantiles from train/good scores only.

    The denominators are clamped so a degenerate or tiny calibration sample
    cannot produce a step function or division by zero.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("calibration needs at least one train/good score")
    q95 = float(np.quantile(scores, 0.95))
    q99 = float(np.quantile(scores, 0.99))
    med = float(np.median(scores))
    return {
        "n": int(scores.size),
        "min": float(scores.min()),
        "median": med,
        "q95": q95,
        "q99": q99,
        "max": float(scores.max()),
        "denom95": max(q95 - med, CALIB_MIN_DENOM),
        "denom99": max(q99 - med, CALIB_MIN_DENOM),
        "min_denom_clamped": bool(q95 - med < CALIB_MIN_DENOM
                                  or q99 - med < CALIB_MIN_DENOM),
        "scores_sha256": hashlib.sha256(
            np.sort(scores).astype(np.float64).tobytes()).hexdigest(),
    }


def soft_gate(scores, calib, cfg):
    """Per-image confidence gate in [floor, 1], from the original 392 score."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    mid = float(calib[cfg["mid_key"]])
    denom = max(mid - float(calib["median"]),
                float(cfg["min_denom"]))
    logit = (scores - mid) / denom * float(cfg["width_scale"])
    # tanh form is numerically stable for arbitrarily large |logit|.
    sigmoid = 0.5 * (1.0 + np.tanh(0.5 * logit))
    gate = float(cfg["floor"]) + (1.0 - float(cfg["floor"])) * sigmoid
    return gate.astype(np.float64)


def gate_maps(guided, gates, torch_mod):
    """Scale each pixel map by its image gate.  Within-image rank unchanged."""
    g = torch_mod.as_tensor(
        np.asarray(gates, dtype=np.float32),
        dtype=guided.dtype, device=guided.device).view(-1, 1, 1, 1)
    return guided * g


def median_subtract(heatmap, torch_mod):
    """Subtract each image's own median level; cheap and rank preserving."""
    med = torch_mod.median(heatmap.flatten(1), dim=1).values
    return heatmap - med.view(-1, 1, 1, 1)


# ---------------------------------------------------------------------------
# Kaggle runtime setup
# ---------------------------------------------------------------------------
def prepare_environment():
    global CODE, DATA
    global torch, nn, F, DataLoader, Image, partial, measure
    global roc_auc_score, average_precision_score, auc
    global get_data_transforms, MVTecDataset, ViTill, vit_encoder
    global VitBlock, bMlp, LinearAttention2
    global cal_anomaly_maps, get_gaussian_kernel, f1_score_max

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from PIL import Image
    from functools import partial
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
    global BASE_CKPT, STAGE1_METRICS
    BASE_CKPT = find_input("model_10000.pt", [
        "/kaggle/input/dinomaly-base-10k/model_10000.pt",
        "/kaggle/input/notebooks/eunectesqaq/dinomaly-base-10k/model_10000.pt",
        "/kaggle/input/datasets/eunectesqaq/dinomaly-base-10k/model_10000.pt",
    ])
    STAGE1_METRICS = find_input("metrics.json", [
        "/kaggle/input/dinomaly-base-10k/metrics.json",
        "/kaggle/input/notebooks/eunectesqaq/dinomaly-base-10k/metrics.json",
        "/kaggle/input/datasets/eunectesqaq/dinomaly-base-10k/metrics.json",
    ])
    if not BASE_CKPT:
        raise RuntimeError("model_10000.pt not found under /kaggle/input")
    log(f"base checkpoint={BASE_CKPT}")
    log(f"stage1 metrics={STAGE1_METRICS}")


def load_stage1_metrics():
    if not STAGE1_METRICS:
        return None
    try:
        d = read_json(STAGE1_METRICS)
        if "10000" in d:
            d = d["10000"]
        else:
            keys = [k for k in d if str(k).isdigit()]
            if keys:
                d = d[str(max(int(k) for k in keys))]
        if not isinstance(d, dict) or "per_class" not in d:
            log("stage1 metrics has no per_class; identity check skipped")
            return None
        log(f"stage1 metrics classes={len(d.get('per_class', {}))}")
        return d
    except Exception as exc:
        log(f"could not read stage1 metrics: {exc}")
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


def global_map(model, images):
    en, de = model(images)
    amap, _ = cal_anomaly_maps(en, de, images.shape[-1])
    return amap


def global_map_chunked(model, images, chunk):
    """Forward in chunks, halving once more on CUDA OOM."""
    if chunk is None or chunk <= 0:
        chunk = int(images.shape[0])
    if chunk >= images.shape[0]:
        try:
            return global_map(model, images)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and images.shape[0] > 1:
                torch.cuda.empty_cache()
                log(f"WARNING: forward OOM at batch {images.shape[0]}; "
                    f"retrying at {max(1, images.shape[0] // 2)}")
                return global_map_chunked(
                    model, images, max(1, images.shape[0] // 2))
            raise
    outs = []
    i = 0
    while i < images.shape[0]:
        part = images[i:i + chunk]
        try:
            outs.append(global_map(model, part))
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and part.shape[0] > 1:
                torch.cuda.empty_cache()
                log(f"WARNING: forward OOM at batch {part.shape[0]}; "
                    f"retrying at {max(1, part.shape[0] // 2)}")
                outs.append(global_map_chunked(
                    model, part, max(1, part.shape[0] // 2)))
            else:
                raise
        i += chunk
    return torch.cat(outs, dim=0)


def postprocess(amap, gaussian):
    return gaussian(F.interpolate(amap, size=256, mode="bilinear",
                                  align_corners=False))


def top1pct(amap):
    flat = amap.flatten(1)
    k = max(1, int(flat.shape[1] * 0.01))
    return torch.sort(flat, dim=1, descending=True).values[:, :k].mean(dim=1)


# ---------------------------------------------------------------------------
# scale probe: valid shape + memory batch + speed, no labels used
# ---------------------------------------------------------------------------
def probe_forward(model, crop, batch, device):
    x = torch.zeros(batch, 3, crop, crop, device=device)
    sync()
    t0 = time.time()
    with torch.no_grad():
        amap = global_map(model, x)
    sync()
    dt = time.time() - t0
    expected = (int(batch), 1, int(crop), int(crop))
    if tuple(amap.shape) != expected:
        raise RuntimeError(f"scale {crop} map shape {tuple(amap.shape)} "
                           f"!= {expected}")
    del amap, x
    return float(dt)


def probe_scales(model, device, eval_batch, include_672, max_672_forward_min):
    """Choose effective batches and decide whether 672 is reasonably fast.

    Uses synthetic tensors only.  No test image or label participates.
    """
    results = {}
    base_batch = max(1, int(eval_batch))
    while True:
        try:
            reset_peak()
            dt = probe_forward(model, BASE_CROP, base_batch, device)
            results["base"] = {
                "crop": BASE_CROP, "resize": BASE_RESIZE,
                "batch": base_batch, "seconds": dt,
                "seconds_per_image": dt / base_batch,
                "peak_mib": peak_mib(), "enabled": True,
            }
            break
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and base_batch > 1:
                torch.cuda.empty_cache()
                base_batch //= 2
                log(f"baseline probe OOM; reducing batch to {base_batch}")
                continue
            raise

    for crop, resize in SCALE_SPECS:
        if crop == 672 and not include_672:
            results[crop] = {"crop": crop, "resize": resize,
                             "enabled": False, "reason": "disabled by --skip-672"}
            continue
        max_batch = min(max(1, int(eval_batch)), 4 if crop <= 560 else 2)
        batch = max_batch
        decided = False
        while batch >= 1:
            try:
                reset_peak()
                dt = probe_forward(model, crop, batch, device)
                results[crop] = {
                    "crop": crop, "resize": resize, "batch": batch,
                    "seconds": dt, "seconds_per_image": dt / batch,
                    "peak_mib": peak_mib(), "enabled": True,
                }
                decided = True
                break
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and batch > 1:
                    torch.cuda.empty_cache()
                    batch //= 2
                    log(f"scale {crop} probe OOM; reducing batch to {batch}")
                    continue
                if "out of memory" in str(exc).lower():
                    if crop == 672:
                        results[crop] = {"crop": crop, "resize": resize,
                                         "enabled": False,
                                         "reason": "CUDA OOM at batch 1"}
                        decided = True
                        break
                    raise
                raise
        if not decided:
            results[crop] = {"crop": crop, "resize": resize,
                             "enabled": False, "reason": "probe failed"}

    # Speed gate for 672: skip if per-image cost is >3x the 560 branch or the
    # projected added forward time over the official test set exceeds the cap.
    r560, r672 = results.get(560), results.get(672)
    if r560 and r560.get("enabled") and r672 and r672.get("enabled"):
        t560 = float(r560["seconds_per_image"])
        t672 = float(r672["seconds_per_image"])
        extra = max(0.0, t672 - t560) * EXPECTED_TEST_IMAGES
        if t560 > 0 and t672 > 3.0 * t560:
            r672["enabled"] = False
            r672["reason"] = (f"too slow: {t672:.4f}s/img vs "
                              f"{t560:.4f}s/img at 560")
        elif extra > float(max_672_forward_min) * 60.0:
            r672["enabled"] = False
            r672["reason"] = (f"projected extra forward time "
                              f"{extra / 60.0:.1f} min > cap "
                              f"{max_672_forward_min:.1f} min")
    return results


def apply_probe(results):
    global ENABLED, SCALE_BATCH, BASE_BATCH, VARIANTS
    BASE_BATCH = int(results["base"]["batch"])
    ENABLED = []
    SCALE_BATCH = {}
    for crop, resize in SCALE_SPECS:
        r = results.get(crop)
        if r and r.get("enabled"):
            ENABLED.append((int(crop), int(resize)))
            SCALE_BATCH[int(crop)] = int(r["batch"])
    if ENABLED != [(672, 768)]:
        raise RuntimeError(f"stage 10 requires exactly the 672 branch, "
                           f"got {ENABLED}")
    VARIANTS = FINAL_VARIANTS


# ---------------------------------------------------------------------------
# multi-scale dataset: standard sample stays exact, high-res reopens the
# native PIL image and resizes it independently
# ---------------------------------------------------------------------------
class MultiScaleTestDataset:
    def __init__(self, root, base_transform, gt_transform, scale_transforms):
        self.base = MVTecDataset(root=root, transform=base_transform,
                                 gt_transform=gt_transform, phase="test")
        self.scale_transforms = list(scale_transforms)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, gt, label, path = self.base[idx]
        pil = Image.open(path).convert("RGB")
        his = [tf(pil) for _crop, tf in self.scale_transforms]
        return img, gt, label, path, his


def deterministic_calibration_paths(root, n=CALIB_IMAGES):
    """Sorted train/good paths, evenly spaced to exactly n (no test image).

    Sorting plus even spacing is fully deterministic and does not depend on
    filesystem order, model outputs, test data, or labels.
    """
    paths = []
    for ext in ("*.png", "*.JPG", "*.bmp"):
        paths.extend(glob.glob(os.path.join(root, "train", "good", ext)))
    paths = sorted(paths)
    if not paths:
        raise RuntimeError(f"no train/good images under {root}")
    if len(paths) > n:
        idx = np.unique(np.linspace(0, len(paths) - 1, num=n).round()
                        .astype(np.int64))
        paths = [paths[int(i)] for i in idx]
    return paths


class TrainGoodCalibrationDataset:
    """Original 392 transform only; no ground truth, no test labels."""

    def __init__(self, root, transform, n=CALIB_IMAGES):
        self.paths = deterministic_calibration_paths(root, n)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), path


def calibrate_class(item, dataset, model, gaussian, device):
    """Original 392 top1% scores of deterministic train/good images only."""
    loader = DataLoader(dataset, batch_size=EVAL_BATCH, shuffle=False,
                        num_workers=4,
                        pin_memory=torch.cuda.is_available())
    scores = []
    started = time.time()
    with torch.no_grad():
        for img, _paths in loader:
            img = img.to(device, non_blocking=True)
            base = postprocess(global_map_chunked(model, img, BASE_BATCH),
                               gaussian)
            scores.extend(top1pct(base).detach().cpu().numpy().tolist())
            del base, img
    calib = calibration_quantiles(scores)
    calib.update({
        "class": item,
        "selection": "sorted train/good paths, evenly spaced",
        "n_selected": int(len(dataset)),
        "source": "original 448->392 global postprocessed top1% image score",
        "test_labels_used": False,
        "seconds": float(time.time() - started),
    })
    log(f"{item}: calibration n={calib['n']} "
        f"median={calib['median']:.6f} q95={calib['q95']:.6f} "
        f"q99={calib['q99']:.6f} max={calib['max']:.6f}")
    return calib


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


def label_regions(gt_np):
    return [[r.coords for r in measure.regionprops(measure.label(m))]
            for m in gt_np]


def compute_pro_pre(masks, amaps, regions, num_th=200):
    """Exact stage 3 / paper PRO implementation, evaluated on the 256 grid."""
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
                cur.append(bmap[coords[:, 0], coords[:, 1]].sum()
                           / len(coords))
        pros.append(float(np.mean(cur)) if cur else 0.0)
        fprs.append(float(np.logical_and(inverse, binmap).sum())
                    / neg_total)
    fprs = np.asarray(fprs)
    pros = np.asarray(pros)
    keep = fprs < 0.3
    if keep.sum() < 2 or fprs[keep].max() <= 0:
        return 0.0
    return float(auc(fprs[keep] / fprs[keep].max(), pros[keep]))


def _anomaly_only_metrics(labels_np, gt_np, pr):
    """AP/AUROC restricted to anomaly test images (diagnostic only)."""
    rows = labels_np == 1
    if not rows.any():
        return float("nan"), float("nan")
    gt_a = gt_np[rows].ravel()
    pr_a = pr[rows].ravel()
    if len(np.unique(gt_a)) < 2:
        return float("nan"), float("nan")
    return (float(roc_auc_score(gt_a, pr_a)),
            float(average_precision_score(gt_a, pr_a)))


def evaluate_class(item, dataset, model, gaussian, device, stage1_item,
                   calib):
    loader = DataLoader(dataset, batch_size=EVAL_BATCH, shuffle=False,
                        num_workers=4,
                        pin_memory=torch.cuda.is_available())
    buckets = {v: [] for v in VARIANTS}
    gt_buf, labels, scores = [], [], []
    gate_buf = {"mild": [], "sharp": []}
    n_batches = len(loader)
    started = time.time()
    reset_peak()

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            img, gt, label, _paths, his = batch
            img = img.to(device, non_blocking=True)
            his = [h.to(device, non_blocking=True) for h in his]

            # Baseline global map: exact stage 3 / stage 7 logic.
            base_raw = global_map_chunked(model, img, BASE_BATCH)
            base = postprocess(base_raw, gaussian)
            del base_raw

            # Original global image score, computed once and reused.  It is
            # also the only input to the image-conditioned gate.
            scores_b = top1pct(base).detach().cpu().numpy().astype(np.float64)
            scores.extend(scores_b.tolist())
            gates = {
                "mild": soft_gate(scores_b, calib, GATE_MILD),
                "sharp": soft_gate(scores_b, calib, GATE_SHARP),
            }
            for k in gate_buf:
                gate_buf[k].extend(gates[k].tolist())

            # Exact 672 reference and exact stage-7 full guided map.
            x_hi = his[0]
            hi_raw = global_map_chunked(model, x_hi, SCALE_BATCH[672])
            hi = postprocess(hi_raw, gaussian)
            del hi_raw
            fused = 0.5 * base + 0.5 * hi
            guide = guide_from_672_image(x_hi)
            guided = guided_filter(guide, fused)
            per_variant = {
                "guide_baseline": guided,
                "guide_gate_mild": gate_maps(guided, gates["mild"], torch),
                "guide_gate_sharp": gate_maps(guided, gates["sharp"], torch),
                "guide_gate_mild_medsub": gate_maps(
                    median_subtract(guided, torch), gates["mild"], torch),
            }

            for v in VARIANTS:
                buckets[v].append(per_variant[v].cpu())
            del per_variant, base, hi, fused, guide, guided
            del his, img, x_hi

            # Masks come from the exact original 448->392 pipeline.
            gt256 = F.interpolate(gt.float(), size=256, mode="nearest").bool()
            if gt256.shape[1] > 1:
                gt256 = gt256.max(dim=1, keepdim=True)[0]
            gt_buf.append(gt256.cpu())
            labels.extend(label.numpy().tolist())

            if bi == 0 or (bi + 1) % 10 == 0 or (bi + 1) == n_batches:
                log(f"  {item} batch {bi + 1}/{n_batches} "
                    f"images={len(labels)}/{len(dataset)} "
                    f"elapsed={time.time() - started:.1f}s")

    gt_np = torch.cat(gt_buf, dim=0)[:, 0].numpy()
    gt_np = (gt_np > 0.5).astype(np.int64)
    gt_flat = gt_np.ravel()
    labels_np = np.asarray(labels, dtype=np.int64)
    scores_np = np.asarray(scores, dtype=np.float64)
    regions = label_regions(gt_np)

    img_metrics = image_metrics_from_scores(labels_np, scores_np)

    variants = {}
    for v in VARIANTS:
        pr = torch.cat(buckets[v], dim=0)[:, 0].numpy().astype(np.float32)
        buckets[v] = None
        pr_flat = pr.ravel()
        recall, thresholds = fixed_fpr_recall(pr_flat, gt_flat, FPRS)
        anom_auroc, anom_ap = _anomaly_only_metrics(labels_np, gt_np, pr)
        row = {
            "I-AUROC": img_metrics["I-AUROC"],
            "I-AP": img_metrics["I-AP"],
            "I-F1": img_metrics["I-F1"],
            "P-AUROC": float(roc_auc_score(gt_flat, pr_flat)),
            "P-AP": float(average_precision_score(gt_flat, pr_flat)),
            "P-F1": float(f1_score_max(gt_flat, pr_flat)),
            "P-AUPRO": float(compute_pro_pre(gt_np, pr, regions)),
            "P-AUROC-anom": anom_auroc,
            "P-AP-anom": anom_ap,
        }
        variants[v] = {"metrics": row, "recall": recall,
                       "thresholds": thresholds}

    gate_values = {k: np.asarray(v, dtype=np.float64)
                   for k, v in gate_buf.items()}
    gate_stats = {}
    for k, vals in gate_values.items():
        cfg = GATE_MILD if k == "mild" else GATE_SHARP
        gate_stats[k] = {
            "definition": gate_definition(cfg),
            "mean": float(vals.mean()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "frac_below_0.2": float((vals < 0.2).mean()),
            "frac_above_0.8": float((vals > 0.8).mean()),
        }

    return {
        "class": item,
        "fingerprint": FP,
        "n_test": int(len(labels_np)),
        "n_normal": int((labels_np == 0).sum()),
        "n_anomaly": int((labels_np == 1).sum()),
        "n_gt_positive_pixels": int(gt_flat.sum()),
        "n_gt_negative_pixels": int((gt_flat == 0).sum()),
        "scales": [{"name": f"hi{c}", "crop": c, "resize": r,
                    "fov": fov(c, r), "batch": SCALE_BATCH[c]}
                   for c, r in ENABLED],
        "image_metrics": img_metrics,
        "image_scores": {
            "source": "baseline 448->392 global top1% of the postprocessed "
                      "256 map, computed once and reused by every variant; "
                      "never gated or replaced",
            "reused_for_all_variants": True,
            "max_abs_diff_across_variants": 0.0,
            "sha256": hashlib.sha256(
                scores_np.astype(np.float64).tobytes()).hexdigest(),
        },
        "calibration": calib,
        "gate": gate_stats,
        "guide": {
            "radius": GUIDE_RADIUS,
            "epsilon": GUIDE_EPS,
            "baseline_variant": "stage-7 full grayscale guided filter output",
        },
        "variants": variants,
        "seconds": float(time.time() - started),
        "gpu_peak_mib": peak_mib(),
    }


# ---------------------------------------------------------------------------
# aggregation / report
# ---------------------------------------------------------------------------
def aggregate(per_class, item_list, stage1):
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

    image_mean = {k: finite_mean(
        [per_class[it]["image_metrics"][k] for it in item_list])
        for k in ("I-AUROC", "I-AP", "I-F1")}

    reference = "guide_baseline"
    comparisons = {}
    for v in VARIANTS:
        if v != reference:
            comparisons[f"{v} - {reference}"] = (v, reference)

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
                    k: per_class[it]["variants"][a]["recall"].get(
                           k, float("nan"))
                       - per_class[it]["variants"][b]["recall"].get(
                           k, float("nan"))
                    for k in RECALL_KEYS},
            }

    stage1_identity = None
    if stage1 and "base_global" in VARIANTS:
        per_class_diff = {}
        for it in item_list:
            src = stage1.get("per_class", {}).get(it)
            if not src:
                continue
            row = per_class[it]["variants"]["base_global"]["metrics"]
            per_class_diff[it] = {
                k: float(row[k] - float(src[k]))
                for k in METRIC_KEYS if k in src
            }
        key_max = {}
        for k in METRIC_KEYS:
            vals = [abs(per_class_diff[it][k]) for it in per_class_diff
                    if k in per_class_diff[it]]
            key_max[k] = max(vals) if vals else None
        mean_delta = {}
        if "mean" in stage1:
            for k in METRIC_KEYS:
                if k in stage1["mean"]:
                    mean_delta[k] = float(
                        mean["base_global"]["metrics"][k]
                        - float(stage1["mean"][k]))
        stage1_identity = {
            "source": STAGE1_METRICS,
            "per_key_max_abs_per_class_delta": key_max,
            "mean_delta": mean_delta,
            "per_class_max_abs_delta": per_class_diff,
            "note": "Our baseline forward uses effective batch "
                    f"{BASE_BATCH}; stage 1 used batch 16. Tiny numerical "
                    "differences are expected.",
        }

    return {
        "mean": mean,
        "image_mean": image_mean,
        "deltas": deltas,
        "per_class_deltas": per_class_deltas,
        "stage1_identity": stage1_identity,
    }


def pct(x):
    return f"{100.0 * x:.2f}"


def gate_definition(cfg):
    if cfg["mid_key"] == "q95":
        return (f"g = {cfg['floor']:.2f} + {1 - cfg['floor']:.2f} * "
                "sigmoid((s - q95) / max(q95 - median, "
                f"{cfg['min_denom']:g}) * {cfg['width_scale']:g})")
    return (f"g = {cfg['floor']:.2f} + {1 - cfg['floor']:.2f} * "
            "sigmoid((s - q99) / max(q99 - median, "
            f"{cfg['min_denom']:g}) * {cfg['width_scale']:g})")


def variant_definition(v):
    if v == "guide_baseline":
        return ("exact stage-7 full grayscale guided-filter output on "
                f"0.5*base_global + 0.5*hi672 (radius {GUIDE_RADIUS}, "
                f"eps {GUIDE_EPS:g}); no gate")
    if v == "guide_gate_mild":
        return "guide_baseline * mild image gate; " + gate_definition(GATE_MILD)
    if v == "guide_gate_sharp":
        return "guide_baseline * sharp image gate; " + gate_definition(
            GATE_SHARP)
    if v == "guide_gate_mild_medsub":
        return ("mild gate * (guide_baseline - per-image median); "
                + gate_definition(GATE_MILD))
    return ""


def write_markdown(report, path):
    lines = []
    lines.append("# Dinomaly image-conditioned calibration (stage 10)")
    lines.append("")
    lines.append(f"Slug: `{SLUG}`  ")
    lines.append(f"Generated: {report['experiment']['generated']}  ")
    lines.append(f"Checkpoint: `{report['checkpoints']['base']['path']}` "
                 f"(sha256 `{report['checkpoints']['base']['sha256'][:16]}...`)  ")
    lines.append(f"Test images: {report['config']['test_images_evaluated']} "
                 f"across {len(report['config']['eval_classes'])} classes  ")
    lines.append(f"Enabled high-res scales: "
                 f"{report['config']['enabled_scales']}  ")
    lines.append("")
    lines.append("## What is compared")
    lines.append("")
    lines.append("All branches use the same frozen 10k checkpoint. The "
                 "baseline is exactly stage 3's global branch. High-res "
                 "branches independently resize the native PIL image to "
                 "640/768, centre-crop 560/672 (same 7/8 FOV), and run the "
                 "full image through ViTill; DINOv2 interpolates its "
                 "position embeddings and ViTill derives the square token "
                 "side from the actual token count.")
    lines.append("")
    lines.append("| variant | definition |")
    lines.append("|---|---|")
    for v in VARIANTS:
        lines.append(f"| `{v}` | {variant_definition(v)} |")
    lines.append("")
    lines.append("Every variant is evaluated on the same 256 grid against "
                 "masks produced by the exact original 448->392 ground-truth "
                 "transform. The image score is always the original baseline "
                 "global top-1% score and is reused by every variant.")
    lines.append("")
    n_eval = len(report["config"]["eval_classes"])
    title = "15-class means" if n_eval == 15 else f"{n_eval}-class means"
    lines.append(f"## {title} (%)")
    lines.append("")
    lines.append("| variant | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | "
                 "P-F1 | P-AUPRO |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        m = report["results"]["mean"][v]["metrics"]
        lines.append("| `" + v + "` | " + " | ".join(
            pct(m[k]) for k in METRIC_KEYS) + " |")
    lines.append("")
    lines.append("Image metrics are identical by construction. Only P-* "
                 "columns and recall differ.")
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
    lines.append("Thresholds are computed from each variant's own per-class "
                 "test negative pixels, matching stage 3's corrected recall; "
                 "they are metric thresholds, not model fitting.")
    lines.append("")
    lines.append("## Mean pixel deltas (percentage points)")
    lines.append("")
    lines.append("| comparison | P-AUROC | P-AP | P-F1 | P-AUPRO |")
    lines.append("|---|---|---|---|---|")
    for name in report["deltas"]:
        d = report["deltas"][name]["metrics"]
        lines.append(f"| {name} | " + " | ".join(
            f"{100 * d[k]:+.2f}" for k in
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
    lines.append(f"- Image score is the original baseline global top-1% and "
                 f"is reused by all variants "
                 f"(max |delta| across variants = "
                 f"{ident['image_score_max_abs_diff_across_variants']:.3e}).")
    lines.append("- No training was performed; the unchanged 10k checkpoint "
                 "was reused.")
    lines.append("- The high-res scales were fixed a priori and no model or "
                 "calibration parameter was fitted. Test labels enter only "
                 "through the standard metric definitions (AUROC/AP/F1/PRO "
                 "and per-variant fixed-FPR recall), exactly as stage 3.")
    lines.append("- High-res maps are computed on the full high-res image, "
                 "then bilinearly reduced to 256 for metrics; raw high-res "
                 "maps are not saved.")
    s1 = ident.get("stage1_baseline_comparison")
    if s1:
        md = s1.get("mean_delta", {})
        lines.append("- Recomputed baseline vs stage 1 reported metrics: "
                 + ", ".join(f"{k} {100 * md[k]:+.4f} pp"
                             for k in md))
        km = s1.get("per_key_max_abs_per_class_delta", {})
        lines.append("- Max per-class |delta| vs stage 1: "
                 + ", ".join(f"{k} {100 * km[k]:.4f} pp"
                             for k in km if km.get(k) is not None))
    else:
        lines.append("- Stage 1 metrics unavailable; baseline identity check "
                     "skipped.")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    rt = report["runtime"]
    lines.append(f"- Device: `{rt['device']}`; torch `{rt['torch']}`")
    lines.append(f"- Classes: {rt['classes_evaluated']}; test images: "
                 f"{rt['test_images_evaluated']}")
    lines.append(f"- Forward passes per test image: "
                 f"{rt['forwards_per_test_image']} "
                 f"(1 baseline + {len(ENABLED)} high-res)")
    lines.append(f"- Effective batches: baseline {rt['base_batch']}, "
                 f"scales {rt['scale_batches']}")
    lines.append(f"- Probe: {json.dumps(rt['probe'], sort_keys=True)}")
    lines.append(f"- Evaluation seconds (sum of per-class): "
                 f"{rt['seconds_total']:.1f}")
    lines.append("")
    lines.append("Full per-class metrics, thresholds and deltas: "
                 "`context_res_report.json`.")
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

    for crop, resize in ((392, 448), (672, 768)):
        ok(abs(fov(crop, resize) - 0.875) < 1e-12,
           f"7/8 FOV for {crop}/{resize}")
    ok(FINAL_VARIANTS == ("guide_baseline", "guide_gate_mild",
                          "guide_gate_sharp", "guide_gate_mild_medsub"),
       "exactly four stage-10 variants")

    import torch as _torch
    import torch.nn.functional as _F
    globals()["torch"] = _torch
    globals()["F"] = _F

    # Flat guide + flat heatmap: the guided filter must be identity.
    g_flat = _torch.full((1, 1, 32, 32), 0.5)
    p_flat = _torch.full((1, 1, 32, 32), 0.3)
    q_flat = guided_filter(g_flat, p_flat)
    ok(bool(_torch.isfinite(q_flat).all()), "flat guided finite")
    ok(float((q_flat - p_flat).abs().max()) < 1e-6,
       "flat guide guided-filter identity")

    # Synthetic vertical edge: guide and heatmap step together.
    g_edge = _torch.empty((1, 1, 32, 32))
    g_edge[:, :, :, :16] = 0.2
    g_edge[:, :, :, 16:] = 0.8
    p_edge = _torch.empty_like(g_edge)
    p_edge[:, :, :, :16] = 0.1
    p_edge[:, :, :, 16:] = 0.7
    g_edge_out = guided_filter(g_edge, p_edge)
    ok(bool(_torch.isfinite(g_edge_out).all()), "edge guided finite")
    guided_step = float(g_edge_out[:, :, :, 16:].mean()
                        - g_edge_out[:, :, :, :16].mean())
    ok(guided_step > 0.4, f"guided preserves edge step ({guided_step:.3f})")

    # Calibration quantiles and gates.
    cal_scores = np.linspace(0.0, 1.0, 48)
    calib = calibration_quantiles(cal_scores)
    ok(calib["q95"] > calib["median"] and calib["q99"] >= calib["q95"],
       "calibration quantiles ordered")
    ok(calib["denom95"] >= CALIB_MIN_DENOM
       and calib["denom99"] >= CALIB_MIN_DENOM,
       "calibration denominators clamped")
    test_scores = np.array([-10.0, calib["q95"], calib["q99"], 10.0])
    gm = soft_gate(test_scores, calib, GATE_MILD)
    gs = soft_gate(test_scores, calib, GATE_SHARP)
    ok(bool(np.isfinite(gm).all()) and bool(np.isfinite(gs).all()),
       "gates finite")
    ok(float(gm.min()) >= GATE_MILD["floor"] - 1e-12
       and float(gm.max()) <= 1.0 + 1e-12,
       "mild gate in [floor, 1]")
    ok(float(gs.min()) >= GATE_SHARP["floor"] - 1e-12
       and float(gs.max()) <= 1.0 + 1e-12,
       "sharp gate in [floor, 1]")
    ok(np.all(np.diff(gm) > 0), "mild gate monotone")
    ok(abs(float(gm[1]) - (GATE_MILD["floor"]
                           + (1 - GATE_MILD["floor"]) * 0.5)) < 1e-7,
       "mild gate is 0.5 at q95")
    ok(abs(float(gs[2]) - (GATE_SHARP["floor"]
                           + (1 - GATE_SHARP["floor"]) * 0.5)) < 1e-7,
       "sharp gate is 0.5 at q99")
    flat_cal = calibration_quantiles(np.full(48, 0.25))
    flat_gate = soft_gate(np.array([0.24, 0.25, 0.26]), flat_cal, GATE_MILD)
    ok(bool(np.isfinite(flat_gate).all()), "flat calibration finite")
    ok(abs(float(flat_gate[1]) - (GATE_MILD["floor"] + 0.5
                                  * (1 - GATE_MILD["floor"]))) < 1e-12,
       "flat calibration midpoint gate")

    # Gating is a per-image non-negative scale: within-image order preserved.
    maps = _torch.rand(3, 1, 16, 16)
    gate = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    gated = gate_maps(maps, gate, _torch)
    for i in range(3):
        ratio = gated[i] / maps[i]
        ok(float(ratio.min()) > 0 and float(ratio.max()) < 1.0001,
           f"gated map {i} scaled")
        ok(abs(float(ratio.mean()) - gate[i]) < 1e-5,
           f"gated map {i} scale equals gate")
        order_a = _torch.argsort(maps[i].flatten())
        order_b = _torch.argsort(gated[i].flatten())
        ok(bool(_torch.equal(order_a, order_b)),
           f"gated map {i} preserves within-image rank")
    centered = median_subtract(maps, _torch)
    ok(float(centered.flatten(1).median(dim=1).values.abs().max()) < 1e-6,
       "median subtraction centers each image")
    order_a = _torch.argsort(maps[0].flatten())
    order_b = _torch.argsort(centered[0].flatten())
    ok(bool(_torch.equal(order_a, order_b)),
       "median subtraction preserves within-image rank")

    n_neg, n_pos = 200, 50
    gt = np.concatenate([np.zeros(n_neg, dtype=np.int64),
                         np.ones(n_pos, dtype=np.int64)])
    neg_a = rng.normal(0.0, 1.0, n_neg)
    pos_a = rng.normal(2.0, 1.0, n_pos)
    a = np.concatenate([neg_a, pos_a])
    b_scores = 10.0 * a + 5.0
    d_scores = -a
    ra, ta = fixed_fpr_recall(a, gt)
    rb, tb = fixed_fpr_recall(b_scores, gt)
    rd, _td = fixed_fpr_recall(d_scores, gt)
    ok(abs(ra["R@FPR0.01"] - rb["R@FPR0.01"]) < 1e-12,
       "affine transform preserves recall")
    ok(abs(ta["T@FPR0.01"] - tb["T@FPR0.01"]) > 1e-6,
       "thresholds differ per variant")
    manual = float((pos_a >= np.quantile(neg_a, 0.99)).mean())
    ok(abs(ra["R@FPR0.01"] - manual) < 1e-12,
       "recall uses this variant's negatives")
    ok(abs(rd["R@FPR0.01"] - ra["R@FPR0.01"]) > 0.1,
       "reversed ranks change recall")

    # Aggregation must never KeyError on any registered variant/key.
    saved = globals()["VARIANTS"]
    globals()["VARIANTS"] = FINAL_VARIANTS
    try:
        fake = {}
        for it, offset in (("alpha", 0.0), ("beta", 0.01)):
            fake[it] = {
                "image_metrics": {k: 0.9 + offset
                                  for k in ("I-AUROC", "I-AP", "I-F1")},
                "variants": {},
            }
            for vi, v in enumerate(FINAL_VARIANTS):
                row = {k: 0.5 + 0.01 * vi + offset for k in METRIC_KEYS}
                rec = {k: 0.2 + 0.01 * vi + offset for k in RECALL_KEYS}
                fake[it]["variants"][v] = {"metrics": row, "recall": rec}
        agg = aggregate(fake, ["alpha", "beta"], None)
        ok(set(agg["mean"]) == set(FINAL_VARIANTS),
           "aggregate covers every variant")
        ok(all(set(agg["mean"][v]["metrics"]) == set(METRIC_KEYS)
               for v in FINAL_VARIANTS),
           "aggregate covers every metric key")
        ok(f"guide_gate_mild - guide_baseline" in agg["deltas"],
           "aggregate deltas use guide reference")
        ok(len(agg["per_class_deltas"]) == 3,
           "aggregate per-class deltas complete")
    finally:
        globals()["VARIANTS"] = saved

    print(f"SELF-TEST OK ({checks} checks)")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    global LOG_PATH, FP, EVAL_BATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--eval-batch", type=int, default=EVAL_BATCH)
    ap.add_argument("--skip-672", action="store_true")
    ap.add_argument("--max-672-forward-min", type=float, default=45.0)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--limit-classes", type=int, default=0,
                    help="smoke only: evaluate first N classes")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()

    EVAL_BATCH = int(args.eval_batch)
    if EVAL_BATCH < 1:
        raise ValueError("--eval-batch must be >= 1")
    os.makedirs(OUT, exist_ok=True)
    LOG_PATH = os.path.join(OUT, "stage10.log")
    log(f"=== {SLUG} start ===")
    log(f"args={vars(args)}")

    run_self_test()
    log("self-test passed")

    prepare_environment()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"device={device} torch={torch.__version__}")
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    find_checkpoints()
    base_sha = sha256_file(BASE_CKPT)
    log(f"base sha256={base_sha}")
    # Stage 10's reference is the exact recomputed stage-7 guided map, not
    # stage 1.  Stage-1 metrics are not needed for this image-level test.
    stage1 = None

    item_list = sorted(
        d for d in os.listdir(DATA)
        if os.path.isdir(os.path.join(DATA, d)))
    if len(item_list) != 15:
        raise RuntimeError(f"expected 15 MVTec classes, got "
                           f"{len(item_list)}: {item_list}")
    log(f"classes={len(item_list)}")

    # Model + synthetic probe decide effective batches and whether 672 is
    # reasonably fast.  No test image or label is involved.
    model = build_model(device)
    st_base = load_ckpt(model, BASE_CKPT, device)
    model.eval()
    log(f"checkpoint iter={st_base.get('iter')} kind={st_base.get('kind')}")
    probe = probe_scales(model, device, EVAL_BATCH, not args.skip_672,
                         args.max_672_forward_min)
    apply_probe(probe)
    # probe has both "base" and integer crop keys; normalize for JSON.
    probe_json = {str(k): v for k, v in probe.items()}
    log(f"probe={json.dumps(probe_json, sort_keys=True)}")
    log(f"enabled scales={ENABLED} base_batch={BASE_BATCH} "
        f"scale_batches={SCALE_BATCH} variants={VARIANTS}")

    # Standard branch uses the exact original transform; each high-res branch
    # reopens the native PIL image and resizes it independently.
    base_transform, gt_transform = get_data_transforms(BASE_RESIZE, BASE_CROP)
    scale_transforms = [
        (crop, get_data_transforms(resize, crop)[0])
        for crop, resize in ENABLED
    ]
    test_data_list = {
        item: MultiScaleTestDataset(
            root=os.path.join(DATA, item),
            base_transform=base_transform,
            gt_transform=gt_transform,
            scale_transforms=scale_transforms)
        for item in item_list
    }
    counts = {it: len(ds) for it, ds in test_data_list.items()}
    total_test = int(sum(counts.values()))
    log(f"test images={total_test} per-class={counts}")
    if total_test != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"expected {EXPECTED_TEST_IMAGES} official test images, "
            f"got {total_test}: {counts}")

    eval_items = list(item_list)
    if args.limit_classes and args.limit_classes > 0:
        eval_items = eval_items[:args.limit_classes]
    test_images_evaluated = int(sum(counts[it] for it in eval_items))
    partial = len(eval_items) != len(item_list)

    # Image-level calibration from deterministic train/good images only.
    # This never reads a test image, mask, or label.
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    calibration = {}
    for item in eval_items:
        calib_ds = TrainGoodCalibrationDataset(
            os.path.join(DATA, item), base_transform, CALIB_IMAGES)
        calibration[item] = calibrate_class(
            item, calib_ds, model, gaussian, device)
    log(f"calibration complete for {len(calibration)} classes "
        f"({CALIB_IMAGES} train/good images per class)")

    enabled_scales = [{"name": f"hi{c}", "crop": c, "resize": r,
                       "fov": fov(c, r), "batch": SCALE_BATCH[c]}
                      for c, r in ENABLED]
    config = {
        "slug": SLUG,
        "stage": 10,
        "script": os.path.basename(os.path.abspath(__file__)),
        "code_sha256": sha256_file(os.path.abspath(__file__)),
        "baseline": {"resize": BASE_RESIZE, "crop": BASE_CROP,
                     "fov": fov(BASE_CROP, BASE_RESIZE)},
        "enabled_scales": enabled_scales,
        "variants": list(VARIANTS),
        "fusion": {
            "reference": "0.5 * base_global + 0.5 * hi672",
            "pixel_map": "exact stage-7 full grayscale guided filter on the "
                         "fusion; guide_baseline is ungated",
        },
        "guide": {
            "source": "original 672 crop de-normalized to RGB 0..1, "
                      "bilinear resized to 256, luminance grayscale",
            "radius": GUIDE_RADIUS,
            "epsilon": GUIDE_EPS,
            "box_pooling": "avg_pool2d(count_include_pad=False)",
        },
        "calibration": {
            "source": "train/good only; original 448->392 top1% score",
            "images_per_class": CALIB_IMAGES,
            "selection": "sorted paths, evenly spaced",
            "classes": calibration,
            "test_labels_used": False,
        },
        "gate": {
            "mild": dict(GATE_MILD, definition=gate_definition(GATE_MILD)),
            "sharp": dict(GATE_SHARP, definition=gate_definition(GATE_SHARP)),
            "application": "per-image multiplication of the pixel map; "
                           "within-image ranking unchanged",
            "median_subtraction_variant": "mild gate * (map - per-image "
                                          "median); cheap, post-hoc",
        },
        "metrics": {
            "keys": list(METRIC_KEYS),
            "recall": list(RECALL_KEYS),
            "evaluation_grid": 256,
            "mask_source": "exact original 448->392 gt transform, nearest "
                           "to 256",
            "image_score": "original baseline global top1% of the "
                           "postprocessed 256 map, reused by all variants; "
                           "never gated or replaced",
            "diagnostic_anom_only": "P-*-anom restrict pixel metrics to "
                                    "anomaly test images only",
            "pro": "stage 3 compute_pro_pre, 200 thresholds, FPR < 0.3",
        },
        "no_training": True,
        "test_labels_used_for_selection_or_fit": False,
        "classes_total": len(item_list),
        "eval_classes": eval_items,
        "partial": partial,
        "test_images_total": total_test,
        "test_images_evaluated": test_images_evaluated,
        "test_counts": counts,
        "eval_batch": EVAL_BATCH,
        "base_batch": BASE_BATCH,
        "scale_batches": {str(k): v for k, v in SCALE_BATCH.items()},
        "probe": probe_json,
        "base_checkpoint": BASE_CKPT,
        "base_sha256": base_sha,
        "stage1_metrics": STAGE1_METRICS,
    }
    # Fingerprint only the semantic configuration.  Probe timings and
    # calibration measurements are recomputed each run and must not
    # invalidate completed classes; the calibration *protocol* is fixed
    # above and recorded in config.json.
    fp_cfg = {k: v for k, v in config.items()
              if k not in ("probe", "calibration")}
    FP = config_fingerprint(fp_cfg)
    log(f"config fingerprint={FP}")
    write_json_atomic(os.path.join(OUT, "config.json"), config)

    per_class_dir = os.path.join(OUT, "per_class")
    os.makedirs(per_class_dir, exist_ok=True)
    progress_path = os.path.join(OUT, "progress.json")
    status_path = os.path.join(OUT, "status.json")

    def set_status(status, **extra):
        write_json_atomic(status_path, {
            "slug": SLUG, "fingerprint": FP, "status": status,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"), **extra})

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
            "slug": SLUG, "fingerprint": FP,
            "classes_total": len(eval_items), "classes_done": done,
            "current": current, "latest": latest,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S")})

    set_status("running", classes_done=done,
               classes_total=len(eval_items), current=None)
    update_progress()

    for item in eval_items:
        if item in set(done):
            log(f"{item}: resume skip")
            continue
        set_status("running", classes_done=done,
                   classes_total=len(eval_items), current=item)
        update_progress(current=item)
        stage1_item = None
        if stage1:
            stage1_item = stage1.get("per_class", {}).get(item)
        log(f"{item}: evaluating {counts[item]} test images "
            f"(base_batch={BASE_BATCH}, scales={ENABLED})")
        result = evaluate_class(item, test_data_list[item], model,
                                gaussian, device, stage1_item,
                                calibration[item])
        write_json_atomic(per_class_path(item), result)
        done.append(item)
        latest = {
            "class": item,
            "seconds": result["seconds"],
            "P-AP": {v: result["variants"][v]["metrics"]["P-AP"]
                     for v in VARIANTS},
            "P-AUPRO": {v: result["variants"][v]["metrics"]["P-AUPRO"]
                        for v in VARIANTS},
            "P-AP-anom": {v: result["variants"][v]["metrics"]["P-AP-anom"]
                          for v in VARIANTS},
        }
        update_progress(current=None, latest=latest)
        set_status("running", classes_done=done,
                   classes_total=len(eval_items), current=None,
                   latest=latest)
        log(f"{item}: done in {result['seconds']:.1f}s "
            f"peak={result['gpu_peak_mib']:.0f}MiB " + " ".join(
                f"{v}:AP={result['variants'][v]['metrics']['P-AP']:.4f}"
                f"/PRO={result['variants'][v]['metrics']['P-AUPRO']:.4f}"
                for v in VARIANTS))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_class = {}
    for item in eval_items:
        d = load_per_class(item)
        if d is None:
            raise RuntimeError(f"missing per-class result for {item}")
        per_class[item] = d

    agg = aggregate(per_class, eval_items, stage1)

    calibration_mean = {
        k: finite_mean([per_class[it]["calibration"].get(k)
                        for it in eval_items])
        for k in ("median", "q95", "q99", "denom95", "denom99")}
    gate_mean = {}
    for gk in ("mild", "sharp"):
        gate_mean[gk] = {
            k: finite_mean([per_class[it]["gate"][gk].get(k)
                            for it in eval_items])
            for k in ("mean", "min", "max",
                      "frac_below_0.2", "frac_above_0.8")}

    report = {
        "experiment": {
            "slug": SLUG,
            "stage": 10,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "partial": partial,
            "hypothesis": "the original 392 image score is excellent at "
                          "image level, but normal-image heatmaps still "
                          "compete in pixel ranking; soft per-image "
                          "confidence gating may suppress those false "
                          "positives without hurting localization",
        },
        "config": config,
        "checkpoints": {
            "base": {
                "path": BASE_CKPT,
                "sha256": base_sha,
                "iter": (int(st_base["iter"])
                         if st_base.get("iter") is not None else None),
                "kind": st_base.get("kind"),
                "reused_unchanged": True,
            },
        },
        "identity": {
            "image_scores_identical_across_variants": True,
            "image_score_max_abs_diff_across_variants": 0.0,
            "image_scores_source": "one original 392 top1% score per test "
                                   "image, preserved unchanged for image "
                                   "metrics and gating",
            "scale_selection": "fixed a priori: only 392 base and 672 "
                               "(768->672); no 560 forward; no test labels "
                               "used",
            "no_training": True,
            "checkpoint_reused_unchanged": True,
            "high_res_geometry": "672 branch independently resizes the native "
                                 "PIL image; same 7/8 FOV as the 448->392 "
                                 "baseline",
            "position_interpolation": "DINOv2 interpolate_pos_encoding; "
                                      "ViTill side derived from token count",
            "guide_parameters_fixed_a_priori": True,
            "reference": "guide_baseline = exact stage-7 full guided filter "
                         "on 0.5*base_global + 0.5*hi672",
            "gate_input_scope": "original 392 image score only; no pixel "
                                "map, mask, or test label participates",
            "calibration_scope": "deterministic train/good images only",
            "stage1_baseline_comparison": None,
        },
        "results": {
            "per_class": per_class,
            "mean": agg["mean"],
            "calibration_mean": calibration_mean,
            "gate_mean": gate_mean,
        },
        "image_metrics": {
            "mean": agg["image_mean"],
            "note": "identical for every variant; from the original 392 "
                    "baseline global map and never gated",
        },
        "diagnostic": {
            "P-AUROC-anom": "pixel AUROC restricted to anomaly test images",
            "P-AP-anom": "pixel AP restricted to anomaly test images",
            "interpretation": "If overall P-AP rises while anomaly-only "
                              "P-AP is flat, the gain is normal-image false-"
                              "positive suppression; if anomaly-only P-AP "
                              "also rises, localization improved",
        },
        "deltas": agg["deltas"],
        "per_class_deltas": agg["per_class_deltas"],
        "runtime": {
            "device": device,
            "torch": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "classes_evaluated": len(eval_items),
            "test_images_evaluated": test_images_evaluated,
            "eval_batch": EVAL_BATCH,
            "base_batch": BASE_BATCH,
            "scale_batches": {str(k): v for k, v in SCALE_BATCH.items()},
            "enabled_scales": list(ENABLED),
            "forwards_per_test_image": 1 + len(ENABLED),
            "probe": probe_json,
            "seconds_total": float(sum(per_class[it]["seconds"]
                                       for it in eval_items)),
            "gpu_peak_mib_max": max(
                (per_class[it]["gpu_peak_mib"] for it in eval_items),
                default=0.0),
        },
        "files": {
            "per_class_dir": per_class_dir,
            "progress": progress_path,
            "status": status_path,
        },
    }

    report_json = os.path.join(OUT, "image_conditioned_report.json")
    write_json_atomic(report_json, report)
    set_status("complete", classes_done=done,
               classes_total=len(eval_items),
               report=report_json, markdown=None)
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
