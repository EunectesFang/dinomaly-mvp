"""Stage 7: edge-guided boundary refinement of the 672 token map.

Frozen 10k checkpoint, no training, no test-label fitting.

Hypothesis: the 48x48 (672 input) token reconstruction map is coarse relative
to object boundaries, so its bilinear 256 map spreads anomaly mass across image
edges.  A grayscale guided filter, using the de-normalized 672 crop as the
edge guide, may move local mass toward real edges.

Protocol is intentionally identical to stage 4 for the two forwards used:
    original image -> Resize(448) -> CenterCrop(392) -> Normalize -> 392 map
    original image -> Resize(768) -> CenterCrop(672) -> Normalize -> 672 map
    both maps -> bilinear 256 -> gaussian(5,4)
    exact reference = 0.5 * base_global + 0.5 * hi672
The original baseline top-1% image score is reused unchanged by every variant.
Masks are the exact original 448->392 ground-truth transform nearest-resized
to 256.  All 15 classes / 1725 official test images are evaluated.

Four pixel variants on the same 256 grid:
    base_fuse672       exact 0.5 * base_global + 0.5 * hi672
    box2_fuse672       two-pass unguided box smoothing of base_fuse672
    guide_fuse672      stable grayscale guided-filter output (radius 7, eps .0025)
    detail_fuse672     base_fuse672 + 0.5 * clip(guided - box2, +-cap)
The detail variant adds only the edge-guided high-pass residual, preserving the
exact base localization.  Its local detail is capped per image at 10% of the
heatmap 1st-99th percentile range to limit texture transfer.

No test labels are used to select, tune, or fit anything.  The 392/672 scales,
guided-filter parameters, detail strength and cap fraction are fixed a priori.
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
import traceback

import numpy as np

OUT = "/kaggle/working"
SLUG = "eunectesqaq/dinomaly-guided-boundary-v7"
BASE_RESIZE = 448
BASE_CROP = 392
SCALE_SPECS = ((672, 768),)              # only the 48x48 token branch
EXPECTED_TEST_IMAGES = 1725
EVAL_BATCH = 4
FPRS = (0.001, 0.01, 0.05)
METRIC_KEYS = ("I-AUROC", "I-AP", "I-F1",
               "P-AUROC", "P-AP", "P-F1", "P-AUPRO")
RECALL_KEYS = tuple(f"R@FPR{f:g}" for f in FPRS)
GUIDE_RADIUS = 7
GUIDE_EPS = 0.0025
DETAIL_STRENGTH = 0.5
DETAIL_CAP_FRAC = 0.10
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FINAL_VARIANTS = ("base_fuse672", "box2_fuse672",
                  "guide_fuse672", "detail_fuse672")

LOG_PATH = None
FP = None
CODE = None
DATA = None
BASE_CKPT = None
STAGE1_METRICS = None
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
        raise RuntimeError(f"stage 7 requires exactly the 672 branch, "
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


def evaluate_class(item, dataset, model, gaussian, device, stage1_item):
    loader = DataLoader(dataset, batch_size=EVAL_BATCH, shuffle=False,
                        num_workers=4,
                        pin_memory=torch.cuda.is_available())
    buckets = {v: [] for v in VARIANTS}
    gt_buf, labels, scores = [], [], []
    guide_clip_fracs = []
    n_batches = len(loader)
    started = time.time()
    reset_peak()

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            img, gt, label, _paths, his = batch
            img = img.to(device, non_blocking=True)
            his = [h.to(device, non_blocking=True) for h in his]

            # Baseline global map: exact stage 3 logic.
            base_raw = global_map_chunked(model, img, BASE_BATCH)
            base = postprocess(base_raw, gaussian)
            del base_raw

            # Exact 672 reference used by every stage-7 refinement.
            x_hi = his[0]
            hi_raw = global_map_chunked(model, x_hi, SCALE_BATCH[672])
            hi = postprocess(hi_raw, gaussian)
            del hi_raw
            fused = 0.5 * base + 0.5 * hi
            guide = guide_from_672_image(x_hi)
            box2, guided, detail, _cap, clip_frac = refine_heatmap(
                fused, guide)
            per_variant = {
                "base_fuse672": fused,
                "box2_fuse672": box2,
                "guide_fuse672": guided,
                "detail_fuse672": detail,
            }
            guide_clip_fracs.append(float(clip_frac.detach().cpu()))

            # Original global image score, computed once and reused.
            scores.extend(top1pct(base).cpu().numpy().tolist())

            for v in VARIANTS:
                buckets[v].append(per_variant[v].cpu())
            del per_variant, base, hi, fused, guide, box2, guided, detail
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
        row = {
            "I-AUROC": img_metrics["I-AUROC"],
            "I-AP": img_metrics["I-AP"],
            "I-F1": img_metrics["I-F1"],
            "P-AUROC": float(roc_auc_score(gt_flat, pr_flat)),
            "P-AP": float(average_precision_score(gt_flat, pr_flat)),
            "P-F1": float(f1_score_max(gt_flat, pr_flat)),
            "P-AUPRO": float(compute_pro_pre(gt_np, pr, regions)),
        }
        variants[v] = {"metrics": row, "recall": recall,
                       "thresholds": thresholds}

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
                      "256 map, computed once and reused by every variant",
            "reused_for_all_variants": True,
            "max_abs_diff_across_variants": 0.0,
            "sha256": hashlib.sha256(
                scores_np.astype(np.float64).tobytes()).hexdigest(),
        },
        "guide": {
            "radius": GUIDE_RADIUS,
            "epsilon": GUIDE_EPS,
            "detail_strength": DETAIL_STRENGTH,
            "detail_cap_fraction": DETAIL_CAP_FRAC,
            "detail_clipped_fraction": (
                float(np.mean(guide_clip_fracs)) if guide_clip_fracs
                else None),
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

    reference = "base_fuse672"
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


def variant_definition(v):
    if v == "base_fuse672":
        return "exact 0.5 * base_global + 0.5 * hi672"
    if v == "box2_fuse672":
        return "two-pass unguided box smoothing control on base_fuse672"
    if v == "guide_fuse672":
        return ("full grayscale guided-filter output on base_fuse672 "
                f"(radius {GUIDE_RADIUS}, eps {GUIDE_EPS:g})")
    if v == "detail_fuse672":
        return (f"base_fuse672 + {DETAIL_STRENGTH:g} * clipped("
                "guided - box2), cap 10% of 1st-99th percentile range")
    return ""


def write_markdown(report, path):
    lines = []
    lines.append("# Dinomaly context resolution v4 (stage 4)")
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
    ok(FINAL_VARIANTS == ("base_fuse672", "box2_fuse672",
                          "guide_fuse672", "detail_fuse672"),
       "exactly four stage-7 variants")

    import torch as _torch
    import torch.nn.functional as _F
    globals()["torch"] = _torch
    globals()["F"] = _F

    # Flat guide + flat heatmap: guided and box outputs must be identity.
    g_flat = _torch.full((1, 1, 32, 32), 0.5)
    p_flat = _torch.full((1, 1, 32, 32), 0.3)
    q_flat = guided_filter(g_flat, p_flat)
    b_flat, gfull_flat, d_flat, cap_flat, _ = refine_heatmap(
        p_flat, g_flat)
    ok(bool(_torch.isfinite(q_flat).all()), "flat guided finite")
    ok(float((q_flat - p_flat).abs().max()) < 1e-6,
       "flat guide guided-filter identity")
    ok(float((b_flat - p_flat).abs().max()) < 1e-6,
       "flat guide box control identity")
    ok(float((d_flat - p_flat).abs().max()) < 1e-6,
       "flat guide detail identity")
    ok(float(cap_flat.min()) > 0, "flat cap positive")

    # Synthetic vertical edge: guide and heatmap step together.  The guided
    # filter should preserve the step better than unguided box smoothing.
    g_edge = _torch.empty((1, 1, 32, 32))
    g_edge[:, :, :, :16] = 0.2
    g_edge[:, :, :, 16:] = 0.8
    p_edge = _torch.empty_like(g_edge)
    p_edge[:, :, :, :16] = 0.1
    p_edge[:, :, :, 16:] = 0.7
    b_edge, g_edge_out, d_edge, cap_edge, _ = refine_heatmap(
        p_edge, g_edge)
    ok(bool(_torch.isfinite(g_edge_out).all()), "edge guided finite")
    guided_step = float(g_edge_out[:, :, :, 16:].mean()
                        - g_edge_out[:, :, :, :16].mean())
    box_step = float(b_edge[:, :, :, 16:].mean()
                     - b_edge[:, :, :, :16].mean())
    ok(guided_step > box_step + 0.1,
       f"guided preserves edge step ({guided_step:.3f} vs {box_step:.3f})")
    ok(float((d_edge - p_edge).abs().max())
       <= DETAIL_STRENGTH * float(cap_edge.max()) + 1e-6,
       "detail residual capped")

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
    LOG_PATH = os.path.join(OUT, "stage7.log")
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
    # Stage 7's reference is the exact recomputed base_fuse672, not stage 1.
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

    enabled_scales = [{"name": f"hi{c}", "crop": c, "resize": r,
                       "fov": fov(c, r), "batch": SCALE_BATCH[c]}
                      for c, r in ENABLED]
    config = {
        "slug": SLUG,
        "stage": 7,
        "script": os.path.basename(os.path.abspath(__file__)),
        "code_sha256": sha256_file(os.path.abspath(__file__)),
        "baseline": {"resize": BASE_RESIZE, "crop": BASE_CROP,
                     "fov": fov(BASE_CROP, BASE_RESIZE)},
        "enabled_scales": enabled_scales,
        "variants": list(VARIANTS),
        "fusion": {
            "reference": "0.5 * base_global + 0.5 * hi672",
            "box_control": "two-pass avg_pool2d box smoothing",
            "guided": "grayscale local-linear guided filter",
            "detail": "base_fuse672 + strength * clipped(guided - box2)",
        },
        "guide": {
            "source": "original 672 crop de-normalized to RGB 0..1, "
                      "bilinear resized to 256, luminance grayscale",
            "radius": GUIDE_RADIUS,
            "epsilon": GUIDE_EPS,
            "detail_strength": DETAIL_STRENGTH,
            "detail_cap_fraction": DETAIL_CAP_FRAC,
            "box_pooling": "avg_pool2d(count_include_pad=False)",
        },
        "metrics": {
            "keys": list(METRIC_KEYS),
            "recall": list(RECALL_KEYS),
            "evaluation_grid": 256,
            "mask_source": "exact original 448->392 gt transform, nearest "
                           "to 256",
            "image_score": "original baseline global top1% of the "
                           "postprocessed 256 map, reused by all variants",
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
    # Fingerprint only the semantic configuration.  Probe timings and peak
    # memory vary run to run and must not invalidate completed classes.
    fp_cfg = {k: v for k, v in config.items() if k != "probe"}
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

    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
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
                                gaussian, device, stage1_item)
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
    report = {
        "experiment": {
            "slug": SLUG,
            "stage": 7,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "partial": partial,
            "hypothesis": "coarse 48x48 token reconstruction spreads across "
                          "image edges; edge-guided filtering may sharpen "
                          "boundaries",
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
            "image_scores_source": "one baseline-global score per test image, "
                                   "reused by every variant",
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
            "reference": "base_fuse672 = 0.5 * base_global + 0.5 * hi672",
            "stage1_baseline_comparison": None,
        },
        "results": {
            "per_class": per_class,
            "mean": agg["mean"],
        },
        "image_metrics": {
            "mean": agg["image_mean"],
            "note": "identical for every variant; from the original baseline "
                    "global map",
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

    report_json = os.path.join(OUT, "guided_boundary_report.json")
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
