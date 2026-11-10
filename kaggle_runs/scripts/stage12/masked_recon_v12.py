"""Stage 12: masked local feature reconstruction on the 48x48 token grid.

Frozen 10k Dinomaly checkpoint.  No encoder or decoder parameter is updated.
Only a shared lightweight 2-conv prediction head is trained.

Protocol (fixed a priori, see experiment/MASKED_LOCAL_RECONSTRUCTION_PLAN.md):
    original -> Resize(448) -> CenterCrop(392) -> Normalize -> 392 map
    original -> Resize(768) -> CenterCrop(672) -> Normalize -> 672 map
    both maps -> bilinear 256 -> gaussian(5,4)
    B0 = exact stage-7 guide_fuse672: guided_filter on
         0.5 * base_global + 0.5 * hi672
    official masks = exact original 448->392 gt transform, nearest to 256

Masking:
    The 48x48 patch grid is partitioned into 4x4-patch blocks (12x12 blocks).
    Blocks are assigned to four groups by (block_row % 2, block_col % 2).
    Each group covers exactly 25% of all patches and every patch is covered
    exactly once.  At inference all four groups are visited; each position is
    scored only from the pass in which its own block group was masked.

Training conditions (four diagnostic classes: leather, capsule, grid, screw):
    C1  clean encoder input, same head/loss positions, error at all positions
    M1  masked encoder input (masking before attention at token entrance),
        target is the detached clean encoder feature, loss only at masked
        positions.
Both conditions use the same 96 train/good images per class, the same 16
disjoint normal calibration images per class, seed 1, 10 epochs, lr 3e-4,
batch 2, the same head initialization and the same per-image mask schedule.
The target branch is always the frozen clean encoder output and is detached.

Anomaly maps:
    b0       exact stage-7 guide_fuse672 (raw; used for faithful identity)
    c1       C1 feature error (calibrated)
    m1       M1 masked feature error composed from the four passes (calibrated)
    b0_c1    0.75 * calibrated B0 + 0.25 * calibrated C1
    b0_m1    0.75 * calibrated B0 + 0.25 * calibrated M1
Calibration uses only the 16 held-out train/good images per class; the scale
is max(p95 - median, 0.005) of each method's own map.

Metrics: full-test pooled pixel AP, anomaly-only pooled pixel AP, per-image
(valid anomaly image) pixel AP averaged within class then macro-averaged,
P-AUROC/P-F1/P-AUPRO, recall at fixed pixel FPR, image-level metrics, class
and image counts, paired per-image deltas, and fixed entry-condition checks.
No markdown is produced.  All JSON is strict and written atomically.
"""
import os
import sys
import glob
import json
import math
import time
import argparse
import hashlib
import tempfile
import traceback

import numpy as np

OUT = "/kaggle/working"
SLUG = "eunectesqaq/dinomaly-masked-recon-v12"
STAGE = 12

BASE_RESIZE = 448
BASE_CROP = 392
HI_RESIZE = 768
HI_CROP = 672
GRID = 48
BLOCK = 4
N_GROUPS = 4
PATCHES = GRID * GRID
FEAT_DIM = 768
FEAT_GROUPS = 2
HEAD_HIDDEN = 256
HEAD_GROUPS = 2

TARGET_CLASSES = ("leather", "capsule", "grid", "screw")
EXPECTED_TEST_IMAGES = {
    "bottle": 83, "cable": 150, "capsule": 132, "carpet": 117,
    "grid": 78, "hazelnut": 110, "leather": 124, "metal_nut": 115,
    "pill": 167, "screw": 160, "tile": 117, "toothbrush": 42,
    "transistor": 100, "wood": 79, "zipper": 151,
}
TARGET_TEST_TOTAL = sum(EXPECTED_TEST_IMAGES[c] for c in TARGET_CLASSES)

TRAIN_PER_CLASS = 96
CALIB_PER_CLASS = 16
SEED = 1
TRAIN_EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 0.0
TRAIN_BATCH = 2
EVAL_BATCH = 2
HI_BATCH = 2
FUSION_BASE_WEIGHT = 0.75
FUSION_HEAD_WEIGHT = 0.25
SCALE_FLOOR = 0.005
GUIDE_RADIUS = 7
GUIDE_EPS = 0.0025
GAUSSIAN_KERNEL = 5
GAUSSIAN_SIGMA = 4
FPRS = (0.001, 0.01, 0.05)
METRIC_KEYS = ("I-AUROC", "I-AP", "I-F1",
               "P-AUROC", "P-AP", "P-F1", "P-AUPRO",
               "P-AUROC-anom", "P-AP-anom", "P-AP-image")
RECALL_KEYS = tuple(f"R@FPR{f:g}" for f in FPRS)
VARIANTS = ("b0", "c1", "m1", "b0_c1", "b0_m1")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Frozen stage-7 guide_fuse672 reference numbers.  Used only for a faithful
# B0 identity check; no value here participates in training or calibration.
STAGE7_REFERENCE = {
    "leather": {
        "I-AUROC": 1.0, "I-AP": 1.0000000000000002,
        "I-F1": 0.9999999500000026,
        "P-AUROC": 0.9937065567545976, "P-AP": 0.5361944371174091,
        "P-F1": 0.5467858418062905, "P-AUPRO": 0.9728077340757638,
    },
    "capsule": {
        "I-AUROC": 0.9804547267650578, "I-AP": 0.9957602425175316,
        "I-F1": 0.9771688997727346,
        "P-AUROC": 0.9875661525509197, "P-AP": 0.6280941945267333,
        "P-F1": 0.6166233547422824, "P-AUPRO": 0.9710460006127057,
    },
    "grid": {
        "I-AUROC": 0.9983291562238931, "I-AP": 0.9994052928932504,
        "I-F1": 0.9911503924817943,
        "P-AUROC": 0.9943297949917914, "P-AP": 0.5735487161229744,
        "P-F1": 0.5963522038827771, "P-AUPRO": 0.9745899062858504,
    },
    "screw": {
        "I-AUROC": 0.983398237343718, "I-AP": 0.9946887003395251,
        "I-F1": 0.9626555516675014,
        "P-AUROC": 0.9935720754748925, "P-AP": 0.6227282359303126,
        "P-F1": 0.6156344291088014, "P-AUPRO": 0.9672399066274383,
    },
}

LOG_PATH = None
FP = None
CODE = None
DATA = None
BASE_CKPT = None
STAGE1_METRICS = None
SMOKE = False
TRAIN_PER_CLASS_EFF = TRAIN_PER_CLASS
CALIB_PER_CLASS_EFF = CALIB_PER_CLASS
EPOCHS_EFF = TRAIN_EPOCHS
LIMIT_TEST = 0

torch = nn = F = DataLoader = Image = partial = measure = None
roc_auc_score = average_precision_score = auc = None
get_data_transforms = MVTecDataset = ViTill = vit_encoder = None
VitBlock = bMlp = LinearAttention2 = None
cal_anomaly_maps = get_gaussian_kernel = f1_score_max = None
_MASKS = None


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


def sha256_text_list(values):
    h = hashlib.sha256()
    for v in values:
        h.update(str(v).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def sha256_array(a):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(a).tobytes())
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


def fov(crop, resize):
    return float(crop) / float(resize)


def finite_mean(values):
    vals = [float(v) for v in values
            if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# mask geometry (pure, self-testable)
# ---------------------------------------------------------------------------
def group_index_grid():
    """[GRID, GRID] group id in [0, 3] by block-row/col parity."""
    blocks = GRID // BLOCK
    br = np.arange(blocks)
    bc = np.arange(blocks)
    block_group = (br[:, None] % 2) * 2 + (bc[None, :] % 2)
    return np.repeat(np.repeat(block_group, BLOCK, axis=0), BLOCK, axis=1)


def group_index_flat():
    return group_index_grid().reshape(-1).astype(np.int64)


def numpy_group_masks():
    groups = group_index_flat()
    return np.stack([groups == g for g in range(N_GROUPS)], axis=0)


def torch_group_masks(device):
    return torch.as_tensor(numpy_group_masks(), dtype=torch.bool,
                           device=device)


def masks_for_groups(device, group_ids):
    """[B, PATCHES] bool masks for integer group ids (tensor/array)."""
    global _MASKS
    if _MASKS is None or _MASKS.device != device:
        _MASKS = torch_group_masks(device)
    ids = torch.as_tensor(group_ids, dtype=torch.long, device=device)
    return _MASKS[ids]


def compose_group_maps(errors, mask2d=None):
    """Fill [B,H,W] from [4,B,H,W] using each group's own positions."""
    n_groups, b, h, w = errors.shape
    if n_groups != N_GROUPS or h != GRID or w != GRID:
        raise ValueError(f"unexpected group error shape {tuple(errors.shape)}")
    if mask2d is None:
        mask2d = torch.as_tensor(
            group_index_grid(), device=errors.device)  # [H,W] ids
        out = torch.zeros((b, h, w), dtype=errors.dtype,
                          device=errors.device)
        for g in range(N_GROUPS):
            sel = mask2d == g
            out[:, sel] = errors[g][:, sel]
        return out
    # Boolean masks [4,H,W]
    out = torch.zeros((b, h, w), dtype=errors.dtype, device=errors.device)
    for g in range(N_GROUPS):
        sel = mask2d[g]
        out[:, sel] = errors[g][:, sel]
    return out


def patch_mask_to_pixel_mask(mask_flat, size=HI_CROP):
    """[B,PATCHES] bool -> [B,1,size,size] bool nearest patch expansion."""
    b = mask_flat.shape[0]
    m = mask_flat.reshape(b, 1, GRID, GRID).float()
    m = F.interpolate(m, size=size, mode="nearest")
    return m.bool()


def split_train_calibration(paths, n_train=TRAIN_PER_CLASS,
                            n_calib=CALIB_PER_CLASS, seed=SEED):
    """Deterministic disjoint split of sorted train/good paths."""
    paths = sorted(str(p) for p in paths)
    if len(paths) < n_train + n_calib:
        raise ValueError(f"need at least {n_train + n_calib} train/good "
                         f"images, got {len(paths)}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(paths))
    calib = sorted(paths[int(i)] for i in perm[:n_calib])
    train = sorted(paths[int(i)] for i in perm[n_calib:n_calib + n_train])
    if set(train) & set(calib):
        raise AssertionError("train/calibration paths overlap")
    return train, calib


def make_training_schedule(n_images, epochs=EPOCHS_EFF, seed=SEED):
    rng = np.random.default_rng(seed + 7919)
    order = np.stack([rng.permutation(n_images) for _ in range(epochs)])
    groups = rng.integers(0, N_GROUPS, size=(epochs, n_images))
    return order, groups.astype(np.int64)


# ---------------------------------------------------------------------------
# pure-numpy metrics / calibration helpers
# ---------------------------------------------------------------------------
def fixed_fpr_recall(pr_flat, gt_flat, fprs=FPRS):
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


def anomaly_only_metrics(labels_np, gt_np, pr):
    rows = labels_np == 1
    if not rows.any():
        return float("nan"), float("nan")
    gt_a = gt_np[rows].ravel()
    pr_a = pr[rows].ravel()
    if len(np.unique(gt_a)) < 2:
        return float("nan"), float("nan")
    return (float(roc_auc_score(gt_a, pr_a)),
            float(average_precision_score(gt_a, pr_a)))


def per_image_pixel_ap(gt_np, pr, labels_np):
    """AP per valid anomaly image; NaN elsewhere."""
    n = len(labels_np)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if int(labels_np[i]) != 1:
            continue
        g = gt_np[i].reshape(-1)
        if g.sum() == 0 or g.sum() == g.size:
            continue
        out[i] = float(average_precision_score(g, pr[i].reshape(-1)))
    return out


def calibration_stats(values):
    """median / p95 / scale / clamped from one method's calibration pixels."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0 or not np.isfinite(v).any():
        return {"median": 0.0, "q95": 0.0, "scale": SCALE_FLOOR,
                "clamped": True, "n_pixels": 0}
    v = v[np.isfinite(v)]
    med = float(np.quantile(v, 0.5))
    q95 = float(np.quantile(v, 0.95))
    raw_scale = q95 - med
    scale = max(raw_scale, SCALE_FLOOR)
    return {"median": med, "q95": q95, "scale": float(scale),
            "raw_scale": float(raw_scale),
            "clamped": bool(raw_scale < SCALE_FLOOR),
            "n_pixels": int(v.size)}


def normalize_map(m, stat):
    return (m - float(stat["median"])) / float(stat["scale"])


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


# ---------------------------------------------------------------------------
# stage-7 faithful helpers
# ---------------------------------------------------------------------------
def _box_mean(x, radius):
    radius = int(radius)
    k = 2 * radius + 1
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=radius,
                        count_include_pad=False)


def guided_filter(guide, heatmap, radius=GUIDE_RADIUS, eps=GUIDE_EPS):
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


def guide_from_hi(x_hi):
    mean = x_hi.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x_hi.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    rgb = (x_hi * std + mean).clamp(0.0, 1.0)
    rgb = F.interpolate(rgb, size=256, mode="bilinear",
                        align_corners=False)
    return (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2]
            + 0.114 * rgb[:, 2:3]).clamp(0.0, 1.0)


def postprocess(amap, gaussian):
    return gaussian(F.interpolate(amap, size=256, mode="bilinear",
                                  align_corners=False))


def top1pct(amap):
    flat = amap.flatten(1)
    k = max(1, int(flat.shape[1] * 0.01))
    return torch.sort(flat, dim=1, descending=True).values[:, :k].mean(dim=1)


# ---------------------------------------------------------------------------
# runtime setup
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
        "E:/dinomaly-mvp/Dinomaly",
        "/kaggle/input/dinomaly-code",
        "/kaggle/input/datasets/eunectesqaq/dinomaly-code",
    ])
    DATA = locate("bottle", [
        "E:/dinomaly-mvp/data/mvtec_anomaly_detection",
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
        "E:/dinomaly-mvp/kaggle_runs/dinomaly-base-10k/model_10000.pt",
        "/kaggle/input/dinomaly-base-10k/model_10000.pt",
        "/kaggle/input/notebooks/eunectesqaq/dinomaly-base-10k/model_10000.pt",
        "/kaggle/input/datasets/eunectesqaq/dinomaly-base-10k/model_10000.pt",
    ])
    STAGE1_METRICS = find_input("metrics.json", [
        "E:/dinomaly-mvp/kaggle_runs/dinomaly-base-10k/metrics.json",
        "/kaggle/input/dinomaly-base-10k/metrics.json",
        "/kaggle/input/notebooks/eunectesqaq/dinomaly-base-10k/metrics.json",
        "/kaggle/input/datasets/eunectesqaq/dinomaly-base-10k/metrics.json",
    ])
    if not BASE_CKPT:
        raise RuntimeError("model_10000.pt not found under /kaggle/input")
    log(f"base checkpoint={BASE_CKPT}")
    log(f"stage1 metrics={STAGE1_METRICS}")


# ---------------------------------------------------------------------------
# model + feature extraction
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


def encoder_input_tokens(encoder, x, masks=None):
    """Token entrance: masking happens here, before any attention block."""
    return encoder.prepare_tokens(x, masks=masks)


def encode_fused(encoder, x, masks=None):
    """Frozen encoder -> two fused intermediate feature groups.

    Returns [B, 2, 768, GRID, GRID].  Exactly the two stage-7 encoder groups
    [layers 2-5] and [layers 6-9], class/register tokens removed.
    """
    tokens = encoder_input_tokens(encoder, x, masks=masks)
    en_list = []
    for i, blk in enumerate(encoder.blocks):
        if i > 9:
            break
        with torch.no_grad():
            tokens = blk(tokens)
        if i in (2, 3, 4, 5, 6, 7, 8, 9):
            en_list.append(tokens)
    reg = int(getattr(encoder, "num_register_tokens", 0))
    side = int(math.sqrt(en_list[0].shape[1] - 1 - reg))
    if side != GRID:
        raise RuntimeError(f"token grid {side} != {GRID}")
    groups = []
    for idxs in ((0, 1, 2, 3), (4, 5, 6, 7)):
        e = torch.stack([en_list[i] for i in idxs], dim=0).mean(dim=0)
        e = e[:, 1 + reg:, :]
        groups.append(e.permute(0, 2, 1).reshape(
            x.shape[0], -1, side, side).contiguous())
    return torch.stack(groups, dim=1)


def vitill_forward_with_clean(model, x):
    """Exact ViTill forward + clean encoder fused features from one pass."""
    encoder = model.encoder
    tokens = encoder_input_tokens(encoder, x, masks=None)
    en_list = []
    for i, blk in enumerate(encoder.blocks):
        if i > model.target_layers[-1]:
            break
        with torch.no_grad():
            tokens = blk(tokens)
        if i in model.target_layers:
            en_list.append(tokens)
    reg = int(getattr(encoder, "num_register_tokens", 0))
    side = int(math.sqrt(en_list[0].shape[1] - 1 - reg))
    clean_groups = []
    for idxs in ((0, 1, 2, 3), (4, 5, 6, 7)):
        e = torch.stack([en_list[i] for i in idxs], dim=0).mean(dim=0)
        e = e[:, 1 + reg:, :]
        clean_groups.append(e.permute(0, 2, 1).reshape(
            x.shape[0], -1, side, side).contiguous())
    clean = torch.stack(clean_groups, dim=1)

    feat = model.fuse_feature(en_list)
    for blk in model.bottleneck:
        feat = blk(feat)
    de_list = []
    for blk in model.decoder:
        feat = blk(feat, attn_mask=None)
        de_list.append(feat)
    de_list = de_list[::-1]
    en = [model.fuse_feature([en_list[idx] for idx in idxs])
          for idxs in model.fuse_layer_encoder]
    de = [model.fuse_feature([de_list[idx] for idx in idxs])
          for idxs in model.fuse_layer_decoder]
    en = [e[:, 1 + reg:, :] for e in en]
    de = [d[:, 1 + reg:, :] for d in de]
    en = [e.permute(0, 2, 1).reshape(
        x.shape[0], -1, side, side).contiguous() for e in en]
    de = [d.permute(0, 2, 1).reshape(
        x.shape[0], -1, side, side).contiguous() for d in de]
    amap, _ = cal_anomaly_maps(en, de, HI_CROP)
    return amap, clean


def global_map(model, images):
    en, de = model(images)
    amap, _ = cal_anomaly_maps(en, de, images.shape[-1])
    return amap


def global_map_chunked(model, images, chunk):
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


def stage7_reference_b0(model, x_base, x_hi, gaussian):
    """Independent copy of the stage-7 guide_fuse672 code path."""
    base_raw = global_map_chunked(model, x_base, BASE_BATCH_EFF)
    base = postprocess(base_raw, gaussian)
    hi_raw = global_map_chunked(model, x_hi, HI_BATCH_EFF)
    hi = postprocess(hi_raw, gaussian)
    fused = 0.5 * base + 0.5 * hi
    guide = guide_from_hi(x_hi)
    return guided_filter(guide, fused)


# ---------------------------------------------------------------------------
# prediction head / losses
# ---------------------------------------------------------------------------
from torch.nn import Module


class MaskedFeatureHead(Module):
    """Shared two-conv head with a grouped output projection."""

    def __init__(self, channels=FEAT_GROUPS * FEAT_DIM,
                 hidden=HEAD_HIDDEN, groups=HEAD_GROUPS):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=3,
                               padding=1, bias=True)
        self.norm1 = nn.GroupNorm(8, hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3,
                               padding=1, bias=True)
        self.norm2 = nn.GroupNorm(8, hidden)
        self.act = nn.GELU()
        self.proj = nn.Conv2d(hidden, channels, kernel_size=1,
                              groups=groups, bias=True)

    def forward(self, feats):
        x = feats.flatten(1, 2)  # [B, 2*768, H, W]
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        x = self.proj(x)
        return x.reshape(feats.shape[0], FEAT_GROUPS, FEAT_DIM,
                         feats.shape[3], feats.shape[4])


def cosine_error_map(pred, target):
    """Mean over the two feature groups of 1 - cosine similarity."""
    cos = F.cosine_similarity(pred.flatten(3), target.flatten(3), dim=2)
    return (1.0 - cos).mean(dim=1).reshape(
        pred.shape[0], 1, GRID, GRID)


def masked_cosine_loss(pred, target, mask_flat):
    """Mean cosine distance only at masked positions, groups equal weight."""
    cos = F.cosine_similarity(pred.flatten(3), target.flatten(3), dim=2)
    err = 1.0 - cos
    w = mask_flat.float().unsqueeze(1)
    per_group = (err * w).sum(dim=2) / w.sum(dim=2).clamp_min(1.0)
    return per_group.mean()


def head_group_losses(pred, target, mask_flat):
    cos = F.cosine_similarity(pred.flatten(3), target.flatten(3), dim=2)
    err = 1.0 - cos
    w = mask_flat.float().unsqueeze(1)
    per_group = (err * w).sum(dim=2) / w.sum(dim=2).clamp_min(1.0)
    return per_group.mean(dim=0)


# ---------------------------------------------------------------------------
# training cache (clean targets, reusable as C1 input)
# ---------------------------------------------------------------------------
def train_good_paths(root):
    paths = []
    for ext in ("*.png", "*.JPG", "*.jpg", "*.bmp"):
        paths.extend(glob.glob(os.path.join(root, "train", "good", ext)))
    return sorted(paths)


def build_clean_cache(model, items, splits, hi_tf, device, cache_root,
                      force=False):
    """Cache clean fused features as float16 .npy, one file per image."""
    os.makedirs(cache_root, exist_ok=True)
    meta_path = os.path.join(cache_root, "meta.json")
    entries = []
    started = time.time()
    for item in items:
        item_dir = os.path.join(cache_root, item)
        os.makedirs(item_dir, exist_ok=True)
        for j, path in enumerate(splits[item]["train"]):
            entries.append({
                "item": item, "local_index": j, "path": path,
                "npy": os.path.join(item_dir, f"{j:04d}.npy"),
            })
    meta = {
        "fingerprint": FP,
        "feature_shape": [FEAT_GROUPS, FEAT_DIM, GRID, GRID],
        "dtype": "float16",
        "entries": entries,
        "split_sha": {it: sha256_text_list(splits[it]["train"])
                      for it in items},
        "source": "frozen encoder clean forward on 768->672 crop",
    }
    reuse = False
    if not force and os.path.isfile(meta_path):
        try:
            old = read_json(meta_path)
            reuse = (old.get("fingerprint") == FP
                     and old.get("split_sha") == meta["split_sha"]
                     and len(old.get("entries", [])) == len(entries))
        except Exception:
            reuse = False
    if reuse:
        log(f"clean cache reuse: {len(entries)} files")
        return entries, meta

    encoder = model.encoder
    log(f"building clean cache: {len(entries)} images -> {cache_root}")
    with torch.no_grad():
        for start in range(0, len(entries), TRAIN_BATCH):
            chunk = entries[start:start + TRAIN_BATCH]
            xh = []
            for e in chunk:
                pil = Image.open(e["path"]).convert("RGB")
                xh.append(hi_tf(pil))
            x = torch.stack(xh).to(device)
            feats = encode_fused(encoder, x, masks=None)
            feats = feats.detach().half().cpu().numpy()
            if not np.isfinite(feats.astype(np.float32)).all():
                raise RuntimeError("non-finite clean features")
            for k, e in enumerate(chunk):
                np.save(e["npy"], feats[k])
            if (start // TRAIN_BATCH) % 25 == 0:
                log(f"  cache {start + len(chunk)}/{len(entries)} "
                    f"elapsed={time.time() - started:.1f}s")
    write_json_atomic(meta_path, meta)
    log(f"clean cache done in {time.time() - started:.1f}s")
    return entries, meta


def load_clean_batch(entries, ids, device):
    arrs = [np.load(entries[int(i)]["npy"]).astype(np.float32)
            for i in ids]
    return torch.from_numpy(np.stack(arrs, 0)).to(device)


def load_hi_batch(entries, ids, hi_tf, device):
    tensors = []
    for i in ids:
        pil = Image.open(entries[int(i)]["path"]).convert("RGB")
        tensors.append(hi_tf(pil))
    return torch.stack(tensors).to(device)


def init_head(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return MaskedFeatureHead().to(DEVICE if "DEVICE" in globals() else "cpu")


def train_heads(model, entries, schedule, device):
    """Train C1 and M1 heads with identical init/order/updates."""
    order, groups = schedule
    head_c1 = MaskedFeatureHead().to(device)
    # Re-seed immediately before the second head so initialization is identical.
    state_c1 = {k: v.detach().clone() for k, v in head_c1.state_dict().items()}
    head_m1 = MaskedFeatureHead().to(device)
    head_m1.load_state_dict(state_c1)
    opt_c1 = torch.optim.AdamW(head_c1.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)
    opt_m1 = torch.optim.AdamW(head_m1.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)
    encoder = model.encoder
    n = len(entries)
    hist = {"epochs": [], "c1_loss": [], "m1_loss": [],
            "c1_group_loss": [], "m1_group_loss": []}
    started = time.time()
    for epoch in range(EPOCHS_EFF):
        c1_losses, m1_losses = [], []
        c1_gl, m1_gl = [], []
        t0 = time.time()
        perm = order[epoch]
        for start in range(0, n, TRAIN_BATCH):
            ids = perm[start:start + TRAIN_BATCH]
            gids = groups[epoch, ids]
            clean = load_clean_batch(entries, ids, device)
            clean_target = clean.detach()  # cached no-grad target
            masks = masks_for_groups(device, gids)

            # C1: full clean input, same head, same loss positions.
            pred_c1 = head_c1(clean)
            loss_c1 = masked_cosine_loss(pred_c1, clean_target, masks)
            opt_c1.zero_grad(set_to_none=True)
            loss_c1.backward()
            opt_c1.step()
            gl_c1 = head_group_losses(pred_c1.detach(), clean_target,
                                      masks)

            # M1: masking at token entrance, clean target detached.
            x_hi = load_hi_batch(entries, ids, HI_TF_EFF, device)
            with torch.no_grad():
                masked_feats = encode_fused(encoder, x_hi, masks=masks)
            pred_m1 = head_m1(masked_feats)
            loss_m1 = masked_cosine_loss(pred_m1, clean_target, masks)
            opt_m1.zero_grad(set_to_none=True)
            loss_m1.backward()
            opt_m1.step()
            gl_m1 = head_group_losses(pred_m1.detach(), clean_target,
                                      masks)

            c1_losses.append(float(loss_c1.detach()))
            m1_losses.append(float(loss_m1.detach()))
            c1_gl.append(gl_c1.detach().cpu().numpy())
            m1_gl.append(gl_m1.detach().cpu().numpy())
            if (start // TRAIN_BATCH) % 50 == 0:
                log(f"  epoch {epoch + 1}/{EPOCHS_EFF} "
                    f"step {start // TRAIN_BATCH + 1}/"
                    f"{(n + TRAIN_BATCH - 1) // TRAIN_BATCH} "
                    f"c1={float(loss_c1):.4f} m1={float(loss_m1):.4f}")
        epoch_row = {
            "epoch": epoch + 1,
            "c1_loss": float(np.mean(c1_losses)),
            "m1_loss": float(np.mean(m1_losses)),
            "c1_group_loss": np.mean(np.stack(c1_gl), axis=0).tolist(),
            "m1_group_loss": np.mean(np.stack(m1_gl), axis=0).tolist(),
            "seconds": float(time.time() - t0),
        }
        hist["epochs"].append(epoch_row)
        hist["c1_loss"].append(epoch_row["c1_loss"])
        hist["m1_loss"].append(epoch_row["m1_loss"])
        hist["c1_group_loss"].append(epoch_row["c1_group_loss"])
        hist["m1_group_loss"].append(epoch_row["m1_group_loss"])
        log(f"epoch {epoch + 1}/{EPOCHS_EFF} "
            f"c1={epoch_row['c1_loss']:.4f} "
            f"m1={epoch_row['m1_loss']:.4f} "
            f"sec={epoch_row['seconds']:.1f}")
    hist["seconds"] = float(time.time() - started)
    hist["updates_per_head"] = int(EPOCHS_EFF * math.ceil(n / TRAIN_BATCH))
    return head_c1, head_m1, hist


# ---------------------------------------------------------------------------
# inference: one batch -> all raw variant maps
# ---------------------------------------------------------------------------
def infer_batch(model, heads, x_base, x_hi, gaussian, device):
    """Return base 256 map + raw B0/C1/M1 256 maps for a batch."""
    base_raw = global_map_chunked(model, x_base, BASE_BATCH_EFF)
    base = postprocess(base_raw, gaussian)
    del base_raw
    hi_raw, clean = vitill_forward_with_clean(model, x_hi)
    hi = postprocess(hi_raw, gaussian)
    del hi_raw
    fused = 0.5 * base + 0.5 * hi
    guide = guide_from_hi(x_hi)
    b0_raw = guided_filter(guide, fused)

    # C1: one clean pass, error at every position.
    clean_round = clean.half().float()
    pred_c1 = heads["c1"](clean_round)
    c1_raw = postprocess(cosine_error_map(pred_c1, clean_round), gaussian)

    # M1: four masked passes; each position uses its own masked pass.
    group_errors = []
    for g in range(N_GROUPS):
        gids = torch.full((x_hi.shape[0],), g, dtype=torch.long,
                          device=device)
        masks = masks_for_groups(device, gids)
        with torch.no_grad():
            masked_feats = encode_fused(model.encoder, x_hi, masks=masks)
        pred = heads["m1"](masked_feats)
        err = cosine_error_map(pred, clean_round)
        group_errors.append(err)
    stacked = torch.cat(group_errors, dim=0).squeeze(1)  # [4*B,H,W]
    stacked = stacked.reshape(N_GROUPS, x_hi.shape[0], GRID, GRID)
    composed = compose_group_maps(stacked)
    m1_raw = postprocess(composed.unsqueeze(1), gaussian)

    return {
        "base": base,
        "b0": b0_raw,
        "c1": c1_raw,
        "m1": m1_raw,
    }


def make_variant_maps(raw, stats):
    b0 = raw["b0"]
    c1 = normalize_map(raw["c1"], stats["c1"])
    m1 = normalize_map(raw["m1"], stats["m1"])
    b0n = normalize_map(raw["b0"], stats["b0"])
    return {
        "b0": b0,                       # faithful raw stage-7 map
        "c1": c1,
        "m1": m1,
        "b0_c1": FUSION_BASE_WEIGHT * b0n + FUSION_HEAD_WEIGHT * c1,
        "b0_m1": FUSION_BASE_WEIGHT * b0n + FUSION_HEAD_WEIGHT * m1,
    }, b0n


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def compute_calibration(model, heads, items, splits, base_tf, hi_tf,
                        gaussian, device):
    result = {}
    started = time.time()
    for item in items:
        paths = splits[item]["calib"]
        values = {"b0": [], "c1": [], "m1": []}
        with torch.no_grad():
            for start in range(0, len(paths), EVAL_BATCH):
                chunk = paths[start:start + EVAL_BATCH]
                xb, xh = [], []
                for p in chunk:
                    pil = Image.open(p).convert("RGB")
                    xb.append(base_tf(pil))
                    xh.append(hi_tf(pil))
                raw = infer_batch(model, heads, torch.stack(xb).to(device),
                                  torch.stack(xh).to(device), gaussian,
                                  device)
                for k in values:
                    values[k].append(raw[k].detach().cpu().numpy())
        stats = {k: calibration_stats(np.concatenate(v, axis=0))
                 for k, v in values.items()}
        result[item] = {
            "n_images": len(paths),
            "paths_sha256": sha256_text_list(paths),
            "stats": stats,
        }
        log(f"calibration {item}: " + " ".join(
            f"{k} med={stats[k]['median']:.6f} "
            f"scale={stats[k]['scale']:.6f}"
            f"{'*' if stats[k]['clamped'] else ''}"
            for k in ("b0", "c1", "m1")))
        del values
    result["_seconds"] = float(time.time() - started)
    return result


# ---------------------------------------------------------------------------
# test data + per-class evaluation
# ---------------------------------------------------------------------------
class MultiScaleTestDataset:
    def __init__(self, root, base_transform, gt_transform, hi_transform):
        self.base = MVTecDataset(root=root, transform=base_transform,
                                 gt_transform=gt_transform, phase="test")
        self.hi_transform = hi_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, gt, label, path = self.base[idx]
        pil = Image.open(path).convert("RGB")
        hi = self.hi_transform(pil)
        return img, gt, label, path, hi


def summarize_variant(gt_np, pr, labels_np, img_metrics, regions, fprs=FPRS):
    gt_flat = gt_np.reshape(-1)
    pr_flat = pr.reshape(-1)
    recall, thresholds = fixed_fpr_recall(pr_flat, gt_flat, fprs)
    anom_auroc, anom_ap = anomaly_only_metrics(labels_np, gt_np, pr)
    per_img = per_image_pixel_ap(gt_np, pr, labels_np)
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
        "P-AP-image": float(np.nanmean(per_img))
        if np.isfinite(per_img).any() else float("nan"),
    }
    return {"metrics": row, "recall": recall, "thresholds": thresholds,
            "per_image_ap": per_img}


def evaluate_class(item, dataset, model, heads, gaussian, stats, device):
    loader = DataLoader(dataset, batch_size=EVAL_BATCH, shuffle=False,
                        num_workers=0 if os.name == "nt" else 4,
                        pin_memory=torch.cuda.is_available())
    buckets = {v: [] for v in VARIANTS}
    raw_buckets = {"b0": [], "c1": [], "m1": []}
    base_scores, variant_scores = [], {v: [] for v in VARIANTS}
    gt_buf, labels, paths = [], [], []
    n_batches = len(loader)
    started = time.time()
    reset_peak()

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            img, gt, label, path, his = batch
            img = img.to(device, non_blocking=True)
            his = his.to(device, non_blocking=True)
            raw = infer_batch(model, heads, img, his, gaussian, device)
            maps, _b0n = make_variant_maps(raw, stats)

            base_scores.extend(top1pct(raw["base"]).cpu().numpy().tolist())
            for v in VARIANTS:
                buckets[v].append(maps[v].cpu())
                variant_scores[v].extend(
                    top1pct(maps[v]).cpu().numpy().tolist())
            for k in raw_buckets:
                raw_buckets[k].append(raw[k].cpu())

            gt256 = F.interpolate(gt.float(), size=256,
                                  mode="nearest").bool()
            if gt256.shape[1] > 1:
                gt256 = gt256.max(dim=1, keepdim=True)[0]
            gt_buf.append(gt256.cpu())
            labels.extend(label.numpy().tolist())
            paths.extend(list(path))

            del raw, maps, img, his
            if bi == 0 or (bi + 1) % 10 == 0 or (bi + 1) == n_batches:
                log(f"  {item} batch {bi + 1}/{n_batches} "
                    f"images={len(labels)}/{len(dataset)} "
                    f"elapsed={time.time() - started:.1f}s")

    gt_np = torch.cat(gt_buf, dim=0)[:, 0].numpy()
    gt_np = (gt_np > 0.5).astype(np.uint8)
    labels_np = np.asarray(labels, dtype=np.int64)
    scores_base = np.asarray(base_scores, dtype=np.float64)
    regions = label_regions(gt_np)
    per_image_index = np.arange(len(labels_np))

    variants = {}
    preds16 = {}
    per_image_rows = []
    for v in VARIANTS:
        pr = torch.cat(buckets[v], dim=0)[:, 0].numpy().astype(np.float32)
        preds16[v] = pr.astype(np.float16)
        img_metrics = image_metrics_from_scores(
            labels_np, np.asarray(variant_scores[v]))
        summary = summarize_variant(gt_np, pr, labels_np, img_metrics,
                                    regions)
        variants[v] = summary
        for i in per_image_index:
            per_image_rows.append({
                "index": int(i), "path": os.path.basename(str(paths[i])),
                "label": int(labels_np[i]),
                "gt_positive_pixels": int(gt_np[i].sum()),
                "variant": v,
                "image_score": float(np.asarray(variant_scores[v])[i]),
                "pixel_ap": (float(summary["per_image_ap"][i])
                             if np.isfinite(summary["per_image_ap"][i])
                             else None),
            })
        buckets[v] = None

    img_metrics_original = image_metrics_from_scores(labels_np, scores_base)
    raw_metrics = {}
    for k, maps in raw_buckets.items():
        pr = torch.cat(maps, dim=0)[:, 0].numpy().astype(np.float32)
        raw_metrics[k] = {
            "P-AP": float(average_precision_score(gt_np.ravel(),
                                                  pr.reshape(-1))),
            "P-AUROC": float(roc_auc_score(gt_np.ravel(), pr.reshape(-1))),
        }
        raw_buckets[k] = None

    result = {
        "class": item,
        "fingerprint": FP,
        "n_test": int(len(labels_np)),
        "n_normal": int((labels_np == 0).sum()),
        "n_anomaly": int((labels_np == 1).sum()),
        "n_valid_anomaly_images": int(np.isfinite(
            variants["b0"]["per_image_ap"]).sum()),
        "n_gt_positive_pixels": int(gt_np.sum()),
        "n_gt_negative_pixels": int((gt_np == 0).sum()),
        "calibration": stats,
        "image_metrics_by_variant": {
            v: image_metrics_from_scores(
                labels_np, np.asarray(variant_scores[v])) for v in VARIANTS},
        "image_metrics_original_base": img_metrics_original,
        "image_scores_sha256": {
            "original_base": sha256_array(scores_base.astype(np.float64)),
            **{v: sha256_array(np.asarray(variant_scores[v],
                                          dtype=np.float64))
               for v in VARIANTS},
        },
        "variants": {
            v: {"metrics": variants[v]["metrics"],
                "recall": variants[v]["recall"],
                "thresholds": variants[v]["thresholds"],
                "per_image_ap": [
                    (float(x) if np.isfinite(x) else None)
                    for x in variants[v]["per_image_ap"]]}
            for v in VARIANTS
        },
        "raw_metrics": raw_metrics,
        "per_image": per_image_rows,
        "seconds": float(time.time() - started),
        "gpu_peak_mib": peak_mib(),
    }
    arrays = {
        "gt": gt_np,
        "labels": labels_np,
        "paths": np.asarray(paths, dtype=str),
        "scores_base": scores_base,
        "scores": {v: np.asarray(variant_scores[v], dtype=np.float64)
                   for v in VARIANTS},
        "preds": preds16,
    }
    return result, arrays


def save_pred_npz(path, arrays):
    payload = {
        "gt": arrays["gt"],
        "labels": arrays["labels"],
        "paths": arrays["paths"],
        "scores_base": arrays["scores_base"],
    }
    for v in VARIANTS:
        payload[f"pred_{v}"] = arrays["preds"][v]
        payload[f"score_{v}"] = arrays["scores"][v]
    np.savez(path, **payload)


def load_pred_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {
            "gt": z["gt"],
            "labels": z["labels"],
            "paths": z["paths"],
            "scores_base": z["scores_base"],
            "preds": {v: z[f"pred_{v}"] for v in VARIANTS},
            "scores": {v: z[f"score_{v}"] for v in VARIANTS},
        }


def render_examples(item, dataset, selected, model, heads, gaussian,
                    stats, device, out_dir):
    """Save one success, one false positive and one miss contact sheet."""
    if not selected:
        return []
    rows = []
    for kind, idx in selected:
        img, gt, label, path, hi = dataset[int(idx)]
        x_base = img.unsqueeze(0).to(device)
        x_hi = hi.unsqueeze(0).to(device)
        with torch.no_grad():
            raw = infer_batch(model, heads, x_base, x_hi, gaussian, device)
            maps, b0n = make_variant_maps(raw, stats)
        panels = []
        rgb = (x_hi * x_hi.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
               + x_hi.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        rgb = rgb.clamp(0, 1)[0].mul(255).byte().cpu().numpy()
        rgb = np.asarray(Image.fromarray(
            rgb.transpose(1, 2, 0)).resize((256, 256), Image.BILINEAR))
        panels.append(("orig", rgb))
        g = (F.interpolate(gt.unsqueeze(0).float(), size=256,
                           mode="nearest")[0, 0].cpu().numpy() * 255)
        panels.append(("gt", np.repeat(g[:, :, None], 3, axis=2).astype(
            np.uint8)))
        order = ("b0", "c1", "m1", "b0_c1", "b0_m1")
        for name in order:
            if name == "b0":
                m = b0n
            else:
                m = maps[name]
            arr = m[0, 0].detach().cpu().numpy()
            arr = np.clip(arr / 3.0, 0.0, 1.0) * 255.0
            arr = np.repeat(arr[:, :, None].astype(np.uint8), 3, axis=2)
            panels.append((name, arr))
        sheet = np.concatenate([p[1] for p in panels], axis=1)
        os.makedirs(out_dir, exist_ok=True)
        outp = os.path.join(out_dir, f"{item}_{kind}.png")
        Image.fromarray(sheet).save(outp)
        rows.append({"kind": kind, "index": int(idx),
                     "path": str(path), "label": int(label),
                     "panel_order": [p[0] for p in panels],
                     "file": outp})
    return rows


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def aggregate(per_class, items, stage7_reference=None):
    mean = {}
    recall_mean = {}
    for v in VARIANTS:
        mean[v] = {k: finite_mean(
            [per_class[it]["variants"][v]["metrics"].get(k)
             for it in items]) for k in METRIC_KEYS}
        recall_mean[v] = {k: finite_mean(
            [per_class[it]["variants"][v]["recall"].get(k)
             for it in items]) for k in RECALL_KEYS}

    image_mean = {
        v: {k: finite_mean(
            [per_class[it]["image_metrics_by_variant"][v].get(k)
             for it in items]) for k in ("I-AUROC", "I-AP", "I-F1")}
        for v in VARIANTS
    }
    image_mean_original = {
        k: finite_mean([per_class[it]["image_metrics_original_base"].get(k)
                        for it in items])
        for k in ("I-AUROC", "I-AP", "I-F1")}

    comparisons = {}
    reference = "b0"
    for v in VARIANTS:
        if v != reference:
            comparisons[f"{v} - {reference}"] = (v, reference)
    comparisons["b0_m1 - b0_c1"] = ("b0_m1", "b0_c1")

    deltas = {}
    for name, (a, b) in comparisons.items():
        deltas[name] = {
            "metrics": {k: mean[a][k] - mean[b][k] for k in METRIC_KEYS},
            "recall": {k: recall_mean[a][k] - recall_mean[b][k]
                       for k in RECALL_KEYS},
            "image": {k: image_mean[a][k] - image_mean[b][k]
                      for k in ("I-AUROC", "I-AP", "I-F1")},
            "per_class_P-AP": {
                it: (per_class[it]["variants"][a]["metrics"]["P-AP"]
                     - per_class[it]["variants"][b]["metrics"]["P-AP"])
                for it in items},
        }

    paired = {}
    for name, (a, b) in comparisons.items():
        per_class_paired = {}
        for it in items:
            pa = np.asarray(
                per_class[it]["variants"][a]["per_image_ap"],
                dtype=np.float64)
            pb = np.asarray(
                per_class[it]["variants"][b]["per_image_ap"],
                dtype=np.float64)
            valid = np.isfinite(pa) & np.isfinite(pb)
            per_class_paired[it] = (
                float(np.mean(pa[valid] - pb[valid]))
                if valid.any() else None)
        paired[name] = {
            "per_class_mean_delta": per_class_paired,
            "macro_mean_delta": finite_mean(list(per_class_paired.values())),
            "n_images_with_valid_pixel_ap": int(sum(
                np.isfinite(np.asarray(
                    per_class[it]["variants"][a]["per_image_ap"],
                    dtype=np.float64)).sum() for it in items)),
        }

    b0_identity = None
    if stage7_reference:
        rows = {}
        for it in items:
            if it not in stage7_reference:
                continue
            got = per_class[it]["variants"]["b0"]["metrics"]
            ref = stage7_reference[it]
            rows[it] = {k: float(got[k] - ref[k])
                        for k in ref if k in got}
        b0_identity = {
            "reference": "stage7 guide_fuse672 (frozen constants)",
            "per_class_delta": rows,
            "max_abs_delta": {
                k: max(abs(r[k]) for r in rows.values() if k in r)
                for k in ("I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP",
                          "P-F1", "P-AUPRO")
                if any(k in r for r in rows.values())},
        }

    # Fixed entry criterion, decided before running.
    gain = deltas["b0_m1 - b0"]["metrics"]["P-AP"]
    vs_c1 = deltas["b0_m1 - b0_c1"]["metrics"]["P-AP"]
    per_class_d = deltas["b0_m1 - b0"]["per_class_P-AP"]
    n_no_drop = sum(1 for x in per_class_d.values() if x >= 0.0)
    min_delta = min(per_class_d.values()) if per_class_d else float("nan")
    per_image_gain = deltas["b0_m1 - b0"]["metrics"]["P-AP-image"]
    passed = bool(
        np.isfinite(gain) and gain >= 0.01
        and np.isfinite(vs_c1) and vs_c1 > 0.0
        and n_no_drop >= 3
        and np.isfinite(min_delta) and min_delta >= -0.02)
    entry = {
        "b0_m1_minus_b0_P-AP": gain,
        "b0_m1_minus_b0_c1_P-AP": vs_c1,
        "classes_with_nonnegative_delta": int(n_no_drop),
        "classes_total": len(items),
        "min_per_class_delta": min_delta,
        "per_image_P-AP_direction": per_image_gain,
        "thresholds": {"P-AP_gain_min": 0.01, "must_exceed_b0_c1": True,
                       "min_nonnegative_classes": 3,
                       "max_allowed_class_drop": 0.02},
        "passed": passed,
    }

    return {
        "mean": {v: {"metrics": mean[v], "recall": recall_mean[v]}
                 for v in VARIANTS},
        "image_mean": image_mean,
        "image_mean_original_base": image_mean_original,
        "deltas": deltas,
        "paired_image_ap_deltas": paired,
        "entry_condition": entry,
        "b0_identity_vs_stage7": b0_identity,
    }


def pooled_metrics(gt_list, labels_list, pred_lists, scores_lists,
                   regions_list):
    gt = np.concatenate(gt_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    out = {}
    for v in VARIANTS:
        pr = np.concatenate(pred_lists[v], axis=0)
        img = image_metrics_from_scores(
            labels, np.concatenate(scores_lists[v], axis=0))
        anom_auroc, anom_ap = anomaly_only_metrics(labels, gt, pr)
        out[v] = {
            "P-AUROC": float(roc_auc_score(gt.ravel(), pr.ravel())),
            "P-AP": float(average_precision_score(gt.ravel(), pr.ravel())),
            "P-F1": float(f1_score_max(gt.ravel(), pr.ravel())),
            # Pooled P-AUPRO over 494 full-size maps is prohibitively slow;
            # per-class P-AUPRO and its class macro mean are reported.
            "P-AUPRO": None,
            "P-AUROC-anom": anom_auroc,
            "P-AP-anom": anom_ap,
            "I-AUROC": img["I-AUROC"], "I-AP": img["I-AP"],
            "I-F1": img["I-F1"],
            "n_images": int(len(labels)),
            "n_anomaly": int((labels == 1).sum()),
            "n_gt_positive_pixels": int(gt.sum()),
        }
    return out


# ---------------------------------------------------------------------------
# self-test (pure CPU; also the first assertion in every run)
# ---------------------------------------------------------------------------
def run_self_test():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        if not cond:
            raise AssertionError(msg)
        checks += 1

    import torch as _torch
    import torch.nn.functional as _F
    from sklearn.metrics import (average_precision_score as _ap,
                                 roc_auc_score as _roc)

    globals()["torch"] = _torch
    globals()["nn"] = _torch.nn
    globals()["F"] = _F
    globals()["average_precision_score"] = _ap
    globals()["roc_auc_score"] = _roc

    ok(abs(fov(BASE_CROP, BASE_RESIZE) - 0.875) < 1e-12, "base FOV")
    ok(abs(fov(HI_CROP, HI_RESIZE) - 0.875) < 1e-12, "hi FOV")
    ok(TARGET_TEST_TOTAL == 494, "four-class test total 494")
    ok(GRID == 48 and HI_CROP // 14 == 48, "48x48 token grid at 672")
    ok(BLOCK == 4 and (GRID // BLOCK) % 2 == 0, "12x12 blocks, even parity")
    ok(len(VARIANTS) == 5, "five variants")
    ok(set(TARGET_CLASSES) == {"leather", "capsule", "grid", "screw"},
       "target classes")

    # Exact four-group partition: every patch covered once, 25% each.
    masks = numpy_group_masks()
    ok(masks.shape == (N_GROUPS, PATCHES), "mask shape")
    ok(bool((masks.sum(axis=0) == 1).all()), "every patch covered once")
    ok(bool((masks.sum(axis=1) == PATCHES // N_GROUPS).all()),
       "each group covers 25%")
    ok(int(masks.sum()) == PATCHES, "total covered patches")
    grid_ids = group_index_grid()
    ok(int(grid_ids.min()) == 0 and int(grid_ids.max()) == 3, "group ids")
    for g in range(N_GROUPS):
        blocks = grid_ids.reshape(GRID // BLOCK, BLOCK,
                                  GRID // BLOCK, BLOCK)
        # block ids in group g must form a 6x6 parity checkerboard
        b_ids = blocks[:, 0, :, 0]
        ok(int((b_ids == g).sum()) == (GRID // BLOCK // 2) ** 2,
           f"group {g} block count")

    # Patch-error routing uses the exact coordinate of each group's pass.
    b, h, w = 2, GRID, GRID
    synthetic = torch.arange(1, N_GROUPS + 1, dtype=torch.float32)
    errors = synthetic.view(N_GROUPS, 1, 1, 1).expand(N_GROUPS, b, h, w)
    composed = compose_group_maps(errors)
    for g in range(N_GROUPS):
        sel = torch.as_tensor(grid_ids == g)
        ok(bool((composed[:, sel] == float(g + 1)).all()),
           f"compose group {g} coordinates")
    ok(bool((composed > 0).all()), "compose fills every position")

    # Pixel-mask expansion covers the same 4x4 patches.
    pmask = patch_mask_to_pixel_mask(
        torch.as_tensor(masks[0]).unsqueeze(0).expand(1, -1), size=HI_CROP)
    ok(pmask.shape == (1, 1, HI_CROP, HI_CROP), "pixel mask shape")
    ok(int(pmask.sum()) == (HI_CROP // 14) ** 2 * 14 * 14 // N_GROUPS,
       "pixel mask patch area")

    # Deterministic disjoint split.
    paths = [f"train/good/{i:04d}.png" for i in range(150)]
    tr, ca = split_train_calibration(paths, 96, 16, SEED)
    tr2, ca2 = split_train_calibration(paths, 96, 16, SEED)
    ok(tr == tr2 and ca == ca2, "split deterministic")
    ok(len(tr) == 96 and len(ca) == 16, "split sizes")
    ok(not (set(tr) & set(ca)), "split disjoint")
    ok(tr == sorted(tr) and ca == sorted(ca), "split sorted")

    order, groups = make_training_schedule(37, epochs=3)
    order2, groups2 = make_training_schedule(37, epochs=3)
    ok(np.array_equal(order, order2) and np.array_equal(groups, groups2),
       "schedule deterministic")
    ok(groups.shape == (3, 37) and groups.min() == 0
       and groups.max() == 3, "schedule groups")
    ok(all(sorted(order[e].tolist()) == list(range(37))
           for e in range(3)), "schedule permutations")

    # Calibration statistics and positive floor.
    vals = np.linspace(0.0, 1.0, 1001)
    st = calibration_stats(vals)
    ok(abs(st["median"] - 0.5) < 1e-12, "calibration median")
    ok(abs(st["q95"] - 0.95) < 1e-12, "calibration q95")
    ok(abs(st["scale"] - 0.45) < 1e-12, "calibration scale")
    flat = calibration_stats(np.zeros(100))
    ok(flat["scale"] == SCALE_FLOOR and flat["clamped"], "floor")
    nm = normalize_map(np.array([0.5, 0.95]), st)
    ok(abs(nm[0]) < 1e-12 and abs(nm[1] - 1.0) < 1e-12,
       "normalize median/q95")

    # Metrics: pooled AP, anomaly-only AP, per-image AP, macro pairing.
    rng = np.random.default_rng(123)
    gt = np.zeros((3, 8, 8), dtype=np.uint8)
    gt[1, 1:4, 1:4] = 1
    gt[2, 5:8, 5:8] = 1
    labels = np.array([0, 1, 1])
    good = rng.normal(0, 0.1, size=(3, 8, 8)) + gt * 1.5
    bad = -good
    gt_flat = gt.reshape(-1)
    ok(abs(average_precision_score(gt_flat, good.reshape(-1))
           - _ap(gt_flat, good.reshape(-1))) < 1e-12, "pooled AP")
    a_auc, a_ap = anomaly_only_metrics(labels, gt, good)
    ok(np.isfinite(a_auc) and np.isfinite(a_ap), "anomaly-only AP finite")
    pia = per_image_pixel_ap(gt, good, labels)
    ok(np.isnan(pia[0]), "normal image has no per-image AP")
    ok(np.isfinite(pia[1]) and np.isfinite(pia[2]), "valid anomaly AP")
    ok(float(np.nanmean(pia)) > 0.9, "planted anomaly per-image AP high")
    pia_bad = per_image_pixel_ap(gt, bad, labels)
    ok(float(np.nanmean(pia_bad)) < 0.9, "reversed map lowers AP")
    delta = np.nanmean(pia - pia_bad)
    ok(delta > 0, "paired per-image delta direction")

    recall_a, thr_a = fixed_fpr_recall(good.reshape(-1), gt_flat)
    recall_b, thr_b = fixed_fpr_recall(10.0 * good.reshape(-1) + 3,
                                       gt_flat)
    ok(all(abs(recall_a[k] - recall_b[k]) < 1e-12 for k in recall_a),
       "affine invariance of fixed-FPR recall")
    ok(all(abs(thr_a[k] - thr_b[k]) > 1e-9 for k in thr_a),
       "thresholds are method-specific")

    # Head / loss gradient check on synthetic features.
    _torch.manual_seed(0)
    head = MaskedFeatureHead()
    feats = _torch.randn(2, FEAT_GROUPS, FEAT_DIM, GRID, GRID)
    target = feats.detach().clone()
    m = _torch.as_tensor(masks[0]).unsqueeze(0).expand(2, -1)
    pred = head(feats)
    loss = masked_cosine_loss(pred, target, m)
    ok(bool(_torch.isfinite(loss)), "loss finite")
    loss.backward()
    ok(not target.requires_grad, "clean target detached")
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    ok(len(grads) > 0 and any(float(g.abs().sum()) > 0 for g in grads),
       "head receives gradient")
    emap = cosine_error_map(pred.detach(), target)
    ok(tuple(emap.shape) == (2, 1, GRID, GRID), "error map shape")

    # Strict JSON round trip.
    tmp = tempfile.mkdtemp(prefix="stage12_selftest_")
    p = os.path.join(tmp, "x.json")
    write_json_atomic(p, {"a": float("nan"), "b": np.float32(1.5),
                          "c": np.int64(3), "d": [np.bool_(True)]})
    d = read_json(p)
    ok(d["a"] is None and d["b"] == 1.5 and d["c"] == 3
       and d["d"] == [True], "strict JSON sanitization")

    ok(len(STAGE7_REFERENCE) == 4, "stage7 reference for four classes")
    for it in TARGET_CLASSES:
        ok(it in STAGE7_REFERENCE, f"stage7 reference {it}")
        ok(len(STAGE7_REFERENCE[it]) == 7, f"stage7 metrics {it}")

    print(f"SELF-TEST OK ({checks} checks)")
    return 0


# ---------------------------------------------------------------------------
# GPU smoke: assertions in the same notebook before any training
# ---------------------------------------------------------------------------
def run_gpu_smoke(model, device, classes, base_tf, hi_tf, gaussian, out_dir):
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        if not cond:
            raise AssertionError("GPU SMOKE: " + msg)
        checks += 1

    item = None
    probe = None
    for c in classes:
        root = os.path.join(DATA, c)
        ds = MultiScaleTestDataset(root, base_tf, None, hi_tf)
        # gt_transform None is not used until __getitem__; pass a real one.
        if len(ds) > 0:
            item, probe = c, ds
            break
    if probe is None:
        raise RuntimeError("GPU smoke: no test image found")
    # Rebuild with a real gt transform for the selected sample.
    gt_tf = get_data_transforms(BASE_RESIZE, BASE_CROP)[1]
    ds = MultiScaleTestDataset(os.path.join(DATA, item), base_tf, gt_tf,
                               hi_tf)
    img, gt, label, path, hi = ds[0]
    x_base = img.unsqueeze(0).to(device)
    x_hi = hi.unsqueeze(0).to(device)
    log(f"GPU smoke image={path}")

    with torch.no_grad():
        tokens = encoder_input_tokens(model.encoder, x_hi, masks=None)
        n_tok = tokens.shape[1]
        reg = int(getattr(model.encoder, "num_register_tokens", 0))
        ok(n_tok == 1 + reg + PATCHES,
           f"672 token count {n_tok} == 1+{reg}+{PATCHES}")
        ok(int(math.sqrt(n_tok - 1 - reg)) == GRID, "token side 48")
        ok(torch.equal(torch_group_masks(device).sum(0),
                       torch.ones(PATCHES, dtype=torch.long,
                                 device=device)), "mask coverage once")
        ok(torch.equal(torch_group_masks(device).sum(1),
                       torch.full((N_GROUPS,), PATCHES // N_GROUPS,
                                 dtype=torch.long, device=device)),
           "mask 25% per group")

        # Masking happens at token entrance: changing pixels only inside the
        # masked 4x4 blocks must not change the token input or encoder output.
        m0 = masks_for_groups(device, torch.zeros(1, dtype=torch.long,
                                                  device=device))
        pmask = patch_mask_to_pixel_mask(m0, HI_CROP)
        x_pert = x_hi.clone()
        x_pert = torch.where(pmask.expand_as(x_pert),
                             torch.full_like(x_pert, 0.37), x_pert)
        tok_a = encoder_input_tokens(model.encoder, x_hi, masks=m0)
        tok_b = encoder_input_tokens(model.encoder, x_pert, masks=m0)
        max_tok = float((tok_a - tok_b).abs().max())
        ok(max_tok < 1e-6, f"preattention token invariance ({max_tok:.2e})")
        feat_a = encode_fused(model.encoder, x_hi, masks=m0)
        feat_b = encode_fused(model.encoder, x_pert, masks=m0)
        max_feat = float((feat_a - feat_b).abs().max())
        ok(max_feat < 1e-6,
           f"masked encoder invariance ({max_feat:.2e})")

        # Combined forward must reproduce the exact stage-7 hi map and the
        # separate clean encoder features.
        hi_combined, clean_combined = vitill_forward_with_clean(model, x_hi)
        hi_direct = global_map(model, x_hi)
        clean_direct = encode_fused(model.encoder, x_hi, masks=None)
        d_hi = float((hi_combined - hi_direct).abs().max())
        d_clean = float((clean_combined - clean_direct).abs().max())
        ok(d_hi < 1e-5, f"combined hi map identity ({d_hi:.2e})")
        ok(d_clean < 1e-5, f"clean feature identity ({d_clean:.2e})")

        base_raw = global_map(model, x_base)
        base = postprocess(base_raw, gaussian)
        hi_map = postprocess(hi_combined, gaussian)
        b0 = guided_filter(guide_from_hi(x_hi),
                           0.5 * base + 0.5 * hi_map)
        b0_ref = stage7_reference_b0(model, x_base, x_hi, gaussian)
        d_b0 = float((b0 - b0_ref).abs().max())
        ok(d_b0 < 1e-5, f"faithful stage-7 B0 ({d_b0:.2e})")

        # Four-pass M1 composition is finite and uses every position.
        heads = {"c1": MaskedFeatureHead().to(device),
                 "m1": MaskedFeatureHead().to(device)}
        raw = infer_batch(model, heads, x_base, x_hi, gaussian, device)
        ok(tuple(raw["m1"].shape) == (1, 1, 256, 256), "M1 map shape")
        ok(bool(torch.isfinite(raw["b0"]).all()), "B0 finite")
        ok(bool(torch.isfinite(raw["c1"]).all()), "C1 finite")
        ok(bool(torch.isfinite(raw["m1"]).all()), "M1 finite")

        # Gradient: target detached, head receives gradient.  The smoke body
        # is under no_grad for the frozen encoder; enable grad around the
        # head-only check.
        with torch.enable_grad():
            feats = clean_combined.detach()
            target = feats.clone()
            m = masks_for_groups(device, torch.zeros(1, dtype=torch.long,
                                                     device=device))
            head = MaskedFeatureHead().to(device)
            pred = head(feats)
            loss = masked_cosine_loss(pred, target, m)
            ok(bool(torch.isfinite(loss)), "smoke loss finite")
            loss.backward()
            ok(not target.requires_grad, "smoke target detached")
            ok(any(p.grad is not None and float(p.grad.abs().sum()) > 0
                   for p in head.parameters()), "smoke head gradient")
            ok(tuple(cosine_error_map(pred.detach(), target).shape)
               == (1, 1, GRID, GRID), "smoke error map shape")

        # Actual short-run timings (synchronized) for the baseline and each
        # masked pass.  Reported once; the per-class evaluation timer is
        # separate.
        sync()
        t = time.time()
        _ = global_map(model, x_base)
        sync()
        t_base = 1000.0 * (time.time() - t)
        sync()
        t = time.time()
        _ = vitill_forward_with_clean(model, x_hi)
        sync()
        t_hi = 1000.0 * (time.time() - t)
        sync()
        t = time.time()
        for g in range(N_GROUPS):
            mg = masks_for_groups(device, torch.full(
                (1,), g, dtype=torch.long, device=device))
            _ = encode_fused(model.encoder, x_hi, masks=mg)
        sync()
        t_masked = 1000.0 * (time.time() - t)
        sync()
        t = time.time()
        _ = heads["m1"](clean_combined)
        sync()
        t_head = 1000.0 * (time.time() - t)

    payload = {"ok": True, "checks": checks, "item": item, "image": path,
               "preattention_max_token_delta": max_tok,
               "masked_encoder_max_delta": max_feat,
               "combined_hi_max_delta": d_hi,
               "clean_feature_max_delta": d_clean,
               "b0_max_delta_vs_stage7_code": d_b0,
               "timing_ms_batch1": {
                   "base_392": t_base,
                   "hi_672_combined_forward": t_hi,
                   "four_masked_encoder_672": t_masked,
                   "head_forward": t_head,
                   "masked_per_pass": t_masked / float(N_GROUPS),
               }}
    os.makedirs(out_dir, exist_ok=True)
    write_json_atomic(os.path.join(out_dir, "gpu_smoke.json"), payload)
    log(f"GPU SMOKE OK ({checks} checks)")
    return payload


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
BASE_BATCH_EFF = EVAL_BATCH
HI_BATCH_EFF = HI_BATCH
HI_TF_EFF = None
DEVICE = "cpu"


def resolve_out_dir(args):
    if args.out:
        return args.out
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working"
    return "E:/dinomaly-mvp/kaggle_runs/stage12-local"


def main():
    global LOG_PATH, FP, EVAL_BATCH, HI_BATCH, TRAIN_BATCH
    global BASE_BATCH_EFF, HI_BATCH_EFF, HI_TF_EFF, DEVICE, SMOKE
    global TRAIN_PER_CLASS_EFF, CALIB_PER_CLASS_EFF, EPOCHS_EFF, LIMIT_TEST
    global OUT

    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="assertions + tiny end-to-end run, then exit")
    ap.add_argument("--skip-smoke", action="store_true",
                    help="skip GPU assertions (reruns only)")
    ap.add_argument("--classes", type=str, default=",".join(TARGET_CLASSES))
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    ap.add_argument("--train-per-class", type=int, default=TRAIN_PER_CLASS)
    ap.add_argument("--calib-per-class", type=int, default=CALIB_PER_CLASS)
    ap.add_argument("--eval-batch", type=int, default=EVAL_BATCH)
    ap.add_argument("--train-batch", type=int, default=TRAIN_BATCH)
    ap.add_argument("--limit-test", type=int, default=0,
                    help="smoke only: max test images per class")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()

    OUT = resolve_out_dir(args)
    os.makedirs(OUT, exist_ok=True)
    LOG_PATH = os.path.join(OUT, "stage12.log")
    SMOKE = bool(args.smoke)

    EVAL_BATCH = max(1, int(args.eval_batch))
    HI_BATCH = max(1, min(EVAL_BATCH, 2))
    BASE_BATCH_EFF = EVAL_BATCH
    HI_BATCH_EFF = HI_BATCH
    TRAIN_BATCH = max(1, int(args.train_batch))
    TRAIN_PER_CLASS_EFF = max(1, int(args.train_per_class))
    CALIB_PER_CLASS_EFF = max(1, int(args.calib_per_class))
    EPOCHS_EFF = max(1, int(args.epochs))
    LIMIT_TEST = max(0, int(args.limit_test))
    if SMOKE:
        # Tiny end-to-end local/Kaggle validation; never a result run.
        TRAIN_PER_CLASS_EFF = min(TRAIN_PER_CLASS_EFF, 4)
        CALIB_PER_CLASS_EFF = min(CALIB_PER_CLASS_EFF, 2)
        EPOCHS_EFF = min(EPOCHS_EFF, 1)
        LIMIT_TEST = LIMIT_TEST or 4

    log(f"=== {SLUG} start ===")
    log(f"args={vars(args)} OUT={OUT} smoke={SMOKE}")

    run_self_test()
    log("self-test passed")

    prepare_environment()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE = device
    log(f"device={device} torch={torch.__version__}")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    find_checkpoints()
    base_sha = sha256_file(BASE_CKPT)
    log(f"base sha256={base_sha}")
    if base_sha != "f5689b76aafa5c9d2b36a0df8e87ef7bdc82075a042c76bcf7933e0edd637218":
        log("WARNING: checkpoint sha256 differs from archived model_10000.pt")

    all_classes = sorted(
        d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d)))
    requested = [c.strip() for c in args.classes.split(",") if c.strip()]
    missing = [c for c in requested if c not in all_classes]
    if missing:
        raise RuntimeError(f"requested classes missing from DATA: {missing}; "
                           f"available={all_classes}")
    item_list = requested

    model = build_model(device)
    st_base = load_ckpt(model, BASE_CKPT, device)
    model.eval()
    log(f"checkpoint iter={st_base.get('iter')} kind={st_base.get('kind')}")

    base_tf, gt_tf = get_data_transforms(BASE_RESIZE, BASE_CROP)
    hi_tf = get_data_transforms(HI_RESIZE, HI_CROP)[0]
    HI_TF_EFF = hi_tf
    gaussian = get_gaussian_kernel(kernel_size=GAUSSIAN_KERNEL,
                                   sigma=GAUSSIAN_SIGMA).to(device)

    if not args.skip_smoke:
        run_gpu_smoke(model, device, item_list, base_tf, hi_tf, gaussian,
                      OUT)
        if SMOKE:
            log("smoke assertions complete; continuing tiny end-to-end run")

    # Data split (train/good only, disjoint calibration).
    splits = {}
    for item in item_list:
        paths = train_good_paths(os.path.join(DATA, item))
        train_p, calib_p = split_train_calibration(
            paths, TRAIN_PER_CLASS_EFF, CALIB_PER_CLASS_EFF, SEED)
        splits[item] = {
            "train": train_p, "calib": calib_p,
            "n_available": len(paths),
            "train_sha256": sha256_text_list(train_p),
            "calib_sha256": sha256_text_list(calib_p),
        }
        log(f"{item}: train/good available={len(paths)} "
            f"train={len(train_p)} calib={len(calib_p)} "
            f"overlap={len(set(train_p) & set(calib_p))}")

    # Test sets and count assertions.
    test_sets = {}
    counts = {}
    for item in item_list:
        ds = MultiScaleTestDataset(os.path.join(DATA, item), base_tf,
                                   gt_tf, hi_tf)
        if LIMIT_TEST:
            ds.base.img_paths = ds.base.img_paths[:LIMIT_TEST]
            ds.base.gt_paths = ds.base.gt_paths[:LIMIT_TEST]
            ds.base.labels = ds.base.labels[:LIMIT_TEST]
            ds.base.types = ds.base.types[:LIMIT_TEST]
        test_sets[item] = ds
        counts[item] = len(ds)
    total_test = int(sum(counts.values()))
    log(f"test images={total_test} per-class={counts}")
    if not SMOKE and not LIMIT_TEST:
        for item in item_list:
            if item in EXPECTED_TEST_IMAGES and counts[item] != \
                    EXPECTED_TEST_IMAGES[item]:
                raise RuntimeError(
                    f"expected {EXPECTED_TEST_IMAGES[item]} test images for "
                    f"{item}, got {counts[item]}")
        if set(item_list) == set(TARGET_CLASSES) and total_test != 494:
            raise RuntimeError(f"expected 494 four-class test images, "
                               f"got {total_test}")

    # Semantic config / fingerprint.
    config = {
        "slug": SLUG, "stage": STAGE,
        "script": os.path.basename(os.path.abspath(__file__)),
        "code_sha256": sha256_file(os.path.abspath(__file__)),
        "baseline": {"resize": BASE_RESIZE, "crop": BASE_CROP,
                     "fov": fov(BASE_CROP, BASE_RESIZE)},
        "high_res": {"resize": HI_RESIZE, "crop": HI_CROP,
                     "fov": fov(HI_CROP, HI_RESIZE),
                     "token_grid": [GRID, GRID], "patch_size": 14},
        "masking": {
            "block": BLOCK, "block_grid": [GRID // BLOCK, GRID // BLOCK],
            "groups": N_GROUPS, "group_rule": "(block_row%2)*2 + block_col%2",
            "coverage": "exactly once per patch; 25% per group",
            "entrance": "encoder.prepare_tokens(x, masks=...) before attention",
        },
        "variants": list(VARIANTS),
        "fusion": {"b0_c1": "0.75*b0_norm + 0.25*c1_norm",
                   "b0_m1": "0.75*b0_norm + 0.25*m1_norm",
                   "scale": "per class/method max(p95-median, 0.005)"},
        "guide": {"radius": GUIDE_RADIUS, "epsilon": GUIDE_EPS,
                  "gaussian": [GAUSSIAN_KERNEL, GAUSSIAN_SIGMA]},
        "training": {
            "source": "MVTec train/good only",
            "classes": item_list,
            "train_per_class": TRAIN_PER_CLASS_EFF,
            "calib_per_class": CALIB_PER_CLASS_EFF,
            "seed": SEED, "epochs": EPOCHS_EFF, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "batch": TRAIN_BATCH,
            "optimizer": "AdamW", "schedule": "constant lr",
            "head": "Conv3x3(1536->256)-GroupNorm-GELU-"
                    "Conv3x3(256->256)-GroupNorm-GELU-"
                    "Conv1x1(256->1536, groups=2)",
            "head_parameters": int(sum(
                p.numel() for p in MaskedFeatureHead().parameters())),
            "loss": "per-group mean cosine distance at masked positions; "
                    "equal group weights; clean target detached",
            "c1": "clean input, same head/loss positions, "
                  "test error at all positions",
            "m1": "masked input at token entrance, same head/loss positions, "
                  "test only at each position's masked pass",
            "same_initialization": True,
            "same_updates_per_head": True,
            "calibration_disjoint_from_training": True,
            "test_images_or_labels_used_for_fit": False,
            "external_data": False,
        },
        "metrics": {
            "keys": list(METRIC_KEYS), "recall": list(RECALL_KEYS),
            "evaluation_grid": 256,
            "mask_source": "exact original 448->392 gt transform, nearest "
                           "to 256",
            "primary": "full-test pooled pixel AP (P-AP); class macro mean",
            "anomaly_only_pooled": "P-AP-anom over anomaly test images",
            "per_image": "P-AP-image: mean within class of per-image AP on "
                         "valid anomaly images, then class macro",
        },
        "entry_condition_frozen": {
            "min_P-AP_gain_vs_b0": 0.01,
            "must_exceed": "b0_c1",
            "min_nonnegative_class_count": 3,
            "max_allowed_class_drop": 0.02,
        },
        "classes_total": len(item_list),
        "eval_classes": item_list,
        "test_images_total": total_test,
        "test_counts": counts,
        "partial": SMOKE or bool(LIMIT_TEST),
        "base_checkpoint": BASE_CKPT,
        "base_sha256": base_sha,
        "stage1_metrics": STAGE1_METRICS,
        "deviations": [
            "Masked features are recomputed on the fly once per image per "
            "epoch instead of cached: four masked patterns per image would "
            "need ~13.6 GB float16. Clean targets are cached as float16 .npy "
            "and reused as C1 input, which is memory bounded.",
            "C1 is scored at all 48x48 positions from one clean pass; M1 is "
            "scored only at each position's own masked pass, as specified.",
            "The original 448->392 top-1% image score is reported separately; "
            "per-variant image metrics use each variant's own map top-1%.",
            "Two condition-specific heads with identical architecture and "
            "identical initialization are used rather than one head updated "
            "by both conditions; joint updates would confound C1 and M1. "
            "Within each condition the head is shared across both feature "
            "groups.",
            "AdamW with weight_decay=0.0 is used; the plan fixed only seed, "
            "epochs, learning rate and masks, so no extra regularization was "
            "introduced.",
        ],
    }
    FP = config_fingerprint(config)
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
        return d if d.get("fingerprint") == FP else None

    done = []
    if not args.fresh:
        for item in item_list:
            if load_per_class(item) is not None:
                done.append(item)
    log(f"resume: {len(done)}/{len(item_list)} classes complete")

    def update_progress(current=None, latest=None):
        write_json_atomic(progress_path, {
            "slug": SLUG, "fingerprint": FP,
            "classes_total": len(item_list), "classes_done": done,
            "current": current, "latest": latest,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S")})

    set_status("running", classes_done=done,
               classes_total=len(item_list), current=None)
    update_progress()

    # 1) clean target cache (also C1 input).  Always built over the complete
    # protocol class set, never just the classes still missing, so a resumed
    # run cannot silently train a different head than the one reported.
    all_done = len(done) == len(item_list)
    cache_entries = []
    cache_seconds = 0.0
    heads = {}
    history = {}
    head_dir = os.path.join(OUT, "heads")
    heads_meta_path = os.path.join(head_dir, "meta.json")
    if not all_done:
        cache_root = os.path.join(OUT, "cache_clean")
        t_cache = time.time()
        cache_entries, _cache_meta = build_clean_cache(
            model, item_list, splits, hi_tf, device, cache_root,
            force=args.fresh)
        cache_seconds = float(time.time() - t_cache)

    # 2) train heads (same schedule/updates, all protocol classes).
    reuse_heads = False
    if not all_done and not args.fresh and os.path.isfile(heads_meta_path):
        try:
            hm = read_json(heads_meta_path)
            reuse_heads = (hm.get("fingerprint") == FP
                           and hm.get("classes") == item_list
                           and all(os.path.isfile(os.path.join(
                               head_dir, f"{name}.pt"))
                               for name in ("c1", "m1")))
        except Exception:
            reuse_heads = False
    if not all_done:
        if reuse_heads:
            log("reusing trained heads")
            for name in ("c1", "m1"):
                h = MaskedFeatureHead().to(device)
                h.load_state_dict(torch.load(
                    os.path.join(head_dir, f"{name}.pt"),
                    map_location=device))
                h.eval()
                heads[name] = h
            if os.path.isfile(os.path.join(head_dir, "history.json")):
                history = read_json(os.path.join(head_dir, "history.json"))
        else:
            schedule = make_training_schedule(len(cache_entries),
                                              EPOCHS_EFF, SEED)
            log(f"training heads: images={len(cache_entries)} "
                f"epochs={EPOCHS_EFF} batch={TRAIN_BATCH} lr={LR} "
                f"updates/head={EPOCHS_EFF * math.ceil(len(cache_entries) / TRAIN_BATCH)}")
            reset_peak()
            head_c1, head_m1, hist = train_heads(
                model, cache_entries, schedule, device)
            heads["c1"] = head_c1.eval()
            heads["m1"] = head_m1.eval()
            history = hist
            history["gpu_peak_mib"] = peak_mib()
            history["schedule_sha256"] = (
                sha256_array(schedule[0].astype(np.int64)),
                sha256_array(schedule[1].astype(np.int64)))
            os.makedirs(head_dir, exist_ok=True)
            for name in ("c1", "m1"):
                p = os.path.join(head_dir, f"{name}.pt")
                torch.save(heads[name].state_dict(), p)
                history[f"{name}_sha256"] = sha256_file(p)
            write_json_atomic(os.path.join(head_dir, "history.json"),
                              history)
            write_json_atomic(heads_meta_path, {
                "fingerprint": FP, "classes": item_list,
                "epochs": EPOCHS_EFF, "train_batch": TRAIN_BATCH,
                "lr": LR, "updates_per_head":
                    history["updates_per_head"],
                "config": config["training"],
            })
            log(f"heads saved: c1={history.get('c1_sha256')} "
                f"m1={history.get('m1_sha256')}")
            del schedule
    else:
        log("all classes already evaluated; skipping cache/training")
        # Prefer the saved history, if present, for the report.
        hp = os.path.join(head_dir, "history.json")
        if os.path.isfile(hp):
            try:
                history = read_json(hp)
            except Exception:
                history = {}

    # 3) calibration (train/good only, disjoint).
    calibration = {}
    calibration_seconds = 0.0
    if not all_done:
        calibration = compute_calibration(
            model, heads, item_list, splits, base_tf, hi_tf, gaussian,
            device)
        calibration_seconds = calibration.pop("_seconds", 0.0)
    else:
        for item in item_list:
            d = load_per_class(item)
            calibration[item] = {
                "n_images": len(splits[item]["calib"]),
                "paths_sha256": sha256_text_list(splits[item]["calib"]),
                "stats": d["calibration"],
            }

    # 4) evaluate classes.
    pooled_arrays = {}
    for item in item_list:
        npz_path = os.path.join(per_class_dir, item + "_preds.npz")
        if item in set(done):
            log(f"{item}: resume skip")
            if os.path.isfile(npz_path):
                pooled_arrays[item] = load_pred_npz(npz_path)
            continue
        set_status("running", classes_done=done,
                   classes_total=len(item_list), current=item)
        update_progress(current=item)
        log(f"{item}: evaluating {counts[item]} test images")
        stats = calibration[item]["stats"]
        result, run_arrays = evaluate_class(
            item, test_sets[item], model, heads, gaussian, stats, device)
        save_pred_npz(npz_path, run_arrays)
        pooled_arrays[item] = run_arrays

        # Examples: success / high-score normal false positive / miss.
        rows = result["per_image"]
        valid_anom = [r for r in rows
                      if r["variant"] == "b0_m1" and r["label"] == 1
                      and r["pixel_ap"] is not None]
        normals = [r for r in rows
                   if r["variant"] == "b0_m1" and r["label"] == 0]
        selected = []
        if valid_anom:
            best = max(valid_anom, key=lambda r: (r["pixel_ap"],
                                                  r["image_score"]))
            selected.append(("success", best["index"]))
        if normals:
            fp = max(normals, key=lambda r: r["image_score"])
            selected.append(("false_positive", fp["index"]))
        if valid_anom:
            miss = min(valid_anom, key=lambda r: r["pixel_ap"])
            selected.append(("miss", miss["index"]))
        try:
            examples = render_examples(
                item, test_sets[item], selected, model, heads, gaussian,
                stats, device, os.path.join(OUT, "examples"))
        except Exception as exc:
            log(f"{item}: example rendering failed: {exc}")
            examples = []
        result["examples"] = examples

        write_json_atomic(per_class_path(item), result)
        done.append(item)
        latest = {
            "class": item, "seconds": result["seconds"],
            "P-AP": {v: result["variants"][v]["metrics"]["P-AP"]
                     for v in VARIANTS},
            "P-AP-anom": {v: result["variants"][v]["metrics"]["P-AP-anom"]
                          for v in VARIANTS},
            "P-AP-image": {v: result["variants"][v]["metrics"]["P-AP-image"]
                           for v in VARIANTS},
        }
        update_progress(current=None, latest=latest)
        set_status("running", classes_done=done,
                   classes_total=len(item_list), current=None,
                   latest=latest)
        log(f"{item}: done in {result['seconds']:.1f}s "
            f"peak={result['gpu_peak_mib']:.0f}MiB " + " ".join(
                f"{v}:AP={result['variants'][v]['metrics']['P-AP']:.4f}"
                f"/imgAP={result['variants'][v]['metrics']['P-AP-image']:.4f}"
                for v in VARIANTS))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 5) aggregate from per-class JSON (always available, including resume).
    per_class = {}
    for item in item_list:
        d = load_per_class(item)
        if d is None:
            raise RuntimeError(f"missing per-class result for {item}")
        per_class[item] = d

    agg = aggregate(per_class, item_list, STAGE7_REFERENCE)

    pooled = None
    if len(pooled_arrays) == len(item_list):
        pooled = pooled_metrics(
            [pooled_arrays[it]["gt"] for it in item_list],
            [pooled_arrays[it]["labels"] for it in item_list],
            {v: [pooled_arrays[it]["preds"][v] for it in item_list]
             for v in VARIANTS},
            {v: [pooled_arrays[it]["scores"][v] for it in item_list]
             for v in VARIANTS},
            None,
        )
        pooled["_note"] = ("pooled pixel AP across all "
                           f"{sum(counts.values())} test images; pooled "
                           "P-AUPRO is omitted because the full-size region "
                           "computation is prohibitive, per-class P-AUPRO is "
                           "reported")
    else:
        log("WARNING: pooled metrics unavailable because prediction arrays "
            "for some resumed classes are missing")

    report = {
        "experiment": {
            "slug": SLUG, "stage": STAGE,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "smoke": SMOKE, "partial": SMOKE or bool(LIMIT_TEST),
            "hypothesis": "masked local feature prediction from surrounding "
                          "context may localize defects better than clean "
                          "reconstruction at the same 48x48 grid",
            "plan": "experiment/MASKED_LOCAL_RECONSTRUCTION_PLAN.md",
        },
        "config": config,
        "checkpoints": {
            "base": {"path": BASE_CKPT, "sha256": base_sha,
                     "iter": (int(st_base["iter"])
                              if st_base.get("iter") is not None else None),
                     "kind": st_base.get("kind"),
                     "reused_unchanged": True},
        },
        "identity": {
            "b0_is_stage7_guide_fuse672": True,
            "b0_geometry": "original -> Resize(768) -> CenterCrop(672) and "
                           "original -> Resize(448) -> CenterCrop(392); both "
                           "to 256 with bilinear+gaussian(5,4)",
            "mask_geometry": "48x48 grid; 4x4 blocks; four parity groups; "
                             "each patch exactly once; 25% per group",
            "masking_entrance": "encoder token embedding before any "
                                "attention block (prepare_tokens masks=)",
            "clean_target_detached": True,
            "encoder_frozen": True,
            "head_only_training": True,
            "test_labels_used_for_selection_or_fit": False,
            "calibration_source":
                f"{CALIB_PER_CLASS_EFF} disjoint train/good images per class",
            "b0_identity_vs_stage7": agg["b0_identity_vs_stage7"],
        },
        "training": {
            "history": history,
            "split": {it: {
                "n_available": splits[it]["n_available"],
                "n_train": len(splits[it]["train"]),
                "n_calib": len(splits[it]["calib"]),
                "train_sha256": splits[it]["train_sha256"],
                "calib_sha256": splits[it]["calib_sha256"],
                "overlap": len(set(splits[it]["train"])
                               & set(splits[it]["calib"])),
                "train_paths": splits[it]["train"],
                "calib_paths": splits[it]["calib"],
            } for it in item_list},
            "seconds": history.get("seconds"),
        },
        "calibration": {
            "seconds": calibration_seconds,
            "per_class": {it: calibration[it] for it in item_list},
        },
        "results": {
            "per_class": per_class,
            "mean": agg["mean"],
            "image_mean": agg["image_mean"],
            "image_mean_original_base": agg["image_mean_original_base"],
            "deltas": agg["deltas"],
            "paired_image_ap_deltas": agg["paired_image_ap_deltas"],
            "entry_condition": agg["entry_condition"],
            "pooled": pooled,
        },
        "counts": {
            "classes": len(item_list),
            "test_images": int(sum(per_class[it]["n_test"]
                                   for it in item_list)),
            "anomaly_images": int(sum(per_class[it]["n_anomaly"]
                                      for it in item_list)),
            "normal_images": int(sum(per_class[it]["n_normal"]
                                     for it in item_list)),
            "per_class": {it: {"n_test": per_class[it]["n_test"],
                               "n_anomaly": per_class[it]["n_anomaly"],
                               "n_normal": per_class[it]["n_normal"],
                               "n_valid_anomaly_images":
                                   per_class[it]["n_valid_anomaly_images"]}
                          for it in item_list},
        },
        "runtime": {
            "device": device,
            "torch": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "eval_batch": EVAL_BATCH, "hi_batch": HI_BATCH,
            "train_batch": TRAIN_BATCH,
            "classes_evaluated": len(item_list),
            "test_images_evaluated": int(sum(per_class[it]["n_test"]
                                             for it in item_list)),
            "forwards_per_test_image":
                "1 base(392) + 1 full hi(672) + 4 masked encoder(672)",
            "training_seconds": history.get("seconds"),
            "cache_seconds": cache_seconds,
            "calibration_seconds": calibration_seconds,
            "evaluation_seconds_total": float(sum(
                per_class[it]["seconds"] for it in item_list)),
            "gpu_peak_mib_max": max(
                (per_class[it]["gpu_peak_mib"] for it in item_list),
                default=0.0),
            "training_gpu_peak_mib": history.get("gpu_peak_mib"),
        },
        "files": {"per_class_dir": per_class_dir,
                  "progress": progress_path, "status": status_path,
                  "examples_dir": os.path.join(OUT, "examples")},
    }
    report_json = os.path.join(OUT, "masked_recon_report.json")
    write_json_atomic(report_json, report)
    set_status("complete", classes_done=done,
               classes_total=len(item_list), report=report_json,
               markdown=None)
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
