"""Stage 11: synthetic-defect refinement head on frozen Dinomaly maps.

Frozen 10k checkpoint and the exact stage-7 maps/geometry:
    original -> Resize(448) -> CenterCrop(392) -> Normalize -> 392 map
    original -> Resize(768) -> CenterCrop(672) -> Normalize -> 672 map
    both maps -> bilinear 256 -> gaussian(5,4)
    v7 reference = guided filter on 0.5 * base_global + 0.5 * hi672
    official masks = exact 448->392 transform, nearest to 256
    image score = original base_global top 1%, unchanged for every variant

A tiny shared 2-layer 3x3 conv head is fitted ONLY on train/good images from
the 15 MVTec classes.  For 16 deterministic good images per class we cache
the compact maps of an unmodified image and one synthetic local defect
(irregular elliptic mask, moved/replaced/contrast patch).  Thus 240 normal
and 240 synthetic training images, no external data and no test image or
label in fitting.  Loss is pixel BCE (balanced pos_weight) + soft Dice.

Evaluation variants (at most three):
    v7_guide       exact stage-7 guided-filter output
    learned_head   pure synthetic-trained head probability
    blend_guarded  v7 guide + fixed guarded residual toward the head

No markdown is produced.  Logs, strict JSON report and per-class JSON are
written atomically.  Runtime imports are delayed so --self-test is local.
"""
import os
import sys
import glob
import json
import math
import time
import argparse
import hashlib
import traceback

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

OUT = 'E:/dinomaly-mvp/kaggle_runs/stage11-local'
SLUG = "eunectesqaq/dinomaly-synthetic-refine-v11"
BASE_RESIZE = 448
BASE_CROP = 392
HI_RESIZE = 768
HI_CROP = 672
EXPECTED_TEST_IMAGES = 322
EVAL_BATCH = 4
HI_BATCH = 2
NUM_TRAIN_IMAGES = 16
TRAIN_EPOCHS = 20
TRAIN_BATCH = 8
LR = 1e-3
SEED = 1105
FPRS = (0.001, 0.01, 0.05)
METRIC_KEYS = ("I-AUROC", "I-AP", "I-F1",
               "P-AUROC", "P-AP", "P-F1", "P-AUPRO")
RECALL_KEYS = tuple(f"R@FPR{f:g}" for f in FPRS)
GUIDE_RADIUS = 7
GUIDE_EPS = 0.0025
BLEND_ALPHA = 0.5
GUARD_CAP = 0.25
N_FEATURES = 10
VARIANTS = ("v7_guide", "learned_head", "blend_guarded")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LOG_PATH = None
FP = None
CODE = None
DATA = None
BASE_CKPT = None
STAGE1_METRICS = None

F = DataLoader = measure = None
roc_auc_score = average_precision_score = auc = None
get_data_transforms = MVTecDataset = ViTill = vit_encoder = None
VitBlock = bMlp = LinearAttention2 = None
cal_anomaly_maps = get_gaussian_kernel = f1_score_max = None
partial = None


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


def fingerprint(cfg):
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


def finite_mean(values):
    vals = [float(v) for v in values
            if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


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


def refine_heatmap(heatmap, guide):
    box2 = _box_mean(_box_mean(heatmap, GUIDE_RADIUS), GUIDE_RADIUS)
    return box2, guided_filter(guide, heatmap)


def fixed_fpr_recall(pr_flat, gt_flat, fprs=FPRS):
    pr_flat = np.asarray(pr_flat).reshape(-1)
    gt_flat = np.asarray(gt_flat).reshape(-1)
    neg, pos = pr_flat[gt_flat == 0], pr_flat[gt_flat == 1]
    recall, thresholds = {}, {}
    if neg.size == 0 or pos.size == 0:
        return recall, thresholds
    for fpr in fprs:
        thr = float(np.quantile(neg, 1.0 - fpr))
        recall[f"R@FPR{fpr:g}"] = float((pos >= thr).mean())
        thresholds[f"T@FPR{fpr:g}"] = thr
    return recall, thresholds


# ---------------------------------------------------------------------------
# pure synthetic training data
# ---------------------------------------------------------------------------
def elliptical_mask(h, w, rng):
    """Irregular ellipse, scale varied, kept away from image borders."""
    base = float(rng.uniform(0.075, 0.19)) * min(h, w)
    ax = max(10.0, base * float(rng.uniform(0.70, 1.45)))
    ay = max(10.0, base * float(rng.uniform(0.55, 1.25)))
    lo_x = max(1.30 * ax + 4.0, 0.08 * w)
    hi_x = min(w - 1.30 * ax - 4.0, 0.92 * w)
    lo_y = max(1.30 * ay + 4.0, 0.08 * h)
    hi_y = min(h - 1.30 * ay - 4.0, 0.92 * h)
    cx = (float(rng.uniform(lo_x, hi_x)) if hi_x > lo_x else w * 0.5)
    cy = (float(rng.uniform(lo_y, hi_y)) if hi_y > lo_y else h * 0.5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xr, yr = xx - cx, yy - cy
    theta = np.arctan2(yr, xr)
    phase1, phase2 = rng.uniform(0, 2 * np.pi, size=2)
    radial = (1.0 + 0.16 * np.sin(3.0 * theta + phase1)
              + 0.085 * np.sin(5.0 * theta + phase2))
    rr = (xr / ax) ** 2 + (yr / ay) ** 2
    mask = rr <= radial ** 2
    mask[:2] = False
    mask[-2:] = False
    mask[:, :2] = False
    mask[:, -2:] = False
    if mask.sum() < max(20, int(0.0015 * h * w)):
        rr0 = (xr / max(ax * 0.7, 4.0)) ** 2 + \
              (yr / max(ay * 0.7, 4.0)) ** 2
        mask = rr0 <= 1.0
        mask[:2] = False
        mask[-2:] = False
        mask[:, :2] = False
        mask[:, -2:] = False
    return mask, (cx, cy, ax, ay)


def _shift_reflect(img, dy, dx):
    h, w = img.shape[:2]
    pad = int(max(abs(dy), abs(dx)) + 2)
    p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    y0, x0 = pad + int(dy), pad + int(dx)
    return p[y0:y0 + h, x0:x0 + w].copy()


def synthesize(pil, seed):
    """Local moved/replaced/contrast defect + native-resolution mask."""
    rng = np.random.default_rng(int(seed))
    img = np.asarray(pil.convert("RGB")).copy()
    h, w = img.shape[:2]
    mask, (cx, cy, ax, ay) = elliptical_mask(h, w, rng)
    span = max(ax, ay)
    angle = rng.uniform(0, 2 * np.pi)
    dist = float(rng.uniform(0.55, 1.05)) * span
    dy = int(round(float(np.sin(angle)) * dist))
    dx = int(round(float(np.cos(angle)) * dist))
    shifted = _shift_reflect(img, dy, dx)
    gain = float(rng.uniform(0.62, 1.38))
    bias = float(rng.uniform(-26.0, 26.0))
    src = (shifted.astype(np.float32) * gain + bias).clip(0, 255)
    out = img.copy()
    out[mask] = src[mask].astype(np.uint8)
    return Image.fromarray(out), mask.astype(np.uint8)


def mask_to_256(mask_np, resize, crop):
    m = Image.fromarray(mask_np.astype(np.uint8) * 255)
    m = m.resize((int(resize), int(resize)), Image.NEAREST)
    off = (int(resize) - int(crop)) // 2
    m = m.crop((off, off, off + int(crop), off + int(crop)))
    t = torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0)
    t = t.view(1, 1, int(crop), int(crop))
    return (F.interpolate(t, size=256, mode="nearest") > 0.5).float()


# ---------------------------------------------------------------------------
# compact map features and tiny shared head
# ---------------------------------------------------------------------------
def map_stats(x):
    flat = x.flatten(1)
    lo = torch.quantile(flat, 0.01, dim=1).view(-1, 1, 1, 1)
    hi = torch.quantile(flat, 0.99, dim=1).view(-1, 1, 1, 1)
    return lo, hi, (hi - lo).clamp_min(1e-4)


def make_features(base, hi, guide):
    fused = 0.5 * base + 0.5 * hi
    lo, _hq, scale = map_stats(fused)
    b = (base - lo) / scale
    h = (hi - lo) / scale
    f = (fused - lo) / scale
    residual = ((hi - base) / scale).clamp(-2.0, 2.0)
    g = guide.clamp(0.0, 1.0)
    bg = F.avg_pool2d(f, 15, 1, 7, count_include_pad=False)
    local = (f - bg).clamp(-2.0, 2.0)
    gm = f.mean(dim=(2, 3), keepdim=True)
    gs = f.std(dim=(2, 3), keepdim=True).clamp(0.0, 2.0)
    absolute = torch.log1p(fused.clamp_min(0.0)).clamp(0.0, 20.0)
    return torch.cat([
        b, h, f, residual, g, bg, local,
        gm.expand_as(f), gs.expand_as(f), absolute,
    ], dim=1)


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(N_FEATURES, 24, 3, padding=1)
        self.norm = nn.GroupNorm(8, 24)
        self.conv2 = nn.Conv2d(24, 1, 3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        nn.init.constant_(self.conv2.bias, -2.0)

    def forward(self, x):
        x = F.relu(self.norm(self.conv1(x)))
        return self.conv2(x)


def blend_guarded(guided, head_prob):
    lo, hi, scale = map_stats(guided)
    base_norm = (guided - lo) / scale
    delta = (head_prob - base_norm).clamp(-GUARD_CAP, GUARD_CAP)
    return guided + float(BLEND_ALPHA) * scale * delta


# ---------------------------------------------------------------------------
# runtime
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
    from sklearn.metrics import roc_auc_score, average_precision_score, auc

    CODE = locate("dinomaly_mvtec_uni.py", [
        'E:/dinomaly-mvp/Dinomaly',
        "/kaggle/input/dinomaly-code",
        "/kaggle/input/datasets/eunectesqaq/dinomaly-code",
    ])
    DATA = locate("bottle", [
        'E:/dinomaly-mvp/data/mvtec_anomaly_detection',
        "/kaggle/input/mvtec-ad",
        "/kaggle/input/datasets/ipythonx/mvtec-ad",
    ])
    sys.path.insert(0, CODE)
    os.chdir(CODE)
    from dataset import get_data_transforms, MVTecDataset
    from models.uad import ViTill
    from models import vit_encoder
    from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
    from utils import cal_anomaly_maps, get_gaussian_kernel, f1_score_max
    log(f"CODE={CODE}")
    log(f"DATA={DATA}")


def find_checkpoints():
    global BASE_CKPT, STAGE1_METRICS
    BASE_CKPT = find_input("model_10000.pt", [
        'E:/dinomaly-mvp/kaggle_runs/dinomaly-base-10k/model_10000.pt',
        "/kaggle/input/dinomaly-base-10k/model_10000.pt",
        "/kaggle/input/notebooks/eunectesqaq/dinomaly-base-10k/model_10000.pt",
        "/kaggle/input/datasets/eunectesqaq/dinomaly-base-10k/model_10000.pt",
    ])
    STAGE1_METRICS = find_input("metrics.json", [
        'E:/dinomaly-mvp/kaggle_runs/dinomaly-base-10k/metrics.json',
        "/kaggle/input/dinomaly-base-10k/metrics.json",
        "/kaggle/input/notebooks/eunectesqaq/dinomaly-base-10k/metrics.json",
        "/kaggle/input/datasets/eunectesqaq/dinomaly-base-10k/metrics.json",
    ])
    if not BASE_CKPT:
        raise RuntimeError("model_10000.pt not found under /kaggle/input")
    log(f"base checkpoint={BASE_CKPT}")
    log(f"stage1 metrics={STAGE1_METRICS}")


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
    if chunk is None or chunk <= 0:
        chunk = int(images.shape[0])
    if chunk >= images.shape[0]:
        try:
            return global_map(model, images)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and images.shape[0] > 1:
                torch.cuda.empty_cache()
                return global_map_chunked(model, images,
                                          max(1, images.shape[0] // 2))
            raise
    outs = []
    for i in range(0, images.shape[0], chunk):
        part = images[i:i + chunk]
        try:
            outs.append(global_map(model, part))
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and part.shape[0] > 1:
                torch.cuda.empty_cache()
                outs.append(global_map_chunked(model, part,
                                               max(1, part.shape[0] // 2)))
            else:
                raise
    return torch.cat(outs, dim=0)


def postprocess(amap, gaussian):
    return gaussian(F.interpolate(amap, size=256, mode="bilinear",
                                  align_corners=False))


def top1pct(amap):
    flat = amap.flatten(1)
    k = max(1, int(flat.shape[1] * 0.01))
    return torch.sort(flat, dim=1, descending=True).values[:, :k].mean(dim=1)


def map_triplet(model, x_base, x_hi, gaussian):
    base = postprocess(global_map_chunked(model, x_base, EVAL_BATCH),
                       gaussian)
    hi = postprocess(global_map_chunked(model, x_hi, HI_BATCH), gaussian)
    guide = guide_from_hi(x_hi)
    return base, hi, guide


# ---------------------------------------------------------------------------
# training data cache
# ---------------------------------------------------------------------------
def train_good_paths(root):
    paths = []
    for ext in ("*.png", "*.JPG", "*.jpg", "*.bmp"):
        paths.extend(glob.glob(os.path.join(root, "train", "good", ext)))
    return sorted(paths)


def build_cache(model, item_list, base_tf, hi_tf, gaussian, device):
    cache = []
    started = time.time()
    for ci, item in enumerate(item_list):
        paths = train_good_paths(os.path.join(DATA, item))
        if not paths:
            raise RuntimeError(f"no train/good images for {item}")
        n = min(NUM_TRAIN_IMAGES, len(paths))
        idxs = np.linspace(0, len(paths) - 1, num=n, dtype=int)
        for k, idx in enumerate(idxs):
            orig = Image.open(paths[int(idx)]).convert("RGB")
            syn, native_mask = synthesize(orig, SEED + ci * 100003 + k)
            pair = [(orig, False), (syn, True)]
            xb = torch.stack([base_tf(p) for p, _ in pair]).to(device)
            xh = torch.stack([hi_tf(p) for p, _ in pair]).to(device)
            with torch.no_grad():
                base, hi, guide = map_triplet(model, xb, xh, gaussian)
                m256 = mask_to_256(native_mask, HI_RESIZE, HI_CROP).to(device)
            for j, (_p, is_syn) in enumerate(pair):
                m = m256 if is_syn else torch.zeros_like(m256)
                cache.append({
                    "base": base[j:j + 1].detach().half().cpu(),
                    "hi": hi[j:j + 1].detach().half().cpu(),
                    "guide": guide[j:j + 1].detach().half().cpu(),
                    "mask": m.detach().half().cpu(),
                    "class": item,
                    "synthetic": bool(is_syn),
                })
            del xb, xh, base, hi, guide, m256
        log(f"cache {item}: {n} good + {n} synthetic "
            f"({len(cache)} images, elapsed={time.time() - started:.1f}s)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return cache


def train_head(cache, device):
    head = TinyHead().to(device)
    pos = sum(float(c["mask"].sum()) for c in cache)
    total = float(len(cache) * 256 * 256)
    pos_weight = torch.tensor(min(50.0, (total - pos) / max(pos, 1.0)),
                              device=device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(TRAIN_EPOCHS * 0.55), int(TRAIN_EPOCHS * 0.82)],
        gamma=0.3)
    rng = np.random.default_rng(SEED)
    log(f"train head: images={len(cache)} positives={int(pos)} "
        f"pos_weight={float(pos_weight):.2f} epochs={TRAIN_EPOCHS}")
    head.train()
    for epoch in range(TRAIN_EPOCHS):
        order = rng.permutation(len(cache))
        losses, dices = [], []
        t0 = time.time()
        for i in range(0, len(order), TRAIN_BATCH):
            ids = order[i:i + TRAIN_BATCH]
            base = torch.cat([cache[j]["base"] for j in ids]).to(device).float()
            hi = torch.cat([cache[j]["hi"] for j in ids]).to(device).float()
            guide = torch.cat([cache[j]["guide"] for j in ids]).to(device).float()
            target = torch.cat([cache[j]["mask"] for j in ids]).to(device).float()
            logits = head(make_features(base, hi, guide))
            bce = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pos_weight)
            prob = torch.sigmoid(logits)
            dice = 1.0 - ((2.0 * (prob * target).sum() + 1.0)
                          / (prob.sum() + target.sum() + 1.0))
            loss = bce + 0.5 * dice
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            dices.append(float(dice.detach()))
        sched.step()
        log(f"epoch {epoch + 1}/{TRAIN_EPOCHS} loss={np.mean(losses):.4f} "
            f"dice={np.mean(dices):.4f} lr={opt.param_groups[0]['lr']:.2e} "
            f"sec={time.time() - t0:.1f}")
    head.eval()
    return head


# ---------------------------------------------------------------------------
# test evaluation
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
        return img, gt, label, path, self.hi_transform(pil)


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
        fprs.append(float(np.logical_and(inverse, binmap).sum()) / neg_total)
    fprs, pros = np.asarray(fprs), np.asarray(pros)
    keep = fprs < 0.3
    if keep.sum() < 2 or fprs[keep].max() <= 0:
        return 0.0
    return float(auc(fprs[keep] / fprs[keep].max(), pros[keep]))


def evaluate_class(item, dataset, model, head, gaussian, device):
    loader = DataLoader(dataset, batch_size=EVAL_BATCH, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())
    buckets = {v: [] for v in VARIANTS}
    gt_buf, labels, scores = [], [], []
    started = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            img, gt, label, _paths, x_hi = batch
            img, x_hi = img.to(device), x_hi.to(device)
            base = postprocess(global_map_chunked(model, img, EVAL_BATCH),
                               gaussian)
            hi = postprocess(global_map_chunked(model, x_hi, HI_BATCH),
                             gaussian)
            fused = 0.5 * base + 0.5 * hi
            guide = guide_from_hi(x_hi)
            _box2, guided = refine_heatmap(fused, guide)
            feat = make_features(base, hi, guide)
            head_prob = torch.sigmoid(head(feat))
            per_variant = {
                "v7_guide": guided,
                "learned_head": head_prob,
                "blend_guarded": blend_guarded(guided, head_prob),
            }
            scores.extend(top1pct(base).cpu().numpy().tolist())
            for v in VARIANTS:
                buckets[v].append(per_variant[v].cpu())
            gt256 = F.interpolate(gt.float(), size=256, mode="nearest").bool()
            if gt256.shape[1] > 1:
                gt256 = gt256.max(dim=1, keepdim=True)[0]
            gt_buf.append(gt256.cpu())
            labels.extend(label.numpy().tolist())
            if bi == 0 or (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
                log(f"  {item} batch {bi + 1}/{len(loader)} "
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
        recall, thresholds = fixed_fpr_recall(pr_flat, gt_flat)
        variants[v] = {
            "metrics": {
                "I-AUROC": img_metrics["I-AUROC"],
                "I-AP": img_metrics["I-AP"],
                "I-F1": img_metrics["I-F1"],
                "P-AUROC": float(roc_auc_score(gt_flat, pr_flat)),
                "P-AP": float(average_precision_score(gt_flat, pr_flat)),
                "P-F1": float(f1_score_max(gt_flat, pr_flat)),
                "P-AUPRO": float(compute_pro_pre(gt_np, pr, regions)),
            },
            "recall": recall,
            "thresholds": thresholds,
        }
    return {
        "class": item,
        "fingerprint": FP,
        "n_test": int(len(labels_np)),
        "n_normal": int((labels_np == 0).sum()),
        "n_anomaly": int((labels_np == 1).sum()),
        "n_gt_positive_pixels": int(gt_flat.sum()),
        "image_metrics": img_metrics,
        "image_scores": {
            "source": "original baseline 448->392 global top1%",
            "reused_for_all_variants": True,
            "sha256": hashlib.sha256(
                scores_np.astype(np.float64).tobytes()).hexdigest(),
        },
        "variants": variants,
        "seconds": float(time.time() - started),
    }


def aggregate(per_class, items):
    mean = {}
    for v in VARIANTS:
        mean[v] = {
            "metrics": {k: finite_mean([per_class[i]["variants"][v]
                                        ["metrics"][k] for i in items])
                        for k in METRIC_KEYS},
            "recall": {k: finite_mean([per_class[i]["variants"][v]
                                       ["recall"].get(k) for i in items])
                       for k in RECALL_KEYS},
        }
    image_mean = {k: finite_mean([per_class[i]["image_metrics"][k]
                                  for i in items])
                  for k in ("I-AUROC", "I-AP", "I-F1")}
    deltas = {}
    for v in VARIANTS:
        if v == "v7_guide":
            continue
        deltas[f"{v} - v7_guide"] = {
            "metrics": {k: mean[v]["metrics"][k]
                        - mean["v7_guide"]["metrics"][k]
                        for k in METRIC_KEYS},
            "recall": {k: mean[v]["recall"][k]
                       - mean["v7_guide"]["recall"][k]
                       for k in RECALL_KEYS},
        }
    return {"mean": mean, "image_mean": image_mean, "deltas": deltas}


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def run_self_test():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        if not cond:
            raise AssertionError(msg)
        checks += 1

    ok((392 / 448) == (672 / 768), "same 7/8 FOV for both branches")
    ok(VARIANTS == ("v7_guide", "learned_head", "blend_guarded"),
       "exactly three variants")
    areas = []
    for seed in range(12):
        mask, geom = elliptical_mask(700, 700, np.random.default_rng(seed))
        ok(mask.shape == (700, 700), "mask native shape")
        ok(mask.dtype == np.bool_, "mask bool")
        ok(mask.sum() > 20, "mask non-empty")
        ok(not mask[0].any() and not mask[-1].any()
           and not mask[:, 0].any() and not mask[:, -1].any(),
           "mask inside border")
        areas.append(float(mask.mean()))
    ok(len(set(round(a, 5) for a in areas)) > 6, "varied mask scale")

    import torch as _torch
    import torch.nn.functional as _F
    globals()["torch"] = _torch
    globals()["F"] = _F
    g = _torch.full((1, 1, 32, 32), 0.5)
    p = _torch.full((1, 1, 32, 32), 0.3)
    ok(float((guided_filter(g, p) - p).abs().max()) < 1e-6,
       "flat guided identity")
    base = _torch.rand(2, 1, 64, 64)
    hi = _torch.rand(2, 1, 64, 64)
    guide = _torch.rand(2, 1, 64, 64)
    feat = make_features(base, hi, guide)
    ok(tuple(feat.shape) == (2, N_FEATURES, 64, 64), "feature shape")
    head = TinyHead()
    out = head(feat)
    ok(tuple(out.shape) == (2, 1, 64, 64), "head output shape")
    prob = _torch.sigmoid(out)
    blend = blend_guarded(base, prob)
    ok(tuple(blend.shape) == (2, 1, 64, 64), "guarded blend shape")
    ok(bool(_torch.isfinite(blend).all()), "guarded blend finite")
    m = mask_to_256(np.ones((700, 700), np.uint8), 768, 672)
    ok(tuple(m.shape) == (1, 1, 256, 256), "mask 256 shape")
    ok(float(m.mean()) > 0.5, "mask transform keeps positives")
    print(f"SELF-TEST OK ({checks} checks)")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    global LOG_PATH, FP, EVAL_BATCH, HI_BATCH, TRAIN_EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--eval-batch", type=int, default=EVAL_BATCH)
    ap.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--limit-classes", type=int, default=0)
    args = ap.parse_args()
    if args.self_test:
        return run_self_test()
    EVAL_BATCH = max(1, int(args.eval_batch))
    HI_BATCH = max(1, min(2, EVAL_BATCH))
    TRAIN_EPOCHS = max(1, int(args.epochs))

    os.makedirs(OUT, exist_ok=True)
    LOG_PATH = os.path.join(OUT, "stage11.log")
    log(f"=== {SLUG} start ===")
    log(f"args={vars(args)}")
    run_self_test()
    log("self-test passed")

    prepare_environment()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"device={device} torch={torch.__version__}")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    find_checkpoints()
    base_sha = sha256_file(BASE_CKPT)
    log(f"base sha256={base_sha}")

    item_list = sorted(d for d in os.listdir(DATA)
                       if os.path.isdir(os.path.join(DATA, d)))
    if len(item_list) != 3:
        raise RuntimeError(f"expected 3 local MVTec classes, got {item_list}")

    model = build_model(device)
    st_base = load_ckpt(model, BASE_CKPT, device)
    model.eval()
    log(f"checkpoint iter={st_base.get('iter')} kind={st_base.get('kind')}")

    base_tf, gt_tf = get_data_transforms(BASE_RESIZE, BASE_CROP)
    hi_tf = get_data_transforms(HI_RESIZE, HI_CROP)[0]
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    log("building compact synthetic cache (train/good only)")
    cache = build_cache(model, item_list, base_tf, hi_tf, gaussian, device)
    if len(cache) != 2 * NUM_TRAIN_IMAGES * len(item_list):
        raise RuntimeError(f"unexpected cache size {len(cache)}")
    log("training shared refinement head")
    head = train_head(cache, device)
    head_path = os.path.join(OUT, "refine_head.pt")
    torch.save({"state_dict": head.state_dict(),
                "features": N_FEATURES,
                "train_images": len(cache),
                "synthetic_per_class": NUM_TRAIN_IMAGES,
                "epochs": TRAIN_EPOCHS,
                "seed": SEED}, head_path)
    log(f"head saved {head_path} sha256={sha256_file(head_path)}")
    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    test_sets = {
        item: MultiScaleTestDataset(
            root=os.path.join(DATA, item), base_transform=base_tf,
            gt_transform=gt_tf, hi_transform=hi_tf)
        for item in item_list
    }
    counts = {it: len(ds) for it, ds in test_sets.items()}
    total_test = int(sum(counts.values()))
    log(f"test images={total_test} per-class={counts}")
    if total_test != EXPECTED_TEST_IMAGES:
        raise RuntimeError(f"expected {EXPECTED_TEST_IMAGES}, got {total_test}")

    eval_items = list(item_list)
    if args.limit_classes and args.limit_classes > 0:
        eval_items = eval_items[:args.limit_classes]
    partial_run = len(eval_items) != len(item_list)

    config = {
        "slug": SLUG,
        "stage": 11,
        "script": os.path.basename(os.path.abspath(__file__)),
        "code_sha256": sha256_file(os.path.abspath(__file__)),
        "baseline": {"resize": BASE_RESIZE, "crop": BASE_CROP},
        "high_res": {"resize": HI_RESIZE, "crop": HI_CROP},
        "variants": list(VARIANTS),
        "fusion": "0.5 * base_global + 0.5 * hi672",
        "guide": {"radius": GUIDE_RADIUS, "epsilon": GUIDE_EPS},
        "training": {
            "source": "MVTec train/good only",
            "classes": item_list,
            "good_images_per_class": NUM_TRAIN_IMAGES,
            "synthetic_images_per_class": NUM_TRAIN_IMAGES,
            "pairs_total": 2 * NUM_TRAIN_IMAGES * len(item_list),
            "synthetic": "irregular ellipse; moved/replaced/contrast patch",
            "seed": SEED,
            "epochs": TRAIN_EPOCHS,
            "batch": TRAIN_BATCH,
            "optimizer": "AdamW lr 1e-3 wd 1e-4",
            "schedule": "MultiStepLR 0.3 at 55% and 82%",
            "loss": "pixel BCE pos_weight + 0.5 soft Dice",
            "head": f"Conv3x3({N_FEATURES}->24)-GroupNorm-ReLU-"
                    "Conv3x3(24->1)",
            "test_images_or_labels_used_for_fit": False,
            "external_data": False,
        },
        "blend": {"alpha": BLEND_ALPHA, "guard_cap": GUARD_CAP,
                  "definition": "guided + 0.5*scale*clip(head_prob - "
                                "normalized(guided), -0.25, 0.25)"},
        "metrics": {
            "keys": list(METRIC_KEYS), "recall": list(RECALL_KEYS),
            "evaluation_grid": 256,
            "mask_source": "exact original 448->392 gt, nearest to 256",
            "image_score": "original baseline top1% reused unchanged",
        },
        "classes_total": len(item_list),
        "eval_classes": eval_items,
        "partial": partial_run,
        "test_images_total": total_test,
        "test_images_evaluated": int(sum(counts[i] for i in eval_items)),
        "test_counts": counts,
        "base_checkpoint": BASE_CKPT,
        "base_sha256": base_sha,
    }
    FP = fingerprint(config)
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

    def ppath(item):
        return os.path.join(per_class_dir, item + ".json")

    def load_pc(item):
        try:
            d = read_json(ppath(item))
            return d if d.get("fingerprint") == FP else None
        except Exception:
            return None

    done = []
    if not args.fresh and os.path.isfile(progress_path):
        try:
            prog = read_json(progress_path)
            if prog.get("fingerprint") == FP:
                done = [i for i in prog.get("classes_done", [])
                        if i in eval_items and load_pc(i) is not None]
        except Exception:
            pass
    log(f"resume: {len(done)}/{len(eval_items)} complete")

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
        if item in done:
            log(f"{item}: resume skip")
            continue
        set_status("running", classes_done=done,
                   classes_total=len(eval_items), current=item)
        update_progress(current=item)
        log(f"{item}: evaluating {counts[item]} test images")
        result = evaluate_class(item, test_sets[item], model, head,
                                gaussian, device)
        write_json_atomic(ppath(item), result)
        done.append(item)
        latest = {"class": item, "seconds": result["seconds"],
                  "P-AP": {v: result["variants"][v]["metrics"]["P-AP"]
                            for v in VARIANTS}}
        update_progress(current=None, latest=latest)
        set_status("running", classes_done=done,
                   classes_total=len(eval_items), current=None,
                   latest=latest)
        log(f"{item}: done in {result['seconds']:.1f}s " + " ".join(
            f"{v}:AP={result['variants'][v]['metrics']['P-AP']:.4f}"
            f"/PRO={result['variants'][v]['metrics']['P-AUPRO']:.4f}"
            for v in VARIANTS))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_class = {}
    for item in eval_items:
        d = load_pc(item)
        if d is None:
            raise RuntimeError(f"missing per-class result {item}")
        per_class[item] = d
    agg = aggregate(per_class, eval_items)
    report = {
        "experiment": {
            "slug": SLUG, "stage": 11,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "partial": partial_run,
            "hypothesis": "synthetic train/good local defects teach a shared "
                          "refinement head to suppress normal-texture false "
                          "positives while preserving the frozen v7 baseline",
        },
        "config": config,
        "checkpoints": {
            "base": {"path": BASE_CKPT, "sha256": base_sha,
                     "iter": st_base.get("iter"),
                     "kind": st_base.get("kind"),
                     "unchanged": True},
            "head": {"path": head_path,
                     "sha256": sha256_file(head_path)},
        },
        "identity": {
            "image_score_unchanged": True,
            "image_score_source": "baseline 448->392 global top1% reused by "
                                  "all variants",
            "test_labels_used_for_fit": False,
            "test_images_used_for_fit": False,
            "external_data": False,
            "frozen_checkpoint": True,
            "exact_v7_geometry": True,
            "exact_official_masks": True,
        },
        "results": {"per_class": per_class, "mean": agg["mean"]},
        "image_metrics": {"mean": agg["image_mean"]},
        "deltas": agg["deltas"],
        "runtime": {
            "device": device, "torch": getattr(torch, "__version__", None),
            "classes_evaluated": len(eval_items),
            "test_images_evaluated": config["test_images_evaluated"],
            "seconds_total": float(sum(per_class[i]["seconds"]
                                       for i in eval_items)),
        },
        "files": {"per_class_dir": per_class_dir,
                  "progress": progress_path, "status": status_path,
                  "head": head_path},
    }
    report_path = os.path.join(OUT, "synthetic_refine_report.json")
    write_json_atomic(report_path, report)
    set_status("complete", classes_done=done,
               classes_total=len(eval_items), report=report_path,
               markdown=None)
    log("DONE")
    log(f"report={report_path}")
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
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
        except Exception:
            pass
        raise
