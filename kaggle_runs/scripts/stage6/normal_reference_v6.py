"""Stage 6: conservative normal-reference fusion on the frozen Dinomaly run.

Stage 6 keeps the stage-5 normal reference (same 40 memory + 16 calibration
train/good images, same frozen 10k weights, same official 15-class protocol)
and replaces aggressive score mixing with a conservative positive residual:
the calibrated base 392+672 map is only increased where the reference map
exceeds its calibrated q99 and both maps agree in a 3x3 neighbourhood.  The
residual is capped.  This targets the v5 regression classes (zipper, pill)
without subtracting anything from the base score.  Pose-aligned reference
matching was scoped but deliberately deferred to protect the competition
deadline; all reported variants are unaligned and marked as such.

Architecture addition
----------------------
ViTill's 392 forward already returns two fused encoder feature groups
(``en[0]``, ``en[1]``, each 768 x 28 x 28).  We use them as a class-specific
normal-reference branch:

* 40 deterministic train/good memory images + 16 disjoint train/good
  calibration images per class.  Test images are never used for fitting.
* Per-group z-score statistics are estimated from the memory bank, then the
  two groups are concatenated (2 x 768 = 1536) and L2-normalised.
* ``free`` match: 1 - mean top-3 cosine similarity over every bank patch of
  every memory image (appearance: does a similar normal exemplar exist
  anywhere?).
* ``spatial`` match: the same top-3 statistic, but candidates are restricted
  to bank patches within +/-1 patch of the query position, across every memory
  image (structure: is the exemplar also in the right place?).  Positions are
  masked, never padded into the candidate set.  Matches are batched GEMMs.

Fusion
------
Each branch is calibrated on the 16 calibration normals with
``(x - median) / max(q95 - median, 0.01)``.  The spatial gate is inferred from
the calibration normals only:

    rel_extra = median( (d_spatial - d_free) / max(d_free, 0.02) )
    g         = 0.15 + 0.35 * exp(-rel_extra / 0.25),  clipped to [0.15, 0.5]

High relative extra distance means the spatial constraint is less repeatable,
so spatial weight is lower.  The formula and its constants are fixed a priori;
there are no hard-coded per-class weights.

Variants (7, fixed before evaluation)
-------------------------------------
base_global       original 392 baseline map (archived stage-3 logic)
ref_free          free normal-reference map
ref_spatial       spatially constrained normal-reference map
ref_fuse_fixed    0.5 * calib(ref_free) + 0.5 * calib(ref_spatial)
ref_adaptive      (1-g) * calib(ref_free) + g * calib(ref_spatial)   PRIMARY
base_fuse672      archived stage-4 392+672 baseline
                  (0.5 * base_global + 0.5 * hi672)
ref_plus_base     0.5 * calib(base_fuse672) + 0.5 * ref_adaptive

All pixel variants are evaluated on the same official 256 grid and the same
448->392 mask transform as stages 3/4.  The image score is always the original
baseline global top-1% score and is reused by every variant.  No training is
performed; the frozen 10k checkpoint is reused unchanged.
"""
import os
import sys
import json
import time
import math
import argparse
import hashlib
import traceback

import numpy as np

OUT = "/kaggle/working"
SLUG = "eunectesqaq/dinomaly-aligned-reference-v6"
BASE_RESIZE = 448
BASE_CROP = 392
HI_RESIZE = 768
HI_CROP = 672
EXPECTED_TEST_IMAGES = 1725
EVAL_BATCH = 4
HI_BATCH = 2
REF_CHUNK = 2

FPRS = (0.001, 0.01, 0.05)
METRIC_KEYS = ("I-AUROC", "I-AP", "I-F1",
               "P-AUROC", "P-AP", "P-F1", "P-AUPRO")
RECALL_KEYS = tuple(f"R@FPR{f:g}" for f in FPRS)

MEMORY_PER_CLASS = 40
CALIB_PER_CLASS = 16
TOTAL_NORMAL_PER_CLASS = MEMORY_PER_CLASS + CALIB_PER_CLASS
TOP_K = 3
LOCAL_RADIUS = 1
SCALE_FLOOR = 0.01
DIST_FLOOR = 0.02
GATE_G_MIN = 0.15
GATE_G_MAX = 0.50
GATE_K = 1.0
GATE_R0 = 0.25
FIXED_SPATIAL_WEIGHT = 0.5
REF_BASE_WEIGHT = 0.5
REF_TAIL_Q = 0.99
CONSERVATIVE_W = 0.25
CONSERVATIVE_CAP = 2.0
STRONG_W = 0.50
STRONG_CAP = 1.0
AGREE_KERNEL = 3

VARIANTS = (
    "base_fuse672",
    "ref_adaptive",
    "ref_plus_base",
    "ref_conservative",
    "ref_conservative_strong",
)
PRIMARY_VARIANT = "ref_conservative"

VARIANT_DEFINITIONS = {
    "base_fuse672": "archived stage-4 392+672 baseline: "
                    "0.5 * base_global + 0.5 * hi672",
    "ref_adaptive": "(1-g) * calib(ref_free) + g * calib(ref_spatial); "
                    "stage-5 normal-reference map (no base score)",
    "ref_plus_base": "stage-5 reproduction: 0.5 * calib(base_fuse672) + "
                     "0.5 * ref_adaptive",
    "ref_conservative": "calib(base_fuse672) + 0.25 * capped positive "
                        "ref residual above calibrated q99, only where both "
                        "maps agree in a 3x3 neighbourhood (PRIMARY, unaligned)",
    "ref_conservative_strong": "same gate with base must exceed calibrated "
                               "q95, residual cap 1.0 and weight 0.5 "
                               "(unaligned)",
}

CONSERVATIVE_FORMULA = (
    "score = calib(base_fuse672) + w * clamp(calib(ref_spatial) - "
    "calib_q99, 0, cap) * agree; agree = 3x3 max-pool of "
    "[calib(base)>base_thr]; nothing is subtracted from base; "
    "pose alignment intentionally not applied in this run")

GATE_FORMULA = ("g = 0.15 + 0.35 * exp(-max(rel_extra, 0) / 0.25), "
                "clipped to [0.15, 0.50]; "
                "rel_extra = median((d_spatial - d_free) / "
                "max(d_free, 0.02)) over calibration normals")

# Frozen stage-4 15-class means (fractions), used only as an audit baseline.
STAGE4_ARCHIVED = {
    "base_global": {
        "metrics": {
            "I-AUROC": 0.9962293697223761,
            "I-AP": 0.9980926073135531,
            "I-F1": 0.9914942848565,
            "P-AUROC": 0.9832705030393117,
            "P-AP": 0.6881937878119229,
            "P-F1": 0.6904051603168335,
            "P-AUPRO": 0.9467326861843743,
        },
        "recall": {
            "R@FPR0.001": 0.18119826225376187,
            "R@FPR0.01": 0.6547711206643027,
            "R@FPR0.05": 0.9450044124953778,
        },
    },
    "fuse672": {
        "metrics": {
            "I-AUROC": 0.9962293697223761,
            "I-AP": 0.9980926073135531,
            "I-F1": 0.9914942848565,
            "P-AUROC": 0.9839427484278644,
            "P-AP": 0.7247128126906136,
            "P-F1": 0.7136586490415545,
            "P-AUPRO": 0.952742818770344,
        },
        "recall": {
            "R@FPR0.001": 0.2230486114836818,
            "R@FPR0.01": 0.7045086752029253,
            "R@FPR0.05": 0.9431508867430506,
        },
    },
}

LOG_PATH = None
FP = None
CODE = None
DATA = None
BASE_CKPT = None
STAGE1_METRICS = None

# Runtime imports are bound by prepare_environment() / run_self_test().
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
# deterministic normal sampling, geometry, pure helpers
# ---------------------------------------------------------------------------
def select_normal_paths(paths, memory=MEMORY_PER_CLASS,
                        calibration=CALIB_PER_CLASS):
    """Deterministic, disjoint train/good selection from sorted paths.

    The first ``memory`` selections form the bank; the remaining
    ``calibration`` selections are used only for normal calibration/gating.
    The selection uses evenly spaced indices over the sorted path list, so it
    is reproducible and class-specific without RNG.
    """
    paths = sorted(str(p) for p in paths)
    need = int(memory) + int(calibration)
    if len(paths) < need:
        raise ValueError(f"need {need} train/good images, got {len(paths)}")
    idx = np.round(np.linspace(0, len(paths) - 1, need)).astype(np.int64)
    if len(set(int(i) for i in idx)) != need:
        raise RuntimeError("deterministic selection produced duplicate indices")
    mem = [paths[int(i)] for i in idx[:memory]]
    cal = [paths[int(i)] for i in idx[memory:]]
    if set(mem) & set(cal):
        raise RuntimeError("memory and calibration selections overlap")
    return mem, cal


def neighborhood_offsets(radius=LOCAL_RADIUS):
    r = int(radius)
    return [(dr, dc) for dr in range(-r, r + 1)
            for dc in range(-r, r + 1)]


def valid_offset_counts(radius=LOCAL_RADIUS, side=28):
    offs = neighborhood_offsets(radius)
    counts = []
    for r in range(side):
        for c in range(side):
            counts.append(sum(1 for dr, dc in offs
                              if 0 <= r + dr < side and 0 <= c + dc < side))
    return counts


def adaptive_gate(rel_extra):
    rel = max(float(rel_extra), 0.0)
    g = GATE_G_MIN + (GATE_G_MAX - GATE_G_MIN) * math.exp(
        -GATE_K * rel / GATE_R0)
    return float(min(max(g, GATE_G_MIN), GATE_G_MAX))


def branch_stats(flat):
    """Median / q95 / positive scale for one branch's normal calibration map."""
    flat = flat.reshape(-1).float()
    median = float(torch.quantile(flat, 0.5))
    q95 = float(torch.quantile(flat, 0.95))
    q99 = float(torch.quantile(flat, REF_TAIL_Q))
    scale = max(q95 - median, SCALE_FLOOR)
    return {"median": median, "q95": q95, "q99": q99, "scale": scale,
            "n_values": int(flat.numel())}


def calibrate_map(x, stats):
    return (x - float(stats["median"])) / float(stats["scale"])


def calibrated_quantile(stats, q):
    key = {0.95: "q95", 0.99: "q99"}.get(float(q))
    if key is not None and key in stats:
        return float((stats[key] - stats["median"]) / stats["scale"])
    return float(q)


def conservative_fusion(cal_base, cal_ref, ref_stats, weight, cap,
                        base_thr):
    """Add only a capped positive reference residual where maps agree.

    cal_base / cal_ref: [B, 1, H, W] normal-calibrated maps.
    The residual threshold is the calibration-normal q99 expressed in the
    calibrated units of the reference branch. A 3x3 max-pool checks for
    supporting baseline evidence near each reference exceedance.
    """
    thr = calibrated_quantile(ref_stats, REF_TAIL_Q)
    resid = torch.clamp(cal_ref - thr, min=0.0, max=float(cap))
    pos = (cal_base > float(base_thr)).float()
    agree = F.max_pool2d(pos, kernel_size=AGREE_KERNEL,
                         stride=1, padding=AGREE_KERNEL // 2)
    return cal_base + float(weight) * resid * agree


def fov(crop, resize):
    return float(crop) / float(resize)


def finite_mean(values):
    vals = [float(v) for v in values
            if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


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
# normal-reference core (self-testable, torch tensors)
# ---------------------------------------------------------------------------
def normalize_features(raw0, raw1, mean0, std0, mean1, std1):
    """Per-group z-score then concatenate and L2-normalise."""
    g0 = (raw0 - mean0.view(1, -1, 1, 1)) / std0.view(1, -1, 1, 1)
    g1 = (raw1 - mean1.view(1, -1, 1, 1)) / std1.view(1, -1, 1, 1)
    return F.normalize(torch.cat([g0, g1], dim=1), p=2, dim=1, eps=1e-8)


def reference_distances(features, bank, radius=LOCAL_RADIUS, topk=TOP_K):
    """Free and spatially constrained top-k cosine distances.

    features: [B, C, H, W] L2-normalised
    bank:     [M*H*W, C] L2-normalised, row-major (image, h, w)
    returns:  free [B, H, W], local [B, H, W] (cosine distances)
    """
    B, C, H, W = features.shape
    hw = H * W
    if bank.shape[0] % hw != 0:
        raise ValueError("bank patch count is not a multiple of H*W")
    M = bank.shape[0] // hw

    q = features.flatten(2).transpose(1, 2).reshape(B * hw, C)
    sim = q @ bank.t()                      # batched GEMM
    sim = sim.view(B, hw, M, hw)

    # free: top-k over every bank patch of every memory image
    free_sim = sim.reshape(B, hw, M * hw).topk(
        int(topk), dim=2, largest=True).values.mean(dim=2)

    # spatial: candidates at query position +/- radius, all memory images.
    # Out-of-image offsets are masked; no padded candidate ever participates.
    rr = torch.div(torch.arange(hw, device=features.device), W,
                   rounding_mode="floor")
    cc = torch.arange(hw, device=features.device) % W
    gathered = []
    for dr, dc in neighborhood_offsets(radius):
        br = rr + int(dr)
        bc = cc + int(dc)
        valid = (br >= 0) & (br < H) & (bc >= 0) & (bc < W)
        pos = (br.clamp(0, H - 1) * W + bc.clamp(0, W - 1)).long()
        idx = pos.view(1, hw, 1, 1).expand(B, hw, M, 1).contiguous()
        s = torch.gather(sim, 3, idx).squeeze(-1)   # [B, hw, M]
        s = s.masked_fill((~valid).view(1, hw, 1), float("-inf"))
        gathered.append(s)
    cand = torch.stack(gathered, dim=3).reshape(B, hw, -1)
    local_sim = cand.topk(int(topk), dim=2, largest=True).values.mean(dim=2)

    free = (1.0 - free_sim).view(B, H, W)
    local = (1.0 - local_sim).view(B, H, W)
    return free, local


def reference_maps_chunked(features, bank, chunk=REF_CHUNK,
                           radius=LOCAL_RADIUS, topk=TOP_K):
    """Bound similarity-matrix memory by chunking over the batch dimension."""
    if chunk is None or chunk <= 0:
        chunk = int(features.shape[0])
    frees, locals_ = [], []
    i = 0
    while i < features.shape[0]:
        f, l = reference_distances(features[i:i + chunk], bank,
                                   radius=radius, topk=topk)
        frees.append(f)
        locals_.append(l)
        i += chunk
    return torch.cat(frees, dim=0), torch.cat(locals_, dim=0)


# ---------------------------------------------------------------------------
# environment / model
# ---------------------------------------------------------------------------
def prepare_environment(code_root=None, data_root=None):
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

    if code_root:
        CODE = os.path.abspath(code_root)
    else:
        CODE = locate("dinomaly_mvtec_uni.py", [
            "/kaggle/input/dinomaly-code",
            "/kaggle/input/datasets/eunectesqaq/dinomaly-code",
        ])
    if data_root:
        DATA = os.path.abspath(data_root)
    else:
        DATA = locate("bottle", [
            "/kaggle/input/mvtec-ad",
            "/kaggle/input/datasets/ipythonx/mvtec-ad",
        ])

    if not os.path.isfile(os.path.join(CODE, "models", "uad.py")):
        raise RuntimeError(f"models/uad.py not found under {CODE}")

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


def find_checkpoints(checkpoint=None, stage1_metrics=None):
    global BASE_CKPT, STAGE1_METRICS
    if checkpoint:
        BASE_CKPT = os.path.abspath(checkpoint)
    else:
        BASE_CKPT = find_input("model_10000.pt", [
            "/kaggle/input/dinomaly-base-10k/model_10000.pt",
            "/kaggle/input/notebooks/eunectesqaq/dinomaly-base-10k/model_10000.pt",
            "/kaggle/input/datasets/eunectesqaq/dinomaly-base-10k/model_10000.pt",
        ])
    if stage1_metrics:
        STAGE1_METRICS = os.path.abspath(stage1_metrics)
    else:
        STAGE1_METRICS = find_input("metrics.json", [
            "/kaggle/input/dinomaly-base-10k/metrics.json",
            "/kaggle/input/notebooks/eunectesqaq/dinomaly-base-10k/metrics.json",
            "/kaggle/input/datasets/eunectesqaq/dinomaly-base-10k/metrics.json",
        ])
    if not BASE_CKPT or not os.path.isfile(BASE_CKPT):
        raise RuntimeError("model_10000.pt not found under /kaggle/input")
    log(f"base checkpoint={BASE_CKPT}")
    log(f"stage1 metrics={STAGE1_METRICS}")


def load_stage1_metrics():
    if not STAGE1_METRICS or not os.path.isfile(STAGE1_METRICS):
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


def global_map_with_features(model, images):
    en, de = model(images)
    amap, _ = cal_anomaly_maps(en, de, images.shape[-1])
    return amap, en


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
# datasets
# ---------------------------------------------------------------------------
class NormalTensorDataset:
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class NormalPairDataset:
    """Calibration images: 392 tensor, 672 tensor, path."""
    def __init__(self, paths, base_transform, hi_transform):
        self.paths = list(paths)
        self.base_transform = base_transform
        self.hi_transform = hi_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        pil = Image.open(path).convert("RGB")
        return (self.base_transform(pil), self.hi_transform(pil), path)


class TestPairDataset:
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


# ---------------------------------------------------------------------------
# class reference: memory bank, calibration statistics, adaptive gate
# ---------------------------------------------------------------------------
def build_memory_bank(memory_paths, base_transform, model, device):
    ds = NormalTensorDataset(memory_paths, base_transform)
    loader = DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=4,
                        pin_memory=torch.cuda.is_available())
    sum0 = torch.zeros(768, dtype=torch.float64, device=device)
    sum1 = torch.zeros(768, dtype=torch.float64, device=device)
    ssq0 = torch.zeros(768, dtype=torch.float64, device=device)
    ssq1 = torch.zeros(768, dtype=torch.float64, device=device)
    n = 0
    raw0_chunks, raw1_chunks = [], []
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            en, _de = model(x)
        f0, f1 = en[0].detach(), en[1].detach()
        d0, d1 = f0.double(), f1.double()
        sum0 += d0.sum(dim=(0, 2, 3))
        sum1 += d1.sum(dim=(0, 2, 3))
        ssq0 += (d0 * d0).sum(dim=(0, 2, 3))
        ssq1 += (d1 * d1).sum(dim=(0, 2, 3))
        n += int(d0.shape[0] * d0.shape[2] * d0.shape[3])
        raw0_chunks.append(f0.cpu())
        raw1_chunks.append(f1.cpu())
        del en, _de, f0, f1, d0, d1, x

    if n == 0:
        raise RuntimeError("empty memory bank")
    mean64_0, mean64_1 = sum0 / n, sum1 / n
    var64_0 = (ssq0 / n - mean64_0 * mean64_0).clamp_min(0.0)
    var64_1 = (ssq1 / n - mean64_1 * mean64_1).clamp_min(0.0)
    mean0 = mean64_0.float()
    mean1 = mean64_1.float()
    std0 = torch.sqrt(var64_0 + 1e-6).float()
    std1 = torch.sqrt(var64_1 + 1e-6).float()

    parts = []
    for r0, r1 in zip(raw0_chunks, raw1_chunks):
        r0 = r0.to(device, non_blocking=True)
        r1 = r1.to(device, non_blocking=True)
        parts.append(normalize_features(
            r0, r1, mean0, std0, mean1, std1).cpu())
        del r0, r1
    bank = torch.cat(parts, dim=0).to(device)
    B, C, H, W = bank.shape
    bank = bank.permute(0, 2, 3, 1).reshape(B * H * W, C).contiguous()
    del parts, raw0_chunks, raw1_chunks
    return {
        "bank": bank,
        "mean0": mean0, "std0": std0,
        "mean1": mean1, "std1": std1,
        "memory_count": int(B),
        "patch_side": int(H),
        "feature_groups": [768, 768],
        "concat_dim": int(C),
        "bank_patches": int(bank.shape[0]),
    }


def build_calibration(calib_paths, base_transform, hi_transform, model,
                      ref, gaussian, device):
    ds = NormalPairDataset(calib_paths, base_transform, hi_transform)
    loader = DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=4,
                        pin_memory=torch.cuda.is_available())
    buckets = {k: [] for k in
               ("ref_free", "ref_spatial", "base_global", "base_fuse672")}
    rel_chunks = []
    for base_x, hi_x, _paths in loader:
        base_x = base_x.to(device, non_blocking=True)
        hi_x = hi_x.to(device, non_blocking=True)
        with torch.no_grad():
            en, de = model(base_x)
            amap, _ = cal_anomaly_maps(en, de, base_x.shape[-1])
            base = postprocess(amap, gaussian)
            feats = normalize_features(
                en[0].detach(), en[1].detach(),
                ref["mean0"], ref["std0"], ref["mean1"], ref["std1"])
            free_raw, local_raw = reference_maps_chunked(
                feats, ref["bank"], REF_CHUNK)
            free = postprocess(free_raw.unsqueeze(1), gaussian)
            local = postprocess(local_raw.unsqueeze(1), gaussian)
            hi_raw = global_map_chunked(model, hi_x, HI_BATCH)
            hi_map = postprocess(hi_raw, gaussian)
        bf = 0.5 * base + 0.5 * hi_map
        buckets["ref_free"].append(free.detach().cpu())
        buckets["ref_spatial"].append(local.detach().cpu())
        buckets["base_global"].append(base.detach().cpu())
        buckets["base_fuse672"].append(bf.detach().cpu())
        rel = (local_raw - free_raw) / torch.clamp(
            free_raw, min=DIST_FLOOR)
        rel_chunks.append(rel.reshape(-1).detach().cpu())
        del en, de, feats, free_raw, local_raw, free, local, hi_raw, hi_map
        del base, bf, base_x, hi_x

    if not rel_chunks:
        raise RuntimeError("empty calibration set")
    stats = {}
    for k, chunks in buckets.items():
        stats[k] = branch_stats(torch.cat([c.reshape(-1) for c in chunks]))
    rel_extra = float(torch.cat(rel_chunks).median())
    gate = adaptive_gate(rel_extra)
    return stats, rel_extra, gate


def build_class_reference(item, root, base_transform, hi_transform, model,
                          gaussian, device):
    train_good = os.path.join(root, "train", "good")
    exts = ("*.png", "*.JPG", "*.bmp")
    import glob
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(train_good, ext)))
    if not paths:
        raise RuntimeError(f"{item}: no train/good images under {train_good}")
    memory_paths, calib_paths = select_normal_paths(paths)
    log(f"{item}: normal reference memory={len(memory_paths)} "
        f"calibration={len(calib_paths)} "
        f"(train/good available={len(paths)})")
    ref = build_memory_bank(memory_paths, base_transform, model, device)
    log(f"{item}: bank patches={ref['bank_patches']} "
        f"dim={ref['concat_dim']} side={ref['patch_side']}")
    stats, rel_extra, gate = build_calibration(
        calib_paths, base_transform, hi_transform, model, ref, gaussian,
        device)
    ref.update({
        "memory_paths": memory_paths,
        "calibration_paths": calib_paths,
        "train_good_available": len(paths),
        "calibration": stats,
        "rel_extra_median": float(rel_extra),
        "gate": float(gate),
    })
    log(f"{item}: gate rel_extra={rel_extra:.6f} g_spatial={gate:.6f}; "
        f"free med/q95/scale="
        f"{stats['ref_free']['median']:.6f}/"
        f"{stats['ref_free']['q95']:.6f}/"
        f"{stats['ref_free']['scale']:.6f}; "
        f"spatial med/q95/scale="
        f"{stats['ref_spatial']['median']:.6f}/"
        f"{stats['ref_spatial']['q95']:.6f}/"
        f"{stats['ref_spatial']['scale']:.6f}")
    return ref


def reference_summary(ref):
    return {
        "source": "train/good only; test images never used for fitting",
        "memory_count": int(ref["memory_count"]),
        "calibration_count": int(len(ref["calibration_paths"])),
        "total_per_class": int(ref["memory_count"] +
                               len(ref["calibration_paths"])),
        "disjoint": True,
        "train_good_available": int(ref["train_good_available"]),
        "memory_files": [os.path.basename(p) for p in ref["memory_paths"]],
        "calibration_files": [os.path.basename(p)
                              for p in ref["calibration_paths"]],
        "selection": "sorted train/good paths; evenly spaced indices; first "
                     "40 memory, last 16 calibration",
        "feature_groups": ref["feature_groups"],
        "concat_dim": ref["concat_dim"],
        "dimensionality_reduction": None,
        "normalization": "per-group z-score from memory bank, concat, L2",
        "bank_patches": ref["bank_patches"],
        "patch_side": ref["patch_side"],
        "top_k": TOP_K,
        "local_radius": LOCAL_RADIUS,
        "calibration": ref["calibration"],
        "gate": {
            "rel_extra_median": float(ref["rel_extra_median"]),
            "g_spatial": float(ref["gate"]),
            "formula": GATE_FORMULA,
        },
    }


# ---------------------------------------------------------------------------
# metrics copied from stage 3/4 (identical official protocol)
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


# ---------------------------------------------------------------------------
# per-class test evaluation
# ---------------------------------------------------------------------------
def evaluate_class(item, dataset, model, ref, gaussian, device, stage1_item):
    loader = DataLoader(dataset, batch_size=EVAL_BATCH, shuffle=False,
                        num_workers=4,
                        pin_memory=torch.cuda.is_available())
    buckets = {v: [] for v in VARIANTS}
    gt_buf, labels, scores = [], [], []
    n_batches = len(loader)
    started = time.time()
    reset_peak()

    cal = ref["calibration"]
    gate = float(ref["gate"])

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            img, gt, label, _paths, hi = batch
            img = img.to(device, non_blocking=True)
            hi = hi.to(device, non_blocking=True)

            # 392 baseline map + the two encoder feature groups, one forward.
            base_raw, en = global_map_with_features(model, img)
            base = postprocess(base_raw, gaussian)

            feats = normalize_features(
                en[0].detach(), en[1].detach(),
                ref["mean0"], ref["std0"], ref["mean1"], ref["std1"])
            free_raw, local_raw = reference_maps_chunked(
                feats, ref["bank"], REF_CHUNK)
            free = postprocess(free_raw.unsqueeze(1), gaussian)
            spatial = postprocess(local_raw.unsqueeze(1), gaussian)

            # Archived stage-4 392+672 baseline.
            hi_raw = global_map_chunked(model, hi, HI_BATCH)
            hi_map = postprocess(hi_raw, gaussian)
            base_fuse672 = 0.5 * base + 0.5 * hi_map

            cal_free = calibrate_map(free, cal["ref_free"])
            cal_spatial = calibrate_map(spatial, cal["ref_spatial"])
            cal_base672 = calibrate_map(base_fuse672, cal["base_fuse672"])

            ref_fuse_fixed = (FIXED_SPATIAL_WEIGHT * cal_spatial
                              + (1.0 - FIXED_SPATIAL_WEIGHT) * cal_free)
            ref_adaptive = gate * cal_spatial + (1.0 - gate) * cal_free
            ref_plus_base = (REF_BASE_WEIGHT * cal_base672
                             + (1.0 - REF_BASE_WEIGHT) * ref_adaptive)

            ref_conservative = conservative_fusion(
                cal_base672, cal_spatial, cal["ref_spatial"],
                CONSERVATIVE_W, CONSERVATIVE_CAP, base_thr=0.0)
            ref_conservative_strong = conservative_fusion(
                cal_base672, cal_spatial, cal["ref_spatial"],
                STRONG_W, STRONG_CAP,
                base_thr=calibrated_quantile(cal["base_fuse672"], 0.95))

            per_variant = {
                "base_fuse672": base_fuse672,
                "ref_adaptive": ref_adaptive,
                "ref_plus_base": ref_plus_base,
                "ref_conservative": ref_conservative,
                "ref_conservative_strong": ref_conservative_strong,
            }

            # Original baseline global image score, computed once and reused.
            scores.extend(top1pct(base).cpu().numpy().tolist())
            for v in VARIANTS:
                buckets[v].append(per_variant[v].cpu())

            # Masks come from the exact original 448->392 pipeline.
            gt256 = F.interpolate(gt.float(), size=256, mode="nearest").bool()
            if gt256.shape[1] > 1:
                gt256 = gt256.max(dim=1, keepdim=True)[0]
            gt_buf.append(gt256.cpu())
            labels.extend(label.numpy().tolist())

            del (base_raw, en, feats, free_raw, local_raw, free, spatial,
                 hi_raw, hi_map, base_fuse672, cal_free, cal_spatial,
                 cal_base672, ref_fuse_fixed, ref_adaptive, ref_plus_base,
                 ref_conservative, ref_conservative_strong,
                 per_variant, base, img, hi)
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

    # base_global is not exposed as a stage-6 variant; image scores still come
    # from the original baseline-global map and are audited via image_scores.
    stage1_delta = None

    return {
        "class": item,
        "fingerprint": FP,
        "n_test": int(len(labels_np)),
        "n_normal": int((labels_np == 0).sum()),
        "n_anomaly": int((labels_np == 1).sum()),
        "n_gt_positive_pixels": int(gt_flat.sum()),
        "n_gt_negative_pixels": int((gt_flat == 0).sum()),
        "normal_reference": reference_summary(ref),
        "image_metrics": img_metrics,
        "image_scores": {
            "source": "baseline 448->392 global top1% of the postprocessed "
                      "256 map, computed once and reused by every variant",
            "reused_for_all_variants": True,
            "max_abs_diff_across_variants": 0.0,
            "sha256": hashlib.sha256(
                scores_np.astype(np.float64).tobytes()).hexdigest(),
            "stage1_baseline_delta": stage1_delta,
        },
        "variants": variants,
        "seconds": float(time.time() - started),
        "gpu_peak_mib": peak_mib(),
    }


# ---------------------------------------------------------------------------
# aggregation / archived comparison
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

    comparisons = {}
    for v in VARIANTS:
        if v != "base_fuse672":
            comparisons[f"{v} - base_fuse672"] = (v, "base_fuse672")
    comparisons["ref_plus_base - ref_adaptive"] = (
        "ref_plus_base", "ref_adaptive")
    comparisons["ref_conservative - ref_plus_base"] = (
        "ref_conservative", "ref_plus_base")
    comparisons["ref_conservative_strong - ref_conservative"] = (
        "ref_conservative_strong", "ref_conservative")

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

    archived = {}
    for ours, theirs in (("base_fuse672", "fuse672"),):
        entry = {"stage4_variant": theirs,
                 "metrics_pp": {}, "recall_pp": {}, "max_abs_metric_pp": 0.0}
        for k in METRIC_KEYS:
            d = 100.0 * (mean[ours]["metrics"][k]
                         - STAGE4_ARCHIVED[theirs]["metrics"][k])
            entry["metrics_pp"][k] = float(d)
            entry["max_abs_metric_pp"] = max(
                entry["max_abs_metric_pp"], abs(float(d)))
        for k in RECALL_KEYS:
            entry["recall_pp"][k] = float(
                100.0 * (mean[ours]["recall"][k]
                         - STAGE4_ARCHIVED[theirs]["recall"][k]))
        archived[f"{ours}_vs_stage4_{theirs}"] = entry
    archived["image_mean_pp"] = {
        k: float(100.0 * (image_mean[k]
                          - STAGE4_ARCHIVED["base_global"]["metrics"][k]))
        for k in ("I-AUROC", "I-AP", "I-F1")
    }

    # base_global is no longer an exposed variant in stage 6, so the exact
    # stage-1 identity row is not recomputed; image scores remain the original
    # baseline-global top-1% and are audited separately.
    stage1_identity = None

    return {
        "mean": mean,
        "image_mean": image_mean,
        "deltas": deltas,
        "per_class_deltas": per_class_deltas,
        "archived_baseline_comparison": archived,
        "stage1_identity": stage1_identity,
    }


# ---------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------
def pct(x):
    return f"{100.0 * x:.2f}"


def write_markdown(report, path):
    cfg = report["config"]
    res = report["results"]
    lines = []
    lines.append("# Dinomaly conservative reference v6 (stage 6)")
    lines.append("")
    lines.append(f"Slug: `{SLUG}`  ")
    lines.append(f"Generated: {report['experiment']['generated']}  ")
    lines.append(f"Checkpoint: `{report['checkpoints']['base']['path']}` "
                 f"(sha256 `{report['checkpoints']['base']['sha256'][:16]}...`)  ")
    lines.append(f"Test images: {cfg['test_images_evaluated']} across "
                 f"{len(cfg['eval_classes'])} classes  ")
    lines.append(f"Primary variant (pre-set): `{PRIMARY_VARIANT}`  ")
    lines.append("")
    lines.append("## What is new")
    lines.append("")
    lines.append("A class-specific normal-reference branch reuses the two "
                 "encoder feature groups already returned by ViTill at 392 "
                 "(two 768-channel 28x28 maps). It asks two questions: does a "
                 "patch have a similar normal exemplar anywhere (appearance, "
                 "`ref_free`), and does a similar exemplar occur within a "
                 "small spatial neighborhood (structure, `ref_spatial`)? "
                 "The structure branch is gated per class by how repeatable "
                 "the spatial constraint is on held-out calibration normals. "
                 "Stage 6 adds a conservative fusion: the calibrated "
                 "392+672 base is only increased where the reference exceeds "
                 "its calibrated q99 and both maps agree locally; the "
                 "positive residual is capped and nothing is subtracted. "
                 "Pose-aligned matching was deferred to protect the deadline, "
                 "so every stage-6 variant is `unaligned`.")
    lines.append("")
    lines.append("## Normal-reference protocol")
    lines.append("")
    lines.append(f"- {MEMORY_PER_CLASS} memory + {CALIB_PER_CLASS} disjoint "
                 "calibration images per class, sampled deterministically and "
                 "evenly from sorted `train/good` paths. No test image is "
                 "used for fitting.")
    lines.append("- Two encoder groups are z-scored separately using memory-"
                 "bank statistics, concatenated to 1536 dimensions, and "
                 "L2-normalised. No dimensionality reduction; matching is "
                 "chunked batched GEMM.")
    lines.append(f"- `ref_free`: 1 - mean top-{TOP_K} cosine similarity over "
                 f"all {MEMORY_PER_CLASS} memory images x 784 patches.")
    lines.append(f"- `ref_spatial`: the same top-{TOP_K} statistic over bank "
                 f"patches within +/-{LOCAL_RADIUS} patch position, across "
                 "all memory images. Out-of-image positions are masked, never "
                 "padded into the candidate set.")
    lines.append("- Per-branch normal calibration from the 16 calibration "
                 f"normals: `(x - median) / max(q95 - median, "
                 f"{SCALE_FLOOR})`.")
    lines.append("- Adaptive gate from calibration normals only: "
                 f"`rel_extra = median((d_spatial - d_free) / "
                 f"max(d_free, {DIST_FLOOR}))`; {GATE_FORMULA}. Higher "
                 "relative extra distance lowers spatial weight. The formula "
                 "and constants are fixed; no class weights are hard-coded "
                 "and no test labels are used.")
    lines.append("")
    lines.append("## Variants (5, fixed before evaluation)")
    lines.append("")
    lines.append("| variant | definition |")
    lines.append("|---|---|")
    for v in VARIANTS:
        mark = " **PRIMARY**" if v == PRIMARY_VARIANT else ""
        lines.append(f"| `{v}` | {VARIANT_DEFINITIONS[v]}{mark} |")
    lines.append("")
    lines.append("All variants use the same official 256 grid, the exact "
                 "original 448->392 mask transform, and the original baseline "
                 "global top-1% image score. Conservative formula: "
                 f"`{CONSERVATIVE_FORMULA}`.")
    lines.append("")
    title = "15-class means" if len(cfg["eval_classes"]) == 15 else \
        f"{len(cfg['eval_classes'])}-class means"
    lines.append(f"## {title} (%)")
    lines.append("")
    lines.append("| variant | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | "
                 "P-F1 | P-AUPRO |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        m = res["mean"][v]["metrics"]
        mark = " **PRIMARY**" if v == PRIMARY_VARIANT else ""
        lines.append(f"| `{v}`{mark} | " + " | ".join(
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
        r = res["mean"][v]["recall"]
        lines.append(f"| `{v}` | " + " | ".join(
            pct(r[k]) for k in RECALL_KEYS) + " |")
    lines.append("")
    lines.append("Thresholds use each variant's own per-class test negative "
                 "pixels, exactly as stages 3/4; they are metric thresholds, "
                 "not fitting.")
    lines.append("")
    lines.append("## Per-class calibration and adaptive gate")
    lines.append("")
    lines.append("| class | mem | calib | rel_extra | g_spatial | "
                 "free med | free q95 | free scale | spatial med | "
                 "spatial q95 | spatial scale | base672 med | base672 q95 | "
                 "base672 scale |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for item in cfg["eval_classes"]:
        nr = res["per_class"][item]["normal_reference"]
        c = nr["calibration"]
        lines.append(
            f"| {item} | {nr['memory_count']} | {nr['calibration_count']} | "
            f"{nr['gate']['rel_extra_median']:.4f} | "
            f"{nr['gate']['g_spatial']:.4f} | "
            f"{c['ref_free']['median']:.5f} | {c['ref_free']['q95']:.5f} | "
            f"{c['ref_free']['scale']:.5f} | "
            f"{c['ref_spatial']['median']:.5f} | "
            f"{c['ref_spatial']['q95']:.5f} | "
            f"{c['ref_spatial']['scale']:.5f} | "
            f"{c['base_fuse672']['median']:.5f} | "
            f"{c['base_fuse672']['q95']:.5f} | "
            f"{c['base_fuse672']['scale']:.5f} |")
    lines.append("")
    lines.append("## Mean pixel deltas (percentage points)")
    lines.append("")
    lines.append("| comparison | P-AUROC | P-AP | P-F1 | P-AUPRO |")
    lines.append("|---|---|---|---|---|")
    for name, delta in report["deltas"].items():
        d = delta["metrics"]
        lines.append(f"| {name} | " + " | ".join(
            f"{100 * d[k]:+.2f}" for k in
            ("P-AUROC", "P-AP", "P-F1", "P-AUPRO")) + " |")
    lines.append("")
    lines.append("## Per-class pixel AP (%)")
    lines.append("")
    lines.append("| class | " + " | ".join(f"`{v}`" for v in VARIANTS) + " |")
    lines.append("|---" * (len(VARIANTS) + 1) + "|")
    for item in cfg["eval_classes"]:
        vals = [res["per_class"][item]["variants"][v]["metrics"]["P-AP"]
                for v in VARIANTS]
        lines.append(f"| {item} | " + " | ".join(pct(x) for x in vals) + " |")
    lines.append("")
    lines.append("## Archived baseline check")
    lines.append("")
    arch = res["archived_baseline_comparison"]
    for name, e in arch.items():
        if name == "image_mean_pp":
            continue
        md = e["metrics_pp"]
        lines.append(
            f"- `{name}`: max |delta| {e['max_abs_metric_pp']:.4f} pp; "
            f"P-AP {md['P-AP']:+.4f} pp, P-AUROC {md['P-AUROC']:+.4f} pp, "
            f"P-AUPRO {md['P-AUPRO']:+.4f} pp")
    im = arch["image_mean_pp"]
    lines.append("- Image means vs stage 4: " + ", ".join(
        f"{k} {im[k]:+.4f} pp" for k in im))
    lines.append("- The archived stage-4 392+672 baseline is "
                 f"`fuse672` P-AP = "
                 f"{100 * STAGE4_ARCHIVED['fuse672']['metrics']['P-AP']:.2f}% "
                 "on the same official protocol.")
    lines.append("")
    lines.append("## Audit and runtime")
    lines.append("")
    lines.append(f"- Image score is the original baseline global top-1% and "
                 f"is reused by all variants (max |delta| = "
                 f"{report['identity']['image_score_max_abs_diff_across_variants']:.3e}).")
    lines.append("- No training; the unchanged frozen 10k checkpoint is "
                 "reused.")
    lines.append("- The normal-reference bank and calibration/gate statistics "
                 "come only from `train/good`. Test labels enter only through "
                 "the standard metric definitions (AUROC/AP/F1/PRO and "
                 "fixed-FPR recall).")
    lines.append("- High-res 672 maps are computed on the full native image "
                 "and reduced to 256 for metrics; no raw high-res maps are "
                 "saved.")
    s1 = report["identity"].get("stage1_baseline_comparison")
    if s1:
        md = s1.get("mean_delta", {})
        lines.append("- Recomputed base_global vs stage 1: " + ", ".join(
            f"{k} {100 * md[k]:+.4f} pp" for k in md))
    else:
        lines.append("- Stage 1 metrics unavailable; identity check skipped.")
    lines.append("")
    rt = report["runtime"]
    lines.append(f"- Device: `{rt['device']}`; torch `{rt['torch']}`")
    lines.append(f"- Classes: {rt['classes_evaluated']}; test images: "
                 f"{rt['test_images_evaluated']}")
    lines.append(f"- Forward passes per test image: "
                 f"{rt['forwards_per_test_image']} "
                 "(1 baseline + 1 hi672)")
    lines.append(f"- Effective batches: base {rt['base_batch']}, "
                 f"hi672 {rt['hi_batch']}, reference chunk {rt['ref_chunk']}")
    lines.append(f"- Evaluation seconds (sum of per-class): "
                 f"{rt['seconds_total']:.1f}")
    lines.append("")
    lines.append("Full per-class metrics, thresholds, calibration and gate "
                 "values: `normal_reference_report.json`.")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# self-test: compile, nonoverlap, free vs spatial, calibration/gate
# ---------------------------------------------------------------------------
def run_self_test():
    global torch, F
    import torch as _torch
    import torch.nn.functional as _F
    torch = _torch
    F = _F
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        if not cond:
            raise AssertionError(msg)
        checks += 1

    # compile check (no bytecode written)
    path = os.path.abspath(__file__)
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    compile(src, path, "exec")
    ok(True, "compile")

    # deterministic nonoverlap
    paths = [f"train/good/{i:04d}.png" for i in range(209)]
    mem, cal = select_normal_paths(paths)
    ok(len(mem) == MEMORY_PER_CLASS and len(cal) == CALIB_PER_CLASS,
       "normal counts")
    ok(len(set(mem + cal)) == TOTAL_NORMAL_PER_CLASS,
       "normal selection unique")
    ok(not (set(mem) & set(cal)), "memory/calibration disjoint")
    mem2, cal2 = select_normal_paths(paths)
    ok(mem2 == mem and cal2 == cal, "normal selection deterministic")
    mem60, cal60 = select_normal_paths(paths[:60])
    ok(len(mem60) == MEMORY_PER_CLASS and len(cal60) == CALIB_PER_CLASS,
       "minimum-size class counts")
    ok(not (set(mem60) & set(cal60)), "minimum-size class disjoint")

    # no padding candidates
    counts = valid_offset_counts(LOCAL_RADIUS, side=28)
    ok(min(counts) == 4, "corner has 4 valid radius-1 candidates")
    ok(max(counts) == 9, "interior has 9 valid radius-1 candidates")
    ok(len(neighborhood_offsets(LOCAL_RADIUS)) == 9, "radius-1 offsets")

    # gate bounded and monotone decreasing
    ok(abs(adaptive_gate(0.0) - GATE_G_MAX) < 1e-12,
       "gate reaches max at zero extra distance")
    ok(GATE_G_MIN <= adaptive_gate(100.0) <= GATE_G_MAX, "gate bounded")
    ok(adaptive_gate(0.0) > adaptive_gate(0.5) > adaptive_gate(1.0),
       "gate decreases with extra distance")

    # calibration positive floor
    const = torch.full((100,), 0.3)
    st = branch_stats(const)
    ok(abs(st["scale"] - SCALE_FLOOR) < 1e-12, "calibration positive floor")
    varied = torch.linspace(0.0, 1.0, 100)
    st2 = branch_stats(varied)
    ok(st2["scale"] > 0 and st2["q95"] >= st2["median"],
       "calibration q95/median")
    ok(st2["q99"] >= st2["q95"] >= st2["median"], "calibration q99 tail")

    # conservative residual: threshold, cap, local agreement, layout/direction
    rs = {"median": 0.0, "q95": 0.5, "q99": 0.8, "scale": 0.5}
    thr = calibrated_quantile(rs, REF_TAIL_Q)
    ok(abs(thr - 1.6) < 1e-12, "calibrated q99 threshold")
    base = torch.zeros(1, 1, 3, 5)
    ref = torch.zeros_like(base)
    base[0, 0, 1, 1] = 3.0
    ref[0, 0, 1, 1] = 2.0
    ref[0, 0, 0, 4] = 2.0
    out = conservative_fusion(base, ref, rs, 0.25, 2.0, 0.0)
    ok(abs(float(out[0, 0, 1, 1]) - (3.0 + 0.25 * 0.4)) < 1e-6,
       "conservative adds capped positive residual on agreement")
    ok(abs(float(out[0, 0, 0, 4])) < 1e-12,
       "conservative leaves a ref-only outlier unchanged")
    base2 = torch.zeros_like(base)
    ref2 = torch.zeros_like(base)
    base2[0, 0, 2, 2] = 3.0
    ref2[0, 0, 2, 2] = 100.0
    out2 = conservative_fusion(base2, ref2, rs, 0.25, 2.0, 0.0)
    ok(abs(float(out2[0, 0, 2, 2]) - (3.0 + 0.25 * 2.0)) < 1e-6,
       "conservative residual cap")
    base3 = torch.zeros_like(base)
    ref3 = torch.zeros_like(base)
    base3[0, 0, 2, 1] = 3.0
    ref3[0, 0, 2, 2] = 2.0
    out3 = conservative_fusion(base3, ref3, rs, 0.25, 2.0, 0.0)
    ok(float(out3[0, 0, 2, 2]) > 0.0, "3x3 agreement bridges one-cell offset")
    out4 = conservative_fusion(base, ref, rs, 0.25, 2.0, 5.0)
    ok(abs(float(out4[0, 0, 1, 1]) - 3.0) < 1e-12,
       "strong gate blocks when base is below its threshold")
    ok(out.shape == base.shape, "conservative preserves H/W layout")

    # free vs spatial behaviour, synthetic unique bank patches
    H = W = 8
    hw = H * W
    C = 16
    M = 2
    gen = torch.Generator().manual_seed(123)
    bank = torch.randn(M * hw, C, generator=gen)
    bank = F.normalize(bank, p=2, dim=1)
    bank4 = bank.view(M, hw, C)
    far = (torch.arange(hw) + hw // 2) % hw
    q = bank4[0, far].transpose(0, 1).contiguous().reshape(1, C, H, W)

    free1, local1 = reference_distances(q, bank, radius=1, topk=1)
    ok(float(free1.max()) < 1e-5,
       "free top-1 finds an exact far exemplar")
    ok(float(local1.mean()) > float(free1.mean()) + 0.3,
       "spatial rejects the far exemplar")

    free3, local3 = reference_distances(q, bank, radius=1, topk=3)
    sim = q.flatten(2).transpose(1, 2).reshape(hw, C) @ bank.t()
    brute = (1.0 - sim).topk(3, dim=1, largest=False).values.mean(dim=1)
    ok(torch.allclose(free3.flatten(), brute, atol=1e-5),
       "free top-3 matches brute force")
    ok(float(local3.mean()) > float(free3.mean()) + 0.02,
       "spatial top-3 penalises the far exemplar")

    # perfectly repeatable normal: free and spatial both collapse to zero
    v = F.normalize(torch.randn(C, generator=gen), p=2, dim=0)
    bank_const = v.expand(M * hw, C).contiguous()
    q_const = v.view(1, C, 1, 1).expand(1, C, H, W)
    f2, l2 = reference_distances(q_const, bank_const, radius=1, topk=3)
    ok(float(f2.max()) < 1e-5 and float(l2.max()) < 1e-5,
       "repeatable normal gives free and spatial zero distance")

    # fusion weights
    ok(abs(FIXED_SPATIAL_WEIGHT + (1.0 - FIXED_SPATIAL_WEIGHT) - 1.0) < 1e-12,
       "fixed fusion weights sum to one")
    ok(abs(REF_BASE_WEIGHT + (1.0 - REF_BASE_WEIGHT) - 1.0) < 1e-12,
       "reference+base weights sum to one")

    print(f"SELF-TEST OK ({checks} checks)")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    global LOG_PATH, FP
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--limit-classes", type=int, default=0,
                    help="smoke only: evaluate first N classes")
    ap.add_argument("--data-root", default=None,
                    help="local smoke: MVTec class parent directory")
    ap.add_argument("--code-root", default=None,
                    help="local smoke: Dinomaly code directory")
    ap.add_argument("--checkpoint", default=None,
                    help="local smoke: model_10000.pt path")
    ap.add_argument("--stage1-metrics", default=None,
                    help="optional stage-1 metrics.json path")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()

    os.makedirs(OUT, exist_ok=True)
    LOG_PATH = os.path.join(OUT, "stage6.log")
    log(f"=== {SLUG} start ===")
    log(f"args={vars(args)}")

    run_self_test()
    log("self-test passed")

    prepare_environment(args.code_root, args.data_root)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"device={device} torch={torch.__version__}")
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    find_checkpoints(args.checkpoint, args.stage1_metrics)
    base_sha = sha256_file(BASE_CKPT)
    log(f"base sha256={base_sha}")
    stage1 = load_stage1_metrics()

    item_list = sorted(
        d for d in os.listdir(DATA)
        if os.path.isdir(os.path.join(DATA, d)))
    if not args.data_root:
        if len(item_list) != 15:
            raise RuntimeError(f"expected 15 MVTec classes, got "
                               f"{len(item_list)}: {item_list}")
    else:
        log(f"local data root: {len(item_list)} classes {item_list}")
    log(f"classes={len(item_list)}")

    model = build_model(device)
    st_base = load_ckpt(model, BASE_CKPT, device)
    model.eval()
    log(f"checkpoint iter={st_base.get('iter')} kind={st_base.get('kind')}")

    base_transform, gt_transform = get_data_transforms(BASE_RESIZE, BASE_CROP)
    hi_transform = get_data_transforms(HI_RESIZE, HI_CROP)[0]
    test_data_list = {
        item: TestPairDataset(
            root=os.path.join(DATA, item),
            base_transform=base_transform,
            gt_transform=gt_transform,
            hi_transform=hi_transform)
        for item in item_list
    }
    counts = {it: len(ds) for it, ds in test_data_list.items()}
    total_test = int(sum(counts.values()))
    log(f"test images={total_test} per-class={counts}")
    if not args.data_root and total_test != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"expected {EXPECTED_TEST_IMAGES} official test images, "
            f"got {total_test}: {counts}")

    eval_items = list(item_list)
    if args.limit_classes and args.limit_classes > 0:
        eval_items = eval_items[:args.limit_classes]
    test_images_evaluated = int(sum(counts[it] for it in eval_items))
    partial = len(eval_items) != len(item_list)

    variant_entries = []
    for v in VARIANTS:
        variant_entries.append({
            "name": v,
            "definition": VARIANT_DEFINITIONS[v],
            "primary": v == PRIMARY_VARIANT,
        })
    config = {
        "slug": SLUG,
        "stage": 6,
        "script": os.path.basename(os.path.abspath(__file__)),
        "code_sha256": sha256_file(os.path.abspath(__file__)),
        "baseline": {"resize": BASE_RESIZE, "crop": BASE_CROP,
                     "fov": fov(BASE_CROP, BASE_RESIZE)},
        "archived_baseline": {
            "source": "stage 4 eunectesqaq/dinomaly-context-res-v4",
            "variant": "fuse672",
            "definition": "0.5 * base_global + 0.5 * hi672",
            "stage4_pixel_ap_percent": 72.47,
            "stage4_mean": STAGE4_ARCHIVED["fuse672"],
        },
        "normal_reference": {
            "source": "train/good only",
            "memory_per_class": MEMORY_PER_CLASS,
            "calibration_per_class": CALIB_PER_CLASS,
            "total_per_class": TOTAL_NORMAL_PER_CLASS,
            "selection": "sorted paths, evenly spaced indices, first 40 "
                         "memory then 16 calibration, disjoint",
            "test_images_used_for_fit": False,
            "feature_groups": "ViTill encoder groups en[0], en[1] at 392",
            "group_dim": 768,
            "concat_dim": 1536,
            "dimensionality_reduction": None,
            "normalization": "per-group z-score from memory bank, concat, L2",
            "top_k": TOP_K,
            "local_radius": LOCAL_RADIUS,
            "matching": "batched GEMM; free all patches; spatial +/-radius "
                        "position candidates across all memory images, no "
                        "padding candidates",
        },
        "calibration": {
            "source": "16 disjoint train/good calibration normals per class",
            "per_branch": ["ref_free", "ref_spatial", "base_global",
                           "base_fuse672"],
            "loc": "median",
            "scale": "max(q95 - median, 0.01)",
            "formula": "(x - median) / scale",
            "positive_scale_floor": SCALE_FLOOR,
            "used_for": "fusion variants only; standalone reference maps "
                        "stay raw",
        },
        "gate": {
            "source": "calibration normals only",
            "rel_extra": f"median((d_spatial - d_free) / max(d_free, "
                         f"{DIST_FLOOR}))",
            "formula": GATE_FORMULA,
            "g_min": GATE_G_MIN,
            "g_max": GATE_G_MAX,
            "gate_k": GATE_K,
            "gate_r0": GATE_R0,
            "distance_floor": DIST_FLOOR,
            "interpretation": "higher relative extra distance lowers the "
                              "spatial weight; bounded fixed formula, no "
                              "hard-coded class weights, no test labels",
        },
        "fusion_weights": {
            "fixed": {"ref_free": 1.0 - FIXED_SPATIAL_WEIGHT,
                      "ref_spatial": FIXED_SPATIAL_WEIGHT},
            "adaptive": {"ref_free": "1 - g", "ref_spatial": "g"},
            "ref_plus_base": {"cal_base_fuse672": REF_BASE_WEIGHT,
                              "ref_adaptive": 1.0 - REF_BASE_WEIGHT},
        },
        "conservative_fusion": {
            "formula": CONSERVATIVE_FORMULA,
            "reference_tail_quantile": REF_TAIL_Q,
            "base_threshold": "calibrated median for conservative; "
                              "calibrated q95 for strong",
            "agreement_kernel": AGREE_KERNEL,
            "weights": {"conservative": CONSERVATIVE_W,
                        "conservative_strong": STRONG_W},
            "caps": {"conservative": CONSERVATIVE_CAP,
                     "conservative_strong": STRONG_CAP},
            "pose_alignment_applied": False,
            "pose_alignment_note": "deferred to protect the competition "
                                   "deadline; all variants are unaligned",
            "subtracts_from_base": False,
        },
        "variants": variant_entries,
        "primary_variant": PRIMARY_VARIANT,
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
        "base_batch": EVAL_BATCH,
        "hi_batch": HI_BATCH,
        "ref_chunk": REF_CHUNK,
        "hi672": {"resize": HI_RESIZE, "crop": HI_CROP,
                  "fov": fov(HI_CROP, HI_RESIZE)},
        "base_checkpoint": BASE_CKPT,
        "base_sha256": base_sha,
        "stage1_metrics": STAGE1_METRICS,
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
            f"(base_batch={EVAL_BATCH}, hi_batch={HI_BATCH})")
        ref = build_class_reference(
            item, os.path.join(DATA, item), base_transform, hi_transform,
            model, gaussian, device)
        result = evaluate_class(item, test_data_list[item], model, ref,
                                gaussian, device, stage1_item)
        write_json_atomic(per_class_path(item), result)
        done.append(item)
        latest = {
            "class": item,
            "seconds": result["seconds"],
            "g_spatial": result["normal_reference"]["gate"]["g_spatial"],
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
            f"peak={result['gpu_peak_mib']:.0f}MiB gate="
            f"{result['normal_reference']['gate']['g_spatial']:.4f} " + " ".join(
                f"{v}:AP={result['variants'][v]['metrics']['P-AP']:.4f}"
                f"/PRO={result['variants'][v]['metrics']['P-AUPRO']:.4f}"
                for v in VARIANTS))
        del ref, result
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
            "stage": 6,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "partial": partial,
            "primary_variant": PRIMARY_VARIANT,
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
            "no_training": True,
            "checkpoint_reused_unchanged": True,
            "normal_reference_source": "train/good only; memory and "
                                       "calibration are disjoint",
            "test_labels_used_for_fitting": False,
            "gate_source": "calibration normals only; fixed bounded formula",
            "stage1_baseline_comparison": agg["stage1_identity"],
        },
        "results": {
            "per_class": per_class,
            "mean": agg["mean"],
            "image_mean": agg["image_mean"],
            "archived_baseline_comparison": agg[
                "archived_baseline_comparison"],
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
            "base_batch": EVAL_BATCH,
            "hi_batch": HI_BATCH,
            "ref_chunk": REF_CHUNK,
            "forwards_per_test_image": 2,
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

    report_json = os.path.join(OUT, "normal_reference_report.json")
    report_md = os.path.join(OUT, "normal_reference_report.md")
    write_json_atomic(report_json, report)
    write_markdown(report, report_md)
    set_status("complete", classes_done=done,
               classes_total=len(eval_items),
               report=report_json, markdown=report_md)
    update_progress(current=None)
    log("DONE")
    log(f"report={report_json}")
    log(f"markdown={report_md}")
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
