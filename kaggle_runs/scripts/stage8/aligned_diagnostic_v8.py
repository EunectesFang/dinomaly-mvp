"""Stage 8: targeted per-reference-image shift alignment diagnostic.

TARGETED DIAGNOSTIC, NOT A FULL BENCHMARK
-----------------------------------------
Only four MVTec classes are evaluated (578 official test images):
``transistor`` (v5 spatial win), ``zipper`` (v5 fused loss),
``pill`` (v5 reference loss), ``screw`` (v5 spatial collapse).  The official
full 15-class / 1725-image test set is still constructed and counted first, so
the selected class samples are provably the official ones; the run is then
restricted to the four classes and every report marks itself as diagnostic.

Core independent variable
-------------------------
Stage 5's ``ref_spatial`` candidate set is replaced.  Instead of always
comparing query patch ``(r, c)`` with the same-coordinate +/-1 neighbourhood
across every memory image, each (test image, memory image) pair first selects
one coherent patch shift ``(sr, sc)`` in ``[-2, 2]^2``.  The shift is the
argmax of the mean cosine similarity over all in-image query positions, minus
a 0.005 * (|sr| + |sc|) Manhattan prior.  Candidates are then the +/-1
neighbourhood around the shifted coordinate, so local pose tolerance is
preserved.  Out-of-image positions are always masked.  If a query row has
fewer than top-k finite candidates under the selected shift (possible at
image corners), that row falls back to the original zero-shift +/-1
neighbourhood.  The free (appearance) search is unchanged.

This reuses the existing ``[B, HW, M, HW]`` similarity tensor.  No new
forward pass, no new embedding, and no test image is used for fitting.

Reference branch and fusion
---------------------------
40 deterministic train/good memory images + 16 disjoint train/good
calibration images per class.  The two ViTill encoder groups (768 each) are
z-scored from the memory bank, concatenated to 1536, and L2-normalised.
Free matching is the unchanged stage-5 top-3 statistic.  The aligned spatial
branch is calibrated on the disjoint calibration normals with the exact stage
5 formula ``(x - median) / max(q95 - median, 0.01)``.  The stage-5 adaptive
gate is also recomputed for the aligned branch from calibration normals:

    rel_extra = median( (d_aligned - d_free) / max(d_free, 0.02) )
    g         = 0.15 + 0.35 * exp(-rel_extra / 0.25), clipped to [0.15, 0.5]

Variants (4, fixed before evaluation)
-------------------------------------
base_fuse672         0.5 * base_global + 0.5 * hi672 (392+672 control)
ref_spatial_aligned  raw aligned spatial normal-reference map (standalone)
ref_adaptive_aligned (1-g) * calib(free) + g * calib(aligned spatial)
aligned_plus_base    0.5 * calib(base_fuse672) + 0.5 * ref_adaptive_aligned
                     (stage-5 ref_plus_base formula with the aligned branch;
                     PRIMARY)

The downloaded stage-5 per-class JSON is embedded as an unchanged control.
Only the four control classes are compared; nothing is fitted on test data.
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
SLUG = "eunectesqaq/dinomaly-aligned-diagnostic-v8"
DIAGNOSTIC_CLASSES = ("transistor", "zipper", "pill", "screw")
DIAGNOSTIC_NOTE = ("targeted four-class diagnostic (578 official test "
                   "images), explicitly not a full 15-class benchmark")
BASE_RESIZE = 448
BASE_CROP = 392
HI_RESIZE = 768
HI_CROP = 672
EXPECTED_TEST_IMAGES = 1725
EXPECTED_DIAGNOSTIC_IMAGES = 578
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
SHIFT_RADIUS = 2
SHIFT_PRIOR = 0.005
SCALE_FLOOR = 0.01
DIST_FLOOR = 0.02
GATE_G_MIN = 0.15
GATE_G_MAX = 0.50
GATE_K = 1.0
GATE_R0 = 0.25
FIXED_SPATIAL_WEIGHT = 0.5
REF_BASE_WEIGHT = 0.5

VARIANTS = (
    "base_fuse672",
    "ref_spatial_aligned",
    "ref_adaptive_aligned",
    "aligned_plus_base",
)
PRIMARY_VARIANT = "aligned_plus_base"

VARIANT_DEFINITIONS = {
    "base_fuse672": "archived stage-4 392+672 baseline: "
                    "0.5 * base_global + 0.5 * hi672 (control)",
    "ref_spatial_aligned": "raw aligned spatial normal-reference map: "
                           "1 - mean top-3 cosine sim around the per-"
                           "(test image, memory image) coherent shift "
                           "selected in [-2,2]^2, +/-1 local tolerance",
    "ref_adaptive_aligned": "(1-g) * calib(ref_free) + "
                            "g * calib(ref_spatial_aligned); g from aligned "
                            "calibration normals with the exact stage-5 gate",
    "aligned_plus_base": "0.5 * calib(base_fuse672) + "
                         "0.5 * ref_adaptive_aligned (stage-5 ref_plus_base "
                         "formula with the aligned branch; PRIMARY)",
}

ALIGNMENT_DESCRIPTION = (
    "per (test image, memory image) coherent shift (sr,sc) in [-2,2]^2; "
    "shift score = mean cosine similarity over in-image query positions "
    "minus 0.005*(|sr|+|sc|); candidates = +/-1 neighbourhood around the "
    "shifted coordinate; out-of-image positions masked; row falls back to "
    "the original zero-shift neighbourhood when fewer than top-3 finite "
    "candidates remain")

GATE_FORMULA = ("g = 0.15 + 0.35 * exp(-max(rel_extra, 0) / 0.25), "
                "clipped to [0.15, 0.50]; "
                "rel_extra = median((d_aligned - d_free) / "
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

# Downloaded stage-5 per-class results for the four diagnostic classes.  These
# are an unchanged control: they are read from Kaggle's completed v5 run and
# are never used for fitting, calibration or variant selection.
V5_CONTROL = {
    "transistor": {
        "n_test": 100,
        "base_fuse672": {
            "metrics": {
                "I-AUROC": 0.99125, "I-AP": 0.983562569235511,
                "I-F1": 0.9749999500000025,
                "P-AUROC": 0.9248275333413035,
                "P-AP": 0.5895012291765824,
                "P-F1": 0.5663132803352096,
                "P-AUPRO": 0.7699350571586548},
            "recall": {"R@FPR0.001": 0.1300812033536242,
                       "R@FPR0.01": 0.39600000959000337,
                       "R@FPR0.05": 0.642614139021484}},
        "ref_spatial": {
            "metrics": {
                "I-AUROC": 0.99125, "I-AP": 0.983562569235511,
                "I-F1": 0.9749999500000025,
                "P-AUROC": 0.9529322558260097,
                "P-AP": 0.7437148158071508,
                "P-F1": 0.6845801878623365,
                "P-AUPRO": 0.8963697803015155},
            "recall": {"R@FPR0.001": 0.2491291078180105,
                       "R@FPR0.01": 0.5713843889129971,
                       "R@FPR0.05": 0.8012375899362505}},
        "ref_plus_base": {
            "metrics": {
                "I-AUROC": 0.99125, "I-AP": 0.983562569235511,
                "I-F1": 0.9749999500000025,
                "P-AUROC": 0.9525976212423726,
                "P-AP": 0.6847992555458892,
                "P-F1": 0.6505734016713519,
                "P-AUPRO": 0.8590391683115273},
            "recall": {"R@FPR0.001": 0.15565054986681884,
                       "R@FPR0.01": 0.49004677524148826,
                       "R@FPR0.05": 0.7727169198827143}},
    },
    "zipper": {
        "n_test": 151,
        "base_fuse672": {
            "metrics": {
                "I-AUROC": 1.0, "I-AP": 0.9999999999999998,
                "I-F1": 0.9999999500000026,
                "P-AUROC": 0.9955812541386245,
                "P-AP": 0.8671961556390471,
                "P-F1": 0.8143222353029622,
                "P-AUPRO": 0.980588373974324},
            "recall": {"R@FPR0.001": 0.4424409210633102,
                       "R@FPR0.01": 0.9081260744140063,
                       "R@FPR0.05": 0.9895011767830661}},
        "ref_spatial": {
            "metrics": {
                "I-AUROC": 1.0, "I-AP": 0.9999999999999998,
                "I-F1": 0.9999999500000026,
                "P-AUROC": 0.8271785784496536,
                "P-AP": 0.3665753563126447,
                "P-F1": 0.4127755933984812,
                "P-AUPRO": 0.6784522122525529},
            "recall": {"R@FPR0.001": 0.15473775075808438,
                       "R@FPR0.01": 0.3515261290108382,
                       "R@FPR0.05": 0.5104176657640974}},
        "ref_plus_base": {
            "metrics": {
                "I-AUROC": 1.0, "I-AP": 0.9999999999999998,
                "I-F1": 0.9999999500000026,
                "P-AUROC": 0.9810730688044449,
                "P-AP": 0.7549975827894198,
                "P-F1": 0.7056820277747335,
                "P-AUPRO": 0.947687534028262},
            "recall": {"R@FPR0.001": 0.33764820457580474,
                       "R@FPR0.01": 0.736754734799578,
                       "R@FPR0.05": 0.9225757899940239}},
    },
    "pill": {
        "n_test": 167,
        "base_fuse672": {
            "metrics": {
                "I-AUROC": 0.9920894708128751,
                "I-AP": 0.9986187195143849,
                "I-F1": 0.9858155528368819,
                "P-AUROC": 0.9754204475463726,
                "P-AP": 0.7425064490658501,
                "P-F1": 0.7093033833782402,
                "P-AUPRO": 0.97712044500069},
            "recall": {"R@FPR0.001": 0.1650509665019773,
                       "R@FPR0.01": 0.6620855512734078,
                       "R@FPR0.05": 0.8742953742742431}},
        "ref_spatial": {
            "metrics": {
                "I-AUROC": 0.9920894708128751,
                "I-AP": 0.9986187195143849,
                "I-F1": 0.9858155528368819,
                "P-AUROC": 0.8677015972901764,
                "P-AP": 0.5084259983535028,
                "P-F1": 0.5504685897412528,
                "P-AUPRO": 0.8937167107712383},
            "recall": {"R@FPR0.001": 0.09546684913311665,
                       "R@FPR0.01": 0.42732770504835027,
                       "R@FPR0.05": 0.6550840720876644}},
        "ref_plus_base": {
            "metrics": {
                "I-AUROC": 0.9920894708128751,
                "I-AP": 0.9986187195143849,
                "I-F1": 0.9858155528368819,
                "P-AUROC": 0.9577910282274725,
                "P-AP": 0.7087673546860803,
                "P-F1": 0.6858441403020662,
                "P-AUPRO": 0.9692415028194312},
            "recall": {"R@FPR0.001": 0.15560832771511085,
                       "R@FPR0.01": 0.6247959830547701,
                       "R@FPR0.05": 0.8460942452630837}},
    },
    "screw": {
        "n_test": 160,
        "base_fuse672": {
            "metrics": {
                "I-AUROC": 0.983398237343718,
                "I-AP": 0.9946887003395251,
                "I-F1": 0.9626555516675014,
                "P-AUROC": 0.994477720544336,
                "P-AP": 0.6592856246268033,
                "P-F1": 0.6475957672209693,
                "P-AUPRO": 0.9749371644660824},
            "recall": {"R@FPR0.001": 0.5741478418717225,
                       "R@FPR0.01": 0.9197256958450988,
                       "R@FPR0.05": 0.9759479628882614}},
        "ref_spatial": {
            "metrics": {
                "I-AUROC": 0.983398237343718,
                "I-AP": 0.9946887003395251,
                "I-F1": 0.9626555516675014,
                "P-AUROC": 0.9180159430490314,
                "P-AP": 0.12841266378333271,
                "P-F1": 0.22323813677069623,
                "P-AUPRO": 0.7658040825923749},
            "recall": {"R@FPR0.001": 0.11020068576038726,
                       "R@FPR0.01": 0.42259983864461476,
                       "R@FPR0.05": 0.6895169423154498}},
        "ref_plus_base": {
            "metrics": {
                "I-AUROC": 0.983398237343718,
                "I-AP": 0.9946887003395251,
                "I-F1": 0.9626555516675014,
                "P-AUROC": 0.98944801844117,
                "P-AP": 0.6796340636302691,
                "P-F1": 0.6528638569532407,
                "P-AUPRO": 0.9631443791149754},
            "recall": {"R@FPR0.001": 0.5946702299314239,
                       "R@FPR0.01": 0.8974384832593788,
                       "R@FPR0.05": 0.9644261799112546}},
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


def shift_offsets(shift_radius=SHIFT_RADIUS):
    r = int(shift_radius)
    return [(sr, sc) for sr in range(-r, r + 1)
            for sc in range(-r, r + 1)]


def valid_offset_counts(radius=LOCAL_RADIUS, side=28):
    offs = neighborhood_offsets(radius)
    counts = []
    for r in range(side):
        for c in range(side):
            counts.append(sum(1 for dr, dc in offs
                              if 0 <= r + dr < side and 0 <= c + dc < side))
    return counts


def merge_shift_diag(diags):
    """Sum per-chunk alignment diagnostics into one JSON-safe summary."""
    total_rows = sum(int(d["rows"]) for d in diags)
    fallback_rows = sum(int(d["fallback_rows"]) for d in diags)
    counts = {}
    for d in diags:
        for k, v in d["shift_counts"].items():
            counts[k] = counts.get(k, 0) + int(v)
    abs_sr = [float(d["mean_abs_sr"]) for d in diags if d["pairs"]]
    abs_sc = [float(d["mean_abs_sc"]) for d in diags if d["pairs"]]
    pairs = sum(int(d["pairs"]) for d in diags)
    zero = sum(int(d["zero_shift_pairs"]) for d in diags)
    invalid = sum(int(d["invalid_shift_score_pairs"]) for d in diags)
    return {
        "shift_radius": int(SHIFT_RADIUS),
        "shift_prior_manhattan": float(SHIFT_PRIOR),
        "shift_counts": counts,
        "pairs": int(pairs),
        "rows": int(total_rows),
        "fallback_rows": int(fallback_rows),
        "fallback_fraction": (float(fallback_rows) / float(total_rows)
                              if total_rows else 0.0),
        "zero_shift_pairs": int(zero),
        "zero_shift_fraction": (float(zero) / float(pairs)
                                if pairs else 0.0),
        "invalid_shift_score_pairs": int(invalid),
        "mean_abs_sr": (float(np.mean(abs_sr)) if abs_sr else 0.0),
        "mean_abs_sc": (float(np.mean(abs_sc)) if abs_sc else 0.0),
    }


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
    scale = max(q95 - median, SCALE_FLOOR)
    return {"median": median, "q95": q95, "scale": scale,
            "n_values": int(flat.numel())}


def calibrate_map(x, stats):
    return (x - float(stats["median"])) / float(stats["scale"])


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


def _shifted_index(rr, cc, sr, sc, H, W):
    """Flattened bank index and in-image mask for a shift applied to queries."""
    br = rr + int(sr)
    bc = cc + int(sc)
    valid = (br >= 0) & (br < H) & (bc >= 0) & (bc < W)
    pos = (br.clamp(0, H - 1) * W + bc.clamp(0, W - 1)).long()
    return pos, valid


def _gather_shifted(sim, rr, cc, sr, sc, H, W):
    """Gather one shifted same-coordinate candidate per (B, query, M)."""
    B, hw, M, _ = sim.shape
    pos, valid = _shifted_index(rr, cc, sr, sc, H, W)
    idx = pos.view(1, hw, 1, 1).expand(B, hw, M, 1).contiguous()
    s = torch.gather(sim, 3, idx).squeeze(-1)      # [B, hw, M]
    return s.masked_fill((~valid).view(1, hw, 1), float("-inf"))


def _gather_selected_neighborhood(sim, rr, cc, sel_sr, sel_sc, dr, dc, H, W):
    """Gather one +/- radius offset relative to each pair's selected shift.

    sel_sr / sel_sc: [B, M] per (test image, memory image) coherent shift.
    """
    B, hw, M, _ = sim.shape
    br = rr.view(1, hw, 1) + sel_sr[:, None, :] + int(dr)
    bc = cc.view(1, hw, 1) + sel_sc[:, None, :] + int(dc)
    valid = (br >= 0) & (br < H) & (bc >= 0) & (bc < W)
    pos = (br.clamp(0, H - 1) * W + bc.clamp(0, W - 1)).long()
    s = torch.gather(sim, 3, pos.unsqueeze(-1)).squeeze(-1)  # [B, hw, M]
    return s.masked_fill(~valid, float("-inf"))


def _zero_shift_candidates(sim, rr, cc, radius, H, W):
    """The original stage-5 +/-radius same-coordinate candidate set."""
    B, hw, M, _ = sim.shape
    gathered = []
    for dr, dc in neighborhood_offsets(radius):
        gathered.append(_gather_shifted(sim, rr, cc, dr, dc, H, W))
    return torch.stack(gathered, dim=3).reshape(B, hw, M * len(
        neighborhood_offsets(radius)))


def reference_distances(features, bank, radius=LOCAL_RADIUS, topk=TOP_K,
                        shift_radius=SHIFT_RADIUS, prior=SHIFT_PRIOR):
    """Free and shift-aligned spatial top-k cosine distances.

    features: [B, C, H, W] L2-normalised
    bank:     [M*H*W, C] L2-normalised, row-major (image, h, w)

    The free branch is exactly the stage-5 statistic.  The spatial branch
    chooses, for every (test image b, memory image m), one coherent shift
    (sr, sc) in [-shift_radius, shift_radius]^2 by maximising the mean cosine
    similarity over in-image query positions minus a Manhattan prior.  It then
    gathers the +/-radius neighbourhood around the shifted coordinate and
    takes the top-k mean.  Rows with fewer than top-k finite aligned candidates
    fall back to the original zero-shift +/-radius neighbourhood.

    returns free [B,H,W], local [B,H,W], diag dict
    """
    B, C, H, W = features.shape
    hw = H * W
    if bank.shape[0] % hw != 0:
        raise ValueError("bank patch count is not a multiple of H*W")
    M = bank.shape[0] // hw
    device = features.device

    q = features.flatten(2).transpose(1, 2).reshape(B * hw, C)
    sim = q @ bank.t()                      # batched GEMM
    sim = sim.view(B, hw, M, hw)

    # free: top-k over every bank patch of every memory image (unchanged)
    free_sim = sim.reshape(B, hw, M * hw).topk(
        int(topk), dim=2, largest=True).values.mean(dim=2)

    rr = torch.div(torch.arange(hw, device=device), W,
                   rounding_mode="floor")
    cc = torch.arange(hw, device=device) % W
    shifts = shift_offsets(shift_radius)
    zero_idx = shifts.index((0, 0)) if (0, 0) in shifts else 0

    # 1) Score every coherent shift per (B, M): mean similarity over valid
    #    query positions, then subtract the fixed Manhattan prior.
    shift_scores = []
    for sr, sc in shifts:
        s = _gather_shifted(sim, rr, cc, sr, sc, H, W)
        _, valid = _shifted_index(rr, cc, sr, sc, H, W)
        n_valid = int(valid.sum())
        if n_valid == 0:
            score = torch.full((B, M), float("-inf"), device=device,
                               dtype=s.dtype)
        else:
            finite = torch.where(torch.isfinite(s), s,
                                 torch.zeros_like(s))
            score = finite.sum(dim=1) / float(n_valid)
        shift_scores.append(score)
    shift_scores = torch.stack(shift_scores, dim=-1)     # [B, M, S]
    penalty = torch.tensor(
        [float(prior) * (abs(sr) + abs(sc)) for sr, sc in shifts],
        device=device, dtype=shift_scores.dtype)
    best = (shift_scores - penalty.view(1, 1, -1)).argmax(dim=-1)  # [B, M]
    invalid_pairs = ~torch.isfinite(shift_scores.max(dim=-1).values)
    if bool(invalid_pairs.any()):
        best = torch.where(invalid_pairs,
                           torch.full_like(best, int(zero_idx)), best)
    shift_table = torch.tensor(shifts, device=device, dtype=torch.long)
    sel_sr = shift_table[best][..., 0]                  # [B, M]
    sel_sc = shift_table[best][..., 1]

    # 2) Gather the +/-radius neighbourhood around the selected shift.
    offs = neighborhood_offsets(radius)
    gathered = []
    for dr, dc in offs:
        gathered.append(_gather_selected_neighborhood(
            sim, rr, cc, sel_sr, sel_sc, dr, dc, H, W))
    cand = torch.stack(gathered, dim=3).reshape(B, hw, M * len(offs))
    finite_per_row = torch.isfinite(cand).sum(dim=-1)   # [B, hw]
    fallback = finite_per_row < int(topk)

    # 3) Edge fallback: rows with too few aligned candidates use the original
    #    zero-shift neighbourhood.  At a corner this still has 4*M candidates.
    if bool(fallback.any()):
        zero_cand = _zero_shift_candidates(sim, rr, cc, radius, H, W)
        cand = torch.where(fallback.unsqueeze(-1), zero_cand, cand)

    local_sim = cand.topk(int(topk), dim=2, largest=True).values.mean(dim=2)

    free = (1.0 - free_sim).view(B, H, W)
    local = (1.0 - local_sim).view(B, H, W)

    best_cpu = best.detach().cpu()
    flat_best = best_cpu.reshape(-1)
    counts = torch.bincount(flat_best, minlength=len(shifts)).tolist()
    pairs = int(flat_best.numel())
    zero_pairs = int((flat_best == int(zero_idx)).sum().item())
    diag = {
        "rows": int(B * hw),
        "pairs": int(pairs),
        "fallback_rows": int(fallback.sum().item()),
        "shift_counts": {f"{sr},{sc}": int(counts[i])
                         for i, (sr, sc) in enumerate(shifts)
                         if counts[i]},
        "zero_shift_pairs": zero_pairs,
        "invalid_shift_score_pairs": int(invalid_pairs.sum().item()),
        "mean_abs_sr": float(sel_sr.abs().float().mean().item()),
        "mean_abs_sc": float(sel_sc.abs().float().mean().item()),
    }
    return free, local, diag


def reference_maps_chunked(features, bank, chunk=REF_CHUNK,
                           radius=LOCAL_RADIUS, topk=TOP_K,
                           shift_radius=SHIFT_RADIUS,
                           prior=SHIFT_PRIOR):
    """Bound similarity-matrix memory by chunking over the batch dimension."""
    if chunk is None or chunk <= 0:
        chunk = int(features.shape[0])
    frees, locals_, diags = [], [], []
    i = 0
    while i < features.shape[0]:
        f, l, d = reference_distances(
            features[i:i + chunk], bank, radius=radius, topk=topk,
            shift_radius=shift_radius, prior=prior)
        frees.append(f)
        locals_.append(l)
        diags.append(d)
        i += chunk
    return (torch.cat(frees, dim=0), torch.cat(locals_, dim=0),
            merge_shift_diag(diags))


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
               ("ref_free", "ref_spatial_aligned", "base_fuse672")}
    rel_chunks = []
    shift_diags = []
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
            free_raw, aligned_raw, adiag = reference_maps_chunked(
                feats, ref["bank"], REF_CHUNK)
            free = postprocess(free_raw.unsqueeze(1), gaussian)
            aligned = postprocess(aligned_raw.unsqueeze(1), gaussian)
            hi_raw = global_map_chunked(model, hi_x, HI_BATCH)
            hi_map = postprocess(hi_raw, gaussian)
        bf = 0.5 * base + 0.5 * hi_map
        buckets["ref_free"].append(free.detach().cpu())
        buckets["ref_spatial_aligned"].append(aligned.detach().cpu())
        buckets["base_fuse672"].append(bf.detach().cpu())
        rel = (aligned_raw - free_raw) / torch.clamp(
            free_raw, min=DIST_FLOOR)
        rel_chunks.append(rel.reshape(-1).detach().cpu())
        shift_diags.append(adiag)
        del en, de, feats, free_raw, aligned_raw, free, aligned
        del hi_raw, hi_map, base, bf, base_x, hi_x

    if not rel_chunks:
        raise RuntimeError("empty calibration set")
    stats = {}
    for k, chunks in buckets.items():
        stats[k] = branch_stats(torch.cat([c.reshape(-1) for c in chunks]))
    rel_extra = float(torch.cat(rel_chunks).median())
    gate = adaptive_gate(rel_extra)
    return stats, rel_extra, gate, merge_shift_diag(shift_diags)


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
    stats, rel_extra, gate, cal_shift_diag = build_calibration(
        calib_paths, base_transform, hi_transform, model, ref, gaussian,
        device)
    ref.update({
        "memory_paths": memory_paths,
        "calibration_paths": calib_paths,
        "train_good_available": len(paths),
        "calibration": stats,
        "rel_extra_median": float(rel_extra),
        "gate": float(gate),
        "calibration_shift_alignment": cal_shift_diag,
    })
    log(f"{item}: gate(rel_extra aligned)={rel_extra:.6f} "
        f"g_spatial={gate:.6f}; free med/q95/scale="
        f"{stats['ref_free']['median']:.6f}/"
        f"{stats['ref_free']['q95']:.6f}/"
        f"{stats['ref_free']['scale']:.6f}; "
        f"aligned spatial med/q95/scale="
        f"{stats['ref_spatial_aligned']['median']:.6f}/"
        f"{stats['ref_spatial_aligned']['q95']:.6f}/"
        f"{stats['ref_spatial_aligned']['scale']:.6f}; "
        f"shift zero_frac={cal_shift_diag['zero_shift_fraction']:.4f} "
        f"fallback_frac={cal_shift_diag['fallback_fraction']:.6f}")
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
        "shift_radius": SHIFT_RADIUS,
        "shift_prior_manhattan": SHIFT_PRIOR,
        "alignment": ALIGNMENT_DESCRIPTION,
        "calibration": ref["calibration"],
        "calibration_shift_alignment": ref["calibration_shift_alignment"],
        "gate": {
            "rel_extra_median": float(ref["rel_extra_median"]),
            "g_spatial": float(ref["gate"]),
            "source": "disjoint calibration normals; aligned spatial branch",
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


def v5_control_comparison(item, variants):
    """Compare this run with the downloaded, unchanged stage-5 result."""
    ctrl = V5_CONTROL.get(item)
    if ctrl is None:
        return None
    pairs = (
        ("base_fuse672", "base_fuse672"),
        ("ref_spatial_aligned", "ref_spatial"),
        ("aligned_plus_base", "ref_plus_base"),
    )
    out = {
        "source": "downloaded stage-5 per-class JSON; unchanged control",
        "v5_n_test": int(ctrl["n_test"]),
        "v5_metrics": {k: ctrl[k] for k in
                       ("base_fuse672", "ref_spatial", "ref_plus_base")},
        "deltas_pp": {},
    }
    for ours, theirs in pairs:
        out["deltas_pp"][f"{ours} - v5_{theirs}"] = {
            "metrics": {
                k: 100.0 * (variants[ours]["metrics"][k]
                            - ctrl[theirs]["metrics"][k])
                for k in METRIC_KEYS},
            "recall": {
                k: 100.0 * (variants[ours]["recall"].get(k, float("nan"))
                            - ctrl[theirs]["recall"].get(k, float("nan")))
                for k in RECALL_KEYS},
        }
    return out


# ---------------------------------------------------------------------------
# per-class test evaluation
# ---------------------------------------------------------------------------
def evaluate_class(item, dataset, model, ref, gaussian, device, stage1_item):
    loader = DataLoader(dataset, batch_size=EVAL_BATCH, shuffle=False,
                        num_workers=4,
                        pin_memory=torch.cuda.is_available())
    buckets = {v: [] for v in VARIANTS}
    gt_buf, labels, scores = [], [], []
    shift_diags = []
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
            free_raw, aligned_raw, adiag = reference_maps_chunked(
                feats, ref["bank"], REF_CHUNK)
            free = postprocess(free_raw.unsqueeze(1), gaussian)
            aligned = postprocess(aligned_raw.unsqueeze(1), gaussian)
            shift_diags.append(adiag)

            # Archived stage-4 392+672 baseline.
            hi_raw = global_map_chunked(model, hi, HI_BATCH)
            hi_map = postprocess(hi_raw, gaussian)
            base_fuse672 = 0.5 * base + 0.5 * hi_map

            cal_free = calibrate_map(free, cal["ref_free"])
            cal_aligned = calibrate_map(
                aligned, cal["ref_spatial_aligned"])
            cal_base672 = calibrate_map(base_fuse672, cal["base_fuse672"])

            ref_adaptive_aligned = (gate * cal_aligned
                                    + (1.0 - gate) * cal_free)
            aligned_plus_base = (REF_BASE_WEIGHT * cal_base672
                                 + (1.0 - REF_BASE_WEIGHT)
                                 * ref_adaptive_aligned)

            per_variant = {
                "base_fuse672": base_fuse672,
                "ref_spatial_aligned": aligned,
                "ref_adaptive_aligned": ref_adaptive_aligned,
                "aligned_plus_base": aligned_plus_base,
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

            del (base_raw, en, feats, free_raw, aligned_raw, free, aligned,
                 hi_raw, hi_map, base_fuse672, cal_free, cal_aligned,
                 cal_base672, ref_adaptive_aligned, aligned_plus_base,
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
            "stage1_baseline_delta": None,
        },
        "test_shift_alignment": merge_shift_diag(shift_diags),
        "v5_control_comparison": v5_control_comparison(item, variants),
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

    comparisons = {
        "aligned_plus_base - base_fuse672": (
            "aligned_plus_base", "base_fuse672"),
        "aligned_plus_base - ref_adaptive_aligned": (
            "aligned_plus_base", "ref_adaptive_aligned"),
        "ref_adaptive_aligned - base_fuse672": (
            "ref_adaptive_aligned", "base_fuse672"),
        "ref_spatial_aligned - base_fuse672": (
            "ref_spatial_aligned", "base_fuse672"),
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

    v5 = {}
    names = None
    for it in item_list:
        entry = per_class[it].get("v5_control_comparison")
        if entry is None:
            continue
        names = names or list(entry["deltas_pp"].keys())
        for name in entry["deltas_pp"]:
            v5.setdefault(name, {"metrics": {}, "recall": {}})
    if names:
        for name in names:
            for k in METRIC_KEYS:
                v5[name]["metrics"][k] = finite_mean([
                    per_class[it]["v5_control_comparison"]["deltas_pp"][
                        name]["metrics"][k]
                    for it in item_list
                    if per_class[it].get("v5_control_comparison")])
            for k in RECALL_KEYS:
                v5[name]["recall"][k] = finite_mean([
                    per_class[it]["v5_control_comparison"]["deltas_pp"][
                        name]["recall"].get(k)
                    for it in item_list
                    if per_class[it].get("v5_control_comparison")])
    v5_mean = {}
    for v5v in ("base_fuse672", "ref_spatial", "ref_plus_base"):
        v5_mean[v5v] = {
            "metrics": {k: finite_mean([
                V5_CONTROL[it][v5v]["metrics"][k]
                for it in item_list if it in V5_CONTROL]) for k in METRIC_KEYS},
            "recall": {k: finite_mean([
                V5_CONTROL[it][v5v]["recall"].get(k)
                for it in item_list if it in V5_CONTROL])
                for k in RECALL_KEYS},
        }
    v5_control = {
        "source": "downloaded stage-5 per-class JSON; unchanged diagnostic "
                  "control, never used for fitting",
        "classes": [it for it in item_list if it in V5_CONTROL],
        "v5_mean": v5_mean,
        "mean_deltas_pp": v5,
    } if names else None

    return {
        "mean": mean,
        "image_mean": image_mean,
        "deltas": deltas,
        "per_class_deltas": per_class_deltas,
        "archived_baseline_comparison": archived,
        "stage1_identity": None,
        "v5_control": v5_control,
    }


# ---------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------
def pct(x):
    return f"{100.0 * x:.2f}"


def write_markdown(report, path):
    cfg = report.get("config", {})
    res = report.get("results", {})
    ev = report.get("experiment", {})
    lines = []
    lines.append("# Dinomaly aligned-reference v8 (targeted diagnostic)")
    lines.append("")
    lines.append(f"Slug: `{SLUG}`  ")
    lines.append(f"Generated: {ev.get('generated')}  ")
    ck = report.get("checkpoints", {}).get("base", {})
    lines.append(f"Checkpoint: `{ck.get('path')}` "
                 f"(sha256 `{str(ck.get('sha256'))[:16]}...`)  ")
    lines.append(f"Test images: {cfg.get('test_images_evaluated')} across "
                 f"{len(cfg.get('eval_classes', []))} classes  ")
    lines.append(f"Primary variant (pre-set): `{PRIMARY_VARIANT}`  ")
    lines.append("")
    lines.append("## TARGETED DIAGNOSTIC - NOT A FULL BENCHMARK")
    lines.append("")
    lines.append("Only the four official classes selected after the full "
                 "15-class test set was initialised are evaluated: "
                 + ", ".join(f"`{c}`" for c in cfg.get("eval_classes", []))
                 + f" ({cfg.get('test_images_evaluated')} official test "
                 "images).  These are v5 win/loss/pose cases, chosen before "
                 "this run.  Do not read the class means as a 15-class "
                 "benchmark result.")
    lines.append("")
    lines.append("## Core independent variable")
    lines.append("")
    lines.append("The stage-5 same-coordinate spatial search is replaced by "
                 "a per-(test image, memory image) coherent shift alignment. "
                 f"Each pair selects one shift `(sr, sc)` in "
                 f"`[-{SHIFT_RADIUS}, {SHIFT_RADIUS}]^2` by maximising the "
                 "mean cosine similarity over in-image query positions, minus "
                 f"a {SHIFT_PRIOR} * (|sr| + |sc|) Manhattan prior. Candidates "
                 f"are the +/-{LOCAL_RADIUS} neighbourhood around the shifted "
                 "coordinate, preserving local tolerance. Out-of-image "
                 "positions are masked. A query row with fewer than "
                 f"{TOP_K} finite aligned candidates falls back to the "
                 "original zero-shift neighbourhood. The free appearance "
                 "search is unchanged.")
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
                 f"all {MEMORY_PER_CLASS} memory images x 784 patches "
                 "(unchanged stage 5).")
    lines.append("- `ref_spatial_aligned`: 1 - mean top-k cosine similarity "
                 "over the shifted +/-1 neighbourhood, as described above.")
    lines.append("- Per-branch normal calibration from the 16 calibration "
                 f"normals: `(x - median) / max(q95 - median, "
                 f"{SCALE_FLOOR})`; the aligned spatial branch is recalibrated "
                 "on normals.")
    lines.append("- Adaptive gate from calibration normals only: "
                 f"`rel_extra = median((d_aligned - d_free) / "
                 f"max(d_free, {DIST_FLOOR}))`; {GATE_FORMULA}. Higher "
                 "relative extra distance lowers spatial weight. The formula "
                 "and constants are fixed; no class weights are hard-coded "
                 "and no test labels are used.")
    lines.append("")
    lines.append(f"## Variants ({len(VARIANTS)}, fixed before evaluation)")
    lines.append("")
    lines.append("| variant | definition |")
    lines.append("|---|---|")
    for v in VARIANTS:
        mark = " **PRIMARY**" if v == PRIMARY_VARIANT else ""
        lines.append(f"| `{v}` | {VARIANT_DEFINITIONS[v]}{mark} |")
    lines.append("")
    lines.append("All variants use the same official 256 grid, the exact "
                 "original 448->392 mask transform, and the original baseline "
                 "global top-1% image score.")
    lines.append("")
    title = (f"{len(cfg.get('eval_classes', []))}-class means "
             "(diagnostic subset)")
    lines.append(f"## {title} (%)")
    lines.append("")
    lines.append("| variant | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | "
                 "P-F1 | P-AUPRO |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        m = res.get("mean", {}).get(v, {}).get("metrics", {})
        mark = " **PRIMARY**" if v == PRIMARY_VARIANT else ""
        lines.append(f"| `{v}`{mark} | " + " | ".join(
            pct(m.get(k, float("nan"))) for k in METRIC_KEYS) + " |")
    lines.append("")
    lines.append("Image metrics are identical by construction. Only P-* "
                 "columns and recall differ.")
    lines.append("")
    lines.append("## Recall at fixed pixel FPR (class mean, %)")
    lines.append("")
    lines.append("| variant | R@FPR0.001 | R@FPR0.01 | R@FPR0.05 |")
    lines.append("|---|---|---|---|")
    for v in VARIANTS:
        r = res.get("mean", {}).get(v, {}).get("recall", {})
        lines.append(f"| `{v}` | " + " | ".join(
            pct(r.get(k, float("nan"))) for k in RECALL_KEYS) + " |")
    lines.append("")
    lines.append("Thresholds use each variant's own per-class test negative "
                 "pixels, exactly as stages 3/4; they are metric thresholds, "
                 "not fitting.")
    lines.append("")
    lines.append("## Shift-alignment diagnostics")
    lines.append("")
    lines.append("| class | calib zero frac | calib fallback frac | "
                 "test zero frac | test fallback frac | test mean |sr| | "
                 "test mean |sc| | invalid shift pairs |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for item in cfg.get("eval_classes", []):
        pc = res.get("per_class", {}).get(item, {})
        cal_d = pc.get("normal_reference", {}).get(
            "calibration_shift_alignment", {})
        test_d = pc.get("test_shift_alignment", {})
        lines.append(
            f"| {item} | {cal_d.get('zero_shift_fraction', float('nan')):.4f} "
            f"| {cal_d.get('fallback_fraction', float('nan')):.6f} "
            f"| {test_d.get('zero_shift_fraction', float('nan')):.4f} "
            f"| {test_d.get('fallback_fraction', float('nan')):.6f} "
            f"| {test_d.get('mean_abs_sr', float('nan')):.4f} "
            f"| {test_d.get('mean_abs_sc', float('nan')):.4f} "
            f"| {test_d.get('invalid_shift_score_pairs', 0)} |")
    lines.append("")
    lines.append("A nonzero fallback fraction is the explicit edge case: a "
                 "shifted +/-1 neighbourhood left fewer than top-3 finite "
                 "candidates for that query row, so the zero-shift "
                 "neighbourhood was used instead.")
    lines.append("")
    lines.append("## Per-class calibration and adaptive gate")
    lines.append("")
    lines.append("| class | mem | calib | rel_extra (aligned) | g_spatial | "
                 "free med | free q95 | free scale | aligned med | "
                 "aligned q95 | aligned scale | base672 med | base672 q95 | "
                 "base672 scale |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for item in cfg.get("eval_classes", []):
        nr = res.get("per_class", {}).get(item, {}).get(
            "normal_reference", {})
        c = nr.get("calibration", {})
        f = c.get("ref_free", {})
        a = c.get("ref_spatial_aligned", {})
        b = c.get("base_fuse672", {})
        lines.append(
            f"| {item} | {nr.get('memory_count')} | "
            f"{nr.get('calibration_count')} | "
            f"{nr.get('gate', {}).get('rel_extra_median', float('nan')):.4f} "
            f"| {nr.get('gate', {}).get('g_spatial', float('nan')):.4f} "
            f"| {f.get('median', float('nan')):.5f} "
            f"| {f.get('q95', float('nan')):.5f} "
            f"| {f.get('scale', float('nan')):.5f} "
            f"| {a.get('median', float('nan')):.5f} "
            f"| {a.get('q95', float('nan')):.5f} "
            f"| {a.get('scale', float('nan')):.5f} "
            f"| {b.get('median', float('nan')):.5f} "
            f"| {b.get('q95', float('nan')):.5f} "
            f"| {b.get('scale', float('nan')):.5f} |")
    lines.append("")
    lines.append("## Mean pixel deltas (percentage points)")
    lines.append("")
    lines.append("| comparison | P-AUROC | P-AP | P-F1 | P-AUPRO |")
    lines.append("|---|---|---|---|---|")
    for name, delta in report.get("deltas", {}).items():
        d = delta.get("metrics", {})
        lines.append(f"| {name} | " + " | ".join(
            f"{100 * d.get(k, float('nan')):+.2f}" for k in
            ("P-AUROC", "P-AP", "P-F1", "P-AUPRO")) + " |")
    lines.append("")
    lines.append("## Per-class pixel AP (%)")
    lines.append("")
    lines.append("| class | " + " | ".join(f"`{v}`" for v in VARIANTS) + " |")
    lines.append("|---" * (len(VARIANTS) + 1) + "|")
    for item in cfg.get("eval_classes", []):
        vals = [res.get("per_class", {}).get(item, {}).get(
                    "variants", {}).get(v, {}).get("metrics", {}).get(
                    "P-AP", float("nan")) for v in VARIANTS]
        lines.append(f"| {item} | " + " | ".join(pct(x) for x in vals) + " |")
    lines.append("")
    v5 = res.get("v5_control")
    lines.append("## Downloaded v5 control (unchanged)")
    lines.append("")
    if v5:
        lines.append("v5 class means on the same four classes (%):")
        lines.append("")
        lines.append("| v5 variant | P-AUROC | P-AP | P-F1 | P-AUPRO |")
        lines.append("|---|---|---|---|---|")
        for v5v in ("base_fuse672", "ref_spatial", "ref_plus_base"):
            m = v5.get("v5_mean", {}).get(v5v, {}).get("metrics", {})
            lines.append(f"| `{v5v}` | " + " | ".join(
                pct(m.get(k, float("nan"))) for k in
                ("P-AUROC", "P-AP", "P-F1", "P-AUPRO")) + " |")
        lines.append("")
        lines.append("This-run minus v5 (percentage points):")
        lines.append("")
        lines.append("| comparison | P-AUROC | P-AP | P-F1 | P-AUPRO |")
        lines.append("|---|---|---|---|---|")
        for name, d in v5.get("mean_deltas_pp", {}).items():
            m = d.get("metrics", {})
            lines.append(f"| {name} | " + " | ".join(
                f"{m.get(k, float('nan')):+.2f}" for k in
                ("P-AUROC", "P-AP", "P-F1", "P-AUPRO")) + " |")
    else:
        lines.append("No v5 control entry available.")
    lines.append("")
    lines.append("## Archived baseline check")
    lines.append("")
    arch = res.get("archived_baseline_comparison", {})
    for name, e in arch.items():
        if name == "image_mean_pp":
            continue
        md = e.get("metrics_pp", {})
        lines.append(
            f"- `{name}`: max |delta| "
            f"{e.get('max_abs_metric_pp', float('nan')):.4f} pp; "
            f"P-AP {md.get('P-AP', float('nan')):+.4f} pp, "
            f"P-AUROC {md.get('P-AUROC', float('nan')):+.4f} pp, "
            f"P-AUPRO {md.get('P-AUPRO', float('nan')):+.4f} pp")
    im = arch.get("image_mean_pp", {})
    if im:
        lines.append("- Image means vs stage 4: " + ", ".join(
            f"{k} {v:+.4f} pp" for k, v in im.items()))
    lines.append("")
    lines.append("## Audit and runtime")
    lines.append("")
    ident = report.get("identity", {})
    lines.append(f"- Image score is the original baseline global top-1% and "
                 f"is reused by all variants (max |delta| = "
                 f"{ident.get('image_score_max_abs_diff_across_variants', float('nan')):.3e}).")
    lines.append("- No training; the unchanged frozen 10k checkpoint is "
                 "reused.")
    lines.append("- Only `train/good` images form the memory bank, and the "
                 "calibration set is disjoint from it. Test labels enter only "
                 "through standard metric definitions.")
    lines.append("- High-res 672 maps are computed on the full native image "
                 "and reduced to 256 for metrics; no raw high-res maps are "
                 "saved.")
    lines.append("- The stage-5 control is a downloaded JSON and was not "
                 "refitted or altered.")
    lines.append("")
    rt = report.get("runtime", {})
    lines.append(f"- Device: `{rt.get('device')}`; torch `{rt.get('torch')}`")
    lines.append(f"- Classes: {rt.get('classes_evaluated')}; test images: "
                 f"{rt.get('test_images_evaluated')}")
    lines.append(f"- Forward passes per test image: "
                 f"{rt.get('forwards_per_test_image')} "
                 "(1 baseline + 1 hi672)")
    lines.append(f"- Effective batches: base {rt.get('base_batch')}, "
                 f"hi672 {rt.get('hi_batch')}, "
                 f"reference chunk {rt.get('ref_chunk')}")
    lines.append(f"- Evaluation seconds (sum of per-class): "
                 f"{rt.get('seconds_total', float('nan')):.1f}")
    lines.append("")
    lines.append("Full per-class metrics, thresholds, shift diagnostics and "
                 "calibration values: `aligned_diagnostic_report.json`.")
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

    free1, local1, _d1 = reference_distances(q, bank, radius=1, topk=1)
    ok(float(free1.max()) < 1e-5,
       "free top-1 finds an exact far exemplar")
    ok(float(local1.mean()) > float(free1.mean()) + 0.3,
       "aligned spatial rejects a 4-row-shifted exemplar outside [-2,2]")

    free3, local3, _d3 = reference_distances(q, bank, radius=1, topk=3)
    sim = q.flatten(2).transpose(1, 2).reshape(hw, C) @ bank.t()
    brute = (1.0 - sim).topk(3, dim=1, largest=False).values.mean(dim=1)
    ok(torch.allclose(free3.flatten(), brute, atol=1e-5),
       "free top-3 matches brute force (unchanged search)")
    ok(float(local3.mean()) > float(free3.mean()) + 0.02,
       "aligned spatial top-3 penalises the far exemplar")

    # perfectly repeatable normal: free and spatial both collapse to zero,
    # and the Manhattan prior selects the zero shift on a perfect tie.
    v = F.normalize(torch.randn(C, generator=gen), p=2, dim=0)
    bank_const = v.expand(M * hw, C).contiguous()
    q_const = v.view(1, C, 1, 1).expand(1, C, H, W)
    f2, l2, d2 = reference_distances(q_const, bank_const, radius=1, topk=3)
    ok(float(f2.max()) < 1e-5 and float(l2.max()) < 1e-5,
       "repeatable normal gives free and spatial zero distance")
    ok(d2["zero_shift_pairs"] == d2["pairs"],
       "Manhattan prior selects zero shift on an exact tie")

    # direction test: bank = query rolled by (sr0, sc0).  The aligned search
    # must select exactly that shift, not its negation or a transposition.
    for sr0, sc0 in ((2, -1), (-2, 2), (1, 1)):
        q_dir = F.normalize(torch.randn(1, C, H, W, generator=gen),
                            p=2, dim=1)
        bank_dir = torch.roll(q_dir, shifts=(sr0, sc0), dims=(2, 3))
        bank_dir = bank_dir.permute(0, 2, 3, 1).reshape(hw, C).contiguous()
        _f, l_dir, d_dir = reference_distances(
            q_dir, bank_dir, radius=1, topk=1,
            shift_radius=SHIFT_RADIUS, prior=SHIFT_PRIOR)
        key = f"{sr0},{sc0}"
        ok(d_dir["shift_counts"].get(key) == d_dir["pairs"],
           f"direction: selected shift is {key}")
        rr = torch.arange(H).view(H, 1).expand(H, W)
        cc = torch.arange(W).view(1, W).expand(H, W)
        center_valid = ((rr + sr0 >= 0) & (rr + sr0 < H)
                        & (cc + sc0 >= 0) & (cc + sc0 < W))
        ok(float(l_dir[0][center_valid].max()) < 1e-5,
           f"direction: top-1 exact match where shift {key} is in-image")

    # edge case: a shift of (-2,-2) leaves no in-image candidates for the
    # top/left border rows.  Those rows must fall back to the zero-shift
    # neighbourhood, stay finite, and still produce a valid top-3 mean.
    q_edge = F.normalize(torch.randn(1, C, H, W, generator=gen), p=2, dim=1)
    bank_edge = torch.roll(q_edge, shifts=(-2, -2), dims=(2, 3))
    bank_edge = bank_edge.permute(0, 2, 3, 1).reshape(hw, C).contiguous()
    _f, l_edge, d_edge = reference_distances(
        q_edge, bank_edge, radius=1, topk=1,
        shift_radius=SHIFT_RADIUS, prior=SHIFT_PRIOR)
    ok(bool(torch.isfinite(l_edge).all()),
       "edge fallback keeps every distance finite")
    ok(d_edge["fallback_rows"] > 0,
       "edge rows with no shifted candidates use the fallback")
    ok(float(l_edge[0, 2, 2]) < 1e-5,
       "interior position still matches the selected shift exactly")
    ok(float(l_edge[0, 0, 0]) > 0.0,
       "corner falls back to zero-shift and is not an exact match")
    _f3, l_edge3, _d3e = reference_distances(
        q_edge, bank_edge, radius=1, topk=3,
        shift_radius=SHIFT_RADIUS, prior=SHIFT_PRIOR)
    ok(bool(torch.isfinite(l_edge3).all()),
       "edge fallback keeps top-3 means finite at every position")

    # renderer smoke test: no KeyError even on a minimal report
    import tempfile
    smoke = {
        "config": {"test_images_evaluated": 1, "eval_classes": ["x"]},
        "results": {"mean": {}, "per_class": {},
                    "archived_baseline_comparison": {}, "v5_control": None},
        "experiment": {"generated": "self-test"},
        "checkpoints": {"base": {"path": "p", "sha256": "abc"}},
        "deltas": {},
        "identity": {},
        "runtime": {},
    }
    smoke_path = os.path.join(tempfile.gettempdir(),
                              "dinomaly_v8_renderer_smoke.md")
    write_markdown(smoke, smoke_path)
    ok(os.path.isfile(smoke_path) and os.path.getsize(smoke_path) > 0,
       "markdown renderer completes without KeyError")

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
    LOG_PATH = os.path.join(OUT, "stage8.log")
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

    # The full official test set is already constructed and counted above.
    # Only now select the four pre-registered diagnostic classes.
    missing = [c for c in DIAGNOSTIC_CLASSES if c not in item_list]
    if missing:
        raise RuntimeError(f"diagnostic classes missing: {missing}")
    eval_items = list(DIAGNOSTIC_CLASSES)
    if args.limit_classes and args.limit_classes > 0:
        eval_items = eval_items[:args.limit_classes]
    test_images_evaluated = int(sum(counts[it] for it in eval_items))
    partial = len(eval_items) != len(item_list)
    if not args.data_root:
        if len(eval_items) != len(DIAGNOSTIC_CLASSES):
            raise RuntimeError("official diagnostic must evaluate exactly "
                               f"{DIAGNOSTIC_CLASSES}")
        if test_images_evaluated != EXPECTED_DIAGNOSTIC_IMAGES:
            raise RuntimeError(
                f"expected {EXPECTED_DIAGNOSTIC_IMAGES} diagnostic test "
                f"images, got {test_images_evaluated}: "
                f"{ {it: counts[it] for it in eval_items} }")
    log(f"TARGETED DIAGNOSTIC: classes={eval_items} "
        f"images={test_images_evaluated}; NOT a full benchmark "
        f"(full test set was {total_test} images / {len(item_list)} classes)")

    variant_entries = []
    for v in VARIANTS:
        variant_entries.append({
            "name": v,
            "definition": VARIANT_DEFINITIONS[v],
            "primary": v == PRIMARY_VARIANT,
        })
    config = {
        "slug": SLUG,
        "stage": 8,
        "diagnostic": True,
        "not_full_benchmark": True,
        "diagnostic_note": DIAGNOSTIC_NOTE,
        "diagnostic_class_rationale": {
            "transistor": "v5 aligned-spatial-style win (ref_spatial P-AP "
                          "74.37 vs base 60.31)",
            "zipper": "v5 fused-reference loss (ref_plus_base P-AP 75.50 vs "
                      "base_fuse672 86.72)",
            "pill": "v5 reference loss (ref_adaptive 54.06 vs base 74.50)",
            "screw": "v5 spatial collapse (ref_spatial 12.84 vs base 61.27)",
        },
        "diagnostic_selected_after_full_test_init": True,
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
            "shift_radius": SHIFT_RADIUS,
            "shift_prior_manhattan": SHIFT_PRIOR,
            "alignment": ALIGNMENT_DESCRIPTION,
            "matching": "batched GEMM; free all patches (unchanged); "
                        "per-(test image, memory image) coherent shift in "
                        "[-2,2]^2 selected by mean similarity minus 0.005 "
                        "Manhattan prior; +/-1 local tolerance around the "
                        "shifted coordinate; out-of-image masked; zero-shift "
                        "fallback for rows with fewer than top-k finite "
                        "candidates",
        },
        "calibration": {
            "source": "16 disjoint train/good calibration normals per class",
            "per_branch": ["ref_free", "ref_spatial_aligned",
                           "base_fuse672"],
            "loc": "median",
            "scale": "max(q95 - median, 0.01)",
            "formula": "(x - median) / scale",
            "positive_scale_floor": SCALE_FLOOR,
            "used_for": "fusion variants only; standalone reference maps "
                        "stay raw",
        },
        "gate": {
            "source": "calibration normals only; aligned spatial branch",
            "rel_extra": f"median((d_aligned - d_free) / max(d_free, "
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
            "adaptive_aligned": {"ref_free": "1 - g",
                                 "ref_spatial_aligned": "g"},
            "aligned_plus_base": {
                "cal_base_fuse672": REF_BASE_WEIGHT,
                "ref_adaptive_aligned": 1.0 - REF_BASE_WEIGHT},
            "stage5_formula_preserved": True,
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
        "stage5_control_embedded_unchanged": True,
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
            "stage": 8,
            "diagnostic": True,
            "not_full_benchmark": True,
            "diagnostic_note": DIAGNOSTIC_NOTE,
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
            "gate_source": "calibration normals only; fixed bounded stage-5 "
                           "formula applied to the aligned spatial branch",
            "stage1_baseline_comparison": None,
        },
        "results": {
            "per_class": per_class,
            "mean": agg["mean"],
            "image_mean": agg["image_mean"],
            "archived_baseline_comparison": agg[
                "archived_baseline_comparison"],
            "v5_control": agg["v5_control"],
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

    report_json = os.path.join(OUT, "aligned_diagnostic_report.json")
    report_md = os.path.join(OUT, "aligned_diagnostic_report.md")
    write_json_atomic(report_json, report)
    markdown_ok = True
    try:
        write_markdown(report, report_md)
    except Exception:
        markdown_ok = False
        log("WARNING: markdown renderer failed; JSON report is complete\n"
            + traceback.format_exc())
    set_status("complete", classes_done=done,
               classes_total=len(eval_items),
               report=report_json,
               markdown=(report_md if markdown_ok else None),
               markdown_error=(None if markdown_ok
                               else "renderer failed; JSON saved"))
    update_progress(current=None)
    log("DONE")
    log(f"report={report_json}")
    log(f"markdown={report_md if markdown_ok else 'SKIPPED'}")
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
