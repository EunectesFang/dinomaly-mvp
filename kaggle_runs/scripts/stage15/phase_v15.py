"""Stage 15: half-token phase of the frozen Dinomaly 672 residual.

The checkpoint, the two-group cosine mean, the gaussian, and the 256 grid
stay as they are. Nothing is trained and nothing is fused into guide_fuse672.

The same frozen model is copied onto both T4 GPUs. Classes are split by
the known test-image counts, 861 on cuda:0 and 864 on cuda:1. One 768
resize produces seven 672 windows. Shifts are (dy, dx) from the
center-crop origin. Half-token phases (0,0),(7,0),(0,7),(7,7) are interleaved
onto the 7px lattice by writing each token only into its top-left 7x7 cell.
Full-token phases (0,0),(14,0),(0,14),(14,14) are averaged on the shared 14px
lattice. The single window is (0,0) alone. All three maps are scored only on
the intersection of the seven windows, then bilinearly resized to 256 and
smoothed with gaussian(5, sigma=4).

Pass, fixed before any score exists: 15-class macro pixel AP of the half-token
map beats both the full-token mean and the single window by at least 0.02.
Against the full-token mean, screw and capsule do not drop, leather and grid
drop by at most 0.02, and no class drops by more than 0.05. The screw
R@FPR0.001 gap versus the single window is reported and does not flip the flag.
"""

import hashlib
import json
import os
import sys
import threading
import time
from functools import partial

import numpy as np

SLUG = "eunectesqaq/dinomaly-phase-v15"
STAGE = 15
HI_RESIZE = 768
HI_CROP = 672
PATCH = 14
CENTER = (HI_RESIZE - HI_CROP) // 2
SHIFTS = (
    (0, 0), (7, 0), (0, 7), (7, 7),
    (14, 0), (0, 14), (14, 14),
)
HALF_SHIFTS = ((0, 0), (7, 0), (0, 7), (7, 7))
FULL_SHIFTS = ((0, 0), (14, 0), (0, 14), (14, 14))
SHIFT_INDEX = {shift: i for i, shift in enumerate(SHIFTS)}
GAUSSIAN_KERNEL = 5
GAUSSIAN_SIGMA = 4
FORWARD_CHUNK = 2
CLASSES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
)
EXPECTED_TEST_IMAGES = {
    "bottle": 83, "cable": 150, "capsule": 132, "carpet": 117, "grid": 78,
    "hazelnut": 110, "leather": 124, "metal_nut": 115, "pill": 167,
    "screw": 160, "tile": 117, "toothbrush": 42, "transistor": 100,
    "wood": 79, "zipper": 151,
}
# Fixed before the run. Counts are the public test-set sizes, not scores.
GPU0_CLASSES = (
    "pill", "zipper", "capsule", "carpet", "metal_nut", "transistor", "wood",
)
GPU1_CLASSES = (
    "screw", "cable", "leather", "tile", "hazelnut", "bottle", "grid",
    "toothbrush",
)
EXPECTED_SHA = (
    "f5689b76aafa5c9d2b36a0df8e87ef7bdc82075a042c76bcf7933e0edd637218"
)
FPRS = (0.001, 0.01, 0.05)
VARIANTS = ("half", "full", "single")

LOG_PATH = None
CODE = None
DATA = None
BASE_CKPT = None
torch = nn = F = DataLoader = None
roc_auc_score = average_precision_score = auc = None
get_data_transforms = MVTecDataset = ViTill = vit_encoder = None
VitBlock = bMlp = LinearAttention2 = None
get_gaussian_kernel = f1_score_max = measure = None


LOG_LOCK = threading.Lock()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with LOG_LOCK:
        print(line, flush=True)
        if LOG_PATH:
            with open(LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def window_box(dy, dx):
    y0 = CENTER + int(dy)
    x0 = CENTER + int(dx)
    return y0, y0 + HI_CROP, x0, x0 + HI_CROP


def intersection_box():
    boxes = [window_box(*shift) for shift in SHIFTS]
    return (
        max(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        min(box[3] for box in boxes),
    )


def aligned_bounds(delta):
    """Token slice of one axis onto the intersection's 14px lattice."""
    if delta == 0:
        return 1, 48
    if delta == 14:
        return 0, 47
    raise ValueError(f"shift {delta} is not on the 14px lattice")


def aligned_crop(tokens, dy, dx):
    y0, y1 = aligned_bounds(dy)
    x0, x1 = aligned_bounds(dx)
    return tokens[SHIFT_INDEX[(dy, dx)], y0:y1, x0:x1]


def half_index():
    u = np.arange(2, 96)
    v = np.arange(2, 96)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    dy = 7 * (uu % 2)
    dx = 7 * (vv % 2)
    ti = ((uu - (uu % 2)) // 2).astype(np.int64)
    tj = ((vv - (vv % 2)) // 2).astype(np.int64)
    src = np.empty(uu.shape, dtype=np.int64)
    src[(dy == 0) & (dx == 0)] = SHIFT_INDEX[(0, 0)]
    src[(dy == 7) & (dx == 0)] = SHIFT_INDEX[(7, 0)]
    src[(dy == 0) & (dx == 7)] = SHIFT_INDEX[(0, 7)]
    src[(dy == 7) & (dx == 7)] = SHIFT_INDEX[(7, 7)]
    return src, ti, tj


HALF_SRC, HALF_TI, HALF_TJ = half_index()


def assemble(tokens):
    """tokens: [7, 48, 48] in SHIFTS order. Returns half, full, single."""
    if tokens.shape != (len(SHIFTS), 48, 48):
        raise RuntimeError(f"token stack {tokens.shape} is not 7x48x48")
    single = aligned_crop(tokens, 0, 0)
    full = np.mean(
        [aligned_crop(tokens, dy, dx) for dy, dx in FULL_SHIFTS], axis=0)
    half = tokens[HALF_SRC, HALF_TI, HALF_TJ]
    if single.shape != (47, 47) or full.shape != (47, 47):
        raise RuntimeError(f"aligned map {single.shape} {full.shape}")
    if half.shape != (94, 94):
        raise RuntimeError(f"half map {half.shape}")
    return half, full, single


def decide(per_class_ap, screw_recall_delta):
    names = list(CLASSES)
    d_full = {
        name: per_class_ap[name]["half"] - per_class_ap[name]["full"]
        for name in names
    }
    d_single = {
        name: per_class_ap[name]["half"] - per_class_ap[name]["single"]
        for name in names
    }
    macro_full = float(np.mean([d_full[name] for name in names]))
    macro_single = float(np.mean([d_single[name] for name in names]))
    clauses = {
        "macro_vs_full": bool(macro_full >= 0.02),
        "macro_vs_single": bool(macro_single >= 0.02),
        "screw_vs_full": bool(d_full["screw"] >= 0.0),
        "capsule_vs_full": bool(d_full["capsule"] >= 0.0),
        "leather_vs_full": bool(d_full["leather"] >= -0.02),
        "grid_vs_full": bool(d_full["grid"] >= -0.02),
        "any_class_floor": bool(all(value >= -0.05 for value in d_full.values())),
    }
    recall_delta = None if screw_recall_delta is None else float(screw_recall_delta)
    return {
        "passed": bool(all(clauses.values())),
        "clauses": clauses,
        "macro_half_minus_full": macro_full,
        "macro_half_minus_single": macro_single,
        "per_class_half_minus_full": d_full,
        "per_class_half_minus_single": d_single,
        "screw_half_minus_single_R@FPR0.001": recall_delta,
        "dilution_supported_on_screw": bool(
            recall_delta is not None and recall_delta > 0.0),
        "fusion_counts": False,
        "rule": {
            "macro_gain_min": 0.02,
            "compared_with": ["full", "single"],
            "screw_and_capsule": "not below full",
            "leather_and_grid_floor": -0.02,
            "any_class_floor": -0.05,
            "score_region": "intersection of the seven 672 windows",
        },
    }


def run_self_test():
    checks = 0

    def ok(flag, message):
        nonlocal checks
        if not flag:
            raise AssertionError(message)
        checks += 1

    y0, y1, x0, x1 = intersection_box()
    ok((y0, y1, x0, x1) == (62, 720, 62, 720), "intersection box")
    span = y1 - y0
    ok(span == 658 and span % 14 == 0 and span % 7 == 0, "intersection span")
    ok(sum(EXPECTED_TEST_IMAGES[name] for name in CLASSES) == 1725,
       "15-class image count")
    ok(set(GPU0_CLASSES).isdisjoint(GPU1_CLASSES), "gpu groups disjoint")
    ok(set(GPU0_CLASSES) | set(GPU1_CLASSES) == set(CLASSES), "gpu groups cover")
    n0 = sum(EXPECTED_TEST_IMAGES[name] for name in GPU0_CLASSES)
    n1 = sum(EXPECTED_TEST_IMAGES[name] for name in GPU1_CLASSES)
    ok(n0 == 861 and n1 == 864, f"gpu image split {n0}/{n1}")
    ok(HALF_SRC.shape == (94, 94), "half index shape")
    ok(set(np.unique(HALF_SRC).tolist()) == {0, 1, 2, 3}, "half phases")
    ok(int((HALF_SRC == 0).sum()) == 94 * 94 // 4, "balanced phases")
    ok(int(HALF_TI.min()) >= 0 and int(HALF_TI.max()) <= 47, "token rows")
    ok(int(HALF_TJ.min()) >= 0 and int(HALF_TJ.max()) <= 47, "token cols")

    tokens = np.zeros((len(SHIFTS), 48, 48), dtype=np.int64)
    for index, (dy, dx) in enumerate(SHIFTS):
        for row in range(48):
            y_start = CENTER + dy + PATCH * row
            x_start = CENTER + dx + PATCH * np.arange(48)
            tokens[index, row, :] = y_start * 100000 + x_start
    half, full, single = assemble(tokens)
    rows = 62 + 7 * np.arange(94)
    cols = 62 + 7 * np.arange(94)
    expected_half = rows[:, None] * 100000 + cols[None, :]
    ok(np.array_equal(half, expected_half), "half cells use token top-left")
    coarse_r = 62 + 14 * np.arange(47)
    coarse_c = 62 + 14 * np.arange(47)
    expected_coarse = coarse_r[:, None] * 100000 + coarse_c[None, :]
    ok(np.array_equal(single, expected_coarse), "single window is aligned")
    ok(np.allclose(full, expected_coarse), "full-token mean is aligned")

    good = {}
    for name in CLASSES:
        good[name] = {"half": 0.50, "full": 0.47, "single": 0.47}
    good["screw"]["half"] = 0.55
    good["capsule"]["half"] = 0.54
    passed = decide(good, 0.01)
    ok(passed["passed"], "synthetic pass")
    short = {
        name: {"half": 0.519, "full": 0.500, "single": 0.490}
        for name in CLASSES
    }
    ok(not decide(short, 0.01)["passed"], "macro under 2pp fails")
    screw_drop = {name: dict(values) for name, values in good.items()}
    screw_drop["screw"]["half"] = screw_drop["screw"]["full"] - 0.001
    ok(not decide(screw_drop, 0.01)["passed"], "screw drop fails")
    leather_drop = {name: dict(values) for name, values in good.items()}
    leather_drop["leather"]["half"] = leather_drop["leather"]["full"] - 0.021
    ok(not decide(leather_drop, 0.01)["passed"], "leather floor fails")
    one_class = {name: dict(values) for name, values in good.items()}
    one_class["tile"]["half"] = one_class["tile"]["full"] - 0.051
    ok(not decide(one_class, 0.01)["passed"], "5pp class drop fails")
    weak_single = {name: dict(values) for name, values in good.items()}
    for name in CLASSES:
        weak_single[name]["single"] = weak_single[name]["half"] - 0.01
    ok(not decide(weak_single, 0.01)["passed"], "single-window gap fails")
    reading = decide(good, -0.01)
    ok(reading["passed"] and not reading["dilution_supported_on_screw"],
       "recall gap does not flip the flag")
    log(f"self-test passed checks={checks}")
    return checks


def locate(marker, preferred):
    for candidate in preferred:
        if os.path.isdir(candidate) and os.path.exists(
                os.path.join(candidate, marker)):
            return candidate
    if os.path.isdir("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if marker in dirs or marker in files:
                return root
    raise RuntimeError(f"cannot locate {marker}")


def find_input(name, preferred):
    for path in preferred:
        if os.path.isfile(path):
            return path
    if os.path.isdir("/kaggle/input"):
        for root, _dirs, files in os.walk("/kaggle/input"):
            if name in files:
                return os.path.join(root, name)
    return None


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_writable(candidates):
    for candidate in candidates:
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = os.path.join(candidate, ".write_probe")
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("ok")
            os.remove(probe)
            return candidate
        except OSError:
            continue
    return None


def prepare_environment():
    global CODE, DATA
    global torch, nn, F, DataLoader
    global roc_auc_score, average_precision_score, auc
    global get_data_transforms, MVTecDataset, ViTill, vit_encoder
    global VitBlock, bMlp, LinearAttention2
    global get_gaussian_kernel, f1_score_max
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch.utils.data import DataLoader

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
    from dataset import MVTecDataset, get_data_transforms
    from models import vit_encoder
    from models.uad import ViTill
    from models.vision_transformer import Block as VitBlock
    from models.vision_transformer import LinearAttention2, bMlp
    from utils import f1_score_max, get_gaussian_kernel
    log(f"CODE={CODE}")
    log(f"DATA={DATA}")


def find_checkpoint():
    global BASE_CKPT
    BASE_CKPT = find_input("model_10000.pt", [
        "E:/dinomaly-mvp/kaggle_runs/dinomaly-base-10k/model_10000.pt",
        "/kaggle/input/dinomaly-base-10k/model_10000.pt",
        "/kaggle/input/notebooks/eunectesqaq/dinomaly-base-10k/model_10000.pt",
        "/kaggle/input/datasets/eunectesqaq/dinomaly-base-10k/model_10000.pt",
    ])
    if not BASE_CKPT:
        raise RuntimeError("model_10000.pt not found")
    log(f"base checkpoint={BASE_CKPT}")


def build_model(device):
    encoder = vit_encoder.load("dinov2reg_vit_base_14")
    bottleneck = nn.ModuleList([bMlp(768, 768 * 4, 768, drop=0.2)])
    decoder = nn.ModuleList([
        VitBlock(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-8),
                 attn=LinearAttention2)
        for _ in range(8)
    ])
    model = ViTill(
        encoder=encoder, bottleneck=bottleneck, decoder=decoder,
        target_layers=[2, 3, 4, 5, 6, 7, 8, 9], mask_neighbor_size=0,
        fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]])
    return model.to(device)


def load_ckpt(model, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.bottleneck.load_state_dict(state["bottleneck"])
    model.decoder.load_state_dict(state["decoder"])
    model.eval()
    return state


def token_residual(model, images, chunk):
    outs = []
    start = 0
    while start < images.shape[0]:
        part = images[start:start + chunk]
        try:
            with torch.inference_mode():
                encoded, decoded = model(part)
                groups = [
                    1.0 - F.cosine_similarity(enc, dec, dim=1)
                    for enc, dec in zip(encoded, decoded)
                ]
                outs.append(torch.stack(groups, dim=0).mean(dim=0))
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and part.shape[0] > 1:
                torch.cuda.empty_cache()
                smaller = max(1, part.shape[0] // 2)
                log(f"WARNING: OOM at batch {part.shape[0]}; retry {smaller}")
                outs.append(token_residual(model, part, smaller))
            else:
                raise
        start += part.shape[0]
    stacked = torch.cat(outs, dim=0)
    if stacked.shape[0] != images.shape[0] or stacked.shape[-2:] != (48, 48):
        raise RuntimeError(f"residual shape {tuple(stacked.shape)}")
    return stacked


def forward_windows(model, image, chunk):
    if image.dim() == 4:
        image = image[0]
    crops = []
    for dy, dx in SHIFTS:
        top, bottom, left, right = window_box(dy, dx)
        crops.append(image[:, top:bottom, left:right])
    batch = torch.stack(crops, dim=0)
    return token_residual(model, batch, chunk).detach().cpu().numpy()


def smooth_to_256(amap, gaussian):
    tensor = torch.from_numpy(np.asarray(amap, dtype=np.float32))
    tensor = tensor.view(1, 1, *amap.shape).to(gaussian.weight.device)
    out = gaussian(F.interpolate(
        tensor, size=256, mode="bilinear", align_corners=False))
    return out[0, 0].detach().cpu().numpy()


class IntersectionTestSet:
    def __init__(self, root):
        img_tf, gt_tf = get_data_transforms(HI_RESIZE, HI_RESIZE)
        self.base = MVTecDataset(
            root=root, transform=img_tf, gt_transform=gt_tf, phase="test")
        self.box = intersection_box()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, gt, label, path = self.base[idx]
        y0, y1, x0, x1 = self.box
        if gt.shape[0] > 1:
            gt = gt.max(dim=0, keepdim=True).values
        gt = gt[:, y0:y1, x0:x1].contiguous()
        return img, gt, label, path


def fixed_fpr_recall(scores, gt):
    neg = scores[gt == 0]
    pos = scores[gt == 1]
    recall, thresholds = {}, {}
    if neg.size == 0 or pos.size == 0:
        return recall, thresholds
    for rate in FPRS:
        thr = float(np.quantile(neg, 1.0 - rate))
        recall[f"R@FPR{rate:g}"] = float((pos >= thr).mean())
        thresholds[f"T@FPR{rate:g}"] = thr
    return recall, thresholds


def per_image_ap(gt, pred, labels):
    values = []
    for index in range(gt.shape[0]):
        if int(labels[index]) == 0 or int(gt[index].sum()) == 0:
            continue
        values.append(float(average_precision_score(
            gt[index].ravel(), pred[index].ravel())))
    if not values:
        return float("nan")
    return float(np.mean(values))


def summarize(gt, pred, labels):
    flat_gt = gt.ravel()
    flat_pr = pred.ravel()
    recall, thresholds = fixed_fpr_recall(flat_pr, flat_gt)
    metrics = {
        "P-AUROC": float(roc_auc_score(flat_gt, flat_pr)),
        "P-AP": float(average_precision_score(flat_gt, flat_pr)),
        "P-F1": float(f1_score_max(flat_gt, flat_pr)),
        "P-AP-image": per_image_ap(gt, pred, labels),
    }
    return {"metrics": metrics, "recall": recall, "thresholds": thresholds}


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(key): json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(value) for value in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def write_json(path, payload):
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
    os.replace(temp, path)


def evaluate_class(name, dataset, model, gaussian, device, chunk):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    stored = {variant: [] for variant in VARIANTS}
    gt_buf, labels = [], []
    started = time.time()
    total = len(dataset)
    for index, batch in enumerate(loader):
        img, gt, label, _path = batch
        img = img.to(device, non_blocking=True)
        tokens = forward_windows(model, img, chunk)
        maps = dict(zip(VARIANTS, assemble(tokens)))
        for variant in VARIANTS:
            stored[variant].append(smooth_to_256(maps[variant], gaussian))
        gt256 = F.interpolate(gt.float(), size=256, mode="nearest").bool()
        if gt256.shape[1] > 1:
            gt256 = gt256.max(dim=1, keepdim=True)[0]
        gt_buf.append(gt256.cpu())
        labels.append(int(label.view(-1)[0]))
        del img, tokens, maps
        if index == 0 or (index + 1) % 20 == 0 or index + 1 == total:
            log(f"  {name} {index + 1}/{total} "
                f"elapsed={time.time() - started:.1f}s")
    gt_np = torch.cat(gt_buf, dim=0)[:, 0].numpy()
    gt_np = (gt_np > 0.5).astype(np.uint8)
    labels_np = np.asarray(labels, dtype=np.int64)
    variants = {}
    for variant in VARIANTS:
        pred = np.stack(stored[variant], axis=0).astype(np.float32)
        stored[variant] = None
        variants[variant] = summarize(gt_np, pred, labels_np)
    return {
        "class": name,
        "n_test": int(len(labels_np)),
        "n_anomaly": int((labels_np == 1).sum()),
        "n_gt_positive_pixels": int(gt_np.sum()),
        "seconds": time.time() - started,
        "variants": variants,
    }


def mean_metrics(per_class):
    out = {}
    for variant in VARIANTS:
        metrics, recall = {}, {}
        metric_keys = per_class[CLASSES[0]]["variants"][variant]["metrics"]
        recall_keys = per_class[CLASSES[0]]["variants"][variant]["recall"]
        for key in metric_keys:
            metrics[key] = float(np.mean([
                per_class[name]["variants"][variant]["metrics"][key]
                for name in CLASSES
            ]))
        for key in recall_keys:
            recall[key] = float(np.mean([
                per_class[name]["variants"][variant]["recall"][key]
                for name in CLASSES
            ]))
        out[variant] = {"metrics": metrics, "recall": recall}
    return out


def run_group(names, model, gaussian, device, out_dir, bucket, errors, stop):
    try:
        torch.cuda.set_device(device)
        class_dir = os.path.join(out_dir, "per_class")
        n_images = sum(EXPECTED_TEST_IMAGES[name] for name in names)
        log(f"{device} images={n_images} classes={','.join(names)}")
        for name in names:
            if stop.is_set():
                raise RuntimeError(f"{device} stopped because the other GPU failed")
            root = os.path.join(DATA, name)
            dataset = IntersectionTestSet(root)
            if len(dataset) != EXPECTED_TEST_IMAGES[name]:
                raise RuntimeError(
                    f"{name} has {len(dataset)} images, "
                    f"expected {EXPECTED_TEST_IMAGES[name]}")
            result = evaluate_class(
                name, dataset, model, gaussian, device, FORWARD_CHUNK)
            with LOG_LOCK:
                bucket[name] = result
            write_json(os.path.join(class_dir, f"{name}.json"), result)
            half_ap = result["variants"]["half"]["metrics"]["P-AP"]
            full_ap = result["variants"]["full"]["metrics"]["P-AP"]
            single_ap = result["variants"]["single"]["metrics"]["P-AP"]
            log(f"{name} n={result['n_test']} half={half_ap:.4f} "
                f"full={full_ap:.4f} single={single_ap:.4f} "
                f"d_full={half_ap - full_ap:+.4f} "
                f"d_single={half_ap - single_ap:+.4f} device={device}")
            del dataset
    except Exception as exc:
        stop.set()
        errors.append(exc)
        log(f"ERROR {device} {type(exc).__name__}: {exc}")


def main():
    global LOG_PATH
    if "--self-test" in sys.argv:
        run_self_test()
        return
    run_self_test()
    out_dir = first_writable([
        "/kaggle/working",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
    ])
    if out_dir is None:
        raise RuntimeError("no writable output directory")
    LOG_PATH = os.path.join(out_dir, "stage15.log")
    log(f"=== {SLUG} start ===")
    prepare_environment()
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"need 2 GPUs, found {torch.cuda.device_count()}")
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    find_checkpoint()
    digest = sha256_file(BASE_CKPT)
    log(f"base sha256={digest}")
    if digest != EXPECTED_SHA:
        raise RuntimeError("checkpoint sha256 does not match model_10000.pt")
    model0 = build_model(device0)
    state = load_ckpt(model0, BASE_CKPT, device0)
    model1 = build_model(device1)
    model1.load_state_dict(model0.state_dict())
    model1.eval()
    gaussian0 = get_gaussian_kernel(
        kernel_size=GAUSSIAN_KERNEL, sigma=GAUSSIAN_SIGMA).to(device0)
    gaussian1 = get_gaussian_kernel(
        kernel_size=GAUSSIAN_KERNEL, sigma=GAUSSIAN_SIGMA).to(device1)
    log(f"checkpoint iter={state.get('iter', 'unknown')} gpus=2 "
        f"windows={len(SHIFTS)} chunk={FORWARD_CHUNK}")
    os.makedirs(os.path.join(out_dir, "per_class"), exist_ok=True)
    bucket, errors = {}, []
    stop = threading.Event()
    workers = (
        threading.Thread(
            target=run_group,
            args=(GPU0_CLASSES, model0, gaussian0, device0,
                  out_dir, bucket, errors, stop),
            name="gpu0"),
        threading.Thread(
            target=run_group,
            args=(GPU1_CLASSES, model1, gaussian1, device1,
                  out_dir, bucket, errors, stop),
            name="gpu1"),
    )
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if errors:
        raise errors[0]
    if set(bucket) != set(CLASSES):
        missing = sorted(set(CLASSES) - set(bucket))
        raise RuntimeError(f"missing classes: {missing}")
    per_class = {name: bucket[name] for name in CLASSES}
    ap = {
        name: {
            variant: per_class[name]["variants"][variant]["metrics"]["P-AP"]
            for variant in VARIANTS
        }
        for name in CLASSES
    }
    screw_recall = (
        per_class["screw"]["variants"]["half"]["recall"]["R@FPR0.001"]
        - per_class["screw"]["variants"]["single"]["recall"]["R@FPR0.001"]
    )
    decision = decide(ap, screw_recall)
    report = {
        "experiment": {
            "slug": SLUG,
            "stage": STAGE,
            "idea": "half-token phase of the frozen 672 residual",
            "base_sha256": digest,
        },
        "config": {
            "classes": list(CLASSES),
            "resize": HI_RESIZE,
            "crop": HI_CROP,
            "shifts": [list(shift) for shift in SHIFTS],
            "intersection": list(intersection_box()),
            "gaussian": [GAUSSIAN_KERNEL, GAUSSIAN_SIGMA],
            "trained": False,
            "fusion_counts": False,
            "forward_chunk": FORWARD_CHUNK,
            "gpus": {
                "cuda:0": list(GPU0_CLASSES),
                "cuda:1": list(GPU1_CLASSES),
            },
        },
        "per_class": per_class,
        "mean": mean_metrics(per_class),
        "decision": decision,
    }
    write_json(os.path.join(out_dir, "phase_report.json"), report)
    log(f"decision={decision}")
    log("=== stage15 done ===")


if __name__ == "__main__":
    main()
