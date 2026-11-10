"""Bottle Dinomaly diagnostic: global map vs fixed local zoom vs their average.

No training. This file is the only new code. It does not modify weights or
bottle_sep_b6.py. Window geometry is fixed and does not depend on the test set.
"""

import gc
import importlib.util
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.signal.windows import hann as hann_window
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

REPO = Path(r"E:\dinomaly-mvp\Dinomaly")
DATA = Path(r"E:\dinomaly-mvp\data\mvtec_anomaly_detection")
SEP_PATH = Path(r"E:\dinomaly-mvp\experiment\bottle_sep_b6.py")
WEIGHTS = Path(r"E:\dinomaly-mvp\experiment\runs\bottle_sep_b6\dinomaly_bottle.pt")
RUN = Path(r"E:\dinomaly-mvp\experiment\runs\bottle_local_zoom")
CLASS_NAME = "bottle"
IMAGE_SIZE = 448
CROP_SIZE = 392
WINDOW = 196
STRIDE = 98
STARTS = (0, 98, 196)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
REFERENCE_PIXEL_AP = 0.8842199527833942
FPR_TARGETS = (0.001, 0.01, 0.05)
GPU_USED_LIMIT_MIB = 1500
GPU_WAIT_SECONDS = 40 * 60
GPU_POLL_SECONDS = 30


def log(message):
    print(message, flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def load_sep_module():
    spec = importlib.util.spec_from_file_location("bottle_sep_b6_evalonly", SEP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def query_memory_used_mib():
    output = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    token = output.strip().splitlines()[0].strip().split()[0]
    return int(token)


def wait_for_gpu():
    started = time.time()
    while True:
        used = query_memory_used_mib()
        log(f"nvidia-smi cuda:0 memory.used={used} MiB")
        if used <= GPU_USED_LIMIT_MIB:
            return used
        elapsed = time.time() - started
        if elapsed >= GPU_WAIT_SECONDS:
            raise RuntimeError(
                f"GPU memory.used stayed above {GPU_USED_LIMIT_MIB} MiB for {GPU_WAIT_SECONDS}s; last={used} MiB"
            )
        log(f"memory.used>{GPU_USED_LIMIT_MIB}, sleep {GPU_POLL_SECONDS}s ({elapsed:.0f}s/{GPU_WAIT_SECONDS}s)")
        time.sleep(GPU_POLL_SECONDS)


def check_paths():
    required = {
        "repo": REPO,
        "data": DATA / CLASS_NAME / "test",
        "weights": WEIGHTS,
        "sep": SEP_PATH,
    }
    missing = [f"{name}:{path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing " + ", ".join(missing))


def check_window_geometry():
    if STARTS != tuple(range(0, CROP_SIZE - WINDOW + 1, STRIDE)):
        raise RuntimeError(f"window starts {STARTS} do not cover {CROP_SIZE}")
    if len(STARTS) != 3 or WINDOW + 2 * STRIDE != CROP_SIZE:
        raise RuntimeError("expected exactly 3x3 windows with no remainder")
    cover = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.int32)
    for y in STARTS:
        for x in STARTS:
            cover[y : y + WINDOW, x : x + WINDOW] += 1
    if int(cover.min()) < 1:
        raise RuntimeError("window grid leaves uncovered pixels")
    hann = hann_window(WINDOW, sym=True)
    if float(hann[0]) != 0.0 or float(hann[-1]) != 0.0:
        raise RuntimeError(f"hann endpoints must be 0, got {hann[0]} {hann[-1]}")
    weight = 0.1 + 0.9 * np.outer(hann, hann)
    return weight.astype(np.float32), {
        "crop_size": CROP_SIZE,
        "window": WINDOW,
        "stride": STRIDE,
        "starts": list(STARTS),
        "n_windows": len(STARTS) * len(STARTS),
        "covers_full_crop": True,
        "no_remainder_window": True,
        "hann": "scipy.signal.windows.hann(196, sym=True), endpoints are 0",
        "overlap_weight": "w = 0.1 + 0.9 * outer(hann(196), hann(196))",
        "weight_min": float(weight.min()),
        "weight_max": float(weight.max()),
        "weight_corner": float(weight[0, 0]),
        "rgb_zoom": "bilinear 196 to 392, align_corners=False, from RGB in [0, 1]",
        "rgb_then": "ImageNet normalize after zoom; normalized tensors are not zoomed",
        "map_paste": "28x28 two-group mean distance, bilinear to 196, align_corners=True",
        "every_test_image_runs_all_9_windows": True,
        "no_high_score_crop": True,
    }


def native_mean_distance(en, de):
    pieces = [(1.0 - F.cosine_similarity(encoder, decoder)).unsqueeze(1) for encoder, decoder in zip(en, de)]
    return torch.stack(pieces, dim=0).mean(dim=0)


def denormalize(images):
    mean = images.new_tensor(MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(STD).view(1, 3, 1, 1)
    return images * std + mean


def fuse_local(model, rgb, weight):
    batch, _, height, width = rgb.shape
    if height != CROP_SIZE or width != CROP_SIZE:
        raise RuntimeError(f"expected {CROP_SIZE} crop, got {tuple(rgb.shape)}")
    acc = torch.zeros(batch, 1, CROP_SIZE, CROP_SIZE, device=rgb.device, dtype=rgb.dtype)
    wsum = torch.zeros(batch, 1, CROP_SIZE, CROP_SIZE, device=rgb.device, dtype=rgb.dtype)
    mean = rgb.new_tensor(MEAN).view(1, 3, 1, 1)
    std = rgb.new_tensor(STD).view(1, 3, 1, 1)
    for y in STARTS:
        for x in STARTS:
            patch = rgb[:, :, y : y + WINDOW, x : x + WINDOW]
            zoom = F.interpolate(patch, size=CROP_SIZE, mode="bilinear", align_corners=False)
            model_input = (zoom - mean) / std
            en, de = model(model_input)
            dist28 = native_mean_distance(en, de)
            if dist28.shape[-2:] != (28, 28):
                raise RuntimeError(f"expected 28x28 distance, got {tuple(dist28.shape)}")
            pasted = F.interpolate(dist28, size=WINDOW, mode="bilinear", align_corners=True)
            acc[:, :, y : y + WINDOW, x : x + WINDOW] += pasted * weight
            wsum[:, :, y : y + WINDOW, x : x + WINDOW] += weight
    if not torch.isfinite(wsum).all() or float(wsum.min()) <= 0:
        raise RuntimeError("fusion weight sum is not strictly positive")
    return acc / wsum, float(wsum.min())


def is_oom(exc):
    return "out of memory" in str(exc).lower()


def evaluate(sep, model, gaussian, weight, loader, device):
    buckets = {name: [] for name in ("global_only", "local_only", "fused")}
    masks, labels, image_scores = [], [], []
    windows_run = 0
    max_abs_global_diff = 0.0
    weight_sum_min = None
    rgb_min, rgb_max = None, None
    seen = 0
    total = len(loader.dataset)
    started = time.time()
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch[0].to(device, non_blocking=True)
            target = batch[1].to(device, non_blocking=True)
            label = batch[2]
            if images.shape[-2:] != (CROP_SIZE, CROP_SIZE):
                raise RuntimeError(f"test tensor is not {CROP_SIZE}, got {tuple(images.shape)}")
            try:
                en, de = model(images)
                global_392, _ = sep.cal_anomaly_maps(en, de, images.shape[-1])
                native = native_mean_distance(en, de)
                rebuilt = F.interpolate(native, size=CROP_SIZE, mode="bilinear", align_corners=True)
                max_abs_global_diff = max(max_abs_global_diff, float((rebuilt - global_392).abs().max()))
                rgb = denormalize(images)
                if rgb_min is None:
                    rgb_min = float(rgb.min())
                    rgb_max = float(rgb.max())
                local_392, this_wmin = fuse_local(model, rgb, weight)
                weight_sum_min = this_wmin if weight_sum_min is None else min(weight_sum_min, this_wmin)
                fused_392 = 0.5 * global_392 + 0.5 * local_392
                finished = {
                    "global_only": gaussian(sep.upsample_to_256(global_392, already_392=True)),
                    "local_only": gaussian(sep.upsample_to_256(local_392, already_392=True)),
                    "fused": gaussian(sep.upsample_to_256(fused_392, already_392=True)),
                }
            except RuntimeError as exc:
                if is_oom(exc):
                    raise
                raise
            for name, amap in finished.items():
                if not torch.isfinite(amap).all():
                    raise RuntimeError(f"non-finite {name} map")
                buckets[name].append(amap.cpu())
            image_scores.append(sep.image_score(finished["global_only"]).cpu())
            mask_256 = F.interpolate(target, size=256, mode="nearest")
            masks.append(mask_256.bool().float().cpu())
            labels.append(label)
            windows_run += int(images.shape[0]) * 9
            seen += int(images.shape[0])
            log(
                f"batch {batch_index} images={seen}/{total} "
                f"elapsed_min={(time.time() - started) / 60:.2f} "
                f"alloc_mib={torch.cuda.memory_allocated(device) / (1024 ** 2):.0f}"
            )
    maps = {name: torch.cat(items, dim=0)[:, 0].numpy() for name, items in buckets.items()}
    mask_np = torch.cat(masks, dim=0)[:, 0].numpy()
    label_np = torch.cat(labels).flatten().numpy()
    scores = torch.cat(image_scores).numpy()
    return {
        "maps": maps,
        "masks": mask_np,
        "labels": label_np,
        "image_scores": scores,
        "windows_run": windows_run,
        "max_abs_global_diff": max_abs_global_diff,
        "weight_sum_min": weight_sum_min,
        "rgb_min": rgb_min,
        "rgb_max": rgb_max,
        "forward_seconds": time.time() - started,
    }


def f1_score_max(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-7)
    f1_scores = f1_scores[:-1]
    return float(f1_scores.max())


def recall_at_fprs(y_true, y_score, targets):
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=False)
    last = np.concatenate([np.flatnonzero(np.diff(fpr) > 0), [len(fpr) - 1]])
    fpr_u = fpr[last]
    tpr_u = tpr[last]
    interpolated, discrete = {}, {}
    for target in targets:
        key = f"{target:g}"
        interpolated[key] = float(np.interp(target, fpr_u, tpr_u))
        index = int(np.searchsorted(fpr_u, target, side="right") - 1)
        index = max(0, min(index, len(fpr_u) - 1))
        discrete[key] = {"recall": float(tpr_u[index]), "realized_fpr": float(fpr_u[index])}
    return interpolated, discrete


def build_regions(masks):
    structure = np.ones((9, 9), dtype=bool)
    regions = []
    empty = 0
    for mask in masks:
        defect = mask.astype(bool)
        interior = ndimage.binary_erosion(defect, structure=structure, iterations=1, border_value=0, origin=0)
        dilated = ndimage.binary_dilation(defect, structure=structure, iterations=1, border_value=0, origin=0)
        boundary = np.logical_and(dilated, np.logical_not(interior))
        far = np.logical_and(np.logical_not(dilated), np.logical_not(defect))
        if not interior.any():
            empty += 1
        regions.append((interior, boundary, far))
    return regions, empty


def micro_means(maps, regions):
    sums = {"interior": 0.0, "boundary": 0.0, "far_normal": 0.0}
    counts = {key: 0 for key in sums}
    for amap, (interior, boundary, far) in zip(maps, regions):
        for key, region in (("interior", interior), ("boundary", boundary), ("far_normal", far)):
            count = int(region.sum())
            if count:
                sums[key] += float(amap[region].astype(np.float64).sum())
                counts[key] += count
    means = {key: (sums[key] / counts[key] if counts[key] else None) for key in sums}
    return means, counts


def pixel_block(prediction, gt, regions):
    gt_flat = gt.ravel().astype(np.uint8)
    pred_flat = prediction.ravel()
    recalls, discrete = recall_at_fprs(gt_flat, pred_flat, FPR_TARGETS)
    means, counts = micro_means(prediction, regions)
    return {
        "pixel_auroc": float(roc_auc_score(gt_flat, pred_flat)),
        "pixel_ap": float(average_precision_score(gt_flat, pred_flat)),
        "pixel_f1_max": f1_score_max(gt_flat, pred_flat),
        "pixel_f1_max_note": "threshold selected on the test set; optimistic",
        "defect_recall_at_normal_fpr": recalls,
        "defect_recall_at_normal_fpr_definition": (
            "Linear interpolation on the pixel ROC. "
            "FPR is the false-positive rate among gt==0 pixels. "
            "Recall is the true-positive rate among gt==1 pixels. "
            "The operating FPR is exactly the named value. Higher score is more anomalous."
        ),
        "defect_recall_discrete_fpr_le_target": discrete,
        "region_micro_mean": means,
        "region_pixel_counts": counts,
    }


def py(value):
    if isinstance(value, dict):
        return {str(key): py(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [py(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    check_paths()
    weight_np, window_info = check_window_geometry()
    log("paths and window geometry ok; waiting for GPU before loading the model")
    memory_before = wait_for_gpu()
    sep = load_sep_module()
    from dataset import MVTecDataset, get_data_transforms

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log("load frozen bottle Dinomaly")
    model = sep.build_model(device)
    state = torch.load(WEIGHTS, map_location=device, weights_only=False)
    if "bottleneck" not in state or "decoder" not in state:
        raise KeyError(f"checkpoint keys are {list(state)} not bottleneck/decoder")
    model.bottleneck.load_state_dict(state["bottleneck"])
    model.decoder.load_state_dict(state["decoder"])
    del state
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    gaussian = sep.get_gaussian_kernel().to(device)
    gaussian.eval()
    weight = torch.from_numpy(weight_np).to(device=device, dtype=torch.float32).view(1, 1, WINDOW, WINDOW)

    data_transform, gt_transform = get_data_transforms(IMAGE_SIZE, CROP_SIZE)
    test_set = MVTecDataset(
        root=str(DATA / CLASS_NAME),
        transform=data_transform,
        gt_transform=gt_transform,
        phase="test",
    )
    log(f"test_images={len(test_set)}")

    packed = None
    used_batch = None
    for batch_size in (6, 4):
        loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        try:
            torch.cuda.reset_peak_memory_stats(device)
            log(f"evaluate batch_size={batch_size}")
            packed = evaluate(sep, model, gaussian, weight, loader, device)
            used_batch = batch_size
            break
        except RuntimeError as exc:
            if batch_size == 4 or not is_oom(exc):
                raise
            log(f"batch 6 OOM, empty cache and retry batch 4\n{exc}")
            gc.collect()
            torch.cuda.empty_cache()
    if packed is None:
        raise RuntimeError("evaluation did not run")

    if packed["max_abs_global_diff"] > 1e-3:
        raise RuntimeError(
            "average-then-upsample disagrees with official upsample-then-average "
            f"by {packed['max_abs_global_diff']}"
        )

    regions, empty_interior = build_regions(packed["masks"])
    empty_good = 0
    empty_anomaly = 0
    for label, (interior, _, _) in zip(packed["labels"], regions):
        if interior.any():
            continue
        if int(label) == 0:
            empty_good += 1
        else:
            empty_anomaly += 1
    methods = {}
    for name in ("global_only", "local_only", "fused"):
        methods[name] = pixel_block(packed["maps"][name], packed["masks"], regions)
        row = methods[name]
        log(
            f"{name}: P-AP={row['pixel_ap']:.6f} P-AUROC={row['pixel_auroc']:.6f} "
            f"P-F1max={row['pixel_f1_max']:.6f} "
            f"R@FPR={row['defect_recall_at_normal_fpr']} "
            f"regions={row['region_micro_mean']}"
        )
    image_auroc = float(roc_auc_score(packed["labels"], packed["image_scores"]))
    global_ap = methods["global_only"]["pixel_ap"]
    in_band = bool(0.86 <= global_ap <= 0.91)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    report = {
        "class_name": CLASS_NAME,
        "task": "fixed-window local zoom diagnostic, no retraining",
        "weights": str(WEIGHTS),
        "checkpoint_keys": ["bottleneck", "decoder"],
        "device": "cuda:0",
        "batch_size": used_batch,
        "test_images": int(len(test_set)),
        "test_anomaly": int((packed["labels"] == 1).sum()),
        "test_good": int((packed["labels"] == 0).sum()),
        "defect_pixels": int(packed["masks"].astype(bool).sum()),
        "normal_pixels": int((~packed["masks"].astype(bool)).sum()),
        "windows_forwarded": int(packed["windows_run"]),
        "window": window_info,
        "postprocess": {
            "global_only": "28x28 two-group mean cosine distance -> bilinear 392 align_corners=True -> bilinear 256 align_corners=False -> conv gaussian kernel 5x5 sigma=4",
            "official_equivalence": "scored global map uses bottle_sep_b6.cal_anomaly_maps (interpolate each group to 392, then mean). This is the linear equivalent of averaging the two 28x28 distances first.",
            "max_abs_diff_average_first_vs_official": packed["max_abs_global_diff"],
            "local_only": "fused 392 map -> bilinear 256 align_corners=False -> same gaussian",
            "fused": "0.5 * global_392 + 0.5 * local_392, then the same 256 resize and gaussian. No per-image min-max.",
            "higher_is_more_anomalous": True,
        },
        "fusion_weight_sum_min": packed["weight_sum_min"],
        "denormalized_rgb_min": packed["rgb_min"],
        "denormalized_rgb_max": packed["rgb_max"],
        "reference_official_pixel_ap": REFERENCE_PIXEL_AP,
        "global_only_pixel_ap": global_ap,
        "global_only_pixel_ap_abs_diff_vs_0.8842199527833942": abs(global_ap - REFERENCE_PIXEL_AP),
        "global_only_pixel_ap_in_0.86_0.91": in_band,
        "image_auroc_global_only_top1pct_mean": image_auroc,
        "image_auroc_note": "Computed once from global_only. Score is the mean of the highest 1% pixels on the final 256 gaussian map.",
        "empty_interior_images": int(empty_interior),
        "empty_interior_good_images": int(empty_good),
        "empty_interior_anomaly_images": int(empty_anomaly),
        "region_definition": {
            "structure": "9x9 all ones, one erosion and one dilation, origin centered",
            "interior": "eroded defect mask",
            "boundary": "dilation minus erosion",
            "far_normal": "outside dilation and gt==0",
            "average": "micro-average over pooled pixels, including good images where the dilation is empty",
        },
        "methods": methods,
        "peak_allocated_mib": round(float(peak_allocated), 1),
        "peak_reserved_mib": round(float(peak_reserved), 1),
        "gpu_memory_used_before_load_mib": int(memory_before),
        "forward_seconds": round(float(packed["forward_seconds"]), 1),
        "torch": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(device),
    }
    (RUN / "report.json").write_text(json.dumps(py(report), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {RUN / 'report.json'}")
    log(
        f"global_only pixel_ap={global_ap:.6f} in_band={in_band} "
        f"abs_diff={abs(global_ap - REFERENCE_PIXEL_AP):.6f} "
        f"image_auroc={image_auroc:.6f} peak_mib={peak_allocated:.1f} empty_interior={empty_interior}"
    )
    if not in_band:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        RUN.mkdir(parents=True, exist_ok=True)
        text = traceback.format_exc()
        (RUN / "error.txt").write_text(text, encoding="utf-8")
        log(text)
        raise
