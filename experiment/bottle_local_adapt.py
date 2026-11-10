"""Local-view adaptation screen on the bottle test set.

The test set has already been inspected. This is a development-stage run:
development_stage is true, there is no held-out normal validation set, and
nothing in this file changes the learning rate, step count, fusion weight,
or window after looking at test scores. Only the step-1000 weights are saved.

Geometry, overlap weights, and the 392-to-256 postprocess are imported from
bottle_local_zoom.py. The hard-mining loss and StableAdamW are imported from
bottle_sep_b6.py. Neither of those files, nor the 5000-step checkpoint, is
modified.
"""

import gc
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

REPO = Path(r"E:\dinomaly-mvp\Dinomaly")
DATA = Path(r"E:\dinomaly-mvp\data\mvtec_anomaly_detection")
ZOOM_PATH = Path(r"E:\dinomaly-mvp\experiment\bottle_local_zoom.py")
WEIGHTS = Path(r"E:\dinomaly-mvp\experiment\runs\bottle_sep_b6\dinomaly_bottle.pt")
RUN = Path(r"E:\dinomaly-mvp\experiment\runs\bottle_local_adapt")
CLASS_NAME = "bottle"
SEED = 1
STEPS = 1000
LR = 2e-4
P_HARD = 0.9
FACTOR = 0.1
BATCH_CANDIDATES = (6, 4)
FPR_TARGETS = (0.001, 0.01, 0.05)
REFERENCE_PIXEL_AP = 0.8842199527833942
DEFECT_TYPES = ("broken_large", "broken_small", "contamination")
METHOD_NAMES = (
    "global_base",
    "local_unadapted",
    "fuse_unadapted",
    "local_adapted",
    "fuse_adapted",
    "global_extra",
    "local_extra",
    "fuse_extra",
)
SPLIT_METHODS = ("global_base", "fuse_unadapted", "fuse_adapted", "fuse_extra")
GPU_USED_LIMIT_MIB = 1500
GPU_WAIT_SECONDS = 40 * 60
GPU_POLL_SECONDS = 30


class RestartBatch(Exception):
    def __init__(self, batch_size, kind):
        self.batch_size = batch_size
        self.kind = kind
        super().__init__(f"{kind} first step OOM, restart both runs at batch {batch_size}")


class PipelineError(RuntimeError):
    pass


def log(message):
    text = str(message)
    print(text, flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


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
    if isinstance(value, np.ndarray):
        return py(value.tolist())
    return value


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    stat = path.stat()
    return {"sha256": digest.hexdigest(), "bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


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


def load_zoom_module():
    spec = importlib.util.spec_from_file_location("bottle_local_zoom_geom", ZOOM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_oom(exc):
    return "out of memory" in str(exc).lower()


class RGBSet(Dataset):
    def __init__(self, rgb):
        self.rgb = rgb

    def __len__(self):
        return int(self.rgb.shape[0])

    def __getitem__(self, index):
        return self.rgb[index]


def normalize_full(rgb, zoom):
    mean = rgb.new_tensor(zoom.MEAN).view(1, 3, 1, 1)
    std = rgb.new_tensor(zoom.STD).view(1, 3, 1, 1)
    return (rgb - mean) / std


def zoom_selected_windows(rgb, window_index, zoom):
    """One uniformly chosen window per image, using fuse_local's crop and zoom.

    fuse_local walks STARTS in row-major order. Index i // 3 is the y start and
    i % 3 is the x start. The resize is bilinear 196 -> 392 with
    align_corners=False, then ImageNet normalization. Normalized tensors are
    not resized.
    """
    starts = tuple(zoom.STARTS)
    patches = []
    for row, index in enumerate(window_index.tolist()):
        if index < 0 or index >= len(starts) * len(starts):
            raise RuntimeError(f"window index {index} is outside the 3x3 grid")
        y = starts[index // len(starts)]
        x = starts[index % len(starts)]
        patches.append(rgb[row : row + 1, :, y : y + zoom.WINDOW, x : x + zoom.WINDOW])
    patch = torch.cat(patches, dim=0)
    zoomed = torch.nn.functional.interpolate(patch, size=zoom.CROP_SIZE, mode="bilinear", align_corners=False)
    return normalize_full(zoomed, zoom)


def assert_zoom_matches_fuse_local(zoom):
    rgb = torch.rand(2, 3, zoom.CROP_SIZE, zoom.CROP_SIZE)
    chosen = torch.tensor([0, 4], dtype=torch.long)
    out = zoom_selected_windows(rgb, chosen, zoom)
    starts = tuple(zoom.STARTS)
    for row, index in enumerate(chosen.tolist()):
        y = starts[index // 3]
        x = starts[index % 3]
        patch = rgb[row : row + 1, :, y : y + zoom.WINDOW, x : x + zoom.WINDOW]
        ref = torch.nn.functional.interpolate(patch, size=zoom.CROP_SIZE, mode="bilinear", align_corners=False)
        ref = normalize_full(ref, zoom)
        if not torch.allclose(out[row], ref[0]):
            raise PipelineError("selected-window zoom disagrees with fuse_local's per-window math")


def load_train_rgb(sep):
    folder = DATA / CLASS_NAME / "train" / "good"
    paths = sorted(folder.glob("*.png"))
    if len(paths) != 209:
        raise PipelineError(f"expected 209 train/good images, found {len(paths)} in {folder}")
    arrays = [sep.load_crop(path) for path in paths]
    rgb = torch.from_numpy(np.stack(arrays, axis=0)).permute(0, 3, 1, 2).contiguous()
    if rgb.shape != (209, 3, 392, 392):
        raise PipelineError(f"train rgb shape {tuple(rgb.shape)}")
    if float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0:
        raise PipelineError(f"train rgb outside [0, 1]: {float(rgb.min())} {float(rgb.max())}")
    return rgb, [str(path) for path in paths]


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if "bottleneck" not in state or "decoder" not in state:
        raise KeyError(f"checkpoint keys are {list(state)} not bottleneck/decoder")
    model.bottleneck.load_state_dict(state["bottleneck"])
    model.decoder.load_state_dict(state["decoder"])
    return state


def freeze_encoder_train_heads(model):
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    for param in model.bottleneck.parameters():
        param.requires_grad_(True)
    for param in model.decoder.parameters():
        param.requires_grad_(True)
    model.train()
    drop = model.bottleneck[0].drop
    if not model.training or not drop.training or abs(float(drop.p) - 0.2) > 1e-8:
        raise PipelineError(
            f"dropout not left open: model.training={model.training} drop.training={drop.training} p={drop.p}"
        )
    if any(param.requires_grad for param in model.encoder.parameters()):
        raise PipelineError("encoder still requires grad")
    if not any(param.requires_grad for param in model.bottleneck.parameters()):
        raise PipelineError("bottleneck is frozen")
    if not any(param.requires_grad for param in model.decoder.parameters()):
        raise PipelineError("decoder is frozen")


def clear_saved_weights():
    for name in ("local_adapt_step1000.pt", "full_extra_step1000.pt"):
        path = RUN / name
        if path.exists():
            path.unlink()


def train_one(model, rgb, batch_size, kind, device, sep, zoom):
    sep.setup_seed(SEED)
    load_checkpoint(model, WEIGHTS, device)
    freeze_encoder_train_heads(model)
    trainable = nn.ModuleList([model.bottleneck, model.decoder])
    optimizer = sep.StableAdamW(
        [{"params": trainable.parameters()}],
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=True,
        eps=1e-8,
    )
    loader_gen = torch.Generator()
    loader_gen.manual_seed(SEED)
    window_gen = torch.Generator()
    window_gen.manual_seed(SEED)
    loader = DataLoader(
        RGBSet(rgb),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=loader_gen,
    )
    torch.cuda.reset_peak_memory_stats(device)
    losses = []
    step_times = []
    loss_at_step = {}
    started = time.perf_counter()
    step = 0
    log(
        f"train {kind} batch={batch_size} steps={STEPS} lr={LR} p={P_HARD} factor={FACTOR} "
        f"pool={len(rgb)} drop_last_remainder={len(rgb) % batch_size}"
    )
    while step < STEPS:
        for batch in loader:
            torch.cuda.synchronize(device)
            step_started = time.perf_counter()
            images = batch.to(device, non_blocking=True)
            try:
                if kind == "local_adapt":
                    choice = torch.randint(0, 9, (images.shape[0],), generator=window_gen)
                    model_in = zoom_selected_windows(images, choice, zoom)
                elif kind == "full_extra":
                    model_in = normalize_full(images, zoom)
                else:
                    raise RuntimeError(f"unknown kind {kind}")
                en, de = model(model_in)
                loss = sep.global_cosine_hm_percent(en, de, p=P_HARD, factor=FACTOR)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                torch.cuda.synchronize(device)
            except RuntimeError as exc:
                if step == 0 and is_oom(exc):
                    raise RestartBatch(4, kind) from exc
                raise
            losses.append(float(loss.detach()))
            step_times.append(time.perf_counter() - step_started)
            step += 1
            if step == 1 or step % 100 == 0:
                loss_at_step[step] = losses[-1]
                peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                log(
                    f"{kind} {step}/{STEPS} loss={losses[-1]:.6f} lr={optimizer.param_groups[0]['lr']:.8f} "
                    f"step_s={step_times[-1]:.3f} peak_mib={peak:.0f} "
                    f"elapsed_min={(time.perf_counter() - started) / 60:.2f}"
                )
            if step == STEPS:
                break
    if step != STEPS or len(losses) != STEPS:
        raise PipelineError(f"{kind} stopped at step {step}")
    if abs(float(optimizer.param_groups[0]["lr"]) - LR) > 1e-12:
        raise PipelineError(f"{kind} learning rate drifted to {optimizer.param_groups[0]['lr']}")
    save_path = RUN / f"{kind}_step1000.pt"
    torch.save(
        {
            "bottleneck": model.bottleneck.state_dict(),
            "decoder": model.decoder.state_dict(),
            "step": STEPS,
            "kind": kind,
            "base_weights": str(WEIGHTS),
            "batch_size": batch_size,
            "lr": LR,
            "p": P_HARD,
            "factor": FACTOR,
        },
        save_path,
    )
    model.eval()
    loss_every_100 = [loss_at_step[s] for s in range(100, STEPS + 1, 100)]
    loss_mean_every_100 = [float(np.mean(losses[i : i + 100])) for i in range(0, STEPS, 100)]
    step_mean_every_100 = [float(np.mean(step_times[i : i + 100])) for i in range(0, STEPS, 100)]
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    report = {
        "kind": kind,
        "steps": STEPS,
        "saved_step": STEPS,
        "checkpoint": str(save_path),
        "batch_size": batch_size,
        "lr": LR,
        "lr_schedule": "constant",
        "p": P_HARD,
        "factor": FACTOR,
        "optimizer": "StableAdamW betas=(0.9, 0.999) weight_decay=1e-4 amsgrad=True eps=1e-8",
        "grad_clip_max_norm": 0.1,
        "encoder_frozen": True,
        "model_train": True,
        "bottleneck_dropout_p": 0.2,
        "dropout_left_on": True,
        "sampling_pool": 209,
        "no_held_out_train_images": True,
        "drop_last": True,
        "images_dropped_per_epoch": int(len(rgb) % batch_size),
        "seconds": float(time.perf_counter() - started),
        "step_seconds_mean": float(np.mean(step_times)),
        "step_seconds_mean_every_100": step_mean_every_100,
        "loss_every_100": loss_every_100,
        "loss_mean_every_100": loss_mean_every_100,
        "peak_allocated_mib": round(float(peak), 1),
        "seed": SEED,
    }
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return report


def postprocess(sep, gaussian, amap_392):
    return gaussian(sep.upsample_to_256(amap_392, already_392=True))


def full_and_local(sep, zoom, model, images, weight):
    en, de = model(images)
    global_392, _ = sep.cal_anomaly_maps(en, de, images.shape[-1])
    native = zoom.native_mean_distance(en, de)
    rebuilt = torch.nn.functional.interpolate(native, size=zoom.CROP_SIZE, mode="bilinear", align_corners=True)
    diff = float((rebuilt - global_392).abs().max())
    rgb = zoom.denormalize(images)
    local_392, weight_min = zoom.fuse_local(model, rgb, weight)
    return global_392, local_392, diff, float(weight_min), float(rgb.min()), float(rgb.max())


def collect_base(sep, zoom, model, gaussian, weight, loader, device):
    model.eval()
    paths, labels, masks = [], [], []
    buckets = {name: [] for name in ("global_base", "local_unadapted", "fuse_unadapted")}
    global_392_parts = []
    image_scores = []
    max_diff = 0.0
    weight_min = None
    rgb_min, rgb_max = None, None
    full_forwards = 0
    window_forwards = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch[0].to(device, non_blocking=True)
            target = batch[1].to(device, non_blocking=True)
            label = batch[2]
            path = batch[3]
            global_392, local_392, diff, this_wmin, this_min, this_max = full_and_local(
                sep, zoom, model, images, weight
            )
            max_diff = max(max_diff, diff)
            weight_min = this_wmin if weight_min is None else min(weight_min, this_wmin)
            rgb_min = this_min if rgb_min is None else min(rgb_min, this_min)
            rgb_max = this_max if rgb_max is None else max(rgb_max, this_max)
            fused_392 = 0.5 * global_392 + 0.5 * local_392
            finished = {
                "global_base": postprocess(sep, gaussian, global_392),
                "local_unadapted": postprocess(sep, gaussian, local_392),
                "fuse_unadapted": postprocess(sep, gaussian, fused_392),
            }
            for name, amap in finished.items():
                if not torch.isfinite(amap).all():
                    raise PipelineError(f"non-finite {name}")
                buckets[name].append(amap.cpu())
            image_scores.append(sep.image_score(finished["global_base"]).cpu())
            global_392_parts.append(global_392.cpu())
            mask_256 = torch.nn.functional.interpolate(target, size=256, mode="nearest")
            masks.append(mask_256.bool().float().cpu())
            label_tensor = label if torch.is_tensor(label) else torch.as_tensor(label)
            labels.append(label_tensor.reshape(-1).cpu())
            paths.extend([str(item) for item in path])
            full_forwards += int(images.shape[0])
            window_forwards += int(images.shape[0]) * 9
            log(f"base batch {batch_index} images={len(paths)} elapsed_min={(time.perf_counter() - started) / 60:.2f}")
    maps = {name: torch.cat(items, dim=0)[:, 0].numpy() for name, items in buckets.items()}
    return {
        "maps": maps,
        "global_392": torch.cat(global_392_parts, dim=0),
        "masks": torch.cat(masks, dim=0)[:, 0].numpy(),
        "labels": torch.cat(labels).flatten().numpy(),
        "paths": paths,
        "image_scores": torch.cat(image_scores).numpy(),
        "max_abs_global_diff": max_diff,
        "weight_sum_min": weight_min,
        "rgb_min": rgb_min,
        "rgb_max": rgb_max,
        "full_forwards": full_forwards,
        "window_forwards": window_forwards,
        "seconds": time.perf_counter() - started,
    }


def collect_local_fused(sep, zoom, model, gaussian, weight, loader, global_392, device, local_name, fuse_name):
    model.eval()
    local_parts, fuse_parts = [], []
    paths = []
    offset = 0
    window_forwards = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch[0].to(device, non_blocking=True)
            path = batch[3]
            count = int(images.shape[0])
            rgb = zoom.denormalize(images)
            local_392, _ = zoom.fuse_local(model, rgb, weight)
            base_392 = global_392[offset : offset + count].to(device, non_blocking=True)
            if base_392.shape != local_392.shape:
                raise PipelineError(f"cached global {tuple(base_392.shape)} != local {tuple(local_392.shape)}")
            fused_392 = 0.5 * base_392 + 0.5 * local_392
            local_256 = postprocess(sep, gaussian, local_392)
            fuse_256 = postprocess(sep, gaussian, fused_392)
            if not torch.isfinite(local_256).all() or not torch.isfinite(fuse_256).all():
                raise PipelineError(f"non-finite {local_name}")
            local_parts.append(local_256.cpu())
            fuse_parts.append(fuse_256.cpu())
            paths.extend(list(path))
            offset += count
            window_forwards += count * 9
            log(
                f"{local_name} batch {batch_index} images={offset} "
                f"elapsed_min={(time.perf_counter() - started) / 60:.2f}"
            )
    if offset != int(global_392.shape[0]):
        raise PipelineError(f"{local_name} saw {offset} images, cache has {global_392.shape[0]}")
    return {
        local_name: torch.cat(local_parts, dim=0)[:, 0].numpy(),
        fuse_name: torch.cat(fuse_parts, dim=0)[:, 0].numpy(),
        "paths": paths,
        "window_forwards": window_forwards,
        "full_forwards": 0,
        "seconds": time.perf_counter() - started,
    }


def collect_extra(sep, zoom, model, gaussian, weight, loader, device):
    model.eval()
    buckets = {name: [] for name in ("global_extra", "local_extra", "fuse_extra")}
    paths = []
    full_forwards = 0
    window_forwards = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch[0].to(device, non_blocking=True)
            path = batch[3]
            global_392, local_392, _, _, _, _ = full_and_local(sep, zoom, model, images, weight)
            fused_392 = 0.5 * global_392 + 0.5 * local_392
            finished = {
                "global_extra": postprocess(sep, gaussian, global_392),
                "local_extra": postprocess(sep, gaussian, local_392),
                "fuse_extra": postprocess(sep, gaussian, fused_392),
            }
            for name, amap in finished.items():
                if not torch.isfinite(amap).all():
                    raise PipelineError(f"non-finite {name}")
                buckets[name].append(amap.cpu())
            paths.extend(list(path))
            count = int(images.shape[0])
            full_forwards += count
            window_forwards += count * 9
            log(f"extra batch {batch_index} images={len(paths)} elapsed_min={(time.perf_counter() - started) / 60:.2f}")
    maps = {name: torch.cat(items, dim=0)[:, 0].numpy() for name, items in buckets.items()}
    maps["paths"] = paths
    maps["full_forwards"] = full_forwards
    maps["window_forwards"] = window_forwards
    maps["seconds"] = time.perf_counter() - started
    return maps


def run_eval(fn):
    for batch_size in BATCH_CANDIDATES:
        try:
            torch.cuda.empty_cache()
            return fn(batch_size), batch_size
        except RuntimeError as exc:
            if batch_size == BATCH_CANDIDATES[-1] or not is_oom(exc):
                raise
            log(f"eval batch {batch_size} OOM, retry batch 4\n{exc}")
            gc.collect()
            torch.cuda.empty_cache()
    raise PipelineError("evaluation did not run")


def build_regions(masks):
    structure = np.ones((9, 9), dtype=bool)
    regions = []
    empty = 0
    for index, mask in enumerate(masks):
        defect = mask.astype(bool)
        interior = ndimage.binary_erosion(defect, structure=structure, iterations=1, border_value=0, origin=0)
        dilated = ndimage.binary_dilation(defect, structure=structure, iterations=1, border_value=0, origin=0)
        inner_edge = np.logical_and(defect, np.logical_not(interior))
        outer = np.logical_and(dilated, np.logical_not(defect))
        far = np.logical_and(np.logical_not(dilated), np.logical_not(defect))
        covered = interior.astype(np.int32) + inner_edge.astype(np.int32) + outer.astype(np.int32) + far.astype(np.int32)
        if int(covered.min()) != 1 or int(covered.max()) != 1:
            raise PipelineError(f"region partition failed on image {index}")
        if not interior.any():
            empty += 1
        regions.append(
            {
                "interior": interior,
                "inner_edge": inner_edge,
                "outer_neighborhood": outer,
                "far_normal": far,
            }
        )
    return regions, empty, {
        "structure": "9x9 all ones, one binary erosion and one binary dilation, origin centered, border_value 0",
        "interior": "eroded defect mask",
        "inner_edge": "original defect mask minus erosion",
        "outer_neighborhood": "dilation minus original defect mask",
        "far_normal": "outside dilation and gt==0",
        "average": "micro-average over pooled pixels; good images have empty defect masks and contribute to far_normal",
    }


def count_empty(labels, regions):
    empty_good = 0
    empty_anomaly = 0
    for label, region in zip(labels, regions):
        if region["interior"].any():
            continue
        if int(label) == 0:
            empty_good += 1
        else:
            empty_anomaly += 1
    return empty_good, empty_anomaly


def micro_means(amap, regions):
    keys = ("interior", "inner_edge", "outer_neighborhood", "far_normal")
    sums = {key: 0.0 for key in keys}
    counts = {key: 0 for key in keys}
    for image, region in zip(amap, regions):
        for key in keys:
            mask = region[key]
            count = int(mask.sum())
            if count:
                sums[key] += float(image[mask].astype(np.float64).sum())
                counts[key] += count
    means = {key: (sums[key] / counts[key] if counts[key] else None) for key in keys}
    return means, counts


def pixel_block(prediction, gt, regions, zoom):
    gt_flat = gt.ravel().astype(np.uint8)
    pred_flat = prediction.ravel()
    recalls, discrete = zoom.recall_at_fprs(gt_flat, pred_flat, FPR_TARGETS)
    means, counts = micro_means(prediction, regions)
    return {
        "pixel_auroc": float(roc_auc_score(gt_flat, pred_flat)),
        "pixel_ap": float(average_precision_score(gt_flat, pred_flat)),
        "pixel_f1_max": zoom.f1_score_max(gt_flat, pred_flat),
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


def subset_pixel_ap(prediction, gt, indices):
    chosen_pred = prediction[indices]
    chosen_gt = gt[indices].ravel().astype(np.uint8)
    return {
        "pixel_ap": float(average_precision_score(chosen_gt, chosen_pred.ravel())),
        "pixel_auroc": float(roc_auc_score(chosen_gt, chosen_pred.ravel())),
        "n_images": int(len(indices)),
        "positive_pixels": int(chosen_gt.sum()),
        "negative_pixels": int((chosen_gt == 0).sum()),
    }


def recall_at_fpr_sets(pos_scores, neg_scores, zoom, target=0.001):
    y_true = np.concatenate(
        [np.zeros(neg_scores.size, dtype=np.uint8), np.ones(pos_scores.size, dtype=np.uint8)]
    )
    y_score = np.concatenate([neg_scores.ravel(), pos_scores.ravel()])
    recalls, discrete = zoom.recall_at_fprs(y_true, y_score, (target,))
    return recalls[f"{target:g}"], discrete[f"{target:g}"]


def type_and_area_splits(maps, masks, labels, types, zoom):
    good = np.flatnonzero(labels == 0)
    anomaly = np.flatnonzero(labels == 1)
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.int64)
    anomaly_areas = areas[anomaly]
    median = float(np.median(anomaly_areas))
    small = anomaly[anomaly_areas < median]
    large = anomaly[anomaly_areas >= median]
    neg_good = None
    splits = {"by_defect_type": {}, "by_area": {}}
    for defect_type in DEFECT_TYPES:
        type_idx = np.flatnonzero(types == defect_type)
        if type_idx.size == 0:
            raise PipelineError(f"missing defect type {defect_type}")
        image_idx = np.concatenate([good, type_idx])
        entry = {}
        for name in SPLIT_METHODS:
            block = subset_pixel_ap(maps[name], masks, image_idx)
            pos = maps[name][type_idx][masks[type_idx].astype(bool)]
            neg = maps[name][good].ravel()
            recall, discrete = recall_at_fpr_sets(pos, neg, zoom, target=0.001)
            block["recall_at_normal_fpr_0.001"] = recall
            block["recall_at_normal_fpr_0.001_discrete"] = discrete
            block["recall_normal_pixels"] = "all pixels of all normal test images"
            block["recall_positive_pixels"] = "gt==1 pixels of this defect type only"
            block["pixel_ap_images"] = "all normal test images plus this defect type"
            entry[name] = block
        splits["by_defect_type"][defect_type] = {
            "n_defect_images": int(type_idx.size),
            "methods": entry,
        }
    for group_name, group_idx in (("small", small), ("large", large)):
        image_idx = np.concatenate([good, group_idx])
        entry = {}
        for name in SPLIT_METHODS:
            block = subset_pixel_ap(maps[name], masks, image_idx)
            block["pixel_ap_images"] = "all normal test images plus this area group"
            entry[name] = block
        splits["by_area"][group_name] = {
            "rule": "anomaly test images only; small means defect_pixels_256 < median; large means defect_pixels_256 >= median",
            "median_used_only_to_describe_groups": True,
            "median_defect_pixels_256": median,
            "n_defect_images": int(group_idx.size),
            "methods": entry,
        }
    splits["area_counts"] = {
        "n_anomaly": int(anomaly.size),
        "n_small": int(small.size),
        "n_large": int(large.size),
        "median_defect_pixels_256": median,
        "min_defect_pixels_256": int(anomaly_areas.min()),
        "max_defect_pixels_256": int(anomaly_areas.max()),
        "median_is_not_a_score_threshold": True,
    }
    return splits, areas


def image_region_mean(amap, mask):
    if mask is None or not np.any(mask):
        return None
    return float(amap[mask].astype(np.float64).mean())


def select_cases(paths, types, labels, areas, maps, regions):
    recovery = []
    for index, label in enumerate(labels):
        if int(label) != 1:
            continue
        global_interior = image_region_mean(maps["global_base"][index], regions[index]["interior"])
        adapted_interior = image_region_mean(maps["local_adapted"][index], regions[index]["interior"])
        if global_interior is None or adapted_interior is None:
            continue
        gap = adapted_interior - global_interior
        if gap > 0:
            recovery.append((gap, index, global_interior))
    if len(recovery) < 3:
        recovery = []
        for index, label in enumerate(labels):
            if int(label) != 1:
                continue
            global_interior = image_region_mean(maps["global_base"][index], regions[index]["interior"])
            adapted_interior = image_region_mean(maps["local_adapted"][index], regions[index]["interior"])
            if global_interior is None or adapted_interior is None:
                continue
            recovery.append((adapted_interior - global_interior, index, global_interior))
    interior_median = float(np.median([item[2] for item in recovery])) if recovery else None
    missed = [item for item in recovery if interior_median is not None and item[2] < interior_median]
    missed.sort(key=lambda item: item[0], reverse=True)
    recovery_rule = (
        "anomaly images with nonempty interior, local_adapted interior mean > global_base interior mean, "
        "and global_base interior mean below the median of those positive-gap interiors; largest gap first"
    )
    chosen_recovery = missed[:3]
    if len(chosen_recovery) < 3:
        recovery_rule += "; fewer than 3 passed the median filter, so the largest positive interior gaps are listed"
        chosen_recovery = sorted(recovery, key=lambda item: item[0], reverse=True)[:3]

    raised = []
    far_values = []
    for index, region in enumerate(regions):
        global_far = image_region_mean(maps["global_base"][index], region["far_normal"])
        adapted_far = image_region_mean(maps["local_adapted"][index], region["far_normal"])
        if global_far is None or adapted_far is None:
            continue
        far_values.append(global_far)
        gap = adapted_far - global_far
        if gap > 0:
            raised.append((gap, index, global_far))
    if len(raised) < 3:
        raised = []
        for index, region in enumerate(regions):
            global_far = image_region_mean(maps["global_base"][index], region["far_normal"])
            adapted_far = image_region_mean(maps["local_adapted"][index], region["far_normal"])
            if global_far is None or adapted_far is None:
                continue
            raised.append((adapted_far - global_far, index, global_far))
    far_median = float(np.median(far_values)) if far_values else None
    low = [item for item in raised if far_median is not None and item[2] < far_median]
    low.sort(key=lambda item: item[0], reverse=True)
    raised_rule = (
        "test images whose global_base far-normal mean is below the median far-normal mean, "
        "and whose local_adapted far-normal mean is higher than global_base; largest increase first"
    )
    chosen_raised = low[:3]
    if len(chosen_raised) < 3:
        raised_rule += "; fewer than 3 passed the median filter, so the largest far-normal increases are listed"
        chosen_raised = sorted(raised, key=lambda item: item[0], reverse=True)[:3]

    def pack(index):
        interior = {}
        far = {}
        for name in ("global_base", "local_unadapted", "local_adapted"):
            interior[name] = image_region_mean(maps[name][index], regions[index]["interior"])
            far[name] = image_region_mean(maps[name][index], regions[index]["far_normal"])
        return {
            "path": paths[index],
            "defect_type": str(types[index]),
            "label": int(labels[index]),
            "defect_pixels_256": int(areas[index]),
            "interior_mean": interior,
            "far_normal_mean": far,
        }

    return {
        "global_missed_adapted_local_recovered": {
            "rule": recovery_rule,
            "images": [pack(index) for _, index, _ in chosen_recovery],
        },
        "global_far_low_adapted_local_raises_far_normal": {
            "rule": raised_rule,
            "images": [pack(index) for _, index, _ in chosen_raised],
        },
    }


def recall_of(block):
    return float(block["defect_recall_at_normal_fpr"]["0.001"])


def build_verdict(methods):
    local_u = methods["local_unadapted"]["region_micro_mean"]
    local_a = methods["local_adapted"]["region_micro_mean"]
    far_down = local_a["far_normal"] < local_u["far_normal"]
    outer_down = local_a["outer_neighborhood"] < local_u["outer_neighborhood"]
    adapted = methods["fuse_adapted"]
    unadapted = methods["fuse_unadapted"]
    base = methods["global_base"]
    extra = methods["fuse_extra"]
    ap_vs_unadapted = adapted["pixel_ap"] >= unadapted["pixel_ap"]
    ap_vs_base = adapted["pixel_ap"] >= base["pixel_ap"]
    recall_vs_unadapted = recall_of(adapted) >= recall_of(unadapted)
    recall_vs_base = recall_of(adapted) >= recall_of(base)
    holds = ap_vs_unadapted and ap_vs_base and recall_vs_unadapted and recall_vs_base
    extra_ap = extra["pixel_ap"] >= adapted["pixel_ap"]
    extra_recall = recall_of(extra) >= recall_of(adapted)
    extra_reaches = extra_ap and extra_recall
    far_word = "降低" if far_down else "没有降低"
    outer_word = "降低" if outer_down else "没有降低"
    hold_word = "都保持或提高" if holds else "没有同时保持或提高"
    if extra_reaches:
        extra_sentence = (
            "fuse_extra 的像素 AP 和 0.001 召回都已经达到或超过 fuse_adapted，"
            "收益更像来自多训 1000 步，而不是局部视野本身。"
        )
    else:
        extra_sentence = (
            "fuse_extra 没有在像素 AP 和 0.001 召回上同时达到 fuse_adapted。"
            f"像素 AP {'达到或超过' if extra_ap else '低于'} fuse_adapted，"
            f"0.001 召回 {'达到或超过' if extra_recall else '低于'} fuse_adapted。"
        )
    text = (
        f"适配后的局部图相对未适配局部图，远处正常均分从 {local_u['far_normal']:.6f} 到 "
        f"{local_a['far_normal']:.6f}，{far_word}；外侧邻域均分从 {local_u['outer_neighborhood']:.6f} 到 "
        f"{local_a['outer_neighborhood']:.6f}，{outer_word}。"
        f"fuse_adapted 的像素 AP={adapted['pixel_ap']:.6f}、召回@0.001={recall_of(adapted):.6f}，"
        f"相对 fuse_unadapted（AP={unadapted['pixel_ap']:.6f}，召回={recall_of(unadapted):.6f}）和 "
        f"global_base（AP={base['pixel_ap']:.6f}，召回={recall_of(base):.6f}）{hold_word}。"
        + extra_sentence
    )
    checks = {
        "local_far_normal_mean_decreased_vs_local_unadapted": bool(far_down),
        "local_outer_neighborhood_mean_decreased_vs_local_unadapted": bool(outer_down),
        "fuse_adapted_pixel_ap_ge_fuse_unadapted": bool(ap_vs_unadapted),
        "fuse_adapted_pixel_ap_ge_global_base": bool(ap_vs_base),
        "fuse_adapted_recall_0.001_ge_fuse_unadapted": bool(recall_vs_unadapted),
        "fuse_adapted_recall_0.001_ge_global_base": bool(recall_vs_base),
        "fuse_adapted_holds_or_improves_ap_and_recall_vs_both": bool(holds),
        "fuse_extra_pixel_ap_ge_fuse_adapted": bool(extra_ap),
        "fuse_extra_recall_0.001_ge_fuse_adapted": bool(extra_recall),
        "fuse_extra_reaches_or_exceeds_fuse_adapted_on_ap_and_recall_0.001": bool(extra_reaches),
        "comparison_rule": (
            "Decreased means the adapted local micro-mean is strictly lower. "
            "Held or improved means fuse_adapted is greater than or equal to both baselines "
            "on pixel AP and on recall at normal FPR 0.001. "
            "fuse_extra reaches fuse_adapted only when both of those two metrics are greater than or equal."
        ),
    }
    return text, checks


def deltas(splits):
    out = {"by_defect_type": {}, "by_area": {}}
    for defect_type, payload in splits["by_defect_type"].items():
        adapted = payload["methods"]["fuse_adapted"]["pixel_ap"]
        base = payload["methods"]["global_base"]["pixel_ap"]
        out["by_defect_type"][defect_type] = {
            "fuse_adapted_pixel_ap": adapted,
            "global_base_pixel_ap": base,
            "fuse_adapted_minus_global_base": adapted - base,
        }
    for group_name, payload in splits["by_area"].items():
        adapted = payload["methods"]["fuse_adapted"]["pixel_ap"]
        base = payload["methods"]["global_base"]["pixel_ap"]
        out["by_area"][group_name] = {
            "fuse_adapted_pixel_ap": adapted,
            "global_base_pixel_ap": base,
            "fuse_adapted_minus_global_base": adapted - base,
        }
    return out


def make_loader(test_set, batch_size):
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    weight_before = fingerprint(WEIGHTS)
    log("load geometry module and frozen recipe")
    zoom = load_zoom_module()
    sep = zoom.load_sep_module()
    weight_np, window_info = zoom.check_window_geometry()
    assert_zoom_matches_fuse_local(zoom)
    memory_before = wait_for_gpu()
    sep.setup_seed(SEED)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    log("build model and load the 5000-step bottleneck/decoder")
    model = sep.build_model(device)
    base_state = load_checkpoint(model, WEIGHTS, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    gaussian = sep.get_gaussian_kernel().to(device)
    gaussian.eval()
    weight = torch.from_numpy(np.ascontiguousarray(weight_np)).to(device=device, dtype=torch.float32)
    weight = weight.view(1, 1, zoom.WINDOW, zoom.WINDOW)

    from dataset import MVTecDataset, get_data_transforms

    data_transform, gt_transform = get_data_transforms(sep.IMAGE_SIZE, sep.CROP_SIZE)
    test_set = MVTecDataset(
        root=str(DATA / CLASS_NAME),
        transform=data_transform,
        gt_transform=gt_transform,
        phase="test",
    )
    path_to_type = {str(path): str(kind) for path, kind in zip(test_set.img_paths, test_set.types)}
    log(f"test_images={len(test_set)}")

    def base_fn(batch_size):
        log(f"evaluate base batch_size={batch_size}")
        return collect_base(sep, zoom, model, gaussian, weight, make_loader(test_set, batch_size), device)

    base, eval_batch = run_eval(base_fn)
    if base["max_abs_global_diff"] > 1e-3:
        raise PipelineError(f"global map disagrees with native distance by {base['max_abs_global_diff']}")
    regions, empty_interior, region_definition = build_regions(base["masks"])
    empty_good, empty_anomaly = count_empty(base["labels"], regions)
    region_definition["empty_interior_images"] = int(empty_interior)
    region_definition["empty_interior_good_images"] = int(empty_good)
    region_definition["empty_interior_anomaly_images"] = int(empty_anomaly)
    defect_pixels = int(base["masks"].astype(bool).sum())
    interior_pixels = int(sum(int(region["interior"].sum()) for region in regions))
    inner_pixels = int(sum(int(region["inner_edge"].sum()) for region in regions))
    outer_pixels = int(sum(int(region["outer_neighborhood"].sum()) for region in regions))
    far_pixels = int(sum(int(region["far_normal"].sum()) for region in regions))
    log(
        f"mask pixels defect={defect_pixels} interior={interior_pixels} "
        f"inner={inner_pixels} outer={outer_pixels} far={far_pixels}"
    )
    if (defect_pixels, interior_pixels, far_pixels, inner_pixels + outer_pixels) != (425068, 301636, 4881606, 256246):
        raise PipelineError(
            "region or mask counts disagree with the previous bottle_local_zoom run; fix the pipeline before training"
        )
    global_block = pixel_block(base["maps"]["global_base"], base["masks"], regions, zoom)
    global_ap = global_block["pixel_ap"]
    log(
        f"global_base pixel_ap={global_ap:.10f} abs_diff_vs_reference={abs(global_ap - REFERENCE_PIXEL_AP):.10f} "
        f"empty_interior={region_definition['empty_interior_images']}"
    )
    if not (0.86 <= global_ap <= 0.91):
        raise PipelineError(f"global_base pixel_ap {global_ap} is outside [0.86, 0.91]; fix the pipeline before training")
    if abs(global_ap - REFERENCE_PIXEL_AP) > 1e-3:
        raise PipelineError(
            f"global_base pixel_ap {global_ap} drifted from {REFERENCE_PIXEL_AP} by more than 1e-3"
        )

    log("cache train/good rgb; no image is held out")
    train_rgb, train_paths = load_train_rgb(sep)
    for param in model.bottleneck.parameters():
        param.requires_grad_(True)
    for param in model.decoder.parameters():
        param.requires_grad_(True)

    batch_size = BATCH_CANDIDATES[0]
    oom_restarted = False
    try:
        local_train = train_one(model, train_rgb, batch_size, "local_adapt", device, sep, zoom)
        extra_train = train_one(model, train_rgb, batch_size, "full_extra", device, sep, zoom)
    except RestartBatch as restart:
        oom_restarted = True
        log(str(restart))
        clear_saved_weights()
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        batch_size = restart.batch_size
        local_train = train_one(model, train_rgb, batch_size, "local_adapt", device, sep, zoom)
        extra_train = train_one(model, train_rgb, batch_size, "full_extra", device, sep, zoom)

    load_checkpoint(model, RUN / "local_adapt_step1000.pt", device)
    model.eval()

    def adapted_fn(batch):
        log(f"evaluate adapted local batch_size={batch}")
        return collect_local_fused(
            sep,
            zoom,
            model,
            gaussian,
            weight,
            make_loader(test_set, batch),
            base["global_392"],
            device,
            "local_adapted",
            "fuse_adapted",
        )

    adapted, adapted_batch = run_eval(adapted_fn)
    if adapted["paths"] != base["paths"]:
        raise PipelineError("adapted evaluation image order differs from global_base")
    if adapted["full_forwards"] != 0:
        raise PipelineError("adapted weights were used for a full-image forward")

    load_checkpoint(model, RUN / "full_extra_step1000.pt", device)
    model.eval()

    def extra_fn(batch):
        log(f"evaluate extra batch_size={batch}")
        return collect_extra(sep, zoom, model, gaussian, weight, make_loader(test_set, batch), device)

    extra, extra_batch = run_eval(extra_fn)
    if extra["paths"] != base["paths"]:
        raise PipelineError("extra evaluation image order differs from global_base")

    maps = dict(base["maps"])
    maps["local_adapted"] = adapted["local_adapted"]
    maps["fuse_adapted"] = adapted["fuse_adapted"]
    maps["global_extra"] = extra["global_extra"]
    maps["local_extra"] = extra["local_extra"]
    maps["fuse_extra"] = extra["fuse_extra"]
    types = np.asarray([path_to_type[path] for path in base["paths"]])
    if set(types.tolist()) != {"good", *DEFECT_TYPES}:
        raise PipelineError(f"unexpected defect types {sorted(set(types.tolist()))}")

    methods = {}
    for name in METHOD_NAMES:
        methods[name] = pixel_block(maps[name], base["masks"], regions, zoom)
        row = methods[name]
        log(
            f"{name}: P-AP={row['pixel_ap']:.6f} P-AUROC={row['pixel_auroc']:.6f} "
            f"P-F1max={row['pixel_f1_max']:.6f} R@FPR={row['defect_recall_at_normal_fpr']} "
            f"regions={row['region_micro_mean']}"
        )
    splits, areas = type_and_area_splits(maps, base["masks"], base["labels"], types, zoom)
    cases = select_cases(base["paths"], types, base["labels"], areas, maps, regions)
    verdict, verdict_checks = build_verdict(methods)
    image_auroc = float(roc_auc_score(base["labels"], base["image_scores"]))
    weight_after = fingerprint(WEIGHTS)
    if weight_before != weight_after:
        raise PipelineError("the 5000-step checkpoint file changed during the run")
    report = {
        "development_stage": True,
        "development_stage_note": (
            "The official bottle test set has been inspected in earlier runs. "
            "Scores below are development-stage results, not a locked confirmation."
        ),
        "no_independent_normal_validation_set": True,
        "no_independent_normal_validation_set_note": (
            "The 5000-step model has already seen all 209 normal training images. "
            "This run uses all 209 again and does not hold any of them out. "
            "There is no independent normal validation set, and the checkpoint is not chosen on validation or test."
        ),
        "class_name": CLASS_NAME,
        "hypothesis": "匹配局部视野的正常重建训练，能降低局部误报，并保留局部与全局的互补。",
        "device": "cuda:0",
        "seed": SEED,
        "weights": str(WEIGHTS),
        "weights_unchanged": weight_before,
        "checkpoint_keys_loaded": ["bottleneck", "decoder"],
        "base_checkpoint_step": int(base_state.get("step", 5000)),
        "saved_checkpoints": {
            "local_adapt": local_train["checkpoint"],
            "full_extra": extra_train["checkpoint"],
            "saved_steps": [STEPS],
            "best_checkpoint_selection": False,
        },
        "hyperparameters_fixed_before_test": {
            "steps": STEPS,
            "lr": LR,
            "p": P_HARD,
            "factor": FACTOR,
            "fusion": "0.5*global+0.5*local",
            "window": window_info,
            "batch_candidates": list(BATCH_CANDIDATES),
            "grad_clip_max_norm": 0.1,
        },
        "train_good": len(train_paths),
        "test_images": int(len(test_set)),
        "test_anomaly": int((base["labels"] == 1).sum()),
        "test_good": int((base["labels"] == 0).sum()),
        "training": {
            "local_adapt": local_train,
            "full_extra": extra_train,
            "batch_size": batch_size,
            "oom_restarted_both_from_the_5000_step_copy": oom_restarted,
        },
        "evaluation_batch_size": {
            "base": eval_batch,
            "adapted": adapted_batch,
            "extra": extra_batch,
        },
        "forwards": {
            "base_full": int(base["full_forwards"]),
            "base_windows": int(base["window_forwards"]),
            "adapted_full": 0,
            "adapted_windows": int(adapted["window_forwards"]),
            "extra_full": int(extra["full_forwards"]),
            "extra_windows": int(extra["window_forwards"]),
            "fuse_adapted_global_source": "original 5000-step checkpoint, not the adapted weights",
        },
        "window": window_info,
        "postprocess": {
            "distance": "two-group mean cosine distance",
            "to_392": "bilinear align_corners=True",
            "to_256": "bilinear align_corners=False",
            "gaussian": "5x5 sigma=4",
            "fusion_before_256": "0.5*global_392+0.5*local_392",
            "local_rgb": "denormalize to RGB, crop 196, bilinear zoom to 392 align_corners=False, then ImageNet normalize",
            "max_abs_diff_average_first_vs_official_global": base["max_abs_global_diff"],
            "fusion_weight_sum_min": base["weight_sum_min"],
        },
        "reference_global_base_pixel_ap": REFERENCE_PIXEL_AP,
        "global_base_pixel_ap": global_ap,
        "global_base_pixel_ap_abs_diff_vs_0.8842199527833942": abs(global_ap - REFERENCE_PIXEL_AP),
        "global_base_pixel_ap_in_0.86_0.91": True,
        "image_auroc_global_base_top1pct_mean": image_auroc,
        "image_auroc_note": "Computed once from global_base. Score is the mean of the highest 1% pixels on the final 256 gaussian map.",
        "region_definition": region_definition,
        "methods": methods,
        "splits": splits,
        "fuse_adapted_minus_global_base_pixel_ap": deltas(splits),
        "cases": cases,
        "verdict_checks": verdict_checks,
        "verdict": verdict,
        "gpu_memory_used_before_load_mib": int(memory_before),
        "peak_allocated_mib": round(float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)), 1),
        "torch": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(device),
    }
    (RUN / "report.json").write_text(json.dumps(py(report), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {RUN / 'report.json'}")
    log(verdict)


def region_definition_empty(regions):
    return sum(1 for region in regions if not region["interior"].any())


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
