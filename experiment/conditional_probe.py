"""Conditional-normality probe on raw patches.

Frozen before scores. Same 8x8 RGB patches, three references fit on train/good only:

- global: nearest patch anywhere
- position: nearest patch at the same cell
- aligned: same cell after one FFT shift against the train mean image
- phase: grid only, same phase of the train-estimated lattice

A branch is supported only if its pooled pixel AP beats global by at least
0.02 and its within-image top-1% far-false-positive fraction is lower.
Screw is a harm check, not part of the grid or transistor decision.
Absolute AP is not comparable to the archived 73.09% full-frame result.
"""
import json
import os
import sys
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score

DATA = r"e:\dinomaly-mvp\data\mvtec"
OUT = r"e:\dinomaly-mvp\kaggle_runs\stage16-local\conditional_probe.json"
SIZE = 256
PATCH = 8
GRID = SIZE // PATCH
FEAT = PATCH * PATCH * 3
SHIFT_LIMIT = 32
NEAR_PX = 14
TOP_FRACTION = 0.01
GAIN_MIN = 0.02
CLASSES = ("grid", "transistor", "screw")
EXPECTED_TEST = {"grid": 78, "transistor": 100, "screw": 160}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def list_images(folder):
    paths = []
    for ext in ("*.png", "*.JPG", "*.jpg", "*.bmp"):
        paths.extend(glob(os.path.join(folder, ext)))
    return sorted(paths)


def load_rgb(path):
    image = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def load_mask(path):
    image = Image.open(path).convert("L").resize((SIZE, SIZE), Image.NEAREST)
    return np.asarray(image, dtype=np.float32) > 0


def test_records(root):
    image_root = os.path.join(root, "test")
    mask_root = os.path.join(root, "ground_truth")
    records = []
    for defect in sorted(os.listdir(image_root)):
        folder = os.path.join(image_root, defect)
        if not os.path.isdir(folder):
            continue
        for path in list_images(folder):
            if defect == "good":
                mask = np.zeros((SIZE, SIZE), dtype=bool)
            else:
                name = os.path.splitext(os.path.basename(path))[0] + "_mask.png"
                mask_path = os.path.join(mask_root, defect, name)
                if not os.path.isfile(mask_path):
                    raise FileNotFoundError(mask_path)
                mask = load_mask(mask_path)
            records.append({"path": path, "defect": defect, "mask": mask,
                            "rgb": load_rgb(path)})
    return records


def patches_from_rgb(rgb):
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    unfolded = F.unfold(tensor, kernel_size=PATCH, stride=PATCH)
    feats = unfolded.squeeze(0).transpose(0, 1).reshape(GRID, GRID, FEAT)
    return feats


def l2_normalize(feats):
    flat = feats.reshape(-1, FEAT)
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return flat.reshape(feats.shape)


def estimate_periods(gray):
    centered = gray - gray.mean()
    spectrum = torch.fft.rfft2(centered)
    corr = torch.fft.irfft2(spectrum.abs() ** 2, s=gray.shape)
    cy = cx = SIZE // 2
    shifted = torch.fft.fftshift(corr)
    periods = {}
    for axis, line in (("y", shifted[:, cx]), ("x", shifted[cy, :])):
        segment = line.clone()
        segment[cy - 11:cy + 12] = -1e30
        window = torch.zeros_like(segment, dtype=torch.bool)
        window[cy - 80:cy + 81] = True
        window[cy - 11:cy + 12] = False
        segment = segment.masked_fill(~window, -1e30)
        index = int(segment.argmax())
        periods[axis] = abs(index - cy)
    return periods["y"], periods["x"]


def phase_key(periods):
    period_y, period_x = periods
    yy = torch.arange(GRID) * PATCH + PATCH // 2
    xx = torch.arange(GRID) * PATCH + PATCH // 2
    key_y = (yy % period_y) // PATCH
    key_x = (xx % period_x) // PATCH
    return key_y[:, None].expand(GRID, GRID), key_x[None, :].expand(GRID, GRID)


def shift_to_template(gray, template):
    left = gray - gray.mean()
    right = template - template.mean()
    cross = torch.fft.rfft2(left) * torch.fft.rfft2(right).conj()
    cross = cross / cross.abs().clamp_min(1e-6)
    corr = torch.fft.irfft2(cross, s=gray.shape)
    ys = list(range(0, SHIFT_LIMIT + 1)) + list(range(SIZE - SHIFT_LIMIT, SIZE))
    xs = ys
    best = None
    for y in ys:
        row = corr[y, xs]
        value, index = row.max(dim=0)
        if best is None or float(value) > best[0]:
            x = xs[int(index)]
            dy = y if y <= SHIFT_LIMIT else y - SIZE
            dx = x if x <= SHIFT_LIMIT else x - SIZE
            best = (float(value), dy, dx)
    return best[1], best[2]


def max_cosine_global(queries, memory, chunk=1024):
    out = torch.empty(queries.shape[0], device=queries.device)
    for start in range(0, queries.shape[0], chunk):
        block = queries[start:start + chunk]
        best = torch.full((block.shape[0],), -1.0, device=queries.device)
        for mem_start in range(0, memory.shape[0], 65536):
            mem = memory[mem_start:mem_start + 65536]
            score = block @ mem.T
            best = torch.maximum(best, score.max(dim=1).values)
        out[start:start + chunk] = best
    return out


def broadcast(cell_scores):
    return cell_scores.repeat_interleave(PATCH, dim=-2).repeat_interleave(PATCH, dim=-1)


def distance_to_defect(mask):
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    # A cheap Euclidean approximation is unnecessary: dilate by a square of
    # radius NEAR_PX and treat the ring as near. The split only needs a
    # frozen spatial cutoff, not a metric distance.
    kernel = 2 * NEAR_PX + 1
    dilated = F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=NEAR_PX)
    near = dilated[0, 0].numpy() > 0.5
    return near


def top_split(score, mask):
    flat = score.reshape(-1)
    k = max(1, int(round(flat.numel() * TOP_FRACTION)))
    threshold = torch.topk(flat, k).values.min()
    chosen = score >= threshold
    near = distance_to_defect(mask)
    gt = mask
    chosen_np = chosen.cpu().numpy()
    hit = int((chosen_np & gt).sum())
    near_fp = int((chosen_np & ~gt & near).sum())
    far_fp = int((chosen_np & ~gt & ~near).sum())
    total = int(chosen_np.sum())
    positive = int(gt.sum())
    return {
        "hit_fraction": hit / total if total else 0.0,
        "near_fp_fraction": near_fp / total if total else 0.0,
        "far_fp_fraction": far_fp / total if total else 0.0,
        "recall_in_top": hit / positive if positive else float("nan"),
    }


def image_ap(score, mask):
    if int(mask.sum()) == 0:
        return None
    return float(average_precision_score(mask.reshape(-1), score.reshape(-1).cpu().numpy()))


def pooled_ap(scores, masks):
    pred = np.concatenate([score.reshape(-1).cpu().numpy() for score in scores])
    gt = np.concatenate([mask.reshape(-1) for mask in masks])
    return float(average_precision_score(gt, pred))


def position_score(test_feats, memory):
    # test_feats [G,G,D], memory [N,G,G,D]
    cosine = (test_feats[None] * memory).sum(dim=-1)
    return 1.0 - cosine.max(dim=0).values


def phase_score(test_feats, memory, key_y, key_x):
    scores = torch.empty(GRID, GRID, device=test_feats.device)
    keys = key_y * (int(key_x.max()) + 1) + key_x
    flat_memory = memory.reshape(memory.shape[0], -1, FEAT)
    flat_test = test_feats.reshape(-1, FEAT)
    flat_scores = torch.empty(GRID * GRID, device=test_feats.device)
    for key in keys.unique():
        select = (keys == key).reshape(-1)
        bank = flat_memory[:, select, :].reshape(-1, FEAT)
        cosine = flat_test[select] @ bank.T
        flat_scores[select] = 1.0 - cosine.max(dim=1).values
    return flat_scores.reshape(GRID, GRID)


def aligned_score(test_feats, memory, dy, dx):
    sy = int(np.rint(-dy / PATCH))
    sx = int(np.rint(-dx / PATCH))
    scores = torch.empty(GRID, GRID, device=test_feats.device)
    fallback = 0
    for i in range(GRID):
        for j in range(GRID):
            ii, jj = i + sy, j + sx
            if not (0 <= ii < GRID and 0 <= jj < GRID):
                ii, jj = i, j
                fallback += 1
            cosine = test_feats[i, j] @ memory[:, ii, jj, :].T
            scores[i, j] = 1.0 - cosine.max()
    return scores, fallback / (GRID * GRID), (sy, sx)


def evaluate_class(name, root):
    train_paths = list_images(os.path.join(root, "train", "good"))
    records = test_records(root)
    if len(records) != EXPECTED_TEST[name]:
        raise RuntimeError(
            f"{name} test images={len(records)} expected {EXPECTED_TEST[name]}")
    train_rgb = [load_rgb(path) for path in train_paths]
    train_feats = l2_normalize(torch.stack(
        [patches_from_rgb(rgb) for rgb in train_rgb])).to(DEVICE)
    train_gray = torch.stack([
        torch.from_numpy(rgb).mean(dim=-1) for rgb in train_rgb]).to(DEVICE)
    template = train_gray.mean(dim=0)
    periods = estimate_periods(template.cpu())
    key_y, key_x = phase_key(periods)
    key_y, key_x = key_y.to(DEVICE), key_x.to(DEVICE)
    memory_flat = train_feats.reshape(-1, FEAT)

    variants = ("global", "position", "aligned", "phase") if name == "grid" else (
        "global", "position", "aligned")
    collected = {variant: [] for variant in variants}
    shifts = []
    per_defect = {variant: {} for variant in variants}
    splits = {variant: [] for variant in variants}
    masks = []

    for record in records:
        feats = l2_normalize(patches_from_rgb(record["rgb"])).to(DEVICE)
        gray = torch.from_numpy(record["rgb"]).mean(dim=-1).to(DEVICE)
        dy, dx = shift_to_template(gray, template)
        shifts.append((abs(dy), abs(dx)))
        cell = {
            "global": (1.0 - max_cosine_global(
                feats.reshape(-1, FEAT), memory_flat)).reshape(GRID, GRID),
            "position": position_score(feats, train_feats),
        }
        aligned, fallback, cell_shift = aligned_score(feats, train_feats, dy, dx)
        cell["aligned"] = aligned
        if name == "grid":
            cell["phase"] = phase_score(feats, train_feats, key_y, key_x)
        mask = record["mask"]
        masks.append(mask)
        for variant, cells in cell.items():
            score = broadcast(cells.detach().cpu())
            collected[variant].append(score)
            if record["defect"] != "good":
                value = image_ap(score, mask)
                per_defect[variant].setdefault(record["defect"], []).append(value)
                splits[variant].append(top_split(score, mask))
        record["fallback"] = fallback
        record["shift"] = (dy, dx, cell_shift)

    result = {
        "n_train": len(train_paths),
        "n_test": len(records),
        "period_y": int(periods[0]),
        "period_x": int(periods[1]),
        "mean_abs_shift_px": {
            "dy": float(np.mean([item[0] for item in shifts])),
            "dx": float(np.mean([item[1] for item in shifts])),
        },
        "variants": {},
    }
    global_ap = pooled_ap(collected["global"], masks)
    global_far = float(np.mean([item["far_fp_fraction"] for item in splits["global"]]))
    for variant in variants:
        ap = pooled_ap(collected[variant], masks)
        far = float(np.mean([item["far_fp_fraction"] for item in splits[variant]]))
        result["variants"][variant] = {
            "P-AP": ap,
            "delta_vs_global": ap - global_ap,
            "far_fp_fraction": far,
            "near_fp_fraction": float(np.mean(
                [item["near_fp_fraction"] for item in splits[variant]])),
            "hit_fraction": float(np.mean(
                [item["hit_fraction"] for item in splits[variant]])),
            "recall_in_top": float(np.nanmean(
                [item["recall_in_top"] for item in splits[variant]])),
            "per_defect_image_ap": {
                defect: float(np.mean(values))
                for defect, values in sorted(per_defect[variant].items())
            },
        }
    primary = {"grid": "phase", "transistor": "aligned", "screw": "aligned"}[name]
    branch = result["variants"][primary]
    supported = branch["delta_vs_global"] >= GAIN_MIN and branch["far_fp_fraction"] < global_far
    result["primary"] = primary
    harmed = (
        result["variants"]["position"]["delta_vs_global"] <= -0.05
        or result["variants"]["aligned"]["delta_vs_global"] <= -0.05
    )
    result["supported"] = bool(supported)
    result["position_harms"] = bool(harmed)
    result["global_far_fp_fraction"] = global_far
    return result


def main():
    chosen = [item for item in sys.argv[1:] if item in EXPECTED_TEST]
    names = chosen or CLASSES
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    report = {
        "device": DEVICE,
        "size": SIZE,
        "patch": PATCH,
        "fit": "train/good only, files present on disk",
        "gain_min": GAIN_MIN,
        "near_px": NEAR_PX,
        "classes": {},
    }
    if os.path.isfile(OUT):
        with open(OUT, encoding="utf-8") as handle:
            previous = json.load(handle)
        report["classes"].update(previous.get("classes", {}))
    for name in names:
        root = os.path.join(DATA, name)
        if not os.path.isdir(os.path.join(root, "train", "good")):
            report["classes"][name] = {"missing": True}
            print(f"{name} missing", flush=True)
            continue
        print(f"running {name}", flush=True)
        report["classes"][name] = evaluate_class(name, root)
        item = report["classes"][name]
        print(
            f"{name} supported={item['supported']} harms={item['position_harms']} "
            + " ".join(
                f"{variant}={row['P-AP']:.4f}({row['delta_vs_global']:+.4f})"
                for variant, row in item["variants"].items()),
            flush=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
