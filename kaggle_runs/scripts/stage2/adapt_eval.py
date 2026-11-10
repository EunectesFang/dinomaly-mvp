"""Dinomaly local-view adaptation on the 15-class MVTec-AD official protocol.

Starts from the 10k-iteration unified baseline produced by stage 1, then:
  * local_adapt: 1000 steps on random 196->392 normal crops (9-window grid)
  * full_extra:  1000 steps on full 392 normal images (compute-matched control)

Evaluates global / 9-window local / 0.5-0.5 fusion maps on all 15 classes with
the paper's 7 metrics, recall at fixed pixel FPR, and defect-region means.
"""
import os, sys, json, time, argparse, math
import numpy as np

OUT = "/kaggle/working"


def locate(marker, preferred):
    for c in preferred:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, marker)):
            return c
    for root, dirs, files in os.walk("/kaggle/input"):
        if marker in dirs or marker in files:
            return root
    raise RuntimeError(f"cannot locate {marker}")


CODE = locate("dinomaly_mvtec_uni.py", [
    "/kaggle/input/dinomaly-code",
    "/kaggle/input/datasets/eunectesqaq/dinomaly-code",
])
DATA = locate("bottle", [
    "/kaggle/input/mvtec-ad",
    "/kaggle/input/datasets/ipythonx/mvtec-ad",
])
def find_base_ckpt():
    """Prefer model_10000.pt, else the largest-iteration checkpoint available."""
    found = []
    roots = ["/kaggle/input/dinomaly-base-10k", "/kaggle/input/dinomaly-base",
             "/kaggle/input/datasets/eunectesqaq/dinomaly-base-10k",
             "/kaggle/input/datasets/eunectesqaq/dinomaly-base"]
    for r in roots:
        if os.path.isdir(r):
            for f in os.listdir(r):
                if f.startswith("model_") and f.endswith(".pt"):
                    found.append(os.path.join(r, f))
    if not found:
        for root, _dirs, files in os.walk("/kaggle/input"):
            for f in files:
                if f.startswith("model_") and f.endswith(".pt"):
                    found.append(os.path.join(root, f))
    if not found:
        raise RuntimeError("no model_*.pt checkpoint found under /kaggle/input")
    def it_of(p):
        try:
            return int(os.path.basename(p).split("_")[1].split(".")[0])
        except Exception:
            return -1
    found.sort(key=it_of)
    return found[-1]


BASE_CKPT = find_base_ckpt()

sys.path.insert(0, CODE)
os.chdir(CODE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from functools import partial
from scipy.signal.windows import hann
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import measure
from sklearn.metrics import roc_auc_score, average_precision_score, auc

from dataset import get_data_transforms, MVTecDataset
from models.uad import ViTill
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from optimizers import StableAdamW
from utils import cal_anomaly_maps, get_gaussian_kernel, f1_score_max, global_cosine_hm_percent

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CROP = 392
WINDOW = 196
STARTS = (0, 98, 196)
STEPS = 1000
BATCH = 6
LR = 2e-4
SEED = 1
EVAL_BATCH = 8
VARIANTS = ("global", "local", "fuse")
REGION_NAMES = ("interior", "inner_edge", "outer_neighborhood", "far_normal")
LOG_PATH = os.path.join(OUT, "stage2.log")


def log(msg):
    os.makedirs(OUT, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def setup_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(device):
    encoder = vit_encoder.load("dinov2reg_vit_base_14")
    bottleneck = nn.ModuleList([bMlp(768, 768 * 4, 768, drop=0.2)])
    decoder = nn.ModuleList([
        VitBlock(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=LinearAttention2)
        for _ in range(8)])
    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                   target_layers=[2, 3, 4, 5, 6, 7, 8, 9], mask_neighbor_size=0,
                   fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
                   fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]])
    return model.to(device)


def load_ckpt(model, path, device):
    st = torch.load(path, map_location=device, weights_only=False)
    model.bottleneck.load_state_dict(st["bottleneck"])
    model.decoder.load_state_dict(st["decoder"])
    return st


def load_train_arrays(item_list):
    prep = transforms.Compose([transforms.Resize((448, 448)),
                               transforms.ToTensor(),
                               transforms.CenterCrop(CROP)])
    arrays = []
    for item in item_list:
        folder = os.path.join(DATA, item, "train", "good")
        paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                       if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
        for p in paths:
            img = Image.open(p).convert("RGB")
            arrays.append((prep(img) * 255).round().clamp(0, 255).to(torch.uint8))
        log(f"train {item}: {len(paths)}")
    rgb = torch.stack(arrays, dim=0).permute(0, 2, 3, 1).contiguous()  # N,H,W,3 uint8
    log(f"train pool total={len(rgb)} shape={tuple(rgb.shape)}")
    return rgb


def norm_tensors(device):
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return mean, std


def normalize(x, mean, std):
    return (x - mean) / std


def local_batch_from_uint8(rgb_u8, choice, device, mean, std):
    x = rgb_u8.float().div(255).permute(0, 3, 1, 2)
    patches = []
    for i in range(x.shape[0]):
        y = STARTS[int(choice[i]) // 3]
        x0 = STARTS[int(choice[i]) % 3]
        patches.append(x[i:i + 1, :, y:y + WINDOW, x0:x0 + WINDOW])
    p = torch.cat(patches, dim=0).to(device)
    z = F.interpolate(p, size=CROP, mode="bilinear", align_corners=False)
    return normalize(z, mean, std)


def full_batch_from_uint8(rgb_u8, device, mean, std):
    x = rgb_u8.float().div(255).permute(0, 3, 1, 2).to(device)
    return normalize(x, mean, std)


def train_kind(model, rgb, kind, device, steps=STEPS, batch=BATCH, lr=LR, seed=SEED):
    setup_seed(seed)
    mean, std = norm_tensors(device)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.bottleneck.parameters():
        p.requires_grad_(True)
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    model.train()
    trainable = nn.ModuleList([model.bottleneck, model.decoder])
    optimizer = StableAdamW([{"params": trainable.parameters()}], lr=lr,
                            betas=(0.9, 0.999), weight_decay=1e-4,
                            amsgrad=True, eps=1e-8)
    gen = torch.Generator().manual_seed(seed)
    n = len(rgb)
    losses = []
    step = 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    peak = 0.0
    while step < steps:
        order = torch.randperm(n, generator=gen)
        for b in range(0, n - batch + 1, batch):
            idx = order[b:b + batch]
            chunk = rgb.index_select(0, idx)
            if kind == "local_adapt":
                choice = torch.randint(0, 9, (int(batch),), generator=gen)
                model_in = local_batch_from_uint8(chunk, choice, device, mean, std)
            elif kind == "full_extra":
                model_in = full_batch_from_uint8(chunk, device, mean, std)
            else:
                raise ValueError(kind)
            en, de = model(model_in)
            loss = global_cosine_hm_percent(en, de, p=0.9, factor=0.1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            losses.append(float(loss.detach()))
            step += 1
            peak = max(peak, torch.cuda.max_memory_allocated() / 1024 ** 2)
            if step == 1 or step % 100 == 0:
                log(f"{kind} {step}/{steps} loss={losses[-1]:.6f} "
                    f"peak_mib={peak:.0f} elapsed_min={(time.time()-started)/60:.1f}")
            if step >= steps:
                break
    model.eval()
    ckpt = os.path.join(OUT, f"{kind}.pt")
    torch.save({"bottleneck": model.bottleneck.state_dict(),
                "decoder": model.decoder.state_dict(),
                "kind": kind, "steps": steps, "batch": batch, "lr": lr,
                "base": BASE_CKPT}, ckpt)
    return {"kind": kind, "steps": steps, "batch": batch, "lr": lr, "seed": seed,
            "loss_first100": float(np.mean(losses[:100])),
            "loss_last100": float(np.mean(losses[-100:])),
            "minutes": (time.time() - started) / 60, "peak_mib": peak,
            "checkpoint": ckpt}


def global_map(model, images):
    en, de = model(images)
    amap, _ = cal_anomaly_maps(en, de, images.shape[-1])
    return amap


def local_map(model, images, weight, mean, std):
    rgb = images * std + mean
    b = images.shape[0]
    acc = torch.zeros(b, 1, CROP, CROP, device=images.device)
    wsum = torch.zeros(b, 1, CROP, CROP, device=images.device)
    for y in STARTS:
        for x0 in STARTS:
            patch = rgb[:, :, y:y + WINDOW, x0:x0 + WINDOW]
            z = F.interpolate(patch, size=CROP, mode="bilinear", align_corners=False)
            en, de = model(normalize(z, mean, std))
            dist = torch.stack([1 - F.cosine_similarity(e, d) for e, d in zip(en, de)], 0).mean(0).unsqueeze(1)
            pasted = F.interpolate(dist, size=WINDOW, mode="bilinear", align_corners=True)
            acc[:, :, y:y + WINDOW, x0:x0 + WINDOW] += pasted * weight
            wsum[:, :, y:y + WINDOW, x0:x0 + WINDOW] += weight
    return acc / wsum


def postprocess(amap, gaussian):
    return gaussian(F.interpolate(amap, size=256, mode="bilinear", align_corners=False))


def top1pct(amap):
    flat = amap.flatten(1)
    k = max(1, int(flat.shape[1] * 0.01))
    return torch.sort(flat, dim=1, descending=True).values[:, :k].mean(dim=1)


def label_regions(gt_np):
    return [[r.coords for r in measure.regionprops(measure.label(m))] for m in gt_np]


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
                cur.append(bmap[coords[:, 0], coords[:, 1]].sum() / len(coords))
        pros.append(float(np.mean(cur)) if cur else 0.0)
        fprs.append(float(np.logical_and(inverse, binmap).sum()) / neg_total)
    fprs = np.asarray(fprs)
    pros = np.asarray(pros)
    keep = fprs < 0.3
    if keep.sum() < 2 or fprs[keep].max() <= 0:
        return 0.0
    return float(auc(fprs[keep] / fprs[keep].max(), pros[keep]))


def collect_and_metrics(model, item_list, test_data_list, gaussian, weight, mean, std, device, tag):
    model.eval()
    per_variant = {v: {} for v in VARIANTS}
    global_scores = {}
    struct = np.ones((9, 9), dtype=bool)
    with torch.no_grad():
        for item, test_data in zip(item_list, test_data_list):
            loader = DataLoader(test_data, batch_size=EVAL_BATCH, shuffle=False, num_workers=4)
            buckets = {v: [] for v in VARIANTS}
            score_buf = {v: [] for v in VARIANTS}
            gt_buf = []
            labels = []
            region_acc = {v: {n: [] for n in REGION_NAMES} for v in VARIANTS}
            started = time.time()
            for img, gt, label, _path in loader:
                img = img.to(device)
                g392 = global_map(model, img)
                l392 = local_map(model, img, weight, mean, std)
                f392 = 0.5 * g392 + 0.5 * l392
                raw = {"global": g392, "local": l392, "fuse": f392}
                done = {}
                for v in VARIANTS:
                    m = postprocess(raw[v], gaussian)
                    done[v] = m
                    buckets[v].append(m.cpu())
                    score_buf[v].extend(top1pct(m).cpu().numpy().tolist())
                gt256 = F.interpolate(gt.float(), size=256, mode="nearest").bool()
                if gt256.shape[1] > 1:
                    gt256 = gt256.max(dim=1, keepdim=True)[0]
                gt_buf.append(gt256.cpu())
                labels.extend(label.numpy().tolist())

                mask_np = gt256[:, 0].numpy()
                maps_np = {v: done[v][:, 0].cpu().numpy() for v in VARIANTS}
                for i in range(mask_np.shape[0]):
                    defect = mask_np[i]
                    if defect.sum() == 0:
                        continue
                    interior = binary_erosion(defect, structure=struct)
                    dilated = binary_dilation(defect, structure=struct)
                    masks = {"interior": interior,
                             "inner_edge": defect & ~interior,
                             "outer_neighborhood": dilated & ~defect,
                             "far_normal": ~dilated}
                    for v in VARIANTS:
                        for n in REGION_NAMES:
                            m = masks[n]
                            if m.sum() > 0:
                                region_acc[v][n].append(float(maps_np[v][i][m].mean()))

            gt_np = torch.cat(gt_buf, dim=0)[:, 0].numpy().astype(np.int64)
            regions = label_regions(gt_np)
            gt_flat = gt_np.ravel()
            neg = None
            for v in VARIANTS:
                pr = torch.cat(buckets[v], dim=0)[:, 0].numpy().astype(np.float32)
                pr_flat = pr.ravel()
                if neg is None:
                    neg = pr_flat[gt_flat == 0]
                pos = pr_flat[gt_flat == 1]
                sp = np.asarray(score_buf[v], dtype=np.float32)
                row = {
                    "I-AUROC": float(roc_auc_score(labels, sp)),
                    "I-AP": float(average_precision_score(labels, sp)),
                    "I-F1": float(f1_score_max(labels, sp)),
                    "P-AUROC": float(roc_auc_score(gt_flat, pr_flat)),
                    "P-AP": float(average_precision_score(gt_flat, pr_flat)),
                    "P-F1": float(f1_score_max(gt_flat, pr_flat)),
                    "P-AUPRO": compute_pro_pre(gt_np, pr, regions),
                }
                rec = {}
                for fpr in (0.001, 0.01, 0.05):
                    if len(neg) and len(pos):
                        thr = np.quantile(neg, 1 - fpr)
                        rec[f"R@FPR{fpr:g}"] = float((pos >= thr).mean())
                reg = {n: (float(np.mean(region_acc[v][n])) if region_acc[v][n] else float("nan"))
                       for n in REGION_NAMES}
                per_variant[v][item] = {"metrics": row, "recall": rec, "regions": reg}
            global_scores[item] = score_buf["global"]
            log(f"{tag} {item}: " + " ".join(
                f"{v} AP={per_variant[v][item]['metrics']['P-AP']:.4f} "
                f"F1={per_variant[v][item]['metrics']['P-F1']:.4f}" for v in VARIANTS)
                + f" ({time.time()-started:.1f}s)")

    means = {}
    for v in VARIANTS:
        m = {k: float(np.mean([per_variant[v][it]["metrics"][k] for it in item_list]))
             for k in per_variant[v][item_list[0]]["metrics"]}
        rec = {}
        for key in per_variant[v][item_list[0]]["recall"]:
            rec[key] = float(np.mean([per_variant[v][it]["recall"][key] for it in item_list]))
        reg = {}
        for n in REGION_NAMES:
            vals = [per_variant[v][it]["regions"][n] for it in item_list
                    if not math.isnan(per_variant[v][it]["regions"][n])]
            reg[n] = float(np.mean(vals)) if vals else float("nan")
        means[v] = {"metrics": m, "recall": rec, "regions": reg}
    return {"per_class": per_variant, "mean": means, "global_scores": global_scores}


def main():
    global EVAL_BATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--eval_batch", type=int, default=EVAL_BATCH)
    args = ap.parse_args()
    EVAL_BATCH = args.eval_batch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    log(f"device={device} torch={torch.__version__} base={BASE_CKPT}")
    setup_seed(SEED)

    item_list = sorted(d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d)))
    if len(item_list) != 15:
        raise RuntimeError(f"expected 15 classes, got {len(item_list)}")
    data_transform, gt_transform = get_data_transforms(448, CROP)
    test_data_list = [MVTecDataset(root=os.path.join(DATA, item), transform=data_transform,
                                   gt_transform=gt_transform, phase="test") for item in item_list]
    log(f"classes={len(item_list)} test images={sum(len(d) for d in test_data_list)}")

    rgb = load_train_arrays(item_list)
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    h = torch.from_numpy(hann(WINDOW, sym=True)).float()
    weight = (0.1 + 0.9 * torch.outer(h, h)).view(1, 1, WINDOW, WINDOW).to(device)
    mean, std = norm_tensors(device)

    report = {"config": {"steps": args.steps, "batch": args.batch, "lr": LR, "seed": SEED,
                         "window": WINDOW, "starts": list(STARTS), "crop": CROP,
                         "base_checkpoint": BASE_CKPT, "device": device,
                         "eval_batch": EVAL_BATCH},
              "train": {}, "results": {}, "deltas": {}, "identity": {}}

    model = build_model(device)

    log("=== eval base ===")
    load_ckpt(model, BASE_CKPT, device)
    base = collect_and_metrics(model, item_list, test_data_list, gaussian, weight, mean, std, device, "base")
    report["results"]["base"] = base

    log("=== train local_adapt ===")
    load_ckpt(model, BASE_CKPT, device)
    report["train"]["local_adapt"] = train_kind(model, rgb, "local_adapt", device,
                                                steps=args.steps, batch=args.batch)
    log("=== eval adapted ===")
    adapted = collect_and_metrics(model, item_list, test_data_list, gaussian, weight, mean, std, device, "adapted")
    report["results"]["adapted"] = adapted

    log("=== train full_extra ===")
    load_ckpt(model, BASE_CKPT, device)
    report["train"]["full_extra"] = train_kind(model, rgb, "full_extra", device,
                                               steps=args.steps, batch=args.batch)
    log("=== eval extra ===")
    extra = collect_and_metrics(model, item_list, test_data_list, gaussian, weight, mean, std, device, "extra")
    report["results"]["extra"] = extra

    # image-score identity: encoder + global path are frozen, scores must be bit-identical
    diffs = []
    for it in item_list:
        a = np.asarray(base["global_scores"][it], dtype=np.float64)
        b = np.asarray(adapted["global_scores"][it], dtype=np.float64)
        diffs.append(float(np.abs(a - b).max()))
    report["identity"]["global_score_max_abs_diff_base_vs_adapted"] = max(diffs)
    report["identity"]["per_class_max_abs_diff"] = dict(zip(item_list, diffs))

    combos = {
        "adapted_local - base_global": ("adapted", "local", "base", "global"),
        "adapted_local - base_local": ("adapted", "local", "base", "local"),
        "adapted_local - extra_local": ("adapted", "local", "extra", "local"),
        "adapted_fuse - base_global": ("adapted", "fuse", "base", "global"),
        "adapted_fuse - base_fuse": ("adapted", "fuse", "base", "fuse"),
        "adapted_fuse - extra_fuse": ("adapted", "fuse", "extra", "fuse"),
        "extra_local - base_local": ("extra", "local", "base", "local"),
        "extra_fuse - base_fuse": ("extra", "fuse", "base", "fuse"),
        "extra_global - base_global": ("extra", "global", "base", "global"),
    }
    for name, (ra, va, rb, vb) in combos.items():
        report["deltas"][name] = {
            k: report["results"][ra]["mean"][va]["metrics"][k]
               - report["results"][rb]["mean"][vb]["metrics"][k]
            for k in report["results"][ra]["mean"][va]["metrics"]}

    report["per_class_delta_adapted_local_minus_base_global"] = {
        it: {k: report["results"]["adapted"]["per_class"]["local"][it]["metrics"][k]
                - report["results"]["base"]["per_class"]["global"][it]["metrics"][k]
             for k in report["results"]["base"]["mean"]["global"]["metrics"]}
        for it in item_list}

    with open(os.path.join(OUT, "adapt_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log("DONE")


if __name__ == "__main__":
    main()
