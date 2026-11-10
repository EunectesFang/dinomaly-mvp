"""Single-class bottle run: official Dinomaly separation recipe, then two localization heads.

The official test set is read only after synthetic-validation thresholds are fixed.
Batch size starts at 6. If the first reconstruction step runs out of memory, the
run reseeds and restarts at batch size 4.
"""

import gc
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF

REPO = Path(r"E:\dinomaly-mvp\Dinomaly")
DATA = Path(r"E:\dinomaly-mvp\data\mvtec_anomaly_detection")
RUN = Path(r"E:\dinomaly-mvp\experiment\runs\bottle_sep_b6")
CLASS_NAME = "bottle"
SEED = 1
TOTAL_ITERS = 5000
LOC_STEPS = 1000
BATCH_CANDIDATES = (6, 4)
VAL_FRACTION = 0.2
P_CLEAN = 0.5
IMAGE_SIZE = 448
CROP_SIZE = 392
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

os.chdir(REPO)
sys.path.insert(0, str(REPO))

from dinov1.utils import trunc_normal_  # noqa: E402
from models import vit_encoder  # noqa: E402
from models.uad import ViTill  # noqa: E402
from models.vision_transformer import Block as VitBlock  # noqa: E402
from models.vision_transformer import LinearAttention2, bMlp  # noqa: E402

_stable_spec = importlib.util.spec_from_file_location(
    "stable_adamw", REPO / "optimizers" / "StableAdamW.py"
)
_stable_mod = importlib.util.module_from_spec(_stable_spec)
_stable_spec.loader.exec_module(_stable_mod)
StableAdamW = _stable_mod.StableAdamW


class RestartBatch(Exception):
    def __init__(self, batch_size):
        self.batch_size = batch_size


class WarmCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for _ in self.base_lrs]
        return [self.schedule[self.last_epoch] for _ in self.base_lrs]


def log(message):
    print(message, flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "train.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def modify_grad(grad, inds, factor):
    inds = inds.expand_as(grad)
    grad = grad.clone()
    grad[inds] *= factor
    return grad


def global_cosine_hm_percent(en, de, p, factor):
    cosine = nn.CosineSimilarity()
    loss = 0.0
    for encoder_feat, decoder_feat in zip(en, de):
        encoder_det = encoder_feat.detach()
        with torch.no_grad():
            point_dist = 1 - cosine(encoder_det, decoder_feat).unsqueeze(1)
            k = max(1, int(point_dist.numel() * (1 - p)))
            thresh = torch.topk(point_dist.reshape(-1), k=k).values[-1]
            easy = point_dist < thresh
        loss = loss + torch.mean(
            1
            - cosine(
                encoder_det.reshape(encoder_det.shape[0], -1),
                decoder_feat.reshape(decoder_feat.shape[0], -1),
            )
        )
        decoder_feat.register_hook(partial(modify_grad, inds=easy, factor=factor))
    return loss / len(en)


def get_gaussian_kernel(kernel_size=5, sigma=4, channels=1):
    coords = torch.arange(kernel_size)
    grid_x = coords.repeat(kernel_size).view(kernel_size, kernel_size)
    grid_y = grid_x.t()
    xy_grid = torch.stack([grid_x, grid_y], dim=-1).float()
    mean = (kernel_size - 1) / 2.0
    variance = sigma ** 2
    kernel = (1.0 / (2.0 * np.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2, dim=-1) / (2 * variance)
    )
    kernel = kernel / torch.sum(kernel)
    weight = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels, bias=False)
    conv.weight.data = weight
    conv.weight.requires_grad = False
    return conv


def build_model(device):
    encoder = vit_encoder.load("dinov2reg_vit_base_14")
    embed_dim, num_heads = 768, 12
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse = [[0, 1, 2, 3], [4, 5, 6, 7]]
    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
    decoder = nn.ModuleList(
        [
            VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn_drop=0.0,
                attn=LinearAttention2,
            )
            for _ in range(8)
        ]
    )
    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=fuse,
        fuse_layer_decoder=fuse,
    )
    for module in nn.ModuleList([bottleneck, decoder]).modules():
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    return model.to(device)


class LocBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.ModuleList([nn.Conv2d(768, 32, kernel_size=1) for _ in range(2)])
        self.conv = nn.Conv2d(66, 64, kernel_size=3, padding=1)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, residuals, distances):
        reduced = [layer(residual) for layer, residual in zip(self.reduce, residuals)]
        stacked = torch.cat([distances[0], distances[1], reduced[0], reduced[1]], dim=1)
        return self.out(F.gelu(self.conv(stacked)))


def residual_and_distance(en, de):
    residuals, distances = [], []
    for encoder_feat, decoder_feat in zip(en, de):
        encoder_n = F.normalize(encoder_feat, dim=1)
        decoder_n = F.normalize(decoder_feat, dim=1)
        residuals.append(encoder_n - decoder_n)
        distances.append(1 - (encoder_n * decoder_n).sum(dim=1, keepdim=True))
    return residuals, distances


def distance_only_residuals(distances):
    return [distance.expand(-1, 768, -1, -1).contiguous() for distance in distances]


def upsample_to_256(anomaly_map, already_392=False):
    if not already_392:
        anomaly_map = F.interpolate(anomaly_map, size=CROP_SIZE, mode="bilinear", align_corners=True)
    return F.interpolate(anomaly_map, size=256, mode="bilinear", align_corners=False)


def cal_anomaly_maps(en, de, out_size):
    group_maps = []
    for encoder_feat, decoder_feat in zip(en, de):
        anomaly = (1 - F.cosine_similarity(encoder_feat, decoder_feat)).unsqueeze(1)
        group_maps.append(F.interpolate(anomaly, size=out_size, mode="bilinear", align_corners=True))
    return torch.cat(group_maps, dim=1).mean(dim=1, keepdim=True), group_maps


def image_score(anomaly_map):
    flat = anomaly_map.flatten(1)
    count = int(flat.shape[1] * 0.01)
    return torch.sort(flat, dim=1, descending=True).values[:, :count].mean(dim=1)


class TensorSet(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        return self.tensors[index], 0


def load_crop(path):
    image = Image.open(path).convert("RGB")
    image = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(image)
    image = transforms.CenterCrop(CROP_SIZE)(image)
    return np.asarray(image).astype(np.float32) / 255.0


def to_model_tensor(array):
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    return TF.normalize(tensor, MEAN, STD)


class PasteSet(Dataset):
    def __init__(self, items, mode):
        self.items = items
        self.mode = mode
        self.by_class = list(range(len(items)))

    def __len__(self):
        return len(self.items)

    def _paste(self, index, image, rng):
        donor_index = index
        while donor_index == index:
            donor_index = rng.choice(self.by_class)
        donor = self.items[donor_index][1]
        side = rng.randint(28, 140)
        y = rng.randint(0, CROP_SIZE - side)
        x = rng.randint(0, CROP_SIZE - side)
        alpha = rng.uniform(0.6, 1.0)
        image = image.copy()
        dst = image[y : y + side, x : x + side]
        src = donor[y : y + side, x : x + side]
        image[y : y + side, x : x + side] = (1 - alpha) * dst + alpha * src
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        mask[y : y + side, x : x + side] = 1.0
        return image, mask

    def __getitem__(self, index):
        image = self.items[index][1]
        if self.mode == "clean":
            mask = np.zeros(image.shape[:2], dtype=np.float32)
            image = image.copy()
        elif self.mode == "paste":
            image, mask = self._paste(index, image, random.Random(SEED + 10007 + index))
        elif random.random() < P_CLEAN:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
            image = image.copy()
        else:
            image, mask = self._paste(index, image, random)
        return to_model_tensor(image), torch.from_numpy(mask)[None]


def split_train_images():
    folder = DATA / CLASS_NAME / "train"
    dataset = ImageFolder(folder, transform=transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.CenterCrop(CROP_SIZE),
        transforms.Normalize(MEAN, STD),
    ]))
    crops = []
    for path, _ in dataset.samples:
        crops.append((path, load_crop(path)))
    rng = random.Random(SEED)
    order = list(range(len(crops)))
    rng.shuffle(order)
    n_val = max(1, int(round(VAL_FRACTION * len(order))))
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    return dataset, [crops[i] for i in train_idx], [crops[i] for i in val_idx]


def train_dinomaly(batch_size, device, train_tensors):
    setup_seed(SEED)
    model = build_model(device)
    trainable = nn.ModuleList([model.bottleneck, model.decoder])
    optimizer = StableAdamW(
        [{"params": trainable.parameters()}],
        lr=2e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=True,
        eps=1e-8,
    )
    scheduler = WarmCosineScheduler(
        optimizer, base_value=2e-3, final_value=2e-4, total_iters=TOTAL_ITERS, warmup_iters=100
    )
    loader = DataLoader(
        TensorSet(train_tensors),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    losses = []
    started = time.time()
    step = 0
    model.train()
    while step < TOTAL_ITERS:
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            try:
                en, de = model(images)
                p = min(0.9 * step / 1000.0, 0.9)
                loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                scheduler.step()
            except RuntimeError as exc:
                if step == 0 and "out of memory" in str(exc).lower():
                    raise RestartBatch(4) from exc
                raise
            losses.append(float(loss.detach()))
            step += 1
            if step % 100 == 0 or step == 1 or step == TOTAL_ITERS:
                peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
                elapsed = time.time() - started
                log(
                    f"recon {step}/{TOTAL_ITERS} loss={losses[-1]:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6f} peak_mib={peak:.0f} "
                    f"elapsed_min={elapsed / 60:.1f}"
                )
            if step == TOTAL_ITERS:
                break
    torch.save(
        {"bottleneck": model.bottleneck.state_dict(), "decoder": model.decoder.state_dict(), "step": step},
        RUN / "dinomaly_bottle.pt",
    )
    return model, {
        "seconds": time.time() - started,
        "last100_mean": float(np.mean(losses[-100:])),
        "first100_mean": float(np.mean(losses[:100])),
        "losses_every_100": [round(float(np.mean(losses[i : i + 100])), 5) for i in range(0, len(losses), 100)],
    }


def train_loc(model, head, items, batch_size, device, name):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        PasteSet(items, mode="mixed"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    losses = []
    started = time.time()
    step = 0
    while step < LOC_STEPS:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.no_grad():
                en, de = model(images)
                residuals, distances = residual_and_distance(en, de)
                if name == "distance":
                    residuals = distance_only_residuals(distances)
            logits = head(residuals, distances)
            mask_28 = F.adaptive_avg_pool2d(masks, 28)
            loss = F.binary_cross_entropy_with_logits(logits, mask_28)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            step += 1
            if step % 100 == 0 or step == 1 or step == LOC_STEPS:
                log(f"loc-{name} {step}/{LOC_STEPS} loss={losses[-1]:.4f}")
            if step == LOC_STEPS:
                break
    return {
        "seconds": time.time() - started,
        "first100_mean": float(np.mean(losses[:100])),
        "last100_mean": float(np.mean(losses[-100:])),
    }


def compute_pro(masks, amaps, num_th=200):
    masks = (masks > 0.5).astype(np.uint8)
    structure = np.ones((3, 3), dtype=int)
    regions = []
    for mask in masks:
        labeled, count = ndimage.label(mask, structure=structure)
        regions.append([np.argwhere(labeled == index) for index in range(1, count + 1)])
    inverse = 1 - masks
    inverse_sum = float(inverse.sum())
    pros, fprs = [], []
    min_th, max_th = float(amaps.min()), float(amaps.max())
    if max_th <= min_th or inverse_sum == 0:
        return float("nan")
    delta = (max_th - min_th) / num_th
    for threshold in np.arange(min_th, max_th, delta):
        binary = amaps > threshold
        overlaps = []
        for binary_map, regs in zip(binary, regions):
            for coords in regs:
                hits = binary_map[coords[:, 0], coords[:, 1]].sum()
                overlaps.append(hits / len(coords))
        if not overlaps:
            continue
        fpr = np.logical_and(inverse, binary).sum() / inverse_sum
        pros.append(float(np.mean(overlaps)))
        fprs.append(float(fpr))
    fprs = np.asarray(fprs)
    pros = np.asarray(pros)
    keep = fprs < 0.3
    if keep.sum() < 2 or fprs[keep].max() <= 0:
        return float("nan")
    fprs = fprs[keep] / fprs[keep].max()
    pros = pros[keep]
    order = np.argsort(fprs, kind="mergesort")
    return float(auc(fprs[order], pros[order]))


def f1_max(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-7)
    return float(f1_scores[:-1].max())


def best_f1_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-7)
    f1_scores = f1_scores[:-1]
    valid = ~np.isnan(f1_scores)
    f1_scores = f1_scores[valid]
    thresholds = thresholds[valid]
    return float(thresholds[int(np.argmax(f1_scores))])


def f1_at(y_true, y_score, threshold):
    prediction = y_score >= threshold
    true = y_true.astype(bool)
    tp = np.logical_and(prediction, true).sum()
    fp = np.logical_and(prediction, ~true).sum()
    fn = np.logical_and(~prediction, true).sum()
    return float(2 * tp / (2 * tp + fp + fn + 1e-7))


@torch.no_grad()
def collect_maps(model, heads, loader, device, gaussian, synthetic=False):
    model.eval()
    for head in heads.values():
        head.eval()
    buckets = {name: [] for name in ("official_gaussian", "no_gaussian", "low_group_no_smooth", "distance_branch", "full_branch")}
    image_scores, labels, masks = [], [], []
    for batch in loader:
        images, target = batch[0].to(device), batch[1].to(device)
        en, de = model(images)
        official_392, group_maps = cal_anomaly_maps(en, de, images.shape[-1])
        official_256 = upsample_to_256(official_392, already_392=True)
        smooth = gaussian(official_256)
        low_256 = upsample_to_256(group_maps[0], already_392=True)
        residuals, distances = residual_and_distance(en, de)
        full_256 = upsample_to_256(torch.sigmoid(heads["full"](residuals, distances)))
        distance_256 = upsample_to_256(
            torch.sigmoid(heads["distance"](distance_only_residuals(distances), distances))
        )
        buckets["official_gaussian"].append(smooth)
        buckets["no_gaussian"].append(official_256)
        buckets["low_group_no_smooth"].append(low_256)
        buckets["distance_branch"].append(distance_256)
        buckets["full_branch"].append(full_256)
        image_scores.append(image_score(smooth))
        if synthetic:
            mask_256 = F.interpolate(target, size=256, mode="nearest")
            masks.append((mask_256 > 0.5).float())
        else:
            mask_256 = F.interpolate(target, size=256, mode="nearest")
            labels.append(batch[2])
            masks.append(mask_256.bool().float())
    maps = {name: torch.cat(items, dim=0)[:, 0].cpu().numpy() for name, items in buckets.items()}
    mask_np = torch.cat(masks, dim=0)[:, 0].cpu().numpy()
    scores = torch.cat(image_scores).cpu().numpy()
    label_np = None if synthetic else torch.cat(labels).flatten().cpu().numpy()
    return maps, mask_np, scores, label_np


def pixel_metrics(gt, prediction, image_gt, image_score_values, threshold):
    gt_flat = gt.ravel().astype(np.uint8)
    pred_flat = prediction.ravel()
    return {
        "pixel_auroc": float(roc_auc_score(gt_flat, pred_flat)),
        "pixel_ap": float(average_precision_score(gt_flat, pred_flat)),
        "pixel_f1_max": f1_max(gt_flat, pred_flat),
        "pixel_aupro": compute_pro(gt, prediction),
        "pixel_f1_at_val_threshold": f1_at(gt_flat, pred_flat, threshold),
        "val_threshold": threshold,
        "good_fp_frac_at_val_threshold": float((prediction[image_gt == 0] >= threshold).mean())
        if np.any(image_gt == 0)
        else None,
        "good_mean_score": float(prediction[image_gt == 0].mean()) if np.any(image_gt == 0) else None,
        "image_auroc": float(roc_auc_score(image_gt, image_score_values)),
        "image_ap": float(average_precision_score(image_gt, image_score_values)),
        "image_f1_max": f1_max(image_gt.astype(np.uint8), image_score_values),
    }


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    log("cache bottle train crops")
    official_set, loc_train, loc_val = split_train_images()
    train_tensors = [official_set[i][0] for i in range(len(official_set))]
    log(f"train_good={len(train_tensors)} loc_train={len(loc_train)} loc_val={len(loc_val)}")

    batch_size = BATCH_CANDIDATES[0]
    try:
        model, recon_report = train_dinomaly(batch_size, device, train_tensors)
    except RestartBatch as restart:
        log(f"batch {batch_size} OOM on first step, restart at {restart.batch_size}")
        gc.collect()
        torch.cuda.empty_cache()
        batch_size = restart.batch_size
        model, recon_report = train_dinomaly(batch_size, device, train_tensors)
    recon_report["batch_size"] = batch_size

    heads = {}
    loc_reports = {}
    for offset, name in ((1, "distance"), (2, "full")):
        setup_seed(SEED + offset)
        head = LocBranch().to(device)
        loc_reports[name] = train_loc(model, head, loc_train, batch_size, device, name)
        heads[name] = head
        torch.save(head.state_dict(), RUN / f"loc_{name}.pt")

    gaussian = get_gaussian_kernel().to(device)
    val_clean = DataLoader(PasteSet(loc_val, mode="clean"), batch_size=batch_size, shuffle=False)
    val_paste = DataLoader(PasteSet(loc_val, mode="paste"), batch_size=batch_size, shuffle=False)
    clean_maps, clean_masks, _, _ = collect_maps(model, heads, val_clean, device, gaussian, synthetic=True)
    paste_maps, paste_masks, _, _ = collect_maps(model, heads, val_paste, device, gaussian, synthetic=True)
    thresholds = {}
    for name in clean_maps:
        gt = np.concatenate([clean_masks, paste_masks], axis=0)
        prediction = np.concatenate([clean_maps[name], paste_maps[name]], axis=0)
        thresholds[name] = best_f1_threshold(gt.ravel().astype(np.uint8), prediction.ravel())
        log(f"val threshold {name}={thresholds[name]:.6f}")

    from dataset import MVTecDataset, get_data_transforms

    data_transform, gt_transform = get_data_transforms(IMAGE_SIZE, CROP_SIZE)
    test_set = MVTecDataset(
        root=str(DATA / CLASS_NAME),
        transform=data_transform,
        gt_transform=gt_transform,
        phase="test",
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_maps, test_masks, test_scores, test_labels = collect_maps(
        model, heads, test_loader, device, gaussian, synthetic=False
    )
    rows = {}
    for name, prediction in test_maps.items():
        rows[name] = pixel_metrics(test_masks, prediction, test_labels, test_scores, thresholds[name])
        row = rows[name]
        log(
            f"{name}: I-AUROC={row['image_auroc']:.4f} P-AUROC={row['pixel_auroc']:.4f} "
            f"P-AP={row['pixel_ap']:.4f} P-F1max={row['pixel_f1_max']:.4f} "
            f"P-AUPRO={row['pixel_aupro']:.4f} P-F1val={row['pixel_f1_at_val_threshold']:.4f} "
            f"good_fp={row['good_fp_frac_at_val_threshold']:.4f}"
        )
    image_aurocs = [row["image_auroc"] for row in rows.values()]
    report = {
        "class_name": CLASS_NAME,
        "protocol": "dinomaly_mvtec_sep.py single-class, 5000 iterations, cuda:0",
        "batch_size": batch_size,
        "batch_requested": BATCH_CANDIDATES[0],
        "recon": recon_report,
        "loc": loc_reports,
        "counts": {
            "train_good": len(train_tensors),
            "loc_train": len(loc_train),
            "loc_val": len(loc_val),
            "test": int(len(test_set)),
            "test_anomaly": int(test_labels.sum()),
            "test_good": int((test_labels == 0).sum()),
        },
        "thresholds_from_synthetic_val": thresholds,
        "rows": rows,
        "flags": {
            "full_pixel_ap_above_official": rows["full_branch"]["pixel_ap"] > rows["official_gaussian"]["pixel_ap"],
            "full_pixel_f1max_above_official": rows["full_branch"]["pixel_f1_max"] > rows["official_gaussian"]["pixel_f1_max"],
            "full_pixel_ap_above_no_gaussian": rows["full_branch"]["pixel_ap"] > rows["no_gaussian"]["pixel_ap"],
            "full_pixel_ap_above_distance": rows["full_branch"]["pixel_ap"] > rows["distance_branch"]["pixel_ap"],
            "image_auroc_same_path": max(image_aurocs) - min(image_aurocs) < 1e-6,
        },
        "peak_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1),
        "scope": "one class, batch may differ from the paper's 16, not the 15-class unified checkpoint",
    }
    (RUN / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"wrote {RUN / 'report.json'}")
    log(json.dumps(report["flags"]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        RUN.mkdir(parents=True, exist_ok=True)
        text = traceback.format_exc()
        (RUN / "error.txt").write_text(text, encoding="utf-8")
        log(text)
        raise
