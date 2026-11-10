"""Short screening run for the frozen-Dinomaly localization branch.

This script does not read test/ or ground_truth/. It only checks whether a
short reconstruction stage and a short localization stage fit in 8 GB, whether
their losses fall, and whether held-out normal images get painted as anomalous.
"""

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
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

REPO = Path(r"E:\dinomaly-mvp\Dinomaly")
DATA = Path(r"E:\dinomaly-mvp\data\mvtec_anomaly_detection")
RUN = Path(r"E:\dinomaly-mvp\experiment\runs\screen_seed1")

# Locked before any test score was inspected. Do not change.
CLASSES = ("wood", "bottle", "screw")
SEED = 1
BATCH_SIZE = 4
PRETRAIN_STEPS = 300
LOC_STEPS = 200
WARMUP_STEPS = 20
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


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def modify_grad(grad, inds, factor):
    inds = inds.expand_as(grad)
    grad = grad.clone()
    grad[inds] *= factor
    return grad


def global_cosine_hm_percent(en, de, p, factor):
    """Same reduction as utils.global_cosine_hm_percent in the official repo."""
    cosine = nn.CosineSimilarity()
    loss = 0.0
    for a, b in zip(en, de):
        a_det = a.detach()
        with torch.no_grad():
            point_dist = 1 - cosine(a_det, b).unsqueeze(1)
            k = max(1, int(point_dist.numel() * (1 - p)))
            thresh = torch.topk(point_dist.reshape(-1), k=k).values[-1]
            easy = point_dist < thresh
        loss = loss + torch.mean(
            1
            - cosine(
                a_det.reshape(a_det.shape[0], -1),
                b.reshape(b.shape[0], -1),
            )
        )
        b.register_hook(partial(modify_grad, inds=easy, factor=factor))
    return loss / len(en)


def build_model(device):
    encoder_name = "dinov2reg_vit_base_14"
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse = [[0, 1, 2, 3], [4, 5, 6, 7]]
    encoder = vit_encoder.load(encoder_name)
    embed_dim, num_heads = 768, 12
    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
    decoder = nn.ModuleList(
        [
            VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
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
    model = model.to(device)
    return model


class LocBranch(nn.Module):
    """1x1 reduce each residual 768->32, concat with two distance maps, then 3x3 and 1x1."""

    def __init__(self):
        super().__init__()
        self.reduce = nn.ModuleList([nn.Conv2d(768, 32, kernel_size=1) for _ in range(2)])
        self.conv = nn.Conv2d(66, 64, kernel_size=3, padding=1)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, residuals, distances):
        reduced = [layer(residual) for layer, residual in zip(self.reduce, residuals)]
        x = torch.cat([distances[0], distances[1], reduced[0], reduced[1]], dim=1)
        return self.out(F.gelu(self.conv(x)))


def residual_and_distance(en, de):
    residuals, distances = [], []
    for e, d in zip(en, de):
        e_n = F.normalize(e, dim=1)
        d_n = F.normalize(d, dim=1)
        residuals.append(e_n - d_n)
        distances.append(1 - (e_n * d_n).sum(dim=1, keepdim=True))
    return residuals, distances


def list_good(class_name):
    folder = DATA / class_name / "train" / "good"
    paths = []
    for pattern in ("*.png", "*.JPG", "*.jpg", "*.bmp"):
        paths.extend(folder.glob(pattern))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(folder)
    for path in paths:
        if "test" in path.parts or "ground_truth" in path.parts:
            raise RuntimeError(f"refusing a non-training path: {path}")
    return paths


def split_items():
    rng = random.Random(SEED)
    train_items, val_items, counts = [], [], {}
    for class_name in CLASSES:
        paths = list_good(class_name)
        order = paths[:]
        rng.shuffle(order)
        n_val = max(1, int(round(VAL_FRACTION * len(order))))
        val_paths, train_paths = order[:n_val], order[n_val:]
        train_items.extend((str(path), class_name) for path in train_paths)
        val_items.extend((str(path), class_name) for path in val_paths)
        counts[class_name] = {
            "train_good": len(train_paths),
            "val_good": len(val_paths),
            "total_good": len(paths),
        }
    return train_items, val_items, counts


_CROP_CACHE = {}


def load_crop(path):
    key = str(path)
    cached = _CROP_CACHE.get(key)
    if cached is None:
        image = Image.open(path).convert("RGB")
        image = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(image)
        image = transforms.CenterCrop(CROP_SIZE)(image)
        cached = np.asarray(image).astype(np.float32) / 255.0
        _CROP_CACHE[key] = cached
    return cached.copy()


def to_model_tensor(array):
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    return TF.normalize(tensor, MEAN, STD)


class CleanSet(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, _ = self.items[index]
        return to_model_tensor(load_crop(path))


class PasteSet(Dataset):
    """Same-class rectangle paste. Label is the replaced-pixel fraction, not blend strength.

    mode='mixed' uses the process RNG and keeps untouched images with probability P_CLEAN.
    mode='clean' never pastes. mode='paste' always pastes, with a per-index seed, so
    validation images stay comparable before and after training.
    """

    def __init__(self, items, mode="mixed"):
        self.items = items
        self.mode = mode
        self.by_class = {}
        for index, (_, class_name) in enumerate(items):
            self.by_class.setdefault(class_name, []).append(index)

    def __len__(self):
        return len(self.items)

    def _paste(self, index, image, rng):
        _, class_name = self.items[index]
        donors = self.by_class[class_name]
        donor_index = index
        if len(donors) > 1:
            while donor_index == index:
                donor_index = rng.choice(donors)
        donor = load_crop(self.items[donor_index][0])
        side = rng.randint(28, 140)
        y = rng.randint(0, CROP_SIZE - side)
        x = rng.randint(0, CROP_SIZE - side)
        alpha = rng.uniform(0.6, 1.0)
        dst = image[y : y + side, x : x + side]
        src = donor[y : y + side, x : x + side]
        image[y : y + side, x : x + side] = (1 - alpha) * dst + alpha * src
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        mask[y : y + side, x : x + side] = 1.0
        return image, mask

    def __getitem__(self, index):
        path, _ = self.items[index]
        image = load_crop(path)
        if self.mode == "clean":
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        elif self.mode == "paste":
            image, mask = self._paste(index, image, random.Random(SEED + 10007 + index))
        else:
            if random.random() < P_CLEAN:
                mask = np.zeros(image.shape[:2], dtype=np.float32)
            else:
                image, mask = self._paste(index, image, random)
        mask_28 = F.adaptive_avg_pool2d(torch.from_numpy(mask)[None, None], 28)[0]
        return to_model_tensor(image), mask_28


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def mean_of(values):
    if not values:
        return None
    return float(sum(values) / len(values))


def pretrain(model, loader, device):
    trainable = nn.ModuleList([model.bottleneck, model.decoder])
    optimizer = StableAdamW(
        [{"params": trainable.parameters()}],
        lr=2e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=True,
        eps=1e-10,
    )
    losses = []
    model.train()
    stream = cycle_loader(loader)
    started = time.time()
    for step in range(PRETRAIN_STEPS):
        if step < WARMUP_STEPS:
            lr = 2e-3 * float(step + 1) / WARMUP_STEPS
        else:
            lr = 2e-3
        for group in optimizer.param_groups:
            group["lr"] = lr
        images = next(stream).to(device, non_blocking=True)
        en, de = model(images)
        p = min(0.9 * step / 1000.0, 0.9)
        loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(trainable.parameters(), max_norm=0.1)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 20 == 0 or step + 1 == PRETRAIN_STEPS:
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            print(
                f"pretrain {step + 1}/{PRETRAIN_STEPS} loss={losses[-1]:.4f} lr={lr:.6f} peak_mib={peak:.0f}",
                flush=True,
            )
    return {
        "seconds": time.time() - started,
        "first20_mean": mean_of(losses[:20]),
        "after_warmup_mean": mean_of(losses[WARMUP_STEPS : WARMUP_STEPS + 20]),
        "last20_mean": mean_of(losses[-20:]),
        "losses_head": [round(v, 5) for v in losses[:5]],
        "losses_tail": [round(v, 5) for v in losses[-5:]],
    }


def train_loc(model, head, loader, device):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    losses = []
    started = time.time()
    stream = cycle_loader(loader)
    for step in range(LOC_STEPS):
        images, masks = next(stream)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.no_grad():
            en, de = model(images)
            residuals, distances = residual_and_distance(en, de)
        logits = head(residuals, distances)
        loss = F.binary_cross_entropy_with_logits(logits, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(head.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 20 == 0 or step + 1 == LOC_STEPS:
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            print(
                f"loc {step + 1}/{LOC_STEPS} loss={losses[-1]:.4f} peak_mib={peak:.0f}",
                flush=True,
            )
    return {
        "seconds": time.time() - started,
        "first20_mean": mean_of(losses[:20]),
        "last20_mean": mean_of(losses[-20:]),
        "losses_head": [round(v, 5) for v in losses[:5]],
        "losses_tail": [round(v, 5) for v in losses[-5:]],
    }


@torch.no_grad()
def _accumulate(model, head, loader, device, kind):
    sig_sum = 0.0
    sig_count = 0
    hot_sum = 0.0
    interior_sum = 0.0
    interior_count = 0
    border_sum = 0.0
    border_count = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        en, de = model(images)
        prob = torch.sigmoid(head(*residual_and_distance(en, de)))
        if kind == "clean":
            sig_sum += prob.sum().item()
            sig_count += prob.numel()
            hot_sum += (prob > 0.5).float().sum().item()
        else:
            interior = masks > 0.8
            border = (masks > 0.05) & (masks <= 0.8)
            if interior.any():
                interior_sum += prob[interior].sum().item()
                interior_count += int(interior.sum())
            if border.any():
                border_sum += prob[border].sum().item()
                border_count += int(border.sum())
    return {
        "sig_sum": sig_sum,
        "sig_count": sig_count,
        "hot_sum": hot_sum,
        "interior_sum": interior_sum,
        "interior_count": interior_count,
        "border_sum": border_sum,
        "border_count": border_count,
    }


@torch.no_grad()
def evaluate_val(model, head, items, device):
    model.eval()
    head.eval()
    clean_loader = DataLoader(
        PasteSet(items, mode="clean"), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    paste_loader = DataLoader(
        PasteSet(items, mode="paste"), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    clean = _accumulate(model, head, clean_loader, device, "clean")
    paste = _accumulate(model, head, paste_loader, device, "paste")
    return {
        "clean_mean_sigmoid": clean["sig_sum"] / clean["sig_count"],
        "clean_frac_above_0.5": clean["hot_sum"] / clean["sig_count"],
        "paste_interior_mean_sigmoid": paste["interior_sum"] / max(paste["interior_count"], 1),
        "paste_border_mean_sigmoid": paste["border_sum"] / max(paste["border_count"], 1),
        "n_val": len(items),
        "interior_tokens": paste["interior_count"],
        "border_tokens": paste["border_count"],
    }


def main():
    setup_seed(SEED)
    RUN.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in py312")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()
    train_items, val_items, counts = split_items()
    print("counts", counts, flush=True)
    model = build_model(device)
    clean_loader = DataLoader(
        CleanSet(train_items),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    pretrain_report = pretrain(model, clean_loader, device)
    setup_seed(SEED + 1)
    paste_loader = DataLoader(
        PasteSet(train_items),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    head = LocBranch().to(device)
    before = evaluate_val(model, head, val_items, device)
    loc_report = train_loc(model, head, paste_loader, device)
    after = evaluate_val(model, head, val_items, device)
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    report = {
        "classes_locked": list(CLASSES),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "pretrain_steps": PRETRAIN_STEPS,
        "loc_steps": LOC_STEPS,
        "counts": counts,
        "peak_mib": round(peak_mib, 1),
        "fits_8gb": peak_mib < 7500,
        "pretrain": pretrain_report,
        "loc": loc_report,
        "val_before_loc": before,
        "val_after_loc": after,
        "flags": {
            "recon_loss_down": pretrain_report["last20_mean"] < pretrain_report["after_warmup_mean"],
            "loc_loss_down": loc_report["last20_mean"] < loc_report["first20_mean"],
            "clean_not_painted": after["clean_frac_above_0.5"] is not None
            and after["clean_frac_above_0.5"] < 0.20,
            "paste_above_clean": (
                after["paste_interior_mean_sigmoid"] is not None
                and after["clean_mean_sigmoid"] is not None
                and after["paste_interior_mean_sigmoid"] > after["clean_mean_sigmoid"]
            ),
        },
        "not_evaluated": "official test set and ground_truth were not read",
        "not_a_paper_baseline": "3 classes, batch 4, 300 reconstruction steps; not the 15-class 10000-iter checkpoint",
    }
    torch.save(
        {
            "bottleneck": model.bottleneck.state_dict(),
            "decoder": model.decoder.state_dict(),
            "loc_branch": head.state_dict(),
            "report": report,
        },
        RUN / "screen_seed1.pt",
    )
    (RUN / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["flags"], indent=2), flush=True)
    print(f"peak_mib={report['peak_mib']}", flush=True)
    print(f"wrote {RUN / 'report.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        RUN.mkdir(parents=True, exist_ok=True)
        text = traceback.format_exc()
        (RUN / "error.txt").write_text(text, encoding="utf-8")
        print(text, flush=True)
        raise
