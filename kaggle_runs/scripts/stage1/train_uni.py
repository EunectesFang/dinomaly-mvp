"""Dinomaly multi-class MVTec-AD baseline, official protocol.

Paper setting: 10,000 iterations, batch 16, DINOv2-Register ViT-B/14 frozen,
392 input, StableAdamW lr 2e-3, warmup 100, cosine to 2e-4, hard-mining
p -> 0.9 over 1000 iters, factor 0.1, grad clip 0.1, seed 1.

Only changes relative to the released dinomaly_mvtec_uni.py:
  * device cuda:0 instead of cuda:1
  * checkpoint saving (commented out upstream)
  * metrics written to JSON
  * CUDA_LAUNCH_BLOCKING not set (speed only)
"""
import os, sys, json, time, random, argparse

OUT = "/kaggle/working"


def locate(marker, preferred):
    for c in preferred:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, marker)):
            return c
    for root, dirs, files in os.walk("/kaggle/input"):
        if marker in dirs or marker in files:
            return root
    raise RuntimeError(f"cannot locate {marker} under /kaggle/input")


CODE = locate("dinomaly_mvtec_uni.py", [
    "/kaggle/input/dinomaly-code",
    "/kaggle/input/datasets/eunectesqaq/dinomaly-code",
])
DATA = locate("bottle", [
    "/kaggle/input/mvtec-ad",
    "/kaggle/input/datasets/ipythonx/mvtec-ad",
])

sys.path.insert(0, CODE)
os.chdir(CODE)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from functools import partial

from dataset import get_data_transforms, MVTecDataset
from models.uad import ViTill
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from dinov1.utils import trunc_normal_
from optimizers import StableAdamW
from utils import evaluation_batch, global_cosine_hm_percent, WarmCosineScheduler
import utils as _utils
from skimage import measure
from sklearn.metrics import auc

LOG_PATH = os.path.join(OUT, "train.log")


def compute_pro_numpy(masks, amaps, num_th=200):
    """Pandas-free replacement for utils.compute_pro (pandas>=2 removed append)."""
    min_th, max_th = float(amaps.min()), float(amaps.max())
    if not np.isfinite(min_th) or not np.isfinite(max_th) or max_th <= min_th:
        return 0.0
    delta = (max_th - min_th) / num_th
    inverse = 1 - masks
    neg_total = int(inverse.sum())
    pros, fprs = [], []
    for th in np.arange(min_th, max_th, delta):
        binmap = amaps > th
        cur = []
        for bmap, mask in zip(binmap, masks):
            for region in measure.regionprops(measure.label(mask)):
                coords = region.coords
                cur.append(bmap[coords[:, 0], coords[:, 1]].sum() / region.area)
        pros.append(float(np.mean(cur)) if cur else 0.0)
        fprs.append(float(np.logical_and(inverse, binmap).sum()) / max(neg_total, 1))
    fprs = np.asarray(fprs)
    pros = np.asarray(pros)
    keep = fprs < 0.3
    if keep.sum() < 2 or fprs[keep].max() <= 0:
        return 0.0
    return float(auc(fprs[keep] / fprs[keep].max(), pros[keep]))


_utils.compute_pro = compute_pro_numpy


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_item_lists(data_path):
    items = sorted(d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)))
    if len(items) != 15:
        raise RuntimeError(f"expected 15 MVTec classes, found {len(items)}: {items}")
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--total_iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--save_at", type=str, default="5000,10000")
    parser.add_argument("--eval_at", type=str, default="10000")
    parser.add_argument("--eval", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"device={device} torch={torch.__version__} batch={args.batch_size} iters={args.total_iters}")

    setup_seed(1)
    item_list = build_item_lists(DATA)
    log(f"classes={item_list}")

    total_iters = args.total_iters
    batch_size = args.batch_size
    image_size = 448
    crop_size = 392

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list, test_data_list = [], []
    for i, item in enumerate(item_list):
        train_path = os.path.join(DATA, item, "train")
        test_path = os.path.join(DATA, item)

        train_data = ImageFolder(root=train_path, transform=data_transform)
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data.samples = [(sample[0], i) for sample in train_data.samples]

        test_data = MVTecDataset(root=test_path, transform=data_transform,
                                 gt_transform=gt_transform, phase="test")
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=4, drop_last=True)
    log(f"train images={len(train_data)} steps_per_epoch={len(train_dataloader)}")

    encoder_name = "dinov2reg_vit_base_14"
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)
    embed_dim, num_heads = 768, 12

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=LinearAttention2)
        for _ in range(8)
    ])

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                   target_layers=target_layers, mask_neighbor_size=0,
                   fuse_layer_encoder=fuse_layer_encoder,
                   fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)
    trainable = nn.ModuleList([bottleneck, decoder])

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = StableAdamW([{"params": trainable.parameters()}],
                            lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4,
                            amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=args.lr,
                                       final_value=args.lr / 10,
                                       total_iters=total_iters, warmup_iters=100)

    save_at = {int(x) for x in args.save_at.split(",") if x}
    eval_at = {int(x) for x in args.eval_at.split(",") if x}
    metrics_by_iter = {}
    it = 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device, non_blocking=True)
            en, de = model(img)

            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            lr_scheduler.step()
            loss_list.append(loss.item())
            it += 1

            if it % 100 == 0 or it == 1:
                peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
                log(f"iter {it}/{total_iters} loss={loss_list[-1]:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} peak_mib={peak:.0f} "
                    f"elapsed_min={(time.time()-started)/60:.1f}")

            if it in save_at or (args.eval and it in eval_at):
                ckpt = os.path.join(OUT, f"model_{it}.pt")
                torch.save({"bottleneck": bottleneck.state_dict(),
                            "decoder": decoder.state_dict(),
                            "iter": it, "batch_size": batch_size},
                           ckpt)
                log(f"saved {ckpt}")

            if args.eval and it in eval_at:
                metrics = evaluate(model, item_list, test_data_list, data_transform,
                                   gt_transform, batch_size, device)
                metrics_by_iter[it] = metrics
                with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics_by_iter, f, indent=2, ensure_ascii=False)
                log(f"eval@{it} mean={metrics['mean']}")
                model.train()

            if it == total_iters:
                break
        log(f"iter [{it}/{total_iters}] loss:{np.mean(loss_list):.4f}")
        if it == total_iters:
            break

    if total_iters not in save_at:
        torch.save({"bottleneck": bottleneck.state_dict(),
                    "decoder": decoder.state_dict(),
                    "iter": it, "batch_size": batch_size},
                   os.path.join(OUT, f"model_{it}.pt"))
    log("DONE")


def evaluate(model, item_list, test_data_list, data_transform, gt_transform, batch_size, device):
    model.eval()
    keys = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
    per_class = {}
    means = []
    with torch.no_grad():
        for item, test_data in zip(item_list, test_data_list):
            loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
            res = evaluation_batch(model, loader, device, max_ratio=0.01, resize_mask=256)
            vals = [float(round(r, 6)) for r in res]
            per_class[item] = dict(zip(keys, vals))
            means.append(vals)
            log(f"  {item}: " + " ".join(f"{k}={v:.4f}" for k, v in zip(keys, vals)))
    mean = dict(zip(keys, [float(np.mean([m[i] for m in means])) for i in range(len(keys))]))
    return {"per_class": per_class, "mean": mean}


if __name__ == "__main__":
    main()
