import os, sys, json, time, subprocess, platform

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

def sh(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        out = e.output
    print(out[-3000:])

print("=== env ===")
print("python:", sys.version)
sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))
print("cwd:", os.getcwd())
print("code dir exists:", os.path.isdir(CODE))
print("data dir exists:", os.path.isdir(DATA))

print("=== data classes ===")
if os.path.isdir(DATA):
    items = sorted([d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d))])
    print("classes:", len(items), items)
    for d in items[:3]:
        print(d, os.listdir(os.path.join(DATA, d))[:5])
    n_train = 0
    for d in items:
        p = os.path.join(DATA, d, "train", "good")
        if os.path.isdir(p):
            n_train += len([f for f in os.listdir(p) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))])
    print("total train/good images:", n_train)

print("=== deps ===")
for m in ["timm", "cv2", "skimage", "sklearn", "pandas", "scipy", "ptflops", "tabulate", "matplotlib"]:
    try:
        mod = __import__(m)
        print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(m, "MISSING:", repr(e))

print("=== import repo ===")
sys.path.insert(0, CODE)
os.chdir(CODE)
try:
    from models.uad import ViTill
    from models import vit_encoder
    from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
    from optimizers import StableAdamW
    from dataset import get_data_transforms, MVTecDataset
    print("imports ok; weights file:", os.path.exists("backbones/weights/dinov2_vitb14_reg4_pretrain.pth"))
except Exception as e:
    import traceback; traceback.print_exc(); raise

print("=== one training step, batch 16 ===")
from functools import partial
from torch import nn
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

torch.manual_seed(1)
items = sorted([d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d))])
data_transform, _ = get_data_transforms(448, 392)
train_list = []
for i, item in enumerate(items):
    ds = ImageFolder(root=os.path.join(DATA, item, "train"), transform=data_transform)
    ds.classes = item
    ds.class_to_idx = {item: i}
    ds.samples = [(s[0], i) for s in ds.samples]
    train_list.append(ds)
train_data = ConcatDataset(train_list)
print("concat train size:", len(train_data))
dl = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4, drop_last=True)

encoder = vit_encoder.load("dinov2reg_vit_base_14")
bottleneck = nn.ModuleList([bMlp(768, 768 * 4, 768, drop=0.2)])
decoder = nn.ModuleList([
    VitBlock(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True,
             norm_layer=partial(nn.LayerNorm, eps=1e-8),
             attn=LinearAttention2)
    for _ in range(8)])
model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
               target_layers=[2,3,4,5,6,7,8,9],
               fuse_layer_encoder=[[0,1,2,3],[4,5,6,7]],
               fuse_layer_decoder=[[0,1,2,3],[4,5,6,7]])
model = model.cuda()
trainable = nn.ModuleList([bottleneck, decoder])
opt = StableAdamW([{"params": trainable.parameters()}], lr=2e-3, betas=(0.9,0.999),
                  weight_decay=1e-4, amsgrad=True, eps=1e-10)
from utils import global_cosine_hm_percent
img, label = next(iter(dl))
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
img = img.cuda()
en, de = model(img)
loss = global_cosine_hm_percent(en, de, p=0.0, factor=0.1)
opt.zero_grad(); loss.backward()
nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
opt.step()
torch.cuda.synchronize()
print("step ok: loss=%.6f  time=%.2fs  peak=%.0f MiB  batch=%s" % (
    loss.item(), time.time()-t0, torch.cuda.max_memory_allocated()/1024**2, tuple(img.shape)))
print("SMOKE_OK")
