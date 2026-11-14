"""Debug: why does the heatmap loss not go down?

Checks: (1) do augmented inputs still contain the fundus structure,
(2) does the initial logits map have any spatial variation,
(3) after N manual training steps, does the loss / spatial std move,
(4) what is the theoretical lower bound of the heat loss.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fovea_data import (FoveaDataset, LetterboxCache,
                            build_samples_from_layout, discover)
from src.fovea_model import FoveaLoss, build_model, gaussian_target

S = 256
layout = discover("data_fake")
paths, xy, sizes = build_samples_from_layout(layout, S)
cache = LetterboxCache(S, 512)
ds = FoveaDataset(paths[:32], xy[:32], S, train=True, cache=cache)

print("=== 1) data check ===")
batch = [ds[i] for i in range(8)]
imgs = torch.stack([b[0] for b in batch])
tgt = torch.stack([b[1] for b in batch])
print("input min/max/mean/std: %.2f / %.2f / %.3f / %.3f"
      % (imgs.min(), imgs.max(), imgs.mean(), imgs.std()))
print("targets:", tgt.round(decimals=3).tolist())

out_dir = Path(".tmp/aug_check")
out_dir.mkdir(parents=True, exist_ok=True)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
from torchvision.transforms.functional import to_pil_image
for i in range(4):
    to_pil_image((imgs[i] * std + mean).clamp(0, 1)).save(out_dir / f"aug_{i}.png")

print("\n=== 2) initial forward: is there spatial variation? ===")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = build_model("effnet_b0", pretrained=True, verbose=False).to(dev)
imgs_dev = imgs.to(dev)
with torch.no_grad():
    o = net(imgs_dev)
lg = o["logits"]
print("logits shape", tuple(lg.shape), "global std %.6f" % lg.std().item())
print("per-image spatial std: %.6f" % lg.flatten(2).std(dim=-1).mean().item())
print("coords:", o["coords"].round(decimals=4).tolist())

print("\n=== 3) manual training, 80 steps ===")
crit = FoveaLoss()
opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
tgt_dev = tgt.to(dev)
for step in range(81):
    opt.zero_grad(set_to_none=True)
    out = net(imgs_dev)
    loss, parts = crit(out, tgt_dev)
    loss.backward()
    opt.step()
    if step % 10 == 0:
        lgd = out["logits"].detach()
        print("step %3d  loss %.4f  heat %.4f  xy %.4f  spatial_std %.5f  |W_head| %.5f"
              % (step, loss.item(), parts["heat"], parts["xy"],
                 lgd.flatten(2).std(dim=-1).mean().item(),
                 net.head[-1].weight.abs().mean().item()))

print("\n=== 4) theoretical bounds of the heat loss ===")
with torch.no_grad():
    out = net(imgs_dev)
    h, w = out["logits"].shape[-2:]
    sigma = max(1.0, 0.02 * max(h, w))
    perfect = gaussian_target(tgt_dev, h, w, sigma)
    logp = torch.log(perfect.clamp_min(1e-30))
    print("uniform CE = %.4f  (log(%d))" % (np.log(h * w), h * w))
    print("perfect CE = %.4f  (sigma=%.2f cells)" % (-(perfect * logp).sum().item(), sigma))
