"""吞吐量基准：分别测「增强」与「前向+反向」的耗时，用来定训练预算。

用法：
    .\\run.ps1 -m src_fer.bench --img-size 128 --batch-size 256
"""
from __future__ import annotations

import argparse
import time

import torch

from . import data as D
from .aug import mixup_cutmix, one_hot, random_batch, soft_cross_entropy
from .models import build_model, build_optimizer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/2026task2")
    ap.add_argument("--model", default="convnext_tiny")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args(argv)

    dev = torch.device("cuda")
    ds = D.FerdDataset(args.data)
    images = torch.from_numpy(ds.train_images).to(dev)
    labels = torch.from_numpy(ds.train_labels).to(dev)

    model = build_model(args.model, len(D.CLASSES), img_size=args.img_size,
                        drop_path_rate=0.1, verbose=False).to(dev)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model)
    opt = build_optimizer(model, lr=3e-4)

    # 预热
    sel = torch.randint(0, len(labels), (args.batch_size,), device=dev)
    x = random_batch(images[sel], labels[sel], args.img_size)
    if args.channels_last:
        x = x.to(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        soft_cross_entropy(model(x), one_hot(labels[sel], len(D.CLASSES), 0.1)).backward()
    torch.cuda.synchronize()

    # 1) 增强
    t0 = time.time()
    for _ in range(args.iters):
        sel = torch.randint(0, len(labels), (args.batch_size,), device=dev)
        xb = random_batch(images[sel], labels[sel], args.img_size)
    torch.cuda.synchronize()
    t_aug = (time.time() - t0) / args.iters

    # 2) 前向+反向
    t0 = time.time()
    for _ in range(args.iters):
        sel = torch.randint(0, len(labels), (args.batch_size,), device=dev)
        x = random_batch(images[sel], labels[sel], args.img_size)
        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)
        y = one_hot(labels[sel], len(D.CLASSES), 0.1)
        x, y = mixup_cutmix(x, y, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = soft_cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t_all = (time.time() - t0) / args.iters

    steps_per_epoch = 22620 // args.batch_size
    print(f"配置: {args.model} @{args.img_size}  bs={args.batch_size}  "
          f"channels_last={args.channels_last}  compile={args.compile}")
    print(f"  增强 {t_aug*1000:.1f} ms/batch")
    print(f"  增强+前反向 {t_all*1000:.1f} ms/batch  ->  计算部分 {(t_all-t_aug)*1000:.1f} ms")
    print(f"  每 epoch({steps_per_epoch} 步) 约 {t_all*steps_per_epoch:.1f} s  "
          f"-> 每折 30 epoch 约 {t_all*steps_per_epoch*30/60:.1f} min  "
          f"-> 5 折约 {t_all*steps_per_epoch*30*5/3600:.2f} h")
    print(f"峰值显存 {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
