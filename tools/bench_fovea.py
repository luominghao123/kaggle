"""吞吐基准：分别测「数据增强」与「模型前反向」的耗时，判断训练瓶颈在哪。

用法:
    .\\run.ps1 tools\\bench_fovea.py                 # 默认 320/512 两档
    .\\run.ps1 tools\\bench_fovea.py --sizes 384 512 --batch-size 16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fovea_data import (FoveaDataset, LetterboxCache,
                            build_samples_from_layout, discover)
from src.fovea_model import FoveaLoss, build_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_fake")
    ap.add_argument("--backbone", default="effnet_b0")
    ap.add_argument("--sizes", nargs="+", type=int, default=[320, 512])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--iters", type=int, default=12)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    layout = discover(args.data)
    print(f"samples: {len(layout.rows)}  test: {len(layout.test_images)}")

    for S in args.sizes:
        paths, xy, _ = build_samples_from_layout(layout, S)
        cache = LetterboxCache(S, 1024)
        n = min(len(paths), 128)
        ds = FoveaDataset(paths[:n], xy[:n], S, train=True, cache=cache)

        # --- 数据增强耗时（含首次读图/缓存填充）---
        t0 = time.time()
        for i in range(n):
            ds[i]
        warm = time.time() - t0
        t0 = time.time()
        rounds = 3
        for _ in range(rounds):
            for i in range(n):
                ds[i]
        aug = (time.time() - t0) / (rounds * n)

        # --- 模型前反向 ---
        net = build_model(args.backbone, pretrained=False, verbose=False).to(dev)
        crit = FoveaLoss()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
        x = torch.randn(args.batch_size, 3, S, S, device=dev)
        y = torch.rand(args.batch_size, 2, device=dev)
        for _ in range(3):                       # 预热（cudnn 选算法）
            opt.zero_grad(set_to_none=True)
            out = net(x)
            loss, _ = crit(out, y)
            loss.backward()
            opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            opt.zero_grad(set_to_none=True)
            out = net(x)
            loss, _ = crit(out, y)
            loss.backward()
            opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        step = (time.time() - t0) / args.iters
        per_img_model = step / args.batch_size

        total = aug + per_img_model
        print(f"\n=== img_size {S} ===")
        print(f"  数据增强      : {aug*1000:6.1f} ms/张（首次填充缓存 {warm:.1f}s / {n} 张）")
        print(f"  模型 fwd+bwd  : {per_img_model*1000:6.1f} ms/张（batch {args.batch_size}, {step*1000:.0f} ms/step）")
        print(f"  合计          : {total*1000:6.1f} ms/张")
        for n_train, ep, folds in ((500, 40, 5), (1500, 40, 5), (3000, 30, 5)):
            hours = total * n_train * ep * folds / 3600
            print(f"    按 {n_train} 张 × {ep} epoch × {folds} 折估算: {hours:.1f} 小时")
        if dev.type == "cuda":
            print(f"  显存峰值      : {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
