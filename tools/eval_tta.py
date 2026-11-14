"""在 OOF 上比较推理侧的各种做法：单视角 / 翻转 TTA / 多尺度 TTA。

模型是全卷积的，letterbox 尺寸可以随便换；但换尺寸后
「补边偏移」变了，所以真值坐标必须按同一尺寸重新构建
（`build_samples_from_layout(layout, size)` 已处理）。

用法:
    .\\tools\\py.ps1 tools\\eval_tta.py --data data/task3 --runs runs/t3_b0_512
    .\\tools\\py.ps1 tools\\eval_tta.py --data data/task3 --runs runs/t3_b0_512 --sizes 448 512 640
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fovea_data import build_samples_from_layout, discover
from src.fovea_predict import load_models, predict_paths


def mcrmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean((pred - gt) ** 2, axis=0))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/task3")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, default=[448, 512, 576])
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layout = discover(args.data)
    models = load_models([Path(r) for r in args.runs], dev)

    # 每个尺寸下重建一次 (paths, 真值坐标)
    data_by_size = {}
    for s in args.sizes:
        paths, xy, _ = build_samples_from_layout(layout, s)
        data_by_size[s] = (paths, xy)

    per_fold = []          # [{key: mcrmse}]
    for rd in [Path(r) for r in args.runs]:
        n_folds = len(sorted(rd.glob("fold*_best.pt")))
        for k in range(n_folds):
            idx = np.load(rd / f"val_idx_fold{k}.npy")
            cand = [m for m in models if m["tag"] == rd.name and m["fold"] == k]
            if not cand:
                continue
            m = cand[0]
            scores = {}
            preds_by_size = {}
            for s in args.sizes:
                paths, xy = data_by_size[s]
                va_paths = [paths[i] for i in idx]
                gt = xy[idx]
                p_single = predict_paths(m["model"], va_paths, s, dev, tta=False)
                p_flip = predict_paths(m["model"], va_paths, s, dev, tta=True)
                preds_by_size[s] = (p_single, p_flip, gt)
                scores[f"single@{s}"] = mcrmse(p_single, gt)
                scores[f"flip@{s}"] = mcrmse(p_flip, gt)
            # 多尺度组合（都用翻转 TTA）
            for r in range(2, len(args.sizes) + 1):
                for combo in itertools.combinations(args.sizes, r):
                    gt = preds_by_size[combo[0]][2]
                    mix = np.mean([preds_by_size[s][1] for s in combo], axis=0)
                    scores["flip@" + "+".join(map(str, combo))] = mcrmse(mix, gt)
            per_fold.append(scores)
            print(f"  {rd.name}/f{k}: " + "  ".join(
                f"{kk}={vv:.5f}" for kk, vv in sorted(scores.items(),
                                                      key=lambda kv: kv[1])[:4]))

    keys = sorted({k for sc in per_fold for k in sc})
    print("\n=== 各折平均（MCRMSE，仅统计所有折都有的组合）===")
    ranked = []
    for k in keys:
        vals = [sc[k] for sc in per_fold if k in sc]
        if len(vals) == len(per_fold):
            ranked.append((float(np.mean(vals)), k, len(vals)))
    for m1, k, n in sorted(ranked):
        print(f"  {k:24s} {m1:.5f}  (n_folds={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
