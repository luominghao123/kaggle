"""在 OOF 上评估「多 run 集成」是否真的比单 run 好（各 run 用同一个 seed 划分，所以折号对齐）。

用法:
    .\\tools\\py.ps1 tools\\eval_ensemble.py --data data/task3 --runs runs/t3_b0_512 runs/t3_cnxt_512
    .\\tools\\py.ps1 tools\\eval_ensemble.py --data data/task3 --runs runs/a runs/b runs/c --size 512
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


def pooled(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/task3")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layout = discover(args.data)
    paths, xy, _ = build_samples_from_layout(layout, args.size)
    models = load_models([Path(r) for r in args.runs], dev)

    runs = [Path(r) for r in args.runs]
    n_folds = min(len(sorted(r.glob("fold*_best.pt"))) for r in runs)

    # fold -> run -> (pred, gt)
    store: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for k in range(n_folds):
        idx = np.load(runs[0] / f"val_idx_fold{k}.npy")   # 所有 run 同一 seed，折划分一致
        va_paths = [paths[i] for i in idx]
        gt = xy[idx]
        store[k] = {}
        for rd in runs:
            cand = [m for m in models if m["tag"] == rd.name and m["fold"] == k]
            if not cand:
                continue
            store[k][rd.name] = (predict_paths(cand[0]["model"], va_paths,
                                               cand[0]["img_size"], dev, tta=args.tta), gt)

    combos = []
    for r in range(1, len(runs) + 1):
        for combo in itertools.combinations([r.name for r in runs], r):
            combos.append(combo)

    print("\n=== OOF（越小越好）===")
    out = []
    for combo in combos:
        per_fold, per_fold_pooled = [], []
        for k in range(n_folds):
            parts = [store[k][name][0] for name in combo if name in store[k]]
            if not parts:
                continue
            gt = store[k][combo[0]][1] if combo[0] in store[k] else store[k][parts and list(store[k])[0]][1]
            mix = np.mean(parts, axis=0)
            per_fold.append(mcrmse(mix, gt))
            per_fold_pooled.append(pooled(mix, gt))
        label = " + ".join(combo)
        out.append((float(np.mean(per_fold)), float(np.mean(per_fold_pooled)), label, per_fold))
    for m1, m2, label, folds in sorted(out):
        fstr = " ".join(f"{v:.4f}" for v in folds)
        print(f"  {m1:.5f}  pooled {m2:.5f}  {label}")
        print(f"            各折: {fstr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
