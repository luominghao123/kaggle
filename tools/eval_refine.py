"""在 OOF（各折验证集）上评估「粗预测 / 二段精修 / 不同融合权重」哪个更好。

这是决定提交用哪种推理方式的**证据**，不要凭感觉设 50/50 融合。

用法:
    .\\tools\\py.ps1 tools\\eval_refine.py --data data/task3 --runs runs/t3_b0_512
    .\\tools\\py.ps1 tools\\eval_refine.py --data data/task3 --runs runs/t3_b0_512 --window 0.25 0.35 0.45
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fovea_data import build_samples_from_layout, discover
from src.fovea_predict import load_models, predict_paths, refine_paths


def mcrmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean((pred - gt) ** 2, axis=0))))


def pooled(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/task3")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--window", nargs="+", type=float, default=[0.25, 0.35, 0.5])
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layout = discover(args.data)
    paths, xy, sizes = build_samples_from_layout(layout, 512)
    models = load_models([Path(r) for r in args.runs], dev)
    tta = not args.no_tta

    rows = []
    for rd in [Path(r) for r in args.runs]:
        ckpts = sorted(rd.glob("fold*_best.pt"))
        for k in range(len(ckpts)):
            idx = np.load(rd / f"val_idx_fold{k}.npy")
            va_paths = [paths[i] for i in idx]
            gt = xy[idx]
            cand = [m for m in models if m["tag"] == rd.name and m["fold"] == k]
            if not cand:
                continue
            m = cand[0]

            coarse = predict_paths(m["model"], va_paths, m["img_size"], dev, tta=tta)
            row = {"fold": f"{rd.name}/f{k}", "n": len(idx),
                   "coarse": mcrmse(coarse, gt), "coarse_pooled": pooled(coarse, gt)}
            for w in args.window:
                ref = refine_paths(m["model"], va_paths, coarse, m["img_size"],
                                   layout.unit, dev, window=w)
                row[f"ref@{w}"] = mcrmse(ref, gt)
                row[f"ref@{w}_pooled"] = pooled(ref, gt)
                for a in (0.25, 0.5, 0.75):
                    blend = a * coarse + (1 - a) * ref
                    row[f"blend{a}@{w}"] = mcrmse(blend, gt)
                    row[f"blend{a}@{w}_pooled"] = pooled(blend, gt)
            rows.append(row)
            print(f"  {row['fold']} (n={row['n']}): coarse {row['coarse']:.5f}  "
                  + "  ".join(f"ref@{w}={row[f'ref@{w}']:.5f}" for w in args.window))

    print("\n=== 各折平均（越小越好）===")
    keys = [k for k in rows[0] if k not in ("fold", "n") and not k.endswith("_pooled")]
    for k in sorted(keys, key=lambda kk: float(np.mean([r[kk] for r in rows]))):
        m1 = float(np.mean([r[k] for r in rows]))
        m2 = float(np.mean([r[k + "_pooled"] for r in rows]))
        print(f"  {k:16s} MCRMSE {m1:.5f}   pooled-RMSE {m2:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
