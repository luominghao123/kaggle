"""实验四：OOF 概率加权集成 + 阈值扫描（判断「多骨干集成到底有没有用」）。

输入是 `src.vessel_predict --eval --eval-probs a.npz b.npz ...` 存下来的
**无泄漏** OOF 概率（每张图只用没见过它的那一折模型 + 翻转/多尺度 TTA）。
在这里扫「权重 × 阈值」，避免用测试集调参。

用法
----
    .\\tools\\py.ps1 tools/eval_vessel_ensemble.py `
        --probs runs/_tune/oof_b0.npz runs/_tune/oof_cnxt.npz --weights 0.3 0.7
    .\\tools\\py.ps1 tools/eval_vessel_ensemble.py `
        --probs runs/_tune/oof_b0.npz runs/_tune/oof_cnxt.npz --sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vessel_data import DEFAULT_DATA, discover, load_set  # noqa: E402
from src.vessel_model import postprocess, seg_metrics  # noqa: E402
from src.vessel_predict import dice_512  # noqa: E402


def load_npz(path: Path, ids_order: list[str]) -> np.ndarray:
    z = np.load(path)
    ids = [str(i) for i in z["ids"]]
    order = [ids.index(i) for i in ids_order]
    return z["probs"].astype(np.float32)[order]


def evaluate(prob: np.ndarray, train: list, thr: float, min_size: int = 0) -> dict:
    d, d512, sen, spe = [], [], [], []
    for i, s in enumerate(train):
        m = postprocess(prob[i], s.fov, thr=thr, min_size=min_size)
        mt = seg_metrics(m, s.mask, s.fov)
        d.append(mt["dice"])
        sen.append(mt["sen"])
        spe.append(mt["spe"])
        d512.append(dice_512(m, s.mask))
    return {"thr": thr, "dice": float(np.mean(d)), "dice512": float(np.mean(d512)),
            "sen": float(np.mean(sen)), "spe": float(np.mean(spe))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--probs", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--sweep", action="store_true", help="两两权重扫描")
    ap.add_argument("--thr", type=float, default=0.65)
    ap.add_argument("--thrs", default="0.5:0.85:0.05")
    args = ap.parse_args()

    train_items, _ = discover(args.data)
    train = load_set(train_items)
    ids = [s.image_id for s in train]
    probs = [load_npz(ROOT / p if not Path(p).is_absolute() else Path(p), ids) for p in args.probs]
    for p, pr in zip(args.probs, probs):
        e = evaluate(pr, train, args.thr)
        print(f"[单模型] {Path(p).name:28s} thr={args.thr:.2f} dice={e['dice']:.4f} dice512={e['dice512']:.4f}")

    lo, hi, step = (float(x) for x in args.thrs.split(":"))
    thrs = [round(x, 2) for x in np.arange(lo, hi + 1e-9, step)]

    if args.sweep and len(probs) == 2:
        print("[扫描] w = 第一个模型的权重")
        rows = []
        for w in np.arange(0, 1.001, 0.1):
            mix = w * probs[0] + (1 - w) * probs[1]
            best = max((evaluate(mix, train, t) for t in thrs), key=lambda r: r["dice512"])
            rows.append({"w0": round(float(w), 2), **best})
            print(f"   w0={w:.1f} thr={best['thr']:.2f} dice={best['dice']:.4f} "
                  f"dice512={best['dice512']:.4f} sen={best['sen']:.4f} spe={best['spe']:.4f}")
        best = max(rows, key=lambda r: r["dice512"])
        print(f"[扫描] 最优 w0={best['w0']} thr={best['thr']} dice={best['dice']:.4f} dice512={best['dice512']:.4f}")
        out = ROOT / "runs" / "_tune" / "ensemble_sweep.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"probs": args.probs, "rows": rows}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"[扫描] -> {out}")

    weights = args.weights or [1.0 / len(probs)] * len(probs)
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    mix = sum(wi * pr for wi, pr in zip(w, probs))
    rows = [evaluate(mix, train, t) for t in thrs]
    best = max(rows, key=lambda r: r["dice512"])
    print(f"[集成] 权重 {np.round(w, 3).tolist()}  thr={best['thr']:.2f} "
          f"dice={best['dice']:.4f} dice512={best['dice512']:.4f} "
          f"sen={best['sen']:.4f} spe={best['spe']:.4f}")
    for r in rows:
        print(f"   thr={r['thr']:.2f} dice={r['dice']:.4f} dice512={r['dice512']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
