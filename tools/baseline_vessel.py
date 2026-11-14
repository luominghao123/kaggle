"""实验四：无监督经典基线（Frangi 管状结构滤波 + 阈值）—— 只用来做报告里的对照。

思路
----
血管是「局部暗的管状/线状结构」。Frangi 血管相似度（Hessian 矩阵两个特征值
λ1, λ2 满足 |λ1|≪|λ2|）能在没有标签的情况下把细管状结构点亮，
再在视场内取 Otsu 阈值二值化。它不需要训练，因此是衡量深度学习模型
是否真的学到东西的低成本参照物。

用法
----
    .\\tools\\py.ps1 tools/baseline_vessel.py                 # 在 20 张训练图上评 Dice
    .\\tools\\py.ps1 tools/baseline_vessel.py --submit        # 顺便写一份提交（保底方案）
    .\\tools\\py.ps1 tools/baseline_vessel.py --sigmas 1 2 3 --beta 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vessel_data import (DEFAULT_DATA, discover, imwrite_unicode, load_set,  # noqa: E402
                            mask_to_rle_512)
from src.vessel_model import postprocess, seg_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def frangi_map(green_u8: np.ndarray, sigmas: tuple[float, ...], beta: float = 0.5,
               black_ridges: bool = True) -> np.ndarray:
    from skimage.filters import frangi
    g = green_u8.astype(np.float32) / 255.0
    v = frangi(g, sigmas=sigmas, beta=beta, black_ridges=black_ridges)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / max(hi - lo, 1e-6)


def otsu_threshold(v: np.ndarray, fov: np.ndarray) -> float:
    vals = (v[fov] * 255).astype(np.uint8)
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t) / 255.0


def percentile_threshold(v: np.ndarray, fov: np.ndarray, keep_pct: float) -> float:
    """留出「最像血管的 keep_pct%」像素。比 Otsu 更适合这种极度不平衡的掩膜。"""
    return float(np.percentile(v[fov], 100.0 - keep_pct))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5])
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="otsu 模式=阈值缩放系数；percentile 模式=保留百分比")
    ap.add_argument("--mode", choices=["percentile", "otsu"], default="percentile",
                    help="阈值方式：percentile=留出最像血管的 x%%（更适合不平衡掩膜），otsu=Otsu")
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--sweep", action="store_true", help="扫一遍阈值缩放系数，报最好的")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--save-masks", default=None)
    args = ap.parse_args()

    train_items, test_items = discover(args.data)
    train = load_set(train_items)
    test = load_set(test_items, with_mask=False)

    def vessel_prob(sample) -> np.ndarray:
        return frangi_map(sample.rgb[..., 1], tuple(args.sigmas), beta=args.beta)

    print(f"[baseline] Frangi sigmas={args.sigmas} beta={args.beta}（{len(train)} 张训练图）")
    probs = [vessel_prob(s) for s in train]

    def thr_of(p: np.ndarray, fov: np.ndarray, keep_pct: float) -> float:
        if args.mode == "otsu":
            return otsu_threshold(p, fov) * keep_pct
        return percentile_threshold(p, fov, keep_pct)

    if args.mode == "otsu":
        scales = [round(x, 2) for x in np.arange(0.1, 1.01, 0.1)] if args.sweep else [args.scale]
    else:
        scales = [float(x) for x in range(4, 26, 2)] if args.sweep else [args.scale]
    best = None
    for sc in scales:
        dices, sens, spes = [], [], []
        for p, s in zip(probs, train):
            t = thr_of(p, s.fov, sc)
            m = postprocess(p, s.fov, thr=t, min_size=args.min_size)
            mt = seg_metrics(m, s.mask, s.fov)
            dices.append(mt["dice"])
            sens.append(mt["sen"])
            spes.append(mt["spe"])
        row = {"scale": sc, "dice": float(np.mean(dices)),
               "sen": float(np.mean(sens)), "spe": float(np.mean(spes))}
        if best is None or row["dice"] > best["dice"]:
            best = row
        print(f"  {args.mode}@{sc:.2f} dice={row['dice']:.4f} sen={row['sen']:.4f} spe={row['spe']:.4f}")
    print(f"[baseline] 最好 {args.mode}@{best['scale']} dice={best['dice']:.4f} "
          f"sen={best['sen']:.4f} spe={best['spe']:.4f}")

    if args.submit:
        ids, rles = [], []
        mask_dir = Path(args.save_masks) if args.save_masks else None
        if mask_dir:
            mask_dir.mkdir(parents=True, exist_ok=True)
        for s in test:
            p = vessel_prob(s)
            t = thr_of(p, s.fov, best["scale"])
            m = postprocess(p, s.fov, thr=t, min_size=args.min_size)
            m8 = (m * 255).astype(np.uint8)
            if mask_dir:
                imwrite_unicode(mask_dir / f"{s.image_id}.png", m8)
            ids.append(s.image_id)
            rles.append(mask_to_rle_512(m8))
        out = ROOT / "submissions" / "t4_baseline_frangi.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Id": ids, "Predicted": rles}).to_csv(out, index=False)
        print(f"[baseline] 提交 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
