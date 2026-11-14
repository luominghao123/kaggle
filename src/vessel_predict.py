"""实验四：血管分割推理 / 集成 / 阈值调参 / 生成提交。

三个模式
--------
1) **调参**（只看 OOF，不碰测试集）：
   `--tune` 把若干 run 的 `oof.npz` 平均起来，在训练集 20 张上网格搜索
   阈值 / 最小连通域 / 填洞，打印「全分辨率 Dice」与「走官方 512 RLE 管线的 Dice」。
2) **评估**：`--eval` 用当前 runs 在训练集上重算 OOF 指标（多骨干集成是否真的有收益）。
3) **提交**：默认模式，对 test 20 张出概率 -> 最优后处理 -> 512x512 RLE -> csv。

为什么要同时看两个 Dice
----------------------
官方 `segmentation_to_csv.py` 会先把掩膜 resize 到 512x512 双线性、再 `>0` 二值化，
这等于给定掩膜做了一次半像素级膨胀。如果评测方对 GT 走同一条管线，
那么本地用**同一条管线**算出的 Dice 才和榜单口径一致；全分辨率 Dice 则是
模型真实质量。两个数都打出来，避免自欺欺人。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .vessel_data import (DEFAULT_DATA, discover, imwrite_unicode, load_set,
                          mask_to_rle_512, rle_to_mask)
from .vessel_model import build_model, postprocess, predict_probs, seg_metrics

ROOT = Path(__file__).resolve().parents[1]


def load_models(run_dirs: list[Path], device: torch.device) -> list[dict]:
    models = []
    for rd in run_dirs:
        ckpts = sorted(rd.glob("fold*_best.pt"))
        if not ckpts:
            raise FileNotFoundError(f"{rd} 下没有 fold*_best.pt")
        for ck in ckpts:
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            m = build_model(blob["backbone"], pretrained=False).to(device)
            m.load_state_dict(blob["state_dict"])
            m.eval()
            models.append({"model": m, "fold": blob.get("fold"), "run": rd.name,
                           "backbone": blob.get("backbone"), "val_dice": blob.get("val_dice"),
                           "val_ids": set(blob.get("val_ids") or [])})
    return models


def probs_for(models: list[dict], sample, tta: tuple[str, ...], scales: tuple[float, ...] = (1.0,),
              oof_safe: bool = False) -> np.ndarray:
    """多模型 × 翻转 TTA × 多尺度 平均概率。

    ``oof_safe=True`` 时只用「没见过这张图」的折模型（即该图属于它的验证折），
    否则在训练集上评估会泄漏 —— 阈值调参必须走这条路径。
    """
    if oof_safe:
        sub = [m for m in models if sample.image_id in m["val_ids"]]
        if sub:
            models = sub
    x0 = sample.normalized().transpose(2, 0, 1)
    acc = None
    n = 0
    for m in models:
        for sc in scales:
            x = x0
            if abs(sc - 1.0) > 1e-6:
                h, w = x0.shape[1:]
                nh, nw = int(round(h * sc)), int(round(w * sc))
                x = np.stack([cv2.resize(x0[c], (nw, nh), interpolation=cv2.INTER_LINEAR)
                              for c in range(x0.shape[0])])
            p = predict_probs(m["model"], x, tta_flips=tta)
            if abs(sc - 1.0) > 1e-6:
                p = cv2.resize(p, (x0.shape[2], x0.shape[1]), interpolation=cv2.INTER_LINEAR)
            acc = p if acc is None else acc + p
            n += 1
    return acc / n


def dice_512(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """按官方提交管线（512 双线性 + >0）算 Dice。"""
    a = rle_to_mask(mask_to_rle_512((pred_mask * 255).astype(np.uint8)))
    b = rle_to_mask(mask_to_rle_512((gt_mask * 255).astype(np.uint8)))
    inter = float((a & b).sum())
    return 2 * inter / max(1.0, float(a.sum() + b.sum()))


def grid_search(oof: np.ndarray, samples, thrs, min_sizes, fills, dilates) -> list[dict]:
    rows = []
    for thr in thrs:
        for ms in min_sizes:
            for fh in fills:
                for dl in dilates:
                    d_full, d_512, sen, spe = [], [], [], []
                    for i, s in enumerate(samples):
                        m = postprocess(oof[i], s.fov, thr=thr, min_size=ms, fill_holes=fh, dilate=dl)
                        mt = seg_metrics(m, s.mask, s.fov)
                        d_full.append(mt["dice"])
                        sen.append(mt["sen"])
                        spe.append(mt["spe"])
                        d_512.append(dice_512(m, s.mask))
                    rows.append({"thr": thr, "min_size": ms, "fill": bool(fh), "dilate": dl,
                                 "dice": float(np.mean(d_full)), "dice512": float(np.mean(d_512)),
                                 "sen": float(np.mean(sen)), "spe": float(np.mean(spe))})
    rows.sort(key=lambda r: r["dice512"], reverse=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="实验四 血管分割推理 / 提交")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--runs", nargs="+", required=True, help="一个或多个 runs/<tag> 目录")
    ap.add_argument("--out", default="submissions/t4_vessel.csv")
    ap.add_argument("--tta", default="h,v,hv",
                    help='翻转 TTA，默认 "h,v,hv"（含原图共 4 种）；给空串表示不 TTA')
    ap.add_argument("--scales", default="1.0", help="多尺度推理，例如 0.9,1.0,1.1")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-size", type=int, default=0)
    ap.add_argument("--fill-holes", action="store_true")
    ap.add_argument("--dilate", type=int, default=0)
    ap.add_argument("--tune", action="store_true", help="用 OOF 网格搜索阈值/后处理")
    ap.add_argument("--tune-probs", default=None,
                    help="用指定 npz（含 probs/ids）里的概率调参，而不是 run 里的 oof.npz")
    ap.add_argument("--eval", action="store_true", help="在训练集上重算 OOF 指标（无泄漏）")
    ap.add_argument("--skip-predict", action="store_true",
                    help="只做调参/评估，不跑测试集、不写提交")
    ap.add_argument("--eval-probs", default="runs/_tune/oof_eval_probs.npz",
                    help="--eval 时把 (无泄漏) OOF 概率存到这里，供 --tune-probs 使用")
    ap.add_argument("--save-masks", default=None, help="把预测掩膜写到该目录（png）")
    ap.add_argument("--save-probs", default=None, help="把测试概率图存成 npz（集成/再调参用）")
    ap.add_argument("--clahe-clip", type=float, default=2.5)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = [ROOT / r if not Path(r).is_absolute() else Path(r) for r in args.runs]
    tta = tuple(t.strip() for t in args.tta.split(",") if t.strip())
    scales = tuple(float(s) for s in args.scales.replace(" ", ",").split(",") if s.strip())

    train_items, test_items = discover(args.data)
    train = load_set(train_items, clip=args.clahe_clip)
    test = load_set(test_items, clip=args.clahe_clip, with_mask=False)
    print(f"[data] 训练 {len(train)} / 测试 {len(test)}；runs={[r.name for r in run_dirs]} tta={tta}")

    # ---------------- 1) OOF 网格搜索 ---------------- #
    thr, min_size, fill, dilate = args.thr, args.min_size, args.fill_holes, args.dilate
    if args.tune:
        if args.tune_probs:
            z = np.load(ROOT / args.tune_probs if not Path(args.tune_probs).is_absolute()
                        else args.tune_probs)
            probs = z["probs"].astype(np.float32)
            ids = [str(i) for i in z["ids"]]
            order = [ids.index(s.image_id) for s in train]
            oof = probs[order]
            src = args.tune_probs
        else:
            oofs = []
            for rd in run_dirs:
                f = rd / "oof.npz"
                if not f.exists():
                    raise FileNotFoundError(f"{f} 不存在，先跑完该 run 的所有折")
                z = np.load(f)
                if not bool(np.all(z["filled"])):
                    raise RuntimeError(f"{f} 的 OOF 没填满（缺折），不能用来调参")
                oofs.append(z["probs"].astype(np.float32))
            oof = np.mean(oofs, 0)
            src = " + ".join(r.name for r in run_dirs)
        print(f"[tune] 概率来源：{src}（{oof.shape[0]} 张训练图）")
        rows = grid_search(oof, train,
                           thrs=[round(x, 2) for x in np.arange(0.3, 0.81, 0.05)],
                           min_sizes=[0, 20, 50],
                           fills=[False, True],
                           dilates=[0, 1])
        print("[tune] dice512 前 10 名：")
        for r in rows[:10]:
            print(f"   thr={r['thr']:.2f} min_size={r['min_size']:3d} fill={int(r['fill'])} "
                  f"dilate={r['dilate']} dice={r['dice']:.4f} dice512={r['dice512']:.4f} "
                  f"sen={r['sen']:.4f} spe={r['spe']:.4f}")
        best = rows[0]
        thr, min_size, fill, dilate = best["thr"], best["min_size"], best["fill"], best["dilate"]
        print(f"[tune] 采用 thr={thr} min_size={min_size} fill={fill} dilate={dilate} "
              f"-> dice {best['dice']:.4f} / dice512 {best['dice512']:.4f}")
        (ROOT / "runs" / "_tune").mkdir(parents=True, exist_ok=True)
        (ROOT / "runs" / "_tune" / "grid.json").write_text(
            json.dumps({"source": src, "runs": [r.name for r in run_dirs],
                        "tta": list(tta), "scales": list(scales), "rows": rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 2) 训练集 OOF 复核（无泄漏） ---------------- #
    if args.eval:
        models = load_models(run_dirs, device)
        print(f"[eval] {len(models)} 个权重；逐图只用『没见过该图』的折模型 "
              f"TTA={tta} scales={scales}")
        per_img, probs = [], []
        for s in train:
            p = probs_for(models, s, tta, scales=scales, oof_safe=True)
            probs.append(p)
            m = postprocess(p, s.fov, thr=thr, min_size=min_size, fill_holes=fill, dilate=dilate)
            mt = seg_metrics(m, s.mask, s.fov)
            mt["id"] = s.image_id
            mt["dice512"] = dice_512(m, s.mask)
            per_img.append(mt)
        keys = ("dice", "dice512", "iou", "sen", "spe", "acc")
        mean = {k: float(np.mean([d[k] for d in per_img])) for k in keys}
        print("[eval] 平均 " + " ".join(f"{k} {v:.4f}" for k, v in mean.items()))
        worst = sorted(per_img, key=lambda d: d["dice"])[:5]
        print("[eval] 最差 5 张：" + ", ".join(f"{d['id']}({d['dice']:.3f})" for d in worst))
        out = ROOT / "runs" / "_tune" / "eval_oof.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"mean": mean, "per_image": per_img,
                                   "tta": list(tta), "scales": list(scales),
                                   "post": {"thr": thr, "min_size": min_size,
                                            "fill": bool(fill), "dilate": dilate}},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        if args.eval_probs:
            ep = ROOT / args.eval_probs
            ep.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ep, probs=np.stack(probs).astype(np.float16),
                                ids=np.array([s.image_id for s in train]))
            print(f"[eval] 概率 -> {args.eval_probs}（可传给 --tune-probs 调参）")
        print(f"[eval] -> {out}")

    # ---------------- 3) 测试集推理 + 写提交 ---------------- #
    if args.skip_predict:
        print("[skip] --skip-predict：不跑测试集")
        return 0
    models = load_models(run_dirs, device)
    print(f"[predict] 集成 {len(models)} 个权重 -> {len(test)} 张测试图")
    t0 = time.time()
    ids, rles, mask_dir = [], [], Path(args.save_masks) if args.save_masks else None
    if mask_dir:
        mask_dir.mkdir(parents=True, exist_ok=True)
    probs_store = {}
    for s in test:
        p = probs_for(models, s, tta, scales=scales)
        probs_store[s.image_id] = p.astype(np.float16)
        m = postprocess(p, s.fov, thr=thr, min_size=min_size, fill_holes=fill, dilate=dilate)
        m8 = (m * 255).astype(np.uint8)
        if mask_dir:
            imwrite_unicode(mask_dir / f"{s.image_id}.png", m8)
        ids.append(s.image_id)
        rles.append(mask_to_rle_512(m8))
        print(f"   {s.image_id}: 血管占比 {m.mean():.4f} RLE 段数 {len(rles[-1].split()) // 2}")
    print(f"[predict] 用时 {(time.time() - t0) / 60:.1f} min")

    if args.save_probs:
        np.savez_compressed(ROOT / args.save_probs, ids=np.array(ids),
                            **{k: v for k, v in probs_store.items()})
        print(f"[predict] 概率图 -> {args.save_probs}")

    sub = pd.DataFrame({"Id": ids, "Predicted": rles})
    # 按数字 id 排序，和 sample_submission.csv 的顺序一致（1,2,...,20）
    sub["_k"] = sub["Id"].map(lambda s: (0, int(s)) if str(s).isdigit() else (1, 0))
    sub = sub.sort_values("_k").drop(columns="_k").reset_index(drop=True)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    meta = {
        "out": args.out,
        "runs": [r.name for r in run_dirs],
        "weights": [f"{m['run']}/fold{m['fold']}" for m in models],
        "tta": list(tta),
        "scales": list(scales),
        "post": {"thr": thr, "min_size": min_size, "fill_holes": bool(fill), "dilate": dilate},
        "positive_ratio": {k: float(v.astype(np.float32).__ge__(thr).mean()) for k, v in probs_store.items()},
        "note": "Predicted = 官方 segmentation_to_csv.py 同口径的 512x512 行优先 RLE",
    }
    (out_path.with_suffix(".json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"[done] {out_path}（{len(sub)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
