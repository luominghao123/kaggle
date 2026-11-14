"""任务三（中心凹检测定位）训练脚本：分层 K 折 + 热图回归 + AMP + EMA。

用法
----
    # 0) 自检数据（打印标签表解析结果、坐标范围、测试图数量）
    .\\run.ps1 -m src.fovea_data data

    # 1) 冒烟（1 折、6 epoch、小图，确认全流程通）
    .\\run.ps1 -m src.fovea_train --data data --folds 1 --epochs 6 --img-size 384 --tag smoke3

    # 2) 正式：5 折、512 输入、40 epoch
    .\\run.ps1 -m src.fovea_train --data data --folds 5 --epochs 40 --img-size 512 --tag f512

产物落在 `runs/<tag>/`：
    fold{k}_best.pt          最优权重（可能来自原始权重或 EMA 权重）
    val_pred_fold{k}.npy     验证集预测（letterbox 归一化坐标，顺序 x,y）
    val_gt_fold{k}.npy       验证集真值（同上）
    val_idx_fold{k}.npy      该折验证集在原训练集中的行号
    test_fold{k}.npy         测试集预测（同上坐标制）
    fold{k}_result.json      该折逐 epoch 记录
    summary.json             总体汇总（含 MCRMSE 与像素误差）
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader

from .engine import CosineSchedulerWithWarmup, get_amp_context
from .fovea_data import (
    FoveaDataset,
    LetterboxCache,
    build_samples_from_layout,
    discover,
    letterbox_params,
    norm_to_label,
)
from .fovea_model import FoveaLoss, build_model
from .models import build_optimizer
from .utils import (AverageMeter, describe_device, get_logger, human_time,
                    save_json, seed_everything)

# 竞赛口径：Mean Columnwise RMSE = 各列 RMSE 的平均
# 两列 (x, y) 时 = (RMSE_x + RMSE_y) / 2


def mcrmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean((pred - gt) ** 2, axis=0))))


def pooled_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """把所有 x/y 值放一起算 RMSE。

    2026task3 的 sample_submission 是长表（ImageID, value），只有**一列**数值，
    此时 MCRMSE 就退化成「全部值一起算的 RMSE」——这个数才最接近榜单显示。
    """
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def radial_error_px(pred_norm, gt_norm, paths, sizes, img_size) -> np.ndarray:
    """把归一化坐标还原到原图像素，返回每张图的欧氏距离（像素）。"""
    errs = []
    for (u, v), (gu, gv), p in zip(pred_norm, gt_norm, paths):
        w, h = sizes[Path(p)]
        s, ox, oy = letterbox_params(w, h, img_size)
        px, py = norm_to_label(float(u), float(v), (w, h), "pixel", s, ox, oy, img_size)
        gx, gy = norm_to_label(float(gu), float(gv), (w, h), "pixel", s, ox, oy, img_size)
        errs.append(float(np.hypot(px - gx, py - gy)))
    return np.asarray(errs)


def build_folds(xy: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """按中心凹的粗位置分层切折，保证每折空间分布一致（小数据下比随机切更稳）。"""
    bx = np.digitize(xy[:, 0], [1 / 3, 2 / 3])
    by = np.digitize(xy[:, 1], [1 / 3, 2 / 3])
    strat = bx * 3 + by
    counts = np.bincount(strat, minlength=9)
    if counts.min() >= folds and folds > 1:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return [(tr, va) for tr, va in skf.split(np.zeros(len(xy)), strat)]
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in kf.split(np.zeros(len(xy)))]


@torch.no_grad()
def predict_loader(model, loader, device, amp_dtype="bf16") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (预测坐标 [N,2], 真值坐标 [N,2], 平均 loss 分量)。"""
    model.eval()
    ctx, _, _ = get_amp_context(device, amp_dtype)
    criterion = FoveaLoss()
    preds, gts = [], []
    loss_meter = AverageMeter()
    for images, xy, _ in loader:
        images = images.to(device, non_blocking=True)
        xy = xy.to(device, non_blocking=True)
        with ctx():
            out = model(images)
            loss, _ = criterion(out, xy)
        loss_meter.update(loss.item(), images.size(0))
        preds.append(out["coords"].float().cpu().numpy())
        gts.append(xy.float().cpu().numpy())
    return np.concatenate(preds), np.concatenate(gts), np.array([loss_meter.avg])


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, epoch,
                    *, amp_dtype="bf16", ema=None, ema_base_decay=None, grad_clip=5.0,
                    logger=None) -> dict:
    model.train()
    ctx, scaler, use_scaler = get_amp_context(device, amp_dtype)
    loss_meter = AverageMeter()
    heat_meter = AverageMeter()
    xy_meter = AverageMeter()
    n_steps = max(len(loader), 1)

    for step, (images, xy, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        xy = xy.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with ctx():
            out = model(images)
            loss, parts = criterion(out, xy)

        if use_scaler:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            if ema_base_decay is not None:
                g = (epoch - 1) * n_steps + step + 1
                ema.decay = min(ema_base_decay, (1.0 + g) / (10.0 + g))
            ema.update(model)

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        heat_meter.update(parts["heat"], bs)
        xy_meter.update(parts["xy"], bs)

    return {"loss": loss_meter.avg, "heat": heat_meter.avg, "xy": xy_meter.avg}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="中心凹定位：K 折训练")
    p.add_argument("--data", default="data")
    p.add_argument("--backbone", default="effnet_b0")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=-1, help="-1 表示跑全部折")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--out-stride", type=int, default=4)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--sigma-ratio", type=float, default=0.02)
    p.add_argument("--coord-weight", type=float, default=0.5)
    p.add_argument("--rotate", type=float, default=12.0)
    p.add_argument("--scale-min", type=float, default=0.88)
    p.add_argument("--scale-max", type=float, default=1.12)
    p.add_argument("--shift", type=float, default=0.05)
    p.add_argument("--hflip", type=float, default=0.5)
    p.add_argument("--color-jitter", type=float, default=0.25)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="fovea")
    p.add_argument("--out", default="runs")
    p.add_argument("--cache-mb", type=int, default=3072)
    p.add_argument("--limit", type=int, default=0, help="只用前 N 个训练样本（调试用）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("fovea", out_dir / "log.txt")
    logger.info(f"设备: {describe_device()}")
    logger.info(f"参数: {vars(args)}")

    # ---------------- 数据 ----------------
    layout = discover(args.data)
    logger.info("数据布局:\n" + layout.summary())

    paths, xy, sizes = build_samples_from_layout(layout, args.img_size)
    if args.limit:
        paths, xy = paths[: args.limit], xy[: args.limit]
        logger.info(f"[调试] 只使用前 {args.limit} 个样本")
    logger.info(f"训练样本 {len(paths)}，测试图片 {len(layout.test_images)}，"
                f"坐标单位 {layout.unit}")

    cache = LetterboxCache(args.img_size, args.cache_mb)
    test_ds = FoveaDataset(layout.test_images, None, args.img_size, train=False, cache=cache)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    splits = build_folds(xy, args.folds, args.seed)
    fold_ids = range(len(splits)) if args.fold < 0 else [args.fold]

    criterion = FoveaLoss(coord_weight=args.coord_weight, sigma_ratio=args.sigma_ratio)
    results = []
    t_start = time.time()

    for k in fold_ids:
        tr_idx, va_idx = splits[k]
        logger.info(f"\n{'=' * 70}\n=== Fold {k}: train {len(tr_idx)} / val {len(va_idx)}")
        t0 = time.time()

        tr_ds = FoveaDataset([paths[i] for i in tr_idx], xy[tr_idx], args.img_size,
                             train=True, rotate=args.rotate,
                             scale=(args.scale_min, args.scale_max), shift=args.shift,
                             hflip=args.hflip, color_jitter=args.color_jitter, cache=cache)
        va_ds = FoveaDataset([paths[i] for i in va_idx], xy[va_idx], args.img_size,
                             train=False, cache=cache)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=True, drop_last=len(tr_ds) > args.batch_size)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=0, pin_memory=True)

        model = build_model(args.backbone, pretrained=not args.no_pretrained,
                            head_dim=args.head_dim, out_stride=args.out_stride,
                            drop_path_rate=args.drop_path).to(device)
        optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineSchedulerWithWarmup(
            optimizer, args.lr, args.lr * 0.02, args.epochs, len(tr_loader),
            warmup_epochs=args.warmup_epochs,
        )
        ema = None if args.no_ema else ModelEmaV2(model, decay=args.ema_decay)

        best = {"score": 1e9, "epoch": -1, "src": None, "state": None}
        history = []

        for epoch in range(1, args.epochs + 1):
            tl = train_one_epoch(model, tr_loader, optimizer, scheduler, criterion, device,
                                 epoch, amp_dtype="bf16", ema=ema,
                                 ema_base_decay=None if ema is None else args.ema_decay,
                                 logger=logger)

            raw_pred, raw_gt, raw_loss = predict_loader(model, va_loader, device)
            s_raw = mcrmse(raw_pred, raw_gt)
            rec = {"epoch": epoch, "train": tl, "val_mcrmse": s_raw}

            pick_state, pick_src, pick_score = model.state_dict(), "model", s_raw
            if ema is not None:
                ema_pred, _, ema_loss = predict_loader(ema.module, va_loader, device)
                s_ema = mcrmse(ema_pred, raw_gt)
                rec.update({"val_mcrmse_ema": s_ema,
                            "val_loss": float(raw_loss[0]),
                            "val_loss_ema": float(ema_loss[0])})
                if s_ema < pick_score:
                    pick_state, pick_src, pick_score = ema.module.state_dict(), "ema", s_ema
            history.append(rec)

            if pick_score < best["score"]:
                best = {"score": pick_score, "epoch": epoch, "src": pick_src,
                        "state": {kk: vv.detach().cpu().clone() for kk, vv in pick_state.items()}}
                logger.info(f"  epoch {epoch:3d}  train_loss {tl['loss']:.4f}"
                            f" (heat {tl['heat']:.3f} / xy {tl['xy']:.4f})"
                            f"  val MCRMSE {s_raw:.5f}"
                            + (f"  EMA {rec.get('val_mcrmse_ema', float('nan')):.5f}"
                               if ema is not None else "")
                            + f"  -> best({pick_src}) {pick_score:.5f}")
            elif epoch % 5 == 0 or epoch == args.epochs:
                logger.info(f"  epoch {epoch:3d}  train_loss {tl['loss']:.4f}"
                            f"  val MCRMSE {s_raw:.5f}")

        # ---- 用最优权重出验证/测试预测 ----
        model.load_state_dict(best["state"])
        va_pred, va_gt, _ = predict_loader(model, va_loader, device)
        te_pred, _, _ = predict_loader(model, test_loader, device)

        np.save(out_dir / f"val_pred_fold{k}.npy", va_pred)
        np.save(out_dir / f"val_gt_fold{k}.npy", va_gt)
        np.save(out_dir / f"val_idx_fold{k}.npy", va_idx)
        np.save(out_dir / f"test_fold{k}.npy", te_pred)
        torch.save({"model": best["state"], "args": vars(args),
                    "best_score": best["score"], "best_epoch": best["epoch"],
                    "best_src": best["src"]}, out_dir / f"fold{k}_best.pt")

        px_err = radial_error_px(va_pred, va_gt, [paths[i] for i in va_idx], sizes, args.img_size)
        fold_res = {
            "fold": k, "val_mcrmse": best["score"], "best_epoch": best["epoch"],
            "best_src": best["src"],
            "val_pooled_rmse": pooled_rmse(va_pred, va_gt),
            "pixel_err_mean": float(px_err.mean()),
            "pixel_err_median": float(np.median(px_err)),
            "pixel_err_p90": float(np.percentile(px_err, 90)),
            "seconds": time.time() - t0, "history": history,
        }
        save_json(fold_res, out_dir / f"fold{k}_result.json")
        results.append(fold_res)
        logger.info(f"  fold {k} 完成: MCRMSE {best['score']:.5f}"
                    f"  像素误差 mean {px_err.mean():.1f} / median {np.median(px_err):.1f}"
                    f"  用时 {human_time(time.time() - t0)}")

    # ---------------- 汇总（只统计本次跑过的折）----------------
    summary = {
        "args": vars(args),
        "layout": {"labels_csv": str(layout.labels_csv),
                   "id_column": layout.id_column, "x_column": layout.x_column,
                   "y_column": layout.y_column, "unit": layout.unit,
                   "n_train": len(paths), "n_test": len(layout.test_images),
                   "submission_columns": layout.submission_columns},
        "folds": results,
        "n_folds": len(results),
        "mean_mcrmse": float(np.mean([r["val_mcrmse"] for r in results])),
        "mean_pooled_rmse": float(np.mean([r["val_pooled_rmse"] for r in results])),
        "mean_pixel_err": float(np.mean([r["pixel_err_mean"] for r in results])),
        "elapsed_seconds": time.time() - t_start,
    }
    save_json(summary, out_dir / "summary.json")

    # ---- OOF 汇总（本次跑过的所有折拼起来）----
    rows = []
    for k in [r["fold"] for r in results]:
        pred = np.load(out_dir / f"val_pred_fold{k}.npy")
        gt = np.load(out_dir / f"val_gt_fold{k}.npy")
        idx = np.load(out_dir / f"val_idx_fold{k}.npy")
        for j, i in enumerate(idx):
            rows.append((str(paths[i]), *gt[j].tolist(), *pred[j].tolist()))
    if rows:
        import csv
        with open(out_dir / "oof.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "gt_x", "gt_y", "pred_x", "pred_y"])
            w.writerows(rows)

    logger.info(f"\n{'=' * 70}\n全部完成，用时 {human_time(time.time() - t_start)}")
    logger.info(f"平均 MCRMSE {summary['mean_mcrmse']:.5f}"
                f"  平均 pooled-RMSE {summary['mean_pooled_rmse']:.5f}"
                f"  平均像素误差 {summary['mean_pixel_err']:.1f}px")
    logger.info(f"缓存状态: {cache.info()}")
    logger.info(f"产物目录: {out_dir}")
    logger.info(f"下一步: .\\run.ps1 -m src.fovea_predict --data {args.data} "
                f"--runs {out_dir} --out submissions/{args.tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
