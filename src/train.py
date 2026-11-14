"""训练入口：分层 K 折微调。

用法示例
--------
# 完整 5 折训练 ConvNeXt-Tiny @384
python -m src.train --data data --model convnext_tiny --img-size 384 --epochs 30 --folds 5

# 只跑第 0 折快速验证流程
python -m src.train --data data --model convnext_tiny --img-size 224 --epochs 8 --fold 0

产物（默认写到 runs/<tag>/）
---------------------------
    fold{k}_best.pt     验证集指标最好的权重（含 EMA 权重）
    fold{k}_last.pt     最后一个 epoch 的权重
    oof.npz             交叉验证的 out-of-fold 概率与真实标签
    test_fold{k}.npy    该折模型对测试集的概率（供 predict.py 集成）
    log.txt             训练日志
    summary.json        每折指标汇总
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# 允许 `python src/train.py` 与 `python -m src.train` 两种跑法
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data import (  # type: ignore
        build_eval_transform,
        build_train_transform,
        discover,
        list_test_samples,
        list_train_samples,
        PlantDataset,
    )
    from src.engine import (  # type: ignore
        CosineSchedulerWithWarmup,
        compute_metrics,
        per_class_report,
        train_one_epoch,
        validate,
    )
    from src.models import build_model, build_optimizer, resolve_name  # type: ignore
    from src.utils import (  # type: ignore
        count_params,
        describe_device,
        get_logger,
        human_time,
        load_json,
        save_json,
        seed_everything,
    )
else:
    from .data import (
        build_eval_transform,
        build_train_transform,
        discover,
        list_test_samples,
        list_train_samples,
        PlantDataset,
    )
    from .engine import (
        CosineSchedulerWithWarmup,
        compute_metrics,
        per_class_report,
        train_one_epoch,
        validate,
    )
    from .models import build_model, build_optimizer, resolve_name
    from .utils import (
        count_params,
        describe_device,
        get_logger,
        human_time,
        load_json,
        save_json,
        seed_everything,
    )


# --------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="植物分类 - 迁移学习训练")

    # 数据
    p.add_argument("--data", default="data", help="数据根目录")
    p.add_argument("--out", default="runs", help="输出根目录")
    p.add_argument("--tag", default=None, help="实验名，默认 模型_尺寸_时间戳")

    # 模型
    p.add_argument("--model", default="convnext_tiny", help="模型别名或 timm 全名")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--drop-rate", type=float, default=0.0)
    p.add_argument("--no-pretrained", action="store_true", help="从零训练（消融实验用）")

    # 训练
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.8)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--mixup-off-epochs", type=int, default=3,
                   help="最后 N 个 epoch 关闭 mixup，让模型收敛到真实分布")
    p.add_argument("--ema-decay", type=float, default=0.9998)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])

    # 验证
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=-1, help="-1 表示跑全部折")
    p.add_argument("--skip-existing", action="store_true",
                   help="已存在 fold{k}_result.json 的折直接跳过（用于先跑单折拿提交、再补全其余折）")
    p.add_argument("--seed", type=int, default=42)

    # 其它
    # 注意：本机沙箱禁止命名管道，DataLoader 的 num_workers>0 会直接报 WinError 5，
    # 因此默认 0（主进程加载）。数据量小，配合内存缓存完全够用。
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--no-cache", action="store_true", help="关闭解码图内存缓存")
    p.add_argument("--cache-max-side", type=int, default=576,
                   help="缓存前把长边限制到该值，控制内存占用")
    p.add_argument("--cache-budget-mb", type=int, default=2048, help="缓存内存上限")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--cache-dir", default=None, help="预留：缓存目录")

    return p.parse_args(argv)


# --------------------------------------------------------------------------
# 单折训练
# --------------------------------------------------------------------------


def run_fold(
    args,
    fold: int,
    train_paths: list[str],
    train_labels: list[int],
    val_paths: list[str],
    val_labels: list[int],
    test_paths: list[str],
    classes: list[str],
    out_dir: Path,
    logger,
) -> dict:
    num_classes = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = build_train_transform(args.img_size)
    eval_tf = build_eval_transform(args.img_size)

    ds_kwargs = dict(
        cache=not args.no_cache,
        cache_budget_mb=args.cache_budget_mb,
        cache_max_side=args.cache_max_side,
    )
    train_ds = PlantDataset(train_paths, train_labels, train_tf, **ds_kwargs)
    val_ds = PlantDataset(val_paths, val_labels, eval_tf, **ds_kwargs)
    test_ds = PlantDataset(test_paths, None, eval_tf, **ds_kwargs)

    common = dict(
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        # 单进程时 persistent_workers 必须为 False
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_ds) > args.batch_size, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, **common)

    logger.info(
        f"[fold {fold}] 训练集 {len(train_ds)} 张 / 验证集 {len(val_ds)} 张 / "
        f"测试集 {len(test_ds)} 张"
    )

    model = build_model(
        args.model, num_classes,
        img_size=args.img_size,
        pretrained=not args.no_pretrained,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path,
        verbose=True,
    ).to(device)

    optimizer = build_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay, opt=args.optimizer
    )
    niter_per_ep = len(train_loader)
    scheduler = CosineSchedulerWithWarmup(
        optimizer, args.lr, args.min_lr, args.epochs, niter_per_ep,
        warmup_epochs=args.warmup_epochs,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Mixup / CutMix（timm 实现，输出软标签）
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        from timm.data import Mixup

        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            prob=args.mixup_prob,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=args.label_smoothing,
            num_classes=num_classes,
        )

    ema = None
    if not args.no_ema:
        from timm.utils import ModelEmaV2

        ema = ModelEmaV2(model, decay=args.ema_decay)

    best = {"f1_macro": -1.0, "acc": 0.0, "epoch": -1}
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # 最后若干 epoch 关闭 mixup，让模型对齐真实数据分布
        use_mix = mixup_fn is not None and epoch <= args.epochs - args.mixup_off_epochs

        ep_t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch,
            amp_dtype=args.amp, mixup_fn=mixup_fn if use_mix else None,
            ema=ema, ema_base_decay=None if ema is None else args.ema_decay,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval, logger=logger,
        )

        # 同时评估「当前权重」与「EMA 权重」，取更好的那个
        va = validate(model, val_loader, criterion, device, num_classes,
                      amp_dtype=args.amp)
        va_ema = None
        if ema is not None:
            va_ema = validate(ema.module, val_loader, criterion, device, num_classes,
                              amp_dtype=args.amp)

        pick = va
        pick_src = "raw"
        if va_ema is not None and va_ema["f1_macro"] > va["f1_macro"]:
            pick, pick_src = va_ema, "ema"

        history.append({
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "val_f1_macro": va["f1_macro"],
            "val_f1_macro_ema": None if va_ema is None else va_ema["f1_macro"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - ep_t0,
        })

        logger.info(
            f"[fold {fold}] ep {epoch:3d}/{args.epochs}  "
            f"loss {tr['loss']:.4f}  "
            f"val_acc {va['acc']:.4f}  val_F1 {va['f1_macro']:.4f}"
            + (f"  (EMA {va_ema['acc']:.4f} / {va_ema['f1_macro']:.4f})" if va_ema else "")
            + f"  lr {optimizer.param_groups[0]['lr']:.2e}  {time.time()-ep_t0:.0f}s"
        )

        if pick["f1_macro"] > best["f1_macro"]:
            best = {
                "f1_macro": pick["f1_macro"],
                "acc": pick["acc"],
                "epoch": epoch,
                "src": pick_src,
            }
            ckpt = {
                "model": model.state_dict(),
                "ema": None if ema is None else ema.module.state_dict(),
                "best_src": pick_src,
                "epoch": epoch,
                "metrics": {k: v for k, v in pick.items() if not isinstance(v, np.ndarray)},
                "classes": classes,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / f"fold{fold}_best.pt")
            logger.info(
                f"    -> 保存最佳权重 (epoch {epoch}, {pick_src}, "
                f"F1={pick['f1_macro']:.4f}, acc={pick['acc']:.4f})"
            )

    # 用最佳权重重算验证集与测试集概率
    ckpt_path = out_dir / f"fold{fold}_best.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        weights = state["ema"] if state["best_src"] == "ema" and state["ema"] else state["model"]
        model.load_state_dict(weights)

    val_out = validate(model, val_loader, criterion, device, num_classes,
                       amp_dtype=args.amp)
    test_out = validate(model, test_loader, criterion, device, num_classes,
                        amp_dtype=args.amp)

    np.save(out_dir / f"test_fold{fold}.npy", test_out["probs"])
    np.save(out_dir / f"val_probs_fold{fold}.npy", val_out["probs"])
    np.save(out_dir / f"val_labels_fold{fold}.npy", val_out["labels"])
    torch.save({"model": model.state_dict(), "classes": classes, "epoch": args.epochs},
               out_dir / f"fold{fold}_last.pt")

    logger.info(
        f"[fold {fold}] 完成  最佳 epoch={best['epoch']}  用时 {human_time(time.time()-t0)}\n"
        + per_class_report(val_out["labels"], val_out["probs"].argmax(1), classes)
    )

    result = {
        "fold": fold,
        "best_epoch": best["epoch"],
        "best_src": best.get("src"),
        "best_val_f1_macro": best["f1_macro"],
        "best_val_acc": best["acc"],
        "final_val": {k: v for k, v in val_out.items() if not isinstance(v, np.ndarray)},
        "history": history,
        "seconds": time.time() - t0,
    }
    # 落盘单折结果，供 --skip-existing 复用
    save_json(result, out_dir / f"fold{fold}_result.json")
    return result


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed, deterministic=False)

    layout = discover(args.data)
    classes = layout.classes
    num_classes = len(classes)

    tag = args.tag or f"{args.model}_{args.img_size}_{time.strftime('%m%d_%H%M%S')}"
    out_dir = Path(args.out) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("plant.train", out_dir / "log.txt")
    logger.info("=" * 78)
    logger.info(f"实验: {tag}")
    logger.info(describe_device())
    logger.info(layout.summary())
    logger.info(f"参数: {vars(args)}")
    logger.info("=" * 78)

    if layout.test_dir is None:
        raise RuntimeError("未找到测试集目录，无法训练（需要 test 图片生成提交文件）")

    paths, labels = list_train_samples(layout.train_dir, layout.class_to_idx, args.seed)
    labels_arr = np.array(labels)
    test_paths = list_test_samples(layout.test_dir)

    logger.info(f"训练图片 {len(paths)} 张，测试图片 {len(test_paths)} 张")

    # 类别分布（分层划分的依据，也用于判断是否需要类别加权）
    dist = {classes[i]: int((labels_arr == i).sum()) for i in range(num_classes)}
    logger.info(f"类别分布: {dist}")
    logger.info(f"最大/最小类样本比: {max(dist.values())/max(min(dist.values()),1):.1f}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(skf.split(np.zeros(len(labels_arr)), labels_arr))

    folds = range(args.folds) if args.fold < 0 else [args.fold]

    results = []
    for k in folds:
        res_path = out_dir / f"fold{k}_result.json"
        if args.skip_existing and res_path.exists() and (out_dir / f"fold{k}_best.pt").exists():
            logger.info(f"[fold {k}] 已有产物，跳过训练（--skip-existing）")
            results.append(load_json(res_path))
            continue
        tr_idx, va_idx = splits[k]
        res = run_fold(
            args, k,
            [paths[i] for i in tr_idx], [labels[i] for i in tr_idx],
            [paths[i] for i in va_idx], [labels[i] for i in va_idx],
            test_paths, classes, out_dir, logger,
        )
        results.append(res)

    # 汇总 OOF（只覆盖跑过的折）
    oof_probs = np.zeros((len(labels_arr), num_classes), dtype=np.float32)
    oof_mask = np.zeros(len(labels_arr), dtype=bool)
    for k in folds:
        _, va_idx = splits[k]
        p = np.load(out_dir / f"val_probs_fold{k}.npy")
        oof_probs[va_idx] = p
        oof_mask[va_idx] = True

    if oof_mask.any():
        oof_metrics = compute_metrics(labels_arr[oof_mask], oof_probs[oof_mask].argmax(1),
                                      num_classes)
        np.savez(out_dir / "oof.npz", probs=oof_probs, labels=labels_arr, mask=oof_mask)
        logger.info("=" * 78)
        logger.info(f"OOF 汇总（{oof_mask.sum()} 张）: " + "  ".join(
            f"{k}={v:.4f}" for k, v in oof_metrics.items()))
        if oof_mask.all():
            logger.info(per_class_report(labels_arr, oof_probs.argmax(1), classes))

    summary = {
        "tag": tag,
        "model": args.model,
        "timm_name": resolve_name(args.model)[0],
        "img_size": args.img_size,
        "epochs": args.epochs,
        "folds_run": list(folds),
        "classes": classes,
        "train_size": len(paths),
        "test_size": len(test_paths),
        "class_distribution": dist,
        "results": results,
    }
    save_json(summary, out_dir / "summary.json")
    logger.info(f"全部完成，产物目录: {out_dir.resolve()}")
    return out_dir


if __name__ == "__main__":
    main()
