"""任务二：分层 K 折训练（ConvNeXt/ResNet 等 timm 骨干 + 迁移学习）。

设计要点（对应作业里的「解决思路」）：
1. 数据集构建：`StratifiedKFold` 保证每折的类别比例与全量一致，并产出 out-of-fold
   概率，用于无偏地估计集成权重；同时给出比单次划分更稳的性能估计。
2. 特征工程：不手工设计特征，直接用 ImageNet-22k 预训练骨干做迁移学习，
   48x48 灰度图放大到 128 或 160 再喂给网络（预训练特征的尺度更匹配）。
3. 数据增强：小图 + 类别不平衡，过拟合风险高，因此开启随机裁剪/仿射/翻转/Cutout/
   Mixup-CutMix，最后若干 epoch 关闭 Mixup 让模型回到真实分布。
4. 训练技巧：bf16 AMP、带预热的余弦学习率、label smoothing、EMA 权重、
   drop_path 随机深度；每轮同时评估原始权重与 EMA 权重，取更优者。
5. 之后由 `src_fer/predict.py` 做多折 + 多骨干 + TTA 集成。

用法：
    .\\run.ps1 -m src_fer.train --data data/raw/2026task2 --model convnext_tiny `
        --img-size 128 --epochs 40 --folds 5 --batch-size 128 --tag cxt128
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from src.utils import (AverageMeter, describe_device, get_logger, human_time,
                       save_json, seed_everything)

from . import data as D
from .aug import mixup_cutmix, one_hot, prep_eval, random_batch, soft_cross_entropy
from .infer import predict_probs
from .models import build_model, build_optimizer
from .utils_fer import ModelEMA, cosine_lr, report, set_lr


# --------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="任务二 FER 训练")
    p.add_argument("--data", default="data/raw/2026task2", help="数据根目录（自动探测）")
    p.add_argument("--cache", default=None, help="图像缓存目录，默认 <root>/_cache48")
    p.add_argument("--out", default=None, help="输出目录，默认 runs_fer/<tag>")
    p.add_argument("--tag", default="fer", help="实验名")

    p.add_argument("--model", default="convnext_tiny", help="骨干别名或 timm 全名")
    p.add_argument("--img-size", type=int, default=128, help="网络输入边长（48 -> 放大）")
    p.add_argument("--no-pretrained", action="store_true", help="不用预训练权重（对照实验）")
    p.add_argument("--drop-path", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.2)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--no-mix-last", type=int, default=5, help="最后 N 个 epoch 关闭 mixup")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=5.0)

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=-1, help="-1 表示跑全部折")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tta", type=int, default=1, help="保存测试概率时用的 TTA 视角数")
    p.add_argument("--limit-frac", type=float, default=1.0, help="只用一部分训练数据（调试）")
    p.add_argument("--log-interval", type=int, default=50)
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# 评估
# --------------------------------------------------------------------------


@torch.no_grad()
def eval_probs(model, images: torch.Tensor, idx: np.ndarray, img_size: int,
               batch_size: int, device: torch.device, amp_dtype: str = "bf16") -> np.ndarray:
    """对给定下标做无 TTA 的推理，返回概率（用于每轮验证与 OOF）。"""
    model.eval()
    out = []
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if amp_dtype == "bf16"
           else torch.autocast("cuda", dtype=torch.float16))
    for s in range(0, len(idx), batch_size):
        sel = idx[s:s + batch_size]
        x = prep_eval(images[torch.as_tensor(sel, device=device)], img_size)
        with ctx:
            logits = model(x)
        out.append(torch.softmax(logits.float(), dim=1).cpu())
    return torch.cat(out).numpy()


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(1) == labels).mean())


# --------------------------------------------------------------------------
# 单折训练
# --------------------------------------------------------------------------


def run_fold(args, ds: D.FerdDataset, fold: int, tr_idx: np.ndarray, va_idx: np.ndarray,
             out_dir: Path, device: torch.device, logger, n_classes: int) -> dict:
    logger.info(f"===== fold {fold}  训练 {len(tr_idx)} / 验证 {len(va_idx)} =====")

    model = build_model(args.model, n_classes, img_size=args.img_size,
                        pretrained=not args.no_pretrained,
                        drop_path_rate=args.drop_path).to(device)
    ema = ModelEMA(model, args.ema_decay)
    opt = build_optimizer(model, lr=args.lr, weight_decay=args.wd)

    # 整份训练数据常驻显存（uint8，65 MB），增强在 GPU 上做
    images = torch.from_numpy(np.ascontiguousarray(ds.train_images)).to(device)
    labels_all = torch.from_numpy(ds.train_labels).to(device)
    va_labels = ds.train_labels[va_idx]

    steps_per_epoch = math.ceil(len(tr_idx) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    best = {"acc": -1.0, "src": "raw", "epoch": -1}
    best_state = None
    history: list[dict] = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(tr_idx)
        use_mix = (args.mixup_prob > 0) and (epoch <= args.epochs - args.no_mix_last)
        meter = AverageMeter()
        t0 = time.time()

        for step in range(steps_per_epoch):
            sel = perm[step * args.batch_size:(step + 1) * args.batch_size]
            if len(sel) == 0:
                break
            gstep = (epoch - 1) * steps_per_epoch + step
            set_lr(opt, cosine_lr(gstep, total_steps, args.lr, args.lr * 0.01, warmup_steps))

            sel_t = torch.as_tensor(sel, device=device)
            xb, yb = images[sel_t], labels_all[sel_t]
            x = random_batch(xb, yb, args.img_size)
            y = one_hot(yb, n_classes, args.label_smoothing)
            if use_mix:
                x, y = mixup_cutmix(x, y, mixup_alpha=args.mixup,
                                    cutmix_alpha=args.cutmix, prob=args.mixup_prob)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = soft_cross_entropy(model(x), y)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            ema.update(model, gstep + 1)
            meter.update(loss.item(), len(sel))

            if args.log_interval and (step + 1) % args.log_interval == 0:
                logger.info(f"  epoch {epoch:3d} step {step+1}/{steps_per_epoch} "
                            f"loss={meter.avg:.4f} lr={opt.param_groups[0]['lr']:.2e}")

        # ---- 每轮验证：原始权重 vs EMA 权重 ----
        p_raw = eval_probs(model, images, va_idx, args.img_size, 512, device)
        p_ema = eval_probs(ema.module, images, va_idx, args.img_size, 512, device)
        acc_raw, acc_ema = accuracy(p_raw, va_labels), accuracy(p_ema, va_labels)
        src = "ema" if acc_ema >= acc_raw else "raw"
        acc = max(acc_raw, acc_ema)
        history.append({"epoch": epoch, "loss": meter.avg, "acc_raw": acc_raw,
                        "acc_ema": acc_ema, "mix": bool(use_mix)})
        logger.info(f"  epoch {epoch:3d}/{args.epochs}  loss={meter.avg:.4f}  "
                    f"acc_raw={acc_raw:.4f}  acc_ema={acc_ema:.4f}  "
                    f"耗时 {human_time(time.time()-t0)}")

        if acc > best["acc"]:
            best = {"acc": acc, "src": src, "epoch": epoch}
            src_model = ema.module if src == "ema" else model
            best_state = {k: v.detach().cpu().clone() for k, v in src_model.state_dict().items()}
            logger.info(f"  * 新的最佳：{acc:.4f}（{src} 权重, epoch {epoch}）")

    # ---- 折结束：用最佳权重产出 OOF 与测试概率 ----
    assert best_state is not None
    src_model = ema.module if best["src"] == "ema" else model
    src_model.load_state_dict(best_state)
    src_model.to(device)

    val_probs = eval_probs(src_model, images, va_idx, args.img_size, 512, device)
    logger.info(f"fold {fold} 最佳验证准确率 = {accuracy(val_probs, va_labels):.4f} "
                f"(epoch {best['epoch']}, {best['src']} 权重)")
    logger.info(report(va_labels, val_probs.argmax(1), D.CLASSES))

    test_images = torch.from_numpy(np.ascontiguousarray(ds.test_images)).to(device)
    test_probs = predict_probs(src_model, test_images, img_size=args.img_size,
                               batch_size=256, tta=args.tta)
    logger.info(f"fold {fold} 测试集概率完成（TTA 视角 {args.tta}）")

    np.save(out_dir / f"test_probs_fold{fold}.npy", test_probs.astype(np.float32))
    np.save(out_dir / f"val_probs_fold{fold}.npy", val_probs.astype(np.float32))
    np.save(out_dir / f"val_labels_fold{fold}.npy", va_labels)
    np.save(out_dir / f"val_idx_fold{fold}.npy", va_idx)
    torch.save({"state_dict": best_state, "best": best, "args": vars(args)},
               out_dir / f"fold{fold}_best.pt")
    save_json({"best": best, "history": history, "args": vars(args),
               "fold": fold, "n_train": len(tr_idx), "n_val": len(va_idx),
               "final_val_acc": accuracy(val_probs, va_labels),
               "elapsed": time.time() - t_start},
              out_dir / f"fold{fold}_result.json")

    del model, ema, src_model
    torch.cuda.empty_cache()
    return {"fold": fold, "best_acc": best["acc"], "src": best["src"],
            "final_acc": accuracy(val_probs, va_labels), "elapsed": time.time() - t_start}


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tag = args.tag or args.model
    out_dir = Path(args.out) if args.out else Path("runs_fer") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(f"fer-{tag}", out_dir / "log.txt")
    logger.info(f"设备: {describe_device()}")
    logger.info(f"参数: {json.dumps(vars(args), ensure_ascii=False)}")

    ds = D.FerdDataset(args.data, args.cache)
    logger.info("\n" + ds.describe())
    if ds.sample_submission:
        logger.info(f"样例提交列名与顺序以 {ds.sample_submission} 为准")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(D.CLASSES)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(skf.split(np.zeros(len(ds.train_labels)), ds.train_labels))

    results = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        if args.fold >= 0 and fold != args.fold:
            continue
        if args.limit_frac < 1.0:
            rng = np.random.default_rng(args.seed)
            keep = rng.random(len(tr_idx)) < args.limit_frac
            tr_idx = tr_idx[keep]
        results.append(run_fold(args, ds, fold, tr_idx, va_idx, out_dir, device,
                                logger, n_classes))

    save_json({"args": vars(args), "folds": results}, out_dir / "summary.json")
    logger.info("全部折完成:")
    for r in results:
        logger.info(f"  fold {r['fold']}: best={r['best_acc']:.4f} ({r['src']}) "
                    f"耗时 {human_time(r['elapsed'])}")
    if results:
        mean_acc = float(np.mean([r["final_acc"] for r in results]))
        logger.info(f"各折最终 OOF 准确率均值 = {mean_acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
