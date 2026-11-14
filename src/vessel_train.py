"""实验四：视网膜血管分割训练（K 折 + 补丁训练 + 全图验证）。

为什么是「补丁训练 / 全图验证」
------------------------------
- 训练：全图只有 565x584 且只有 20 张，整图喂进去每轮只有 20 个样本、
  显存还要为整图付账。改成从原分辨率裁 384x384 补丁（前景偏置采样，
  约一半补丁里一定有血管），一个 epoch 就能取几百个样本，
  而且**不做缩放**，1~2 px 的毛细血管不会被插值抹掉。
- 验证/推理：整图一次性前向（padding 到 32 的倍数），
  这样指标的像素单位和官方 512x512 RLE 口径一致。

产物（`runs/<tag>/`）
---------------------
```
fold{k}_best.pt     该折最优权重（raw 与 ema 里取验证 Dice 高的那个）
fold{k}_result.json 该折的超参、最优轮次、验证指标
oof.npz             全量 OOF 概率图（20 张），供阈值/后处理调参使用
summary.json        数据布局 + 全部超参 + 每折指标 + 均值
log.txt             逐轮进度（可直接写进报告）
```
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .vessel_data import DEFAULT_DATA, BatchPrefetch, PatchDataset, discover, load_set
from .vessel_model import SegLoss, build_model, mean_metrics, postprocess, predict_probs, seg_metrics

ROOT = Path(__file__).resolve().parents[1]


class EMA:
    """权重指数滑动平均（20 张图的小数据训练，EMA 通常比最后一轮更稳）。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.n = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for s, p in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                s.copy_(p)


def lr_at(step: int, total: int, base_lr: float, warmup: float = 0.05) -> float:
    w = max(1, int(total * warmup))
    if step < w:
        return base_lr * (step + 1) / w
    t = (step - w) / max(1, total - w)
    return base_lr * 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def eval_samples(model: nn.Module, samples: list, thr: float = 0.5, min_size: int = 0,
                 tta: tuple[str, ...] = ()) -> tuple[list[dict], list[np.ndarray]]:
    model.eval()
    metrics, probs = [], []
    for s in samples:
        p = predict_probs(model, s.normalized().transpose(2, 0, 1), tta_flips=tta)
        m = postprocess(p, s.fov, thr=thr, min_size=min_size)
        metrics.append(seg_metrics(m, s.mask, s.fov))
        probs.append(p.astype(np.float16))
    return metrics, probs


def make_folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return [np.sort(idx[i::k]) for i in range(k)]


def _batches_sync(ds: PatchDataset, rng: random.Random, bs: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _ in range(bs):
        x, y = ds[rng.randrange(len(ds))]
        xs.append(x)
        ys.append(y)
    return np.stack(xs), np.stack(ys)


def train_fold(args, samples, tr_idx: np.ndarray, va_idx: np.ndarray, fold: int,
               log, device: torch.device) -> dict:
    t0 = time.time()
    ds = PatchDataset(samples, indices=list(tr_idx), patch=args.patch, augment=True,
                      fg_bias=args.fg_bias, seed=args.seed + fold)
    va = [samples[i] for i in va_idx]

    model = build_model(args.backbone, pretrained=not args.no_pretrained,
                        in_chans=3, drop=args.drop).to(device)
    loss_fn = SegLoss(pos_weight=args.pos_weight if args.pos_weight > 0 else ds.pos_weight,
                      dice_w=args.dice_w).to(device)
    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * args.steps
    rng = random.Random(args.seed * 101 + fold)

    best = {"dice": -1.0, "epoch": -1, "src": "raw"}
    best_state = None
    gstep = 0
    log(f"[fold{fold}] 训练 {len(tr_idx)} 张 / 验证 {len(va_idx)} 张 "
        f"({[s.image_id for s in va]}) patch={args.patch} "
        f"epochs={args.epochs} steps={args.steps} bs={args.batch_size} "
        f"pos_weight={float(loss_fn.pos_weight):.2f}")

    for ep in range(args.epochs):
        ds.set_epoch(ep)
        model.train()
        run_loss, run_bce, run_dice = 0.0, 0.0, 0.0
        loader = BatchPrefetch(ds, args.batch_size, seed=args.seed * 7919 + fold * 131 + ep,
                               queue_size=max(1, args.prefetch)) if args.prefetch > 0 else None
        try:
            for _ in range(args.steps):
                lr = lr_at(gstep, total_steps, args.lr)
                for g in opt.param_groups:
                    g["lr"] = lr
                if loader is not None:
                    xb, yb = loader.get()
                else:
                    xb, yb = _batches_sync(ds, rng, args.batch_size)
                x = torch.from_numpy(xb).to(device, non_blocking=True)
                y = torch.from_numpy(yb).to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    logits = model(x)
                loss, parts = loss_fn(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                ema.update(model)
                run_loss += float(loss.detach())
                run_bce += parts["bce"]
                run_dice += parts["dice_loss"]
                gstep += 1
        finally:
            loader.close()

        metrics_raw, _ = eval_samples(model, va)
        metrics_ema, _ = eval_samples(ema.shadow, va)
        d_raw = float(np.mean([m["dice"] for m in metrics_raw]))
        d_ema = float(np.mean([m["dice"] for m in metrics_ema]))
        src, dice, mt = ("ema", d_ema, metrics_ema) if d_ema >= d_raw else ("raw", d_raw, metrics_raw)
        flag = ""
        if dice > best["dice"]:
            best = {"dice": dice, "epoch": ep, "src": src,
                    "sen": float(np.mean([m["sen"] for m in mt])),
                    "spe": float(np.mean([m["spe"] for m in mt])),
                    "iou": float(np.mean([m["iou"] for m in mt])), "lr": lr}
            best_state = copy.deepcopy((ema.shadow if src == "ema" else model).state_dict())
            flag = " *"
        log(f"[fold{fold}] ep{ep:03d} loss {run_loss / args.steps:.4f} "
            f"(bce {run_bce / args.steps:.4f} dice {run_dice / args.steps:.4f}) "
            f"val dice raw {d_raw:.4f} ema {d_ema:.4f} lr {lr:.2e}{flag}")

    # 用最优权重回算该折验证概率（OOF 用）
    model.load_state_dict(best_state)
    metrics, probs = eval_samples(model, va, thr=args.thr, min_size=args.min_size)
    dice = float(np.mean([m["dice"] for m in metrics]))
    ckpt = {
        "state_dict": best_state,
        "backbone": args.backbone, "patch": args.patch, "fold": fold,
        "val_ids": [s.image_id for s in va], "val_dice": dice, "src": best["src"],
    }
    torch.save(ckpt, ROOT / "runs" / args.tag / f"fold{fold}_best.pt")
    # 逐折把 OOF 概率落盘：这样训练可以分多次调用跑（前台调用有超时上限），
    # 最后由 _merge_oof() 把已有的折拼成完整 oof.npz。
    np.savez_compressed(ROOT / "runs" / args.tag / f"fold{fold}_oof.npz",
                        probs=np.stack(probs).astype(np.float16),
                        val_idx=np.asarray(va_idx), ids=np.array([s.image_id for s in va]))
    log(f"[fold{fold}] 最优 ep{best['epoch']}（{best['src']}）验证 dice {dice:.4f} "
        f"sen {np.mean([m['sen'] for m in metrics]):.4f} spe {np.mean([m['spe'] for m in metrics]):.4f} "
        f"用时 {(time.time() - t0) / 60:.1f} min")
    return {"fold": fold, "val_idx": list(map(int, va_idx)), "val_ids": [s.image_id for s in va],
            "best": best, "metrics": metrics, "probs": probs,
            "mean": {k: float(np.mean([m[k] for m in metrics])) for k in
                     ("dice", "iou", "sen", "spe", "acc")}}


def main() -> int:
    ap = argparse.ArgumentParser(description="实验四 血管分割训练")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--tag", default="v4_b0_384")
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-fold", type=int, default=-1, help="只跑某一折（调试）")
    ap.add_argument("--only-folds", default="", help="只跑指定的几折，如 0,1,2（用于分批续跑）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=40, help="每个 epoch 的优化步数")
    ap.add_argument("--patch", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--pos-weight", type=float, default=0.0, help="0 表示按数据自动算")
    ap.add_argument("--dice-w", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--fg-bias", type=float, default=0.5)
    ap.add_argument("--thr", type=float, default=0.5, help="导出 OOF 时用的阈值")
    ap.add_argument("--min-size", type=int, default=0)
    ap.add_argument("--clahe-clip", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU 线程数（0=默认）")
    ap.add_argument("--prefetch", type=int, default=3, help="预取队列长度（0 表示关掉后台线程）")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    # 注意：不要开 cudnn.benchmark。本机实测它每个新模型实例/新输入尺寸都要
    # 花 30~150 s 试算法，而 steady-state 速度和 False 完全一样（335 ms/step）。
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = ROOT / "runs" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"
    log_path.write_text("", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    train_items, test_items = discover(args.data)
    train = load_set(train_items, clip=args.clahe_clip)
    test = load_set(test_items, clip=args.clahe_clip, with_mask=False)
    log(f"[data] {args.data}")
    log(f"[data] 训练 {len(train)} 张 / 测试 {len(test)} 张；尺寸 {sorted({s.shape for s in train + test})}")
    log(f"[data] 视场内血管比例 {np.mean([s.mask[s.fov].mean() for s in train]):.4f}")

    folds = make_folds(len(train), args.folds, args.seed)
    if args.only_folds:
        fold_ids = [int(x) for x in args.only_folds.replace(" ", ",").split(",") if x.strip() != ""]
    else:
        fold_ids = list(range(args.folds)) if args.only_fold < 0 else [args.only_fold]
    log(f"[cv] {args.folds} 折，seed={args.seed}，验证集划分：" +
        "; ".join(f"f{i}={[train[j].image_id for j in folds[i]]}" for i in fold_ids))

    results = []
    for fold in fold_ids:
        va_idx = folds[fold]
        tr_idx = np.array([i for i in range(len(train)) if i not in set(va_idx.tolist())])
        res = train_fold(args, train, tr_idx, va_idx, fold, log, device)
        (run_dir / f"fold{fold}_result.json").write_text(
            json.dumps({k: v for k, v in res.items() if k != "probs"}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        results.append(res)

    # OOF 概率合并（把磁盘上已有的 fold*_oof.npz 拼起来，支持分多次调用训练）
    mm = merge_oof(run_dir, train, args, log)
    ordered = merge_summary(run_dir, results, args, train, test, mm)
    if ordered:
        log(f"[done] {len(ordered)} 折的折均 dice "
            f"{float(np.mean([f['mean']['dice'] for f in ordered])):.4f} "
            f"iou {float(np.mean([f['mean']['iou'] for f in ordered])):.4f}")
    return 0


def merge_oof(run_dir: Path, train: list, args, log) -> dict:
    """把 `fold{k}_oof.npz` 拼成完整 OOF 并评估（支持分多次调用训练）。"""
    oof = np.zeros((len(train),) + train[0].shape, np.float16)
    filled = np.zeros(len(train), bool)
    for f in sorted(run_dir.glob("fold*_oof.npz")):
        z = np.load(f)
        for idx, p in zip(z["val_idx"], z["probs"]):
            oof[int(idx)] = p
            filled[int(idx)] = True
    np.savez_compressed(run_dir / "oof.npz", probs=oof, filled=filled,
                        ids=np.array([s.image_id for s in train]),
                        masks=np.stack([s.mask for s in train]),
                        fovs=np.stack([s.fov for s in train]))
    if not filled.all():
        log(f"[oof] 已有 {int(filled.sum())}/{len(train)} 张 OOF（还有折没跑完）")
        return {}
    all_m = [seg_metrics(postprocess(oof[i].astype(np.float32), train[i].fov,
                                     thr=args.thr, min_size=args.min_size),
                         train[i].mask, train[i].fov) for i in range(len(train))]
    mm = mean_metrics(all_m)
    log(f"[oof] 阈值 {args.thr} 下全量 OOF：dice {mm['dice']:.4f} iou {mm['iou']:.4f} "
        f"sen {mm['sen']:.4f} spe {mm['spe']:.4f} acc {mm['acc']:.4f}")
    return mm


def merge_summary(run_dir: Path, results: list[dict], args, train, test, mm: dict) -> list[dict]:
    """把本次跑的折与历史上跑过的折合并写进 summary.json。"""
    summary_path = run_dir / "summary.json"
    prev: dict = {}
    if summary_path.exists():
        try:
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}
    by_fold = {f["fold"]: f for f in prev.get("folds", [])}
    for r in results:
        by_fold[r["fold"]] = {"fold": r["fold"], "val_ids": r["val_ids"],
                              "best": r["best"], "mean": r["mean"]}
    ordered = [by_fold[k] for k in sorted(by_fold)]
    summary = {
        "tag": args.tag,
        "data": str(args.data),
        "n_train": len(train), "n_test": len(test),
        "shape": list(train[0].shape),
        "args": vars(args),
        "folds": ordered,
        "fold_mean": {k: float(np.mean([f["mean"][k] for f in ordered]))
                      for k in ("dice", "iou", "sen", "spe", "acc")} if ordered else {},
        "oof_mean": mm,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return ordered


def summarize(results: list[dict]) -> str:
    m = {k: float(np.mean([r["mean"][k] for r in results])) for k in results[0]["mean"]}
    return (f"折均 dice {m['dice']:.4f} iou {m['iou']:.4f} "
            f"sen {m['sen']:.4f} spe {m['spe']:.4f} acc {m['acc']:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
