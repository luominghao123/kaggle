"""训练引擎：单 epoch 训练 / 验证 / 指标计算。

包含：自动混合精度（bf16 优先）、Mixup-CutMix、EMA、梯度裁剪、
以及同时跟踪 accuracy 与 macro-F1（竞赛指标）。
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .utils import AverageMeter


# --------------------------------------------------------------------------
# 精度上下文
# --------------------------------------------------------------------------


def get_amp_context(device: torch.device, dtype_name: str = "bf16"):
    """返回 (autocast_ctx_factory, scaler, use_scaler)。

    bf16 与 fp16 的区别：bf16 动态范围大，不需要 GradScaler；
    RTX 4060 (Ada) 对两者都支持，默认用 bf16 更省心。
    """
    if device.type != "cuda":
        return (lambda: _nullcontext()), None, False

    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    use_scaler = dtype_name == "fp16"

    def ctx():
        try:
            return torch.amp.autocast(device_type="cuda", dtype=dtype)
        except AttributeError:  # 老版本 torch
            return torch.cuda.amp.autocast(dtype=dtype)

    if use_scaler:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    return ctx, scaler, use_scaler


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """同时给出 accuracy 与多种 F1 平均方式，便于和榜单口径对齐。"""
    labels = list(range(num_classes))
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels,
                                   zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=labels,
                                      zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", labels=labels,
                                   zero_division=0)),
    }


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> str:
    labels = list(range(len(classes)))
    f1s = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    lines = ["", "各类别 F1 / 召回 / 精确率:"]
    lines.append(f"    {'类别':32s} {'n':>5s} {'F1':>8s} {'recall':>8s} {'prec':>8s}")
    for i, c in enumerate(classes):
        tp = cm[i, i]
        n = cm[i].sum()
        recall = tp / n if n else 0.0
        pred_n = cm[:, i].sum()
        prec = tp / pred_n if pred_n else 0.0
        lines.append(f"    {c:32s} {n:5d} {f1s[i]:8.4f} {recall:8.4f} {prec:8.4f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 训练 / 验证
# --------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    *,
    amp_dtype: str = "bf16",
    mixup_fn=None,
    ema=None,
    ema_base_decay: float | None = None,
    grad_clip: float = 5.0,
    log_interval: int = 50,
    logger=None,
) -> dict:
    """训练一个 epoch，返回平均 loss 与训练集准确率。

    ema_base_decay
        若给定，则对 EMA 做「预热」：初期用较小的 decay，随步数增长再逼近
        ema_base_decay。这是必要的——像 0.9998 这种大 decay 的时间常数约 5000 步，
        而小数据集上总步数可能只有几百步，不做预热的话 EMA 权重会一直停留在
        随机初始化附近，EMA 分支等于失效。
    """
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    seen = 0
    n_steps = len(loader)

    ctx, scaler, use_scaler = get_amp_context(device, amp_dtype)

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets_soft = mixup_fn(images, targets)
        else:
            targets_soft = targets

        optimizer.zero_grad(set_to_none=True)
        with ctx():
            outputs = model(images)
            loss = criterion(outputs, targets_soft)

        if use_scaler:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            if ema_base_decay is not None:
                global_step = (epoch - 1) * n_steps + step + 1
                # 标准 EMA 预热：早期 decay 小，后期逼近 base_decay
                ema.decay = min(ema_base_decay, (1.0 + global_step) / (10.0 + global_step))
            ema.update(model)

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        # 训练准确率：mixup 下 targets 是软标签，这里用原始 targets 仅作参考
        with torch.no_grad():
            pred = outputs.argmax(dim=1)
            correct += (pred == targets).sum().item()
            seen += bs

        if logger is not None and log_interval and (step + 1) % log_interval == 0:
            logger.info(
                f"    epoch {epoch} step {step+1}/{len(loader)}  "
                f"loss={loss_meter.avg:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    return {"loss": loss_meter.avg, "acc": correct / max(seen, 1)}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module | None,
    device: torch.device,
    num_classes: int,
    *,
    amp_dtype: str = "bf16",
    return_probs: bool = True,
) -> dict:
    """验证 / 推理。

    兼容两种 DataLoader：
      * 有标签  -> batch = (images, labels_tensor)  ：计算 loss 与指标
      * 无标签  -> batch = (images, paths_list)     ：只返回概率（用于测试集）
    """
    model.eval()
    ctx, _, _ = get_amp_context(device, amp_dtype)

    loss_meter = AverageMeter()
    all_probs, all_labels = [], []
    has_labels = True

    for batch in loader:
        images, targets = batch[0], batch[1]
        images = images.to(device, non_blocking=True)

        with ctx():
            outputs = model(images)

        if isinstance(targets, torch.Tensor):
            targets = targets.to(device, non_blocking=True)
            if criterion is not None:
                loss_meter.update(criterion(outputs, targets).item(), images.size(0))
            all_labels.append(targets.cpu())
        else:
            has_labels = False

        all_probs.append(F.softmax(outputs.float(), dim=1).cpu())

    probs = torch.cat(all_probs).numpy()
    out: dict = {"loss": loss_meter.avg}

    if has_labels and all_labels:
        labels = torch.cat(all_labels).numpy()
        preds = probs.argmax(axis=1)
        out.update(compute_metrics(labels, preds, num_classes))
        out["labels"] = labels
    if return_probs:
        out["probs"] = probs
    return out


# --------------------------------------------------------------------------
# 学习率调度
# --------------------------------------------------------------------------


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    *,
    warmup_epochs: int = 0,
    warmup_prefix: bool = False,
    start_warmup_value: float = 0.0,
):
    """带 warmup 的余弦退火，返回每一项 iter 的 lr 列表。

    参考 DeiT / timm 的实现：前 warmup_epochs 线性升温，其后余弦衰减。
    """
    warmup_iters = max(0, int(warmup_epochs * niter_per_ep))
    total_iters = epochs * niter_per_ep

    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    else:
        warmup_schedule = np.array([])

    iters = np.arange(total_iters - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / max(len(iters), 1))
    )
    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == total_iters, (len(schedule), total_iters)
    return schedule


class CosineSchedulerWithWarmup:
    """把上面的调度表包成可 step() 的对象。"""

    def __init__(self, optimizer, base_lr: float, final_lr: float, epochs: int,
                 niter_per_ep: int, warmup_epochs: int = 0, start_warmup_value: float = 0.0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.schedule = cosine_scheduler(
            base_lr, final_lr, epochs, niter_per_ep,
            warmup_epochs=warmup_epochs, start_warmup_value=start_warmup_value,
        )
        self.it = 0
        self._set_lr(self.schedule[0])

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def step(self):
        if self.it < len(self.schedule) - 1:
            self.it += 1
        self._set_lr(self.schedule[self.it])

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]
