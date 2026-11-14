"""训练用的小工具：EMA、按步进度的学习率调度、指标打印。"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix


class ModelEMA:
    """指数滑动平均权重（带预热）。

    decay=0.999 的时间常数约 1000 步，而本任务每折只有 ~170 步/epoch，
    不做预热的话 EMA 权重会长时间停留在随机初始化附近，等于白加。
    预热公式取自 timm/DeiT：decay_t = min(base, (1+t)/(10+t))。
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.base_decay = decay
        self.decay = 0.0

    @torch.no_grad()
    def update(self, model: nn.Module, step: int | None = None):
        if step is not None:
            self.decay = min(self.base_decay, (1.0 + step) / (10.0 + step))
        d = self.decay
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


def cosine_lr(step: int, total: int, base_lr: float, final_lr: float,
              warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    t = (step - warmup_steps) / max(total - warmup_steps, 1)
    return final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * min(t, 1.0)))


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def report(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    rep = classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                target_names=classes, digits=4, zero_division=0)
    lines = ["混淆矩阵（行=真值, 列=预测）:"]
    header = "        " + "".join(f"{c[:6]:>8s}" for c in classes)
    lines.append(header)
    for i, c in enumerate(classes):
        lines.append(f"{c[:6]:>7s} " + "".join(f"{v:8d}" for v in cm[i]))
    lines.append("")
    lines.append(rep)
    return "\n".join(lines)
