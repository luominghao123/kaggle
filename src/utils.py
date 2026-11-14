"""通用工具：随机种子、指标、EMA、进度与日志。

本文件不依赖具体数据集，可单独测试。
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------
# 随机种子
# --------------------------------------------------------------------------


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """固定所有随机源，保证实验可复现。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 输入尺寸固定时 benchmark 能自动选最快的卷积算法
        torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------
# 日志
# --------------------------------------------------------------------------


def get_logger(name: str = "plant", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # 避免重复添加 handler
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


# --------------------------------------------------------------------------
# 训练辅助
# --------------------------------------------------------------------------


class AverageMeter:
    """滑动平均累计器。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


class Timer:
    """简单计时器，用于打印 epoch 耗时。"""

    def __init__(self) -> None:
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def reset(self) -> None:
        self.t0 = time.time()


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def describe_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"CUDA:{name} ({mem:.1f} GB), torch={torch.__version__}, cuda={torch.version.cuda}"
    return f"CPU only, torch={torch.__version__}"


def human_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
