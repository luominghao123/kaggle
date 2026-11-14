"""任务二的模型构建：timm 预训练骨干 + 优化器参数分组。

与 `src/models.py` 的区别只在于「默认类别数 6」和「推荐的骨干清单更轻」
（输入只有 48x48，放大到 96~160 训练，太重的骨干既慢又容易过拟合）。
"""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 必须在 import timm / huggingface_hub 之前设置（本机直连 huggingface.co 会超时）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

#: 别名 -> timm 权重名
MODEL_ZOO: dict[str, str] = {
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "convnext_small": "convnext_small.fb_in22k_ft_in1k",
    "convnext_nano": "convnext_nano.in12k_ft_in1k",
    "resnet18": "resnet18.a1_in1k",
    "resnet34": "resnet34.a1_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "tf_efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "vit_tiny": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "mobilevit_s": "mobilevit_s.cvnets_in1k",
}

DEFAULT_MODEL = "convnext_tiny"


def resolve_name(name_or_alias: str) -> str:
    return MODEL_ZOO.get(name_or_alias, name_or_alias)


def build_model(name_or_alias: str = DEFAULT_MODEL, num_classes: int = 6, *,
                img_size: int | None = None, pretrained: bool = True,
                drop_path_rate: float = 0.1, drop_rate: float = 0.0,
                verbose: bool = True) -> nn.Module:
    timm_name = resolve_name(name_or_alias)
    kwargs = dict(pretrained=pretrained, num_classes=num_classes,
                  drop_rate=drop_rate, drop_path_rate=drop_path_rate)
    # 只有 ViT/DeiT 这类带固定位置编码的模型需要显式 img_size，且要开动态尺寸
    if "vit" in timm_name or "deit" in timm_name:
        kwargs["img_size"] = img_size or 224
        kwargs["dynamic_img_size"] = True

    model = timm.create_model(timm_name, **kwargs)
    if verbose:
        total = sum(p.numel() for p in model.parameters())
        print(f"[model] {timm_name}  img_size={img_size}  pretrained={pretrained}  "
              f"params={total/1e6:.2f}M  drop_path={drop_path_rate}")
    return model


def split_weight_decay(model: nn.Module, weight_decay: float):
    """bias / norm 类参数不做 weight decay（常规做法，稳定且通常略涨点）。"""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "pos_embed" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, *, lr: float = 3e-4, weight_decay: float = 0.05,
                    opt: str = "adamw") -> torch.optim.Optimizer:
    groups = split_weight_decay(model, weight_decay)
    if opt.lower() == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))
    if opt.lower() == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)
    raise ValueError(f"不支持的优化器: {opt}")
