"""模型层：基于 timm 的迁移学习骨干网络 + 优化器参数分组。

为什么用 timm
-------------
1. 模型库最全（ConvNeXt / Swin / EfficientNetV2 / RegNet / ViT / MaxViT ...），
   方便做多骨干集成冲榜；
2. 权重质量好，`*.in22k_ft_in1k` 系列在 12 类植物幼苗这种小数据上收敛极快；
3. 自带 drop_path、EMA 等训练技巧。

国内网络
--------
timm 的预训练权重托管在 HuggingFace，本机直连 huggingface.co 超时，
但 hf-mirror.com 可达；因此这里在 import timm 之前设置 HF_ENDPOINT 走镜像。
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# 环境变量：必须在 import timm / huggingface_hub 之前设置
# --------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 1) 权重镜像：本机直连 huggingface.co 会超时，走国内镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 2) 缓存目录：默认落在 C:\Users\<user>\.cache\huggingface，位于工作区之外，
#    会被 DSH 文件沙箱拒绝创建（WinError 5）。改到工作区内，顺便方便复用。
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / ".hf_cache"))

# 3) Windows 上 symlink 需要开发者模式，关掉警告并让 hub 退回复制模式
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# --------------------------------------------------------------------------
# 推荐骨干：单卡 8GB 显存下的性价比之选
# --------------------------------------------------------------------------
# (timm 名称, 预训练原生分辨率, 备注, 384 分辨率下 batch size 参考值)
MODEL_ZOO: dict[str, dict] = {
    "convnext_tiny": {
        "timm": "convnext_tiny.fb_in22k_ft_in1k",
        "native": 224,
        "bs384": 16,
        "note": "ConvNeXt-T，精度/速度平衡最好，首选",
    },
    "convnext_small": {
        "timm": "convnext_small.fb_in22k_ft_in1k",
        "native": 224,
        "bs384": 8,
        "note": "更强的 ConvNeXt-S，显存吃紧时降到 320",
    },
    "swin_tiny": {
        "timm": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        "native": 224,
        "bs384": 12,
        "note": "Swin-T，与 CNN 互补，集成增益明显",
    },
    "swin_small": {
        "timm": "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
        "native": 224,
        "bs384": 8,
        "note": "Swin-S",
    },
    "effnetv2_s": {
        "timm": "tf_efficientnetv2_s.in21k_ft_in1k",
        "native": 384,
        "bs384": 16,
        "note": "EfficientNetV2-S，原生 384 分辨率，快且强",
    },
    "effnet_b0": {
        "timm": "efficientnet_b0.ra_in1k",
        "native": 224,
        "bs384": 32,
        "note": "轻量兜底，训练极快",
    },
    "regnetx_016": {
        "timm": "regnetx_016.tv2_in1k",
        "native": 224,
        "bs384": 24,
        "note": "RegNetX-1.6GF，小而稳",
    },
    "vit_base": {
        "timm": "vit_base_patch16_224.augreg_in21k_ft_in1k",
        "native": 224,
        "bs384": 8,
        "note": "ViT-B/16，注意力可视化好看，但需更多 epoch",
    },
    "maxvit_tiny": {
        "timm": "maxvit_tiny_tf_224.in1k",
        "native": 224,
        "bs384": 8,
        "note": "MaxViT-T",
    },
}

DEFAULT_ALIAS = "convnext_tiny"


def resolve_name(name_or_alias: str) -> tuple[str, dict]:
    """把别名（如 convnext_tiny）解析成 (timm 全名, 元信息)。"""
    if name_or_alias in MODEL_ZOO:
        info = MODEL_ZOO[name_or_alias]
        return info["timm"], info
    # 允许直接传 timm 全名
    return name_or_alias, {"timm": name_or_alias, "native": 224, "bs384": 12, "note": "自定义"}


def build_model(
    name_or_alias: str = DEFAULT_ALIAS,
    num_classes: int = 12,
    *,
    img_size: int | None = None,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    verbose: bool = True,
) -> nn.Module:
    """构建分类模型。

    drop_path_rate 是小数据微调的关键正则项：0.05~0.15 通常最好。
    """
    timm_name, info = resolve_name(name_or_alias)
    size = img_size or info.get("native", 224)

    kwargs = dict(
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
    # 只有 ViT/DeiT 这类带固定长度位置编码的模型需要显式声明输入尺寸；
    # ConvNeXt / Swin / EfficientNet 等是纯卷积或窗口注意力，传 img_size 反而会报错。
    if "vit" in timm_name or "deit" in timm_name:
        kwargs["img_size"] = size
        kwargs["dynamic_img_size"] = True

    model = timm.create_model(timm_name, **kwargs)

    if verbose:
        total, trainable = (
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
        print(
            f"[model] {timm_name}  img_size={size}  pretrained={pretrained}  "
            f"params={total/1e6:.2f}M (trainable {trainable/1e6:.2f}M)  "
            f"drop_path={drop_path_rate}"
        )
    return model


# --------------------------------------------------------------------------
# 优化器参数分组
# --------------------------------------------------------------------------


def split_weight_decay(model: nn.Module, weight_decay: float):
    """bias / norm / 位置编码不做 weight decay，其余做（常规做法，稳定且略涨点）。"""
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


def build_optimizer(
    model: nn.Module,
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    betas: tuple[float, float] = (0.9, 0.999),
    opt: str = "adamw",
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    groups = split_weight_decay(model, weight_decay)
    if opt.lower() == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
    if opt.lower() == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)
    raise ValueError(f"不支持的优化器: {opt}")
