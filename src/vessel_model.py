"""实验四：视网膜血管分割模型层（U-Net + timm 预训练编码器）。

设计要点
--------
1. **编码器用 ImageNet 预训练**。只有 20 张训练图，从零训 U-Net 很容易过拟合到
   这 20 张的纹理；预训练的卷积特征（边缘/曲线/管状结构）能明显领先。
   本机 HF 走镜像，`efficientnet_b0 / resnet34 / convnext_tiny` 权重已在 `.hf_cache`。
2. **解码器自顶向下 + 跳连**，最浅一层上采样两次到原分辨率（stride 1），
   血管是 1~2 px 细线，输出必须在全分辨率上，不能留在 stride 2/4。
3. **损失 = BCE（带 pos_weight）+ soft Dice**。血管只占视场 12%，
   纯 BCE 会偏向背景，Dice 项直接优化重叠度。
4. 所有指标都在 **FOV 掩膜内**统计，避免圆形视野外的黑边稀释数字。

模块清单
--------
- `build_model()`     骨干 + 解码器
- `SegLoss`           训练损失
- `predict_probs()`   全图推理（含 padding 与翻转 TTA）
- `postprocess()`     阈值 + 最小连通域过滤 + 填洞
- `seg_metrics()`     dice / iou / sensitivity / specificity / accuracy
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vessel_data import pad_to_stride


# --------------------------------------------------------------------------- #
# 网络
# --------------------------------------------------------------------------- #
def conv_bn(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class UpBlock(nn.Module):
    """上采样 -> 拼跳连 -> 两层卷积。"""

    def __init__(self, cin: int, cskip: int, cout: int, n_conv: int = 2):
        super().__init__()
        self.reduce = conv_bn(cin, cout, 1)
        body = [conv_bn(cout + cskip, cout)]
        body += [conv_bn(cout, cout) for _ in range(n_conv - 1)]
        self.body = nn.Sequential(*body)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.body(torch.cat([x, skip], 1))


class VesselUNet(nn.Module):
    """timm 编码器 + 轻量解码器，输出单通道 logits（与输入同分辨率）。"""

    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True,
                 in_chans: int = 3, decoder_ch: tuple[int, ...] = (256, 128, 96, 64),
                 drop: float = 0.0):
        super().__init__()
        import timm
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, in_chans=in_chans,
            features_only=True, out_indices=None,
        )
        info = self.encoder.feature_info
        chs = list(info.channels())
        reds = list(info.reduction())
        # 只要 stride>=2 的层；最深层做解码起点
        keep = [i for i, r in enumerate(reds) if r >= 2]
        self.keep = keep
        self.chs = [chs[i] for i in keep]
        self.reds = [reds[i] for i in keep]
        if len(self.chs) < 3:
            raise ValueError(f"{backbone} 特征层太少: {self.reds}")

        dec = list(decoder_ch)
        while len(dec) < len(self.chs):
            dec.append(max(32, dec[-1] // 2))
        self.up = nn.ModuleList()
        cin = self.chs[-1]
        for i in range(len(self.chs) - 2, -1, -1):
            cout = dec[i]
            self.up.append(UpBlock(cin, self.chs[i], cout, n_conv=2 if i >= len(self.chs) - 3 else 1))
            cin = cout
        # 最浅特征 stride2 -> 再上采样一次到 stride1
        self.head = nn.Sequential(conv_bn(cin, max(32, cin // 2)), conv_bn(max(32, cin // 2), 32))
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feats = self.encoder(x)
        feats = [feats[i] for i in self.keep]
        x = feats[-1]
        for j, i in enumerate(range(len(feats) - 2, -1, -1)):
            x = self.up[j](x, feats[i])
        # 最浅特征可能是 stride2（EfficientNet/ResNet）也可能是 stride4（ConvNeXt），
        # 所以这里直接对齐到输入尺寸，而不是固定 ×2 —— 否则后者输出会只有一半分辨率。
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = self.drop(self.head(x))
        return self.out(x)


def build_model(backbone: str = "efficientnet_b0", pretrained: bool = True,
                in_chans: int = 3, **kw) -> VesselUNet:
    return VesselUNet(backbone=backbone, pretrained=pretrained, in_chans=in_chans, **kw)


# --------------------------------------------------------------------------- #
# 损失
# --------------------------------------------------------------------------- #
def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


class SegLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, dice_w: float = 1.0, bce_w: float = 1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        self.dice_w = dice_w
        self.bce_w = bce_w

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        dice = soft_dice_loss(logits, target)
        loss = self.bce_w * bce + self.dice_w * dice
        return loss, {"bce": float(bce.detach()), "dice_loss": float(dice.detach())}


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_probs(model: nn.Module, image_chw: np.ndarray, stride: int = 32,
                  tta_flips: tuple[str, ...] = ()) -> np.ndarray:
    """对一张已标准化的 (3,H,W) 图做全分辨率推理，返回 (H,W) float32 概率。

    padding 到 stride 的整数倍再裁回，保证尺寸任意；``tta_flips`` 里可放
    ``'h'``/``'v'``（水平/竖直翻转），眼底图没有固定朝向，翻转 TTA 是合法的。
    """
    x, (h, w) = pad_to_stride(image_chw, stride=stride, value=0.0)
    t = torch.from_numpy(np.ascontiguousarray(x))[None]
    model.eval()
    device = next(model.parameters()).device

    variants = [("",)]
    for f in tta_flips:
        variants.append((f,))

    acc = None
    for (flip,) in variants:
        tt = t
        if "h" in flip:
            tt = torch.flip(tt, dims=[3])
        if "v" in flip:
            tt = torch.flip(tt, dims=[2])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logit = model(tt.to(device, non_blocking=True))
        prob = torch.sigmoid(logit.float())
        if "h" in flip:
            prob = torch.flip(prob, dims=[3])
        if "v" in flip:
            prob = torch.flip(prob, dims=[2])
        p = prob[0, 0, :h, :w].detach().float().cpu().numpy()
        acc = p if acc is None else acc + p
    return acc / len(variants)


def postprocess(prob: np.ndarray, fov: np.ndarray | None = None, thr: float = 0.5,
                min_size: int = 0, fill_holes: bool = False,
                dilate: int = 0) -> np.ndarray:
    """概率 -> 二值掩膜。视场外一律置 0（那些像素本来就不是视网膜）。"""
    m = (prob >= thr).astype(np.uint8)
    if fov is not None:
        m &= fov.astype(np.uint8)
    if min_size > 0 and m.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros(max(n, 2), bool)
        for i in range(1, n):
            keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_size
        m = keep[lab].astype(np.uint8)
    if fill_holes and m.any():
        h, w = m.shape
        ff = m.copy()
        mask_ff = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, mask_ff, (0, 0), 1)
        m = np.clip(m + (ff == 0).astype(np.uint8), 0, 1)
        if fov is not None:
            m &= fov.astype(np.uint8)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        m = cv2.dilate(m, k)
        if fov is not None:
            m &= fov.astype(np.uint8)
    return m.astype(bool)


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def seg_metrics(pred: np.ndarray, gt: np.ndarray, fov: np.ndarray | None = None) -> dict:
    """pred/gt: bool 或 0/1；fov: bool 掩膜（None 表示全图统计）。"""
    p = pred.astype(bool)
    g = gt.astype(bool)
    if fov is not None:
        p = p & fov
        g = g & fov
    tp = float(np.count_nonzero(p & g))
    fp = float(np.count_nonzero(p & ~g))
    fn = float(np.count_nonzero(~p & g))
    tn = float(np.count_nonzero(~p & ~g))
    eps = 1e-7
    return {
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "sen": tp / (tp + fn + eps),
        "spe": tn / (tn + fp + eps),
        "acc": (tp + tn) / (tp + tn + fp + fn + eps),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def mean_metrics(list_of: list[dict]) -> dict:
    keys = [k for k in list_of[0] if not k.startswith("t")]
    return {k: float(np.mean([d[k] for d in list_of])) for k in keys}


__all__ = [
    "VesselUNet", "build_model", "SegLoss", "soft_dice_loss", "predict_probs",
    "postprocess", "seg_metrics", "mean_metrics", "conv_bn",
]
