"""在 GPU 上做的批量化数据增强。

为什么要自己写在 GPU 上跑
------------------------
本机沙箱禁止进程间命名管道，`DataLoader(num_workers>0)` 会直接报 WinError 5，
只能单进程读数据。48x48 的小图虽然解码不慢，但如果把「上采样到 128/160 + 颜色抖动」
放在 CPU 上做，28k 张/epoch 会成为明显瓶颈。

既然整份数据只有 65 MB，干脆把它常驻显存，每个 batch 在 GPU 上：
    随机裁剪 -> 随机仿射（旋转/缩放/平移/错切）-> 上采样 -> 标准化 -> Cutout -> Mixup/CutMix
这样增强几乎不占额外时间。

针对 FER 的增强取舍
------------------
* 水平翻转：人脸表情左右对称，可以翻；**不做垂直翻转**（上下颠倒的人脸不属于自然分布）。
* 旋转限制在 ±12°：人脸有天然朝向，大角度旋转收益为负。
* 缩放/平移：模拟人脸在画面中的位置与大小变化（数据本身就有这种差异）。
* Cutout：遮挡局部（眼睛/嘴）能显著抑制过拟合，FER 上是最有效的正则之一。
* Mixup/CutMix：小数据 + 类别不平衡下有用，最后几个 epoch 关掉，让模型回到真实分布。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# ImageNet 归一化参数（灰度图复制成 3 通道后沿用预训练模型的统计量）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_batch(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """x: [B,3,H,W] 取值 [0,1] -> 标准化。"""
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / s


def prep_eval(images_u8: torch.Tensor, out_size: int) -> torch.Tensor:
    """验证/推理用：uint8 [B,48,48] -> 标准化后的 [B,3,out,out]。"""
    x = images_u8.unsqueeze(1).float().div_(255.0)
    if out_size != x.shape[-1]:
        x = F.interpolate(x, size=(out_size, out_size), mode="bilinear",
                          align_corners=False, antialias=True)
    x = x.repeat(1, 3, 1, 1)
    return normalize_batch(x)


def _random_crop(x: torch.Tensor, crop: int) -> torch.Tensor:
    """批量随机裁剪（所有样本裁到同一尺寸，便于后续一次上采样）。"""
    B, C, H, W = x.shape
    if crop >= H and crop >= W:
        return x
    dy = torch.randint(0, H - crop + 1, (B,), device=x.device)
    dx = torch.randint(0, W - crop + 1, (B,), device=x.device)
    # 用 gather 造索引，避免 python 循环
    rows = dy.view(B, 1, 1) + torch.arange(crop, device=x.device).view(1, crop, 1)
    cols = dx.view(B, 1, 1) + torch.arange(crop, device=x.device).view(1, 1, crop)
    rows = rows.expand(B, crop, crop)
    cols = cols.expand(B, crop, crop)
    idx = rows * W + cols  # [B,crop,crop]
    flat = x.reshape(B, C, H * W)
    idx = idx.unsqueeze(1).expand(B, C, crop, crop).reshape(B, C, -1)
    out = torch.gather(flat, 2, idx)
    return out.reshape(B, C, crop, crop)


def _random_affine(x: torch.Tensor, max_rot: float, max_trans: float,
                   scale_range: tuple[float, float], max_shear: float,
                   flip_p: float) -> torch.Tensor:
    """批量随机仿射：旋转 + 缩放 + 平移 + 错切 + 水平翻转。"""
    B = x.size(0)
    dev = x.device

    ang = (torch.rand(B, device=dev) * 2 - 1) * (max_rot * math.pi / 180.0)
    sc = torch.empty(B, device=dev).uniform_(scale_range[0], scale_range[1])
    tx = (torch.rand(B, device=dev) * 2 - 1) * max_trans
    ty = (torch.rand(B, device=dev) * 2 - 1) * max_trans
    shx = (torch.rand(B, device=dev) * 2 - 1) * (max_shear * math.pi / 180.0)

    flip = (torch.rand(B, device=dev) < flip_p)
    fx = torch.where(flip, -torch.ones_like(sc), torch.ones_like(sc))

    cos, sin = torch.cos(ang), torch.sin(ang)

    # 线性部分 = 翻转 * 旋转 * 错切 * 缩放；theta 是「输出坐标 -> 输入坐标」的映射
    a = cos * fx / sc
    b = (-sin + cos * torch.tan(shx)) * fx / sc
    c = sin / sc
    d = (cos + sin * torch.tan(shx)) / sc

    theta = torch.zeros(B, 2, 3, device=dev, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = a, b, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = c, d, ty

    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)


def _cutout(x: torch.Tensor, p: float, frac_range: tuple[float, float],
            fill: float = 0.0) -> torch.Tensor:
    """随机擦除：按概率把一块矩形区域置为 fill（标准化后 fill=0 即图像均值）。"""
    B, C, H, W = x.shape
    do = torch.rand(B, device=x.device) < p
    if not bool(do.any()):
        return x
    frac = torch.empty(B, device=x.device).uniform_(*frac_range)
    bh = (frac * H).sqrt().round().clamp(2, H).long()
    bw = (frac * W).sqrt().round().clamp(2, W).long()
    y0 = (torch.rand(B, device=x.device) * (H - bh + 1).float()).long()
    x0 = (torch.rand(B, device=x.device) * (W - bw + 1).float()).long()
    yy = torch.arange(H, device=x.device).view(1, H, 1)
    xx = torch.arange(W, device=x.device).view(1, 1, W)
    mask = ((yy >= y0.view(B, 1, 1)) & (yy < (y0 + bh).view(B, 1, 1)) &
            (xx >= x0.view(B, 1, 1)) & (xx < (x0 + bw).view(B, 1, 1)))
    mask = mask & do.view(B, 1, 1)
    return torch.where(mask.unsqueeze(1), torch.full_like(x, fill), x)


def _lightness(x: torch.Tensor, strength: float = 0.15) -> torch.Tensor:
    """亮度/对比度抖动（灰度复制的 3 通道，等价于整体灰度扰动）。"""
    B = x.size(0)
    dev = x.device
    a = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * strength      # 对比度
    b = (torch.rand(B, device=dev) * 2 - 1) * strength            # 亮度
    view = (B, 1, 1, 1)
    return x * a.view(view) + b.view(view)


def random_batch(images_u8: torch.Tensor, labels: torch.Tensor, out_size: int, *,
                 crop_pad: int = 4, max_rot: float = 12.0, max_trans: float = 0.08,
                 scale_range: tuple[float, float] = (1.0, 1.35), max_shear: float = 6.0,
                 flip_p: float = 0.5, cutout_p: float = 0.5,
                 cutout_frac: tuple[float, float] = (0.02, 0.12),
                 jitter: float = 0.15) -> torch.Tensor:
    """一个训练 batch 的增强：uint8 [B,48,48] -> 标准化后的 [B,3,out,out]。

    crop_pad 是随机裁剪掉的边（48 -> 44），配合仿射里的缩放一起构成
    「随机缩放裁剪」的效果。
    """
    x = images_u8.unsqueeze(1).float().div_(255.0)
    crop = x.shape[-1] - crop_pad
    if crop_pad > 0:
        x = _random_crop(x, crop)
    x = _random_affine(x, max_rot, max_trans, scale_range, max_shear, flip_p)
    if out_size != crop:
        x = F.interpolate(x, size=(out_size, out_size), mode="bilinear",
                          align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    if jitter > 0:
        x = _lightness(x, jitter)
    x = normalize_batch(x.clamp_(0.0, 1.0))
    if cutout_p > 0:
        x = _cutout(x, cutout_p, cutout_frac)
    return x.contiguous()


# --------------------------------------------------------------------------
# Mixup / CutMix
# --------------------------------------------------------------------------


def one_hot(labels: torch.Tensor, num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    off = smoothing / num_classes
    on = 1.0 - smoothing + off
    y = torch.full((labels.size(0), num_classes), off, device=labels.device,
                   dtype=torch.float32)
    y.scatter_(1, labels.view(-1, 1), on)
    return y


def mixup_cutmix(x: torch.Tensor, y: torch.Tensor, *,
                 mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0,
                 prob: float = 0.5, switch_p: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (混合后的图, 混合后的软标签)。y 需为 one-hot/平滑后的软标签。"""
    if prob <= 0 or (mixup_alpha <= 0 and cutmix_alpha <= 0):
        return x, y
    if float(torch.rand(1, device=x.device)) > prob:
        return x, y

    B = x.size(0)
    perm = torch.randperm(B, device=x.device)

    if cutmix_alpha > 0 and (mixup_alpha <= 0 or float(torch.rand(1, device=x.device)) < switch_p):
        lam = float(torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample())
        H, W = x.shape[-2:]
        rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        if rh > 0 and rw > 0:
            cy = int(torch.randint(0, H, (1,), device=x.device))
            cx = int(torch.randint(0, W, (1,), device=x.device))
            y0, y1 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
            x0, x1 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
            x[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
            lam = 1 - ((y1 - y0) * (x1 - x0) / (H * W))
        else:
            lam = 1.0
        y = lam * y + (1 - lam) * y[perm]
    else:
        lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
        x = lam * x + (1 - lam) * x[perm]
        y = lam * y + (1 - lam) * y[perm]
    return x, y


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """软标签交叉熵（Mixup/CutMix 与 label smoothing 都走这里）。"""
    logp = F.log_softmax(logits.float(), dim=1)
    return -(target.float() * logp).sum(dim=1).mean()
