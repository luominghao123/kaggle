"""推理：把 uint8 图像批量送进模型，支持多视角 TTA。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .aug import normalize_batch, prep_eval


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _amp_ctx(device: torch.device, amp_dtype: str):
    if device.type != "cuda":
        return _null()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _center_crop_view(images_u8: torch.Tensor, crop: int, out_size: int) -> torch.Tensor:
    """中心裁剪 crop×crop 再放大到 out_size，得到「更近」的尺度视角。"""
    H = images_u8.shape[-1]
    o = max((H - crop) // 2, 0)
    x = images_u8[..., o:o + crop, o:o + crop]
    x = x.unsqueeze(1).float().div_(255.0)
    x = F.interpolate(x, size=(out_size, out_size), mode="bilinear",
                      align_corners=False, antialias=True)
    return x.repeat(1, 3, 1, 1)


def count_views(tta: int) -> int:
    """tta=0 -> 1 视角；tta=1 -> +水平翻转 = 2；tta=2 -> 再加两个尺度×翻转 = 6。"""
    return {0: 1, 1: 2}.get(tta, 6)


def view_batches(images_u8: torch.Tensor, img_size: int, tta: int):
    """逐个产出标准化后的输入视角 [B,3,S,S]。"""
    yield normalize_batch(prep_eval(images_u8, img_size))
    if tta >= 1:
        yield normalize_batch(prep_eval(images_u8.flip(-1), img_size))
    if tta >= 2:
        H = images_u8.shape[-1]
        for crop in (int(round(H * 0.92)), int(round(H * 0.83))):
            v = normalize_batch(_center_crop_view(images_u8, crop, img_size))
            yield v
            yield v.flip(-1)


@torch.no_grad()
def predict_probs(model, images_u8: torch.Tensor, *, img_size: int, batch_size: int = 256,
                  tta: int = 1, amp_dtype: str = "bf16") -> np.ndarray:
    """返回 [N, num_classes] 的 softmax 概率（多视角平均）。

    images_u8: CPU 或 GPU 上的 uint8 张量 [N,48,48]。
    """
    model.eval()
    device = next(model.parameters()).device
    images_u8 = images_u8.to(device, non_blocking=True)

    n = images_u8.shape[0]
    n_views = count_views(tta)
    out: list[torch.Tensor] = []

    for start in range(0, n, batch_size):
        chunk = images_u8[start:start + batch_size]
        acc = None
        for view in view_batches(chunk, img_size, tta):
            with _amp_ctx(device, amp_dtype):
                logits = model(view)
            p = F.softmax(logits.float(), dim=1)
            acc = p if acc is None else acc + p
        out.append((acc / n_views).cpu())

    return torch.cat(out, dim=0).numpy()
