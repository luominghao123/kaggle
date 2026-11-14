"""任务三（中心凹检测定位）模型层：timm 骨干 + FPN 热图头 + soft-argmax。

为什么用「热图回归」而不是「直接回归两个数」
--------------------------------------------
直接回归 (x, y) 把空间结构完全丢掉了：网络必须从全局池化特征里「记住」
坐标，位置分辨率极低。热图回归让网络输出一张**空间响应图**，
再对响应图做 soft-argmax 得到亚像素坐标：

    p_i = softmax(logit)_i          # 整张响应图上的概率分布
    coord = Σ_i p_i · g_i           # g_i 是第 i 个格点的归一化坐标

这样梯度里天然带着「往正确的格子拉」的局部监督，小数据也能训得很好，
而且输出分辨率（stride 4）远高于全局池化。

损失 = 软标签交叉熵（响应图对齐高斯峰） + 坐标 L1（直接优化竞赛指标相关量）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import resolve_name


# ==========================================================================
# 模型
# ==========================================================================


class FoveaNet(nn.Module):
    """骨干 + 金字塔（FPN-lite）热图头。

    参数
    ----
    backbone        : timm 名称或 `src/models.py` 里的别名（convnext_tiny / effnet_b0 ...）
    head_dim        : 侧向 1×1 卷积与融合后的通道数
    out_stride      : 输出热图相对输入的下采样倍数（4 时 512 输入 -> 128×128 响应图）
    """

    def __init__(
        self,
        backbone: str = "effnet_b0",
        *,
        pretrained: bool = True,
        head_dim: int = 64,
        out_stride: int = 4,
        drop_path_rate: float = 0.1,
        drop_rate: float = 0.0,
        verbose: bool = True,
    ):
        super().__init__()
        import timm  # 延迟导入：先让 .models 设置好 HF 镜像环境变量

        timm_name, _ = resolve_name(backbone)
        self.backbone_name = timm_name
        self.out_stride = out_stride

        self.encoder = timm.create_model(
            timm_name,
            pretrained=pretrained,
            features_only=True,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )
        chans = list(self.encoder.feature_info.channels())
        strides = list(self.encoder.feature_info.reduction())

        # 只保留 stride >= out_stride 的层参与融合。
        # 例：EfficientNet 最浅层是 stride 2，如果也拿来融合，
        # 解码器就要在 256×256（512 输入时）上跑 64 通道 3×3 卷积——
        # 实测占了前向开销的一大半，而输出精度本来就只需要 stride 4。
        self.levels = [i for i, s in enumerate(strides) if s >= out_stride] or [len(strides) - 1]
        self.chans = [chans[i] for i in self.levels]
        self.strides = [strides[i] for i in self.levels]

        self.lateral = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, head_dim, 1, bias=False),
                nn.BatchNorm2d(head_dim),
                nn.SiLU(inplace=True),
            )
            for c in self.chans
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(head_dim, head_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_dim),
            nn.SiLU(inplace=True),
        )
        # 输出头：先 3×3 提特征，再 1×1 出单通道响应图
        self.head = nn.Sequential(
            nn.Conv2d(head_dim, head_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_dim, 1, 1),
        )
        # 输出偏置初始化到 0，避免训练初期响应图整体饱和
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.head[-1].weight)

        if verbose:
            total = sum(p.numel() for p in self.parameters())
            print(f"[model] FoveaNet(backbone={timm_name}) "
                  f"levels={list(zip(self.strides, self.chans))} "
                  f"params={total/1e6:.2f}M out_stride={out_stride}")

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.encoder(x)                         # 由浅到深
        used = [feats[i] for i in self.levels]
        # 自顶向下融合
        p = self.lateral[-1](used[-1])
        for i in range(len(used) - 2, -1, -1):
            p = F.interpolate(p, size=used[i].shape[-2:], mode="nearest")
            p = p + self.lateral[i](used[i])
        p = self.smooth(p)

        logits = self.head(p)
        want = x.shape[-2] // self.out_stride, x.shape[-1] // self.out_stride
        if logits.shape[-2:] != want:
            logits = F.interpolate(logits, size=want, mode="bilinear", align_corners=False)

        coords = heatmap_to_coords(logits)
        return {"logits": logits, "coords": coords}


def heatmap_to_coords(logits: torch.Tensor) -> torch.Tensor:
    """soft-argmax：把响应图转成归一化坐标 [0,1]。返回 (B, 2)，顺序 (x, y)。"""
    b, _, h, w = logits.shape
    prob = F.softmax(logits.reshape(b, -1), dim=-1).reshape(b, h, w)

    xs = torch.linspace(0.5 / w, 1 - 0.5 / w, w, device=logits.device, dtype=prob.dtype)
    ys = torch.linspace(0.5 / h, 1 - 0.5 / h, h, device=logits.device, dtype=prob.dtype)

    px = (prob.sum(dim=1) * xs).sum(dim=-1)             # (B,) 按列(行)边缘分布求期望
    py = (prob.sum(dim=2) * ys).sum(dim=-1)
    return torch.stack([px, py], dim=-1)


def gaussian_target(coords: torch.Tensor, h: int, w: int, sigma: float) -> torch.Tensor:
    """按归一化坐标生成高斯响应图（和为 1 的概率分布），形状 (B, 1, h, w)。

    ⚠️ 单位坑：`sigma` 以「**热图格子数**」为单位（例如 2 表示半径约 2 格），
    而下面的 dx2/dy2 是**归一化坐标**（整图跨度为 1）的平方差，
    所以必须先把 sigma 换算成归一化单位 `sigma / max(h, w)`。
    早先版本忘了这一步，σ 相当于覆盖了整张图，
    目标图退化成均匀分布——热图分支因此完全没有监督信号（soft-argmax 只会输出 0.5）。
    """
    b = coords.shape[0]
    s_norm = max(float(sigma), 0.5) / max(h, w)

    xs = torch.linspace(0.5 / w, 1 - 0.5 / w, w, device=coords.device, dtype=coords.dtype)
    ys = torch.linspace(0.5 / h, 1 - 0.5 / h, h, device=coords.device, dtype=coords.dtype)
    gx = coords[:, 0:1]                                 # (B,1)
    gy = coords[:, 1:2]

    dx2 = (xs.view(1, 1, w) - gx.view(b, 1, 1)) ** 2
    dy2 = (ys.view(1, h, 1) - gy.view(b, 1, 1)) ** 2
    heat = torch.exp(-(dx2 + dy2) / (2 * s_norm ** 2))
    heat = heat / heat.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    return heat.unsqueeze(1)


# ==========================================================================
# 损失
# ==========================================================================


class FoveaLoss(nn.Module):
    """软标签交叉熵（响应图） + 坐标光滑 L1。

    sigma 以「热图格子」为单位：默认取热图边长的 2%，
    太小会让监督信号过于稀疏（小数据训不动），太大则定位精度上不去。
    """

    def __init__(self, coord_weight: float = 0.5, sigma_ratio: float = 0.02,
                 min_sigma: float = 1.0):
        super().__init__()
        self.coord_weight = coord_weight
        self.sigma_ratio = sigma_ratio
        self.min_sigma = min_sigma

    def forward(self, out: dict, target_xy: torch.Tensor) -> tuple[torch.Tensor, dict]:
        logits = out["logits"]
        coords = out["coords"]
        h, w = logits.shape[-2:]

        sigma = max(self.min_sigma, self.sigma_ratio * max(h, w))
        target = gaussian_target(target_xy, h, w, sigma).reshape(logits.shape[0], -1)
        log_prob = F.log_softmax(logits.reshape(logits.shape[0], -1), dim=-1)
        loss_heat = -(target * log_prob).sum(dim=-1).mean()

        loss_xy = F.smooth_l1_loss(coords, target_xy, beta=0.01)
        total = loss_heat + self.coord_weight * loss_xy
        return total, {"heat": float(loss_heat.detach()), "xy": float(loss_xy.detach())}


def build_model(
    backbone: str = "effnet_b0",
    *,
    pretrained: bool = True,
    head_dim: int = 64,
    out_stride: int = 4,
    drop_path_rate: float = 0.1,
    verbose: bool = True,
) -> FoveaNet:
    return FoveaNet(
        backbone,
        pretrained=pretrained,
        head_dim=head_dim,
        out_stride=out_stride,
        drop_path_rate=drop_path_rate,
        verbose=verbose,
    )


if __name__ == "__main__":
    import sys
    import time

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_model(sys.argv[1] if len(sys.argv) > 1 else "effnet_b0").to(dev)
    x = torch.randn(2, 3, 512, 512, device=dev)
    t0 = time.time()
    out = net(x)
    print("logits", tuple(out["logits"].shape), "coords", out["coords"].detach().cpu().round(decimals=3).tolist())
    print(f"forward {time.time()-t0:.2f}s on {dev}")
    loss, parts = FoveaLoss()(out, torch.tensor([[0.4, 0.6], [0.3, 0.3]], device=dev))
    print("loss %.4f" % loss.item(), {k: round(v, 4) for k, v in parts.items()})
    if dev == "cuda":
        print("peak VRAM %.0f MB" % (torch.cuda.max_memory_allocated() / 1024**2))
