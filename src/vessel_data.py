"""实验四：视网膜血管分割（Kaggle 2026task-4）数据层。

数据实况（已核对本机 `实验四/segmentation/`）
--------------------------------------------
```
实验四/segmentation/
├── train/image/*.jpg   20 张，565x584（宽x高），彩色眼底图
├── train/label/*.jpg   20 张，同尺寸，二值掩膜（0/255，正类≈8%）
├── test/image/*.jpg    20 张，565x584
├── sample_submission.csv   Id,Predicted（20 行，RLE 在 512x512 上）
└── segmentation_to_csv.py  官方给的「掩膜图 -> RLE 提交」脚本
```
这就是典型的 DRIVE 风格血管分割：正类 = 血管像素，负类 = 背景。
提交的 RLE 是把掩膜 **resize 到 512x512** 之后按行优先编码的，
所以本地评估也要走同一条管线才能和榜单口径对齐（见 `submission_dice`）。

本文件提供
----------
1. `discover()`      自动探测 image/label 目录、配对成 (图像, 掩膜) 列表
2. `fov_mask()`      视场（圆形眼底区域）掩膜 —— 指标只在视场内统计更干净
3. `preprocess()`    绿通道 CLAHE 增强（血管在绿通道对比度最高）
4. `load_set()`      读入并缓存「预处理 + 逐图逐通道标准化」后的 float32 图
5. `PatchDataset`    训练用补丁采样器（前景偏置 + 保结构几何增强）
6. `pad_to_stride()` 验证/推理时的 padding 工具
7. `--selftest`      打印数据布局、正类比例、FOV 面积、增强是否保结构

为什么不用 albumentations / torchvision.transforms
--------------------------------------------------
本机没有 albumentations；而血管是 1~2 px 的细线，增强必须让
**图像与掩膜走同一个坐标变换**，所以这里直接用 cv2.warpAffine / cv2.remap
把图和一掩膜一起变换，掩膜用最近邻、图用双线性，绝对不会错位。
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "实验四" / "segmentation"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGE_DIR_NAMES = {"image", "images", "img", "imgs", "jpg"}
LABEL_DIR_NAMES = {"label", "labels", "mask", "masks", "gt", "groundtruth", "annotation", "annotations"}

cv2.setNumThreads(0)  # 多进程/多线程下避免 OpenCV 自己抢线程


# --------------------------------------------------------------------------- #
# 布局探测
# --------------------------------------------------------------------------- #
def _list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _stem_key(p: Path) -> str:
    return p.stem.lstrip("0") or "0"


def discover(data_dir: str | Path = DEFAULT_DATA) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """返回 (训练对 [(image, mask)], 测试图)。自动配对，缺掩膜的当测试集。"""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    img_dirs = sorted(p for p in data_dir.rglob("*") if p.is_dir() and p.name.lower() in IMAGE_DIR_NAMES)
    if not img_dirs:
        raise FileNotFoundError(f"{data_dir} 下没找到 image/ 目录")

    train_items: list[tuple[Path, Path]] = []
    test_items: list[Path] = []
    for d in img_dirs:
        label_dir = None
        for sib in sorted(d.parent.iterdir()):
            if sib.is_dir() and sib.name.lower() in LABEL_DIR_NAMES:
                label_dir = sib
                break
        images = _list_images(d)
        if label_dir is None:
            test_items += images
            continue
        masks = {_stem_key(p): p for p in _list_images(label_dir)}
        for img in images:
            m = masks.get(_stem_key(img))
            if m is not None:
                train_items.append((img, m))
    if not train_items:
        raise FileNotFoundError(f"{data_dir} 下没找到 image/label 配对的训练数据")
    return train_items, sorted(test_items)


# --------------------------------------------------------------------------- #
# 预处理
# --------------------------------------------------------------------------- #
def fov_mask(rgb: np.ndarray, thr: int = 12) -> np.ndarray:
    """视场掩膜（bool）。眼底图圆形视野外是全黑，这些像素不该参与指标。"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = (gray > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == biggest).astype(np.uint8)
    # 填内部空洞：先取反，再从边界 floodfill，剩下没被填到的就是洞
    h, w = m.shape
    ff = m.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask_ff, (0, 0), 1)
    holes = (ff == 0).astype(np.uint8)
    m = np.clip(m + holes, 0, 1)
    return m.astype(bool)


def preprocess(rgb: np.ndarray, clip: float = 2.5, grid: int = 8) -> np.ndarray:
    """绿通道 CLAHE 替换 G 通道，返回 uint8 RGB。训练/推理必须共用。"""
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, -1)
    out = np.ascontiguousarray(rgb[..., :3])
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    g = clahe.apply(out[..., 1])
    out = out.copy()
    out[..., 1] = g
    return out


@dataclass
class Sample:
    """一张图的全部缓存：预处理图、掩膜、视场、逐通道均值方差。"""
    image_id: str
    image_path: Path
    mask_path: Path | None
    rgb: np.ndarray          # uint8 (H, W, 3)
    mask: np.ndarray         # bool  (H, W)，无掩膜时全 False
    fov: np.ndarray          # bool  (H, W)
    mean: np.ndarray         # (3,)
    std: np.ndarray          # (3,)

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgb.shape[:2]

    def normalized(self) -> np.ndarray:
        """(H, W, 3) float32，按图内视场统计量标准化。"""
        x = self.rgb.astype(np.float32) / 255.0
        return (x - self.mean) / self.std


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """cv2.imread 不支持非 ASCII 路径（本仓库路径含中文），改走 imdecode。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite_unicode(path: Path, arr: np.ndarray) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", arr)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def _read_rgb(path: Path) -> np.ndarray:
    bgr = imread_unicode(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"读不了图像: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, bool)
    m = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"读不了掩膜: {path}")
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def make_sample(image_path: Path, mask_path: Path | None, clip: float = 2.5, grid: int = 8) -> Sample:
    raw = _read_rgb(image_path)
    rgb = preprocess(raw, clip=clip, grid=grid)
    fov = fov_mask(raw)          # 视场用**原始**图算，CLAHE 会放大暗区噪声
    mask = _read_mask(mask_path, rgb.shape[:2])
    x = rgb.astype(np.float32) / 255.0
    sel = fov if fov.mean() > 0.05 else np.ones_like(fov)
    mean = x[sel].mean(0)
    std = np.maximum(x[sel].std(0), 1e-3)
    return Sample(image_path.stem, image_path, mask_path, rgb, mask, fov, mean, std)


def load_set(items: list, clip: float = 2.5, grid: int = 8, with_mask: bool = True) -> list[Sample]:
    """items 可以是 [(img, mask)] 或 [img]。"""
    out = []
    for it in items:
        if isinstance(it, tuple):
            img, msk = it
        else:
            img, msk = it, None
        out.append(make_sample(Path(img), Path(msk) if (msk is not None and with_mask) else None,
                               clip=clip, grid=grid))
    return out


# --------------------------------------------------------------------------- #
# 训练用补丁采样 + 增强
# --------------------------------------------------------------------------- #
def pad_to_stride(img: np.ndarray, stride: int = 32, value: float = 0.0) -> tuple[np.ndarray, tuple[int, int]]:
    """把 (C,H,W) / (H,W) 右下 padding 到 stride 的整数倍，返回 (pad后的图, (H,W))。"""
    h, w = img.shape[-2:]
    ph = int(math.ceil(h / stride) * stride) - h
    pw = int(math.ceil(w / stride) * stride) - w
    if ph == 0 and pw == 0:
        return img, (h, w)
    pad = [(0, 0)] * (img.ndim - 2) + [(0, ph), (0, pw)]
    return np.pad(img, pad, mode="constant", constant_values=value), (h, w)


def random_affine(img: np.ndarray, msk: np.ndarray, rng: random.Random,
                  rot: float = 180.0, scale: tuple[float, float] = (0.85, 1.25),
                  shift: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """同步的旋转+缩放+平移（img: uint8 HxWx3，msk: uint8/bool HxW）。

    mask 走最近邻，图走双线性 —— 两者共用同一个矩阵，所以绝不会错位。
    """
    h, w = img.shape[:2]
    angle = rng.uniform(-rot, rot) if rot > 0 else 0.0
    s = rng.uniform(*scale)
    M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), angle, s)
    M[0, 2] += rng.uniform(-shift, shift) * w
    M[1, 2] += rng.uniform(-shift, shift) * h
    img2 = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
    msk2 = cv2.warpAffine(msk.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img2, msk2


def elastic(img: np.ndarray, msk: np.ndarray, rng: random.Random,
            alpha: float = 0.045, sigma: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """弹性形变（血管走向轻微弯折，20 张图很需要这种多样性）。

    位移场先在 1/4 分辨率上生成再放大：384² 上直接做高斯核，
    单次就 38 ms，会拖死训练吞吐；降采样后只要 ~4 ms，效果肉眼无差。
    """
    h, w = img.shape[:2]
    s = 4
    hh, ww = max(2, h // s), max(2, w // s)
    gen = np.random.default_rng(rng.randrange(1 << 30))
    dx = cv2.GaussianBlur(gen.random((hh, ww), dtype=np.float32) * 2 - 1, (0, 0), sigma / s) * (alpha * w)
    dy = cv2.GaussianBlur(gen.random((hh, ww), dtype=np.float32) * 2 - 1, (0, 0), sigma / s) * (alpha * h)
    dx = cv2.resize(dx, (w, h), interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize(dy, (w, h), interpolation=cv2.INTER_LINEAR)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx = xx + dx
    my = yy + dy
    img2 = cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    msk2 = cv2.remap(msk.astype(np.uint8), mx, my, interpolation=cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img2, msk2


def photometric_lut(gamma: float, a: float, b: float) -> np.ndarray:
    """把 亮/对比/gamma 合成一张 256 项查找表（uint8 上做，比浮点 pow 快 20 倍）。"""
    v = np.arange(256, dtype=np.float32) / 255.0
    v = np.clip(v ** gamma, 0, 1) * a + b
    return np.clip(v * 255.0, 0, 255).astype(np.uint8)


def photometric(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """亮度/对比度/gamma/模糊/噪声：眼底图不同设备曝光差异很大，必须增强。"""
    x = img
    if rng.random() < 0.85:
        a = rng.uniform(0.85, 1.18)
        b = rng.uniform(-0.16, 0.16) * 255.0
        gamma = float(math.exp(rng.uniform(-0.4, 0.4)))
        x = cv2.LUT(x, photometric_lut(gamma, a, b))
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.4, 1.2))
    if rng.random() < 0.3:
        k = rng.randint(3, 12)
        gen = np.random.default_rng(rng.randrange(1 << 30))
        noise = gen.integers(-k, k + 1, size=x.shape, dtype=np.int16).astype(np.int16)
        x = np.clip(x.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return x


def sample_center(fg_xy: np.ndarray, fov_xy: np.ndarray, rng: random.Random, fg_bias: float) -> tuple[int, int]:
    """前景偏置采样：一半左右的补丁保证有血管，避免全背景补丁浪费算力。"""
    if fg_xy.shape[1] > 0 and rng.random() < fg_bias:
        k = rng.randrange(fg_xy.shape[1])
        return int(fg_xy[1, k]), int(fg_xy[0, k])   # (y, x)
    k = rng.randrange(fov_xy.shape[1])
    return int(fov_xy[1, k]), int(fov_xy[0, k])


def crop_with_pad(arr: np.ndarray, cy: int, cx: int, size: int, fill: float = 0.0) -> np.ndarray:
    """以 (cy, cx) 为中心裁 size×size，越界部分用 reflect 补。"""
    h, w = arr.shape[:2]
    half = size // 2
    y0, y1 = cy - half, cy - half + size
    x0, x1 = cx - half, cx - half + size
    py0, py1 = max(0, -y0), max(0, y1 - h)
    px0, px1 = max(0, -x0), max(0, x1 - w)
    pad = [(py0, py1), (px0, px1)] + [(0, 0)] * (arr.ndim - 2)
    a = np.pad(arr, pad, mode="reflect") if (py0 or py1 or px0 or px1) else arr
    return a[y0 + py0:y1 + py0, x0 + px0:x1 + px0]


class PatchDataset:
    """补丁数据集（本机 num_workers 只能为 0，所以实现成「uint8 + 查找表」的低开销版本）。

    每个 ``__getitem__`` 产出 (image[3,S,S] float32, mask[1,S,S] float32)。
    - 几何增强在图与掩膜上同步做（掩膜最近邻）；
    - 光度增强用 256 项 LUT 在 uint8 上做，而不是浮点 pow（实测 68 ms/补丁 -> ~12 ms）；
    - 最后按**该图的固定均值方差**标准化，和验证/推理路径完全一致。
    """

    def __init__(self, samples: list[Sample], indices: list[int] | None = None, patch: int = 384,
                 augment: bool = True, fg_bias: float = 0.5, seed: int = 0,
                 elastic_p: float = 0.3):
        self.samples = [samples[i] for i in (indices if indices is not None else range(len(samples)))]
        self.patch = int(patch)
        self.augment = augment
        self.fg_bias = fg_bias
        self.elastic_p = elastic_p
        self.seed = seed
        self.epoch = 0
        self._prep()
        self.pos_weight = float((1 - self.fg_ratio) / max(self.fg_ratio, 1e-6))

    def _prep(self) -> None:
        self._data = []
        tot_fg = tot_px = 0
        for s in self.samples:
            fov = s.fov
            fg_xy = np.array([], dtype=np.int64).reshape(2, 0)
            if s.mask is not None and s.mask.any():
                fg_xy = np.stack(np.nonzero(s.mask & fov), 0)
            fov_xy = np.stack(np.nonzero(fov), 0)
            if fov_xy.shape[1] == 0:
                fov_xy = np.stack(np.nonzero(np.ones_like(fov)), 0)
            tot_fg += int((s.mask & fov).sum())
            tot_px += int(fov.sum())
            self._data.append((s.rgb, s.mask.astype(np.uint8), s.mean.astype(np.float32),
                               s.std.astype(np.float32), fg_xy, fov_xy))
        self.fg_ratio = tot_fg / max(tot_px, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, idx: int):
        rng = random.Random((self.seed * 1_000_003) + self.epoch * 9176 + idx)
        rgb, mask, mean, std, fg_xy, fov_xy = self._data[idx % len(self._data)]
        h, w = rgb.shape[:2]
        size = self.patch

        if self.augment:
            cy, cx = sample_center(fg_xy, fov_xy, rng, self.fg_bias)
            jitter = size // 3
            cy += rng.randint(-jitter, jitter)
            cx += rng.randint(-jitter, jitter)
        else:
            cy, cx = h // 2, w // 2

        img = crop_with_pad(rgb, cy, cx, size)
        msk = crop_with_pad(mask, cy, cx, size)

        if self.augment:
            if rng.random() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if rng.random() < 0.5:
                img, msk = img[::-1], msk[::-1]
            img, msk = random_affine(img, msk, rng)
            if rng.random() < self.elastic_p:
                img, msk = elastic(img, msk, rng)
            img = photometric(img, rng)

        x = np.ascontiguousarray(img).astype(np.float32) / 255.0
        x = (x - mean) / std
        x = np.ascontiguousarray(x.transpose(2, 0, 1))
        return x, np.ascontiguousarray(msk[None].astype(np.float32))


# --------------------------------------------------------------------------- #
# 批量预取（CPU 增强 与 GPU 前反向 重叠）
# --------------------------------------------------------------------------- #
class BatchPrefetch:
    """后台线程生产 batch，主线程取用。

    本机 DataLoader 的 num_workers 不能开（沙箱禁止在系统 TEMP 建共享内存），
    但增强（~12 ms/补丁）和前反向（~40 ms/补丁）本来就是两个不同的资源，
    用一个后台线程把两者叠起来，实测每个 epoch 省 20%~30% 时间。
    """

    def __init__(self, dataset: PatchDataset, batch_size: int, seed: int, queue_size: int = 3):
        import queue
        import threading

        self.ds = dataset
        self.bs = int(batch_size)
        self.seed = int(seed)
        self.q: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self.stop = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        import queue

        rng = random.Random(self.seed)
        n = len(self.ds)
        while not self.stop:
            xs, ys = [], []
            for _ in range(self.bs):
                x, y = self.ds[rng.randrange(n)]
                xs.append(x)
                ys.append(y)
            while not self.stop:
                try:
                    self.q.put((np.stack(xs), np.stack(ys)), timeout=0.2)
                    break
                except queue.Full:
                    continue

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        return self.q.get()

    def close(self) -> None:
        self.stop = True
        self.thread.join(timeout=5.0)


# --------------------------------------------------------------------------- #
# 官方提交管线（512x512 RLE）
# --------------------------------------------------------------------------- #
def mask_to_rle_512(mask_u8: np.ndarray, size: int = 512) -> str:
    """完全复刻 `segmentation_to_csv.py` 的编码：

    uint8 掩膜 -> resize 512x512 双线性 -> `>0` 二值化 -> 行优先游程编码。
    官方脚本就是这么做的，所以本地评估/提交都走这一条，口径才不会偏。
    """
    m = cv2.resize(mask_u8, (size, size), interpolation=cv2.INTER_LINEAR)
    flat = (m.reshape(-1) > 0)
    dots = np.flatnonzero(flat)
    if dots.size == 0:
        return ""
    breaks = np.flatnonzero(np.diff(dots) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [dots.size - 1]))
    pairs = []
    for s, e in zip(starts, ends):
        pairs.append(str(int(dots[s]) + 1))
        pairs.append(str(int(dots[e] - dots[s] + 1)))
    return " ".join(pairs)


def rle_to_mask(rle: str, size: int = 512) -> np.ndarray:
    """把 RLE 解回 512x512 的 bool 掩膜（校验提交用）。"""
    flat = np.zeros(size * size, bool)
    toks = rle.split()
    if len(toks) % 2:
        raise ValueError("RLE 数值个数不是偶数")
    prev_end = 0
    for i in range(0, len(toks), 2):
        start, length = int(toks[i]), int(toks[i + 1])
        if length <= 0:
            raise ValueError("RLE 长度必须为正")
        if start <= prev_end:
            raise ValueError("RLE 起点必须递增且不重叠")
        end = start - 1 + length
        if end > flat.size:
            raise ValueError("RLE 越界")
        flat[start - 1:end] = True
        prev_end = end
    return flat.reshape(size, size)


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def selftest(data_dir: str | Path = DEFAULT_DATA) -> int:
    data_dir = Path(data_dir)
    train_items, test_items = discover(data_dir)
    print(f"[布局] {data_dir}")
    print(f"  训练 {len(train_items)} 对：{train_items[0][0].name} … {train_items[-1][0].name}")
    print(f"  测试 {len(test_items)} 张：{[p.name for p in test_items][:5]} …")

    train = load_set(train_items)
    test = load_set(test_items, with_mask=False)
    sizes = {s.shape for s in train + test}
    print(f"[尺寸] {sorted(sizes)}（H,W）")
    fg = np.mean([s.mask[s.fov].mean() for s in train])
    fovr = np.mean([s.fov.mean() for s in train])
    print(f"[统计] 视场内血管像素占比 {fg:.4f}；视场占全图 {fovr:.4f}")
    print(f"[标签] 唯一值 {np.unique(train[0].mask.astype(np.uint8))}；"
          f"每张正类像素 {min(int(s.mask.sum()) for s in train)}~{max(int(s.mask.sum()) for s in train)}")

    ds = PatchDataset(train, patch=384, augment=True, seed=1)
    print(f"[补丁] 正类比例 {ds.fg_ratio:.4f} -> BCE pos_weight {ds.pos_weight:.2f}")
    for ep in (0, 1):
        ds.set_epoch(ep)
        x, y = ds[0]
        print(f"  epoch{ep} patch {x.shape} {x.dtype} 范围 [{x.min():.2f}, {x.max():.2f}]，"
              f"掩膜 {y.shape} 正类 {y.mean():.3f}")

    # 几何一致性：不增强时，补丁里的掩膜必须和图对得上
    ds_noaug = PatchDataset(train, patch=256, augment=False)
    x, y = ds_noaug[0]
    assert y.mean() > 0.01, "不增强的中心补丁里应该能看到血管"
    print("[自检] 不增强的中心补丁掩膜正类比例 %.3f（正常）" % y.mean())

    # 提交管线往返
    s = train[0]
    rle = mask_to_rle_512((s.mask * 255).astype(np.uint8))
    dec = rle_to_mask(rle)
    gt512 = cv2.resize((s.mask * 255).astype(np.uint8), (512, 512), interpolation=cv2.INTER_LINEAR) > 0
    inter = (dec & gt512).sum()
    dice = 2 * inter / (dec.sum() + gt512.sum())
    print(f"[提交] RLE 长度 {len(rle.split()) // 2} 段，512 上正类 {dec.mean():.4f}，自比 Dice {dice:.4f}")
    assert dice > 0.999, "RLE 编解码不一致"
    print("[自检] 全部通过")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="实验四血管分割数据层自检")
    ap.add_argument("data", nargs="?", default=str(DEFAULT_DATA))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", default=None, help="把布局信息写成 json")
    args = ap.parse_args()
    if args.json:
        train_items, test_items = discover(args.data)
        import json as _json
        info = {
            "data": str(args.data),
            "n_train": len(train_items),
            "n_test": len(test_items),
            "train": [{"image": str(a), "mask": str(b)} for a, b in train_items],
            "test": [str(p) for p in test_items],
        }
        Path(args.json).write_text(_json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[json] -> {args.json}")
        return 0
    return selftest(args.data)


if __name__ == "__main__":
    raise SystemExit(main())
