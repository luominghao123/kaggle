"""任务三（中心凹检测定位）数据层：布局自动探测、坐标归一化、letterbox、保位置的几何增强。

和任务一（图像分类）的区别
--------------------------
1. 标签不是「每个类别一个子目录」，而是一张表里的 **(x, y) 坐标**；
2. 增强必须**保位置**：翻转 / 旋转 / 缩放要同步变换坐标点，
   因此不能用 `RandomResizedCrop` 这类会破坏坐标的操作；
3. 训练与推理统一走 **letterbox**（等比缩放 + 居中补黑边）。
   若把任意长宽比的眼底图硬拉成正方形，x/y 两个方向比例尺不同，
   坐标回归会被系统性拉偏；letterbox 保证「原图坐标 ↔ 输入坐标」是同一个
   等比缩放 + 平移，可逆且不畸变。

内部统一表示
------------
管线内部所有坐标都是 **letterbox 之后正方形图上、按图边长归一化的 [0, 1] 坐标**。
`unit` 只描述**标签文件里**坐标的单位：
    * `normalized` —— 已经是 [0,1]（这类竞赛最常见）；
    * `pixel`     —— 原图像素坐标，需用每张图的 (w, h) 归一化。
提交时再按 `sample_submission.csv` 的量纲还原（见 `fovea_predict.py`）。
"""
from __future__ import annotations

import csv
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 张量版归一化常数（避免每个样本重复构造）
_MEAN_T = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD_T = torch.tensor(IMAGENET_STD).view(3, 1, 1)

# 列名猜测关键字（按优先级排列，匹配时先精确后包含）
ID_HINTS = ("image_id", "img_id", "imgid", "imageid", "filename", "file_name",
            "file", "path", "name", "picture", "photo", "id", "img", "image")
X_HINTS = ("fovea_x", "fovea_x_coord", "x_fovea", "center_x", "centre_x", "x_center",
           "xcoord", "coord_x", "x_coord", "cx", "fx", "x")
Y_HINTS = ("fovea_y", "fovea_y_coord", "y_fovea", "center_y", "centre_y", "y_center",
           "ycoord", "coord_y", "y_coord", "cy", "fy", "y")

# 长表提交里第一列的写法：81_Fovea_X / 81_Fovea_Y
_LONG_ID_RE = re.compile(r"^\s*(\d+)\s*[_\-\s]+(.+?)\s*$")


# ==========================================================================
# 布局探测
# ==========================================================================


@dataclass
class FoveaLayout:
    """一份中心凹数据集的可训练视图。"""

    labels_csv: Path
    id_column: str
    x_column: str
    y_column: str
    rows: list[tuple[Path, float, float]]      # (图片路径, 原始单位 x, 原始单位 y)
    unit: str                                  # 'normalized' | 'pixel'
    test_images: list[Path]
    sample_submission: Path | None = None
    submission_columns: list[str] = field(default_factory=list)
    image_index: dict[str, Path] = field(default_factory=dict)
    # 提交表形态：'wide' = (ID, x, y) 每图一行；'long' = (ImageID, value) 每图两行
    submission_fmt: str = "wide"
    submission_rows: list[list[str]] = field(default_factory=list)
    # 长表专用：每行对应 (图片路径, 轴 0=x/1=y)；无法解析的行存 None
    submission_pairs: list[tuple[Path, int] | None] = field(default_factory=list)

    @property
    def item_count(self) -> int:
        return len(self.rows)

    def summary(self) -> str:
        xs = np.array([r[1] for r in self.rows], dtype=float)
        ys = np.array([r[2] for r in self.rows], dtype=float)
        lines = [
            f"标签表        : {self.labels_csv}",
            f"  ID 列       : {self.id_column!r}",
            f"  坐标列      : x={self.x_column!r}  y={self.y_column!r}",
            f"  坐标单位    : {self.unit}"
            + ("（像素，提交前需按每张图尺寸换算）" if self.unit == "pixel" else "（已归一化）"),
            f"  训练样本数  : {self.item_count}",
            f"  x 范围      : [{xs.min():.4f}, {xs.max():.4f}]  mean={xs.mean():.4f}",
            f"  y 范围      : [{ys.min():.4f}, {ys.max():.4f}]  mean={ys.mean():.4f}",
            f"测试图片数    : {len(self.test_images)}",
            f"sample_submit : {self.sample_submission}",
            f"  提交表形态  : {self.submission_fmt}"
            + ("（长表：每图两行 _X / _Y）" if self.submission_fmt == "long" else ""),
        ]
        if self.submission_columns:
            lines.append(f"  提交列名    : {self.submission_columns}")
        if self.submission_fmt == "long" and self.submission_rows:
            lines.append(f"  提交行数    : {len(self.submission_rows)}"
                         f"（已解析 {sum(1 for p in self.submission_pairs if p)} 行）")
        return "\n".join(lines)


def _iter_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def _build_image_index(images: list[Path]) -> dict[str, Path]:
    """同时用「完整文件名」「去扩展名」「去前导零的编号」建索引。

    真实数据里常见标签表写 `data=1`、图片叫 `0001.jpg` 的情况，
    不做补零归一化就一行都对不上。
    """
    idx: dict[str, Path] = {}
    for p in images:
        for key in (p.name, p.stem, p.name.lower(), p.stem.lower()):
            idx.setdefault(key, p)
        s = p.stem
        if s.isdigit():
            idx.setdefault(str(int(s)), p)          # "0001" -> "1"
            idx.setdefault(f"{int(s):04d}", p)      # "1" -> "0001"
    return idx


def _read_csv_rows(path: Path, limit: int = 50000) -> tuple[list[str], list[list[str]]]:
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    return [], []
                rows: list[list[str]] = []
                for i, row in enumerate(reader):
                    if i >= limit:
                        break
                    if row:
                        rows.append(row)
                return [h.strip() for h in header], rows
        except UnicodeDecodeError:
            continue
    return [], []


def _as_float(s: str) -> float | None:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _pick_by_hint(columns: list[str], hints: tuple[str, ...]) -> str | None:
    low = {c: c.strip().lower().replace(" ", "_") for c in columns}
    for h in hints:                       # 先精确
        for c, lc in low.items():
            if lc == h:
                return c
    for h in hints:                       # 再包含
        for c, lc in low.items():
            if h in lc:
                return c
    return None


def parse_submission_spec(sample_sub: Path, image_index: dict[str, Path]) -> dict:
    """解析 sample_submission，判断是「宽表」还是「长表」。

    宽表：`ID, x, y`            每张图一行
    长表：`ImageID, value`      每张图两行，第一列形如 `81_Fovea_X` / `81_Fovea_Y`
          （2026task3 的真实数据就是这种，且 value 列是空的）
    """
    header, rows = _read_csv_rows(sample_sub)
    out = {"fmt": "wide", "header": header, "rows": rows,
           "test_images": [], "pairs": []}
    if not header:
        return out

    # ---- 长表判定：两列 + 首列全都形如 <编号>_<后缀> ----
    if len(header) == 2 and rows:
        probe = [(_LONG_ID_RE.match((r[0] or "").strip()) if r else None) for r in rows[:8]]
        if probe and all(probe):
            pairs: list[tuple[Path, int] | None] = []
            for r in rows:
                m = _LONG_ID_RE.match((r[0] or "").strip()) if r else None
                if not m:
                    pairs.append(None)
                    continue
                num, suffix = m.group(1), m.group(2).lower()
                path = image_index.get(num) or image_index.get(f"{int(num):04d}")
                axis = 1 if ("y" in suffix and "x" not in suffix) else 0
                pairs.append((path, axis) if path is not None else None)
            seen: list[Path] = []
            for pp in pairs:
                if pp and pp[0] not in seen:
                    seen.append(pp[0])
            out.update({"fmt": "long", "pairs": pairs, "test_images": seen})
            return out

    # ---- 宽表：按 ID 列匹配图片 ----
    id_col = _pick_by_hint(header, ID_HINTS) or header[0]
    j = header.index(id_col)
    seen2: list[Path] = []
    for r in rows:
        key = r[j].strip() if j < len(r) else ""
        if not key:
            continue
        p = image_index.get(key) or image_index.get(key.lower()) or \
            image_index.get(Path(key).stem)
        if p is not None and p not in seen2:
            seen2.append(p)
    out["test_images"] = seen2
    return out


def _analyse_labels_csv(csv_path: Path, image_index: dict[str, Path]) -> dict | None:
    """判断一个 csv 是否「图片 ID + 两个坐标列」，是则返回解析结果。"""
    header, rows = _read_csv_rows(csv_path)
    if len(header) < 3 or not rows:
        return None

    n = len(rows)
    col_vals: dict[str, list[str]] = {c: [] for c in header}
    for r in rows:
        for j, c in enumerate(header):
            col_vals[c].append(r[j].strip() if j < len(r) else "")

    # 1) ID 列：与图片索引命中比例最高者
    best_id, best_hit = None, 0.0
    for c in header:
        vals = [v for v in col_vals[c] if v]
        if not vals:
            continue
        hit = sum(1 for v in vals if v in image_index or v.lower() in image_index)
        frac = hit / len(vals)
        if frac > best_hit:
            best_id, best_hit = c, frac
    if best_id is None or best_hit < 0.5:
        return None

    # 2) 数值列
    numeric: list[str] = []
    for c in header:
        if c == best_id:
            continue
        ok = sum(1 for v in col_vals[c] if _as_float(v) is not None)
        if ok >= 0.9 * n:
            numeric.append(c)
    if len(numeric) < 2:
        return None

    # 3) 选 x / y：先看列名提示，再退化为「前两个数值列」
    x_col = _pick_by_hint(numeric, X_HINTS)
    y_col = _pick_by_hint(numeric, Y_HINTS)
    if x_col is not None and x_col == y_col:
        x_col = None
    if x_col is None or y_col is None:
        rest = [c for c in numeric if c not in (x_col, y_col)]
        if x_col is None and rest:
            x_col = rest.pop(0)
        if y_col is None and rest:
            y_col = rest.pop(0)
    if x_col is None or y_col is None or x_col == y_col:
        return None

    # 4) 组装样本行
    parsed: list[tuple[Path, float, float]] = []
    for i in range(n):
        key = col_vals[best_id][i]
        if not key:
            continue
        path = image_index.get(key) or image_index.get(key.lower())
        if path is None:
            continue
        x, y = _as_float(col_vals[x_col][i]), _as_float(col_vals[y_col][i])
        if x is None or y is None:
            continue
        parsed.append((path, x, y))
    if not parsed:
        return None

    xs = np.array([r[1] for r in parsed])
    ys = np.array([r[2] for r in parsed])
    unit = "normalized" if max(xs.max(), ys.max()) <= 1.5 else "pixel"

    return {
        "csv": csv_path, "id_col": best_id, "x_col": x_col, "y_col": y_col,
        "rows": parsed, "unit": unit, "hit": best_hit,
    }


def discover(root: str | Path) -> FoveaLayout:
    """在 root 下自动探测中心凹数据集（标签表 + 训练图 + 测试图）。"""
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"数据根目录不存在: {root}")

    images = _iter_images(root)
    if not images:
        raise RuntimeError(f"{root} 下一张图片都没找到。")
    image_index = _build_image_index(images)

    sample_sub: Path | None = None
    cands: list[Path] = []
    for p in sorted(root.rglob("*.csv")):
        name = p.name.lower()
        if "sample" in name and "submission" in name:
            sample_sub = sample_sub or p
            continue
        cands.append(p)

    best = None
    for c in cands:
        info = _analyse_labels_csv(c, image_index)
        if info is None:
            continue
        hinted = (_pick_by_hint([info["x_col"]], X_HINTS) is not None
                  and _pick_by_hint([info["y_col"]], Y_HINTS) is not None)
        score = (len(info["rows"]), 1 if hinted else 0, info["hit"])
        if best is None or score > best[0]:
            best = (score, info)

    if best is None:
        raise RuntimeError(
            f"未在 {root} 下找到「图片 ID + 两个坐标列」的标签表。\n"
            "请确认数据已解压；或用 --labels/--img-dir/--test-dir 显式指定。"
        )
    info = best[1]

    # 测试图：优先按 sample_submission 解析（支持长/宽两种表），其次取「未被标签引用」的图
    train_paths = {r[0] for r in info["rows"]}
    spec = {"fmt": "wide", "header": [], "rows": [], "test_images": [], "pairs": []}
    if sample_sub is not None:
        spec = parse_submission_spec(sample_sub, image_index)
    test_images: list[Path] = list(spec["test_images"])
    if not test_images:
        test_images = [p for p in images if p not in train_paths]

    return FoveaLayout(
        labels_csv=info["csv"],
        id_column=info["id_col"],
        x_column=info["x_col"],
        y_column=info["y_col"],
        rows=info["rows"],
        unit=info["unit"],
        test_images=test_images,
        sample_submission=sample_sub,
        submission_columns=spec["header"],
        image_index=image_index,
        submission_fmt=spec["fmt"],
        submission_rows=spec["rows"],
        submission_pairs=spec["pairs"],
    )


# ==========================================================================
# letterbox 与单位换算
# ==========================================================================


def letterbox_params(w: int, h: int, size: int) -> tuple[float, int, int]:
    """返回 (缩放比 s, 左补边 ox, 上补边 oy)。"""
    s = min(size / w, size / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return s, (size - nw) // 2, (size - nh) // 2


def letterbox(img: Image.Image, size: int, fill=(0, 0, 0)):
    """等比缩放 + 居中补边到 size×size。返回 (新图, s, ox, oy)。"""
    w, h = img.size
    s, ox, oy = letterbox_params(w, h, size)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img.resize((nw, nh), Image.BICUBIC), (ox, oy))
    return canvas, s, ox, oy


def label_to_norm(x: float, y: float, size: tuple[int, int], unit: str,
                  s: float, ox: int, oy: int, out_size: int) -> tuple[float, float]:
    """标签单位坐标 -> letterbox 归一化坐标 [0,1]。"""
    w, h = size
    px = x * w if unit == "normalized" else x
    py = y * h if unit == "normalized" else y
    return (px * s + ox) / out_size, (py * s + oy) / out_size


def norm_to_label(u: float, v: float, size: tuple[int, int], unit: str,
                  s: float, ox: int, oy: int, out_size: int) -> tuple[float, float]:
    """letterbox 归一化坐标 -> 标签单位坐标（预测还原用）。"""
    w, h = size
    px = (u * out_size - ox) / s
    py = (v * out_size - oy) / s
    if unit == "normalized":
        return px / w, py / h
    return px, py


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


# ==========================================================================
# 数据集
# ==========================================================================


class LetterboxCache:
    """全局共享的 letterbox 图缓存。

    本机沙箱禁止命名管道，DataLoader 的 num_workers 只能是 0，
    读图与原图缩放全部落在主进程；K 折训练时同一张图会被多个折反复读取，
    因此用「进程内共享字典」缓存 letterbox 之后的图，避免重复解码大图（眼底图常有 2000+ 像素）。
    """

    def __init__(self, img_size: int, budget_mb: int = 3072):
        self.img_size = img_size
        self.budget = budget_mb * 1024 * 1024
        self.store: dict[str, Image.Image] = {}
        self.bytes = 0
        self.full = False
        self.hits = 0
        self.misses = 0

    def get(self, path: str | Path) -> Image.Image:
        key = str(path)
        img = self.store.get(key)
        if img is not None:
            self.hits += 1
            return img

        self.misses += 1
        with Image.open(key) as im:
            lb, _, _, _ = letterbox(im.convert("RGB"), self.img_size)
        if not self.full:
            nbytes = self.img_size * self.img_size * 3
            if self.bytes + nbytes <= self.budget:
                self.store[key] = lb
                self.bytes += nbytes
            else:
                self.full = True
        return lb

    def info(self) -> str:
        total = self.hits + self.misses
        rate = 100.0 * self.hits / total if total else 0.0
        return (f"letterbox 缓存 {len(self.store)} 张 / {self.bytes / 1024**2:.0f} MB"
                f"，命中率 {rate:.1f}%" + ("（已达上限）" if self.full else ""))


class FoveaDataset(Dataset):
    """中心凹定位数据集。

    `xy` 是 **letterbox 归一化坐标 [0,1]**（由 `build_samples_from_layout` 生成）。
    训练时做保位置的随机几何变换（旋转 / 缩放 / 平移 / 水平翻转），
    坐标点用同一个前向矩阵同步映射——这是与分类任务增强最本质的不同。
    """

    def __init__(
        self,
        paths: list[Path],
        xy: np.ndarray | None,
        img_size: int,
        *,
        train: bool = False,
        rotate: float = 12.0,
        scale: tuple[float, float] = (0.88, 1.12),
        shift: float = 0.05,
        hflip: float = 0.5,
        color_jitter: float = 0.25,
        blur_p: float = 0.15,
        cache: LetterboxCache | None = None,
    ):
        self.paths = [Path(p) for p in paths]
        self.xy = None if xy is None else np.asarray(xy, dtype=np.float32)
        self.img_size = img_size
        self.train = train
        self.rotate = rotate
        self.scale = scale
        self.shift = shift
        self.hflip = hflip
        self.color_jitter = color_jitter
        self.blur_p = blur_p
        self.cache = cache

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, i: int) -> Image.Image:
        if self.cache is not None:
            return self.cache.get(self.paths[i])
        with Image.open(self.paths[i]) as im:
            lb, _, _, _ = letterbox(im.convert("RGB"), self.img_size)
        return lb

    def __getitem__(self, i: int):
        img = self._load(i)
        xy = None if self.xy is None else self.xy[i].copy()

        # 一次性转成 float 张量，之后所有增强都在张量上做：
        # PIL 的 adjust_* / gaussian_blur 每次都要整图拷贝 + 类型转换，
        # 单进程（num_workers=0）下这是训练速度的主要瓶颈。
        t = tvf.pil_to_tensor(img).float().div_(255.0)

        if self.train:
            t, xy = self._random_geom(t, xy)
            t = self._random_color(t)

        t = (t - _MEAN_T) / _STD_T
        if xy is None:
            return t, torch.zeros(2), str(self.paths[i])
        return t, torch.from_numpy(np.asarray(xy, dtype=np.float32)), str(self.paths[i])

    def _random_color(self, t: torch.Tensor) -> torch.Tensor:
        """张量版颜色增强：亮度 / 对比度 / 饱和度 / gamma / 轻微模糊。"""
        j = self.color_jitter
        if j > 0:
            t = t * (1 + random.uniform(-j, j))
            c = 1 + random.uniform(-j, j)
            t = (t - t.mean(dim=(1, 2), keepdim=True)) * c + t.mean(dim=(1, 2), keepdim=True)
            s = 1 + random.uniform(-j, j)
            gray = t.mean(dim=0, keepdim=True)
            t = (t - gray) * s + gray
            t = t.clamp_(0.0, 1.0).pow_(random.uniform(0.85, 1.18))
        if self.blur_p and random.random() < self.blur_p:
            # 注意：GaussianBlur 在 torchvision.transforms.functional 里（不是 torch.nn.functional），
            # 但它直接支持张量输入，因此不需要退回 PIL。
            t = tvf.gaussian_blur(t.unsqueeze(0), kernel_size=3,
                                  sigma=random.uniform(0.3, 1.2))[0]
        return t

    def _random_geom(self, t: torch.Tensor, xy: np.ndarray | None):
        """保位置随机几何变换：图像与坐标共用同一前向矩阵（在张量上完成）。"""
        ang = random.uniform(-self.rotate, self.rotate) if self.rotate else 0.0
        sc = random.uniform(*self.scale)
        tr = np.array([random.uniform(-self.shift, self.shift),
                       random.uniform(-self.shift, self.shift)], dtype=np.float64) * 2.0
        do_flip = self.hflip > 0 and random.random() < self.hflip

        rad = math.radians(ang)
        A = np.array([[math.cos(rad), -math.sin(rad)],
                      [math.sin(rad), math.cos(rad)]], dtype=np.float64) * sc
        if do_flip:
            A = A @ np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

        fwd = np.eye(3)
        fwd[:2, :2] = A
        fwd[:2, 2] = tr
        theta = torch.from_numpy(np.linalg.inv(fwd)[:2]).float().unsqueeze(0)

        grid = F.affine_grid(theta, (1,) + tuple(t.shape), align_corners=False)
        t = F.grid_sample(t.unsqueeze(0), grid, align_corners=False,
                          padding_mode="zeros")[0]

        if xy is not None:
            p = np.array([xy[0] * 2 - 1, xy[1] * 2 - 1, 1.0], dtype=np.float64)
            q = fwd @ p
            xy = np.array([(q[0] + 1) / 2, (q[1] + 1) / 2], dtype=np.float32)
        return t, xy


# ==========================================================================
# 样本构造
# ==========================================================================


def build_samples_from_layout(layout: FoveaLayout, img_size: int):
    """标签表 -> (路径列表, letterbox 归一化坐标, 原图尺寸表)。"""
    paths: list[Path] = []
    xy: list[tuple[float, float]] = []
    sizes: dict[Path, tuple[int, int]] = {}
    out_of_range = 0
    for path, x, y in layout.rows:
        size = sizes.get(path)
        if size is None:
            size = image_size(path)
            sizes[path] = size
        s, ox, oy = letterbox_params(size[0], size[1], img_size)
        u, v = label_to_norm(x, y, size, layout.unit, s, ox, oy, img_size)
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            out_of_range += 1
        paths.append(path)
        xy.append((float(u), float(v)))
    if out_of_range:
        print(f"[!] {out_of_range} 个坐标落在图像范围外"
              "（可能是标注越界，或坐标单位判断错了）")
    return paths, np.asarray(xy, dtype=np.float32), sizes


# ==========================================================================
# 自检
# ==========================================================================


def _selftest_geometry(verbose: bool = True, trials: int = 12) -> bool:
    """验证「图像变换」与「坐标点变换」严格一致：亮点画在标注处，变换后应重合。"""
    S = 128
    ok_all = True
    rng = random.Random(0)
    ds = FoveaDataset([], None, S, train=True, rotate=30, scale=(0.9, 1.1),
                      shift=0.05, hflip=0.5, color_jitter=0.0, blur_p=0.0)
    for trial in range(trials):
        px, py = int(rng.uniform(0.2, 0.8) * S), int(rng.uniform(0.2, 0.8) * S)
        t = torch.zeros(3, S, S)
        t[:, py, px] = 1.0
        xy = np.array([(px + 0.5) / S, (py + 0.5) / S], dtype=np.float32)

        t2, xy2 = ds._random_geom(t, xy)
        flat = t2[0].numpy()
        yy, xx = np.unravel_index(int(np.argmax(flat)), flat.shape)
        got = np.array([(xx + 0.5) / S, (yy + 0.5) / S])
        err = float(np.abs(got - xy2).max())
        ok = err < 2.0 / S
        ok_all &= ok
        if verbose:
            print(f"  trial {trial:2d}: 坐标变换后 {xy2.round(3)}  亮点实际 {got.round(3)}"
                  f"  误差 {err * S:.2f}px  {'OK' if ok else 'FAIL'}")
    return ok_all


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        print("几何变换自检（坐标点必须与图像同步）：")
        good = _selftest_geometry()
        print("\n结果:", "全部通过" if good else "存在失败项")
        sys.exit(0 if good else 1)

    root = sys.argv[1] if len(sys.argv) > 1 else "data"
    layout = discover(root)
    print(layout.summary())
    paths, xy, sizes = build_samples_from_layout(layout, 512)
    print(f"\nletterbox 归一化坐标示例: {xy[:3].round(4).tolist()}")
    print(f"原图尺寸示例: {list(sizes.items())[:3]}")
