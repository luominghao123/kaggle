"""数据层：目录自动探测、数据集封装、数据增强 / TTA。

设计要点
--------
1. 目录结构自动探测：不假设数据一定解压成 `data/train`，而是递归搜索
   形如 `.../train/<类别名>/*.png` 的结构，兼容 `data/raw/new_plant/train/...`
   这类多层嵌套（去年课程就是这么放的）。
2. 提交格式对齐 `sample_submission.csv`：若存在则直接采用它的 ID 顺序与列名。
3. 图像读取统一走 PIL，自动兼容 png/jpg/jpeg/bmp/tif 与灰度图。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# 迁移学习统一使用 ImageNet 统计量
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# 目录探测
# --------------------------------------------------------------------------


@dataclass
class DataLayout:
    """描述一份数据的物理布局。"""

    train_dir: Path
    test_dir: Path | None
    classes: list[str]
    sample_submission: Path | None = None
    class_to_idx: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.class_to_idx:
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    @property
    def idx_to_class(self) -> dict[int, str]:
        return {i: c for c, i in self.class_to_idx.items()}

    def summary(self) -> str:
        lines = [
            f"train_dir        : {self.train_dir}",
            f"test_dir         : {self.test_dir}",
            f"sample_submission: {self.sample_submission}",
            f"类别数           : {len(self.classes)}",
        ]
        for c in self.classes:
            lines.append(f"    - {c}")
        return "\n".join(lines)


def _is_class_dir(d: Path) -> bool:
    """判断目录是否符合 <类别>/*.png 结构。"""
    if not d.is_dir():
        return False
    subs = [x for x in d.iterdir() if x.is_dir()]
    if not subs:
        return False
    # 至少一半子目录里含有图片
    with_imgs = 0
    for s in subs:
        for f in s.iterdir():
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                with_imgs += 1
                break
    return with_imgs >= max(2, len(subs) // 2)


def _find_sample_submission(root: Path) -> Path | None:
    cands = [
        p
        for p in root.rglob("*.csv")
        if "sample" in p.name.lower() and "submission" in p.name.lower()
    ]
    if cands:
        # 取最浅层的那个
        cands.sort(key=lambda p: len(p.parts))
        return cands[0]
    return None


def _find_test_dir(root: Path, train_dir: Path) -> Path | None:
    """测试集是「一层平铺图片」的目录，或含 test 关键字的目录。"""
    # 1) train 的同级 test 目录
    for name in ("test", "Test", "TEST", "test1", "test_set"):
        p = train_dir.parent / name
        if p.is_dir():
            return p
    # 2) 全盘找名为 test 的目录，且里面直接放图片
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in ("test", "test_set") and _has_images_flat(p):
            return p
    # 3) 兜底：任何平铺图片且不含子目录的目录
    for p in sorted(root.rglob("*")):
        if p.is_dir() and _has_images_flat(p) and p != train_dir:
            return p
    return None


def _has_images_flat(d: Path) -> bool:
    try:
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                return True
    except (PermissionError, OSError):
        return False
    return False


def discover(root: str | Path) -> DataLayout:
    """在 root 下自动探测数据布局。

    优先寻找「子目录名含 train」且符合类别目录结构的目录。
    """
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"数据根目录不存在: {root}")

    candidates: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_dir():
            continue
        if p.name.lower() in ("test", "test_set", "__macosx"):
            continue
        if _is_class_dir(p):
            # 只保留「父目录名含 train」或名字含 train 的，优先级更高
            candidates.append(p)

    if not candidates:
        raise RuntimeError(
            f"未在 {root} 下找到形如 <dir>/<类别名>/*.png 的训练集结构。\n"
            "请确认数据已解压，且至少有两层：训练集目录 + 类别子目录。"
        )

    def score(p: Path) -> tuple[int, int]:
        name_bonus = 0 if "train" in p.name.lower() else 1
        if "train" in p.parent.name.lower():
            name_bonus = 0
        # 类别数越多越可信；层数越浅越优先
        n_cls = len([x for x in p.iterdir() if x.is_dir()])
        return (name_bonus, -n_cls, len(p.parts))

    candidates.sort(key=score)
    train_dir = candidates[0]

    classes = sorted([x.name for x in train_dir.iterdir() if x.is_dir()])
    test_dir = _find_test_dir(root, train_dir)
    sample = _find_sample_submission(root)

    return DataLayout(
        train_dir=train_dir,
        test_dir=test_dir,
        classes=classes,
        sample_submission=sample,
    )


# --------------------------------------------------------------------------
# 数据集
# --------------------------------------------------------------------------


def list_train_samples(
    train_dir: Path, class_to_idx: dict[str, int], rng_seed: int | None = None
) -> tuple[list[str], list[int]]:
    """返回 (图片路径列表, 标签列表)。"""
    paths: list[str] = []
    labels: list[int] = []
    for cls, idx in class_to_idx.items():
        cls_dir = train_dir / cls
        if not cls_dir.is_dir():
            continue
        files = sorted(
            f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS
        )
        for f in files:
            paths.append(str(f))
            labels.append(idx)
    if rng_seed is not None:
        order = np.random.RandomState(rng_seed).permutation(len(paths))
        paths = [paths[i] for i in order]
        labels = [labels[i] for i in order]
    return paths, labels


def list_test_samples(test_dir: Path) -> list[str]:
    files = sorted(
        f for f in test_dir.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS
    )
    return [str(f) for f in files]


class PlantDataset(Dataset):
    """通用图像分类数据集：读图 -> 变换 -> (tensor, label)。

    为什么带内存缓存
    ----------------
    本机沙箱禁止进程间命名管道，DataLoader 的 num_workers 只能是 0，
    读图与数据增强全部落在主进程上。把「解码后的图」缓存进内存，
    可以避免每个 epoch 重复解码（小数据集下这是主要瓶颈）。
    缓存会按 cache_budget_mb 控制上限，超预算后自动退化为按需解码。
    """

    def __init__(
        self,
        paths: list[str],
        labels: list[int] | None,
        transform=None,
        *,
        cache: bool = True,
        cache_max_side: int | None = 576,
        cache_budget_mb: int = 2048,
    ):
        self.paths = paths
        self.labels = labels
        self.transform = transform

        self._use_cache = cache and cache_max_side is not None
        self._cache_max_side = cache_max_side
        self._cache_budget = cache_budget_mb * 1024 * 1024
        self._cache: list = [None] * len(paths) if self._use_cache else None
        self._cache_bytes = 0
        self._cache_full = False

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, i: int):
        if self._cache is not None and self._cache[i] is not None:
            return self._cache[i]

        img = Image.open(self.paths[i])
        img = img.convert("RGB")

        if self._use_cache and not self._cache_full:
            w, h = img.size
            if max(w, h) > self._cache_max_side:
                s = self._cache_max_side / max(w, h)
                # 用 BICUBIC 下采样后再缓存；对 384 训练仍有足够细节
                img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
            nbytes = img.size[0] * img.size[1] * 3
            if self._cache_bytes + nbytes <= self._cache_budget:
                self._cache[i] = img
                self._cache_bytes += nbytes
            else:
                self._cache_full = True  # 预算用尽，后续按需解码
        return img

    def __getitem__(self, i: int):
        img = self._load(i)
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is None:
            return img, self.paths[i]
        return img, self.labels[i]

    def cache_info(self) -> str:
        if not self._use_cache:
            return "未启用内存缓存"
        n = sum(1 for x in self._cache if x is not None)
        return (f"内存缓存 {n}/{len(self.paths)} 张，"
                f"占用 {self._cache_bytes/1024**2:.0f} MB" + ("（已达预算上限）" if self._cache_full else ""))


# --------------------------------------------------------------------------
# 数据增强
# --------------------------------------------------------------------------


def build_train_transform(
    img_size: int,
    *,
    scale: tuple[float, float] = (0.45, 1.0),
    ratio: tuple[float, float] = (0.8, 1.25),
    color_jitter: float = 0.3,
    re_prob: float = 0.25,
    use_randaug: bool = True,
    randaug_ops: int = 2,
    randaug_mag: int = 9,
):
    """训练增强。

    植物幼苗是俯拍图，上下/左右翻转与 90° 旋转都符合真实分布，
    因此这里额外加入垂直翻转与随机 90° 旋转。
    """
    ops: list = [
        T.RandomResizedCrop(img_size, scale=scale, ratio=ratio, antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        # 以 1/2 概率做一次 90/180/270 度旋转
        T.RandomApply(
            [
                T.RandomChoice(
                    [
                        T.RandomRotation((90, 90)),
                        T.RandomRotation((180, 180)),
                        T.RandomRotation((270, 270)),
                    ]
                )
            ],
            p=0.5,
        ),
    ]
    if use_randaug:
        ops.append(T.RandAugment(num_ops=randaug_ops, magnitude=randaug_mag))
    else:
        ops.append(T.ColorJitter(brightness=color_jitter, contrast=color_jitter,
                                 saturation=color_jitter, hue=0.05))
    ops += [
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        # 注意：torchvision>=0.17 的 RandomErasing 已移除 mode 参数
        T.RandomErasing(p=re_prob, value="random"),
    ]
    return T.Compose(ops)


def build_eval_transform(img_size: int, crop_pct: float = 0.9):
    """验证 / 测试：短边缩放到 img_size/crop_pct，再中心裁剪。"""
    resize = int(round(img_size / crop_pct))
    return T.Compose(
        [
            T.Resize(resize, antialias=True),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_tta_transforms(
    img_size: int,
    crop_pct: float = 0.9,
    extra_full_view: bool = True,
) -> list[tuple[str, object]]:
    """返回 [(名字, transform)]，用于测试时增强。

    默认 8 个视角 = 7 个几何视角（原图 / 水平翻转 / 垂直翻转 / 双翻转 / 三个 90° 旋转）
    + 1 个整图视角（不做中心裁剪，提供尺度多样性）。
    对俯拍植物图来说这些变换都是「语义不变」的。
    """
    post = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    def make(flip_h: bool, flip_v: bool, rot: int, cp: float):
        ops = [
            T.Resize(int(round(img_size / cp)), antialias=True),
            T.CenterCrop(img_size),
        ]
        if flip_h:
            ops.append(T.RandomHorizontalFlip(p=1.0))
        if flip_v:
            ops.append(T.RandomVerticalFlip(p=1.0))
        if rot:
            ops.append(T.RandomRotation((rot, rot)))
        return T.Compose(ops + post)

    views: list[tuple[str, object]] = []
    for name, fh, fv, rot in (
        ("id", False, False, 0),
        ("hflip", True, False, 0),
        ("vflip", False, True, 0),
        ("hvflip", True, True, 0),
        ("rot90", False, False, 90),
        ("rot180", False, False, 180),
        ("rot270", False, False, 270),
    ):
        views.append((name, make(fh, fv, rot, crop_pct)))

    if extra_full_view:
        views.append(("full", make(False, False, 0, 1.0)))

    return views


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data"
    layout = discover(root)
    print(layout.summary())

    paths, labels = list_train_samples(layout.train_dir, layout.class_to_idx)
    print(f"\n训练图片总数: {len(paths)}")
    counts = np.bincount(labels, minlength=len(layout.classes))
    for c, i in sorted(layout.class_to_idx.items(), key=lambda kv: kv[1]):
        print(f"    {c:32s} idx={i:2d}  n={counts[i]}")

    if layout.test_dir:
        test_paths = list_test_samples(layout.test_dir)
        print(f"\n测试图片总数: {len(test_paths)}")
        print("    示例:", [os.path.basename(p) for p in test_paths[:5]])
    else:
        print("\n[警告] 未找到测试集目录")
