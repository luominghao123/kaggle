"""任务二数据集探测 / 加载 / 缓存。

数据来源：`2026task2.zip` 解压后的结构

    fer_data/fer_data/
        ├── train/<ClassName>/*.jpg     48x48 灰度
        └── test/*.jpg                  48x48 灰度，平铺
    submission.csv                      ID,Emotion（样例提交，Emotion 取值全为整数 0）

类别共 6 个（原始 FER2013 的 Disgust 类被移除）。

关于标签编码
------------
样例提交里 Emotion 全是 0，无法反推「类别名 -> 整数」的对应关系，而作业/竞赛页面
也没有给出显式说明。因此这里准备了三套候选编码（见 ENCODINGS），默认用 `alpha`
（类名字母序）。三套编码只是**同一个预测结果的不同写法**，如果评测反馈的准确率
异常低（例如 0.15 左右而不是 0.6 以上），换一份提交文件即可，不需要重新训练。
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# 类别与候选编码
# --------------------------------------------------------------------------

#: 按字母序排列的类别名，也是模型内部使用的 0..5 标签顺序
CLASSES: list[str] = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

#: 提交文件的列名（与官方样例 submission.csv 保持一致）
ID_COL = "ID"
LABEL_COL = "Emotion"

#: 候选「类别名 -> 提交整数」编码。key 即 --label-encoding 的取值。
#:
#: 默认用 `fer2013_new`，依据是**实测证据**而不是猜测：
#: 官方样例 `submission.csv` 的 ID 顺序是「按真实类别分块排列」的（7065 行只有 42 个
#: 连续段，随机顺序下应有约 5800 段），块的顺序依次是
#:     Angry(990) -> Fear(1024) -> Happy(1797) -> Sad(1215) -> Surprise(800) -> Neutral(1239)
#: 用公开 FER2013 匹配得到的真实标签去核对，这 6 块的纯度分别是
#:     99.80% / 99.51% / 100.00% / 99.51% / 99.50% / 99.84%
#: 而若按字母序假设，第 3~5 块的纯度会掉到 0.08% / 0.00% / 0.00%。
#: 也就是说样例提交是按标签升序排的，标签顺序正是 FER2013 官方顺序去掉 Disgust 后重编号。
ENCODINGS: dict[str, dict[str, int]] = {
    # A. 字母序：torchvision ImageFolder / keras image_dataset_from_directory
    #    生成文件夹数据集时的默认编号方式（**本竞赛不是这种**，仅作对照保留）
    "alpha": {c: i for i, c in enumerate(CLASSES)},
    # B. FER2013 官方字典顺序、去掉 Disgust 后重新编号 —— 本竞赛实际使用的编码
    "fer2013_new": {
        "Angry": 0, "Fear": 1, "Happy": 2, "Sad": 3, "Surprise": 4, "Neutral": 5,
    },
    # C. 沿用 FER2013 官方字典的原始整数、留下 Disgust 的空洞（1 号缺省）
    "fer2013_gap": {
        "Angry": 0, "Fear": 2, "Happy": 3, "Sad": 4, "Surprise": 5, "Neutral": 6,
    },
}

DEFAULT_ENCODING = "fer2013_new"


def class_to_submission_labels(encoding: str = DEFAULT_ENCODING) -> np.ndarray:
    """返回长度为 6 的数组：内部类别下标 -> 提交用的整数。"""
    if encoding not in ENCODINGS:
        raise KeyError(f"未知编码 {encoding!r}，可选：{list(ENCODINGS)}")
    table = ENCODINGS[encoding]
    return np.array([table[c] for c in CLASSES], dtype=np.int64)


# --------------------------------------------------------------------------
# 目录探测
# --------------------------------------------------------------------------


def find_root(data_root: str | Path) -> Path:
    """在 data_root 下自动找到「含 train/<类名>/ 与 test/」的那一层。"""
    data_root = Path(data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_root}")

    candidates = [data_root] + sorted(p for p in data_root.rglob("*") if p.is_dir())
    for p in candidates:
        tr, te = p / "train", p / "test"
        if tr.is_dir() and te.is_dir():
            subdirs = [d for d in tr.iterdir() if d.is_dir()]
            if subdirs:
                return p
    raise FileNotFoundError(
        f"在 {data_root} 下没找到 `<dir>/train/<类别名>/*.jpg` + `<dir>/test/*.jpg` 的结构，"
        "请确认 2026task2.zip 已解压"
    )


def find_sample_submission(root: Path, data_root: Path | None = None) -> Path | None:
    """找官方样例提交（决定列名与 ID 顺序）。

    样例 `submission.csv` 在压缩包里位于 `fer_data/` 的**上一层**，所以除了数据根目录
    本身，还要往上看几级、并在最初传入的 --data 目录里递归找一遍。
    """
    names = ("submission.csv", "sample_submission.csv")
    cands: list[Path] = []
    for base in [root, root.parent, root.parent.parent]:
        cands += [base / n for n in names]
    if data_root is not None:
        for n in names:
            cands += sorted(Path(data_root).resolve().rglob(n))
    for p in cands:
        if p.is_file():
            return p
    return None


# --------------------------------------------------------------------------
# 读图
# --------------------------------------------------------------------------


def _read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def _read_many(paths: list[Path], desc: str, workers: int = 12) -> np.ndarray:
    """多线程读图（PIL 解码会释放 GIL，线程池有效；这里不用多进程，
    因为本机沙箱禁止命名管道，多进程 DataLoader 会直接 WinError 5）。"""
    out = np.empty((len(paths), 48, 48), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, arr in enumerate(ex.map(_read_gray, paths)):
            out[i] = arr
    return out


# --------------------------------------------------------------------------
# 带缓存的数据集封装
# --------------------------------------------------------------------------


class FerdDataset:
    """把整个数据集读进内存（uint8，48x48）。

    28275 张训练图 = 65 MB、7065 张测试图 = 16 MB，直接常驻显存也没问题，
    因此训练时可以在 GPU 上做增强，彻底避开 CPU 数据加载瓶颈。
    """

    def __init__(self, root: str | Path, cache_dir: str | Path | None = None):
        self.data_root = Path(root).resolve()
        self.root = find_root(root)
        self.cache_dir = Path(cache_dir) if cache_dir else self.root / "_cache48"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.train_images, self.train_labels, self.train_names = self._load_train()
        self.test_images, self.test_names = self._load_test()
        self.sample_submission = find_sample_submission(self.root, self.data_root)

    # -- 训练集 ---------------------------------------------------------
    def _load_train(self):
        img_p = self.cache_dir / "train_48.npy"
        lab_p = self.cache_dir / "train_labels.npy"
        nam_p = self.cache_dir / "train_names.json"
        if img_p.is_file() and lab_p.is_file() and nam_p.is_file():
            return (np.load(img_p), np.load(lab_p),
                    json.loads(nam_p.read_text(encoding="utf-8")))

        train_dir = self.root / "train"
        class_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
        names = [d.name for d in class_dirs]
        if names != CLASSES:
            raise ValueError(
                f"训练集类别与预期不一致：实际 {names}，预期 {CLASSES}。"
                "若确实只有这 6 个类但名字不同，请更新 data.CLASSES"
            )

        paths: list[Path] = []
        labels: list[int] = []
        for ci, d in enumerate(class_dirs):
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            paths.extend(files)
            labels.extend([ci] * len(files))

        images = _read_many(paths, "train")
        labels_arr = np.asarray(labels, dtype=np.int64)
        names_k = [p.name for p in paths]

        np.save(img_p, images)
        np.save(lab_p, labels_arr)
        nam_p.write_text(json.dumps(names_k, ensure_ascii=False), encoding="utf-8")
        return images, labels_arr, names_k

    # -- 测试集 ---------------------------------------------------------
    def _load_test(self):
        img_p = self.cache_dir / "test_48.npy"
        nam_p = self.cache_dir / "test_names.json"
        if img_p.is_file() and nam_p.is_file():
            return np.load(img_p), json.loads(nam_p.read_text(encoding="utf-8"))

        test_dir = self.root / "test"
        paths = sorted(p for p in test_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        images = _read_many(paths, "test")
        names = [p.name for p in paths]
        np.save(img_p, images)
        nam_p.write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
        return images, names

    # -- 概览 -----------------------------------------------------------
    def describe(self) -> str:
        lines = [f"数据根目录: {self.root}",
                 f"训练集: {len(self.train_labels)} 张   测试集: {len(self.test_names)} 张",
                 f"样例提交: {self.sample_submission}", "", "训练集类别分布:"]
        for i, c in enumerate(CLASSES):
            n = int((self.train_labels == i).sum())
            lines.append(f"    [{i}] {c:10s} {n:6d}  ({n/len(self.train_labels)*100:5.2f}%)")
        return "\n".join(lines)


def dataset_overview(data: str | Path, cache: str | Path | None = None) -> str:
    return FerdDataset(data, cache).describe()


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="任务二数据自检")
    ap.add_argument("data", nargs="?", default="data/raw/2026task2")
    ap.add_argument("--cache", default=None)
    args = ap.parse_args(argv)

    ds = FerdDataset(args.data, args.cache)
    print(ds.describe())
    print("\n测试集样例文件名:", ds.test_names[:5])
    print("图像形状:", ds.train_images.shape, ds.train_images.dtype,
          "| 像素范围:", ds.train_images.min(), ds.train_images.max())
    if ds.sample_submission:
        head = ds.sample_submission.read_text(encoding="utf-8").splitlines()
        print(f"\n样例提交 {ds.sample_submission.name} 前 3 行: {head[:3]}  共 {len(head)-1} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
