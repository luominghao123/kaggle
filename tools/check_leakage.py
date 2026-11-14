"""数据泄漏自检：检测训练集与测试集之间是否存在重复 / 近似重复图片。

为什么要查
----------
如果测试图片和训练图片是同一张（或只是换了编码/尺寸的副本），模型只要"背答案"
就能拿高分。这种分数是虚的：既不代表模型真实能力，也说明竞赛划分有问题。
本脚本把这件事查清楚，避免自欺欺人。

两层检测
--------
1. 字节级精确重复：直接算文件 MD5。能抓住"同一个文件复制两份"。
2. 感知近似重复：把图片转灰度、缩到 9x8，做水平相邻像素差分得到 dHash（64 bit），
   比 Hamming 距离。能抓住"同一张图重新压缩/缩放/改格式"的情况。
   阈值 <= 4 bit 视为近似重复（经验值，越小越严格）。

用法:
    .\\run.ps1 tools\\check_leakage.py
    .\\run.ps1 tools\\check_leakage.py --max-dist 2      # 更严格
    .\\run.ps1 tools\\check_leakage.py --export dup.csv  # 导出重复对
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dhash(path: Path, hash_size: int = 8) -> int | None:
    """差异哈希：对缩放后的灰度图做水平差分，得到 hash_size*hash_size bit。"""
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
            a = np.asarray(im, dtype=np.int16)
    except Exception:
        return None
    diff = a[:, 1:] > a[:, :-1]  # (8, 8) bool
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def collect(root: Path):
    """返回 [(split, 路径)]，split 取 train 的类别名或 'test'。"""
    items = []
    train_dir = root / "train"
    test_dir = root / "test"
    for cls_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() == ".png":
                items.append((f"train/{cls_dir.name}", f))
    for f in sorted(test_dir.iterdir()):
        if f.suffix.lower() == ".png":
            items.append(("test", f))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--max-dist", type=int, default=4, help="dHash Hamming 距离阈值")
    ap.add_argument("--export", default=None, help="把重复对导出为 csv")
    args = ap.parse_args()

    from src.data import discover

    layout = discover(ROOT / args.data)
    root = layout.train_dir.parent
    print(f"数据根目录: {root}")

    items = collect(root)
    n_train = sum(1 for s, _ in items if s.startswith("train"))
    n_test = sum(1 for s, _ in items if s == "test")
    print(f"图片总数 {len(items)}  (训练 {n_train} / 测试 {n_test})\n")

    # ---------------- 1. 字节级精确重复 ----------------
    print("=" * 74)
    print("[1] 字节级精确重复（MD5）")
    by_md5 = defaultdict(list)
    for split, p in items:
        by_md5[md5_of(p)].append((split, p))

    dup_groups = {h: v for h, v in by_md5.items() if len(v) > 1}
    exact_cross = []
    exact_within = []
    if not dup_groups:
        print("    没有任何两个文件内容完全相同")
    for h, group in dup_groups.items():
        splits = {s for s, _ in group}
        if "test" in splits and any(s.startswith("train") for s in splits):
            exact_cross.append((h, group))
        else:
            exact_within.append((h, group))
    print(f"    训练/测试 跨集重复组: {len(exact_cross)}")
    print(f"    集合内部重复组      : {len(exact_within)}")
    for h, group in exact_cross[:5]:
        print(f"      {h[:10]}  " + "  |  ".join(f"{s}:{p.name}" for s, p in group))

    # ---------------- 2. 感知近似重复 ----------------
    print("\n" + "=" * 74)
    print(f"[2] 感知近似重复（dHash，Hamming <= {args.max_dist}）")
    hashes, splits, paths = [], [], []
    for split, p in items:
        h = dhash(p)
        if h is not None:
            hashes.append(h)
            splits.append(split)
            paths.append(p)
    H = np.array(hashes, dtype=np.uint64)
    n = len(H)
    print(f"    成功计算哈希 {n} 张")

    # 两两汉明距离（分批做，避免 878x878 全矩阵的内存压力）
    near_pairs = []
    for i in range(n):
        d = np.bitwise_count(H[i] ^ H[i + 1:])
        js = np.nonzero(d <= args.max_dist)[0] + i + 1
        for j in js:
            near_pairs.append((int(d[j - i - 1]), i, int(j)))

    cross = [(d, i, j) for d, i, j in near_pairs
             if (splits[i] == "test") != (splits[j] == "test")]
    within = [(d, i, j) for d, i, j in near_pairs
              if (splits[i] == "test") == (splits[j] == "test")]

    print(f"    训练集内部近似重复对: {len(within)}")
    print(f"    测试集内部近似重复对: {sum(1 for d,i,j in within if splits[i]=='test')}")
    print(f"    >>> 训练-测试 跨集近似重复对: {len(cross)}  <<<")

    if cross:
        print("\n    跨集重复样例（距离越小越像，0 = 完全一致）:")
        for d, i, j in sorted(cross)[:10]:
            a, b = (i, j) if splits[i].startswith("train") else (j, i)
            print(f"      dist={d:2d}  {paths[a].parent.name}/{paths[a].name}"
                  f"  <->  test/{paths[b].name}")
        n_test_hit = len({j if splits[j] == "test" else i for _, i, j in cross})
        print(f"\n    受影响的测试图片数: {n_test_hit} / {n_test} "
              f"({n_test_hit / max(n_test,1) * 100:.1f}%)")
    else:
        print("\n    训练集与测试集之间没有近似重复图片")

    if args.export and (cross or exact_cross):
        import csv

        out = Path(args.export)
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["type", "dist", "train_path", "test_path"])
            for h, group in exact_cross:
                tr = [p for s, p in group if s.startswith("train")]
                te = [p for s, p in group if s == "test"]
                for a in tr:
                    for b in te:
                        w.writerow(["md5", 0, a, b])
            for d, i, j in sorted(cross):
                a, b = (i, j) if splits[i].startswith("train") else (j, i)
                w.writerow(["dhash", d, paths[a], paths[b]])
        print(f"\n    重复对已导出: {out}")

    # ---------------- 3. 结论 ----------------
    print("\n" + "=" * 74)
    leaky = len(cross) + sum(
        1 for h, g in exact_cross
        if any(s.startswith("train") for s, _ in g) and any(s == "test" for s, _ in g)
    )
    if leaky == 0:
        print("结论：未发现训练集与测试集之间的重复图片，评测是干净的。")
    else:
        print(f"结论：发现 {len(cross)} 对跨集近似重复，模型分数会被这部分样本抬高，")
        print("      报告里应当如实说明，并给出剔除重复后的指标。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
