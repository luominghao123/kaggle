"""对可疑的图片对做决定性判定：是「同一张图被复制」，还是「两张相似的照片」。

背景
----
dHash 在 64 bit 上做汉明距离，阈值放宽到 6~8 时会捞出一批「看起来像」的图片对，
但这里面绝大多数只是同物种的相似照片，并不是数据泄漏。真正的泄漏必须满足
「同一幅图像内容」——最直接的判据是像素级高度对齐。

判定手段
--------
把两张图各自转灰度、缩放到同一尺寸并做标准化，然后计算归一化互相关 (NCC)：
    NCC = mean((a-mean(a))/std(a) * (b-mean(b))/std(b))   ∈ [-1, 1]
* NCC >= 0.98  -> 几乎可以断定是同一张图（重新编码/缩放/裁剪）
* 0.85 ~ 0.98  -> 高度相似，需要人工看图确认
* < 0.85       -> 只是长得像的同物种照片，不构成泄漏

用法:
    .\\run.ps1 tools\\verify_duplicates.py                 # 扫描全部跨集候选
    .\\run.ps1 tools\\verify_duplicates.py --top 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import discover  # noqa: E402
from tools.check_leakage import collect, dhash  # noqa: E402


def ncc(p1: Path, p2: Path, size: int = 64) -> float:
    """归一化互相关：1 表示完全一致（线性关系），0 表示无关。"""
    def load(p):
        with Image.open(p) as im:
            a = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR),
                           dtype=np.float64)
        a = a - a.mean()
        s = a.std()
        return a / s if s > 1e-6 else a

    a, b = load(p1), load(p2)
    return float((a * b).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--max-dist", type=int, default=12, help="dHash 候选阈值")
    ap.add_argument("--top", type=int, default=25, help="只细查最像的前 N 对")
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    layout = discover(ROOT / args.data)
    root = layout.train_dir.parent
    items = collect(root)

    hashes, splits, paths, wh = [], [], [], []
    for s, p in items:
        h = dhash(p)
        if h is None:
            continue
        with Image.open(p) as im:
            wh.append(im.size)
        hashes.append(h)
        splits.append(s)
        paths.append(p)

    arr = np.array(hashes, dtype=np.uint64)
    n = len(arr)

    cands = []
    for i in range(n):
        d = np.bitwise_count(arr[i] ^ arr[i + 1:])
        for off in np.nonzero(d <= args.max_dist)[0]:
            j = i + 1 + int(off)
            if (splits[i] == "test") != (splits[j] == "test"):
                cands.append((int(d[off]), i, j))
    cands.sort()

    print(f"图片 {n} 张；dHash <= {args.max_dist} 的跨集候选共 {len(cands)} 对")
    print(f"下面按 dHash 距离从小到大细查前 {min(args.top, len(cands))} 对\n")
    print(f"{'dHash':>5s} {'NCC':>7s} {'尺寸1':>12s} {'尺寸2':>12s}  {'判定':<22s} 图片对")
    print("-" * 110)

    for d, i, j in cands[: args.top]:
        ti, tj = (i, j) if splits[i].startswith("train") else (j, i)
        c = ncc(paths[ti], paths[tj], args.size)
        if c >= 0.98:
            verdict = "判定为同一张图(泄漏)"
        elif c >= 0.85:
            verdict = "高度相似，需人工确认"
        else:
            verdict = "仅同物种相似，非泄漏"
        print(f"{d:5d} {c:7.4f} {str(wh[ti]):>12s} {str(wh[tj]):>12s}  {verdict:<22s} "
              f"{splits[ti]}/{paths[ti].name} <-> test/{paths[tj].name}")

    # 统计
    n_dup = 0
    n_sim = 0
    for d, i, j in cands:
        ti, tj = (i, j) if splits[i].startswith("train") else (j, i)
        c = ncc(paths[ti], paths[tj], args.size)
        n_dup += c >= 0.98
        n_sim += 0.85 <= c < 0.98

    print("\n" + "=" * 110)
    print(f"NCC >= 0.98（判定为同一张图，真泄漏）: {n_dup}")
    print(f"0.85 <= NCC < 0.98（高度相似，需人工看）: {n_sim}")
    print(f"其余 {len(cands) - n_dup - n_sim} 对为同物种相似照片，不构成泄漏")
    if n_dup == 0:
        print("\n结论：训练集与测试集之间不存在实质性的图像重复，评测划分是干净的。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
