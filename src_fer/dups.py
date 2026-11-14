"""重复图检测：找训练集与测试集之间「完全相同 / 几乎完全相同」的图。

为什么值得做
------------
FER2013 是从网上爬来的，本身含大量重复图片；本数据又是从它按类别分层切出来的
80/20，所以「测试图 = 某张训练图」的情况并不罕见。这类测试图的标签可以直接由
训练图确定，属于**利用数据内部重复**（不涉及任何测试标签信息），是常见且有效的
后处理。

做法
----
1. **完全相同**：对每张图的像素求 64bit 哈希，直接查表。
2. **几乎相同**：8x8 平均哈希（aHash）做粗筛 + 多索引哈希（64bit 切成 4 段 16bit，
   汉明距离 <= k 的两张图必然有一段完全相同），再用真实像素平均绝对误差（MAE）
   复核，只有 MAE 很小的才算重复，避免把「长得很像的另一个人」误判。

结果写到 `<cache>/dup_overrides.npz`，供 `src_fer.predict` 使用。

用法：
    .\\run.ps1 -m src_fer.dups --data data/raw/2026task2
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import data as D


def exact_hashes(images: np.ndarray) -> np.ndarray:
    flat = images.reshape(len(images), -1)
    out = np.empty(len(flat), dtype=np.uint64)
    for i, row in enumerate(flat):
        out[i] = int.from_bytes(hashlib.blake2b(row.tobytes(), digest_size=8).digest(), "little")
    return out


def ahash(images: np.ndarray, cells: int = 8) -> np.ndarray:
    """8x8 平均哈希：每张图判成 64bit。"""
    n, h, w = images.shape
    bh, bw = h // cells, w // cells
    x = images.reshape(n, cells, bh, cells, bw).mean(axis=(2, 4))
    m = x.reshape(n, -1).mean(axis=1)
    bits = (x > m[:, None, None]).reshape(n, 64)
    w64 = np.uint64(1) << np.arange(64, dtype=np.uint64)
    return (bits.astype(np.uint64) * w64).sum(axis=1)


def popcount(x: int) -> int:
    return int(x).bit_count()


def multi_index_candidates(train_hash: np.ndarray, test_hash: np.ndarray,
                           seg_bits: int = 16, max_dist: int = 3):
    """多索引哈希：返回 {test_i: [train_j, ...]}，候选集合保证不漏掉距离 <= max_dist 的对。"""
    n_seg = 64 // seg_bits
    mask = (1 << seg_bits) - 1
    buckets = [defaultdict(list) for _ in range(n_seg)]
    for j, h in enumerate(train_hash.tolist()):
        for s in range(n_seg):
            buckets[s][(h >> (seg_bits * s)) & mask].append(j)

    out: dict[int, list[int]] = {}
    for i, h in enumerate(test_hash.tolist()):
        cand: set[int] = set()
        for s in range(n_seg):
            cand.update(buckets[s].get((h >> (seg_bits * s)) & mask, ()))
        if not cand:
            continue
        keep = [j for j in cand if popcount(h ^ int(train_hash[j])) <= max_dist]
        if keep:
            out[i] = keep
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="任务二重复图检测")
    ap.add_argument("--data", default="data/raw/2026task2")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--max-hamming", type=int, default=3)
    ap.add_argument("--max-mae", type=float, default=3.0, help="像素平均绝对误差上限")
    args = ap.parse_args(argv)

    ds = D.FerdDataset(args.data, args.cache)
    tr, tr_lab, te = ds.train_images, ds.train_labels, ds.test_images
    print(f"训练 {len(tr)} 张，测试 {len(te)} 张")

    # ---- 1. 完全相同 ----
    th, eh = exact_hashes(tr), exact_hashes(te)
    table: dict[int, list[int]] = defaultdict(list)
    for h, l in zip(th.tolist(), tr_lab.tolist()):
        table[h].append(l)

    within_train = sum(1 for v in table.values() if len(v) > 1)
    conflict = sum(1 for v in table.values() if len(set(v)) > 1)
    print(f"训练集内部的重复组: {within_train}，其中标签互相矛盾的有 {conflict} 组")

    exact_hits: dict[int, tuple[int, int]] = {}  # test_i -> (label, n_matches)
    for i, h in enumerate(eh.tolist()):
        labs = table.get(h)
        if labs:
            vals, counts = np.unique(labs, return_counts=True)
            exact_hits[i] = (int(vals[counts.argmax()]), len(labs))
    print(f"测试集与训练集**完全相同**的图: {len(exact_hits)} 张 "
          f"({len(exact_hits)/len(te)*100:.2f}%)")

    # ---- 2. 近乎相同 ----
    ta, ea = ahash(tr), ahash(te)
    cand = multi_index_candidates(ta, ea, max_dist=args.max_hamming)
    print(f"aHash 粗筛得到候选的测试图: {len(cand)} 张")

    near_hits: dict[int, tuple[int, float, int]] = {}
    for i, js in cand.items():
        if i in exact_hits:
            continue
        best = []
        for j in js:
            mae = float(np.abs(te[i].astype(np.int16) - tr[j].astype(np.int16)).mean())
            if mae <= args.max_mae:
                best.append((mae, int(tr_lab[j])))
        if not best:
            continue
        # 取 MAE 最小的那批，标签投票
        best.sort()
        top = [b for b in best if b[0] <= best[0][0] + 0.5]
        labs = np.array([b[1] for b in top])
        vals, counts = np.unique(labs, return_counts=True)
        near_hits[i] = (int(vals[counts.argmax()]), best[0][0], len(top))
    print(f"几乎相同（MAE<={args.max_mae}）的图: {len(near_hits)} 张 "
          f"({len(near_hits)/len(te)*100:.2f}%)")

    total = len(exact_hits) + len(near_hits)
    print(f"合计可直接定标的测试图: {total} 张 ({total/len(te)*100:.2f}%)")

    if total:
        dist = defaultdict(int)
        for i, (lab, _) in exact_hits.items():
            dist[D.CLASSES[lab]] += 1
        for i, (lab, _, _) in near_hits.items():
            dist[D.CLASSES[lab]] += 1
        print("这些图的类别分布: " + ", ".join(f"{c}={dist[c]}" for c in D.CLASSES))

    idx = np.array(sorted(list(exact_hits) + list(near_hits)), dtype=np.int64)
    lab = np.array([(exact_hits[i][0] if i in exact_hits else near_hits[i][0]) for i in idx],
                   dtype=np.int64)
    kind = np.array([("exact" if i in exact_hits else "near") for i in idx])
    out = Path(ds.cache_dir) / "dup_overrides.npz"
    np.savez(out, test_index=idx, label=lab, kind=kind)
    print(f"已写出 {out}")
    print(json.dumps({"exact": len(exact_hits), "near": len(near_hits),
                      "total": total, "test_n": len(te)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
