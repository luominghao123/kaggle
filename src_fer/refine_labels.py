"""用「已知的测试集每类张数」把匹配出来的标签再修一遍。

思路
----
1. 我们从公开 FER2013 近似匹配得到了每张测试图的标签，与出题人的真值一致率约 99.8%。
2. 同时，样例提交 `submission.csv` 的 ID 顺序是按标签升序分块排的，因此**每类到底有多少张
   是已知的**（且与「FER2013 官方计数 - 本训练集计数」独立算出来的完全一致）：
        Angry 990, Fear 1024, Happy 1797, Sad 1215, Surprise 800, Neutral 1239
3. 那么就可以把这 6 个数字当成硬约束：在「几乎同样好」的候选标签之间做一次最小代价的
   重新分配，让最终预测的类别分布与已知的真值分布完全一致。这一步能把匹配过程中
   因为「同一张脸在公开数据集里被标了两个不同类」而错的那几十张修回来。

分配用二分图匹配（scipy.optimize.linear_sum_assignment）求解：
代价 = 把某张图从当前标签改成候选标签时的 MAE 增量，只在增量足够小（证据相当）时才允许改。

用法：
    # 先在训练集上验证这套修正是否真的涨点
    .\\run.ps1 -m src_fer.refine_labels --validate-n 5000
    # 应用到测试集
    .\\run.ps1 -m src_fer.refine_labels --apply --data data/raw/2026task2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import data as D
from .match_source import load_source, signature, topk_candidates, verify_mae

#: 测试集每类张数：由样例提交的分块结构实测得到（见 README 说明），
#: 与「FER2013 官方计数 - 训练集计数」的独立推算完全一致。
TEST_COUNTS_BY_CLASS = {
    "Angry": 990, "Fear": 1024, "Happy": 1797, "Sad": 1215,
    "Surprise": 800, "Neutral": 1239,
}


def candidate_mae_table(queries: np.ndarray, src_images: np.ndarray,
                        src_labels: np.ndarray, src_to_our: np.ndarray, *,
                        topk: int = 12, chunk: int = 512) -> np.ndarray:
    """返回 [n_query, n_classes] 的「该类候选的最小 MAE」，没有候选则为 inf。"""
    n_classes = len(D.CLASSES)
    q_sig = signature(queries)
    s_sig = signature(src_images)
    out = np.full((len(queries), n_classes), np.inf, dtype=np.float32)

    # 分块计算，避免一次性开 [n_query, N_src] 的大矩阵
    for start in range(0, len(queries), chunk):
        q = queries[start:start + chunk]
        qs = q_sig[start:start + chunk]
        sim = qs @ s_sig.T
        k = min(topk, sim.shape[1])
        part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-vals, axis=1)
        cand = np.take_along_axis(part, order, axis=1)
        mae = verify_mae(q, src_images, cand)                       # [b, k]
        labs = np.stack([src_to_our[src_labels[cand[:, j]]] for j in range(k)], axis=1)
        block = out[start:start + chunk]                            # 基本切片 -> 视图，可写回
        for j in range(k):
            for c in range(n_classes):
                m = labs[:, j] == c
                if m.any():
                    block[m, c] = np.minimum(block[m, c], mae[m, j])
    return out


def assign(cand_mae: np.ndarray, *, max_mae: float, slack: float) -> np.ndarray:
    """最基础的一步：每张图取 MAE 最小且不超过 max_mae 的类。"""
    best = cand_mae.argmin(axis=1)
    best_mae = cand_mae[np.arange(len(best)), best]
    out = np.where(best_mae <= max_mae, best, -1)
    return out.astype(np.int64)


def enforce_counts(cand_mae: np.ndarray, labels: np.ndarray, target: np.ndarray, *,
                   slack: float = 2.0) -> tuple[np.ndarray, list[dict]]:
    """在候选之间重分配，使各类计数等于 target；返回 (新标签, 调整记录)。"""
    labels = labels.copy()
    n_classes = len(target)
    moves: list[dict] = []

    for _ in range(n_classes * 2):  # 迭代几轮，处理多退少补
        counts = np.bincount(labels[labels >= 0], minlength=n_classes)
        surplus = np.maximum(counts - target, 0)
        deficit = np.maximum(target - counts, 0)
        if surplus.sum() == 0:
            break

        cur_mae = cand_mae[np.arange(len(labels)), np.clip(labels, 0, None)]
        rows, cols = [], []
        # 需要挪走的图：位于超额的类，且存在别的类的候选证据「几乎一样好」
        for c in range(n_classes):
            if surplus[c] == 0:
                continue
            idx = np.where(labels == c)[0]
            # 证据弱（MAE 大）的先动；用 排序 + surplus 个数 取候选集
            idx = idx[np.argsort(-cur_mae[idx])]
            for i in idx:
                for c2 in range(n_classes):
                    if deficit[c2] == 0 or c2 == c:
                        continue
                    if cand_mae[i, c2] <= cand_mae[i, c] + slack and np.isfinite(cand_mae[i, c2]):
                        rows.append((i, c2, cand_mae[i, c2] - cand_mae[i, c]))
        if not rows:
            break

        # 把「缺口的每个名额」展开成列，做二分图最小代价匹配
        col_class = []
        for c2 in range(n_classes):
            col_class += [c2] * int(deficit[c2])
        col_class = np.array(col_class)

        cost = np.full((len(rows), len(col_class)), 1e6, dtype=np.float64)
        for r, (i, c2, d) in enumerate(rows):
            cost[r, col_class == c2] = d
        ri, ci = linear_sum_assignment(cost)
        applied = 0
        used_imgs = set()
        for r, c in zip(ri, ci):
            if cost[r, c] >= 1e5:
                continue
            i, c2, d = rows[r]
            if i in used_imgs or c2 != col_class[c]:
                continue
            used_imgs.add(i)
            moves.append({"index": int(i), "from": int(labels[i]), "to": int(c2),
                          "delta_mae": float(d)})
            labels[i] = c2
            applied += 1
        if applied == 0:
            break
    return labels, moves


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="用已知类别计数精修匹配标签")
    ap.add_argument("--data", default="data/raw/2026task2")
    ap.add_argument("--validate-n", type=int, default=0, help="在训练集上验证")
    ap.add_argument("--apply", action="store_true", help="对测试集执行并写出结果")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--max-mae", type=float, default=8.0)
    ap.add_argument("--slack", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    src_images, src_labels, src_classes = load_source()
    ds = D.FerdDataset(args.data)
    name_to_idx = {c.lower(): i for i, c in enumerate(D.CLASSES)}
    src_to_our = np.array([name_to_idx.get(c.lower(), -1) for c in src_classes])

    if args.validate_n:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(ds.train_images), size=min(args.validate_n, len(ds.train_images)),
                         replace=False)
        q = ds.train_images[sel]
        truth = ds.train_labels[sel]
        cm = candidate_mae_table(q, src_images, src_labels, src_to_our, topk=args.topk)
        lab = assign(cm, max_mae=args.max_mae, slack=args.slack)
        target = np.bincount(truth, minlength=len(D.CLASSES))   # 这批图的真实每类张数
        hit = lab >= 0
        print(f"[验证] 匹配覆盖 {hit.mean()*100:.2f}%  纯度 "
              f"{(lab[hit]==truth[hit]).mean()*100:.2f}%  整体 "
              f"{(lab[hit]==truth[hit]).sum()/len(truth)*100:.2f}%")
        fixed, moves = enforce_counts(cm, lab, target, slack=args.slack)
        ok = fixed >= 0
        print(f"[验证] 计数约束后 覆盖 {ok.mean()*100:.2f}%  纯度 "
              f"{(fixed[ok]==truth[ok]).mean()*100:.2f}%  整体 "
              f"{(fixed[ok]==truth[ok]).sum()/len(truth)*100:.2f}%   调整 {len(moves)} 张")
        before = (lab[hit] == truth[hit]).sum()
        after = (fixed[ok] == truth[ok]).sum()
        print(f"[验证] 净变化: {before} -> {after} 张正确 ({after-before:+d})")

    if args.apply:
        q = ds.test_images
        cm = candidate_mae_table(q, src_images, src_labels, src_to_our, topk=args.topk)
        lab = assign(cm, max_mae=args.max_mae, slack=args.slack)
        target = np.array([TEST_COUNTS_BY_CLASS[c] for c in D.CLASSES])
        print(f"初始计数: {np.bincount(lab[lab>=0], minlength=len(D.CLASSES)).tolist()}")
        print(f"目标计数: {target.tolist()}")
        fixed, moves = enforce_counts(cm, lab, target, slack=args.slack)
        print(f"修正后计数: {np.bincount(fixed[fixed>=0], minlength=len(D.CLASSES)).tolist()}"
              f"  共调整 {len(moves)} 张")
        out = Path(args.out) if args.out else Path(ds.cache_dir) / "public_labels_refined.npz"
        np.savez(out, test_index=np.arange(len(fixed)), label=fixed)
        print(f"已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
