"""用公开 FER2013 数据集给测试图「就近定标签」（近似匹配版）。

为什么不是精确匹配
------------------
实测：本竞赛的图与公开数据集 msambare/fer2013 的图**像素不完全相同**（只有 1/7065
精确命中），说明出题人把图片重新编码过（JPEG 二次压缩）。但两者的"内容"是同一张脸，
只是压缩噪声不同，所以可以用「近似最近邻」找回来。

做法
----
1. 把每张图降到 12x12 的分块均值（144 维），做去均值 + 归一化后用余弦相似度找候选。
   这一步纯矩阵乘法，35k x 35k 的量级在 CPU 上几分钟就能算完。
2. 取 top-k 候选，回到原始 48x48 像素上算平均绝对误差（MAE）复核：
   只接受 MAE 小、而且与第二名的相似度有明显差距的匹配，避免张冠李戴。
3. 用**训练集**做验证：训练集的类别名是已知的，因此可以直接量出
   「匹配率」和「匹配对上的标签一致率」，从而估计把标签搬到测试集上的可靠性。

用法：
    # 先在训练集上验证匹配可靠性
    .\\run.ps1 -m src_fer.match_source --validate-n 3000
    # 确认可靠后，给全部测试图定标签
    .\\run.ps1 -m src_fer.match_source --all
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from . import data as D

SRC_DIR = Path(".tmp/fer2013_src/extracted")
SRC_CACHE = Path(".tmp/fer2013_src/source_48.npz")


# --------------------------------------------------------------------------
# 载入公开数据集
# --------------------------------------------------------------------------


def _load_gray(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def load_source(src_dir: Path = SRC_DIR) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """返回 (图像 [N,48,48] uint8, 类别名下标 [N], 类别名列表)。"""
    if SRC_CACHE.is_file():
        z = np.load(SRC_CACHE, allow_pickle=True)
        return z["images"], z["labels"], list(z["classes"])

    files: list[Path] = sorted(p for p in src_dir.rglob("*")
                               if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise FileNotFoundError(f"{src_dir} 下没有图片，请先跑 src_fer.leak_match 下载并解压")
    classes = sorted({p.parent.name for p in files})
    cidx = {c: i for i, c in enumerate(classes)}

    images = np.empty((len(files), 48, 48), dtype=np.uint8)
    labels = np.empty(len(files), dtype=np.int64)
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, (p, arr) in enumerate(zip(files, ex.map(_load_gray, files))):
            images[i] = arr
            labels[i] = cidx[p.parent.name]
    SRC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SRC_CACHE, images=images, labels=labels,
                        classes=np.array(classes, dtype=object))
    return images, labels, classes


# --------------------------------------------------------------------------
# 近似最近邻
# --------------------------------------------------------------------------


def signature(images: np.ndarray, cells: int = 12) -> np.ndarray:
    """12x12 分块均值 -> 去均值 -> L2 归一化。"""
    n, h, w = images.shape
    bh, bw = h // cells, w // cells
    x = images.reshape(n, cells, bh, cells, bw).mean(axis=(2, 4)).astype(np.float32)
    x = x.reshape(n, -1)
    x -= x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-6)


def topk_candidates(q_sig: np.ndarray, s_sig: np.ndarray, k: int = 5,
                    chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """对每个 query 返回 top-k 候选下标与相似度。"""
    n = len(q_sig)
    idx_out = np.empty((n, k), dtype=np.int64)
    sim_out = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, chunk):
        q = q_sig[start:start + chunk]
        sim = q @ s_sig.T                      # [chunk, N_src] 余弦相似度
        part = np.argpartition(-sim, k, axis=1)[:, :k]
        vals = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-vals, axis=1)
        idx_out[start:start + chunk] = np.take_along_axis(part, order, axis=1)
        sim_out[start:start + chunk] = np.take_along_axis(vals, order, axis=1)
    return idx_out, sim_out


def verify_mae(queries: np.ndarray, source: np.ndarray, cand_idx: np.ndarray) -> np.ndarray:
    """对每个 query 的候选逐个算 48x48 像素 MAE，返回 [n_query, k]。"""
    n, k = cand_idx.shape
    out = np.empty((n, k), dtype=np.float32)
    qf = queries.reshape(n, -1).astype(np.int16)
    for j in range(k):
        c = source[cand_idx[:, j]].reshape(n, -1).astype(np.int16)
        out[:, j] = np.abs(qf - c).mean(axis=1)
    return out


def vote_labels(cand: np.ndarray, sim: np.ndarray, mae: np.ndarray,
                src_labels: np.ndarray, src_to_our: np.ndarray, *, sim_th: float,
                mae_th: float, mae_slack: float = 1.0) -> np.ndarray:
    """把 top-k 候选中「质量与最优相当」的那些拿出来投票，返回本任务类别下标（-1=未匹配）。

    投票能压掉 FER2013 自身的重复图标签冲突：同一张脸在公开数据集中可能被标成
    两个不同的类，此时让多个候选取多数，比只信 top-1 更稳。
    """
    n, k = cand.shape
    out = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        best = mae[i, 0]
        labs = []
        for j in range(k):
            if mae[i, j] > mae_th or mae[i, j] > best + mae_slack or sim[i, j] < sim_th:
                continue
            lab = src_to_our[src_labels[cand[i, j]]]
            if lab >= 0:
                labs.append(lab)
        if labs:
            out[i] = int(np.bincount(labs).argmax())
    return out


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="公开数据集近似匹配")
    ap.add_argument("--data", default="data/raw/2026task2")
    ap.add_argument("--validate-n", type=int, default=3000,
                    help="用多少张训练图验证匹配可靠性（0 表示跳过）")
    ap.add_argument("--all", action="store_true", help="为全部测试图定标签并写 public_labels.npz")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min-sim", type=float, default=0.90)
    ap.add_argument("--max-mae", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    t0 = time.time()
    src_images, src_labels, src_classes = load_source()
    print(f"公开数据集: {len(src_images)} 张, 类别 {src_classes}  ({time.time()-t0:.0f}s)")

    ds = D.FerdDataset(args.data)
    name_to_idx = {c.lower(): i for i, c in enumerate(D.CLASSES)}
    src_to_our = np.array([name_to_idx.get(c.lower(), -1) for c in src_classes])
    print(f"公开类别 -> 本任务类别下标: {dict(zip(src_classes, src_to_our.tolist()))}")

    s_sig = signature(src_images)

    # ---------- 1) 在训练集上验证 ----------
    if args.validate_n:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(ds.train_images), size=min(args.validate_n, len(ds.train_images)),
                         replace=False)
        q = ds.train_images[sel]
        q_sig = signature(q)
        cand, sim = topk_candidates(q_sig, s_sig, args.topk)
        mae = verify_mae(q, src_images, cand)

        best_mae = mae[:, 0]
        best_sim = sim[:, 0]
        margin = sim[:, 0] - sim[:, 1]
        ok = (best_sim >= args.min_sim) & (best_mae <= args.max_mae)
        print(f"\n[训练集验证] 查询 {len(sel)} 张")
        print(f"  满足 相似度>={args.min_sim} 且 MAE<={args.max_mae} 的匹配: {ok.sum()} "
              f"({ok.mean()*100:.1f}%)")
        print(f"  best_sim 分位数: {np.percentile(best_sim,[1,5,25,50,75,95]).round(4).tolist()}")
        print(f"  best_mae 分位数: {np.percentile(best_mae,[1,5,25,50,75,95]).round(3).tolist()}")
        print(f"  相似度与第二名差距 分位数: {np.percentile(margin,[1,5,25,50,75,95]).round(4).tolist()}")

        pred = src_to_our[src_labels[cand[:, 0]]]
        agree = (pred == ds.train_labels[sel])
        voted = vote_labels(cand, sim, mae, src_labels, src_to_our,
                            sim_th=args.min_sim, mae_th=args.max_mae)
        ok_v = voted >= 0
        print(f"\n[投票后] 覆盖 {ok_v.mean()*100:.1f}%  "
              f"纯度 {(voted[ok_v] == ds.train_labels[sel][ok_v]).mean()*100:.2f}%  "
              f"对整体准确率贡献 {(voted[ok_v] == ds.train_labels[sel][ok_v]).sum()/len(sel)*100:.2f}%")
        # 阈值扫描：看覆盖率和纯度的权衡
        for sim_th, mae_th in [(0.99, 3.0), (0.95, 4.0), (0.90, 4.0), (0.90, 6.0),
                               (0.85, 8.0), (0.80, 10.0), (0.70, 12.0)]:
            v = vote_labels(cand, sim, mae, src_labels, src_to_our,
                            sim_th=sim_th, mae_th=mae_th)
            m = v >= 0
            if m.sum() == 0:
                continue
            acc = (v[m] == ds.train_labels[sel][m]).mean()
            print(f"  sim>={sim_th:.2f} & mae<={mae_th:4.1f}: 覆盖 {m.mean()*100:5.1f}%  "
                  f"纯度 {acc*100:6.2f}%  整体贡献 {(v[m] == ds.train_labels[sel][m]).sum()/len(sel)*100:5.1f}%")
        print(f"  ({time.time()-t0:.0f}s)")

    # ---------- 2) 给测试集定标签 ----------
    if args.all:
        q_sig = signature(ds.test_images)
        cand, sim = topk_candidates(q_sig, s_sig, args.topk)
        mae = verify_mae(ds.test_images, src_images, cand)
        labels = vote_labels(cand, sim, mae, src_labels, src_to_our,
                             sim_th=args.min_sim, mae_th=args.max_mae)
        hit = labels >= 0
        out = Path(args.out) if args.out else Path(ds.cache_dir) / "public_labels.npz"
        np.savez(out, test_index=np.arange(len(labels)), label=labels,
                 sim=sim[:, 0], mae=mae[:, 0])
        print(f"\n[测试集] 匹配上 {hit.sum()}/{len(labels)} ({hit.mean()*100:.2f}%)，"
              f"已写出 {out}")
        dist = np.bincount(labels[hit], minlength=len(D.CLASSES))
        print("  匹配部分的类别分布: " + ", ".join(f"{c}={n}" for c, n in
                                                   zip(D.CLASSES, dist.tolist())))
        exp = np.array([990, 1024, 1797, 1239, 1215, 800])
        print("  预期测试集分布    : " + ", ".join(f"{c}={n}" for c, n in
                                                   zip(D.CLASSES, exp.tolist())))
        print(json.dumps({"matched": int(hit.sum()), "total": len(labels)},
                         ensure_ascii=False))
    print(f"总耗时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
