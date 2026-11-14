"""用 out-of-fold 概率来验证两个后处理到底有没有用（而不是凭感觉加）。

1. **重复图定标**：在每一折里，只用该折的「训练部分」建哈希表，去覆盖该折验证图的
   预测，比较覆盖前后的准确率——这才是测试时做法的无偏模拟。
2. **类别先验对齐（Sinkhorn）**：分层 K 折的验证集类别分布是已知的（等比例分层），
   所以可以像测试时那样把预测分布对齐到目标分布，再比较准确率。

用法：
    .\\run.ps1 -m src_fer.validate_postproc --data data/raw/2026task2 --runs runs_fer/cxt128
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from . import data as D
from .predict import sinkhorn_prior


def exact_hashes(images: np.ndarray) -> np.ndarray:
    flat = images.reshape(len(images), -1)
    out = np.empty(len(flat), dtype=np.uint64)
    for i, row in enumerate(flat):
        out[i] = int.from_bytes(hashlib.blake2b(row.tobytes(), digest_size=8).digest(), "little")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="后处理效果验证（基于 OOF）")
    ap.add_argument("--data", default="data/raw/2026task2")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--prior-strength", type=float, default=0.5)
    args = ap.parse_args(argv)

    run = Path(args.runs)
    ds = D.FerdDataset(args.data)
    y = ds.train_labels

    # summary.json 只有在全部折训练完之后才会写；只跑了部分折时退化为读 fold*_result.json
    if (run / "summary.json").is_file():
        meta = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    else:
        cand = sorted(run.glob("fold*_result.json"))
        if not cand:
            raise FileNotFoundError(f"{run} 下既没有 summary.json 也没有 fold*_result.json")
        meta = json.loads(cand[0].read_text(encoding="utf-8"))
    seed = meta["args"]["seed"]
    folds = meta["args"]["folds"]
    splits = list(StratifiedKFold(n_splits=folds, shuffle=True,
                                  random_state=seed).split(np.zeros(len(y)), y))

    all_probs = np.zeros((len(y), len(D.CLASSES)), dtype=np.float64)
    covered = np.zeros(len(y), dtype=bool)

    dup_before = dup_after = dup_n = 0
    prior_before = prior_after = 0
    prior_n = 0
    rows = []
    rows_diag = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        vp, vl, vi = (run / f"val_probs_fold{fold}.npy", run / f"val_labels_fold{fold}.npy",
                      run / f"val_idx_fold{fold}.npy")
        if not vp.is_file():
            continue
        probs = np.load(vp).astype(np.float64)
        labels = np.load(vl)
        idx = np.load(vi)
        assert np.array_equal(labels, y[idx]), "验证标签与全局标签不一致"

        all_probs[idx] = probs
        covered[idx] = True

        # ---- 1) 重复图定标（只用该折训练部分） ----
        table: dict[int, list[int]] = defaultdict(list)
        for h, l in zip(exact_hashes(ds.train_images[tr_idx]).tolist(),
                        y[tr_idx].tolist()):
            table[h].append(l)
        vhash = exact_hashes(ds.train_images[idx])
        override = {}
        for k, h in enumerate(vhash.tolist()):
            labs = table.get(h)
            if labs:
                vals, cnt = np.unique(labs, return_counts=True)
                override[k] = int(vals[cnt.argmax()])

        pred_before = probs.argmax(1)
        pred_after = pred_before.copy()
        for k, lab in override.items():
            pred_after[k] = lab
        n = len(override)
        dup_n += n
        dup_before += int((pred_before == labels).sum())
        dup_after += int((pred_after == labels).sum())

        # 折间泄漏诊断：验证集里「在训练部分存在逐像素完全相同图」的比例与准确率，
        # 用来说明分层 K 折给出的验证准确率为什么偏乐观
        covered = np.zeros(len(labels), dtype=bool)
        covered[list(override)] = True
        acc_cov = float((pred_before[covered] == labels[covered]).mean()) if covered.any() else float("nan")
        acc_unc = float((pred_before[~covered] == labels[~covered]).mean())
        rows_diag.append({"fold": fold, "n_val": int(len(labels)), "n_dup": int(covered.sum()),
                          "acc_all": float((pred_before == labels).mean()),
                          "acc_on_dup": acc_cov, "acc_on_unique": acc_unc})

        # ---- 2) 类别先验对齐 ----
        target = np.bincount(labels, minlength=len(D.CLASSES))
        adj = sinkhorn_prior(probs, target, strength=args.prior_strength)
        prior_before += int((probs.argmax(1) == labels).sum())
        prior_after += int((adj.argmax(1) == labels).sum())
        prior_n += len(labels)

        rows.append({"fold": fold, "n_val": int(len(labels)),
                     "acc": float((pred_before == labels).mean()),
                     "dup_covered": n,
                     "acc_dup": float((pred_after == labels).mean()),
                     "acc_prior": float((adj.argmax(1) == labels).mean()),
                     "dup_acc_on_covered": (float((pred_after[list(override)] ==
                                                   labels[list(override)]).mean())
                                            if override else None),
                     "model_acc_on_covered": (float((pred_before[list(override)] ==
                                                     labels[list(override)]).mean())
                                              if override else None)})
        print(f"fold {fold}: acc={rows[-1]['acc']:.4f}  覆盖 {n} 张 "
              f"(模型在这些图上 {rows[-1]['model_acc_on_covered']} -> "
              f"定标后 {rows[-1]['dup_acc_on_covered']})  "
              f"先验对齐后 acc={rows[-1]['acc_prior']:.4f}")

    n_total = int(covered.sum())
    print()
    print(f"OOF 覆盖样本数: {n_total}")
    print(f"基线 OOF 准确率                : {dup_before/n_total:.4f}")
    print(f"+ 重复图定标（覆盖 {dup_n} 张） : {dup_after/n_total:.4f} "
          f"({(dup_after-dup_before)/n_total:+.4f})")
    print(f"基线（先验对齐口径）           : {prior_before/prior_n:.4f}")
    print(f"+ 类别先验对齐（Sinkhorn）     : {prior_after/prior_n:.4f} "
          f"({(prior_after-prior_before)/prior_n:+.4f})")

    print("\n折间泄漏诊断（验证集里「在训练部分存在同像素图」的那部分，模型准确率明显更高）:")
    for d in rows_diag:
        print(f"  fold {d['fold']}: 验证 {d['n_val']} 张，其中 {d['n_dup']} 张"
              f"({d['n_dup']/max(d['n_val'],1)*100:.1f}%) 有同像素训练图 —— "
              f"这部分 acc={d['acc_on_dup']:.4f}，其余 acc={d['acc_on_unique']:.4f}，"
              f"整体 acc={d['acc_all']:.4f}")

    out = {"n_oof": n_total, "base_acc": dup_before / n_total,
           "dup_acc": dup_after / n_total, "dup_covered": dup_n,
           "base_acc_prior": prior_before / prior_n, "prior_acc": prior_after / prior_n,
           "folds": rows}
    (run / "postproc_validation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {run / 'postproc_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
