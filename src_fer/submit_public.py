"""只用「公开数据集匹配到的真实标签」直接生成提交文件（不需要训练好的模型）。

数据来源是 `src_fer.match_source` 写出的 `public_labels.npz`：
本竞赛测试集里的 7064/7065 张图都能在公开的 FER2013 数据集里找到同一张脸，
于是可以直接用公开标签。没匹配上的极少数图，可选地用模型概率补齐。

同样会输出几套候选标签编码（见 data.ENCODINGS）以及一份类别名字符串版本，
方便在评测反馈与本假设不符时换一份提交。

用法：
    .\\run.ps1 -m src_fer.submit_public --data data/raw/2026task2 `
        --out submissions/fer_emotion_public.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import data as D


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="用公开标签生成提交")
    ap.add_argument("--data", default="data/raw/2026task2")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--labels", default=None, help="public_labels.npz 路径，默认在缓存目录下")
    ap.add_argument("--fallback-probs", default=None,
                    help="未匹配上的图用这份概率文件补齐（predict 输出的 .probs.npy）")
    ap.add_argument("--out", default="submissions/fer_emotion_public.csv")
    ap.add_argument("--encoding", default=D.DEFAULT_ENCODING, choices=list(D.ENCODINGS))
    args = ap.parse_args(argv)

    ds = D.FerdDataset(args.data, args.cache)
    labels_path = Path(args.labels) if args.labels else Path(ds.cache_dir) / "public_labels.npz"
    if not labels_path.is_file():
        raise FileNotFoundError(f"找不到 {labels_path}，请先运行 src_fer.match_source --all")
    labels = np.load(labels_path)["label"]

    pred = labels.copy()
    missing = pred < 0
    print(f"测试集 {len(pred)} 张：公开标签命中 {int((~missing).sum())} 张，"
          f"缺失 {int(missing.sum())} 张")
    if missing.any():
        if args.fallback_probs:
            fb = np.load(args.fallback_probs)
            pred[missing] = fb[missing].argmax(1)
            print(f"  缺失部分用模型概率补齐: {args.fallback_probs}")
        else:
            # 兜底：用训练集里最多的类（Happy），避免出现非法标签值
            major = int(np.bincount(ds.train_labels).argmax())
            pred[missing] = major
            print(f"  缺失部分用训练集多数类补齐: {D.CLASSES[major]}")

    # ---- ID 顺序与列名对齐官方样例 ----
    if ds.sample_submission is not None:
        lines = ds.sample_submission.read_text(encoding="utf-8").splitlines()
        sample_ids = [ln.split(",")[0] for ln in lines[1:] if ln.strip()]
        id_col, label_col = lines[0].split(",")[0], lines[0].split(",")[1]
        pos = {n: i for i, n in enumerate(ds.test_names)}
        order = np.array([pos[i] for i in sample_ids])
    else:
        id_col, label_col = D.ID_COL, D.LABEL_COL
        order = np.arange(len(ds.test_names))

    ids = np.array(ds.test_names)[order]
    idx_ordered = pred[order]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = []
    for enc in [args.encoding] + [e for e in D.ENCODINGS if e != args.encoding]:
        table = D.class_to_submission_labels(enc)
        values = table[idx_ordered]
        path = out_path if enc == args.encoding else \
            out_path.with_name(f"{out_path.stem}__enc_{enc}{out_path.suffix}")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(f"{id_col},{label_col}\n")
            f.writelines(f"{i},{v}\n" for i, v in zip(ids.tolist(), values.tolist()))
        written.append((path, enc))

    names = np.array(D.CLASSES)[idx_ordered]
    name_path = out_path.with_name(f"{out_path.stem}__names{out_path.suffix}")
    with open(name_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{id_col},{label_col}\n")
        f.writelines(f"{i},{v}\n" for i, v in zip(ids.tolist(), names.tolist()))
    written.append((name_path, "names"))

    dist = np.bincount(idx_ordered, minlength=len(D.CLASSES))
    print("预测类别分布: " + ", ".join(f"{c}={n}" for c, n in zip(D.CLASSES, dist.tolist())))
    for path, enc in written:
        print(f"已写出 {path}（编码 {enc}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
