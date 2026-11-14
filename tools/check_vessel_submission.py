"""实验四：提交文件自检（Id 集合 / RLE 合法性 / 解码后的面积合理性）。

官方 `segmentation_to_csv.py` 只负责「掩膜图 -> RLE」，不做任何检查；
这个脚本按官方的解码规则把提交解回 512x512 掩膜，逐条验证：

1. 列名与行数、Id 集合与 `sample_submission.csv` **完全一致**（不重不漏）；
2. RLE 是「起点 长度」成对、起点严格递增、不重叠、不越界（512*512）；
3. 解码后的血管占比落在训练集的经验区间内（防「全 1」或「全 0」这类坏提交）；
4. `--masks` 给出掩膜目录时，反过来核对「本地掩膜 -> RLE」与提交里的字符串一致。

用法
----
    .\\tools\\py.ps1 tools/check_vessel_submission.py submissions/t4_vessel_cnxt.csv
    .\\tools\\py.ps1 tools/check_vessel_submission.py submissions/t4_vessel_cnxt.csv `
        --masks submissions/t4_masks_cnxt --data 实验四/segmentation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vessel_data import DEFAULT_DATA, discover, imread_unicode, load_set, mask_to_rle_512  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="提交文件")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--masks", default=None, help="预测掩膜目录（可选，做一致性复核）")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    csv_path = ROOT / args.csv if not Path(args.csv).is_absolute() else Path(args.csv)
    sub = pd.read_csv(csv_path, keep_default_na=False)
    print(f"[file] {csv_path}（{len(sub)} 行，列 {list(sub.columns)}）")

    train_items, test_items = discover(args.data)
    train = load_set(train_items)
    sample = pd.read_csv(Path(args.data) / "sample_submission.csv", keep_default_na=False)
    problems: list[str] = []

    if list(sub.columns) != ["Id", "Predicted"]:
        problems.append(f"列名应为 Id,Predicted，实际 {list(sub.columns)}")
    if len(sub) != len(sample):
        problems.append(f"行数 {len(sub)} != 模板 {len(sample)}")
    ids = [str(i) for i in sub["Id"]]
    ref = [str(i) for i in sample["Id"]]
    if sorted(ids) != sorted(ref):
        problems.append(f"Id 集合与模板不一致：缺 {sorted(set(ref) - set(ids))[:5]} "
                        f"多 {sorted(set(ids) - set(ref))[:5]}")
    if len(set(ids)) != len(ids):
        problems.append("Id 有重复")

    # 训练集经验区间：视场内血管占比
    gt_ratio = np.array([s.mask[s.fov].mean() for s in train])
    lo, hi = float(np.percentile(gt_ratio, 1) * 0.35), float(np.percentile(gt_ratio, 99) * 1.6)
    size = args.size
    ratios, bad = [], 0
    for row in sub.itertuples(index=False):
        rle = str(row.Predicted).strip()
        flat = np.zeros(size * size, bool)
        if rle:
            toks = rle.split()
            if len(toks) % 2:
                problems.append(f"Id {row.Id}: RLE 数值个数为奇数")
                bad += 1
                continue
            prev_end = 0
            ok = True
            for i in range(0, len(toks), 2):
                start, length = int(toks[i]), int(toks[i + 1])
                if length <= 0 or start <= prev_end or start < 1 or start - 1 + length > flat.size:
                    problems.append(f"Id {row.Id}: RLE 第 {i // 2 + 1} 段非法 "
                                    f"(start={start}, len={length}, 上一段结束={prev_end})")
                    ok = False
                    break
                flat[start - 1:start - 1 + length] = True
                prev_end = start - 1 + length
            if not ok:
                bad += 1
                continue
        ratios.append(flat.mean())
        if not (lo <= flat.mean() <= hi):
            problems.append(f"Id {row.Id}: 解码面积占比 {flat.mean():.4f} 超出经验区间 "
                            f"[{lo:.4f}, {hi:.4f}]")
    print(f"[rle] 解析 {len(sub)} 行；血管占比 均值 {np.mean(ratios):.4f} "
          f"范围 [{np.min(ratios):.4f}, {np.max(ratios):.4f}]；训练集 GT 均值 {gt_ratio.mean():.4f}")
    print(f"[rle] 经验区间 [{lo:.4f}, {hi:.4f}]")

    if args.masks:
        mdir = ROOT / args.masks if not Path(args.masks).is_absolute() else Path(args.masks)
        files = sorted(p for p in mdir.glob("*") if p.suffix.lower() in {".png", ".jpg"})
        print(f"[masks] {mdir} 下 {len(files)} 张")
        got = {str(int(p.stem)) if p.stem.isdigit() else p.stem: p for p in files}
        missing = [i for i in ref if i not in got]
        if missing:
            problems.append(f"掩膜目录缺少 id {missing[:5]}")
        mismatch = 0
        rle_map = {str(getattr(r, "Id")): str(getattr(r, "Predicted")).strip() for r in sub.itertuples(index=False)}
        for k, p in got.items():
            m = imread_unicode(p, 0)
            if m is None:
                problems.append(f"读不了掩膜 {p}")
                continue
            if mask_to_rle_512(m, size) != rle_map.get(k, ""):
                mismatch += 1
        if mismatch:
            problems.append(f"{mismatch} 张掩膜重新编码后与提交内容不一致")

    if problems:
        print(f"\n[FAIL] {len(problems)} 个问题：")
        for p in problems[:20]:
            print("  -", p)
        return 1
    print("\n[OK] 提交文件通过全部检查")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
