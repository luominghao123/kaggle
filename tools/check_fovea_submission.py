"""校验中心凹（坐标回归）提交文件：列名、行数、ID 是否与 sample_submission 对齐、坐标范围。

与 `check_submission.py` 的区别：那个是任务一（分类，ID + Category）专用，
拿它检查坐标提交会报一堆假错误，所以这里单写一个。

用法:
    .\\tools\\py.ps1 tools\\check_fovea_submission.py submissions/f512.csv
    .\\tools\\py.ps1 tools\\check_fovea_submission.py submissions/f512.csv --data data/task3
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                rows = [r for r in csv.reader(f) if r]
            return (rows[0], rows[1:]) if rows else ([], [])
        except UnicodeDecodeError:
            continue
    return [], []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--data", default=None, help="数据根目录（用来找 sample_submission.csv）")
    args = ap.parse_args()

    sub = Path(args.submission)
    if not sub.exists():
        print(f"[X] 文件不存在: {sub}")
        return 1

    header, rows = read_csv(sub)
    print("=" * 74)
    print(f"检查提交文件: {sub}")
    print(f"  行数: {len(rows)}   列名: {header}")

    problems: list[str] = []
    warns: list[str] = []

    is_long = len(header) == 2 and all("_fovea_" in (r[0] or "").lower() for r in rows[:4])
    if is_long:
        print("  表形态: 长表（ImageID / value，每图两行 _X / _Y）")
        vals = []
        for r in rows:
            try:
                vals.append(float(r[1]))
            except (ValueError, IndexError):
                problems.append(f"存在非数值行: {r[:2]}")
                break
        if vals:
            v = np.asarray(vals)
            print(f"  value 范围: [{v.min():.3f}, {v.max():.3f}]  mean={v.mean():.3f}")
            if (v == 0).all():
                problems.append("全部填 0 —— 像是没预测成功")
            if v.min() >= 0 and v.max() <= 1.001:
                warns.append("所有值都在 [0,1] 内；若标签是像素坐标（如 700~1800），说明单位还原错了")
    else:
        if len(header) != 3:
            problems.append(f"列数应为 3（ID + 两个坐标），实际 {len(header)}")

        # ---- 数值列范围 ----
        vals = []
        for r in rows:
            try:
                vals.append((float(r[1]), float(r[2])))
            except (ValueError, IndexError):
                problems.append(f"存在非数值坐标行: {r[:3]}")
                break
        if vals:
            arr = np.asarray(vals)
            print(f"  x 范围: [{arr[:,0].min():.4f}, {arr[:,0].max():.4f}]  mean={arr[:,0].mean():.4f}")
            print(f"  y 范围: [{arr[:,1].min():.4f}, {arr[:,1].max():.4f}]  mean={arr[:,1].mean():.4f}")
            if arr.min() < -0.02 or arr.max() > 1.02:
                warns.append("坐标超出 [0,1] —— 若竞赛要求归一化坐标，这会是致命的；"
                             "若要求像素坐标则可忽略")
            if np.allclose(arr, 0.5):
                problems.append("所有坐标都是 0.5，像是没预测成功（占位值）")

    # ---- 与 sample_submission 对齐 ----
    data_root = Path(args.data) if args.data else None
    sample = None
    if data_root is not None and data_root.exists():
        cands = [p for p in data_root.rglob("*.csv")
                 if "sample" in p.name.lower() and "submission" in p.name.lower()]
        sample = cands[0] if cands else None
    if sample is None:
        for p in (ROOT / "data").rglob("*sample*submission*.csv"):
            sample = p
            break

    if sample is None:
        warns.append("没找到 sample_submission.csv，无法校验行数/ID 是否对齐")
    else:
        sh, srows = read_csv(sample)
        print(f"  参照模板: {sample.relative_to(ROOT) if sample.is_relative_to(ROOT) else sample}"
              f"  {len(srows)} 行 / 列名 {sh}")
        if len(srows) != len(rows):
            problems.append(f"行数与模板不一致: {len(rows)} vs {len(srows)}")
        if sh != header:
            warns.append(f"列名与模板不一致: {header} vs {sh}（Kaggle 按列名对齐，请确认）")
        a = [r[0].strip() for r in srows]
        b = [r[0].strip() for r in rows]
        missing = sorted(set(a) - set(b))
        extra = sorted(set(b) - set(a))
        if missing:
            problems.append(f"缺少 {len(missing)} 个 ID，例如 {missing[:5]}")
        if extra:
            problems.append(f"多出 {len(extra)} 个 ID，例如 {extra[:5]}")
        if not missing and not extra:
            print(f"  [ok] ID 集合与模板完全一致（{len(a)} 个）")
            if a != b:
                warns.append("行顺序与模板不同 —— Kaggle 按 ID 匹配，通常不影响得分")

    print()
    for w in warns:
        print(f"  [i] {w}")
    for p in problems:
        print(f"  [X] {p}")
    print("=" * 74)
    if problems:
        print("结论: 存在问题，请先修复")
        return 1
    print("结论: 通过" + ("（有提示项，可用但不放心就再核一下）" if warns else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
