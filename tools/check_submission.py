"""提交文件自检：在上传前把所有会用到的格式问题都查一遍。

用法:
    .\\run.ps1 tools\\check_submission.py submissions/cnxt320_fold0.csv
    .\\run.ps1 tools\\check_submission.py submissions/*.csv --sample submission-for-task2.csv

检查项:
    1. 列名与顺序是否与 sample_submission 一致
    2. 行数是否一致
    3. ID 集合是否完全一致（这是最容易翻车、且会导致 0 分的地方）
    4. ID 顺序是否与 sample_submission 一致（Kaggle 按 ID 匹配，顺序不强制，但一致最稳）
    5. 预测的类别是否都在训练集出现的类别名之内（拼写错误会导致该行判错）
    6. 是否有空值 / 重复 ID
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def known_classes() -> set[str]:
    """从训练集目录读出合法类别名。"""
    try:
        from src.data import discover

        return set(discover(ROOT / "data").classes)
    except Exception:
        return set()


def check(sub_path: Path, sample_path: Path | None) -> bool:
    print("=" * 74)
    print(f"检查提交文件: {sub_path}")
    ok = True

    if not sub_path.exists():
        print("  [X] 文件不存在")
        return False

    sub = pd.read_csv(sub_path, dtype=str)
    print(f"  形状: {sub.shape}")
    print(f"  列名: {list(sub.columns)}")

    # 空值 / 重复
    if sub.isna().any().any():
        print(f"  [X] 存在空值: {sub.columns[sub.isna().any()].tolist()}")
        ok = False
    dup = sub[sub.columns[0]].duplicated().sum()
    if dup:
        print(f"  [X] ID 有 {dup} 个重复")
        ok = False

    if sample_path and sample_path.exists():
        sample = pd.read_csv(sample_path, dtype=str)
        print(f"\n  参照模板: {sample_path}  形状 {sample.shape}")

        if list(sub.columns) != list(sample.columns):
            print(f"  [X] 列名不一致: {list(sub.columns)} vs {list(sample.columns)}")
            ok = False
        else:
            print(f"  [ok] 列名一致: {list(sub.columns)}")

        if len(sub) != len(sample):
            print(f"  [X] 行数不一致: {len(sub)} vs {len(sample)}")
            ok = False
        else:
            print(f"  [ok] 行数一致: {len(sub)}")

        sid, tid = sub.columns[0], sample.columns[0]
        s_set, t_set = set(sub[sid]), set(sample[tid])
        if s_set == t_set:
            print("  [ok] ID 集合完全一致")
        else:
            print(f"  [X] ID 集合不一致: 缺少 {len(t_set - s_set)} 个, 多出 {len(s_set - t_set)} 个")
            print(f"      缺少示例: {sorted(t_set - s_set)[:5]}")
            print(f"      多出示例: {sorted(s_set - t_set)[:5]}")
            ok = False

        if list(sub[sid]) == list(sample[tid]):
            print("  [ok] ID 顺序也完全一致（最稳妥）")
        else:
            print("  [i] ID 顺序与模板不同 —— Kaggle 按 ID 匹配，不影响得分")
    else:
        print("\n  [!] 未提供 sample_submission，跳过 ID 集合校验")

    # 类别合法性
    classes = known_classes()
    if classes:
        label_col = sub.columns[1]
        used = set(sub[label_col].unique())
        bad = used - classes
        print(f"\n  预测用到的类别 ({len(used)}): {sorted(used)}")
        if bad:
            print(f"  [X] 出现训练集里不存在的类别: {sorted(bad)}")
            print(f"      合法类别应为: {sorted(classes)}")
            ok = False
        else:
            print(f"  [ok] 全部落在合法类别内 {sorted(classes)}")
    else:
        print("\n  [!] 未能读到训练集类别，跳过类别名校验")

    print("\n" + ("  结论: 通过，可以上传" if ok else "  结论: 存在问题，请先修复"))
    print("=" * 74)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", nargs="+", help="待检查的提交文件")
    ap.add_argument("--sample", default=None, help="sample_submission.csv 路径")
    args = ap.parse_args()

    sample = Path(args.sample) if args.sample else None
    if sample is None:
        for cand in (ROOT / "submission-for-task2.csv",
                     ROOT / "data" / "raw"):
            if cand.is_file():
                sample = cand
                break
            if cand.is_dir():
                hits = list(cand.rglob("*sample*submission*.csv")) or \
                       list(cand.rglob("submission-for-task2.csv"))
                if hits:
                    sample = hits[0]
                    break

    results = [check(Path(p), sample) for p in args.submission]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
