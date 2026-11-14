"""把下载好的竞赛压缩包自动解压到 data/raw/，并做完整自检。

设计目标：让「拿到数据」这一步只需要把 zip 丢进工作区，剩下的自动完成。

用法:
    .\\run.ps1 tools\\ingest_data.py              # 自动扫描并解压
    .\\run.ps1 tools\\ingest_data.py --src D:\\下载  # 指定压缩包所在目录

处理逻辑:
    1. 在工作区根目录、data/、以及可选 --src 目录里找 .zip/.tar/.gz/.tgz；
    2. 递归解压（Kaggle 的下载包里常常还套一层 zip），最多 3 层；
    3. 解压到 data/raw/ 下；
    4. 调用 src.data.discover 自检，打印类别与样本数；
    5. 校验 sample_submission.csv 的 ID 能否和 test 图片对上（这是能否提交的关键）。
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARCHIVE_SUFFIXES = (".zip", ".tar", ".gz", ".tgz", ".tar.gz")


def find_archives(dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    skip = {".venv", ".hf_cache", ".tmp", "runs", "submissions", "src", "tools", "_ref"}
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if any(part in skip for part in p.parts):
                continue
            if p.name.lower().endswith(ARCHIVE_SUFFIXES):
                out.append(p)
    return sorted(set(out))


def extract(archive: Path, dest: Path) -> Path:
    """解压单个压缩包，返回其解压目录。"""
    target = dest / archive.stem.replace(".tar", "")
    target.mkdir(parents=True, exist_ok=True)

    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    else:
        with tarfile.open(archive) as t:
            t.extractall(target)
    return target


def extract_recursive(start_dirs: list[Path], dest: Path, max_depth: int = 3) -> None:
    """反复解压，直到没有新的压缩包（Kaggle 下载包经常套娃）。"""
    for depth in range(max_depth):
        archives = find_archives(start_dirs)
        if not archives:
            if depth == 0:
                print("[!] 没有找到任何压缩包")
            return
        print(f"\n--- 第 {depth+1} 轮解压，发现 {len(archives)} 个压缩包 ---")
        for a in archives:
            try:
                out = extract(a, dest)
                size_mb = a.stat().st_size / 1024**2
                print(f"    {a.name}  ({size_mb:.1f} MB) -> {out}")
            except Exception as e:
                print(f"    {a.name} 解压失败: {type(e).__name__}: {e}")
        # 下一轮只在刚解压出来的目录里找
        start_dirs = [dest]


def check_sample_submission(layout):
    """校验 sample_submission 的 ID 与 test 图片能否对上——对不上就交不了。"""
    from src.data import list_test_samples

    if layout.sample_submission is None:
        print("\n[!] 没找到 sample_submission.csv —— 提交文件的列名/ID 格式只能靠猜")
        print("    建议补上，否则可能出现「提交成功但分数为 0」")
        return

    import pandas as pd

    df = pd.read_csv(layout.sample_submission)
    print(f"\n[sample_submission] {layout.sample_submission}")
    print(f"    形状 {df.shape}  列名 {list(df.columns)}")
    print("    前 5 行:")
    print("      " + df.head(5).to_string(index=False).replace("\n", "\n      "))

    test_paths = list_test_samples(layout.test_dir)
    sample_ids = df[df.columns[0]].astype(str).tolist()

    by_name = {Path(s).name for s in sample_ids}
    by_stem = {Path(s).stem for s in sample_ids}
    hit_name = sum(1 for p in test_paths if Path(p).name in by_name)
    hit_stem = sum(1 for p in test_paths if Path(p).stem in by_stem)

    print(f"\n    test 图片数 {len(test_paths)} / 提交行数 {len(sample_ids)}")
    print(f"    按完整文件名匹配: {hit_name}/{len(test_paths)}")
    print(f"    按去扩展名匹配  : {hit_stem}/{len(test_paths)}")
    if len(sample_ids) != len(test_paths):
        print("    [!] 行数与图片数不一致，请人工确认测试集是否找对了")
    if hit_name == 0 and hit_stem == 0:
        print("    [!] 完全匹配不上！很可能 test 目录找错了，或 ID 是纯数字序号")
    elif hit_stem > hit_name:
        print("    [i] ID 不含扩展名，predict.py 会自动按去扩展名匹配（已支持）")
    else:
        print("    [ok] ID 与测试图片文件名可直接对应")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="*", default=None, help="额外搜索压缩包的目录")
    ap.add_argument("--dest", default="data/raw", help="解压目标目录")
    ap.add_argument("--keep", action="store_true", help="解压后保留压缩包")
    args = ap.parse_args()

    dest = (ROOT / args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    dirs = [ROOT, ROOT / "data"]
    if args.src:
        dirs += [Path(s).expanduser().resolve() for s in args.src]

    archives_before = find_archives(dirs)
    extract_recursive(dirs, dest, max_depth=3)

    print("\n" + "=" * 74)
    from src.data import discover

    try:
        layout = discover(ROOT / "data")
    except Exception as e:
        print(f"[X] 数据自检失败: {e}")
        print("\n当前 data/ 下的内容:")
        for p in sorted((ROOT / "data").rglob("*"))[:60]:
            print("   ", p.relative_to(ROOT))
        return 1

    print(layout.summary())

    from src.data import list_test_samples, list_train_samples

    paths, labels = list_train_samples(layout.train_dir, layout.class_to_idx)
    import numpy as np

    counts = np.bincount(labels, minlength=len(layout.classes))
    print(f"\n训练图片 {len(paths)} 张，分布:")
    for c, i in sorted(layout.class_to_idx.items(), key=lambda kv: kv[1]):
        print(f"    {c:34s} {counts[i]:5d}")

    if layout.test_dir:
        tp = list_test_samples(layout.test_dir)
        print(f"\n测试图片 {len(tp)} 张，示例: {[Path(p).name for p in tp[:5]]}")
    else:
        print("\n[X] 没有找到测试集目录")

    check_sample_submission(layout)

    if not args.keep:
        for a in archives_before:
            try:
                a.unlink()
                print(f"\n已删除源压缩包: {a.name}")
            except Exception:
                pass

    print("\n" + "=" * 74)
    print("数据就绪。下一步：")
    print("    .\\run.ps1 -m src.train --data data --model convnext_tiny "
          "--img-size 384 --epochs 30 --folds 5 --tag cnxt384")
    return 0


if __name__ == "__main__":
    sys.exit(main())
