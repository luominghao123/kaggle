"""检查训练集内部的近似重复对，重点看是否存在「同图不同标签」的标签冲突。

标签冲突会直接压住模型上限：同一张图既算 A 类又算 B 类，模型无论怎么学都要错一半。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import discover  # noqa: E402
from tools.check_leakage import collect, dhash  # noqa: E402


def main():
    max_dist = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    layout = discover(ROOT / "data")
    root = layout.train_dir.parent
    items = collect(root)

    hashes, splits, paths = [], [], []
    for s, p in items:
        h = dhash(p)
        if h is not None:
            hashes.append(h)
            splits.append(s)
            paths.append(p)

    arr = np.array(hashes, dtype=np.uint64)
    n = len(arr)
    print(f"图片 {n} 张，阈值 dHash Hamming <= {max_dist}（越小越像）\n")

    conflicts = 0
    pairs = 0
    for i in range(n):
        d = np.bitwise_count(arr[i] ^ arr[i + 1:])
        for off in np.nonzero(d <= max_dist)[0]:
            j = i + 1 + int(off)
            pairs += 1
            a, b = splits[i], splits[j]
            same = a == b
            tag = "同类" if same else "!!! 跨类 = 标签冲突"
            if not same:
                conflicts += 1
            print(f"  dist={int(d[off]):2d}  {a:26s} {paths[i].name}"
                  f"  <->  {b:26s} {paths[j].name}    {tag}")

    print(f"\n近似重复对总数: {pairs}   其中跨类（标签冲突）: {conflicts}")
    if conflicts == 0:
        print("结论: 训练集没有标签冲突，标注质量没问题。")
    else:
        print("结论: 存在标签冲突，这些样本会同时拉低训练与验证指标，建议在报告中说明。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
