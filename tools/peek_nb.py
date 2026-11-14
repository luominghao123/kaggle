"""查看参考 notebook 的单元格内容（上一届同课程实验）。

用法:
    python tools/peek_nb.py _ref/lab1_plant_2025.ipynb [起始] [结束]
"""
import json
import sys


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"_ref\lab1_plant_2025.ipynb"
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else lo + 12
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 1500

    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    print(f"# {path}  共 {len(cells)} 个单元格\n")
    for i, c in enumerate(cells):
        if not (lo <= i < hi):
            continue
        src = "".join(c.get("source", []))
        kind = c["cell_type"]
        print("=" * 72)
        print(f"--- cell {i} [{kind}] ---")
        print(src[:limit])
        if len(src) > limit:
            print(f"... (截断，共 {len(src)} 字符)")
    print("=" * 72)


if __name__ == "__main__":
    main()
