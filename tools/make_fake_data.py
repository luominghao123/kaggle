"""生成结构相同的「假数据」，用于在真实数据到位前打通全流程。

只用于冒烟测试，测试完请删除 _smoke_data 目录（避免被 discover() 误选）。

用法:
    python tools/make_fake_data.py [输出目录] [每类训练张数] [测试张数]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = [
    "Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat",
    "Fat Hen", "Loose Silky-bent", "Maize", "Scentless Mayweed",
    "Shepherds Purse", "Small-flowered Cranesbill", "Sugar beet",
]


def make_image(rng, cls_idx: int, size: int = 256) -> Image.Image:
    """按类别生成「有颜色倾向」的噪声图，让模型有机会学到点东西。"""
    base = np.zeros((size, size, 3), dtype=np.float32)
    hue = np.array(
        [(37, 84, 40), (120, 160, 60), (60, 120, 50), (90, 150, 70),
         (180, 170, 90), (70, 110, 45), (45, 95, 55), (140, 180, 80),
         (100, 130, 60), (160, 150, 85), (80, 160, 90), (190, 140, 70)][cls_idx % 12],
        dtype=np.float32,
    )
    base += hue[None, None, :]
    base += rng.normal(0, 18, base.shape)
    # 加一个绿色"叶片"斑块
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = rng.integers(70, size - 70, 2)
    r = rng.integers(35, 75)
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < r**2
    base[mask] += np.array([-40, 40, -30], dtype=np.float32)
    base = np.clip(base, 0, 255).astype(np.uint8)
    return Image.fromarray(base)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "_smoke_data")
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    root = out / "new_plant" / "new_plant"
    train_dir = root / "train"
    test_dir = root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    total = 0
    for ci, cls in enumerate(CLASSES):
        d = train_dir / cls
        d.mkdir(parents=True, exist_ok=True)
        # 模拟类别不平衡：第一个类样本多一些
        n = n_train + (12 if ci == 6 else 0) - (8 if ci == 4 else 0)
        for j in range(max(n, 6)):
            make_image(rng, ci).save(d / f"{cls.lower().replace(' ', '_')}_{j:03d}.png")
            total += 1

    for j in range(n_test):
        make_image(rng, int(rng.integers(0, 12))).save(test_dir / f"test_{j:03d}.png")

    import pandas as pd

    pd.DataFrame(
        {"ID": [f"test_{j:03d}.png" for j in range(n_test)],
         "Category": [CLASSES[0]] * n_test}
    ).to_csv(root / "sample_submission.csv", index=False)

    print(f"假数据已生成: {root}")
    print(f"  训练 {total} 张 / {len(CLASSES)} 类，测试 {n_test} 张")
    print("自检: python -m src.data " + str(out))


if __name__ == "__main__":
    main()
