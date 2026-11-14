"""生成「假眼底图 + 中心凹坐标」数据集，用于在真实数据到手前验证整条管线。

真数据（2026task3）2026-10-20 才开放，但代码必须先在本地跑通：
这个脚本按眼底图的视觉结构造图——
    黑色背景 + 圆形视网膜区域 + 放射状血管 + 亮视盘 + 暗中心凹小斑，
中心凹位置随机，坐标以「按图边长归一化」的形式写进标签表，
和真实竞赛数据（图片 ID + 两个坐标列）的表格结构保持一致。

用法:
    .\\run.ps1 tools\\make_fake_fovea.py --out data_fake --n-train 160 --n-test 40

产物:
    data_fake/
        train/  *.png             训练图
        train.csv                 image_id,x,y
        test/   *.png             测试图
        sample_submission.csv     ID,x,y
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]


def make_fundus(rng: random.Random, w: int, h: int, fx: float, fy: float) -> Image.Image:
    """造一张假眼底图。fx/fy 是中心凹的归一化坐标（0~1，按宽/高归一化）。"""
    # ---- 视网膜圆形区域 ----
    cx = w * 0.5 + rng.uniform(-0.04, 0.04) * w
    cy = h * 0.5 + rng.uniform(-0.04, 0.04) * h
    R = 0.47 * min(w, h) * rng.uniform(0.92, 1.0)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.hypot(xx - cx, yy - cy)
    retina = (dist < R).astype(np.float32)
    # 边缘略暗的径向渐变，模拟眼底照明不均
    vignette = np.clip(1.0 - 0.7 * (dist / R) ** 2, 0.35, 1.0)

    base = np.zeros((h, w, 3), dtype=np.float32)
    base[..., 0] = 205.0
    base[..., 1] = 120.0
    base[..., 2] = 70.0
    base *= vignette[..., None]
    # 轻微噪声
    base += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 4, base.shape).astype(np.float32)
    base *= retina[..., None]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # ---- 血管：从视盘附近向外放射的折线 ----
    od_x = cx + R * 0.45 * (1 if fx < 0.5 else -1)
    od_y = cy + R * 0.10
    od_r = R * 0.14
    for _ in range(rng.randint(7, 12)):
        ang = rng.uniform(0, 2 * math.pi)
        px, py = od_x + math.cos(ang) * od_r * 0.6, od_y + math.sin(ang) * od_r * 0.6
        pts = [(px, py)]
        for _ in range(rng.randint(3, 6)):
            ang += rng.uniform(-0.7, 0.7)
            step = R * rng.uniform(0.12, 0.25)
            px += math.cos(ang) * step
            py += math.sin(ang) * step
            pts.append((px, py))
        width = rng.randint(2, 5)
        draw.line(pts, fill=(105, 40, 30), width=width, joint="curve")

    # ---- 视盘（亮圆）----
    draw.ellipse([od_x - od_r, od_y - od_r, od_x + od_r, od_y + od_r],
                 fill=(247, 226, 190), outline=(210, 180, 140), width=2)

    # ---- 中心凹（暗斑，两层：较宽的黄斑暗区 + 更小的正中心）----
    px, py = fx * w, fy * h
    mac_r = R * rng.uniform(0.10, 0.14)
    draw.ellipse([px - mac_r, py - mac_r, px + mac_r, py + mac_r], fill=(150, 78, 48))
    fov_r = mac_r * rng.uniform(0.28, 0.38)
    draw.ellipse([px - fov_r, py - fov_r, px + fov_r, py + fov_r], fill=(88, 42, 28))

    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 1.4)))
    return img


def sample_point(rng: random.Random) -> tuple[float, float]:
    """中心凹大致在图像中部偏一侧，避免均匀分布（更接近真实眼底图）。"""
    ang = rng.uniform(0.15 * math.pi, 0.85 * math.pi)
    rad = rng.uniform(0.05, 0.30)
    x = 0.5 + math.cos(ang) * rad
    y = 0.5 + math.sin(ang) * rad * 0.9
    return float(np.clip(x, 0.12, 0.88)), float(np.clip(y, 0.12, 0.88))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_fake")
    ap.add_argument("--n-train", type=int, default=160)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-side", type=int, default=480)
    ap.add_argument("--max-side", type=int, default=800)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    rng_np.random()  # noqa: F841

    out = (ROOT / args.out).resolve()
    for sub in ("train", "test"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    def one_image(path: Path, name: str):
        long_side = rng.randint(args.min_side, args.max_side)
        aspect = rng.uniform(0.66, 1.5)
        if aspect >= 1:
            w, h = long_side, int(long_side / aspect)
        else:
            w, h = int(long_side * aspect), long_side
        fx, fy = sample_point(rng)
        img = make_fundus(rng, w, h, fx, fy)
        img.save(path / name)
        return fx, fy, w, h

    train_rows = []
    for i in range(args.n_train):
        name = f"tr_{i:04d}.png"
        fx, fy, w, h = one_image(out / "train", name)
        train_rows.append((name, round(fx, 6), round(fy, 6)))
        if (i + 1) % 40 == 0:
            print(f"  train {i + 1}/{args.n_train}")

    with open(out / "train.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["image_id", "x", "y"])
        wr.writerows(train_rows)

    test_rows = []
    for i in range(args.n_test):
        name = f"te_{i:04d}.png"
        one_image(out / "test", name)
        test_rows.append(name)
        if (i + 1) % 20 == 0:
            print(f"  test {i + 1}/{args.n_test}")

    with open(out / "sample_submission.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["ID", "x", "y"])
        wr.writerows([[n, 0.5, 0.5] for n in test_rows])

    print(f"\n完成: {out}")
    print(f"    训练 {len(train_rows)} 张 -> train.csv (image_id,x,y，归一化坐标)")
    print(f"    测试 {len(test_rows)} 张 -> sample_submission.csv (ID,x,y)")
    print("\n下一步:")
    print(f"    .\\run.ps1 -m src.fovea_data {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
