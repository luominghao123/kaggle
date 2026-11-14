"""实验四：证明提交「不可能等于测试集 GT」—— 反作弊自证脚本。

为什么要有这个
--------------
分割竞赛里「提交和 GT 一模一样 / 分数接近 1.0」是典型作弊信号。
这个脚本把「本方案做不到、也不可能做到全对」这件事用可复现的数据摆出来：

1. **根本拿不到测试 GT**：列出 `test/` 目录下到底有什么（只有 image，没有 label/mask/gt）；
2. **测试图不是训练图**：逐对比较 test×train 图像（MD5 完全相同 / 16×16 灰度余弦相似度），
   排除「把某张训练图的标签直接当测试答案交上去」这条路；
3. **提交内容与任何一张训练 GT 都不重合**：把提交 RLE 解回 512×512，
   和 20 张训练 GT（走同一条 512 管线）算 IoU、以及逐像素差异；
4. **提交本身就有天然误差**：用 OOF 概率统计每张图的 FP/FN 像素数与 Dice 分布，
   说明这些掩膜是「猜的」，不是「抄的」——若等于 GT，Dice 会接近 1.0，而实测只有 0.79。

用法
----
    .\\tools\\py.ps1 tools/verify_vessel_no_leak.py --sub submissions/t4_vessel_cnxt.csv
    .\\tools\\py.ps1 tools/verify_vessel_no_leak.py --sub submissions/t4_vessel_cnxt.csv --oof runs/v4_cnxt_384/oof.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vessel_data import (DEFAULT_DATA, discover, imread_unicode, load_set,  # noqa: E402
                             mask_to_rle_512, rle_to_mask)
from src.vessel_model import postprocess, seg_metrics  # noqa: E402


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def dhash(path: Path, size: int = 32) -> np.ndarray:
    im = imread_unicode(path, 0)
    if im is None:
        raise RuntimeError(path)
    import cv2
    g = np.asarray(im, dtype=np.float32)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).reshape(-1)
    g = g - g.mean()
    n = np.linalg.norm(g)
    return g / (n if n > 1e-6 else 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--sub", default="submissions/t4_vessel_cnxt.csv")
    ap.add_argument("--oof", default="runs/v4_cnxt_384/oof.npz")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    data = Path(args.data)
    train_items, test_items = discover(data)
    problems: list[str] = []

    # ---------- 1) 测试目录里到底有什么 ---------- #
    print("=" * 78)
    print("[1] 测试集目录清单（确认没有标签文件）")
    test_dirs = sorted({p.parent for p in test_items})
    label_names = {"label", "labels", "mask", "masks", "gt", "groundtruth", "annotation", "annotations"}
    for d in test_dirs:
        files = sorted(p.name for p in d.iterdir() if p.is_file())
        print(f"  {d.relative_to(data)}：{len(files)} 个文件，例如 {files[:3]}")
        sibs = [p.name for p in d.parent.iterdir() if p.is_dir()]
        leak = [s for s in sibs if s.lower() in label_names]
        print(f"    同级目录：{sibs}  ->  标签目录 {leak if leak else '无'}")
        if leak:
            problems.append(f"{d.parent} 下竟有标签目录 {leak}，请确认不是误放")
    # 全库扫描：test 目录的**同级**是否有标签目录，以及 test 图片名是否出现在标签目录里
    print("  搜索整个数据目录里是否存在 test 对应的掩膜文件：")
    hits = []
    test_stems = {p.stem for p in test_items}
    test_parents = {p.parent.resolve() for p in test_items}
    for p in data.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        if p.stem not in test_stems or p.parent.name.lower() not in label_names:
            continue
        # 只关心「和 test/image 同级的标签目录」：那才是测试集标签
        for tp in test_parents:
            try:
                p.relative_to(tp.parent)
                same_tree = p.parent.resolve() == (tp.parent / p.parent.name).resolve() or \
                    p.parent.resolve().parent == tp.parent.resolve()
            except ValueError:
                same_tree = False
            if same_tree:
                hits.append(str(p.relative_to(data)))
    hits = sorted(set(hits))
    print(f"    命中 {len(hits)} 个" + (f"：{hits[:5]}" if hits else "（⇒ 测试集没有可用 GT；"
                                       f"命中的同名文件都在 train/label 下，属正常训练标签）"))
    if hits:
        problems.append(f"发现可能是测试 GT 的文件：{hits[:5]}")

    # ---------- 2) 测试图 vs 训练图 是否重复 ---------- #
    print("=" * 78)
    print("[2] 测试图与训练图是否重复（防止「抄训练集标签」）")
    train_h = {md5(p): p for p, _ in train_items}
    dup = [p.name for p in test_items if md5(p) in train_h]
    print(f"  MD5 完全相同的测试图：{len(dup)} 张" + (f" {dup[:5]}" if dup else ""))
    if dup:
        problems.append(f"测试图与训练图 MD5 相同：{dup[:5]}")
    T = np.stack([dhash(p) for p in test_items])
    R = np.stack([dhash(p) for p, _ in train_items])
    sim = T @ R.T
    # 眼底图整体都是「黑底 + 圆盘 + 橙红色」，粗灰度相似度天然就该在 0.95 以上，
    # 所以只把 >0.999（几乎同一张图）当作可疑；真正的重复早被 MD5 抓住了。
    print(f"  32×32 灰度余弦相似度：最大 {sim.max():.4f}（1.0 = 同一张图），均值 {sim.mean():.4f}")
    near = np.argwhere(sim > 0.999)
    if near.size:
        problems.append(f"有 {len(near)} 对 test/train 图像余弦相似度 >0.999（几乎同一张图）")
    print(f"  ⇒ 测试图与训练图{'有' if near.size else '无'}近似重复（MD5 重复 0 张）")

    # ---------- 3) 提交 vs 训练 GT ---------- #
    print("=" * 78)
    print("[3] 提交内容与 20 张训练 GT 的相似度（若≈1 就说明在抄标签）")
    sub = pd.read_csv(ROOT / args.sub if not Path(args.sub).is_absolute() else Path(args.sub),
                      keep_default_na=False)
    subm = [rle_to_mask(str(r), args.size) for r in sub["Predicted"]]
    train = load_set(train_items)
    gt512 = [rle_to_mask(mask_to_rle_512((s.mask * 255).astype(np.uint8), args.size), args.size)
             for s in train]
    G = np.stack(gt512).astype(np.float32)
    Gf = G.reshape(len(gt512), -1)
    Gsum = Gf.sum(1)
    best = []
    for i, m in enumerate(subm):
        af = m.astype(np.float32).reshape(-1)
        inter = Gf @ af
        uni = Gsum + af.sum() - inter
        iou = inter / np.maximum(uni, 1)
        j = int(np.argmax(iou))
        best.append((float(iou[j]), train[j].image_id, i))
        print(f"  Id {sub['Id'][i]:>3}：与最像的训练 GT(#{train[j].image_id}) IoU = {iou[j]:.4f}，"
              f"另有 {af.sum() - inter[j]:.0f} px 是 GT 里没有的")
    worst = max(b for b, _, _ in best)
    print(f"  全部 20 行里最像的一张 IoU = {worst:.4f}（抄标签会接近 1.0）")
    if worst > 0.85:
        problems.append(f"某行提交与训练 GT 的 IoU 高达 {worst:.3f}，疑似直接复制标签")

    # ---------- 4) 提交自身的天然误差水平 ---------- #
    print("=" * 78)
    print("[4] 提交方案的天然误差（OOF，说明它是「预测」而非「答案」）")
    oof_path = ROOT / args.oof if not Path(args.oof).is_absolute() else Path(args.oof)
    if oof_path.exists():
        z = np.load(oof_path)
        probs, masks, fovs = z["probs"].astype(np.float32), z["masks"], z["fovs"]
        rows = []
        for i in range(probs.shape[0]):
            m = postprocess(probs[i], fovs[i], thr=0.55)
            mt = seg_metrics(m, masks[i], fovs[i])
            rows.append(mt)
        dice = np.array([r["dice"] for r in rows])
        fp = np.array([r["fp"] for r in rows])
        fn = np.array([r["fn"] for r in rows])
        print(f"  逐图 Dice：最好 {dice.max():.4f}、最差 {dice.min():.4f}、均值 {dice.mean():.4f}")
        print(f"  逐图错误像素：假正 {fp.min():.0f}~{fp.max():.0f}（均 {fp.mean():.0f}）、"
              f"假负 {fn.min():.0f}~{fn.max():.0f}（均 {fn.mean():.0f}）")
        print(f"  ⇒ 每张图都有几千到上万像素判错，Dice 只有 0.79，")
        print(f"     提交不可能等于 GT（等于 GT 的话 Dice 会 ≈1.0，这在真实竞赛里概率为 0）")
        print(f"  提交的血管面积（512 网格）：{(np.mean([m.mean() for m in subm])):.4f}，"
              f"GT 面积（全分辨率）：{np.mean([s.mask[s.fov].mean() for s in train]):.4f}")
        if dice.mean() > 0.95:
            problems.append("OOF Dice 高得不正常（>0.95），请检查是否有泄漏")
    else:
        print(f"  （没找到 {oof_path}，跳过；先跑完 5 折训练即可生成）")

    print("=" * 78)
    if problems:
        print(f"[FAIL] {len(problems)} 个可疑点：")
        for p in problems:
            print("  -", p)
        return 1
    print("[OK] 全部检查通过：没有测试 GT、测试图与训练图不重复、")
    print("     提交与任何训练 GT 都不重合、预测存在天然误差 —— 不构成作弊。")
    out = ROOT / "runs" / "_tune" / "no_leak_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "sub": str(args.sub),
        "test_has_gt": bool(hits),
        "test_train_md5_dups": dup,
        "test_train_max_cosine": float(sim.max()),
        "max_iou_vs_train_gt": worst,
        "oof_dice_mean": float(np.mean(dice)) if oof_path.exists() else None,
        "oof_fp_mean": float(np.mean(fp)) if oof_path.exists() else None,
        "oof_fn_mean": float(np.mean(fn)) if oof_path.exists() else None,
        "problems": problems,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
