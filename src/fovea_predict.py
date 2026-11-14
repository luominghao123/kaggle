"""任务三（中心凹检测定位）推理 / 提交生成：多折多模型集成 + TTA + 可选二段放大精修。

用法
----
    # 单次训练结果 -> 提交文件
    .\\run.ps1 -m src.fovea_predict --data data --runs runs/f512 --out submissions/f512.csv

    # 多个骨干集成（不同 tag 一起平均）
    .\\run.ps1 -m src.fovea_predict --data data --runs runs/f512 runs/f512_cnxt `
        --out submissions/f_ens.csv

    # 关掉 TTA / 打开二段精修
    .\\run.ps1 -m src.fovea_predict --data data --runs runs/f512 --no-tta
    .\\run.ps1 -m src.fovea_predict --data data --runs runs/f512 --refine

坐标口径
--------
模型内部一律是「letterbox 后正方形图上的归一化坐标」；
提交前用每张原图的 (w, h) 与 letterbox 参数还原成 **标签表所用的单位**
（`FoveaLayout.unit`），保证和 sample_submission 的量纲一致。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .engine import get_amp_context
from .fovea_data import (FoveaDataset, FoveaLayout, build_samples_from_layout,
                         discover, image_size, label_to_norm, letterbox,
                         letterbox_params, norm_to_label)
from .fovea_model import build_model
from .utils import describe_device, get_logger, save_json

# ==========================================================================
# 基础推理
# ==========================================================================


@torch.no_grad()
def predict_paths(model, paths, img_size, device, *, batch_size=16,
                  amp_dtype="bf16", tta=True) -> np.ndarray:
    """对一组图片出 letterbox 归一化坐标（多视角 TTA 平均）。返回 (N, 2)。"""
    ds = FoveaDataset(paths, None, img_size, train=False, cache=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=True)
    model.eval()
    ctx, _, _ = get_amp_context(device, amp_dtype)

    def run(flip: bool) -> np.ndarray:
        out = []
        for images, _, _ in loader:
            images = images.to(device, non_blocking=True)
            if flip:
                images = torch.flip(images, dims=[3])
            with ctx():
                pred = model(images)["coords"]
            pred = pred.float().cpu().numpy()
            if flip:                       # 水平翻转后 x 坐标要镜像回来
                pred[:, 0] = 1.0 - pred[:, 0]
            out.append(pred)
        return np.concatenate(out)

    preds = run(False)
    if tta:
        preds = 0.5 * (preds + run(True))
    return preds


@torch.no_grad()
def refine_paths(model, paths, coarse_norm, img_size, unit, device, *,
                 window: float = 0.35, amp_dtype="bf16") -> np.ndarray:
    """二段精修：以粗预测为中心裁一小块、把这块放大到 img_size 再回归一次。

    中心凹在整张眼底图上只占几个像素，直接在全图上回归的分辨率有限；
    裁窗后目标在输入里的相对尺寸大一个量级，定位精度明显更高。
    """
    model.eval()
    ctx, _, _ = get_amp_context(device, amp_dtype)
    out = coarse_norm.copy()

    for i, path in enumerate(paths):
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            s, ox, oy = letterbox_params(w, h, img_size)
            px, py = norm_to_label(float(coarse_norm[i, 0]), float(coarse_norm[i, 1]),
                                   (w, h), "pixel", s, ox, oy, img_size)

            side = window * min(w, h)
            x0 = int(round(np.clip(px - side / 2, 0, max(w - side, 1))))
            y0 = int(round(np.clip(py - side / 2, 0, max(h - side, 1))))
            x1 = int(round(min(w, x0 + side)))
            y1 = int(round(min(h, y0 + side)))
            crop = im.crop((x0, y0, x1, y1))
            cw, ch = crop.size

            lb, s2, ox2, oy2 = letterbox(crop, img_size)
            ten = torch.from_numpy(np.array(lb)).float().permute(2, 0, 1).div_(255.0)
            ten = (ten - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                  torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            with ctx():
                pred = model(ten.unsqueeze(0).to(device))["coords"].float().cpu().numpy()[0]

            cx, cy = norm_to_label(float(pred[0]), float(pred[1]), (cw, ch), "pixel",
                                   s2, ox2, oy2, img_size)
            cx, cy = cx + x0, cy + y0
            # 换回 letterbox 归一化坐标
            u, v = label_to_norm(cx, cy, (w, h), "pixel", s, ox, oy, img_size)
            out[i] = (u, v)
    return out


# ==========================================================================
# 组装
# ==========================================================================


def load_models(run_dirs: list[Path], device) -> list[dict]:
    """加载所有折 / 所有 run 的最优权重。"""
    models = []
    for rd in run_dirs:
        ckpts = sorted(rd.glob("fold*_best.pt"))
        if not ckpts:
            print(f"[!] {rd} 下没有 fold*_best.pt，跳过")
            continue
        for ck in ckpts:
            state = torch.load(ck, map_location="cpu", weights_only=False)
            a = state.get("args", {})
            net = build_model(
                a.get("backbone", "effnet_b0"),
                pretrained=False,
                head_dim=a.get("head_dim", 64),
                out_stride=a.get("out_stride", 4),
                drop_path_rate=0.0,
                verbose=False,
            )
            net.load_state_dict(state["model"])
            net.to(device).eval()
            models.append({
                "model": net,
                "img_size": a.get("img_size", 512),
                "score": state.get("best_score", float("nan")),
                "tag": rd.name,
                "fold": int(ck.stem.split("fold")[1].split("_")[0]),
            })
            print(f"[load] {ck}  img_size={a.get('img_size', 512)}  "
                  f"val MCRMSE={state.get('best_score', float('nan')):.5f} "
                  f"({state.get('best_src', '?')})")
    if not models:
        raise RuntimeError("没有加载到任何权重，请检查 --runs 路径")
    return models


def convert_to_label_unit(pred_norm: np.ndarray, paths: list[Path], img_size: int,
                         unit: str) -> np.ndarray:
    out = np.zeros_like(pred_norm, dtype=np.float64)
    for i, p in enumerate(paths):
        w, h = image_size(p)
        s, ox, oy = letterbox_params(w, h, img_size)
        out[i, 0], out[i, 1] = norm_to_label(
            float(pred_norm[i, 0]), float(pred_norm[i, 1]), (w, h), unit, s, ox, oy, img_size)
    return out


def read_submission_template(path: Path | None) -> tuple[list[str], list[list[str]]] | None:
    if path is None or not Path(path).exists():
        return None
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                rows = [r for r in csv.reader(f) if r]
            if rows:
                return rows[0], rows[1:]
        except UnicodeDecodeError:
            continue
    return None


def write_submission(out_path: Path, layout: FoveaLayout, paths: list[Path],
                     coords: np.ndarray, x_col: str, y_col: str, id_col: str) -> None:
    """按 sample_submission 的形态写提交文件。

    支持两种表：
      * 宽表 `ID,x,y`                —— 每张图一行
      * 长表 `ImageID,value`         —— 每张图两行（`81_Fovea_X` / `81_Fovea_Y`）
    行顺序与 ID 写法完全照抄模板，错一个 ID 就等于白交。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_name: dict[str, tuple[float, float]] = {}
    by_path: dict[Path, tuple[float, float]] = {}
    for p, (x, y) in zip(paths, coords):
        by_name[p.name] = (float(x), float(y))
        by_name[p.stem] = (float(x), float(y))
        by_path[Path(p)] = (float(x), float(y))

    def lookup(key: str):
        return (by_name.get(key) or by_name.get(Path(key).stem)
                or by_name.get(Path(key).name))

    # ---------------- 长表 ----------------
    if layout.submission_fmt == "long" and layout.submission_rows:
        header = layout.submission_columns or ["ImageID", "value"]
        value_col = 1 if len(header) > 1 else 1
        filled, missing = 0, 0
        out_rows = []
        for i, r in enumerate(layout.submission_rows):
            r = list(r) + [""] * (len(header) - len(r))
            pair = layout.submission_pairs[i] if i < len(layout.submission_pairs) else None
            if pair is None:
                missing += 1
                r[value_col] = "0.0"
            else:
                path, axis = pair
                xy = by_path.get(Path(path))
                if xy is None:
                    missing += 1
                    r[value_col] = "0.0"
                else:
                    r[value_col] = f"{xy[axis]:.6f}"
                    filled += 1
            out_rows.append(r)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(header)
            wr.writerows(out_rows)
        print(f"[提交] {out_path}（长表）行数 {len(out_rows)}"
              f"（匹配 {filled}，缺失 {missing}）")
        if missing:
            print("[!] 有提交行没解析出图片，已填 0.0 占位——请检查 ID 规则")
        return

    # ---------------- 宽表 ----------------
    tpl = read_submission_template(layout.sample_submission)
    if tpl is None:
        header = [id_col or "ID", x_col, y_col]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(header)
            for p, (x, y) in zip(paths, coords):
                wr.writerow([p.name, f"{x:.6f}", f"{y:.6f}"])
        print(f"[提交] 无 sample_submission，按 ID/{x_col}/{y_col} 写出 {len(paths)} 行")
        return

    header, rows = tpl
    ji = header.index(id_col) if id_col in header else 0
    jx = header.index(x_col) if x_col in header else None
    jy = header.index(y_col) if y_col in header else None
    if jx is None or jy is None:
        numeric_cols = [c for c in header if c != header[ji]]
        jx = header.index(numeric_cols[0])
        jy = header.index(numeric_cols[1])
        print(f"[!] 模板列名与标签列名不一致，按位置取 {header[jx]!r}/{header[jy]!r}")

    filled, missing = 0, 0
    out_rows = []
    for r in rows:
        r = list(r) + [""] * (len(header) - len(r))
        key = r[ji].strip()
        hit = lookup(key)
        if hit is None:
            missing += 1
            r[jx], r[jy] = "0.5", "0.5"
        else:
            r[jx], r[jy] = f"{hit[0]:.6f}", f"{hit[1]:.6f}"
            filled += 1
        out_rows.append(r)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(out_rows)
    print(f"[提交] {out_path}  行数 {len(out_rows)}（匹配 {filled}，缺失 {missing}）")
    if missing:
        print("[!] 有 ID 没匹配上图片，已填 0.5 占位——请检查测试图目录是否找对了")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="中心凹定位：集成推理 / 生成提交")
    p.add_argument("--data", default="data")
    p.add_argument("--runs", nargs="+", required=True, help="一个或多个训练产物目录")
    p.add_argument("--out", default="submissions/fovea.csv")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-tta", action="store_true", help="关闭水平翻转 TTA")
    p.add_argument("--refine", action="store_true", help="开启二段放大精修")
    p.add_argument("--refine-window", type=float, default=0.35,
                   help="精修裁窗边长 = 该比例 × 图像短边")
    p.add_argument("--save-probs", action="store_true", help="把归一化坐标另存为 .npy")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("fovea_predict")
    logger.info(f"设备: {describe_device()}")

    layout = discover(args.data)
    logger.info("数据布局:\n" + layout.summary())
    run_dirs = [Path(r) for r in args.runs]

    models = load_models(run_dirs, device)
    test_paths = list(layout.test_images)
    if not test_paths:
        raise RuntimeError("没找到测试图片")

    # ---------- 1) 每个模型 × 每张测试图 ----------
    per_model = []
    for m in models:
        pred = predict_paths(m["model"], test_paths, m["img_size"], device,
                             batch_size=args.batch_size, tta=not args.no_tta)
        per_model.append(pred)
        logger.info(f"  {m['tag']} fold{m['fold']}  img_size={m['img_size']} 完成")
    ens = np.mean(np.stack(per_model, axis=0), axis=0)

    # ---------- 2) 可选二段精修（对集成结果做，再按模型平均） ----------
    if args.refine:
        logger.info("二段精修中（以集成粗预测为中心裁窗）...")
        refined = []
        for m in models:
            refined.append(refine_paths(m["model"], test_paths, ens, m["img_size"],
                                        layout.unit, device, window=args.refine_window))
        ens_ref = np.mean(np.stack(refined, axis=0), axis=0)
        # 两种结果各取一半，避免精修在个别图上跑飞
        ens = 0.5 * (ens + ens_ref)
        logger.info("精修完成（与粗预测 50/50 融合）")

    # ---------- 3) 还原单位并写提交 ----------
    coords = convert_to_label_unit(ens, test_paths, models[0]["img_size"], layout.unit)
    out_path = Path(args.out)
    write_submission(out_path, layout, test_paths, coords,
                     layout.x_column, layout.y_column, layout.id_column)

    if args.save_probs:
        np.save(out_path.with_suffix(".npy"), ens)
        logger.info(f"归一化坐标已存至 {out_path.with_suffix('.npy')}")

    # ---------- 4) 顺带报告各 run 的 OOF 成绩，便于判断集成是否有收益 ----------
    oof_rows = []
    for rd in run_dirs:
        preds, gts = [], []
        for f in sorted(rd.glob("val_pred_fold*.npy")):
            gt_f = rd / f.name.replace("val_pred", "val_gt")
            if gt_f.exists():
                preds.append(np.load(f))
                gts.append(np.load(gt_f))
        if preds:
            pr, gt = np.concatenate(preds), np.concatenate(gts)
            score = float(np.mean(np.sqrt(np.mean((pr - gt) ** 2, axis=0))))
            oof_rows.append({"run": rd.name, "n": len(pr), "oof_mcrmse": score})
            logger.info(f"  {rd.name}: OOF MCRMSE {score:.5f}（{len(pr)} 张）")

    save_json({
        "runs": [str(r) for r in run_dirs],
        "n_models": len(models),
        "n_test": len(test_paths),
        "tta": not args.no_tta,
        "refine": args.refine,
        "unit": layout.unit,
        "oof": oof_rows,
        "out": str(out_path),
        "pred_min": coords.min(axis=0).tolist(),
        "pred_max": coords.max(axis=0).tolist(),
    }, out_path.with_suffix(".json"))

    logger.info(f"\n提交文件: {out_path}")
    logger.info("提交前自查: .\\tools\\py.ps1 tools\\check_fovea_submission.py "
                f"{out_path} --data {args.data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
