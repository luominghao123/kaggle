"""推理入口：多模型 / 多折 / 多视角 TTA 集成，生成提交文件。

用法示例
--------
# 单模型 5 折 + 8 视角 TTA
python -m src.predict --data data --runs runs/convnext_tiny_384_xxxx

# 多模型集成（按 OOF 的 macro-F1 自动加权）
python -m src.predict --data data --runs runs/convnext_xxx runs/swin_xxx runs/effnet_xxx \
    --weights auto --out submissions/ensemble.csv

# 快速模式：直接用训练时保存的 test_fold*.npy（无 TTA）
python -m src.predict --data data --runs runs/convnext_xxx --no-tta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data import (  # type: ignore
        build_eval_transform,
        build_tta_transforms,
        discover,
        list_test_samples,
        PlantDataset,
    )
    from src.engine import compute_metrics, get_amp_context  # type: ignore
    from src.models import build_model  # type: ignore
    from src.utils import describe_device, get_logger, load_json, seed_everything  # type: ignore
else:
    from .data import (
        build_eval_transform,
        build_tta_transforms,
        discover,
        list_test_samples,
        PlantDataset,
    )
    from .engine import compute_metrics, get_amp_context
    from .models import build_model
    from .utils import describe_device, get_logger, load_json, seed_everything


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="植物分类 - 集成推理与提交")
    p.add_argument("--data", default="data")
    p.add_argument("--runs", nargs="+", required=True,
                   help="一个或多个训练产物目录（runs/<tag>）")
    p.add_argument("--out", default=None, help="提交文件路径，默认 submissions/submission.csv")
    p.add_argument("--weights", nargs="*", default=None,
                   help="各 run 的权重；给 'auto' 则按 OOF macro-F1 加权")
    p.add_argument("--tta", dest="tta", action="store_true", default=True)
    p.add_argument("--no-tta", dest="tta", action="store_false")
    p.add_argument("--batch-size", type=int, default=32)
    # 沙箱禁止命名管道 -> num_workers 必须为 0
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-probs", action="store_true", help="同时保存集成后的概率矩阵")
    p.add_argument("--calibrate", default="auto", choices=["off", "auto", "on"],
                   help="按 OOF 搜索每类乘性权重以最大化 macro-F1；"
                        "auto = 仅当 OOF 增益超过 0.003 才启用")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# macro-F1 的类别先验校准
# --------------------------------------------------------------------------


def search_class_weights(
    oof_probs: np.ndarray,
    oof_labels: np.ndarray,
    *,
    iters: int = 4,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """坐标上升搜索每类乘性权重 w，使 argmax(P * w) 的 macro-F1 最大。

    动机：macro-F1 对少数类敏感，而交叉熵训练出的模型倾向于预测多数类。
    在概率上乘一个可学习的类先验，可以在不重训的前提下把决策边界往少数类推。

    返回 (w, 校准前 F1, 校准后 F1)。
    """
    from sklearn.metrics import f1_score

    if grid is None:
        # 搜索范围刻意收窄，避免在 OOF 上过拟合
        grid = np.array([0.80, 0.85, 0.875, 0.925, 0.95, 1.0,
                         1.05, 1.075, 1.125, 1.15, 1.20, 1.30])

    num_classes = oof_probs.shape[1]
    w = np.ones(num_classes, dtype=np.float64)

    def score(weights: np.ndarray) -> float:
        pred = (oof_probs * weights).argmax(axis=1)
        return float(f1_score(oof_labels, pred, average="macro", zero_division=0))

    best = score(w)
    base = best
    for _ in range(iters):
        improved = False
        for c in range(num_classes):
            keep = w[c]
            best_val_c, best_g_c = best, keep
            for g in grid:  # grid 含 1.0，即保留原值的选项始终在候选内
                w[c] = g
                s = score(w)
                if s > best_val_c + 1e-6:
                    best_val_c, best_g_c = s, g
            w[c] = best_g_c
            if best_val_c > best:
                best = best_val_c
                improved = True
        if not improved:
            break
    return w, base, best


# --------------------------------------------------------------------------
# 提交文件
# --------------------------------------------------------------------------


def _norm_stem(name: str) -> str:
    return Path(str(name)).stem


def make_submission(
    test_paths: list[str],
    probs: np.ndarray,
    classes: list[str],
    sample_submission: Path | None,
    out_path: Path,
) -> pd.DataFrame:
    """按 sample_submission 的列名与 ID 格式生成提交文件。

    关键点：ID 必须是评测系统能对上的键。这里优先用 sample_submission.csv
    里的 ID 原样值，通过「文件名 / 去扩展名」两级匹配把它们和测试图片对应起来。
    """
    basenames = [Path(p).name for p in test_paths]
    pred_idx = probs.argmax(axis=1)
    pred_names = [classes[i] for i in pred_idx]

    id_col, label_col = "ID", "Category"
    ids: list[str] = basenames

    if sample_submission is not None and Path(sample_submission).exists():
        sample = pd.read_csv(sample_submission)
        id_col = sample.columns[0]
        label_col = sample.columns[1] if len(sample.columns) > 1 else "Category"
        sample_ids = sample[id_col].astype(str).tolist()

        # 用原图文件名做「预测 -> ID」映射，避免对顺序做任何假设
        sample_by_name = {Path(s).name: s for s in sample_ids}
        sample_by_stem = {_norm_stem(s): s for s in sample_ids}

        mapped, miss = [], 0
        for b in basenames:
            if b in sample_by_name:
                mapped.append(sample_by_name[b])
            elif _norm_stem(b) in sample_by_stem:
                mapped.append(sample_by_stem[_norm_stem(b)])
            else:
                mapped.append(b)
                miss += 1
        ids = mapped
        if miss:
            print(f"[warn] {miss}/{len(basenames)} 张图片在 sample_submission 中未匹配到 ID，"
                  f"已回退为文件名")

    df = pd.DataFrame({id_col: ids, label_col: pred_names})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return df


# --------------------------------------------------------------------------
# 单 run 的测试集概率
# --------------------------------------------------------------------------


@torch.no_grad()
def predict_run(
    run_dir: Path,
    test_paths: list[str],
    classes: list[str],
    args,
    logger,
) -> np.ndarray:
    """把一个 run 目录下所有折的模型做 TTA 推理并平均，返回 (N, C) 概率。"""
    summary_path = run_dir / "summary.json"
    meta = load_json(summary_path) if summary_path.exists() else {}
    model_name = meta.get("model", "convnext_tiny")
    img_size = meta.get("img_size", 384)

    ckpts = sorted(run_dir.glob("fold*_best.pt"))
    if not ckpts:
        ckpts = sorted(run_dir.glob("fold*_last.pt"))
    if not ckpts:
        raise FileNotFoundError(f"{run_dir} 下没有找到 fold*_best.pt / fold*_last.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_tta = args.tta
    views = build_tta_transforms(img_size) if use_tta else [("id", build_eval_transform(img_size))]

    logger.info(f"[{run_dir.name}] 模型={model_name} img_size={img_size} "
                f"检查点={len(ckpts)} TTA视角={len(views)}")

    accum = np.zeros((len(test_paths), len(classes)), dtype=np.float64)
    n_models = 0

    for ckpt_path in ckpts:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        weights = state["ema"] if state.get("best_src") == "ema" and state.get("ema") else state["model"]

        model = build_model(model_name, len(classes), img_size=img_size,
                            pretrained=False, verbose=False).to(device)
        model.load_state_dict(weights)
        model.eval()

        for view_name, tf in views:
            ds = PlantDataset(test_paths, None, tf)
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=(device.type == "cuda"),
                persistent_workers=args.workers > 0,
            )
            probs_list = []
            for images, _paths in loader:
                images = images.to(device, non_blocking=True)
                ctx, _, _ = get_amp_context(device, args.amp)
                with ctx():
                    out = model(images)
                probs_list.append(torch.softmax(out.float(), dim=1).cpu().numpy())
            probs = np.concatenate(probs_list, axis=0)
            accum += probs
            n_models += 1
            logger.info(f"    {ckpt_path.name:20s} view={view_name:8s} 完成")

    accum /= max(n_models, 1)
    return accum


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)

    layout = discover(args.data)
    classes = layout.classes
    test_paths = list_test_samples(layout.test_dir)

    logger = get_logger("plant.predict")
    logger.info("=" * 78)
    logger.info(describe_device())
    logger.info(f"类别 {len(classes)}: {classes}")
    logger.info(f"测试图片 {len(test_paths)} 张")

    run_dirs = [Path(r) for r in args.runs]

    # ---- 权重：auto 表示按 OOF macro-F1 归一化加权 ----
    if args.weights and len(args.weights) == 1 and args.weights[0] == "auto":
        scores = []
        for rd in run_dirs:
            oof_path = rd / "oof.npz"
            if oof_path.exists():
                d = np.load(oof_path)
                m = compute_metrics(d["labels"][d["mask"]], d["probs"][d["mask"]].argmax(1),
                                    len(classes))
                scores.append(m["f1_macro"])
            else:
                scores.append(1.0)
        scores = np.array(scores)
        weights = scores / scores.sum()
        for rd, s, w in zip(run_dirs, scores, weights):
            logger.info(f"  自动权重 {rd.name}: OOF F1={s:.4f} -> w={w:.3f}")
    elif args.weights:
        weights = np.array([float(w) for w in args.weights], dtype=np.float64)
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(run_dirs)) / len(run_dirs)

    all_probs = np.zeros((len(test_paths), len(classes)), dtype=np.float64)
    for rd, w in zip(run_dirs, weights):
        # 已有 best 权重就做 TTA；--no-tta 时优先复用训练时保存的 test_fold*.npy
        if not args.tta:
            fold_files = sorted(rd.glob("test_fold*.npy"))
            if fold_files:
                p = np.mean([np.load(f) for f in fold_files], axis=0)
                logger.info(f"[{rd.name}] 复用训练期保存的 test_fold*.npy（{len(fold_files)} 折）")
                all_probs += w * p
                continue
        p = predict_run(rd, test_paths, classes, args, logger)
        all_probs += w * p

    all_probs /= weights.sum()

    # ---- 集成 OOF + macro-F1 类别先验校准 ----
    # 前提：各 run 训练时用了相同的 seed 与折数，OOF 的划分才可比。
    oof_parts, oof_labels, oof_mask = [], None, None
    for rd, w in zip(run_dirs, weights):
        f = rd / "oof.npz"
        if f.exists():
            d = np.load(f)
            oof_parts.append((d["probs"].astype(np.float64), float(w)))
            oof_labels, oof_mask = d["labels"], d["mask"]
    if oof_parts and oof_mask is not None and oof_mask.any():
        wsum = sum(w for _, w in oof_parts)
        ens_oof = sum(p * w for p, w in oof_parts) / wsum
        y, P = oof_labels[oof_mask], ens_oof[oof_mask]
        base_m = compute_metrics(y, P.argmax(1), len(classes))
        logger.info("集成 OOF 指标: " + "  ".join(f"{k}={v:.4f}" for k, v in base_m.items()))

        if args.calibrate != "off":
            cw, f1_before, f1_after = search_class_weights(P, y)
            gain = f1_after - f1_before
            if args.calibrate == "on" or gain > 0.003:
                all_probs = all_probs * cw
                logger.info(
                    f"启用类别先验校准：OOF macro-F1 {f1_before:.4f} -> {f1_after:.4f} "
                    f"(+{gain:.4f})\n    权重 = " +
                    ", ".join(f"{c}:{v:.2f}" for c, v in zip(classes, cw))
                )
            else:
                logger.info(f"类别先验校准收益仅 {gain:+.4f}，低于阈值，跳过（--calibrate on 可强制开启）")

    out_path = Path(args.out) if args.out else Path("submissions/submission.csv")
    df = make_submission(test_paths, all_probs, classes, layout.sample_submission, out_path)

    logger.info(f"提交文件已写出: {out_path.resolve()}  ({len(df)} 行)")
    logger.info(f"预测类别分布:\n{df.iloc[:, 1].value_counts().to_string()}")

    if args.save_probs:
        np.save(out_path.with_suffix(".probs.npy"), all_probs)
        logger.info(f"概率矩阵已保存: {out_path.with_suffix('.probs.npy')}")

    print(df.head(10).to_string(index=False))
    return out_path


if __name__ == "__main__":
    main()
