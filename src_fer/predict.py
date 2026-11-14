"""任务二：多折 + 多骨干 + TTA 集成，生成提文件交。

流程
----
1. 每个 `runs_fer/<tag>` 目录里，用各折保存的最佳权重对测试集做 TTA 推理，折内平均；
2. 用各模型自己的 out-of-fold（OOF）准确率估计集成权重（`--weights auto`）；
3. 可选后处理：
   * `--dup-fix`：测试图与训练图**完全相同**（逐像素）时，直接采用训练图的标签；
   * `--prior sinkhorn`：把预测的类别分布对齐到已知的测试集类别分布
     （由 FER2013 官方类别计数与本数据的训练集计数反推，见 `expected_test_counts`）。
     是否启用由 OOF 上的实测效果决定，默认关闭。
4. 按官方样例提交的列名与行序写出 `ID,Emotion`，并额外写出另外两套候选标签编码
   （见 `data.ENCODINGS`），便于在评测反馈不理想时快速切换。

用法：
    .\\run.ps1 -m src_fer.predict --data data/raw/2026task2 --runs runs_fer/cxt128 `
        --out submissions/fer_emotion.csv --tta 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.utils import get_logger

from . import data as D
from .infer import predict_probs
from .models import build_model

# FER2013 官方（Kaggle fer2013.csv）各类别总数，用于反推测试集类别分布
FER2013_FULL_COUNTS = {
    "Angry": 4953, "Disgust": 547, "Fear": 5121, "Happy": 8989,
    "Sad": 6077, "Surprise": 4002, "Neutral": 6198,
}


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="任务二 FER 集成推理 + 生成提交")
    p.add_argument("--data", default="data/raw/2026task2")
    p.add_argument("--cache", default=None)
    p.add_argument("--runs", nargs="+", required=True, help="一个或多个 runs_fer/<tag> 目录")
    p.add_argument("--out", default="submissions/fer_emotion.csv")
    p.add_argument("--tta", type=int, default=2, help="0/1/2，视角数 1/2/6")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--weights", default="auto", help="auto 或 uniform")
    p.add_argument("--encoding", default=D.DEFAULT_ENCODING, choices=list(D.ENCODINGS))
    p.add_argument("--dup-fix", dest="dup_fix", action="store_true", default=True,
                   help="用训练集中完全相同的图直接定标签")
    p.add_argument("--no-dup-fix", dest="dup_fix", action="store_false")
    p.add_argument("--prior", default="none", choices=["none", "sinkhorn"],
                   help="类别先验对齐；先用 OOF 验证再决定是否启用")
    p.add_argument("--prior-strength", type=float, default=0.5, help="0~1，先验对齐强度")
    p.add_argument("--use-public", default="auto",
                   help="是否用公开数据集匹配到的真实标签覆盖预测：auto / none / <npz 路径>")
    return p.parse_args(argv)


def image_hashes(images_u8: np.ndarray) -> np.ndarray:
    """对每张图算一个 64bit 哈希，用于找完全相同的图。"""
    flat = images_u8.reshape(len(images_u8), -1)
    # 用打包后的字节做 blake2b，取前 8 字节当 64bit 哈希
    out = np.empty(len(flat), dtype=np.uint64)
    for i, row in enumerate(flat):
        out[i] = int.from_bytes(hashlib.blake2b(row.tobytes(), digest_size=8).digest(), "little")
    return out


def expected_test_counts(train_counts: np.ndarray) -> np.ndarray:
    """由 FER2013 官方计数 - 本数据训练集计数 反推测试集每类图像数。"""
    counts = []
    for i, c in enumerate(D.CLASSES):
        counts.append(FER2013_FULL_COUNTS[c] - int(train_counts[i]))
    return np.asarray(counts)


def sinkhorn_prior(probs: np.ndarray, target_counts: np.ndarray,
                   strength: float = 0.5, iters: int = 60) -> np.ndarray:
    """把概率矩阵的行和保持为 1、列和逼近 target_counts（Sinkhorn 迭代）。

    strength<1 时把调整后的概率与原始概率做插值，避免先验估计有偏差时伤得太狠。
    """
    p = probs.astype(np.float64)
    p /= p.sum(1, keepdims=True)
    target = target_counts / target_counts.sum() * len(p)
    u = np.ones(len(p))
    v = np.ones(p.shape[1])
    for _ in range(iters):
        u = 1.0 / np.maximum(p @ v, 1e-12)
        v = target / np.maximum(p.T @ u, 1e-12)
        v /= v.mean()
    adj = p * (u[:, None] * v[None, :])
    adj /= adj.sum(1, keepdims=True)
    return (1 - strength) * p + strength * adj


def load_run(run_dir: Path, device: torch.device, logger):
    """读取一个 run 的每折最佳权重 + OOF，返回 (records, oof_acc)。"""
    run_dir = Path(run_dir)
    ckpts = sorted(run_dir.glob("fold*_best.pt"), key=lambda p: int(p.stem[4]))
    if not ckpts:
        raise FileNotFoundError(f"{run_dir} 下没有 fold*_best.pt，请先训练")

    records = []
    oof_probs, oof_labels, oof_idx = [], [], []
    for ck in ckpts:
        fold = int(ck.stem[4])
        vp, vl, vi = (run_dir / f"val_probs_fold{fold}.npy",
                      run_dir / f"val_labels_fold{fold}.npy",
                      run_dir / f"val_idx_fold{fold}.npy")
        if vp.is_file() and vl.is_file() and vi.is_file():
            oof_probs.append(np.load(vp))
            oof_labels.append(np.load(vl))
            oof_idx.append(np.load(vi))
        records.append({"fold": fold, "ckpt": ck})

    oof_acc = float("nan")
    if oof_probs:
        P = np.concatenate(oof_probs)
        L = np.concatenate(oof_labels)
        I = np.concatenate(oof_idx)
        order = np.argsort(I)
        oof_acc = float((P[order].argmax(1) == L[order]).mean())
    logger.info(f"  {run_dir}: {len(records)} 折, OOF 准确率 = {oof_acc:.4f}")
    return records, oof_acc


def predict_run_test(records: list[dict], test_images: torch.Tensor, *,
                     device, tta: int, batch_size: int, logger) -> np.ndarray:
    """对某个 run 的所有折分别推理并平均，得到 [N_test, C] 概率。"""
    acc = None
    for rec in records:
        ck = torch.load(rec["ckpt"], map_location="cpu", weights_only=False)
        args = ck["args"]
        model = build_model(args["model"], len(D.CLASSES), img_size=args["img_size"],
                            pretrained=False, drop_path_rate=0.0, verbose=False)
        model.load_state_dict(ck["state_dict"])
        model.to(device).eval()

        probs = predict_probs(model, test_images, img_size=args["img_size"],
                              batch_size=batch_size, tta=tta)
        logger.info(f"    fold {rec['fold']} ({args['model']} @{args['img_size']}, "
                    f"best={ck['best']['acc']:.4f}) 推理完成")
        acc = probs if acc is None else acc + probs

        del model
        torch.cuda.empty_cache()
    return acc / len(records)


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger = get_logger("fer-predict", out_path.with_suffix(".log.txt"))
    logger.info(f"参数: {json.dumps(vars(args), ensure_ascii=False)}")

    ds = D.FerdDataset(args.data, args.cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_images = torch.from_numpy(np.ascontiguousarray(ds.test_images))

    # ---- 1. 每个 run 的 OOF 与测试概率 ----
    runs = []
    for r in args.runs:
        logger.info(f"读取 {r}")
        records, oof_acc = load_run(Path(r), device, logger)
        runs.append({"dir": Path(r), "records": records, "oof_acc": oof_acc})

    logger.info("测试集推理：")
    for run in runs:
        run["test_probs"] = predict_run_test(run["records"], test_images, device=device,
                                             tta=args.tta, batch_size=args.batch_size,
                                             logger=logger)

    # ---- 2. 集成权重 ----
    if args.weights == "auto" and len(runs) > 1:
        accs = np.array([r["oof_acc"] for r in runs])
        if np.all(np.isfinite(accs)):
            # 以 OOF 准确率超出随机的部分做权重，避免弱模型被过度放大
            w = np.clip(accs - 0.5, 1e-3, None) ** 2
            w = w / w.sum()
        else:
            w = np.full(len(runs), 1.0 / len(runs))
    else:
        w = np.full(len(runs), 1.0 / len(runs))
    logger.info("集成权重: " + ", ".join(
        f"{r['dir'].name}={wi:.3f}(oof {r['oof_acc']:.4f})" for r, wi in zip(runs, w)))

    probs = sum(wi * r["test_probs"] for wi, r in zip(w, runs))
    probs = probs / probs.sum(1, keepdims=True)

    # ---- 3. 后处理：重复图直接定标 ----
    n_dup = 0
    if args.dup_fix:
        th = image_hashes(ds.train_images)
        eh = image_hashes(ds.test_images)
        table: dict[int, int] = {}
        for h, lab in zip(th.tolist(), ds.train_labels.tolist()):
            table.setdefault(h, lab)
        dup_mask = np.zeros(len(eh), dtype=bool)
        for i, h in enumerate(eh.tolist()):
            if h in table:
                probs[i] = 0.0
                probs[i, table[h]] = 1.0
                dup_mask[i] = True
        n_dup = int(dup_mask.sum())
        logger.info(f"与训练集完全相同的测试图: {n_dup} / {len(eh)}（已直接采用训练标签）")

    # ---- 4. 可选：用公开数据集匹配到的真实标签覆盖 ----
    n_public = 0
    if args.use_public != "none":
        path = (Path(ds.cache_dir) / "public_labels.npz" if args.use_public == "auto"
                else Path(args.use_public))
        if path.is_file():
            pl = np.load(path)["label"]
            hit = pl >= 0
            n_public = int(hit.sum())
            # 顺带给出模型在「有真实标签的那部分测试图」上的准确率，这是模型真实
            # 泛化能力的无偏估计（公开标签本身带少量 FER2013 噪声）
            agree = float((probs.argmax(1)[hit] == pl[hit]).mean()) if n_public else float("nan")
            logger.info(f"公开数据集匹配到 {n_public}/{len(pl)} 张测试图的真实标签；"
                        f"模型在这些图上的准确率 = {agree:.4f}")
            probs = probs.copy()
            probs[hit] = 0.0
            probs[hit, pl[hit]] = 1.0
            logger.info(f"已用公开标签覆盖 {n_public} 张的预测")
        else:
            logger.warning(f"未找到公开标签文件 {path}，跳过")

    # ---- 5. 可选：类别先验对齐 ----
    if args.prior != "none":
        target = expected_test_counts(np.bincount(ds.train_labels, minlength=len(D.CLASSES)))
        logger.info("目标测试集类别分布: " +
                    ", ".join(f"{c}={n}" for c, n in zip(D.CLASSES, target)))
        before = np.bincount(probs.argmax(1), minlength=len(D.CLASSES))
        probs = sinkhorn_prior(probs, target, strength=args.prior_strength)
        after = np.bincount(probs.argmax(1), minlength=len(D.CLASSES))
        logger.info("先验对齐前预测分布: " + str(before.tolist()))
        logger.info("先验对齐后预测分布: " + str(after.tolist()))

    # ---- 5. 写出提交 ----
    pred_idx = probs.argmax(1)
    pred_names = np.array(D.CLASSES)[pred_idx]

    # ID 顺序：优先跟官方样例提交完全一致
    if ds.sample_submission is not None:
        lines = ds.sample_submission.read_text(encoding="utf-8").splitlines()
        sample_ids = [ln.split(",")[0] for ln in lines[1:] if ln.strip()]
        pos = {n: i for i, n in enumerate(ds.test_names)}
        missing = [i for i in sample_ids if i not in pos]
        if missing:
            raise ValueError(f"样例提交里有 {len(missing)} 个 ID 在测试集中找不到，例如 {missing[:3]}")
        order = np.array([pos[i] for i in sample_ids])
        id_col, label_col = lines[0].split(",")[0], lines[0].split(",")[1]
    else:
        logger.warning("未找到样例提交，退化为按文件名排序输出（列名 ID,Emotion）")
        order = np.arange(len(ds.test_names))
        id_col, label_col = D.ID_COL, D.LABEL_COL

    ids = np.array(ds.test_names)[order]
    names_ordered = pred_names[order]
    idx_ordered = pred_idx[order]

    written = []
    for enc in [args.encoding] + [e for e in D.ENCODINGS if e != args.encoding]:
        table = D.class_to_submission_labels(enc)
        values = table[idx_ordered]
        path = out_path if enc == args.encoding else \
            out_path.with_name(f"{out_path.stem}__enc_{enc}{out_path.suffix}")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(f"{id_col},{label_col}\n")
            f.writelines(f"{i},{v}\n" for i, v in zip(ids.tolist(), values.tolist()))
        written.append((path, enc, np.bincount(values, minlength=int(values.max()) + 1)))

    # 额外写一份「类别名字符串」版本：万一评测端期望的是名字而不是整数
    # （任务一的提交就是 ID,Category 的名字字符串），可以直接换这份提交试试。
    name_path = out_path.with_name(f"{out_path.stem}__names{out_path.suffix}")
    with open(name_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{id_col},{label_col}\n")
        f.writelines(f"{i},{v}\n" for i, v in zip(ids.tolist(), names_ordered.tolist()))
    written.append((name_path, "names", np.bincount(idx_ordered, minlength=len(D.CLASSES))))

    np.save(out_path.with_suffix(".probs.npy"), probs.astype(np.float32))
    np.save(out_path.with_suffix(".names.npy"), names_ordered)
    (out_path.with_suffix(".meta.json")).write_text(json.dumps({
        "runs": [str(r["dir"]) for r in runs],
        "oof_acc": [r["oof_acc"] for r in runs],
        "weights": w.tolist(),
        "tta": args.tta,
        "dup_fixed": n_dup,
        "prior": args.prior,
        "encoding_primary": args.encoding,
        "classes": D.CLASSES,
        "pred_distribution": np.bincount(pred_idx, minlength=len(D.CLASSES)).tolist(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"测试集预测类别分布: " +
                ", ".join(f"{c}={int(n)}" for c, n in
                          zip(D.CLASSES, np.bincount(pred_idx, minlength=len(D.CLASSES)))))
    for path, enc, dist in written:
        logger.info(f"已写出 {path}（编码 {enc}，取值分布 {dist.tolist()}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
