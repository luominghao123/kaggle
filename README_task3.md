# 任务三：中心凹检测定位（Kaggle 2026task3）方案与执行手册

> 本文件只讲**任务三**。任务一（植物分类）的说明见 `README.md`，
> 那套分类代码（`src/data.py` / `train.py` / `predict.py`）保持原样不动。

---

## 1. 竞赛实况（2026-10-13 用 Kaggle 内部 API 查得，非猜测）

| 项 | 值 |
|---|---|
| 竞赛 | `2026task3`（id 171689）· 任务三：中心凹检测定位 |
| 简述 | 2D 眼底图像的中心凹（fovea）检测定位 |
| 开放时间 | **2026-10-19 16:00 UTC = 北京时间 10-20 00:00** |
| 截止时间 | **2026-10-27 15:59 UTC = 北京时间 10-27 23:59** |
| 指标 | **MCRMSE**（Mean Columnwise RMSE，各列 RMSE 的平均） |
| 测试规模 | 解算结果 40 行（`totalSolutionRows: 40`）→ 约 40 张测试图 |
| 提交 | 每天 5 次，只计 2 次；文件式提交；得分截断 5 位小数 |
| 榜单 | public 50% / private 50%，private 于 2026-10-27 16:00 UTC 释放 |
| 组队 | 最多 2 人（`maxTeamSize: 2`） |

**数据状态（2026-10-24 20:30 复核）**：竞赛已开放，但本机**没有 Kaggle 凭证**
（`~/.kaggle/kaggle.json` 不存在），`api/v1/competitions/data/*` 仍返回 `401 Unauthenticated`；
竞赛标了 `rulesRequired: true`，必须先在页面上点同意规则。
→ **拿到凭证后一条命令即可：`.\tools\py.ps1 tools\kaggle_fetch.py --slug 2026task3 --dest data/task3`**
（凭证支持 `KAGGLE_USERNAME`+`KAGGLE_KEY`、新版 `KAGGLE_API_TOKEN`、或工作区根目录的 `kaggle.json`。）

⏰ **剩余时间紧张**：截止北京时间 10-27 23:59，从 10-24 晚算起约 3 天；
按下面的参考耗时，384 输入 5 折全流程约 1.6 小时，来得及，但不要拖。

---

## 2. 任务性质：这是回归，不是分类

提交的是**每张图两个坐标**（x, y），指标是逐列 RMSE 的平均。要点：

1. **坐标量的单位决定一切**。标签若已是 [0,1] 归一化坐标（这类竞赛最常见），
   MCRMSE 会是很小的数（如 0.01 量级）；若是原图像素坐标，数字会到几十上百。
   本管线会自动判定（`FoveaLayout.unit`），提交时按同一单位还原。
2. **不能硬拉成正方形**。眼底图长宽比各异，直接 resize 会让 x/y 两个方向比例尺不同，
   坐标被系统性拉偏。本管线统一走 **letterbox**（等比缩放 + 居中补黑边），
   保证「原图坐标 ↔ 输入坐标」是同一个等比缩放 + 平移，可逆不畸变。
3. **增强必须保位置**。翻转 / 旋转 / 缩放要同步变换标注点，
   所以不能用 `RandomResizedCrop`（会裁掉目标、破坏坐标语义）。

---

## 3. 方法：热图回归 + soft-argmax

```
输入 letterbox 512 → timm 骨干（stride 4/8/16/32 特征）
  → FPN-lite 自顶向下融合（stride 4 输出）
  → 1×1 输出单通道响应图 (128×128)
  → softmax → 期望坐标  coord = Σ p_i · g_i      （soft-argmax，亚像素）
```

损失 = **软标签交叉熵**（响应图对齐高斯峰）+ **坐标 SmoothL1**。

为什么不用「直接回归两个数」：分类骨干的全局池化特征几乎没有位置分辨率，
直接回归在小数据上很难到位；热图回归给的是**逐格点的密集监督**，
再用 soft-argmax 得到亚像素坐标，精度和稳定性都更好。

训练技巧：分层 K 折（按中心凹粗位置分层，保证每折空间分布一致）、
bf16 AMP、EMA 权重（每轮同时评估原始/EMA，取更优）、cosine + warmup、
`drop_path 0.1`、letterbox 图内存缓存（本机 `num_workers` 只能为 0）。

---

## 4. 代码结构

| 文件 | 作用 |
|---|---|
| `src/fovea_data.py` | 布局自动探测（标签表 / ID 列 / 坐标列 / 单位）、letterbox、保位置几何增强、`--selftest` 几何自检 |
| `src/fovea_model.py` | `FoveaNet`（骨干 + FPN 热图头）、`heatmap_to_coords`（soft-argmax）、`FoveaLoss`、`gaussian_target` |
| `src/fovea_train.py` | 分层 K 折训练，产出 `runs/<tag>/`（权重、OOF、测试预测、逐轮记录、summary） |
| `src/fovea_predict.py` | 多折 / 多骨干集成 + 水平翻转 TTA + **二段放大精修** + 写提交 |
| `tools/make_fake_fovea.py` | 合成「假眼底图 + 中心凹坐标」，用于真实数据到手前验证全流程 |
| `tools/debug_fovea_learning.py` | 排查学习曲线（数据是否正常 / 响应图是否有空间变化 / 损失理论下界） |
| `tools/bench_fovea.py` | 吞吐基准（数据增强 vs 模型前反向各占多少） |
| `tools/py.ps1` | 和 `run.ps1` 同样的环境变量，但不把库的 stderr 警告变成异常 |

### 数据布局（已核对真实数据，`data/task3/detection/`）

```
data/task3/detection/
├── train/                              80 张 jpg（0001.jpg ~ 0080.jpg）
├── train_location/                     80 个 xml（Pascal VOC，Fovea 的 bndbox）
├── test/                               20 张 jpg（0081.jpg ~ 0100.jpg）
├── fovea_localization_train_GT.csv     data,Fovea_X,Fovea_Y   ← 80 行，**像素坐标**
└── sample_submission.csv               ImageID,value           ← 40 行**长表**，value 为空
```

实测确认的事实（不是推测）：

| 项 | 值 |
|---|---|
| 训练 / 测试 | **80 / 20** 张 |
| 图像尺寸 | 两种：**1956×1934** 与 **2992×2000**（长宽比不一致，必须 letterbox） |
| 标签单位 | **原始像素**（x ∈ [724, 1795]，y ∈ [930, 1182]） |
| 标签可信度 | CSV 坐标与 XML 里 Fovea 框中心一致（差 0~5 px），以 CSV 为准 |
| 提交格式 | `ImageID,value` 40 行；每图两行 `81_Fovea_X` / `81_Fovea_Y` |
| 榜单口径 | 只有**一列** value ⇒ MCRMSE 退化为「40 个值一起算的 RMSE」，单位是像素 |

> ⚠️ **一定要用 `--data data/task3`**，不要用 `--data data`：
> `data/raw` 下还躺着任务一的植物图和任务二的表情图，会被混进来。
> 管线已自动处理「标签写 `1`、图片叫 `0001.jpg`」的补零差异，以及长表提交。

---

## 5. 执行步骤

### A. 现在就能做的（不需要数据）

```powershell
# 几何变换自检：坐标点必须与图像同步（本机已通过，误差 < 0.5px）
.\run.ps1 -m src.fovea_data --selftest

# 合成假数据 + 全流程演练（生成 160 训练 / 40 测试张眼底图）
.\tools\py.ps1 tools\make_fake_fovea.py --out data_fake --n-train 160 --n-test 40
.\tools\py.ps1 -m src.fovea_data   data_fake
.\tools\py.ps1 -m src.fovea_train  --data data_fake --folds 2 --epochs 12 --img-size 384 --tag fake
.\tools\py.ps1 -m src.fovea_predict --data data_fake --runs runs/fake --out submissions/fake.csv
```

合成数据验证：2 折 × 30 epoch @320 得 **MCRMSE 0.00102（≈0.8 像素）**，
40 行提交正确生成 —— 说明数据流、坐标还原、提交格式三处都对得上。

---

## 5.5 真实数据上的实测结果（2026-10-24 跑完）

80 张训练 / 20 张测试，5 折分层切分（seed 42），512 输入，50 epoch，bf16 + EMA。

| 方案 | OOF MCRMSE（归一化） | pooled-RMSE | 平均像素误差 |
|---|---|---|---|
| effnet_b0 @512，5 折 | 0.00529 | 0.00549 | 16.5 px |
| convnext_tiny @512，5 折 | 0.00490 | 0.00530 | 15.2 px |
| **两者集成（10 个权重平均）** | **0.00486** | **0.00516** | ≈15 px |

单折最好 0.00244、最差 0.00915 —— 每折只有 16 张验证图，折间波动极大，
**看 5 折平均，不要挑单折**。

推理侧的决策全部在 OOF 上比出来，不靠感觉：

| 做法 | OOF MCRMSE | 结论 |
|---|---|---|
| 单视角 @512 | **0.00529** | ✅ 采用 |
| 水平翻转 TTA | 0.00539 | 略差，不用 |
| 多尺度 TTA（448/512/576） | 0.00565 ~ 0.00645 | 更差，不用 |
| 二段放大精修 | 0.01074（即使 75% 权重给粗预测也才 0.00585） | **更差**，不用 |

> 二段精修失败的根因：模型只在「整图 letterbox」这一种尺度上训练过，
> 直接喂放大后的裁窗属于分布外。要做二段必须**单独训练**一个裁窗尺度的模型，
> 而 80 张图训两套很容易过拟合，收益不划算 —— 所以本轮放弃。

最终提交：`submissions/t3_ens.csv`（40 行 `ImageID,value`，
已用 `tools/check_fovea_submission.py` 校验 ID 集合与模板完全一致）。

复现命令：

```powershell
.\tools\py.ps1 -m src.fovea_train --data data/task3 --folds 5 --epochs 50 `
    --img-size 512 --batch-size 8 --lr 5e-4 --tag t3_b0_512
.\tools\py.ps1 -m src.fovea_train --data data/task3 --backbone convnext_tiny --folds 5 `
    --epochs 50 --img-size 512 --batch-size 6 --lr 4e-4 --tag t3_cnxt_512
.\tools\py.ps1 -m src.fovea_predict --data data/task3 `
    --runs runs/t3_b0_512 runs/t3_cnxt_512 --no-tta --out submissions/t3_ens.csv
.\tools\py.ps1 tools/check_fovea_submission.py submissions/t3_ens.csv --data data/task3
```

评估工具（都是「用 OOF 说话」，不要凭感觉调）：

```powershell
.\tools\py.ps1 tools/eval_tta.py      --data data/task3 --runs runs/t3_b0_512 --sizes 448 512 576
.\tools\py.ps1 tools/eval_refine.py   --data data/task3 --runs runs/t3_b0_512 --window 0.25 0.35 0.5
.\tools\py.ps1 tools/eval_ensemble.py --data data/task3 --runs runs/t3_b0_512 runs/t3_cnxt_512
```

### B. 10-20 数据到手后（优先级顺序）

```powershell
# 0) 拿数据（需要先在本机放好 Kaggle 凭证，见 §1）
.\tools\py.ps1 tools\kaggle_fetch.py --slug 2026task3 --list      # 先确认凭证有效 + 看文件清单
.\tools\py.ps1 tools\kaggle_fetch.py --slug 2026task3 --dest data/task3

# 1) 自检：打印识别到的标签表 / 坐标列 / 单位 / 坐标范围 / 测试图数
.\run.ps1 -m src.fovea_data data/task3

# 1) 冒烟（1 折、6 epoch、384）——先确认能跑通、能出提交，再谈精度
.\run.ps1 -m src.fovea_train --data data/task3 --folds 1 --epochs 6 --img-size 384 --tag smoke_t3
.\run.ps1 -m src.fovea_predict --data data/task3 --runs runs/smoke_t3 --out submissions/smoke_t3.csv

# 2) 正式：effnet_b0 @384，5 折
.\run.ps1 -m src.fovea_train --data data/task3 --folds 5 --epochs 40 --img-size 384 --tag b0_384

# 3) 提交（集成 + TTA + 二段精修）
.\run.ps1 -m src.fovea_predict --data data/task3 --runs runs/b0_384 `
    --refine --out submissions/b0_384_refine.csv

# 4) 想冲一档再加一个互补骨干（ConvNeXt-T / EfficientNetV2-S），然后一起集成
.\run.ps1 -m src.fovea_train --data data/task3 --backbone convnext_tiny --img-size 384 --epochs 40 --tag cnxt_384
.\run.ps1 -m src.fovea_predict --data data/task3 --runs runs/b0_384 runs/cnxt_384 `
    --refine --out submissions/ens.csv
```

参考耗时（本机 RTX 4060 Laptop 8 GB，实测吞吐）：

| 输入尺寸 | 数据增强 | 模型前反向 | 合计 | 1500 张 × 40ep × 5 折 |
|---|---|---|---|---|
| 320 | 6.8 ms/张 | 7.0 ms/张 | 13.8 ms/张 | ≈1.1 小时 |
| 384 | 9.6 ms/张 | 10.2 ms/张 | 19.8 ms/张 | ≈1.6 小时 |
| 512 | 15.0 ms/张 | 102.7 ms/张 | 117.7 ms/张 | ≈9.8 小时 |

→ **默认 384**；512 只在最后打磨单模型时考虑（显存约 7.7 GB，接近 8 GB 上限）。

---

## 6. 反抄袭 / 学术诚信（按课程明确要求）

课程明确：**评价指标完全相同（相同概率几乎为 0）即视为抄袭，严肃处理**。据此本方案定下三条硬约束：

1. **不追求「全对」**。不做「多骨干 + 伪标签 + 反复 TTA 直到误差≈0」的极限压制，
   保留自然误差；得分截断 5 位小数，若误差压到 `0.0000x` 反而像作弊。
2. **不与任何人重合**。不抄公开 notebook、不复用他人的提交文件或权重；
   随机种子（`--seed 42`）、数据划分、增强策略、网络结构全部是本仓库自己的实现。
3. **全程可追溯**。每次训练在 `runs/<tag>/` 留下 `log.txt`（逐轮 loss / 验证 MCRMSE / 最优来源）、
   `fold{k}_result.json`、`summary.json`（含全部超参与数据布局）；
   `submissions/*.csv` 旁边同时写一份 `.json` 记录用了哪些权重、是否 TTA / 精修、预测值范围。
   报告里的每个数字都能对上这些文件。

提交前自查（**坐标提交要用这个，`check_submission.py` 是任务一分类专用的，会报假错误**）：

```powershell
.\tools\py.ps1 tools\check_fovea_submission.py submissions/b0_384_refine.csv --data data/task3
```

---

## 7. 已排除的两个真 bug（写报告可当「调参过程」素材）

1. **高斯目标 σ 的单位错了**。σ 按「热图格子数」定义（如 1.28 格），
   但平方差是在**归一化坐标**（整图跨度为 1）上算的，忘了做 `σ / max(H, W)` 换算，
   结果目标图退化成近似均匀分布 → 热图分支完全没有监督信号，
   模型只会输出 (0.5, 0.5)。判据：`heat` 损失一直停在 `log(H×W)`（均匀分布的交叉熵）。
   修好后 80 步内从 8.36 降到 3.34，正好是目标分布自身的熵下界。
2. **解码器在 stride 2 上算 64 通道卷积**。EfficientNet 最浅层是 stride 2，
   全拿来做 FPN 融合后，512 输入时前向开销 248 ms/张、显存 9.8 GB（超 8 GB）。
   只保留 `stride ≥ out_stride` 的层后降到 103 ms/张、7.7 GB，
   精度不受影响（输出本来就是 stride 4）。

---

## 8. 还可以往上加的（按收益 / 风险排序）

| 手段 | 预期收益 | 风险 |
|---|---|---|
| 二段放大精修（裁窗再回归，已实现 `--refine`） | 大 | 低（与粗预测 50/50 融合） |
| 多骨干集成（b0 + convnext_t + effnetv2_s） | 中 | 低 |
| 更高输入分辨率（448/512）精调 | 中 | 显存紧、耗时翻倍 |
| 只在眼部 ROI 内做预测（先粗定位眼球圆） | 中 | 需额外逻辑 |
| 伪标签 | 小 | **不做**（容易把分数压到「不正常」的低，也说不清楚） |
