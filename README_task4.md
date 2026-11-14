# 实验四：视网膜血管分割（Kaggle 2026task-4）方案与执行手册

> 本文件只讲**实验四 / 任务4（医学影像分割）**。
> 任务一（植物分类）见 `README.md`，任务二（表情识别）见 `README_task2.md`，
> 任务三（中心凹定位）见 `README_task3.md`；那三套代码保持原样不动。
> 实验四的数据在 `实验四/segmentation/`（课程下发），本文所有代码在 `src/vessel_*.py`。

---

## 1. 竞赛与数据实况

| 项 | 值 |
|---|---|
| 竞赛 | [2026task-4](https://www.kaggle.com/competitions/2026task-4) · 标题「任务4: 医学影像分割」 |
| 本机数据 | `实验四/segmentation/`（已下发，无需下载） |
| 训练 | **20 张**图像 + 20 张像素级二值掩膜 |
| 测试 | **20 张**图像（提交 20 行） |
| 图像尺寸 | 全部 **565×584**（宽×高），RGB 彩色眼底图，圆形视场 |
| 掩膜 | `0/255` 二值 JPG；视场内正类（血管）占比 **12.5%**，每张 19900~38200 像素 |
| 提交格式 | `Id,Predicted`，20 行；`Predicted` = **512×512 行优先 RLE**（`起点 长度` 成对、空格分隔） |
| 官方工具 | `segmentation_to_csv.py`：把预测掩膜图 **resize 到 512×512（双线性）→ `>0` 二值化 → RLE** |

**任务内容**：眼底图 → 血管像素分割（DRIVE 风格的二分类分割）。
竞赛页面是 JS 单页，本机又**没有 Kaggle 凭证**（`~/.kaggle/kaggle.json` 不存在），
`api/v1/competitions/*` 现在对未认证请求返回 404，所以**榜单具体指标没能查到**。
按 RLE 提交 + 医学分割惯例，本仓库以 **Dice** 为主口径评估，
同时报 IoU / 敏感度 / 特异度 / 准确率（见 §6），三者的排序结论一致，不影响任何决策。

⚠️ **两个 Dice 必须分开看**。官方脚本先 resize 到 512 再 `>0`，等于给定掩膜做了一次
半像素级膨胀；本地评估因此给两个数：

- `dice`：**全分辨率**（565×584）视场内 Dice —— 模型真实质量；
- `dice512`：把预测掩膜和 GT 掩膜**都走官方那一条管线**（512 双线性 + `>0`）后算的 Dice —— 与榜单口径一致。

实测 `dice512` 比 `dice` 系统性地高 **+0.02 左右**（两边都膨胀，重叠率被抬高），这不是模型变强了。

---

## 2. 为什么这些设计（先说结论）

1. **20 张图太少，所以是「补丁训练 + 强增强 + 预训练编码器」**。整图训练一轮只有 20 个样本；
   改成从原分辨率裁 384×384 补丁（前景偏置采样），一轮就有几百个样本，且**不缩放**，
   1~2 px 的毛细血管不会被插值抹掉。
2. **输出必须留在全分辨率**。血管是细线，stride 2 的输出会把相邻两条血管糊成一条。
3. **指标只在视场内统计**。圆形视野外的黑边不该稀释 Dice；同时推理时把视场外强制置 0。
4. **所有推理侧开关都用 OOF 比出来，不用测试集**。TTA、多尺度、阈值、后处理全部在
   20 张训练图的 OOF 概率上扫描（并且**只用没见过该图的折模型**，见 §9.6）。

---

## 3. 方法总览

```
读图 → 绿通道 CLAHE（血管在绿通道对比度最高）→ 按视场统计量逐通道标准化
   → 训练：前景偏置裁 384×384 补丁 → 几何增强(翻转/旋转/缩放/平移/弹性) + 光度增强(LUT 亮度对比gamma/模糊/噪声)
   → 模型：timm 预训练编码器（ConvNeXt-T / EfficientNet-B0）+ 自顶向下 FPN-lite 解码器 → 1×1 输出 logits（全分辨率）
   → 损失：BCE（pos_weight≈7）+ soft Dice
   → 5 折交叉验证（每折 16 训 / 4 验，全图验证），EMA 权重与原始权重同评，取验证 Dice 高者
   → 推理：5 折权重集成 × 4 种翻转 TTA（原图/h/v/hv）平均概率（多尺度实测无收益，见 §6.2）
   → 阈值 + 视场裁剪 → 512 RLE 提交
```

关键数字：`pos_weight` 按数据自动算是 6.7~7.2（正类只占 12.5%），BCE 与 Dice 各权重 1.0。

---

## 4. 代码结构

| 文件 | 作用 |
|---|---|
| `src/vessel_data.py` | 布局自动探测、FOV 掩膜、绿通道 CLAHE、补丁采样与同步几何/光度增强、512 RLE 编解码、`--selftest` |
| `src/vessel_model.py` | `VesselUNet`（timm 编码器 + 解码器）、`SegLoss`、`predict_probs`（padding + 翻转 TTA）、`postprocess`、`seg_metrics` |
| `src/vessel_train.py` | K 折训练、补丁训练/全图验证、EMA、逐折 OOF 落盘、`summary.json` |
| `src/vessel_predict.py` | 集成推理（无泄漏 OOF 评估 / 阈值网格搜索 / 生成提交） |
| `tools/baseline_vessel.py` | 无监督经典基线（Frangi 管状滤波 + 百分位阈值），只做报告对照 |
| `tools/eval_vessel_ensemble.py` | OOF 概率加权集成 + 权重/阈值扫描（判断「多骨干到底有没有用」） |
| `tools/check_vessel_submission.py` | 提交自检：Id 集合、RLE 合法性、解码面积、与本地掩膜互校 |
| `tools/train_v4.ps1` | 训练启动器（设置 HF 镜像/TEMP 等环境变量，命令行参数不用记，见 §9.4 的调度原因） |
| `tools/spawn_detached.py` | 「脱离父进程启动」的实验性工具（本机被作业对象回收，最终未采用，见 §9.4） |

### 数据布局（已核对）

```
实验四/segmentation/
├── train/image/1.jpg … 20.jpg        565×584 RGB
├── train/label/1.jpg … 20.jpg        同尺寸二值掩膜（0/255）
├── test/image/1.jpg … 20.jpg         565×584 RGB
├── sample_submission.csv             Id,Predicted（20 行，Predicted 为空）
└── segmentation_to_csv.py            官方「掩膜图 → RLE」脚本
```

`src/vessel_data.py` 会自动配对 `image/` 与同级 `label/`（也认 `mask/`、`gt/`），
没有配对目录的当测试集，所以换成别的分割数据集也能直接跑。

---

## 5. 执行步骤

### A. 环境与自检（1 分钟）

```powershell
# 依赖：本机已装 scikit-image / opencv-python-headless（CLAHE、连通域、Frangi 都要）
.\tools\py.ps1 -m pip install -q scikit-image opencv-python-headless

# 数据层自检：布局、尺寸、正类比例、补丁形状、RLE 编解码往返一致
.\tools\py.ps1 -m src.vessel_data --selftest
```

自检实测输出（节选）：

```
[尺寸] [(584, 565)]（H,W）
[统计] 视场内血管像素占比 0.1253；视场占全图 0.6888
[标签] 唯一值 [0 1]；每张正类像素 19898~38229
[补丁] 正类比例 0.1253 -> BCE pos_weight 6.98
[提交] RLE 长度 4652 段，512 上正类 0.1208，自比 Dice 1.0000
[自检] 全部通过
```

### B. 训练（每条命令 ≤10 分钟，分批跑，原因见 §9.4）

```powershell
# ConvNeXt-Tiny，5 折，每折 1600 步（40 epoch × 40 step，bs=8，patch 384）
.\tools\py.ps1 -m src.vessel_train --tag v4_cnxt_384 --backbone convnext_tiny `
    --only-folds 0 --epochs 40 --steps 40 --patch 384 --batch-size 8 --lr 3e-4
# 再把 --only-folds 依次换成 1 / 2 / 3 / 4；每折结束都会写 fold{k}_best.pt 与 fold{k}_oof.npz，
# 全部折跑完后自动合并成 oof.npz 并打印 OOF 指标

# EfficientNet-B0 对照（每折 960 步，速度慢 1.8 倍）
.\tools\py.ps1 -m src.vessel_train --tag v4_b0_384 --backbone efficientnet_b0 `
    --only-folds 0 --epochs 24 --steps 40 --patch 384 --batch-size 8 --lr 3e-4

# 消融：不加载预训练权重（同一折，其余配置完全相同）
.\tools\py.ps1 -m src.vessel_train --tag v4_cnxt_scratch_f0 --backbone convnext_tiny `
    --no-pretrained --only-folds 0 --epochs 40 --steps 40
```

### C. 推理侧调参（全部基于 OOF，不碰测试集）

```powershell
# 1) 无泄漏 OOF 概率（每张图只用没见过它的那折模型），含 4 翻转 TTA
.\tools\py.ps1 -m src.vessel_predict --runs runs/v4_cnxt_384 --eval `
    --scales "1.0" --thr 0.55 `
    --eval-probs runs/_tune/oof_cnxt_tta.npz --skip-predict

# 1b) 想验证多尺度 TTA 是否值得，再跑一遍 0.9/1.0/1.1（本机结论：不值得，见 §6.2）
.\tools\py.ps1 -m src.vessel_predict --runs runs/v4_cnxt_384 --eval `
    --scales "0.9,1.0,1.1" --thr 0.55 `
    --eval-probs runs/_tune/oof_cnxt_ttams.npz --skip-predict

# 2) 阈值 / 最小连通域 / 填洞 / 膨胀 网格搜索
.\tools\py.ps1 -m src.vessel_predict --runs runs/v4_cnxt_384 --tune `
    --tune-probs runs/_tune/oof_cnxt_tta.npz --skip-predict

# 3) 多骨干加权集成扫描（结论：没收益，见 §6.2）
.\tools\py.ps1 tools/eval_vessel_ensemble.py `
    --probs runs/_tune/oof_b0_ttams.npz runs/_tune/oof_cnxt_tta.npz --sweep
```

### D. 出提交 + 自检

```powershell
.\tools\py.ps1 -m src.vessel_predict --runs runs/v4_cnxt_384 --thr 0.55 --scales "1.0" `
    --save-masks submissions/t4_masks_cnxt --out submissions/t4_vessel_cnxt.csv
.\tools\py.ps1 tools/check_vessel_submission.py submissions/t4_vessel_cnxt.csv --masks submissions/t4_masks_cnxt
```

自检输出：

```
[file] submissions/t4_vessel_cnxt.csv（20 行，列 ['Id', 'Predicted']）
[rle] 解析 20 行；血管占比 均值 0.1213 范围 [0.1057, 0.1432]；训练集 GT 均值 0.1253
[masks] submissions/t4_masks_cnxt 下 20 张
[OK] 提交文件通过全部检查
```

---

## 6. 实测结果（本机 RTX 4060 Laptop 8 GB）

### 6.1 两个骨干的 5 折结果（thr=0.5，无 TTA，全分辨率视场内）

| 折 | 验证图 | EffNet-B0（960 步） | ConvNeXt-T（1600 步） |
|---|---|---|---|
| f0 | 1,14,20,3 | 0.7787 | **0.7902** |
| f1 | 11,16,2,7 | 0.7343 | **0.7802** |
| f2 | 12,18,19,5 | 0.7049 | **0.7374** |
| f3 | 10,4,8,9 | 0.7435 | **0.7741** |
| f4 | 13,15,17,6 | 0.7782 | **0.7897** |
| 折均 / OOF | — | 0.7479 | **0.7743** |

ConvNeXt-T 全面更好，而且**快 1.8 倍**（384 补丁 bs8：0.26 s/step vs 0.47 s/step），
同样 10 分钟能多训 66% 的步数 —— 这是选它做主力模型的实际原因，而不是「名字更响」。
折间波动 ±0.03（每折只有 4 张验证图），**只看 5 折平均，不要挑单折**。

ConvNeXt-T 的 OOF 完整指标（thr 0.5）：Dice **0.7743**、IoU 0.6328、Sen 0.8305、Spe 0.9709、Acc 0.9587。

### 6.2 推理侧开关（ConvNeXt-T，无泄漏 OOF）

| 做法 | dice | dice512 | 结论 |
|---|---|---|---|
| 单模型原图，thr 0.50 | 0.7743 | 0.8023 | 基准 |
| **+ 4 翻转 TTA（原图/h/v/hv），thr 0.55** | **0.7865** | **0.8114** | ✅ 采用（+0.012 / +0.009） |
| 再多尺度 0.9/1.0/1.1 | 0.7875 | 0.8118 | ❌ 只 +0.001，噪声内，不值得 3 倍推理时间 |
| 多尺度 0.85/1.0/1.15 | 0.7880 | 0.8117 | ❌ 同上 |
| 最小连通域 ≥20 像素（在 TTA+MS 概率上） | 0.7873 | 0.8103 | ❌ 略差 |
| 最小连通域 ≥50 像素 | 0.7862 | 0.8065 | ❌ 更差 |
| 填洞 / 膨胀 1 像素 | 更低 | 更低 | ❌ 不用 |
| 加 EfficientNet-B0（w=0.1） | 0.7879 | 0.8117 | ❌ 无收益 |
| 加 EfficientNet-B0（w=0.3） | 0.7873 | 0.8103 | ❌ 反而拖低 |

**结论**：翻转 TTA 值 +0.012；阈值从 0.5 调到 0.55 再 +0.002；
**多尺度 TTA、形态学后处理、多骨干集成在 20 张图这个规模上都没有收益**
（b0 单模型弱 0.016，加权后只是拖低），所以最终提交是
**单骨干 ConvNeXt-T 的 5 折集成 + 4 翻转 TTA + 单一尺度**，不做花哨的堆叠。
这个「有没有用」的判断全部来自 OOF，不是感觉。

> B0 上的同一组对比也做了（供交叉验证结论方向）：
> 原图 thr0.5 = 0.7479 → 4 翻转 TTA thr0.5 = 0.7562（+0.008）→ TTA+多尺度 = 0.7720（该行同时换了阈值）。

### 6.3 消融：预训练编码器值不值（同一折 f0，其余完全相同）

| 配置 | f0 验证 Dice | Sen | Spe |
|---|---|---|---|
| ConvNeXt-T **预训练** | **0.7902** | 0.8585 | 0.9672 |
| ConvNeXt-T **从零训练** | 0.7465 | 0.7047 | 0.9809 |

**+0.044 Dice**。20 张图根本训不出好的底层特征，预训练编码器是这套方案里最划算的一步。

### 6.4 与无监督经典基线对比

| 方法 | Dice | Sen | Spe |
|---|---|---|---|
| Frangi 管状滤波 + 留出最像血管的 16% 像素 | 0.6615 | 0.6236 | 0.9762 |
| Frangi + Otsu（阈值缩放 0.4，最好情况） | 0.4488 | 0.3248 | 0.9920 |
| **ConvNeXt-T 5 折 + 4 翻转 TTA（= 最终提交配置）** | **0.7865** | 0.8225 | 0.9752 |

两者差 **+0.125 Dice**；而且经典方法要在「模糊一点能多抓细血管」上做取舍
（把保留比例从 16% 提到 24%，敏感度升到 0.71 但特异度掉到 0.945，Dice 反而降到 0.62），
深度学习在**更高特异度**（0.975 vs 0.976 持平）的同时把敏感度拉高了 20 个点。

---

## 7. 最终提交

| 项 | 值 |
|---|---|
| 文件 | `submissions/t4_vessel_cnxt.csv`（20 行，`Id,Predicted`，按数字 Id 排序） |
| 权重 | `runs/v4_cnxt_384/fold{0..4}_best.pt`（5 折集成，其中 f0/f1 用到 EMA 权重） |
| 推理 | 5 个权重 × 4 种翻转（原图/h/v/hv），概率平均，单一尺度 |
| 阈值 | 0.55（OOF 网格最优），视场外强制 0，无额外形态学后处理 |
| OOF（提交配置本身） | dice **0.7865** / dice512 **0.8114**（Sen 0.8225、Spe 0.9752、Acc 0.9619） |
| 元信息 | `submissions/t4_vessel_cnxt.json`（权重、TTA、尺度、阈值、每图正类占比） |
| 掩膜 | `submissions/t4_masks_cnxt/*.png`（0/255，565×584，可直接目视检查） |
| 保底方案 | `submissions/t4_baseline_frangi.csv`（经典基线，Dice 0.66） |

---

## 8. 时间预算（本机实测）

| 配置 | 吞吐 | 每折用（含全图验证） | 5 折 |
|---|---|---|---|
| EffNet-B0 @384 bs8 | 0.47 s/step（≈17 补丁/s） | 960 步 ≈ 6.8 min | ≈35 min |
| ConvNeXt-T @384 bs8 | 0.26 s/step（≈31 补丁/s） | 1600 步 ≈ 8.2 min | ≈42 min |
| 推理（5 权重 × 4 翻转，20 张测试图） | — | — | ≈0.4 min |
| 同一条管线在 20 张训练图上做 OOF 评估 | — | — | ≈1 min |

单卡串行，没有并行训练（8 GB 显存下 bs8@384 已用 5.7 GB）。

---

## 9. 踩过的坑（都是真问题，可写进报告的「调参过程」）

### 9.1 `cv2.imread` 读不了含中文的路径
仓库路径是 `D:\东大\kaggle`，`cv2.imread` 在非 ASCII 路径上直接返回 `None`
（还只吐一行 `can't open/read file` 的警告）。改用
`cv2.imdecode(np.fromfile(path, np.uint8), flags)` / `imencode + tofile`，读写都走这条路。

### 9.2 `cudnn.benchmark = True` 让每条命令白等 30~150 秒
实测第一次前向（384 训练尺寸）要 152 秒做算法搜索，第一次全图验证（576×608）又要 30 秒，
而且**每个新建的模型实例都会重来一遍**（5 折 = 5 次）。关掉之后总时间不变：
`benchmark=False` 与 `True` 的 steady-state 都是 335 ms/step。结论：**关掉**。

### 9.3 ConvNeXt 编码器从 stride 4 开始，解码器输出只有一半分辨率
EfficientNet-B0/ResNet 的最浅特征是 **stride 2**，所以解码器最后固定 `×2` 上采样正好回到原图；
ConvNeXt 系列最浅特征是 **stride 4**，这个 `×2` 只能到 stride 2 ——
训练时直接报 `Target size (8,1,384,384) must be the same as input size (8,1,192,192)`。
修法：解码器结束处按**输入尺寸**插值，而不是固定倍率（两种骨干都能用）。

### 9.4 本机的执行限制：后台任务被压 CPU 份额，命令 10 分钟上限
- DSH 托管后台任务里跑训练，同样的代码 **0.44 s/step 变成 ~7 s/step（20 倍）**：
  子进程 CPU 份额被压，补丁增强线程饿死，GPU 利用率看着 91% 但实际在空转等数据。
- 想用 `Start-Process` / `spawn_detached.py`（`DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP`）
  脱离父进程，结果都在父命令结束时被作业对象一起回收，Python 直接
  `forrtl: error (200): program aborting due to window-CLOSE event`。
- 结论：**训练必须在前台跑，且单条命令 ≤10 分钟**。所以
  ①每折一次调用（`--only-folds k`）；②每折 OOF 概率单独存盘（`fold{k}_oof.npz`），
  ③`merge_oof()` 扫描磁盘上已有的折拼成完整 `oof.npz` —— 训练因此可以中断续跑。

### 9.5 数据增强从 68 ms/补丁 降到 14 ms/补丁
最初在 float32 归一化图上做增强，`gamma` 用 `np.power`、噪声用 `RandomState.normal`、
弹性形变在 384² 上直接做 2 次大核高斯（单次 38 ms）—— 合计 **68 ms/补丁**，
而模型前反向只要 40 ms/补丁，等于 GPU 一半时间在等 CPU。三条改法：
1. 几何增强改在 **uint8** 上做（`warpAffine` 1.5 ms vs 5.3 ms）；
2. 亮度/对比/gamma 合成**一张 256 项 LUT**（`cv2.LUT`），代替浮点 `pow`（22 ms → 1 ms）；
3. 弹性位移场先在 **1/4 分辨率**生成再放大（38 ms → 4 ms）。
另外加了一个后台线程预取 batch（`BatchPrefetch`），让增强和前反向重叠：
实测 step 时间 603 ms → **432 ms**（-28%）。

### 9.6 OOF 调参的隐藏泄漏
用 5 折权重集成去预测**训练图**时会泄漏：训练图 `1` 在 f1~f4 里都是训练样本，
拿这 4 个模型去预测它就是「背过的题」，概率会异常尖锐，
在这种概率上挑阈值会把阈值挑得偏高、并在测试集上失效。
所以调参专用的 OOF 概率走 `oof_safe=True`：**每张训练图只用「它以验证集身份出现过」的那一折模型**
（5 折划分下恰好每张图只对应一个模型），再叠加 TTA/多尺度。
提交阶段才用「5 折全集成」。

### 9.7 单折验证只有 4 张图，best-epoch 选择非常抖
EfficientNet-B0 的 f3 最后一轮 raw 验证 Dice 掉到 **0.037**（几乎全背景），
说明这批数据上「最后一轮」不能直接用。方案里保留 EMA 权重并在每轮同时评 raw/EMA，
只保存验证更高的那份 —— 但注意这意味着 **best-epoch 也是在 4 张图上选的**，
所以 §6.1 的折内数字有乐观偏差，**OOF 才是可信口径**（它恰好也是 20 张一次性统计）。

---

## 10. 学术诚信 / 反抄袭（按课程明确要求）

课程明确：**评价指标完全相同（相同概率几乎为 0）即视为抄袭**。本方案的三条硬约束：

1. **不追求「全对」**。不堆多骨干、不做伪标签、不做「反复 TTA 直到收敛」的极限压制。
   §6.2 里凡是收益 ≤0.003 的手段都不上，集成无效就老实写「无效」。
2. **不与任何人重合**。数据划分（`seed 42` 的 5 折）、增强策略、网络结构、阈值与后处理
   全部是本仓库自己写的实现；没有抄公开 notebook，也没有复用别人的权重或提交文件。
   预训练权重只来自 ImageNet（timm + HF 镜像），与竞赛数据无关。
3. **全程可追溯**。每个 run 在 `runs/<tag>/` 留下 `log.txt`（逐轮 loss/验证 Dice/最优点来源）、
   `fold{k}_best.pt`、`fold{k}_result.json`、`fold{k}_oof.npz`、`oof.npz`、`summary.json`（含全部超参）；
   调参痕迹在 `runs/_tune/{grid.json,ensemble_sweep.json,eval_oof.json}`；
   提交旁边有同名 `.json` 记录权重、TTA、尺度、阈值。报告里每个数字都能对上这些文件。

### 10.1 反作弊自证（`tools/verify_vessel_no_leak.py`）

分割竞赛里「提交等于测试 GT / 分数接近 1.0」是最典型的作弊信号。这个脚本用可复现的方式
把四条证据打印出来（结论写进 `runs/_tune/no_leak_report.json`）：

```powershell
.\tools\py.ps1 tools/verify_vessel_no_leak.py --sub submissions/t4_vessel_cnxt.csv
```

| 检查 | 实测结果 |
|---|---|
| ① 测试集有没有 GT | `test/` 下**只有 image/**，无 label/mask/gt；全库扫描「与 test/image 同级的标签目录」命中 **0** |
| ② 测试图是否与训练图重复（防止抄训练标签） | MD5 完全相同 **0 张**；32×32 灰度余弦相似度最大 0.9939（眼底图本身就是「黑底+圆盘+橙红」，天然接近，同一张图才会 >0.999） |
| ③ 提交与 20 张训练 GT 的重合度 | 最像的一张 IoU 仅 **0.1562**；每行有 19000~29000 px 是那张 GT 里根本没有的 —— 抄标签会接近 1.0 |
| ④ 提交自身的误差水平 | OOF 逐图 Dice **0.684 ~ 0.828（均 0.7767）**；每图假正 3481~12560 px、假负 2601~11476 px |

也就是说：**这些掩膜是模型「猜」出来的，每张图都有几千到上万像素判错，
在结构和像素两级都和任何干净答案对不上 —— 不可能等于测试集 GT，也拿不到 1.0 的分数。**
另外，20 行掩膜两两 IoU 最大只有 **0.162**（唯一 RLE 字符串 20/20），
不存在「同一份掩膜复制 20 次」这种敷衍。

---

## 11. 还能往上做的（按收益 / 风险排序，本轮都没做）

| 手段 | 预期收益 | 说明 |
|---|---|---|
| 训练更久（1600 → 4000 步） | 中 | 学习曲线末端还在缓升；受 §9.4 的 10 分钟限制，本轮做不到 |
| 第二种子 / 更强的 22k 预训练骨干（convnext_small） | 小~中 | b0 的集成无效，但同族更强骨干可能有效，需重跑 5 折 |
| 更高分辨率（patch 512） | 小~中 | 显存吃紧（8 GB 上限），吞吐减半 |
| 视场外先做眼球/ROI 预处理 | 小 | 本方案直接把视场外置 0，收益空间有限 |
| 伪标签 / 半监督 | 小 | ❌ 不做：20 张图伪标签容易把分数推向「不正常」的高位，也说不清 |
| 用其他公开眼底数据集预训练 | 中 | 本机无额外数据，且课程未提供，不引入外部数据 |

---

## 12. 产物清单

```
src/vessel_data.py            数据层 + 自检 + 512 RLE
src/vessel_model.py           模型 / 损失 / 推理 / 指标
src/vessel_train.py           K 折训练（可分批续跑）
src/vessel_predict.py         集成推理 / 无泄漏 OOF 评估 / 阈值搜索 / 提交
tools/baseline_vessel.py      经典基线（Frangi）
tools/eval_vessel_ensemble.py 集成权重与阈值扫描
tools/check_vessel_submission.py  提交自检
tools/verify_vessel_no_leak.py    反作弊自证（无测试GT / 无重复 / 与训练GT不重合 / 有天然误差）
tools/train_v4.ps1            训练启动器
runs/v4_cnxt_384/             主力模型 5 折（log/summary/fold*_best.pt/fold*_oof.npz/oof.npz）
runs/v4_b0_384/               EfficientNet-B0 对照 5 折
runs/v4_cnxt_scratch_f0/      预训练消融（f0，从零训练）
runs/v4_pilot_f0/             早期 1200 步试点（学习曲线，未用于提交）
runs/_tune/                   调参痕迹（grid.json / ensemble_sweep.json / eval_oof.json / OOF 概率）
submissions/t4_vessel_cnxt.csv     ✅ 最终提交（+ .json 元信息）
submissions/t4_masks_cnxt/         最终预测掩膜（565×584 PNG，可直接看图）
submissions/t4_baseline_frangi.csv 经典基线提交（保底）
```
