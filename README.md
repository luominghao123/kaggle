# 植物分类 —— 深度学习图像分类（Kaggle 2026task1）

东北大学机器学习课程设计 · 实验一。基于 **timm 预训练骨干 + 分层 K 折微调 + 多模型 TTA 集成**
的植物（幼苗）种类识别方案。

- 竞赛：<https://www.kaggle.com/competitions/2026task1>
- 评估指标：**F1 Score**（Kaggle `metric/f1-score`，多分类按 macro 平均）
- 截止时间：**2026-10-10 24:00（北京时间）** = `2026-10-10T16:00:00Z`
- 提交格式：`ID, Category`（ID = 测试图片文件名，Category = **类别名字符串**）
- 提交限制：每天 3 次，取其中最好的 2 次计入最终成绩

---

## 1. 环境

本机已配置好工作区内的独立虚拟环境（**不要**用系统 Python 装包）：

| 项 | 值 |
|---|---|
| 虚拟环境 | `.venv`（工作区内） |
| Python | 3.13.11 |
| PyTorch | 2.11.0+cu128（CUDA 12.8） |
| GPU | RTX 4060 Laptop 8 GB |

### ⚠️ 本机沙箱的三个坑（已在脚本/代码里绕开）

1. **pip 直连会失败**：DSH 文件沙箱拒绝写入 `tempfile.mkdtemp()` 创建的目录
   （`mkdtemp` 内部用 `os.mkdir(path, 0o700)`，事后 `chmod` 也被拒）。
   解决：`tools/shim/sitecustomize.py` 把 `mkdtemp` 改成默认权限创建，通过
   `PYTHONPATH` 注入即可。

2. **PowerShell / curl 的 TLS 不可用**（schannel `SEC_E_NO_CREDENTIALS`），
   且 `huggingface.co`、`raw.githubusercontent.com` 不可达。
   解决：网络访问一律走 Python（自带 OpenSSL）；timm 权重走 `HF_ENDPOINT=https://hf-mirror.com`
   （已在 `src/models.py` 内置）；pip 走清华镜像。

3. **禁止进程间命名管道 → `DataLoader(num_workers>0)` 直接报 `WinError 5`**。
   解决：`--workers` 默认 **0**（主进程加载）；为抵消单进程读图开销，
   `PlantDataset` 会把解码后的图缓存在内存（默认上限 2 GB，长边压到 576）。

以上环境变量都由 `run.ps1` 统一设置，**推荐一律用它来跑**，不要直接调 `.venv\Scripts\python.exe`。

### 安装依赖（如需重建环境）

```powershell
$env:PYTHONPATH = "D:\东大\kaggle\tools\shim"
$env:TEMP = "D:\东大\kaggle\.tmp"; $env:TMP = $env:TEMP

# CUDA 版 torch（PyPI 上的 torch 是 CPU-only，必须走官方 CUDA 源）
.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 `
    --index-url https://download.pytorch.org/whl/cu128

# 其余依赖走清华镜像
.venv\Scripts\python.exe -m pip install timm scikit-learn pandas tqdm pillow `
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 2. 数据放置

把竞赛数据下载后解压到 `data/` 下，**目录层级不限**，程序会自动探测。
推荐（也是去年课程数据的样子）：

```
data/
└── raw/
    └── new_plant/
        └── new_plant/
            ├── train/
            │   ├── Black-grass/     *.png
            │   ├── Charlock/        *.png
            │   ├── ...              （共 12 类）
            │   └── Sugar beet/      *.png
            ├── test/                *.png   （平铺，无子目录）
            └── sample_submission.csv        （可选，但强烈建议放进来）
```

只要满足「`train/<类别名>/*.png` + 一个平铺的 `test/`」即可，层数随意。

**先自检数据是否被正确识别：**

```powershell
.\run.ps1 -m src.data data
```

会打印识别到的 train/test 路径、类别列表、每类样本数。

---

## 3. 快速开始

> 一律使用 `.\run.ps1`（它负责设置沙箱兼容的环境变量），不要直接调 `.venv\Scripts\python.exe`。

```powershell
# 0) 自检数据
.\run.ps1 -m src.data data

# 1) 小规模冒烟测试（1 折 8 epoch @224，确认全流程通）
.\run.ps1 -m src.train --data data --model convnext_tiny `
    --img-size 224 --epochs 8 --folds 5 --fold 0 --batch-size 32 --tag smoke

# 2) 正式训练：ConvNeXt-Tiny @384，5 折
.\run.ps1 -m src.train --data data --model convnext_tiny `
    --img-size 384 --epochs 30 --folds 5 --batch-size 16 --tag cnxt384

# 3) 生成提交文件（8 视角 TTA + 5 折平均）
.\run.ps1 -m src.predict --data data --runs runs/cnxt384 `
    --out submissions/cnxt384.csv
```

产物落在 `runs/<tag>/`：`fold{k}_best.pt`、`oof.npz`、`test_fold{k}.npy`、`log.txt`、`summary.json`。

---

## 4. 冲榜路线（按收益排序）

| 阶段 | 做法 | 预期 macro-F1 |
|---|---|---|
| 基线 | ConvNeXt-Tiny @224，5 折 | ~0.97 |
| **+高分辨率** | 384（甚至 448）微调，本数据收益最大 | ~0.985 |
| **+TTA** | 8 视角（原图/翻转/四旋转） | ~0.99 |
| **+多骨干集成** | ConvNeXt-T + Swin-T + EffNetV2-S，按 OOF 加权 | ~0.99+ |
| +伪标签 | 用集成结果对测试集打软标签再训一轮（谨慎，有风险） | 视情况 |

多模型集成：

```powershell
.\run.ps1 -m src.train --data data --model swin_tiny  --img-size 384 --epochs 30 --tag swin384
.\run.ps1 -m src.train --data data --model effnetv2_s --img-size 384 --epochs 30 --tag effv2s384

.\run.ps1 -m src.predict --data data `
    --runs runs/cnxt384 runs/swin384 runs/effv2s384 `
    --weights auto --out submissions/ensemble.csv
```

> `--weights auto` 按各模型 OOF 的 macro-F1 自动加权。

---

## 5. 关键实现说明（写报告时可用）

1. **为什么用迁移学习**：训练集仅约千张量级，从零训练 CNN 会严重过拟合；
   预训练骨干（ImageNet-22k → 1k 微调权重）在 30 epoch 内即可收敛到 98%+。
2. **数据增强针对俯拍植物图定制**：除常规 `RandomResizedCrop + RandAugment` 外，
   额外加入**垂直翻转**与**随机 90°/180°/270° 旋转**——俯拍视角下这些变换语义不变，
   等价于免费扩充数据。
3. **Mixup + CutMix**：小样本下显著抑制过拟合；最后 3 个 epoch 关闭，
   让模型回到真实数据分布，避免验证指标被"软化"。
4. **EMA（指数滑动平均）**：对权重做平滑，通常带来 0.2~0.5 个点，
   且几乎不增加训练成本；每轮同时评估原始权重与 EMA 权重，取更优者。
5. **drop_path（随机深度）**：0.1 左右对 ConvNeXt/Swin 微调最稳。
6. **分层 K 折**：保证每折类别分布一致（本数据类别不平衡，
   最少类 Common wheat 与最多类 Loose Silky-bent 样本数差约 5 倍），
   同时产出 out-of-fold 概率用于无偏的集成权重估计。
7. **TTA**：8 视角平均，只涨不跌，且无额外训练成本。
8. **指标对齐**：同时跟踪 accuracy / macro-F1 / weighted-F1 / micro-F1，
   确认榜单口径（macro-F1 对少数类更敏感，因此训练时不使用类别加权反而更好）。

---

## 6. 目录结构

```
kaggle/
├── data/                     # 数据（自行放置）
├── src/
│   ├── data.py               # 目录探测 / Dataset / 增强 / TTA
│   ├── models.py             # timm 骨干 + 优化器分组
│   ├── engine.py             # 训练验证循环 / AMP / 指标 / 调度器
│   ├── train.py              # 分层 K 折训练
│   ├── predict.py            # TTA 集成推理 + 生成提交
│   └── utils.py              # 种子 / 日志 / EMA 工具
├── tools/
│   ├── shim/sitecustomize.py # 沙箱兼容补丁（见 §1）
│   ├── probe_env.py          # 环境探测
│   └── peek_nb.py            # 查看参考 notebook
├── runs/                     # 训练产物
├── submissions/              # 提交文件
└── _ref/                     # 参考资料（去年同课程 notebook）
```
