"""植物分类 —— 东北大学机器学习课程设计 实验一（Kaggle 2026task1）。

模块划分
--------
    data.py     目录自动探测、数据集、数据增强 / TTA
    models.py   timm 骨干网络构建、优化器参数分组
    engine.py   训练/验证循环、AMP、指标计算、余弦调度
    train.py    分层 K 折训练入口
    predict.py  多模型多折 TTA 集成 + 生成 submission.csv
    utils.py    随机种子、日志、EMA 等通用工具
"""

__version__ = "1.0.0"
