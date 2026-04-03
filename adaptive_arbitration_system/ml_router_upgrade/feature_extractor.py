"""
feature_extractor.py
====================
重构后的在线推理接入模块（替代旧版规则 if-else 逻辑）。

目标：
1. 对外保持接口一致：predict_difficulty(prompt: str) -> int
2. 内部改为“特征工程 + 标准化 + RandomForest 分类”
3. 与现有 arbitrator_core.py 无缝兼容（函数名与返回值类型不变）

使用方式：
- 先运行 train_router.py 生成 router_model.pkl
- 再在主系统中 import 本文件的 predict_difficulty
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import joblib
import numpy as np

# 兼容两种运行方式：
# 1) 作为包被主系统导入：from ml_router_upgrade.feature_extractor import ...
# 2) 直接在当前目录执行脚本：python3 feature_extractor.py
try:
    from .feature_engineering import extract_features
except ImportError:  # pragma: no cover - 脚本直跑兜底
    from feature_engineering import extract_features

# 模型文件路径：与当前脚本同目录。
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "router_model.pkl")


def _load_bundle(path: str) -> Tuple[Any, Any]:
    """
    加载模型与标准化器。

    约定：
    - train_router.py 将保存一个字典对象，至少包含：
      {"model": RandomForestClassifier, "scaler": StandardScaler}

    异常策略：
    - 文件不存在时抛出友好错误，提醒先训练。
    - 文件结构不符合预期时，抛出解释性错误，便于快速排查。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            "未找到 router_model.pkl。请先在同目录运行 train_router.py 完成模型训练与导出。"
        )

    bundle = joblib.load(path)

    if isinstance(bundle, dict):
        model = bundle.get("model")
        scaler = bundle.get("scaler")
    else:
        raise ValueError(
            "router_model.pkl 格式不正确：期望为 dict，且包含 model 与 scaler。"
        )

    if model is None or scaler is None:
        raise ValueError(
            "router_model.pkl 缺少 model/scaler 字段，请重新运行 train_router.py。"
        )

    return model, scaler


# 模块加载时执行一次，避免每次预测重复 IO，保证在线推理延迟极低。
_MODEL, _SCALER = _load_bundle(MODEL_PATH)


def predict_difficulty(prompt: str) -> int:
    """
    对外统一接口：预测难度等级（1/2/3）。

    流程：
    1. extract_features(prompt) -> 5D 向量
    2. scaler.transform() 标准化
    3. RandomForest.predict() 输出类别

    返回：
    - 1（简单）
    - 2（中等）
    - 3（困难）
    """
    text = prompt or ""

    # 形状从 (5,) 调整为 (1,5)，符合 sklearn 批量输入格式。
    feats = np.asarray(extract_features(text), dtype=np.float64).reshape(1, -1)

    feats_scaled = _SCALER.transform(feats)
    pred = int(_MODEL.predict(feats_scaled)[0])

    # 防御性兜底：若模型意外输出越界值，按区间截断，确保兼容旧系统。
    if pred < 1:
        return 1
    if pred > 3:
        return 3
    return pred


if __name__ == "__main__":
    demo_texts = [
        "请回复收到",
        "请简述牛顿第一定律，并举一个生活中的例子。",
        "请阅读这段代码并分析复杂度：```python\ndef f(x):\n    return x*x\n``` 然后解释为什么这样设计。",
    ]
    for t in demo_texts:
        print(f"Prompt: {t}")
        print(f"Pred: {predict_difficulty(t)}")
        print("-" * 50)
