"""
feature_extractor.py
====================
在线推理接入模块：支持随机森林（RF）与多层感知机（MLP）双引擎热切换。

目标：
1. 保持统一接口：predict_difficulty(prompt: str, engine='rf') -> int
2. 支持在随机森林与神经网络之间无缝切换。
3. 在模型文件缺失时提供友好提示，便于排查部署问题。
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional, Tuple

import joblib
import numpy as np

# 兼容两种运行方式：
# 1) 作为包被主系统导入：from ml_router_upgrade.feature_extractor import ...
# 2) 直接在当前目录执行脚本：python3 feature_extractor.py
try:
    from .feature_engineering import extract_features
except ImportError:  # pragma: no cover - 脚本直跑兜底
    from feature_engineering import extract_features


# ========= PyTorch 可选依赖加载 =========
# 为了保证“即使未安装 torch 也能继续使用 RF 引擎”，这里采用防御性加载。
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - 环境无 torch 时兜底
    torch = None
    nn = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class MLPRouter(nn.Module):
        """
        MLP 路由器网络结构（与 train_mlp_router.py 保持完全一致）。

        结构：
        - 输入层：6
        - 隐藏层1：Linear(6,64)+ReLU+Dropout(0.2)
        - 隐藏层2：Linear(64,32)+ReLU
        - 输出层：Linear(32,3)
        """

        def __init__(self, input_dim: int = 6) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
else:
    # 仅用于类型占位，避免 NameError。
    class MLPRouter:  # pragma: no cover
        pass


# ========= 模型文件路径 =========
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RF_MODEL_PATH = os.path.join(CURRENT_DIR, "router_model.pkl")
MLP_MODEL_PATH = os.path.join(CURRENT_DIR, "mlp_router.pth")
MLP_SCALER_PATH = os.path.join(CURRENT_DIR, "mlp_scaler.pkl")


def _load_rf_bundle(path: str) -> Tuple[Optional[Any], Optional[Any]]:
    """加载随机森林模型与标准化器。"""
    if not os.path.exists(path):
        print("[WARN] 未找到 router_model.pkl，RF 引擎不可用。")
        return None, None

    try:
        bundle = joblib.load(path)
        if not isinstance(bundle, dict):
            print("[WARN] router_model.pkl 格式异常，RF 引擎不可用。")
            return None, None

        model = bundle.get("model")
        scaler = bundle.get("scaler")
        if model is None or scaler is None:
            print("[WARN] router_model.pkl 缺少 model/scaler 字段，RF 引擎不可用。")
            return None, None

        print("[INFO] RF 引擎加载成功。")
        return model, scaler
    except Exception as exc:
        print(f"[WARN] RF 引擎加载失败：{exc}")
        return None, None


def _load_mlp_bundle(
    model_path: str,
    scaler_path: str,
) -> Tuple[Optional[Any], Optional[Any], int]:
    """加载 MLP 权重与标准化器。"""
    if not TORCH_AVAILABLE:
        print("[WARN] 当前环境未安装 PyTorch，MLP 引擎不可用。")
        return None, None, 6

    if not os.path.exists(model_path):
        print("[WARN] 未找到 mlp_router.pth，MLP 引擎不可用。")
        return None, None, 6

    if not os.path.exists(scaler_path):
        print("[WARN] 未找到 mlp_scaler.pkl，MLP 引擎不可用。")
        return None, None, 6

    try:
        checkpoint = torch.load(model_path, map_location="cpu")

        # 兼容不同保存格式：
        # 1) dict + model_state_dict
        # 2) 直接 state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            input_dim = int(checkpoint.get("input_dim", 6))
        else:
            state_dict = checkpoint
            input_dim = 6

        model = MLPRouter(input_dim=input_dim)
        model.load_state_dict(state_dict)
        model.eval()

        scaler = joblib.load(scaler_path)

        print("[INFO] MLP 引擎加载成功。")
        return model, scaler, input_dim
    except Exception as exc:
        print(f"[WARN] MLP 引擎加载失败：{exc}")
        return None, None, 6


# 模块加载时执行一次，避免每次推理重复磁盘 IO。
_RF_MODEL, _RF_SCALER = _load_rf_bundle(RF_MODEL_PATH)
_MLP_MODEL, _MLP_SCALER, _MLP_INPUT_DIM = _load_mlp_bundle(MLP_MODEL_PATH, MLP_SCALER_PATH)

# 轻量硬核关键词，用于前置启发式拦截（与 ML 解耦，避免强依赖训练侧词表）。
_LIGHT_HARD_KEYWORDS = [
    "代码",
    "算法",
    "证明",
    "复杂度",
    "设计",
    "推导",
    "原理",
    "逻辑",
    "架构",
    "微服务",
    "并发",
    "二分",
    "排序",
    "图论",
    "动态规划",
    "事务",
    "一致性",
    "实现",
]

# 明确的“短文本低负荷”白名单（仅这些才能直接短路为 Level 1）。
_TRIVIAL_SHORT_TEXTS = {
    "你好",
    "您好",
    "请回复收到",
    "收到",
    "在吗",
    "谢谢",
    "早上好",
    "晚上好",
    "再见",
    "ok",
    "okay",
}

# 简单算术题模式（如 1+1等于几）
_SIMPLE_ARITHMETIC_PATTERN = re.compile(r"^\s*\d+\s*[\+\-\*/]\s*\d+\s*(等于几|\?)?\s*$")

# 明显高难模式（短文本也应判为高难候选）
_HARD_SHORT_PATTERNS = [
    re.compile(r"O\(.+\)", re.IGNORECASE),
    re.compile(r"\b(log|bfs|dfs|sql|rpc|api|raft|paxos)\b", re.IGNORECASE),
    re.compile(r"(快排|归并|最短路|并查集|哈希|拓扑|红黑树|线段树|贪心|递归)"),
]

# Level 3 概率救援阈值：
# 当模型主判为 Level 1 但 Level 3 概率足够高时，强制拉回到 Level 3。
_RF_L3_RESCUE_THRESHOLD = 0.35
_MLP_L3_RESCUE_THRESHOLD = 0.40


def _is_light_hard_prompt(text: str) -> bool:
    """
    判断短文本中是否包含硬核编程/逻辑特征。

    说明：
    - 该判断用于 Rule + ML 混合路由中的“前置拦截”。
    - 只做轻量字符串匹配，保证推理路径开销极低。
    """
    if not text:
        return False

    lowered = text.lower()
    if "{" in text or "def " in lowered or "class " in lowered:
        return True

    if any(keyword in text for keyword in _LIGHT_HARD_KEYWORDS):
        return True

    return any(pattern.search(text) is not None for pattern in _HARD_SHORT_PATTERNS)


def _hard_floor_difficulty(text: str) -> int:
    """
    对“短且明确硬核”的 prompt 施加难度下限。

    返回：
    - 3: 明确硬核短文本（默认提升到困难）
    - 0: 不触发
    """
    if len(text) <= 20 and _is_light_hard_prompt(text):
        return 3
    return 0


def _should_short_circuit_level1(text: str) -> bool:
    """
    前置启发式规则（收敛版）：
    - 仅对“明确低负荷短文本”短路为 Level 1；
    - 不再简单以“长度<=12 且无硬核词”粗暴短路，避免误伤短高难问题。
    """
    if len(text) > 12:
        return False

    if _is_light_hard_prompt(text):
        return False

    lowered = text.lower().strip()
    if lowered in _TRIVIAL_SHORT_TEXTS:
        return True

    if _SIMPLE_ARITHMETIC_PATTERN.match(text):
        return True

    return False


def _predict_with_rf(feats: np.ndarray) -> int:
    """使用随机森林预测难度。"""
    if _RF_MODEL is None or _RF_SCALER is None:
        raise RuntimeError("RF 引擎未就绪。请先运行 train_router.py 生成 router_model.pkl。")

    feats_scaled = _RF_SCALER.transform(feats)
    pred = int(_RF_MODEL.predict(feats_scaled)[0])

    # RF 专用 Level 3 召回救援：
    # 若主判为 1，但类3概率已较高，则拉回 3，降低“3->1”误判。
    try:
        proba = _RF_MODEL.predict_proba(feats_scaled)[0]
        classes = list(_RF_MODEL.classes_)
        if pred == 1 and 3 in classes:
            p3 = float(proba[classes.index(3)])
            if p3 >= _RF_L3_RESCUE_THRESHOLD:
                return 3
    except Exception:
        # 防御性兜底：概率读取失败时退回原始预测。
        pass

    return pred


def _predict_with_mlp(feats: np.ndarray) -> int:
    """使用 MLP 预测难度。"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("MLP 引擎不可用：当前环境未安装 PyTorch。")

    if _MLP_MODEL is None or _MLP_SCALER is None:
        raise RuntimeError("MLP 引擎未就绪。请先运行 train_mlp_router.py 生成 mlp_router.pth 与 mlp_scaler.pkl。")

    feats_scaled = _MLP_SCALER.transform(feats)
    x = torch.tensor(feats_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = _MLP_MODEL(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(logits, dim=1).item())

        # MLP 专用 Level 3 召回救援：
        # pred_idx=0 对应 Level 1；idx=2 对应 Level 3。
        if pred_idx == 0:
            p3 = float(probs[0, 2].item())
            if p3 >= _MLP_L3_RESCUE_THRESHOLD:
                return 3

    # 训练时标签被映射为 0/1/2，这里恢复为 1/2/3。
    return pred_idx + 1


def predict_difficulty(prompt: str, engine: str = "rf") -> int:
    """
    统一难度预测接口。

    参数：
    - prompt: 输入文本
    - engine: 路由引擎，支持 'rf' 或 'mlp'

    返回：
    - 1（简单）
    - 2（中等）
    - 3（困难）
    """
    text = (prompt or "").strip()

    # 加入启发式前置规则，处理短文本极端特征分布问题。
    # 典型场景：如“1+1等于几”这类超短文本，避免被符号特征异常放大后误判为高难任务。
    if _should_short_circuit_level1(text):
        return 1

    hard_floor = _hard_floor_difficulty(text)

    feats = np.asarray(extract_features(text), dtype=np.float64).reshape(1, -1)

    selected_engine = (engine or "rf").strip().lower()

    if selected_engine == "mlp":
        pred = _predict_with_mlp(feats)
    elif selected_engine == "rf":
        pred = _predict_with_rf(feats)
    else:
        raise ValueError("engine 参数仅支持 'rf' 或 'mlp'")

    # 对短硬核任务执行最终下限约束，避免双引擎共同掉到 Level 1。
    if hard_floor > 0:
        pred = max(pred, hard_floor)

    # 防御性兜底。
    if pred < 1:
        return 1
    if pred > 3:
        return 3
    return pred


if __name__ == "__main__":
    demo_texts = [
        "请回复收到",
        "请证明 O(log n)",
        "解释这段代码的异常处理",
    ]

    for eng in ["rf", "mlp"]:
        print(f"\n=== Engine: {eng} ===")
        for t in demo_texts:
            try:
                print(f"Prompt: {t}")
                print(f"Pred: {predict_difficulty(t, engine=eng)}")
            except Exception as exc:
                print(f"Pred Error: {exc}")
            print("-" * 50)
