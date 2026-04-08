"""
feature_extractor.py
====================
在线推理接入模块：支持“Rule + MLP + RF”异构级联路由。

目标：
1. 保持统一接口：predict_difficulty(prompt: str) -> tuple[int, str]
2. 先规则、再 MLP、后 RF 的三级漏斗决策。
3. 返回“最终难度 + 决策引擎名称（Rule/MLP/RF）”。
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

# 中等任务意图关键词（用于难度下限保护）。
_MEDIUM_INTENT_KEYWORDS = [
    "翻译",
    "简述",
    "介绍",
    "总结",
    "通俗",
    "概述",
    "历史意义",
    "解释一下",
    "请解释",
]

# Level 3 概率救援阈值：
# 当模型主判为 Level 1 但类3概率足够高时，强制拉回到 Level 3。
_RF_L3_RESCUE_THRESHOLD = 0.35
_MLP_L3_RESCUE_THRESHOLD = 0.40

# 决策引擎标签（用于 CSV/论文可解释性分析）
DECISION_RULE = "Rule"
DECISION_MLP = "MLP"
DECISION_RF = "RF"


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


def _should_short_circuit_level1(text: str) -> bool:
    """
    第一关规则防线：
    - 极短文本（<=12）
    - 且不含硬核编程/逻辑特征
    满足即直接判为 Level 1。
    """
    if not text:
        return True
    return len(text) <= 12 and (not _is_light_hard_prompt(text))


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


def _clamp_level(level: int) -> int:
    if level < 1:
        return 1
    if level > 3:
        return 3
    return level


def predict_difficulty_with_trace(prompt: str, engine: str = "hybrid") -> Tuple[int, str, str]:
    """
    统一难度预测接口（带流程追踪）。

    参数：
    - prompt: 输入文本
    - engine:
      - "hybrid"（默认）并行机制：
        A) Rule候选 -> MLP复核（仅MLP拍板）
        B) 非Rule候选 -> MLP初判 -> RF终判
      - "rf"：Rule -> RF（兼容旧评测）
      - "mlp"：Rule -> MLP（兼容旧评测）

    返回：
    - (difficulty, decision_engine, decision_flow)
      difficulty: 1/2/3
      decision_engine: Rule / MLP / RF
      decision_flow: Rule/MLP/RF 判定规则切换流程
    """
    text = (prompt or "").strip()
    mode = (engine or "hybrid").strip().lower()

    rule_hit = _should_short_circuit_level1(text)

    mlp_ready = _MLP_MODEL is not None and _MLP_SCALER is not None and TORCH_AVAILABLE
    rf_ready = _RF_MODEL is not None and _RF_SCALER is not None

    # 兼容模式：RF 单引擎
    if mode == "rf":
        if rule_hit:
            return 1, DECISION_RULE, "Rule命中->最终1"
        feats = np.asarray(extract_features(text), dtype=np.float64).reshape(1, -1)
        if not rf_ready:
            raise RuntimeError("RF 模型未就绪，无法执行 engine='rf' 预测。")
        rf_pred = _clamp_level(_predict_with_rf(feats))
        return rf_pred, DECISION_RF, f"RF直判({rf_pred})->最终{rf_pred}"

    # 兼容模式：MLP 单引擎
    if mode == "mlp":
        if rule_hit:
            return 1, DECISION_RULE, "Rule命中->最终1"
        feats = np.asarray(extract_features(text), dtype=np.float64).reshape(1, -1)
        if not mlp_ready:
            raise RuntimeError("MLP 模型未就绪，无法执行 engine='mlp' 预测。")
        mlp_pred = _clamp_level(_predict_with_mlp(feats))
        return mlp_pred, DECISION_MLP, f"MLP直判({mlp_pred})->最终{mlp_pred}"

    # 标准模式：异构级联（Hybrid）
    if mode not in {"hybrid", "cascade"}:
        raise ValueError("engine 参数仅支持 'hybrid'/'cascade' 或兼容模式 'rf'/'mlp'")

    feats = np.asarray(extract_features(text), dtype=np.float64).reshape(1, -1)

    # 并行分支 A：Rule 命中后仅交给 MLP 复核，不再串到 RF。
    # 规则：
    # - MLP 也判 1，则最终为 1；
    # - MLP 判 2/3，则最终按 MLP 输出。
    if rule_hit:
        if not mlp_ready:
            # 若 MLP 不可用，保留 Rule 兜底，避免服务不可用。
            return 1, DECISION_RULE, "Rule命中->MLP不可用->Rule兜底(1)"
        mlp_rule_pred = _clamp_level(_predict_with_mlp(feats))
        if mlp_rule_pred == 1:
            return 1, DECISION_MLP, "Rule命中->MLP复核(1)->最终1"
        return mlp_rule_pred, DECISION_MLP, f"Rule命中->MLP复核({mlp_rule_pred})->最终{mlp_rule_pred}"

    # 并行分支 B：非 Rule 样本执行 MLP -> RF 级联。
    if mlp_ready:
        mlp_pred = _clamp_level(_predict_with_mlp(feats))
    else:
        # MLP 不可用时退化为 RF 决策。
        if rf_ready:
            rf_pred = _clamp_level(_predict_with_rf(feats))
            return rf_pred, DECISION_RF, f"MLP不可用->RF直判({rf_pred})->最终{rf_pred}"
        raise RuntimeError("MLP 与 RF 模型均未就绪，无法进行难度预测。")

    # 第三关（RF 专家）：非 Rule 分支中由 RF 最终拍板。
    # 规则：
    # - 若 MLP=1 且 RF=1，最终才为 1；
    # - 否则按 RF 输出。
    if rf_ready:
        rf_pred = _clamp_level(_predict_with_rf(feats))
        if mlp_pred == 1 and rf_pred == 1:
            return 1, DECISION_RF, "MLP初判(1)->RF复判(1)->最终1"
        return rf_pred, DECISION_RF, f"MLP初判({mlp_pred})->RF复判({rf_pred})->最终{rf_pred}"

    # RF 不可用时兜底：使用 MLP 结果。
    return _clamp_level(mlp_pred), DECISION_MLP, f"MLP初判({mlp_pred})->RF不可用->最终{mlp_pred}"


def predict_difficulty(prompt: str, engine: str = "hybrid") -> Tuple[int, str]:
    """
    兼容接口：返回 (difficulty, decision_engine)。
    """
    difficulty, decision_engine, _ = predict_difficulty_with_trace(prompt, engine=engine)
    return difficulty, decision_engine


if __name__ == "__main__":
    demo_texts = [
        "请回复收到",
        "请证明 O(log n)",
        "解释这段代码的异常处理",
    ]

    print("\n=== Heterogeneous Cascade Demo ===")
    for t in demo_texts:
        try:
            pred, dec, flow = predict_difficulty_with_trace(t)
            print(f"Prompt: {t}")
            print(f"Pred: {pred}, Decision: {dec}, Flow: {flow}")
        except Exception as exc:
            print(f"Pred Error: {exc}")
        print("-" * 50)
