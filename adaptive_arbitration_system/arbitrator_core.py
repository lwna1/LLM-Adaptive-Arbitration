"""
arbitrator_core.py
------------------
本模块是混合自适应仲裁系统的核心调度大脑，负责：
1. 融合“难度预测 + 硬件状态”进行首次路由。
2. 对输出做质量检测。
3. 在质量不达标时进行级联升级（Cascade Retry）。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from config_and_api import LLMAPIError, call_llm
from device_simulator import DeviceSimulator
from ml_router_upgrade.feature_extractor import predict_difficulty


# 模型从小到大顺序，用于级联升级判断。
MODEL_ORDER = ["qwen-0.5b", "llama-1b", "qwen-1.5b"]

# 拒答/低质量关键词，命中即视为可能不达标。
REFUSAL_KEYWORDS = [
    "不知道",
    "抱歉",
    "作为一个AI",
    "无法回答",
    "缺乏信息",
]

# 针对 llama-1b 的特定风险词（系统日志幻觉）。
LLAMA_HALLUCINATION_KEYWORDS = [
    "任务列表",
    "系统状态",
]


class AdaptiveArbitrator:
    """
    多目标仲裁器：
    - 目标 1：在硬件约束下尽量选择合适算力模型。
    - 目标 2：在输出质量不佳时尽可能进行级联补救。
    - 目标 3：记录全链路耗时与平均 TPS，用于后续实验统计。
    """

    def __init__(self, device: Optional[DeviceSimulator] = None) -> None:
        self.device = device or DeviceSimulator()

    @staticmethod
    def _difficulty_to_model(difficulty: int) -> str:
        """NORMAL 情况下的默认映射：1->0.5B, 2->1B, 3->1.5B。"""
        mapping = {
            1: "qwen-0.5b",
            2: "llama-1b",
            3: "qwen-1.5b",
        }
        return mapping.get(difficulty, "llama-1b")

    @staticmethod
    def _constraint_max_model(constraint: str) -> str:
        """根据硬件约束返回当前可用的最大模型。"""
        if constraint == "THROTTLED_05B":
            return "qwen-0.5b"
        if constraint == "THROTTLED_1B":
            return "llama-1b"
        return "qwen-1.5b"

    def _clip_model_by_constraint(self, model_name: str, constraint: str) -> str:
        """若候选模型超过约束上限，则截断到约束允许的最大模型。"""
        max_model = self._constraint_max_model(constraint)
        if MODEL_ORDER.index(model_name) > MODEL_ORDER.index(max_model):
            return max_model
        return model_name

    def _select_initial_model(self, difficulty: int, constraint: str) -> str:
        """
        首次路由策略（严格对应你的需求）：
        1) THROTTLED_05B: 强制 0.5B。
        2) THROTTLED_1B: 难度映射后再截断到 1B。
        3) NORMAL: 1->0.5B, 2->1B, 3->1.5B。
        """
        if constraint == "THROTTLED_05B":
            return "qwen-0.5b"

        preferred = self._difficulty_to_model(difficulty)

        if constraint == "THROTTLED_1B":
            return self._clip_model_by_constraint(preferred, constraint)

        return preferred

    @staticmethod
    def _is_unqualified(answer: str, difficulty: int, model_name: str) -> bool:
        """
        质量评估：输出是否不达标。

        不达标规则：
        1) 包含拒答词。
        2) 对 Level 2/3 问题，回答长度 < 10 字。
        3) 若模型为 llama-1b，且包含“任务列表/系统状态”等幻觉词。
        """
        text = (answer or "").strip()

        if any(word in text for word in REFUSAL_KEYWORDS):
            return True

        if difficulty in (2, 3) and len(text) < 10:
            return True

        if model_name == "llama-1b" and any(word in text for word in LLAMA_HALLUCINATION_KEYWORDS):
            return True

        return False

    @staticmethod
    def _next_model(current_model: str) -> str:
        """返回下一个更大模型；若已是最大模型则返回自身。"""
        idx = MODEL_ORDER.index(current_model)
        if idx >= len(MODEL_ORDER) - 1:
            return current_model
        return MODEL_ORDER[idx + 1]

    def _can_upgrade(self, current_model: str, constraint: str) -> bool:
        """
        判断在当前约束下是否还能升级到更大模型。
        例如：
        - THROTTLED_1B 下，llama-1b 不能再升到 1.5b。
        - NORMAL 下，0.5b/1b 都可继续升级。
        """
        max_model = self._constraint_max_model(constraint)
        return MODEL_ORDER.index(current_model) < MODEL_ORDER.index(max_model)

    def adaptive_process(self, prompt: str) -> Dict[str, object]:
        """
        执行完整的“路由 + 推理 + 质量评估 + 级联补救”流程。

        返回字段：
        - answer: 最终答案
        - total_latency: 全部调用累计耗时（秒）
        - avg_tps: 平均 TPS
        - call_chain: 实际调用模型链路（按顺序）
        - difficulty: 系统预测难度（便于上层 benchmark 直接记录）
        """
        difficulty = predict_difficulty(prompt)
        initial_constraint = self.device.get_hardware_constraint()
        current_model = self._select_initial_model(difficulty, initial_constraint)

        call_chain: List[str] = []
        total_latency = 0.0
        tps_values: List[float] = []
        final_answer = ""

        while True:
            # A. 先执行一次模型调用，并将本次调用带来的硬件损耗计入状态。
            self.device.update_state(current_model)
            call_chain.append(current_model)

            api_failed = False
            try:
                answer, latency, tps = call_llm(current_model, prompt)
            except LLMAPIError as exc:
                # 调用失败同样视作一次低质量结果，后续会按策略尝试级联补救。
                answer = f"[模型调用异常] {exc}"
                latency = float(getattr(exc, "latency", 0.0))
                tps = 0.0
                api_failed = True

            total_latency += latency
            if tps > 0:
                tps_values.append(tps)

            final_answer = answer.strip()
            unqualified = api_failed or self._is_unqualified(final_answer, difficulty, current_model)

            # B. 质量达标，直接结束。
            if not unqualified:
                break

            # C. 质量不达标 -> 判断当前硬件是否允许继续升级。
            current_constraint = self.device.get_hardware_constraint()
            if self._can_upgrade(current_model, current_constraint):
                current_model = self._next_model(current_model)
                continue

            # 已受限且不能升级：按要求拼接硬件限制提示并终止级联。
            if current_constraint in ("THROTTLED_05B", "THROTTLED_1B"):
                final_answer = f"[受硬件过热限制，强制终止级联]{final_answer}"
            break

        avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0.0

        return {
            "answer": final_answer,
            "total_latency": round(total_latency, 4),
            "avg_tps": round(avg_tps, 4),
            "call_chain": call_chain,
            "difficulty": difficulty,
        }
