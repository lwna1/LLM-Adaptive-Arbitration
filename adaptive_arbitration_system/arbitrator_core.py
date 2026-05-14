"""
arbitrator_core.py
------------------
本模块是混合自适应仲裁系统的核心调度大脑，负责：
1. 融合“难度预测 + Safety Score”进行动态路由与自适应降级。
2. 通过零参数轻量质量评估器执行端侧低开销回复验收。
3. 在每次调用后进行状态回写，形成“时间-能耗-温度-路由”闭环。
4. 当端侧无法继续升级且回答仍不合格时，触发云端兜底卸载。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Pattern, Tuple

from config_and_api import LLMAPIError, call_llm
from device_simulator import DeviceSimulator
from ml_router_upgrade.feature_extractor import predict_difficulty_with_trace


MODEL_ORDER = ["qwen-0.5b", "llama-1b", "qwen-1.5b"]

REFUSAL_PATTERNS = [
    "我是一个人工智能",
    "作为一个人工智能",
    "无法回答",
    "我无法回答",
    "不能回答",
    "缺乏信息",
]

ACK_WORDS = {
    "收到",
    "好的",
    "明白",
    "已收到",
    "了解",
    "ok",
    "okay",
}

CLOUD_FALLBACK_TEXT = "这是由云端大模型 API 生成的高质量兜底回答。"
CLOUD_FALLBACK_LATENCY = 1.2

_HAS_LATIN_TEXT_PATTERN = re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]")
_HAS_CJK_TEXT_PATTERN = re.compile(r"[\u4E00-\u9FFF]")
_HAS_JAPANESE_TEXT_PATTERN = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]")
_HAS_KOREAN_TEXT_PATTERN = re.compile(r"[\uAC00-\uD7AF]")
_HAS_CYRILLIC_TEXT_PATTERN = re.compile(r"[\u0400-\u04FF]")
_HAS_ARABIC_TEXT_PATTERN = re.compile(r"[\u0600-\u06FF]")
_LATIN_TOKEN_PATTERN = re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]+")

# 翻译任务目标语言推断规则。
# 说明：
# - 这里做的是“目标语种/语系存在性校验”，不是完整语言识别。
# - 对英语/法语/西班牙语/德语等拉丁字母语种，统一按 Latin Script 检查。
TRANSLATION_TARGET_RULES: List[Tuple[str, Tuple[str, ...], Pattern[str]]] = [
    ("英文", ("英语", "英文", "english"), _HAS_LATIN_TEXT_PATTERN),
    ("中文", ("中文", "汉语", "汉字", "chinese"), _HAS_CJK_TEXT_PATTERN),
    ("法语", ("法语", "法文", "french"), _HAS_LATIN_TEXT_PATTERN),
    ("西班牙语", ("西班牙语", "西语", "spanish"), _HAS_LATIN_TEXT_PATTERN),
    ("德语", ("德语", "德文", "german"), _HAS_LATIN_TEXT_PATTERN),
    ("意大利语", ("意大利语", "italian"), _HAS_LATIN_TEXT_PATTERN),
    ("葡萄牙语", ("葡萄牙语", "portuguese"), _HAS_LATIN_TEXT_PATTERN),
    ("日语", ("日语", "日文", "japanese"), _HAS_JAPANESE_TEXT_PATTERN),
    ("韩语", ("韩语", "韩文", "korean"), _HAS_KOREAN_TEXT_PATTERN),
    ("俄语", ("俄语", "俄文", "russian"), _HAS_CYRILLIC_TEXT_PATTERN),
    ("阿拉伯语", ("阿拉伯语", "arabic"), _HAS_ARABIC_TEXT_PATTERN),
]

TRANSLATION_META_PATTERNS: List[Pattern[str]] = [
    re.compile(r"请提供"),
    re.compile(r"请输入"),
    re.compile(r"我会翻译"),
    re.compile(r"按照输入"),
    re.compile(r"根据输入"),
    re.compile(r"请您让我"),
    re.compile(r"please provide", re.IGNORECASE),
    re.compile(r"provide the input", re.IGNORECASE),
    re.compile(r"i will translate", re.IGNORECASE),
    re.compile(r"go ahead and provide", re.IGNORECASE),
    re.compile(r"following your intended meaning", re.IGNORECASE),
]

# 对常见拉丁字母目标语言增加轻量“目标语言痕迹”校验。
# 说明：
# - 仅在多词输出时启用，避免误伤单词/短语级翻译。
# - 这些标记词不是完整语言识别，只用于拦截明显错语种或英文混充。
TRANSLATION_LANGUAGE_MARKERS: Dict[str, Pattern[str]] = {
    "英文": re.compile(r"\b(the|is|are|to|and|of|in|for|with|this|that|can|need|please|should)\b", re.IGNORECASE),
    "法语": re.compile(r"\b(le|la|les|un|une|des|et|est|pour|avec|dans|que|vous|nous)\b|[àâçéèêëîïôùûüÿœæ]", re.IGNORECASE),
    "西班牙语": re.compile(r"\b(el|la|los|las|un|una|y|es|para|con|que|como|por|usted|nosotros)\b|[áéíóúñ¿¡]", re.IGNORECASE),
    "德语": re.compile(r"\b(der|die|das|ein|eine|und|ist|nicht|mit|für|ich|wir|sie|zu|auf|bitte|deutsch|deutsche|übersetzen)\b|[äöüß]", re.IGNORECASE),
    "意大利语": re.compile(r"\b(il|lo|la|gli|le|un|una|e|è|per|con|che|non|noi|voi)\b|[àèéìíîòóù]", re.IGNORECASE),
    "葡萄牙语": re.compile(r"\b(o|a|os|as|um|uma|e|é|para|com|que|não|nos|vocês)\b|[áâãàçéêíóôõú]", re.IGNORECASE),
}


def _detect_severe_repetition(answer: str) -> bool:
    """
    复读机检测（Repetition）。
    """
    text = (answer or "").strip()
    if not text:
        return False

    compact = re.sub(r"\s+", "", text)
    if len(compact) < 12:
        return False

    # 连续子串重复检测（如 abcabcabc）。
    max_chunk_len = min(12, len(compact) // 3)
    for chunk_len in range(2, max_chunk_len + 1):
        limit = len(compact) - chunk_len * 3 + 1
        for i in range(limit):
            chunk = compact[i : i + chunk_len]
            if compact[i : i + chunk_len * 3] == chunk * 3:
                return True

    # 分句重复检测（同一句段重复 >= 3）。
    segments = [seg.strip() for seg in re.split(r"[。！？!?；;\n，,]", text) if seg.strip()]
    if len(segments) >= 3:
        seg_counter = Counter(segments)
        if max(seg_counter.values()) >= 3:
            return True

    return False


def _infer_translation_target(prompt: str) -> Optional[Tuple[str, Pattern[str]]]:
    """
    从翻译任务中推断目标语言。

    策略：
    - 仅在包含“翻译/译成/译为”等意图时启用
    - 若 prompt 中出现多个语言关键词，取最后出现的一个
      以兼容“从英语翻译成法语”这类同时包含源语言与目标语言的表达
    """
    text = (prompt or "").strip()
    lowered = text.lower()

    if not any(keyword in text for keyword in ["翻译", "译成", "译为"]):
        return None

    best_match: Optional[Tuple[int, str, Pattern[str]]] = None
    for label, aliases, validator in TRANSLATION_TARGET_RULES:
        for alias in aliases:
            index = lowered.rfind(alias.lower())
            if index < 0:
                continue
            if best_match is None or index > best_match[0]:
                best_match = (index, label, validator)

    if best_match is None:
        return None
    return best_match[1], best_match[2]


def _translation_meta_response_detected(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return False
    return any(pattern.search(text) is not None for pattern in TRANSLATION_META_PATTERNS)


def _validate_translation_target_content(target_label: str, answer: str) -> Optional[str]:
    """
    对翻译结果做更严格的目标语言形态校验。

    返回：
    - None: 通过
    - str: 失败原因
    """
    answer_text = (answer or "").strip()
    latin_chars = len(_HAS_LATIN_TEXT_PATTERN.findall(answer_text))
    cjk_chars = len(_HAS_CJK_TEXT_PATTERN.findall(answer_text))
    latin_tokens = _LATIN_TOKEN_PATTERN.findall(answer_text)

    if target_label in TRANSLATION_LANGUAGE_MARKERS:
        if latin_chars <= 0:
            return f"翻译任务缺少{target_label}内容"

        # 非中文目标语下，若中文明显多于目标语内容，通常是回显原文或夹带解释。
        if cjk_chars >= 4 and cjk_chars > latin_chars:
            return "翻译结果疑似仍以中文或提示词回显为主"

        if _translation_meta_response_detected(answer_text):
            return "翻译结果疑似未执行而是回显指令"

        marker_pattern = TRANSLATION_LANGUAGE_MARKERS[target_label]
        # 对较长输出要求至少出现一些目标语言痕迹，防止英文/杂糅文本混充德语/西语等。
        if len(latin_tokens) >= 4 and marker_pattern.search(answer_text) is None:
            return f"翻译结果缺少明显{target_label}语言特征"

        return None

    if target_label == "中文":
        if cjk_chars <= 0:
            return "翻译任务缺少中文内容"
        if cjk_chars < 2 and latin_chars > cjk_chars * 3:
            return "翻译结果疑似仍以外文为主"
        return None

    if target_label in {"日语", "韩语", "俄语", "阿拉伯语"}:
        pattern = {
            "日语": _HAS_JAPANESE_TEXT_PATTERN,
            "韩语": _HAS_KOREAN_TEXT_PATTERN,
            "俄语": _HAS_CYRILLIC_TEXT_PATTERN,
            "阿拉伯语": _HAS_ARABIC_TEXT_PATTERN,
        }[target_label]
        if pattern.search(answer_text) is None:
            return f"翻译任务缺少{target_label}内容"
        if target_label != "日语" and cjk_chars >= 4:
            return "翻译结果疑似仍以中文或提示词回显为主"
        if _translation_meta_response_detected(answer_text):
            return "翻译结果疑似未执行而是回显指令"
        return None

    return None


def evaluate_answer_quality(prompt: str, answer: str) -> Tuple[bool, str]:
    """
    零参数轻量级质量评估器。

    返回：
    - (True, "OK"): 合格
    - (False, "原因"): 不合格
    """
    prompt_text = (prompt or "").strip()
    answer_text = (answer or "").strip()

    if not answer_text:
        return False, "空回复"

    translation_target = _infer_translation_target(prompt_text)

    # 1) 复读机检测
    if _detect_severe_repetition(answer_text):
        return False, "复读机输出"

    # 2) 长问短答（并排除简单确认词）
    if translation_target is None and len(prompt_text) > 20 and len(answer_text) < 10:
        if answer_text.lower() not in ACK_WORDS and answer_text not in ACK_WORDS:
            return False, "长问短答且疑似敷衍"

    # 3) 任务格式匹配
    prompt_lower = prompt_text.lower()
    if any(key in prompt_lower for key in ["代码", "实现", "python", "c++"]):
        if "```" not in answer_text:
            return False, "代码任务缺少代码块"

    if translation_target is not None:
        target_label, target_pattern = translation_target
        if target_pattern.search(answer_text) is None:
            return False, f"翻译任务缺少{target_label}内容"
        translation_reason = _validate_translation_target_content(target_label, answer_text)
        if translation_reason is not None:
            return False, translation_reason

    # 4) 拒答检测
    if any(pattern in answer_text for pattern in REFUSAL_PATTERNS):
        return False, "模型拒答"

    return True, "OK"


def call_cloud_api(prompt: str) -> Tuple[str, float, float]:
    """
    Mock 云端 API 调用。

    返回：
    - answer: 云端高质量兜底回复
    - latency: 固定网络延迟（秒）
    - tps: -1（云端不计入本地算力 TPS）

    关键约束：
    - 云端调用不应更新本地设备状态（不扣电、不升温）。
    """
    _ = prompt  # 预留参数，便于后续替换为真实云端请求
    return CLOUD_FALLBACK_TEXT, CLOUD_FALLBACK_LATENCY, -1.0


class AdaptiveArbitrator:
    """
    动态仲裁器：
    - 难度预测：由 RF/MLP 路由引擎给出 difficulty。
    - 安全决策：由 battery + temperature 计算 Safety Score。
    - 质量闭环：低质量回复触发向上级联重试。
    """

    def __init__(
        self,
        device_simulator: Optional[DeviceSimulator] = None,
        routing_engine: str = "rf",
    ) -> None:
        self.simulator = device_simulator or DeviceSimulator()
        # 兼容旧代码可能引用 self.device 的场景。
        self.device = self.simulator
        self.routing_engine = routing_engine

    @staticmethod
    def _calculate_safety_score(battery: float, temp: float) -> float:
        """
        Safety Score：
        score = (battery * 0.6) - ((temp - 25) * 1.5)
        """
        return (battery * 0.6) - ((temp - 25.0) * 1.5)

    def _get_safety_score(self) -> float:
        state = self.simulator.get_state()
        return self._calculate_safety_score(float(state["battery"]), float(state["temperature"]))

    @staticmethod
    def _next_model(model_name: str) -> str:
        """返回下一个更大模型；若已最大则返回自身。"""
        idx = MODEL_ORDER.index(model_name)
        if idx >= len(MODEL_ORDER) - 1:
            return model_name
        return MODEL_ORDER[idx + 1]

    def _max_model_for_request(self, difficulty: int, score: float) -> str:
        """
        根据任务难度与安全分，得到“当前允许的最大模型”。

        规则：
        - 全局极限保护：score < -10 -> 仅 qwen-0.5b
        - Level 3：score > 20 才允许 qwen-1.5b，否则最多 llama-1b
        - Level 2：score > 0 才允许 llama-1b，否则最多 qwen-0.5b
        """
        if score < -10:
            return "qwen-0.5b"

        if difficulty >= 3:
            return "qwen-1.5b" if score > 20 else "llama-1b"

        if difficulty == 2:
            if score > 20:
                return "qwen-1.5b"
            return "llama-1b" if score > 0 else "qwen-0.5b"

        # Level 1：默认 0.5b，若安全分较高可允许后续升级补救。
        if score > 20:
            return "qwen-1.5b"
        if score > 0:
            return "llama-1b"
        return "qwen-0.5b"

    def _select_initial_model(self, difficulty: int, score: float, chain_logs: List[str]) -> str:
        """
        初始路由（含前置降级）。
        """
        if score < -10:
            chain_logs.append(f"[受限降级]score={score:.2f},强制:qwen-0.5b")
            return "qwen-0.5b"

        if difficulty >= 3:
            if score > 20:
                return "qwen-1.5b"
            chain_logs.append(f"[受限降级]score={score:.2f},跳过:qwen-1.5b")
            return "llama-1b"

        if difficulty == 2:
            if score > 0:
                return "llama-1b"
            chain_logs.append(f"[受限降级]score={score:.2f},跳过:llama-1b")
            return "qwen-0.5b"

        return "qwen-0.5b"

    def adaptive_process(self, prompt: str) -> Dict[str, object]:
        """
        动态级联主循环：
        1) 预测难度并进行安全分选模。
        2) 调用模型并做质量评估。
        3) 不合格则尝试向上升级重试，直到成功或受限。

        关键实现：
        - 引入基于时间的动态能耗与冷却模型。
        - 每次调用后立刻用耗时回写仿真状态（含异常/被拦截场景）。
        """
        # 难度预测改为“异构级联路由”，返回 (难度, 决策引擎, 判定流程)。
        difficulty, decision_engine, decision_flow = predict_difficulty_with_trace(prompt)
        chain_logs: List[str] = []

        total_latency = 0.0
        tps_values: List[float] = []
        final_answer = ""

        initial_score = self._get_safety_score()
        current_model = self._select_initial_model(difficulty, initial_score, chain_logs)
        chain_logs.append(f"[初始路由]{current_model}(score={initial_score:.2f})")

        while True:
            chain_logs.append(f"[调用]{current_model}")

            api_failed = False
            failure_reason = ""
            answer = ""
            latency = 0.0
            tps = 0.0

            try:
                answer, latency, tps = call_llm(current_model, prompt)
            except LLMAPIError as exc:
                api_failed = True
                failure_reason = f"调用异常:{exc}"
                answer = f"[模型调用异常]{exc}"
                latency = float(getattr(exc, "latency", 0.0))
                tps = 0.0
            except Exception as exc:  # 防御性兜底
                api_failed = True
                failure_reason = f"未知异常:{exc}"
                answer = f"[模型调用未知异常]{exc}"
                latency = 0.0
                tps = 0.0

            elapsed = max(0.0, float(latency))

            # 状态闭环回写：只要消耗了时间，就立刻更新能耗与温度。
            # 引入基于时间的动态能耗与冷却模型。
            self.simulator.update_state(current_model, elapsed)
            chain_logs.append(f"[状态回写]{current_model}|t={elapsed:.3f}s")

            total_latency += elapsed
            if tps > 0:
                tps_values.append(tps)

            if api_failed:
                quality_ok, quality_reason = False, failure_reason
            else:
                quality_ok, quality_reason = evaluate_answer_quality(prompt, answer)

            if quality_ok:
                final_answer = (answer or "").strip()
                chain_logs.append("[质量通过]")
                break

            chain_logs.append(f"{current_model} [质量拦截:{quality_reason}]")

            # 根据最新状态动态判断是否还能升级。
            now_score = self._get_safety_score()
            max_model = self._max_model_for_request(difficulty, now_score)
            next_model = self._next_model(current_model)
            can_upgrade = MODEL_ORDER.index(next_model) <= MODEL_ORDER.index(max_model) and next_model != current_model

            if can_upgrade:
                chain_logs.append(f"[升级重试]{current_model}->{next_model}(score={now_score:.2f})")
                current_model = next_model
                continue

            # 触发云端兜底条件：
            # 1) 已到端侧最大模型（如 1.5b）仍不合格；
            # 2) 当前安全分限制导致无法继续升级。
            if current_model == "qwen-1.5b":
                chain_logs.append("[受限/妥协]端侧最高模型仍不达标")
            else:
                chain_logs.append(f"[受限/妥协]score={now_score:.2f},无法升级到:{next_model}")

            chain_logs.append("[云端兜底卸载]")
            cloud_answer, cloud_latency, cloud_tps = call_cloud_api(prompt)
            total_latency += max(0.0, float(cloud_latency))
            if cloud_tps > 0:
                tps_values.append(float(cloud_tps))
            final_answer = cloud_answer.strip()
            chain_logs.append("Cloud-API [成功]")
            break

        avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0.0
        final_score = self._get_safety_score()
        chain_text = " -> ".join(chain_logs)

        return {
            "answer": final_answer,
            "total_latency": round(total_latency, 4),
            "avg_tps": round(avg_tps, 4),
            "call_chain": chain_logs,  # 兼容旧调用方（列表）
            "call_chain_text": chain_text,  # 新增：直接可写入 CSV 的链路文本
            "difficulty": difficulty,
            "decision_engine": decision_engine,
            "decision_flow": decision_flow,
            "safety_score": round(final_score, 4),
        }
