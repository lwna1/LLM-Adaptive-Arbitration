"""
config_and_api.py
-----------------
本模块负责：
1. 维护全局配置（Ollama API 地址、模型信息）。
2. 提供统一的模型调用函数 call_llm()。

说明：
- 该函数会记录请求端到端总延迟（latency）。
- 并基于 Ollama 返回的 eval_count / eval_duration 计算 TPS。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple

import requests

# 强制本地 Ollama 请求绕过任何系统代理，避免 127.0.0.1 被错误代理转发。
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

# ===== 全局配置 =====
# 直接指向 Ollama 的 generate 接口（按你的项目环境固定为本地服务）。
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"

# 模型注册表：记录模型名称与其参数量/能耗等级。
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen-0.5b": {"params_b": 0.5, "energy_level": 1},
    "llama-1b": {"params_b": 1.0, "energy_level": 2},
    "qwen-1.5b": {"params_b": 1.5, "energy_level": 3},
}

# 默认超时时间（秒）。
# 保持 requests.post 的 timeout=120 设定。
DEFAULT_TIMEOUT = 120


class LLMAPIError(RuntimeError):
    """封装底层接口调用异常，便于上层仲裁模块统一处理。"""

    def __init__(self, message: str, latency: float = 0.0) -> None:
        super().__init__(message)
        self.latency = latency


def call_llm(model_name: str, prompt: str, timeout: int = DEFAULT_TIMEOUT) -> Tuple[str, float, float]:
    """
    调用指定模型并返回结果。

    参数：
    - model_name: 目标模型名称（必须在 MODEL_REGISTRY 中）。
    - prompt: 输入提示词。
    - timeout: 请求超时阈值，默认 120 秒。

    返回：
    - response_text: 模型回复文本。
    - latency: 请求总延迟（秒）。
    - tps: Tokens Per Second（基于 eval_count/eval_duration 计算）。

    异常：
    - LLMAPIError: 网络、超时、状态码、JSON 解析等异常统一转译后抛出。
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型：{model_name}，可用模型：{list(MODEL_REGISTRY.keys())}")

    # 给操作系统和模型调度预留更充足的喘息时间，降低高频切换场景下 502 风险。
    time.sleep(1.5)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }

    start_time = time.perf_counter()
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
        # 若 HTTP 状态码非 2xx，会抛出 HTTPError。
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout as exc:
        end_time = time.perf_counter()
        latency = end_time - start_time
        raise LLMAPIError(f"调用模型 {model_name} 超时（{latency:.2f}s）", latency=latency) from exc
    except requests.exceptions.HTTPError as exc:
        end_time = time.perf_counter()
        latency = end_time - start_time
        status_code = getattr(exc.response, "status_code", "unknown")
        raise LLMAPIError(
            f"调用模型 {model_name} 返回非成功状态码：{status_code}（耗时 {latency:.2f}s）",
            latency=latency,
        ) from exc
    except requests.exceptions.RequestException as exc:
        end_time = time.perf_counter()
        latency = end_time - start_time
        raise LLMAPIError(
            f"调用模型 {model_name} 网络异常（耗时 {latency:.2f}s）：{exc}",
            latency=latency,
        ) from exc
    except ValueError as exc:
        # response.json() 解析失败通常会触发 ValueError。
        end_time = time.perf_counter()
        latency = end_time - start_time
        raise LLMAPIError(
            f"调用模型 {model_name} 的响应 JSON 解析失败（耗时 {latency:.2f}s）",
            latency=latency,
        ) from exc
    except Exception as exc:  # 防御性兜底，避免异常漏出导致管线中断。
        end_time = time.perf_counter()
        latency = end_time - start_time
        raise LLMAPIError(
            f"调用模型 {model_name} 出现未知异常（耗时 {latency:.2f}s）：{exc}",
            latency=latency,
        ) from exc

    end_time = time.perf_counter()
    latency = end_time - start_time

    # 响应文本：若字段缺失，按空字符串兜底。
    response_text = str(data.get("response", "")).strip()

    # Ollama 典型字段：
    # eval_count: 生成 token 数
    # eval_duration: 生成耗时（纳秒）
    try:
        eval_count = int(data.get("eval_count", 0) or 0)
        eval_duration_ns = int(data.get("eval_duration", 0) or 0)
    except (TypeError, ValueError):
        # 若字段异常，不阻塞主流程，TPS 记为 0.0。
        eval_count = 0
        eval_duration_ns = 0

    # TPS 计算：token / 秒。
    # eval_duration 单位为 ns，因此需要除以 1e9 转换到秒。
    if eval_count > 0 and eval_duration_ns > 0:
        tps = eval_count / (eval_duration_ns / 1_000_000_000)
    else:
        tps = 0.0

    return response_text, latency, tps
