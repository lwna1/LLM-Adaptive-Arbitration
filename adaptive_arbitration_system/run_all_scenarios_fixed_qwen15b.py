"""
run_all_scenarios_fixed_qwen15b.py
==================================
固定模型基线跑批脚本（不经过仲裁器）。

目标：
1. 复用原有四大场景题库（A/B/C/D，各 40 条）。
2. 每道题直接调用固定模型 qwen-1.5b。
3. 输出与 benchmark 可对比的关键指标（耗时/TPS/电量/温度/链路/回复）。

说明：
- 本脚本不使用 ML 难度分类，不输出“系统预估难度 / 仲裁引擎 / 判定流程”等列。
- 本脚本不进行模型仲裁，真实调用链路固定为“[固定直连]qwen-1.5b”。
"""

from __future__ import annotations

import os
import random
from typing import Dict, List

import pandas as pd

from config_and_api import LLMAPIError, call_llm
from device_simulator import DeviceSimulator
from run_all_scenarios_benchmark import RANDOM_SEED, SCENARIO_DIRS, build_all_scenarios


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")
FIXED_MODEL_NAME = "qwen-1.5b"
OUTPUT_FILE_NAME = "results_qwen15b_direct.csv"


def _calc_safety_score(battery: float, temp: float) -> float:
    """
    与主系统一致的安全分公式，便于横向对比。
    """
    return (battery * 0.6) - ((temp - 25.0) * 1.5)


def run_single_scenario_fixed_model(
    question_bank: List[Dict[str, object]],
    scenario_dir: str,
    model_name: str = FIXED_MODEL_NAME,
) -> str:
    """
    在单个场景中使用固定模型直连跑批（不经过仲裁器）。
    """
    simulator = DeviceSimulator(battery=100.0, temperature=35.0)

    records: List[Dict[str, object]] = []
    total = len(question_bank)
    scenario_name = os.path.basename(scenario_dir)

    print(f"[INFO] 场景={scenario_name} | 固定模型={model_name} | 题目数={total}")

    for idx, item in enumerate(question_bank, start=1):
        prompt = str(item["prompt"])
        gt = int(item["ground_truth"])

        answer = ""
        latency = 0.0
        tps = 0.0
        chain_text = f"[固定直连]{model_name}"

        try:
            answer, latency, tps = call_llm(model_name, prompt)
        except LLMAPIError as exc:
            latency = float(getattr(exc, "latency", 0.0))
            tps = 0.0
            answer = f"[模型调用异常]{exc}"
            chain_text += " -> [调用异常]"
        except Exception as exc:  # 防御性兜底
            latency = 0.0
            tps = 0.0
            answer = f"[模型调用未知异常]{exc}"
            chain_text += " -> [未知异常]"

        elapsed = max(0.0, float(latency))

        # 与主 benchmark 一致：每次请求后回写设备状态并追加间隙冷却。
        simulator.update_state(model_name, elapsed)
        simulator.cool_down()
        state = simulator.get_state()

        clean_answer = str(answer).replace("\n", " ").replace("\r", " ").strip()
        safety_score = _calc_safety_score(float(state["battery"]), float(state["temperature"]))

        records.append(
            {
                "题号": item["id"],
                "输入Prompt": prompt,
                "真实难度标签": gt,
                "真实调用链路": chain_text,
                "最终总耗时": round(elapsed, 4),
                "平均TPS": round(float(tps), 4),
                "安全得分": round(float(safety_score), 4),
                "执行后电量": float(state["battery"]),
                "执行后温度": float(state["temperature"]),
                "最终模型回复": clean_answer,
            }
        )

        print(
            f"  [{idx:02d}/{total}] 真值={gt} | 链路={chain_text} | "
            f"耗时={elapsed:.3f}s | TPS={float(tps):.4f} | "
            f"电量={state['battery']:.2f}% | 温度={state['temperature']:.2f}°C"
        )

    out_csv = os.path.join(scenario_dir, OUTPUT_FILE_NAME)
    pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存：{out_csv}")
    return out_csv


def main() -> None:
    random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    scenarios = build_all_scenarios()
    summary_rows: List[Dict[str, object]] = []

    for key, qbank in scenarios.items():
        scenario_name = SCENARIO_DIRS[key]
        scenario_dir = os.path.join(OUTPUT_ROOT, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        out_csv = run_single_scenario_fixed_model(qbank, scenario_dir, model_name=FIXED_MODEL_NAME)
        df = pd.read_csv(out_csv, encoding="utf-8-sig")

        total_time = float(df["最终总耗时"].sum()) if len(df) else 0.0
        avg_tps = float(df["平均TPS"].mean()) if len(df) else 0.0
        final_battery = float(df["执行后电量"].iloc[-1]) if len(df) else 100.0
        total_battery_drop = max(0.0, 100.0 - final_battery)
        max_temp = float(df["执行后温度"].max()) if len(df) else 0.0

        summary_rows.append(
            {
                "场景": scenario_name,
                "题目数": int(len(df)),
                "总耗时(秒)": round(total_time, 4),
                "平均TPS": round(avg_tps, 4),
                "总耗电(%)": round(total_battery_drop, 4),
                "最高温度(°C)": round(max_temp, 2),
            }
        )

    print("\n=== 固定模型 qwen-1.5b 场景跑批完成 ===")
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

