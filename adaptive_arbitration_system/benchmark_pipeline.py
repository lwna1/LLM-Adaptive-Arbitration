"""
benchmark_pipeline.py
---------------------
本模块用于执行批量实验并导出论文数据：
1. 构建 20 条题库（含真实难度标签）。
2. 逐题调用 adaptive_process()。
3. 每题后调用 cool_down()。
4. 收集关键指标并导出 CSV。
5. 使用混合级联路由（Rule / MLP / RF）执行单次跑批。
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List

from arbitrator_core import AdaptiveArbitrator
from device_simulator import DeviceSimulator


def build_question_bank() -> List[Dict[str, object]]:
    """
    构建实验题库（20 条）。

    结构说明：
    - 每条样本为字典：{"id": "Q01", "prompt": "...", "ground_truth": 1/2/3}
    - ground_truth 由人工依据任务认知负荷标注，供混淆矩阵分析使用。

    覆盖类型：
    - Level 1：极简指令/常识
    - Level 2：中等解释/翻译
    - Level 3：算法、证明、代码与复杂逻辑
    """
    return [
        {"id": "Q01", "prompt": "请回复收到", "ground_truth": 1},
        {"id": "Q02", "prompt": "你好", "ground_truth": 1},
        {"id": "Q03", "prompt": "1+1等于几", "ground_truth": 1},
        {"id": "Q04", "prompt": "中国的首都是哪里？", "ground_truth": 1},
        {"id": "Q05", "prompt": "一年有多少天？", "ground_truth": 1},
        {"id": "Q06", "prompt": "请简要介绍一下长城的历史意义", "ground_truth": 2},
        {"id": "Q07", "prompt": "请将这句话翻译成英语：人工智能正在改变世界", "ground_truth": 2},
        {"id": "Q08", "prompt": "什么是操作系统，请用通俗语言总结", "ground_truth": 2},
        {"id": "Q09", "prompt": "请简述牛顿第一定律", "ground_truth": 2},
        {
            "id": "Q10",
            "prompt": "请解释一下机器学习和深度学习的区别，并给出一个生活中的例子",
            "ground_truth": 2,
        },
        {"id": "Q11", "prompt": "为什么天空是蓝色的？请从光散射原理角度说明", "ground_truth": 3},
        {"id": "Q12", "prompt": "请比较栈和队列的逻辑差异，并给出典型应用场景", "ground_truth": 3},
        {"id": "Q13", "prompt": "请解释一下 TCP 三次握手的过程和设计原因", "ground_truth": 3},
        {"id": "Q14", "prompt": "请从算法复杂度角度，比较冒泡排序与快速排序", "ground_truth": 3},
        {
            "id": "Q15",
            "prompt": "请用Python写一个快速排序算法，并解释每一步的原理",
            "ground_truth": 3,
        },
        {
            "id": "Q16",
            "prompt": "请证明二分查找的时间复杂度是 O(log n)，并说明推导过程",
            "ground_truth": 3,
        },
        {
            "id": "Q17",
            "prompt": "请阅读下面代码并解释可能的问题：def foo(x):\n    if x > 0:\n        return x\n    else:\n        return None\n然后说明如何改进异常处理逻辑",
            "ground_truth": 3,
        },
        {
            "id": "Q18",
            "prompt": "请比较 Transformer 和 RNN 在长序列建模上的优劣，并解释为什么注意力机制更适合并行计算",
            "ground_truth": 3,
        },
        {
            "id": "Q19",
            "prompt": "以下是一个较长任务：请从系统架构角度解释微服务拆分原则、服务发现机制、配置中心设计、链路追踪、熔断降级策略、幂等控制和分布式事务处理，并总结你认为最关键的工程权衡。",
            "ground_truth": 3,
        },
        {
            "id": "Q20",
            "prompt": "请编写一个包含插入、删除、查找操作的哈希表实现，并比较链地址法和开放寻址法的优缺点，同时解释在高负载因子下性能退化的原因。",
            "ground_truth": 3,
        },
    ]


def export_results_to_csv(rows: List[Dict[str, object]], output_path: str) -> None:
    """
    将实验结果导出到 CSV。

    字段定义：
    - 题号
    - 输入Prompt
    - 系统预估难度
    - 仲裁引擎
    - 判定流程
    - 真实难度标签
    - 真实调用链路
    - 最终总耗时
    - 平均TPS
    - 执行后电量
    - 执行后温度
    - 最终模型回复
    """
    fieldnames = [
        "题号",
        "输入Prompt",
        "系统预估难度",
        "仲裁引擎",
        "判定流程",
        "真实难度标签",
        "真实调用链路",
        "最终总耗时",
        "平均TPS",
        "执行后电量",
        "执行后温度",
        "最终模型回复",
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    output_csv_name: str = "arbitration_results_hybrid.csv",
    routing_engine: str = "hybrid",
) -> List[Dict[str, object]]:
    """
    执行完整的批量实验流程。

    参数：
    - output_csv_name: 导出文件名
    - routing_engine: 路由引擎类型，当前建议使用 'hybrid'

    返回：
    - 包含每条样本统计数据的列表（同时也会写入 CSV）。
    """
    # 每次 run 都重新初始化设备状态（满电+基础温度），保证实验公平。
    simulator = DeviceSimulator(battery=100.0, temperature=35.0)
    arbitrator = AdaptiveArbitrator(simulator, routing_engine=routing_engine)

    question_bank = build_question_bank()
    total_count = len(question_bank)

    print(f"=== 开始执行批量测试 | engine={routing_engine} ===")
    print(f"题目总数：{total_count}")

    records: List[Dict[str, object]] = []

    for index, item in enumerate(question_bank, start=1):
        prompt = str(item["prompt"])
        ground_truth = int(item["ground_truth"])

        result = arbitrator.adaptive_process(prompt)

        # 正确接收并展开仲裁结果，便于后续统计与导出。
        answer = str(result.get("answer", ""))
        total_latency = float(result.get("total_latency", 0.0))
        avg_tps = float(result.get("avg_tps", 0.0))
        call_chain = result.get("call_chain", [])
        difficulty = int(result.get("difficulty", 1))
        decision_engine = str(result.get("decision_engine", "Unknown"))
        decision_flow = str(result.get("decision_flow", "")).strip()

        # CSV 对换行符敏感，写入前必须清洗为单行文本。
        clean_answer = answer.replace("\n", " ").replace("\r", " ").strip()

        # 每次完整请求结束后执行冷却。
        simulator.cool_down()

        battery_after = round(simulator.battery, 2)
        temperature_after = round(simulator.temperature, 2)
        chain_str = " -> ".join(call_chain)

        record = {
            "题号": item["id"],
            "输入Prompt": prompt,
            "系统预估难度": difficulty,
            "仲裁引擎": decision_engine,
            "判定流程": decision_flow,
            "真实难度标签": ground_truth,
            "真实调用链路": chain_str,
            "最终总耗时": round(total_latency, 4),
            "平均TPS": round(avg_tps, 4),
            "执行后电量": battery_after,
            "执行后温度": temperature_after,
            "最终模型回复": clean_answer,
        }
        records.append(record)

        print(
            f"[{index:02d}/{total_count}] "
            f"真值={ground_truth} | 预测={difficulty}({decision_flow}) | 链路={chain_str} | "
            f"耗时={total_latency:.4f}s | 平均TPS={avg_tps:.4f} | "
            f"电量={battery_after:>6.2f}% | 温度={temperature_after:>5.2f}℃"
        )

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_csv_name)
    export_results_to_csv(records, output_path)

    print(f"=== 测试完成 | engine={routing_engine} ===")
    print(f"结果已导出：{output_path}")

    return records


if __name__ == "__main__":
    # 单次运行：异构级联路由（Rule -> MLP -> RF）
    run_benchmark(output_csv_name="arbitration_results_hybrid.csv", routing_engine="hybrid")
