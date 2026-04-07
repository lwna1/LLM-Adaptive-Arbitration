"""
run_all_scenarios_benchmark.py
==============================
四大场景统筹跑批脚本（替代旧 benchmark_pipeline.py）。

目标：
1. 自动构造四类场景题库（A/B/C/D）
2. 为每个场景分别执行 RF 与 MLP 双引擎跑批
3. 输出结果到严格目录规范：
   output/scenario_*/results_rf.csv
   output/scenario_*/results_mlp.csv
"""

from __future__ import annotations

import os
import random
import re
from typing import Dict, List

import pandas as pd

from arbitrator_core import AdaptiveArbitrator
from device_simulator import DeviceSimulator


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")

SCENARIO_DIRS = {
    "A": "scenario_A_light",
    "B": "scenario_B_heavy",
    "C": "scenario_C_hallucination",
    "D": "scenario_D_burst",
}

RANDOM_SEED = 42
POOL_SIZE = 40


def _extract_final_model(chain_text: str) -> str:
    matches = re.findall(r"(qwen-0\.5b|llama-1b|qwen-1\.5b)", str(chain_text))
    return matches[-1] if matches else "unknown"


def _assert_pool_size_40() -> None:
    """
    启动时校验所有池子长度，确保后续场景构建稳定可复现。
    """
    pools = {
        "simple_pool": _simple_prompts_pool(),
        "medium_pool": _medium_prompts_pool(),
        "heavy_pool": _heavy_prompts_pool(),
        "hallucination_pool": _hallucination_prompts_pool(),
    }
    for name, pool in pools.items():
        if len(pool) != POOL_SIZE:
            raise RuntimeError(f"{name} 长度异常：期望 {POOL_SIZE}，实际 {len(pool)}")


def _simple_prompts_pool() -> List[str]:
    return [
        "请回复收到",
        "你好",
        "1+1等于几",
        "2+3等于几",
        "3+5等于几",
        "今天星期几",
        "现在几点",
        "请说早上好",
        "请说晚上好",
        "请输出数字7",
        "请输出数字9",
        "北京是中国首都吗",
        "上海在中国吗",
        "苹果是一种水果吗",
        "水会结冰吗",
        "太阳从东边升起吗",
        "请回答：地球是圆的吗",
        "请回答：火会发热吗",
        "请给我一句鼓励的话",
        "请给我一句祝福",
        "请说一句自我介绍",
        "请说一句问候语",
        "请回复：已收到",
        "请告诉我1分钟有多少秒",
        "请告诉我一年有几个月",
        "请问猫会喵喵叫吗",
        "请问鱼生活在水里吗",
        "请问1米等于100厘米吗",
        "请问中国的国庆节在几月",
        "请问红绿灯有几种颜色",
        "请回答：是晴天吗",
        "请回答：你在线吗",
        "请输出OK",
        "请输出收到",
        "请用一个词形容春天",
        "请用一个词形容夏天",
        "请用一个词形容秋天",
        "请用一个词形容冬天",
        "请回答：2是偶数吗",
        "请回答：9是奇数吗",
    ]


def _medium_prompts_pool() -> List[str]:
    return [
        "请解释一下牛顿第一定律，并给一个生活例子。",
        "请将这句话翻译成英语：人工智能正在改变社会。",
        "请总结云计算的三个核心特点。",
        "请简述数据库索引的作用。",
        "请介绍操作系统中进程与线程的区别。",
        "请解释什么是机器学习中的过拟合。",
        "请总结 TCP 三次握手的主要流程。",
        "请将这句话翻译成英语：边缘计算可以降低时延。",
        "请说明哈希表查找效率高的原因。",
        "请解释面向对象编程的封装思想。",
        "请简述什么是 RESTful API。",
        "请总结深度学习与传统机器学习的主要差异。",
        "请介绍消息队列在系统中的常见作用。",
        "请解释数据库事务的 ACID 特性。",
        "请将这句话翻译成英语：数据一致性非常重要。",
        "请简述负载均衡的常见策略。",
        "请解释 HTTP 和 HTTPS 的核心区别，并说明常见应用场景。",
        "请总结 DNS 域名解析的主要流程。",
        "请介绍缓存与数据库一致性的基本处理思路。",
        "请说明二叉树前序、中序、后序遍历的差异。",
        "请将这句话翻译成英语：The experiment is reproducible.",
        "请简述 Linux 文件权限的基本含义。",
        "请解释对象存储与块存储的主要区别。",
        "请简要说明 CAP 理论的核心思想。",
        "请将这句话翻译成英语：Latency is critical for edge devices.",
        "请总结微服务的主要优点与挑战。",
        "请解释 JWT 的基本原理与使用场景。",
        "请说明垃圾回收机制在程序中的作用。",
        "请将这句话翻译成英语：Model routing should balance quality and cost.",
        "请总结 SQL 与 NoSQL 数据库的关键区别。",
        "请解释线程池技术的优势与注意点。",
        "请简述虚拟内存机制的作用。",
        "请将这句话翻译成英语：This system supports dynamic degradation.",
        "请总结 CDN 在互联网系统中的核心价值。",
        "请解释 API 幂等性的概念与工程意义。",
        "请说明队列与栈在数据结构上的区别。",
        "请将这句话翻译成英语：Please keep technical terms unchanged.",
        "请总结可观测性的三大支柱。",
        "请解释重试机制与熔断机制之间的关系。",
        "请将这句话翻译成英语：We need a robust benchmark pipeline.",
    ]


def _heavy_prompts_pool() -> List[str]:
    return [
        "请从算法复杂度角度，比较快速排序、归并排序与堆排序，并分析在不同输入分布下的性能差异，给出详细推导过程。",
        "请证明二分查找在有序数组中的时间复杂度为 O(log n)，并补充递归与迭代实现的边界条件分析。",
        "请设计一个支持高并发读写的缓存系统，讨论一致性、淘汰策略、异常处理、限流熔断与可观测性架构。",
        "请写出 Python 版本的图最短路算法，并比较 Dijkstra 与 Bellman-Ford 在负权边场景的区别与复杂度。",
        "请从微服务架构角度，详细说明服务发现、配置中心、链路追踪、分布式事务与幂等控制的工程权衡。",
        "请阅读以下逻辑并指出潜在 bug：if (x>0) {return f(x);} else {while(true){}}，并给出可证明终止的重构方案。",
        "请比较 Transformer、RNN、LSTM 的长序列建模能力，结合注意力机制原理解释并行优势和内存代价。",
        "请从数学角度推导交叉熵损失与 softmax 梯度，给出每一步公式变形过程，并解释数值稳定性处理。",
        "请基于 CAP 理论分析分布式数据库在一致性与可用性之间的取舍，并给出你推荐的工程实践。",
        "请实现一个支持回滚的并查集结构，并分析其时间复杂度与空间复杂度。",
        "请设计一个高可靠日志采集系统，讨论批处理、背压、重试、去重与最终一致性。",
        "请证明动态规划求解最长公共子序列的状态转移方程正确，并给出边界初始化说明。",
        "请分析微服务熔断降级策略在雪崩场景下的作用，并比较两种实现方案的优缺点。",
        "请用 Python 实现 A* 搜索算法，并说明启发函数不一致时可能导致的问题。",
        "请对比 Raft 与 Paxos 在工程可实现性上的差异，并解释日志复制的关键步骤。",
        "请从内存模型角度解释多线程可见性问题，并结合示例分析 happens-before 关系。",
        "请设计一个支持千万级请求的限流系统，比较令牌桶与漏桶算法。",
        "请实现并证明单调队列求滑动窗口最大值算法的正确性与复杂度。",
        "请分析数据库索引失效的常见原因，并给出可复现实验与优化方案。",
        "请写出最小生成树的 Kruskal 与 Prim 算法，并比较它们的适用场景。",
        "请从系统架构角度设计一个高可用订单系统，讨论分库分表、分布式事务、幂等与补偿机制。",
        "请证明拓扑排序算法在有向无环图上必然终止，并分析其时间复杂度与空间复杂度。",
        "请实现一个支持撤销操作的文本编辑器核心结构，给出数据结构设计与复杂度证明。",
        "请从一致性协议角度比较 ZAB、Raft 与 Multi-Paxos，并说明工程实现差异。",
        "请写出 Python 版 KMP 字符串匹配算法并推导 next 数组构造过程。",
        "请分析高并发下数据库连接池耗尽的根因，并给出监控、限流、扩容的完整方案。",
        "请设计一个分布式定时任务系统，考虑时钟漂移、重复执行、故障恢复与可观测性。",
        "请证明单源最短路径在存在负环时无法得到有意义解，并给出检测策略。",
        "请比较 B+Tree 与 LSM-Tree 在读写放大、压缩策略、范围查询上的工程权衡。",
        "请写出并证明线段树区间更新与区间查询算法的正确性。",
        "请从编译原理角度解释词法分析、语法分析、语义分析三阶段的职责与联系。",
        "请设计一个模型路由系统的在线评估机制，要求覆盖准确率、时延、能耗、温升四维指标。",
        "请实现并推导并行归并排序在多核场景下的复杂度，并讨论线程调度开销。",
        "请分析消息队列积压导致系统雪崩的传播路径，并提出分层限流与降级方案。",
        "请从概率图模型角度解释隐马尔可夫模型，并给出维特比算法推导。",
        "请证明红黑树插入调整后仍满足所有性质，并给出旋转操作的不变式说明。",
        "请设计一个端侧推理缓存系统，要求考虑命中率、过期策略、内存碎片与淘汰公平性。",
        "请写出 Floyd-Warshall 算法并解释其三重循环的动态规划含义。",
        "请对比同步复制与异步复制在数据库高可用中的延迟与一致性风险。",
        "请实现一个支持回滚与快照的并发键值存储，并分析锁粒度对吞吐的影响。",
    ]


def _hallucination_prompts_pool() -> List[str]:
    """
    场景C：特意构造“容易触发质量拦截”的提示词。
    """
    return [
        "请用 python 代码实现快速排序，必须放在```代码块```里，并解释复杂度。",
        "请翻译成英语：我今天在图书馆学习了机器学习，并输出英文结果。",
        "请用 c++ 实现并查集，要求给出完整代码块和注释。",
        "请翻译成英语：请保持句意准确，不要遗漏任何关键词。",
        "请用 python 代码实现二分查找，必须使用```代码块```。",
        "请翻译成英语：端侧部署需要平衡时延与能耗。",
        "请实现一个 LRU 缓存，使用代码块输出并说明设计思路。",
        "请翻译成英语：我们需要一个稳定可复现的实验流程。",
        "请用 python 实现二叉树层序遍历，必须给出```代码块```并附复杂度分析。",
        "请翻译成英语：系统在高并发下需要保持鲁棒性。",
        "请用 c++ 实现最短路径算法，必须输出在```代码块```中。",
        "请翻译成英语：请务必保留术语的准确性与完整性。",
        "请实现哈希表并处理冲突，要求输出完整```代码块```。",
        "请翻译成英语：论文实验需要严格遵循控制变量法。",
        "请给出 Python 版并查集实现，必须使用```代码块```。",
        "请翻译成英语：模型需要在端侧设备上稳定运行。",
        "请输出 C++ 版快速幂实现，必须放进```代码块```。",
        "请翻译成英语：请保持语义一致，不要自由发挥。",
        "请用 python 写一个 BFS，必须使用```代码块```并解释每行作用。",
        "请翻译成英语：我们需要自动化可视化评测流水线。",
        "请用 python 实现 Dijkstra 算法，必须输出```代码块```，不要省略 import。",
        "请翻译成英语：系统需要在端侧设备上稳定低延迟运行。",
        "请用 c++ 实现拓扑排序，必须放到```代码块```里并附注释。",
        "请翻译成英语：请不要改变原句的技术术语。",
        "请实现 Python 版并发队列，必须使用```代码块```输出完整实现。",
        "请翻译成英语：我们需要对模型进行公平对比实验。",
        "请用 c++ 写一个最小堆实现，必须输出```代码块```和测试样例。",
        "请翻译成英语：硬件温度上升会触发动态降级策略。",
        "请用 python 给出 LRU 缓存类，必须使用```代码块```并说明时间复杂度。",
        "请翻译成英语：请确保译文语法正确且自然。",
        "请实现 C++ 版二分查找，必须放入```代码块```，并解释边界处理。",
        "请翻译成英语：论文图表需要统一字体和分辨率。",
        "请用 python 实现 KMP 算法，必须给出```代码块```和 next 数组构造说明。",
        "请翻译成英语：请在译文中保留关键术语原意。",
        "请写一个 c++ 版单例模式实现，必须使用```代码块```并说明线程安全。",
        "请翻译成英语：系统必须支持自动化场景回归测试。",
        "请用 python 实现哈希表冲突处理，必须输出```代码块```。",
        "请翻译成英语：请严格按照输入语义进行翻译。",
        "请给出 C++ 版 BFS 与 DFS，必须使用```代码块```并比较复杂度。",
        "请翻译成英语：我们正在构建高内聚低耦合的评测流水线。",
    ]


def build_scenario_a_light() -> List[Dict[str, object]]:
    """
    场景A：90% 极简短文本（轻负载）。
    共 40 条：36 条简单 + 4 条中等。
    """
    simple_pool = _simple_prompts_pool()
    medium_pool = _medium_prompts_pool()

    prompts: List[Dict[str, object]] = []
    for i in range(36):
        prompts.append(
            {
                "id": f"A{i+1:02d}",
                "prompt": simple_pool[i % len(simple_pool)],
                "ground_truth": 1,
            }
        )
    for i in range(4):
        prompts.append(
            {
                "id": f"A{36+i+1:02d}",
                "prompt": medium_pool[i % len(medium_pool)],
                "ground_truth": 2,
            }
        )
    return prompts


def build_scenario_b_heavy() -> List[Dict[str, object]]:
    """
    场景B：100% 极长复杂逻辑（重负载）。
    """
    heavy_pool = _heavy_prompts_pool()
    prompts: List[Dict[str, object]] = []
    for i in range(40):
        prompts.append(
            {
                "id": f"B{i+1:02d}",
                "prompt": heavy_pool[i % len(heavy_pool)],
                "ground_truth": 3,
            }
        )
    return prompts


def build_scenario_c_hallucination() -> List[Dict[str, object]]:
    """
    场景C：极易触发质量拦截的奇葩要求。
    主要由代码块硬约束/翻译硬约束构成。
    """
    pool = _hallucination_prompts_pool()
    prompts: List[Dict[str, object]] = []
    for i in range(40):
        text = pool[i % len(pool)]
        # 翻译类按中等标注，代码类按困难标注
        gt = 2 if "翻译成英语" in text else 3
        prompts.append(
            {
                "id": f"C{i+1:02d}",
                "prompt": text,
                "ground_truth": gt,
            }
        )
    return prompts


def build_scenario_d_burst() -> List[Dict[str, object]]:
    """
    场景D：冷热脉冲交替。
    按 5难-15简-5难-15简 组织，共 40 条。
    """
    simple_pool = _simple_prompts_pool()
    heavy_pool = _heavy_prompts_pool()

    prompts: List[Dict[str, object]] = []
    idx = 1

    # 5 难
    for i in range(5):
        prompts.append({"id": f"D{idx:02d}", "prompt": heavy_pool[i % len(heavy_pool)], "ground_truth": 3})
        idx += 1
    # 15 简
    for i in range(15):
        prompts.append({"id": f"D{idx:02d}", "prompt": simple_pool[i % len(simple_pool)], "ground_truth": 1})
        idx += 1
    # 5 难
    for i in range(5):
        prompts.append({"id": f"D{idx:02d}", "prompt": heavy_pool[(i + 3) % len(heavy_pool)], "ground_truth": 3})
        idx += 1
    # 15 简
    for i in range(15):
        prompts.append({"id": f"D{idx:02d}", "prompt": simple_pool[(i + 2) % len(simple_pool)], "ground_truth": 1})
        idx += 1

    return prompts


def build_all_scenarios() -> Dict[str, List[Dict[str, object]]]:
    return {
        "A": build_scenario_a_light(),
        "B": build_scenario_b_heavy(),
        "C": build_scenario_c_hallucination(),
        "D": build_scenario_d_burst(),
    }


def run_single_engine(
    question_bank: List[Dict[str, object]],
    engine: str,
    scenario_dir: str,
) -> str:
    """
    在单个场景下运行单个引擎（RF 或 MLP）。
    """
    simulator = DeviceSimulator(battery=100.0, temperature=35.0)
    arbitrator = AdaptiveArbitrator(device_simulator=simulator, routing_engine=engine)

    records = []
    total = len(question_bank)
    print(f"[INFO] 场景={os.path.basename(scenario_dir)} | 引擎={engine} | 题目数={total}")

    for idx, item in enumerate(question_bank, start=1):
        prompt = str(item["prompt"])
        gt = int(item["ground_truth"])

        result = arbitrator.adaptive_process(prompt)
        answer = str(result.get("answer", ""))
        call_chain_list = result.get("call_chain", [])
        call_chain_text = str(result.get("call_chain_text", ""))
        if not call_chain_text and isinstance(call_chain_list, list):
            call_chain_text = " -> ".join(str(x) for x in call_chain_list)

        total_latency = float(result.get("total_latency", 0.0))
        avg_tps = float(result.get("avg_tps", 0.0))
        pred_diff = int(result.get("difficulty", 1))
        safety_score = float(result.get("safety_score", 0.0))

        # 兼容旧逻辑：每轮请求后额外冷却一次，模拟请求间隙。
        simulator.cool_down()
        state = simulator.get_state()

        final_model = _extract_final_model(call_chain_text)
        clean_answer = answer.replace("\n", " ").replace("\r", " ").strip()

        record = {
            "题号": item["id"],
            "输入Prompt": prompt,
            "真实难度标签": gt,
            "系统预估难度": pred_diff,
            "真实调用链路": call_chain_text,
            "最终调用模型": final_model,
            "最终总耗时": round(total_latency, 4),
            "平均TPS": round(avg_tps, 4),
            "安全得分": round(safety_score, 4),
            "执行后电量": float(state["battery"]),
            "执行后温度": float(state["temperature"]),
            "最终模型回复": clean_answer,
        }
        records.append(record)

        print(
            f"  [{idx:02d}/{total}] 预测={pred_diff} 真值={gt} "
            f"耗时={total_latency:.3f}s 电量={state['battery']:.2f}% 温度={state['temperature']:.2f}°C"
        )

    out_csv = os.path.join(scenario_dir, f"results_{engine}.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存：{out_csv}")
    return out_csv


def main() -> None:
    random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    _assert_pool_size_40()

    scenarios = build_all_scenarios()

    for key, qbank in scenarios.items():
        scenario_name = SCENARIO_DIRS[key]
        scenario_dir = os.path.join(OUTPUT_ROOT, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        # 双引擎对撞：RF + MLP
        run_single_engine(qbank, engine="rf", scenario_dir=scenario_dir)
        run_single_engine(qbank, engine="mlp", scenario_dir=scenario_dir)

    print("[INFO] 四大场景跑批完成。")


if __name__ == "__main__":
    main()
