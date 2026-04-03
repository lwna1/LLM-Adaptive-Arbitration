"""
train_router.py
===============
本脚本负责构建“难度路由器”的离线训练流水线：
1. 程序化生成约 150 条模拟数据（Mock Dataset）。
2. 调用 feature_engineering.extract_features() 提取 6D 特征。
3. 训练 RandomForestClassifier（50 棵树，小模型）。
4. 输出 Accuracy / Classification Report / Feature Importance。
5. 将模型与标准化器打包保存为 router_model.pkl。

可直接运行：
    python3 train_router.py
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Dict, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 兼容两种运行方式：
# 1) 在父目录中作为包执行：python3 -m ml_router_upgrade.train_router
# 2) 在当前目录直接执行：python3 train_router.py
try:
    from .feature_engineering import FEATURE_NAMES, extract_features
except ImportError:  # pragma: no cover - 脚本直跑兜底
    from feature_engineering import FEATURE_NAMES, extract_features

# 固定随机种子，保证论文实验可复现。
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def _build_level_1_samples(n: int = 50) -> List[Dict[str, object]]:
    """
    Level 1（简单）样本：
    - 极简问候、指令、常识
    - 保持短文本、低认知负荷
    """
    base_prompts = [
        "你好",
        "请回复收到",
        "1+1等于几",
        "今天天气怎么样",
        "早上好",
        "晚安",
        "谢谢",
        "你是谁",
        "现在几点",
        "再见",
        "吃饭了吗",
        "你在吗",
        "帮我打个招呼",
        "中国的首都是哪里",
        "一年有多少天",
    ]
    suffixes = ["", "。", "!", "？", "，谢谢"]

    samples: List[Dict[str, object]] = []
    for i in range(n):
        root = base_prompts[i % len(base_prompts)]
        suffix = suffixes[(i * 2) % len(suffixes)]
        samples.append({"prompt": f"{root}{suffix}", "label": 1})
    return samples


def _build_level_2_samples(n: int = 50) -> List[Dict[str, object]]:
    """
    Level 2（中等）样本：
    - 中等长度解释、总结、翻译类任务
    - 保留专业性但控制推理深度
    """
    topics = [
        "量子力学的历史背景",
        "文艺复兴对欧洲文化的影响",
        "数据库事务的基本概念",
        "操作系统进程与线程的区别",
        "机器学习中的过拟合问题",
        "英语写作中的被动语态",
        "中国古代丝绸之路的文化交流",
        "区块链共识机制的核心思想",
        "牛顿第一定律",
        "计算机网络分层模型",
    ]

    templates = [
        "请解释一下{topic}，并给出一个通俗例子。",
        "请总结{topic}，要求条理清晰、适合初学者阅读。",
        "请将关于{topic}的一段说明翻译成英语，并解释关键词。",
        "请从定义和应用两个角度介绍{topic}。",
        "请把{topic}写成一段 120 字左右的课堂笔记。",
    ]

    samples: List[Dict[str, object]] = []
    for i in range(n):
        topic = topics[i % len(topics)]
        template = templates[(i * 3 + 1) % len(templates)]
        prompt = template.format(topic=topic)

        if i % 2 == 0:
            prompt += " 输出时先给概念，再给示例。"

        samples.append({"prompt": prompt, "label": 2})

    return samples


def _build_level_3_samples(n: int = 50) -> List[Dict[str, object]]:
    """
    Level 3（困难）样本：
    - 关键改造点：显著增加“短文本但高认知负荷”的样本。
    - 通过硬核关键词（算法/证明/复杂度/异常处理等）和符号（O(log n)、代码符号）
      主动打破“长度=难度”的伪相关。
    """
    short_hard_prompts = [
        "请证明 O(log n)",
        "用Python写快排",
        "解释这段代码的异常处理",
        "比较快排和归并",
        "推导二分查找复杂度",
        "说明微服务架构设计",
        "分析源码逻辑",
        "写一个栈并证明正确性",
        "解释算法原理+复杂度",
        "设计缓存淘汰策略",
        "证明这个递归会终止",
        "比较线程和协程区别",
        "请推导 loss=w*x+b",
        "分析这段代码: if(x>0){...}",
        "写 BFS 并解释复杂度",
        "写 DFS 并比较区别",
        "证明哈希冲突处理正确",
        "解释异常处理设计原则",
        "实现 LRU 并说明原理",
        "比较 RPC 与 HTTP",
        "设计幂等接口逻辑",
        "写并查集并分析复杂度",
        "解释事务隔离级别区别",
        "分析 SQL 执行计划",
        "证明动态规划转移方程",
        "写 Dijkstra 并推导复杂度",
        "解释这段 C++ 源码逻辑",
        "设计限流算法",
        "证明贪心策略最优",
        "写二叉树层序遍历",
        "解释微服务熔断设计",
        "比较一致性哈希方案",
        "写拓扑排序并证明正确",
        "分析异常栈与错误边界",
        "证明单调栈算法正确",
    ]

    long_hard_templates = [
        "请阅读代码并完成问题定位、异常处理设计和复杂度证明：```python\ndef f(arr):\n    return sorted(arr)\n```",
        "请从架构角度比较单体与微服务，补充服务治理、熔断、链路追踪和一致性策略，并说明设计权衡。",
        "请给出快速排序的原理、伪代码、复杂度推导以及最坏情况分析，并比较与归并排序的区别。",
        "请解释一个分布式事务方案，包含异常处理、幂等设计、补偿逻辑与正确性证明。",
        "给定表达式 T(n)=2T(n/2)+n，请推导复杂度并说明推导过程中的关键假设。",
    ]

    samples: List[Dict[str, object]] = []
    short_target = int(n * 0.7)  # 70% 使用短高难样本

    for i in range(short_target):
        prompt = short_hard_prompts[i % len(short_hard_prompts)]
        # 轻微扰动，避免样本完全重复。
        if i % 3 == 0:
            prompt += "。"
        elif i % 3 == 1:
            prompt += "，请给步骤。"
        else:
            prompt += "，并解释原理。"
        samples.append({"prompt": prompt, "label": 3})

    while len(samples) < n:
        idx = len(samples) - short_target
        prompt = long_hard_templates[idx % len(long_hard_templates)]
        if idx % 2 == 0:
            prompt += " 请补充边界条件 x>0, x=0, x<0。"
        samples.append({"prompt": prompt, "label": 3})

    return samples


def generate_mock_data() -> List[Dict[str, object]]:
    """
    按 1:1:1 构建 150 条模拟数据。

    返回格式：
    [
      {"prompt": "...", "label": 1},
      {"prompt": "...", "label": 2},
      {"prompt": "...", "label": 3},
    ]
    """
    data: List[Dict[str, object]] = []
    data.extend(_build_level_1_samples(50))
    data.extend(_build_level_2_samples(50))
    data.extend(_build_level_3_samples(50))

    random.shuffle(data)
    return data


def train_and_save_model(model_path: str) -> None:
    """
    完整训练流程：特征提取 -> 数据集划分 -> 标准化 -> 模型训练 -> 评估 -> 持久化。

    参数：
    - model_path: router_model.pkl 输出路径
    """
    dataset = generate_mock_data()
    print(f"[INFO] 模拟样本总数: {len(dataset)}")

    X = np.array([extract_features(item["prompt"]) for item in dataset], dtype=np.float64)
    y = np.array([item["label"] for item in dataset], dtype=np.int64)

    if X.shape[1] != 6:
        raise ValueError(f"特征维度异常：期望 6，实际 {X.shape[1]}。请检查 feature_engineering.extract_features。")

    # 80% / 20% 划分训练与测试，且按标签分层抽样，保持类别分布一致。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    # 标准化原理：
    # x' = (x - mean) / std
    # 作用：统一各维度量纲，避免单一大尺度特征主导分裂。
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练轻量随机森林（50 棵树）。
    clf = RandomForestClassifier(
        n_estimators=50,
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    # ===== 评估输出（用于论文图表） =====
    acc = accuracy_score(y_test, y_pred)
    print("\n========== 模型评估 ==========")
    print(f"Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[1, 2, 3],
            target_names=["简单(1)", "中等(2)", "困难(3)"],
            digits=4,
            zero_division=0,
        )
    )

    # 特征重要性：值越大表示该特征对分类决策贡献越高。
    print("Feature Importance:")
    display_feature_names = [
        "Length",
        "Entropy",
        "Noun Ratio",
        "Verb Ratio",
        "Symbol Density",
        "Keyword Hits",
    ]
    importances = clf.feature_importances_
    sorted_pairs = sorted(zip(display_feature_names, importances), key=lambda x: x[1], reverse=True)
    for rank, (name, score) in enumerate(sorted_pairs, start=1):
        print(f"{rank}. {name:<15} -> {score:.6f}")

    # ===== 持久化 =====
    bundle = {
        "model": clf,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "version": "rf_router_v2_6d",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_size": len(dataset),
        "feature_dim": int(X.shape[1]),
    }
    joblib.dump(bundle, model_path)
    print(f"\n[INFO] 模型已保存到: {model_path}")


def main() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "router_model.pkl")
    train_and_save_model(model_path)


if __name__ == "__main__":
    main()
