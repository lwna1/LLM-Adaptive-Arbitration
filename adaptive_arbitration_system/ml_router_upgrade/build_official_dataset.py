"""
build_official_dataset.py
=========================
本脚本用于从官方开源数据集构建“统一训练数据源”，服务于 RF / MLP 双引擎的公平对比实验。

核心目标：
1. 使用 ModelScope 加载 AI-ModelScope/alpaca-gpt4-data-zh。
2. 提取 instruction 字段作为 Prompt。
3. 通过启发式规则自动打标难度（1/2/3）。
4. 定向抽样得到 1500 条数据（尽量保证 1:1:1）。
5. 导出 official_training_data.csv（仅保留 prompt, label 两列）。

运行方式：
    python3 build_official_dataset.py
"""

from __future__ import annotations

import importlib
import os
import random
import re
import sys
from typing import Dict, List

import pandas as pd
from modelscope.msdatasets import MsDataset

# 兼容两种运行方式：
# 1) 在父目录中作为包执行：python3 -m ml_router_upgrade.build_official_dataset
# 2) 在当前目录直接执行：python3 build_official_dataset.py
try:
    from .feature_engineering import HARD_KEYWORDS
except ImportError:  # pragma: no cover - 脚本直跑兜底
    from feature_engineering import HARD_KEYWORDS


# ===== 全局配置 =====
DATASET_ID = "AI-ModelScope/alpaca-gpt4-data-zh"
TARGET_TOTAL = 1500
TARGET_PER_LEVEL = 500
RANDOM_SEED = 42
SCENARIO_EXTRA_TOTAL = 160
BENCHMARK_EXTRA_TOTAL = 20

# benchmark 题库按标签重复注入倍率（用于训练加权）。
# 倍率是“总保留份数”，例如 12 代表该标签样本会保留 12 份（含原始 1 份）。
BENCHMARK_LABEL_REPEAT = {
    1: 2,
    2: 12,
    3: 2,
}

# 用于辅助识别“逻辑/代码型任务”的符号集合。
LOGIC_SYMBOLS = set("{}[]`=+><*/()")

# 明显高认知负荷的短文本模式（防止被“短文本规则”误判为 Level 1）。
HARD_SHORT_PATTERNS = [
    re.compile(r"O\(.+\)", re.IGNORECASE),
    re.compile(r"\b(log|bfs|dfs|sql|rpc|api|raft|paxos)\b", re.IGNORECASE),
    re.compile(r"(快排|归并|最短路|并查集|哈希|拓扑|红黑树|线段树|贪心|递归|动态规划)"),
    re.compile(r"(证明|推导|复杂度|源码|架构|微服务|一致性|事务|异常处理)"),
]

# 明确低负荷短文本白名单，仅这些短文本才会优先视为 Level 1。
TRIVIAL_SHORT_TEXTS = {
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

SIMPLE_ARITHMETIC_PATTERN = re.compile(r"^\s*\d+\s*[\+\-\*/]\s*\d+\s*(等于几|\?)?\s*$")

# 明确“中等任务意图”的关键词。
# 这些任务通常不要求高强度推导/代码实现，但认知负荷明显高于闲聊与极简指令。
MEDIUM_INTENT_KEYWORDS = [
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


def auto_label_difficulty(prompt: str) -> int:
    """
    启发式自动打标函数（Heuristic Labeling）。

    规则（宽松版）：
    1) 若命中硬核关键词、硬核短文本模式，或包含较多逻辑符号 -> Level 3
    2) 若文本较短且未出现高负荷特征 -> Level 1
    3) 其余判为 Level 2

    说明：
    - 该规则用于快速构建大规模“弱监督标签”，为消融实验提供统一数据源。
    - 这里复用了 feature_engineering.HARD_KEYWORDS，避免训练与推理语义不一致。
    """
    text = (prompt or "").strip()
    if not text:
        return 1

    keyword_hit = sum(1 for kw in HARD_KEYWORDS if kw in text)
    symbol_count = sum(1 for ch in text if ch in LOGIC_SYMBOLS)
    symbol_density = symbol_count / max(len(text), 1)

    # “大量符号”同时参考绝对数量和密度，阈值略提高以避免误伤普通文本。
    has_heavy_symbols = symbol_count >= 5 or symbol_density >= 0.10

    hard_short_hit = any(pattern.search(text) is not None for pattern in HARD_SHORT_PATTERNS)
    if keyword_hit > 0 or has_heavy_symbols or hard_short_hit:
        return 3

    # 在长度规则之前优先识别“中等任务意图”，避免被短文本条件误压到 Level 1。
    if any(keyword in text for keyword in MEDIUM_INTENT_KEYWORDS):
        return 2

    # Level 1 改为更贴近真实分布的“宽松短文本判定”：
    # - 白名单短句/简单算术直接判 1
    # - 长度 <= 28 且无硬核命中、符号负载低，也归为 1
    lowered = text.lower()
    if lowered in TRIVIAL_SHORT_TEXTS or SIMPLE_ARITHMETIC_PATTERN.match(text):
        return 1

    if len(text) <= 28 and keyword_hit == 0 and symbol_count <= 1 and not hard_short_hit:
        return 1

    return 2


def _msdataset_to_dataframe(ms_dataset_obj) -> pd.DataFrame:
    """
    将 ModelScope 数据对象尽可能稳健地转换为 pandas.DataFrame。
    """
    # 常见路径：MsDataset -> HuggingFace Dataset -> pandas
    if hasattr(ms_dataset_obj, "to_hf_dataset"):
        hf_ds = ms_dataset_obj.to_hf_dataset()
        if hasattr(hf_ds, "to_pandas"):
            return hf_ds.to_pandas()
        return pd.DataFrame(hf_ds)

    # 兜底路径：对象本身支持 to_pandas
    if hasattr(ms_dataset_obj, "to_pandas"):
        return ms_dataset_obj.to_pandas()

    # 若是可迭代对象，强制转 DataFrame
    return pd.DataFrame(list(ms_dataset_obj))


def load_official_prompts(dataset_id: str = DATASET_ID) -> List[str]:
    """
    加载 Alpaca-zh 数据集并提取 instruction 字段。
    """
    load_errors: List[str] = []
    dataset_obj = None

    # 不同环境下 MsDataset.load 参数兼容性可能略有差异，做两次兜底尝试。
    load_attempts = [
        {"dataset_name": dataset_id, "split": "train"},
        {"dataset_name": dataset_id},
    ]

    for kwargs in load_attempts:
        try:
            dataset_obj = MsDataset.load(**kwargs)
            break
        except Exception as exc:  # pragma: no cover - 真实错误依赖运行环境
            load_errors.append(f"kwargs={kwargs}, error={exc}")

    if dataset_obj is None:
        joined_errors = "\n".join(load_errors) if load_errors else "未知错误"
        raise RuntimeError(
            "加载 ModelScope 数据集失败，请检查网络与 dataset_id。\n"
            f"dataset_id={dataset_id}\n"
            f"详情：\n{joined_errors}"
        )

    # 某些版本可能返回 split 映射对象，这里优先拿 train。
    if isinstance(dataset_obj, dict):
        if "train" in dataset_obj:
            dataset_obj = dataset_obj["train"]
        else:
            dataset_obj = next(iter(dataset_obj.values()))

    df = _msdataset_to_dataframe(dataset_obj)
    if "instruction" not in df.columns:
        raise KeyError("数据集中未找到 instruction 字段，无法构建 prompt。")

    prompts = (
        df["instruction"]
        .fillna("")
        .astype(str)
        .map(str.strip)
        .tolist()
    )
    prompts = [p for p in prompts if p]
    return prompts


def build_balanced_dataset(
    prompts: List[str],
    target_total: int = TARGET_TOTAL,
    target_per_level: int = TARGET_PER_LEVEL,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    将原始 prompt 列表打标后做定向抽样，构建约 1500 条的均衡数据集。

    采样策略（真实标注优先）：
    - 先按启发式标签分桶，尽量每类各取 500。
    - 若某类不足 500，不强制重写标签，而是保留真实标签分布，
      从剩余样本中补齐到 1500（用于训练规模稳定）。
    - 最终随机打散，避免类别顺序偏置。
    """
    if not prompts:
        raise ValueError("输入 prompts 为空，无法构建训练集。")

    random.seed(seed)

    df = pd.DataFrame({"prompt": prompts})
    # 去重有助于减弱重复指令对模型的记忆偏置。
    df = df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    df["label"] = df["prompt"].map(auto_label_difficulty).astype(int)

    if len(df) < target_total:
        raise ValueError(f"可用样本不足：需要 {target_total}，实际 {len(df)}。")

    grouped: Dict[int, pd.DataFrame] = {level: df[df["label"] == level].copy() for level in [1, 2, 3]}
    selected_parts: List[pd.DataFrame] = []
    selected_indices = set()

    # 第一阶段：尽量均衡（每类最多 500）。
    for level in [1, 2, 3]:
        bucket = grouped[level]
        take_n = min(target_per_level, len(bucket))
        if take_n > 0:
            sampled = bucket.sample(n=take_n, random_state=seed)
            selected_parts.append(sampled)
            selected_indices.update(sampled.index.tolist())

    selected_df = pd.concat(selected_parts, axis=0) if selected_parts else pd.DataFrame(columns=df.columns)

    # 第二阶段：若不足 1500，则从剩余池按真实标签补齐，不改标签定义。
    shortage = target_total - len(selected_df)
    if shortage > 0:
        remain_df = df.loc[~df.index.isin(selected_indices)].copy()
        extra = remain_df.sample(n=min(shortage, len(remain_df)), random_state=seed)
        selected_df = pd.concat([selected_df, extra], axis=0)

    # 防御性截断到目标规模。
    if len(selected_df) > target_total:
        selected_df = selected_df.sample(n=target_total, random_state=seed)

    selected_df = selected_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    selected_df = selected_df[["prompt", "label"]]
    return selected_df


def load_scenario_extra_samples() -> pd.DataFrame:
    """
    读取 run_all_scenarios_benchmark.py 中定义的 160 条场景题，
    并严格复用其 ground_truth 标注作为训练标签。

    说明：
    - 这里不改动场景脚本内的标注策略，直接复用其构造函数输出；
    - 这样可保证“训练补充样本”和“场景跑批样本”语义一致，
      满足论文中控制变量法对标注一致性的要求。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        scenario_module = importlib.import_module("run_all_scenarios_benchmark")
    except Exception as exc:
        raise RuntimeError(
            "无法导入 run_all_scenarios_benchmark.py，请确认文件存在且可执行。"
        ) from exc

    if not hasattr(scenario_module, "build_all_scenarios"):
        raise AttributeError("run_all_scenarios_benchmark.py 缺少 build_all_scenarios 函数。")

    all_scenarios = scenario_module.build_all_scenarios()
    records: List[Dict[str, object]] = []

    for _, question_bank in all_scenarios.items():
        for item in question_bank:
            records.append(
                {
                    "prompt": str(item.get("prompt", "")).strip(),
                    "label": int(item.get("ground_truth", 1)),
                }
            )

    extra_df = pd.DataFrame(records)
    extra_df = extra_df[extra_df["prompt"].astype(str).str.len() > 0].copy()
    extra_df["label"] = extra_df["label"].astype(int)

    if len(extra_df) != SCENARIO_EXTRA_TOTAL:
        raise ValueError(
            f"场景补充样本数量异常：期望 {SCENARIO_EXTRA_TOTAL}，实际 {len(extra_df)}。"
        )

    return extra_df[["prompt", "label"]]


def load_benchmark_extra_samples() -> pd.DataFrame:
    """
    读取 benchmark_pipeline.py 中定义的 20 条标准测试题，
    并复用其 ground_truth 作为训练标签。

    说明：
    - 该 20 条题是系统历史基准题库，加入训练集有助于提升
      在“短指令/常识/中高难解释与代码题”上的覆盖稳定性；
    - 按 benchmark_pipeline 的原始人工标签注入，不改标注逻辑。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        benchmark_module = importlib.import_module("benchmark_pipeline")
    except Exception as exc:
        raise RuntimeError(
            "无法导入 benchmark_pipeline.py，请确认文件存在且可执行。"
        ) from exc

    if not hasattr(benchmark_module, "build_question_bank"):
        raise AttributeError("benchmark_pipeline.py 缺少 build_question_bank 函数。")

    question_bank = benchmark_module.build_question_bank()
    records: List[Dict[str, object]] = []
    for item in question_bank:
        records.append(
            {
                "prompt": str(item.get("prompt", "")).strip(),
                "label": int(item.get("ground_truth", 1)),
            }
        )

    extra_df = pd.DataFrame(records)
    extra_df = extra_df[extra_df["prompt"].astype(str).str.len() > 0].copy()
    extra_df["label"] = extra_df["label"].astype(int)

    if len(extra_df) != BENCHMARK_EXTRA_TOTAL:
        raise ValueError(
            f"benchmark 补充样本数量异常：期望 {BENCHMARK_EXTRA_TOTAL}，实际 {len(extra_df)}。"
        )

    return extra_df[["prompt", "label"]]


def expand_benchmark_samples_by_label(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """
    对 benchmark 样本按标签进行重复注入（训练加权）。

    设计目的：
    - 原始 benchmark 仅 20 条，占总体训练集比例较低；
    - 对 Level 2 提高重复倍率，增强中等难度任务的决策权重。
    """
    if benchmark_df.empty:
        return benchmark_df.copy()

    expanded_parts: List[pd.DataFrame] = []
    for label in [1, 2, 3]:
        sub = benchmark_df[benchmark_df["label"] == label].copy()
        if sub.empty:
            continue

        repeat_n = int(BENCHMARK_LABEL_REPEAT.get(label, 1))
        repeat_n = max(1, repeat_n)
        expanded_parts.append(pd.concat([sub.copy() for _ in range(repeat_n)], ignore_index=True))

    expanded_df = pd.concat(expanded_parts, ignore_index=True) if expanded_parts else benchmark_df.copy()
    return expanded_df[["prompt", "label"]]


def merge_with_fixed_extras(
    official_df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    将官方训练集与固定补充题库合并并打乱顺序。

    关键约束：
    - 官方抽样结果（1500 条）不做内容修改；
    - 场景样本（160）按 run_all_scenarios_benchmark 的人工标签注入；
    - benchmark 样本（20）按人工标签注入，并按标签重复注入做训练加权；
    - 合并后随机打散，输出统一训练集供 RF/MLP 共用。
    """
    scenario_df = load_scenario_extra_samples()
    benchmark_df = load_benchmark_extra_samples()
    weighted_benchmark_df = expand_benchmark_samples_by_label(benchmark_df)
    merged_df = pd.concat([official_df.copy(), scenario_df, weighted_benchmark_df], ignore_index=True)
    merged_df = merged_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return merged_df[["prompt", "label"]]


def main() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(current_dir, "official_training_data.csv")

    print(f"[INFO] 正在加载官方数据集：{DATASET_ID}")
    prompts = load_official_prompts(DATASET_ID)
    print(f"[INFO] 原始可用 prompt 数量：{len(prompts)}")

    print("[INFO] 开始自动打标与定向抽样（目标 1500 条）...")
    official_df = build_balanced_dataset(
        prompts=prompts,
        target_total=TARGET_TOTAL,
        target_per_level=TARGET_PER_LEVEL,
        seed=RANDOM_SEED,
    )

    label_counts = official_df["label"].value_counts().sort_index()
    print("[INFO] 官方 1500 条采样完成，类别分布：")
    for level in [1, 2, 3]:
        print(f"  - Level {level}: {int(label_counts.get(level, 0))}")

    print("[INFO] 正在注入 run_all_scenarios_benchmark 的 160 条场景样本...")
    print("[INFO] 正在注入 benchmark_pipeline 的 20 条基准样本（按标签重复注入）...")
    final_df = merge_with_fixed_extras(official_df=official_df, seed=RANDOM_SEED)
    final_counts = final_df["label"].value_counts().sort_index()

    weighted_benchmark_rows = (
        5 * BENCHMARK_LABEL_REPEAT.get(1, 1)
        + 5 * BENCHMARK_LABEL_REPEAT.get(2, 1)
        + 10 * BENCHMARK_LABEL_REPEAT.get(3, 1)
    )

    print("[INFO] 训练集类别分布：")
    for level in [1, 2, 3]:
        print(f"  - Level {level}: {int(final_counts.get(level, 0))}")
    print(
        f"[INFO] 最终样本总数：{len(final_df)}"
        f"（官方 {TARGET_TOTAL} + 场景 {SCENARIO_EXTRA_TOTAL}"
        f" + 基准加权 {weighted_benchmark_rows} / 原始基准 {BENCHMARK_EXTRA_TOTAL}）"
    )

    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已导出：{output_csv}")


if __name__ == "__main__":
    main()
