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

import os
import random
import re
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
    print("[INFO] 采样完成，类别分布：")
    for level in [1, 2, 3]:
        print(f"  - Level {level}: {int(label_counts.get(level, 0))}")
    print(f"[INFO] 最终样本总数：{len(official_df)}")

    official_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已导出：{output_csv}")


if __name__ == "__main__":
    main()
