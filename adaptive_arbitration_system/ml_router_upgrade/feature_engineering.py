"""
feature_engineering.py
======================
本文件是“难度预测前端”的特征工程核心库，负责把自然语言 Prompt 映射成固定维度数值向量。

设计目标：
1. 特征数量小（6D），便于端侧快速推理。
2. 特征语义清晰，便于毕业论文中解释“为什么这样设计”。
3. 与具体分类器解耦，后续可替换为 SVM / XGBoost / 轻量神经网络。

输出特征向量顺序（固定）：
[length, entropy, noun_ratio, verb_ratio, symbol_density, keyword_hit]
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List

import jieba.posseg as pseg

# 统一对外暴露的特征名，训练脚本可直接复用这个顺序。
FEATURE_NAMES = [
    "length",
    "entropy",
    "noun_ratio",
    "verb_ratio",
    "symbol_density",
    "keyword_hit",
]

# 高认知负荷关键词集合（用于打破“短文本=简单”的偏见）。
HARD_KEYWORDS = {
    "代码",
    "算法",
    "推导",
    "证明",
    "原理",
    "逻辑",
    "源码",
    "比较",
    "架构",
    "复杂度",
    "微服务",
    "区别",
    "设计",
    "异常处理",
}

# 名词/专有名词词性前缀（jieba 词性体系）。
NOUN_PREFIXES = ("n", "nr", "ns", "nt", "nz")

# 动词词性前缀。
# 注：v 覆盖大部分动词，vn 常表示名动词，这里按“动作语义”纳入动词密度统计。
VERB_PREFIXES = ("v", "vn")

# 特殊逻辑/代码符号集合。
# 用户要求至少覆盖：{} [] ` = + >
# 这里补充了 < 和 *，用于提升对数学/代码推理任务的敏感性。
SYMBOL_SET = set("{}[]`=+><*")


def _shannon_entropy(text: str) -> float:
    """
    计算字符级香农信息熵（Shannon Entropy）。

    信息熵公式：
        H(X) = - Σ p(x_i) * log2(p(x_i))

    解释：
    - 当文本字符分布越均匀、类型越多时，熵越高。
    - 对于“结构复杂、术语多、代码片段多”的文本，通常熵值更高。

    参数：
    - text: 输入文本

    返回：
    - 熵值（float），空文本返回 0.0
    """
    if not text:
        return 0.0

    total = len(text)
    counter = Counter(text)

    entropy = 0.0
    for count in counter.values():
        prob = count / total
        entropy -= prob * math.log2(prob)

    return entropy


def _extract_tokens(text: str):
    """
    使用 jieba 进行分词+词性标注，并过滤空白 token。

    返回：
    - tokens: 形如 pair(word, flag) 的可迭代结果（列表）
    """
    tokens = []
    for token in pseg.cut(text):
        word = token.word.strip()
        if not word:
            continue
        tokens.append(token)
    return tokens


def _pos_ratios_and_keyword_hit(tokens) -> tuple[float, float, float]:
    """
    计算名词密度、动词密度、硬核关键词命中数。

    说明：
    - noun_ratio = 名词类词数 / 总词数
    - verb_ratio = 动词类词数 / 总词数
    - keyword_hit = 分词结果中命中 HARD_KEYWORDS 的次数
      （按题目要求：通过遍历分词结果统计）
    """
    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0.0, 0.0, 0.0

    noun_count = 0
    verb_count = 0
    keyword_hit = 0

    for token in tokens:
        word = token.word
        flag = token.flag or ""

        if flag.startswith(NOUN_PREFIXES):
            noun_count += 1

        if flag.startswith(VERB_PREFIXES):
            verb_count += 1

        if word in HARD_KEYWORDS:
            keyword_hit += 1

    noun_ratio = noun_count / total_tokens
    verb_ratio = verb_count / total_tokens

    return float(noun_ratio), float(verb_ratio), float(keyword_hit)


def _symbol_density(text: str) -> float:
    """
    计算特殊符号密度。

    定义：
        symbol_density = 符号字符总数 / (文本总字符数 + 15)

    作用：
    - 对包含代码、公式、逻辑推理符号的文本更敏感。
    - 采用分母平滑后，可抑制极短文本因分母过小造成的密度虚高问题。
      例如“1+1等于几”这类短文本，不会因单个符号被放大为“高难任务”。
    """
    if not text:
        return 0.0

    symbol_count = sum(1 for ch in text if ch in SYMBOL_SET)
    length = len(text)
    return symbol_count / (length + 15)


def extract_features(prompt: str) -> List[float]:
    """
    核心接口：将原始 Prompt 转为 6D 特征向量。

    参数：
    - prompt: 输入文本

    返回：
    - [length, entropy, noun_ratio, verb_ratio, symbol_density, keyword_hit]

    说明：
    - length 使用 log(length + 1) 做对数平滑，收敛长尾分布并减小量纲差异。
    - entropy 反映文本信息复杂度。
    - noun_ratio / verb_ratio 反映语义构成。
    - symbol_density 反映逻辑/代码符号负载。
    - keyword_hit 反映硬核认知词触发强度，弱化长度偏见。
    """
    text = (prompt or "").strip()

    # 空字符串兜底：返回 6 维全 0，确保下游矩阵维度稳定。
    if not text:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    raw_length = len(text)
    length = float(math.log(raw_length + 1))
    entropy = float(_shannon_entropy(text))

    tokens = _extract_tokens(text)
    noun_ratio, verb_ratio, keyword_hit = _pos_ratios_and_keyword_hit(tokens)

    symbol_density = float(_symbol_density(text))

    return [
        length,
        entropy,
        noun_ratio,
        verb_ratio,
        symbol_density,
        keyword_hit,
    ]


if __name__ == "__main__":
    # 轻量自测，方便独立运行时快速查看特征效果。
    sample = "请证明 O(log n) 并比较快排与归并的复杂度。"
    feats = extract_features(sample)
    print("特征名:", FEATURE_NAMES)
    print("特征值:", [round(v, 6) for v in feats])
