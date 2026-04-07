"""
evaluate_10k_and_global.py
==========================
全局算法级评测脚本（与具体场景无关）。

功能：
1. 自动创建 output/global_algorithm_eval/ 目录
2. 拉取 Alpaca-zh 数据并构建 10,000 条评估样本（启发式打标）
3. 对比 RF / MLP 路由引擎预测能力（混淆矩阵 + 指标表格）
4. 输出随机森林特征重要性与 MLP Loss 曲线

输出文件（严格对齐目录规范）：
- 10k_confusion_matrix.png
- 10k_metrics_compare.md
- feature_importance.png
- mlp_loss_curve.png
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
from modelscope.msdatasets import MsDataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from ml_router_upgrade.feature_extractor import predict_difficulty
from ml_router_upgrade.build_official_dataset import auto_label_difficulty


# ===== 全局样式（防中文乱码 + 学术风）=====
FONT_CANDIDATES = [
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Noto Sans CJK SC",
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
]


def _configure_plot_fonts() -> None:
    """
    稳健配置中文字体，避免 seaborn 覆盖 rcParams 导致方块字。
    """
    font_file_candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fpath in font_file_candidates:
        if os.path.exists(fpath):
            try:
                font_manager.fontManager.addfont(fpath)
            except Exception:
                pass

    sns.set_theme(
        style="ticks",
        context="paper",
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": FONT_CANDIDATES,
            "axes.unicode_minus": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        },
    )

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300


_configure_plot_fonts()

SEED = 42
TARGET_SIZE = 10_000
DATASET_ID = "AI-ModelScope/alpaca-gpt4-data-zh"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GLOBAL_DIR = os.path.join(PROJECT_ROOT, "output", "global_algorithm_eval")
ML_DIR = os.path.join(PROJECT_ROOT, "ml_router_upgrade")

ROUTER_MODEL_PATH = os.path.join(ML_DIR, "router_model.pkl")
MLP_LOSS_CSV_PATHS = [
    os.path.join(PROJECT_ROOT, "output", "mlp_loss_history.csv"),
    os.path.join(ML_DIR, "mlp_loss_history.csv"),
]


def _msdataset_to_dataframe(ms_dataset_obj) -> pd.DataFrame:
    """
    将 ModelScope 数据对象稳健转换为 DataFrame。
    """
    if hasattr(ms_dataset_obj, "to_hf_dataset"):
        hf_ds = ms_dataset_obj.to_hf_dataset()
        if hasattr(hf_ds, "to_pandas"):
            return hf_ds.to_pandas()
        return pd.DataFrame(hf_ds)

    if hasattr(ms_dataset_obj, "to_pandas"):
        return ms_dataset_obj.to_pandas()

    return pd.DataFrame(list(ms_dataset_obj))


def load_alpaca_prompts(dataset_id: str = DATASET_ID) -> List[str]:
    """
    从 Alpaca-zh 数据集中提取 instruction 字段。
    """
    attempts = [
        {"dataset_name": dataset_id, "split": "train"},
        {"dataset_name": dataset_id},
    ]
    dataset_obj = None
    errors = []

    for kwargs in attempts:
        try:
            dataset_obj = MsDataset.load(**kwargs)
            break
        except Exception as exc:  # pragma: no cover
            errors.append(f"{kwargs}: {exc}")

    if dataset_obj is None:
        raise RuntimeError(
            "无法加载 Alpaca 数据集，请检查网络与 ModelScope 环境。\n"
            + "\n".join(errors)
        )

    if isinstance(dataset_obj, dict):
        dataset_obj = dataset_obj.get("train", next(iter(dataset_obj.values())))

    df = _msdataset_to_dataframe(dataset_obj)
    if "instruction" not in df.columns:
        raise KeyError("数据集中缺少 instruction 字段。")

    prompts = (
        df["instruction"]
        .fillna("")
        .astype(str)
        .map(str.strip)
        .tolist()
    )
    prompts = [p for p in prompts if p]
    return prompts


def build_eval_set(size: int = TARGET_SIZE) -> pd.DataFrame:
    """
    构建评估集并附加启发式标签。

    说明：
    - 标签函数直接复用 build_official_dataset.py 中的 auto_label_difficulty，
      确保“训练集构建”和“全局评测”完全一致。
    """
    random.seed(SEED)
    prompts = load_alpaca_prompts(DATASET_ID)
    if len(prompts) < size:
        raise ValueError(f"数据量不足：需要 {size} 条，实际 {len(prompts)} 条")

    sampled = random.sample(prompts, size)
    df = pd.DataFrame({"prompt": sampled})
    df["label"] = df["prompt"].map(auto_label_difficulty).astype(int)
    return df


def evaluate_engines(eval_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    分别用 RF 和 MLP 进行预测。
    """
    y_true = eval_df["label"].to_numpy(dtype=np.int64)

    rf_pred = np.array([predict_difficulty(p, engine="rf") for p in eval_df["prompt"]], dtype=np.int64)
    mlp_pred = np.array([predict_difficulty(p, engine="mlp") for p in eval_df["prompt"]], dtype=np.int64)

    return y_true, rf_pred, mlp_pred


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算 Accuracy / Precision / Recall / F1（macro）。
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def save_confusion_figure(y_true: np.ndarray, rf_pred: np.ndarray, mlp_pred: np.ndarray) -> str:
    """
    输出 1x2 混淆矩阵对比图。
    """
    labels = [1, 2, 3]
    rf_cm = confusion_matrix(y_true, rf_pred, labels=labels)
    mlp_cm = confusion_matrix(y_true, mlp_pred, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    sns.heatmap(
        rf_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=axes[0],
    )
    axes[0].set_title("随机森林混淆矩阵（1万条）")
    axes[0].set_xlabel("预测难度")
    axes[0].set_ylabel("真实难度")

    sns.heatmap(
        mlp_cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_title("多层感知机混淆矩阵（1万条）")
    axes[1].set_xlabel("预测难度")
    axes[1].set_ylabel("")

    fig.tight_layout()
    out_path = os.path.join(GLOBAL_DIR, "10k_confusion_matrix.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_metrics_markdown(rf_metrics: Dict[str, float], mlp_metrics: Dict[str, float]) -> str:
    """
    输出 10k 指标对比 Markdown 表。
    """
    lines = [
        "| 引擎 | Accuracy | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
        f"| 随机森林（RF） | {rf_metrics['accuracy']:.4f} | {rf_metrics['precision']:.4f} | {rf_metrics['recall']:.4f} | {rf_metrics['f1']:.4f} |",
        f"| 多层感知机（MLP） | {mlp_metrics['accuracy']:.4f} | {mlp_metrics['precision']:.4f} | {mlp_metrics['recall']:.4f} | {mlp_metrics['f1']:.4f} |",
    ]
    content = "\n".join(lines) + "\n"
    out_path = os.path.join(GLOBAL_DIR, "10k_metrics_compare.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path


def save_feature_importance() -> str:
    """
    输出随机森林特征重要性图。
    """
    if not os.path.exists(ROUTER_MODEL_PATH):
        raise FileNotFoundError(f"未找到模型：{ROUTER_MODEL_PATH}")

    bundle = joblib.load(ROUTER_MODEL_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("router_model.pkl 格式异常，无法读取")

    model = bundle["model"]
    if not hasattr(model, "feature_importances_"):
        raise ValueError("RF 模型缺少 feature_importances_")

    importances = np.asarray(model.feature_importances_, dtype=float)
    feature_names = bundle.get("feature_names") or [
        "length",
        "entropy",
        "noun_ratio",
        "verb_ratio",
        "symbol_density",
        "keyword_hit",
    ]
    feature_names = feature_names[: len(importances)]

    zh_map = {
        "length": "文本长度(log)",
        "entropy": "信息熵",
        "noun_ratio": "名词比例",
        "verb_ratio": "动词比例",
        "symbol_density": "符号密度",
        "keyword_hit": "硬核词命中",
    }
    labels = [zh_map.get(n, n) for n in feature_names]

    order = np.argsort(importances)
    labels_sorted = [labels[i] for i in order]
    scores_sorted = importances[order]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh(labels_sorted, scores_sorted, color="#4C72B0")
    ax.set_title("随机森林特征重要性")
    ax.set_xlabel("重要性权重")
    ax.set_ylabel("特征维度")

    out_path = os.path.join(GLOBAL_DIR, "feature_importance.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_mlp_loss_curve() -> str:
    """
    输出 MLP Loss 曲线。
    若未找到 loss 历史文件，则生成提示图，保证目录产物完整。
    """
    loss_path = None
    for p in MLP_LOSS_CSV_PATHS:
        if os.path.exists(p):
            loss_path = p
            break

    fig, ax = plt.subplots(figsize=(7, 4.2))

    if loss_path is None:
        ax.text(0.5, 0.5, "未找到 mlp_loss_history.csv", ha="center", va="center", fontsize=12)
        ax.set_title("MLP 训练损失曲线")
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("损失值")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        loss_df = pd.read_csv(loss_path, encoding="utf-8-sig")
        if not {"epoch", "loss"}.issubset(loss_df.columns):
            raise ValueError(f"文件格式异常：{loss_path} 缺少 epoch/loss 列")
        ax.plot(loss_df["epoch"], loss_df["loss"], color="#1f77b4", linewidth=1.8)
        ax.set_title("MLP 训练损失曲线")
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("损失值")
        ax.grid(alpha=0.25, linestyle="--")

    out_path = os.path.join(GLOBAL_DIR, "mlp_loss_curve.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    os.makedirs(GLOBAL_DIR, exist_ok=True)

    print("[INFO] 构建 10,000 条全局评估集...")
    eval_df = build_eval_set(TARGET_SIZE)

    print("[INFO] 执行 RF / MLP 双引擎预测...")
    y_true, rf_pred, mlp_pred = evaluate_engines(eval_df)

    rf_metrics = _calc_metrics(y_true, rf_pred)
    mlp_metrics = _calc_metrics(y_true, mlp_pred)

    out1 = save_confusion_figure(y_true, rf_pred, mlp_pred)
    out2 = save_metrics_markdown(rf_metrics, mlp_metrics)
    out3 = save_feature_importance()
    out4 = save_mlp_loss_curve()

    print("[INFO] 全局评测完成：")
    for p in [out1, out2, out3, out4]:
        print(f" - {p}")


if __name__ == "__main__":
    main()
