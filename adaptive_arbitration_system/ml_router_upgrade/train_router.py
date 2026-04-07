"""
train_router.py
===============
本脚本负责训练 Random Forest 难度路由器（RF 引擎）。

关键目标：
1. 使用统一官方数据源 official_training_data.csv 训练 RF。
2. 输出论文可视化所需的评估产物（metrics/confusion 数据）。
3. 将附加产物统一保存到项目 output/ 目录，便于后续 experiment_analytics.py 一键读取。

可直接运行：
    python3 train_router.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 兼容两种运行方式：
# 1) 在父目录中作为包执行：python3 -m ml_router_upgrade.train_router
# 2) 在当前目录直接执行：python3 train_router.py
try:
    from .feature_engineering import FEATURE_NAMES, extract_features
except ImportError:  # pragma: no cover - 脚本直跑兜底
    from feature_engineering import FEATURE_NAMES, extract_features


SEED = 42


def load_official_dataset(csv_path: str) -> pd.DataFrame:
    """
    加载官方训练集 CSV。

    为消融实验统一数据源，RF 与 MLP 必须读取同一份 prompt/label 数据。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"未找到官方训练集：{csv_path}\n"
            "请先运行 build_official_dataset.py 生成 official_training_data.csv。"
        )

    df = pd.read_csv(csv_path)
    required_cols = {"prompt", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"官方训练集缺少必要列：{missing}")

    df = df.copy()
    df["prompt"] = df["prompt"].fillna("").astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[(df["prompt"] != "") & (df["label"].isin([1, 2, 3]))].copy()
    df["label"] = df["label"].astype(int)

    if len(df) < 100:
        raise ValueError(f"有效样本过少（{len(df)} 条），请检查数据构建流程。")

    return df.reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """构建 6D 特征矩阵 X 与标签向量 y。"""
    X = np.array([extract_features(p) for p in df["prompt"].tolist()], dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)

    if X.ndim != 2 or X.shape[1] != 6:
        raise ValueError(f"特征维度异常：期望 [N, 6]，实际 {X.shape}")
    return X, y


def _build_metrics_dict(report_dict: Dict[str, object], accuracy: float) -> Dict[str, object]:
    """
    将 classification_report 的输出整理为论文常用指标结构。
    """
    macro_avg = report_dict.get("macro avg", {})
    weighted_avg = report_dict.get("weighted avg", {})
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(macro_avg.get("precision", 0.0)),
        "recall_macro": float(macro_avg.get("recall", 0.0)),
        "f1_macro": float(macro_avg.get("f1-score", 0.0)),
        "precision_weighted": float(weighted_avg.get("precision", 0.0)),
        "recall_weighted": float(weighted_avg.get("recall", 0.0)),
        "f1_weighted": float(weighted_avg.get("f1-score", 0.0)),
        "full_report": report_dict,
    }


def train_and_save_model(
    model_path: str,
    dataset_csv_path: str,
    output_dir: str,
) -> None:
    """
    完整训练流程：读取数据 -> 特征提取 -> 训练 -> 评估 -> 持久化。
    """
    os.makedirs(output_dir, exist_ok=True)

    df = load_official_dataset(dataset_csv_path)
    print(f"[INFO] 官方训练样本总数: {len(df)}")

    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=50,
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    report_str = classification_report(
        y_test,
        y_pred,
        labels=[1, 2, 3],
        target_names=["简单(1)", "中等(2)", "困难(3)"],
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=[1, 2, 3],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])

    print("\n========== RF 模型评估 ==========")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report_str)
    print("Confusion Matrix:")
    print(cm)

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

    # ===== 保存模型 =====
    bundle = {
        "model": clf,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "version": "rf_router_v4_official_6d",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_size": int(len(df)),
        "feature_dim": int(X.shape[1]),
        "data_source": os.path.basename(dataset_csv_path),
    }
    joblib.dump(bundle, model_path)
    print(f"\n[INFO] 模型已保存到: {model_path}")

    # ===== 保存论文可视化产物 =====
    metrics_path = os.path.join(output_dir, "rf_metrics.json")
    confusion_path = os.path.join(output_dir, "rf_confusion_data.csv")

    metrics_payload = _build_metrics_dict(report_dict, acc)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    confusion_df = pd.DataFrame({"y_true": y_test.astype(int), "y_pred": y_pred.astype(int)})
    confusion_df.to_csv(confusion_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] 指标已保存: {metrics_path}")
    print(f"[INFO] 混淆矩阵数据已保存: {confusion_path}")


def main() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))  # .../ml_router_upgrade
    project_root = os.path.dirname(current_dir)  # .../adaptive_arbitration_system
    output_dir = os.path.join(project_root, "output")

    model_path = os.path.join(current_dir, "router_model.pkl")
    dataset_csv_path = os.path.join(current_dir, "official_training_data.csv")

    train_and_save_model(
        model_path=model_path,
        dataset_csv_path=dataset_csv_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
