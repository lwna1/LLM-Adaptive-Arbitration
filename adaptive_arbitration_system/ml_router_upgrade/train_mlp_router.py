"""
train_mlp_router.py
===================
本脚本用于训练基于 PyTorch 的 MLP 路由器（对比引擎）。

关键目标：
1. 读取统一官方训练集 official_training_data.csv（控制变量法）。
2. 训练 MLP 并保存模型权重与标准化器。
3. 导出论文可视化与统计分析所需数据：
   - mlp_loss_history.csv
   - mlp_metrics.json
   - mlp_confusion_data.csv

运行方式：
    python3 train_mlp_router.py
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# 兼容两种运行方式：
# 1) 在父目录中作为包执行：python3 -m ml_router_upgrade.train_mlp_router
# 2) 在当前目录直接执行：python3 train_mlp_router.py
try:
    from .feature_engineering import extract_features
except ImportError:  # pragma: no cover - 脚本直跑兜底
    from feature_engineering import extract_features


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_official_dataset(csv_path: str) -> pd.DataFrame:
    """
    加载官方训练集 CSV（prompt + label）。

    说明：
    - MLP 与 RF 必须读取同一数据源，确保对比实验公平。
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
    """构建 MLP 训练所需的 6D 特征矩阵与标签向量。"""
    X = np.array([extract_features(p) for p in df["prompt"].tolist()], dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)

    if X.ndim != 2 or X.shape[1] != 6:
        raise ValueError(f"特征维度异常：期望 [N, 6]，实际 {X.shape}")
    return X, y


class DifficultyDataset(Dataset):
    """
    难度分类数据集。

    注意：
    - 原始标签是 1/2/3。
    - CrossEntropyLoss 需要类别索引 0/1/2，因此这里减 1。
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels - 1, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


class MLPRouter(nn.Module):
    """
    MLP 路由器结构：
    - 输入层：6
    - 隐藏层1：Linear(6,64)+ReLU+Dropout(0.2)
    - 隐藏层2：Linear(64,32)+ReLU
    - 输出层：Linear(32,3)
    """

    def __init__(self, input_dim: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_metrics_dict(report_dict: Dict[str, object], accuracy: float) -> Dict[str, object]:
    """整理 classification_report 为论文常用指标结构。"""
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


def train_and_save_mlp(
    model_path: str,
    scaler_path: str,
    dataset_csv_path: str,
    output_dir: str,
    epochs: int = 120,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
) -> None:
    """
    训练 MLP 并保存模型权重、标准化器与可视化数据文件。
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

    train_ds = DifficultyDataset(X_train_scaled, y_train)
    test_ds = DifficultyDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRouter(input_dim=6).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"[INFO] 使用设备: {device}")
    print(f"[INFO] 开始训练，共 {epochs} 个 Epoch")

    # 记录每个 Epoch 的 loss，用于论文绘制收敛曲线。
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        avg_loss = epoch_loss / len(train_ds)
        loss_history.append({"epoch": epoch, "loss": float(avg_loss)})

        if epoch % 20 == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}/{epochs}] Loss = {avg_loss:.6f}")

    # ===== 测试集评估 =====
    model.eval()
    all_true_idx = []
    all_pred_idx = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)

            all_true_idx.extend(batch_y.cpu().numpy().tolist())
            all_pred_idx.extend(preds.cpu().numpy().tolist())

    # 恢复到原始标签空间（1/2/3）
    y_true = np.asarray(all_true_idx, dtype=np.int64) + 1
    y_pred = np.asarray(all_pred_idx, dtype=np.int64) + 1

    test_acc = accuracy_score(y_true, y_pred)
    report_str = classification_report(
        y_true,
        y_pred,
        labels=[1, 2, 3],
        target_names=["简单(1)", "中等(2)", "困难(3)"],
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[1, 2, 3],
        output_dict=True,
        zero_division=0,
    )

    print("\n========== MLP 模型评估 ==========")
    print(f"Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(report_str)

    # ===== 保存模型 =====
    ckpt = {
        "model_state_dict": model.state_dict(),
        "input_dim": 6,
        "num_classes": 3,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "dataset_size": int(len(df)),
        "data_source": os.path.basename(dataset_csv_path),
    }
    torch.save(ckpt, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] 模型权重已保存: {model_path}")
    print(f"[INFO] 标准化器已保存: {scaler_path}")

    # ===== 保存论文可视化产物 =====
    loss_path = os.path.join(output_dir, "mlp_loss_history.csv")
    metrics_path = os.path.join(output_dir, "mlp_metrics.json")
    confusion_path = os.path.join(output_dir, "mlp_confusion_data.csv")

    pd.DataFrame(loss_history).to_csv(loss_path, index=False, encoding="utf-8-sig")

    metrics_payload = _build_metrics_dict(report_dict, test_acc)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
        confusion_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[INFO] Loss 历史已保存: {loss_path}")
    print(f"[INFO] 指标已保存: {metrics_path}")
    print(f"[INFO] 混淆矩阵数据已保存: {confusion_path}")


def main() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))  # .../ml_router_upgrade
    project_root = os.path.dirname(current_dir)  # .../adaptive_arbitration_system
    output_dir = os.path.join(project_root, "output")

    model_path = os.path.join(current_dir, "mlp_router.pth")
    scaler_path = os.path.join(current_dir, "mlp_scaler.pkl")
    dataset_csv_path = os.path.join(current_dir, "official_training_data.csv")

    train_and_save_mlp(
        model_path=model_path,
        scaler_path=scaler_path,
        dataset_csv_path=dataset_csv_path,
        output_dir=output_dir,
        epochs=120,
        batch_size=16,
        learning_rate=1e-3,
    )


if __name__ == "__main__":
    main()
