import os
import joblib
import torch
import torch.nn as nn
from ml_router_upgrade.feature_engineering import extract_features

# 1. 提取底层特征
prompt = "1+1等于几"
features = extract_features(prompt)
feature_names = ["Log长度", "信息熵", "名词比例", "动词比例", "平滑符号密度", "硬核词命中"]

print("========== [1+1等于几] 6D特征透视 ==========")
for name, val in zip(feature_names, features):
    print(f"[{name}]: {val:.4f}")

# 2. 模拟 Random Forest 的思考过程
try:
    rf_data = joblib.load('/root/LAA/adaptive_arbitration_system/ml_router_upgrade/router_model.pkl')
    scaler = rf_data['scaler']
    rf_model = rf_data['model']
    
    # 打印 Scaler 缩放后的样子
    scaled_features = scaler.transform([features])
    print("\n========== Scaler 归一化后的数据 ==========")
    print(scaled_features[0])
    
    # 打印 RF 各个类别的概率
    rf_probs = rf_model.predict_proba(scaled_features)[0]
    print("\n========== 随机森林(RF) 的内部投票 ==========")
    print(f"Level 1 概率: {rf_probs[0]:.2%}")
    print(f"Level 2 概率: {rf_probs[1]:.2%}")
    print(f"Level 3 概率: {rf_probs[2]:.2%}")
    print(f"RF 最终决定: Level {rf_model.predict(scaled_features)[0]}")
except Exception as e:
    print("\nRF 分析失败:", e)