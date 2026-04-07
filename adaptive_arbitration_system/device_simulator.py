"""
device_simulator.py
-------------------
本模块用于模拟端侧设备状态：
- 电量（battery）按“功率 * 推理耗时”的动态能耗模型衰减。
- 温度（temperature）按任务能量输入升高，并叠加牛顿冷却模型散热。

核心说明：
- 引入基于时间的动态能耗与冷却模型，替代固定扣电/固定升温逻辑。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class DeviceSimulator:
    """
    端侧设备状态仿真器（Time-Aware 版本）。

    初始状态：
    - battery = 100.0 (%)
    - temperature = 35.0 (℃)

    模型额定功率（W）：
    - qwen-0.5b: 5.0
    - llama-1b: 10.0
    - qwen-1.5b: 20.0

    状态更新公式：
    - E = Power * inference_time_seconds
    - battery = max(0, battery - E * 0.015)
    - temp_rise = E * 0.04
    - T_next = T_env + (T_curr - T_env) * exp(-k)
      其中 T_curr 为“升温后的温度”，T_env=25.0，k=0.1
    """

    battery: float = 100.0
    temperature: float = 35.0
    env_temperature: float = 25.0
    cooling_k: float = 0.1

    # 不同模型额定功率（单位：W）
    model_power_watts: Dict[str, float] = field(
        default_factory=lambda: {
            "qwen-0.5b": 5.0,
            "llama-1b": 10.0,
            "qwen-1.5b": 20.0,
        }
    )

    def _apply_newton_cooling(self, temp_curr: float) -> float:
        """
        牛顿冷却（离散步）：
        T_next = T_env + (T_curr - T_env) * e^(-k)
        """
        temp_next = self.env_temperature + (temp_curr - self.env_temperature) * math.exp(-self.cooling_k)
        return max(self.env_temperature, temp_next)

    def update_state(self, model_name: str, inference_time_seconds: float) -> None:
        """
        在每次模型调用后更新设备状态（闭环入口）。

        参数：
        - model_name: 模型名称
        - inference_time_seconds: 本次推理耗时（秒）
        """
        if model_name not in self.model_power_watts:
            raise ValueError(f"未知模型：{model_name}")

        # 防御性处理：避免负耗时导致“反向充电/降温”。
        elapsed = max(0.0, float(inference_time_seconds))
        power = self.model_power_watts[model_name]

        # 动态能耗计算：E = P * t
        energy = power * elapsed

        # 电量衰减：battery = max(0, battery - E * 0.015)
        self.battery = max(0.0, self.battery - energy * 0.002)

        # 温度上升后再执行自然散热，模拟“发热 + 冷却”叠加过程。
        temp_after_rise = self.temperature + energy * 0.025
        self.temperature = self._apply_newton_cooling(temp_after_rise)

        # 统一保留两位小数，便于日志和 CSV 可读性。
        self.battery = round(self.battery, 2)
        self.temperature = round(self.temperature, 2)

    def cool_down(self) -> None:
        """
        额外散热接口（兼容旧跑批脚本）。

        说明：
        - 该方法不再绑定固定降温值，而是沿用牛顿冷却。
        - update_state 已内置一次冷却；该方法可用于“请求间隙”的补充散热。
        """
        self.temperature = round(self._apply_newton_cooling(self.temperature), 2)

    def get_state(self) -> Dict[str, float]:
        """
        返回当前设备状态。

        返回：
        - battery: 当前电量百分比（0~100）
        - temperature: 当前温度（℃）
        """
        return {
            "battery": round(self.battery, 2),
            "temperature": round(self.temperature, 2),
        }
