"""
device_simulator.py
-------------------
本模块用于模拟端侧设备状态：
- 电量（battery）随模型调用逐步衰减。
- 温度（temperature）随模型调用逐步升高。
- 每个请求结束后可调用 cool_down() 模拟自然降温。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class DeviceSimulator:
    """
    端侧设备状态仿真器。

    初始状态：
    - battery = 100.0 (%)
    - temperature = 35.0 (℃)
    """

    battery: float = 100.0
    temperature: float = 35.0

    # 不同模型对硬件状态的影响：
    # 映射关系为 model_name -> (battery_drop, temperature_rise)
    model_impact: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "qwen-0.5b": (0.5, 0.5),
            "llama-1b": (1.5, 1.5),
            "qwen-1.5b": (3.0, 3.0),
        }
    )

    def update_state(self, model_name: str) -> None:
        """
        在每次模型调用后更新设备状态。

        规则：
        - qwen-0.5b：battery -= 0.5, temperature += 0.5
        - llama-1b： battery -= 1.5, temperature += 1.5
        - qwen-1.5b：battery -= 3.0, temperature += 3.0
        """
        if model_name not in self.model_impact:
            raise ValueError(f"未知模型：{model_name}")

        battery_drop, temperature_rise = self.model_impact[model_name]
        self.battery = max(0.0, round(self.battery - battery_drop, 2))
        self.temperature = round(self.temperature + temperature_rise, 2)

    def cool_down(self) -> None:
        """
        每次完整请求结束后执行一次降温。

        规则：
        - temperature 降低 1.0℃
        - 最低不低于 35.0℃
        """
        self.temperature = round(max(35.0, self.temperature - 1.0), 2)

    def get_hardware_constraint(self) -> str:
        """
        根据当前电量与温度评估硬件约束等级。

        约束规则（按优先级从高到低）：
        1) 若 temperature >= 45.0 或 battery <= 15.0
           -> "THROTTLED_05B"（严重受限）

        2) 若 temperature >= 40.0 或 battery <= 30.0
           -> "THROTTLED_1B"（轻度受限）

        3) 其余
           -> "NORMAL"
        """
        if self.temperature >= 45.0 or self.battery <= 15.0:
            return "THROTTLED_05B"

        if self.temperature >= 40.0 or self.battery <= 30.0:
            return "THROTTLED_1B"

        return "NORMAL"
