#!/usr/bin/env python3
"""
demo_presentation.py
====================
答辩演示脚本：端侧大模型自适应仲裁系统（Rich 终端版）

功能清单：
1) 剧本化场景演示（4个代表场景）
2) 自由交互模式（含链路可视化）
3) 物理状态实时注入（/temp, /batt）
4) 实验数据实时回溯（平均TPS、总耗电、云端卸载率）

说明：
- 本脚本直接复用项目现有主链路：
  arbitrator_core + device_simulator + feature_extractor
- Demo 仅负责终端展示与状态注入，不再维护独立仲裁逻辑。
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Dict, List, Tuple

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.table import Table
from rich.text import Text


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


from arbitrator_core import AdaptiveArbitrator as CoreAdaptiveArbitrator
from arbitrator_core import MODEL_ORDER
from device_simulator import DeviceSimulator
from ml_router_upgrade.feature_engineering import extract_features
from ml_router_upgrade.feature_extractor import predict_difficulty_with_trace


MODEL_COLORS = {
    "qwen-0.5b": "bright_green",
    "llama-1b": "bright_yellow",
    "qwen-1.5b": "bright_cyan",
    "Cloud-API": "bright_magenta",
}


class DemoRuntime:
    """演示期运行时：封装项目主链路与会话统计。"""

    def __init__(self) -> None:
        self.simulator = DeviceSimulator(battery=80.0, temperature=35.0)
        self.arbitrator = CoreAdaptiveArbitrator(
            device_simulator=self.simulator,
            routing_engine="hybrid",
        )
        self.records: List[Dict[str, object]] = []
        self.request_count = 0

    def safety_score(self) -> float:
        state = self.simulator.get_state()
        return self.arbitrator._calculate_safety_score(
            float(state["battery"]),
            float(state["temperature"]),
        )

    def inject_temperature(self, value: float) -> None:
        self.simulator.temperature = round(
            max(self.simulator.env_temperature, min(120.0, float(value))),
            2,
        )

    def inject_battery(self, value: float) -> None:
        self.simulator.battery = round(max(0.0, min(100.0, float(value))), 2)

    def preview_request(self, prompt: str) -> Tuple[List[float], int, str, str, float, str]:
        features = [float(x) for x in extract_features(prompt)]
        level, engine, flow = predict_difficulty_with_trace(prompt, engine="hybrid")
        score = self.safety_score()

        preview_chain: List[str] = []
        initial_model = self.arbitrator._select_initial_model(int(level), score, preview_chain)
        return features, int(level), str(engine), str(flow), score, initial_model

    @staticmethod
    def _extract_initial_model(call_chain: List[str]) -> str:
        if not call_chain:
            return "unknown"
        match = re.search(r"\[初始路由\]([^(]+)", str(call_chain[0]))
        return match.group(1).strip() if match else "unknown"

    @staticmethod
    def _extract_final_model(call_chain: List[str]) -> str:
        joined = " -> ".join(str(step) for step in call_chain)
        matches = re.findall(r"(qwen-0\.5b|llama-1b|qwen-1\.5b|Cloud-API)", joined)
        return matches[-1] if matches else "unknown"

    def process_request(self, prompt: str, purpose: str = "Live") -> Dict[str, object]:
        self.request_count += 1
        req_id = self.request_count

        state_before = self.simulator.get_state()
        score_start = self.safety_score()
        result = self.arbitrator.adaptive_process(prompt)
        state_after = self.simulator.get_state()

        call_chain = [str(x) for x in result.get("call_chain", [])]
        call_chain_text = " -> ".join(call_chain)
        initial_model = self._extract_initial_model(call_chain)
        final_model = self._extract_final_model(call_chain)
        cloud_used = "[云端兜底卸载]" in call_chain_text or final_model == "Cloud-API"

        record = {
            "id": req_id,
            "purpose": purpose,
            "prompt": prompt,
            "difficulty": int(result.get("difficulty", 1)),
            "decision_engine": str(result.get("decision_engine", "Unknown")),
            "decision_flow": str(result.get("decision_flow", "")),
            "features": [round(float(x), 4) for x in extract_features(prompt)],
            "safety_start": round(score_start, 4),
            "safety_end": round(float(result.get("safety_score", score_start)), 4),
            "initial_model": initial_model,
            "final_model": final_model,
            "call_chain": call_chain_text,
            "latency": round(float(result.get("total_latency", 0.0)), 4),
            "tps": round(float(result.get("avg_tps", 0.0)), 4),
            "answer": str(result.get("answer", "")),
            "cloud_used": cloud_used,
            "battery_before": round(float(state_before["battery"]), 2),
            "battery_after": round(float(state_after["battery"]), 2),
            "temperature_before": round(float(state_before["temperature"]), 2),
            "temperature_after": round(float(state_after["temperature"]), 2),
            "battery_delta": round(
                max(0.0, float(state_before["battery"]) - float(state_after["battery"])),
                4,
            ),
        }
        self.records.append(record)
        return record

    def session_metrics(self) -> Dict[str, float]:
        total = len(self.records)
        avg_tps = sum(float(r.get("tps", 0.0)) for r in self.records) / total if total else 0.0
        total_battery = sum(float(r.get("battery_delta", 0.0)) for r in self.records)
        cloud_count = sum(1 for r in self.records if bool(r.get("cloud_used", False)))
        cloud_rate = (cloud_count / total * 100.0) if total else 0.0
        return {
            "total_requests": float(total),
            "avg_tps": round(avg_tps, 4),
            "total_battery_drop": round(total_battery, 4),
            "cloud_offload_rate": round(cloud_rate, 2),
            "cloud_count": float(cloud_count),
        }


class DemoUI:
    """Rich 终端交互与可视化层。"""

    def __init__(self) -> None:
        self.console = Console()
        self.runtime = DemoRuntime()

    def _model_style(self, model_name: str) -> str:
        return MODEL_COLORS.get(model_name, "white")

    def render_status_panel(self) -> Panel:
        state = self.runtime.simulator.get_state()
        score = self.runtime.safety_score()
        score_style = "green" if score > 20 else ("yellow" if score > 0 else "red")

        metrics = self.runtime.session_metrics()
        text = Text()
        text.append("温度: ", style="bold")
        text.append(f"{state['temperature']:.2f}°C", style="bright_red")
        text.append("    电量: ", style="bold")
        text.append(f"{state['battery']:.2f}%", style="bright_green")
        text.append("    Safety Score: ", style="bold")
        text.append(f"{score:.2f}", style=f"bold {score_style}")
        text.append("\n模式: ", style="bold")
        text.append("项目主链路直连", style="cyan")
        text.append("    核心: ", style="bold")
        text.append("arbitrator_core + device_simulator", style="white")
        text.append("\n调用统计: ", style="bold")
        text.append(
            f"requests={int(metrics['total_requests'])} | cloud={int(metrics['cloud_count'])}",
            style="white",
        )

        return Panel(text, title="系统状态 / Status Bar", border_style="bright_blue")

    def render_stats_table(self, max_rows: int = 8) -> Table:
        metrics = self.runtime.session_metrics()

        table = Table(title="推理统计 / Inference Statistics", box=box.SIMPLE_HEAVY)
        table.add_column("请求ID", justify="right")
        table.add_column("场景")
        table.add_column("难度")
        table.add_column("仲裁引擎")
        table.add_column("最终模型")
        table.add_column("耗时(s)", justify="right")
        table.add_column("TPS", justify="right")
        table.add_column("云端", justify="center")

        for rec in self.runtime.records[-max_rows:]:
            model = str(rec["final_model"])
            table.add_row(
                str(rec["id"]),
                str(rec["purpose"]),
                f"L{rec['difficulty']}",
                str(rec["decision_engine"]),
                f"[{self._model_style(model)}]{model}[/{self._model_style(model)}]",
                f"{float(rec['latency']):.3f}",
                f"{float(rec['tps']):.2f}",
                "是" if bool(rec["cloud_used"]) else "否",
            )

        table.caption = (
            f"会话累计: 请求={int(metrics['total_requests'])} | 平均TPS={metrics['avg_tps']:.2f} | "
            f"总耗电={metrics['total_battery_drop']:.3f}% | 云端卸载率={metrics['cloud_offload_rate']:.2f}%"
        )
        return table

    def print_header(self) -> None:
        self.console.clear()
        self.console.print(
            Panel(
                "[bold cyan]端侧大模型自适应仲裁系统 - 答辩演示终端[/bold cyan]\n"
                "[white]演示层直连项目主链路：Rule + MLP + RF + 安全分降级 + 云端兜底[/white]",
                border_style="cyan",
            )
        )
        self.console.print(self.render_status_panel())
        self.console.print(self.render_stats_table())

    def _stream_answer(self, answer: str, style: str = "white") -> None:
        words = answer.split(" ")
        self.console.print("[bold]模型回复:[/bold] ", end="")
        for word in words:
            self.console.print(f"[{style}]{word}[/{style}]", end=" ")
            time.sleep(0.03)
        self.console.print()

    def _trace_and_run(self, prompt: str, purpose: str) -> Dict[str, object]:
        with Status("[cyan]🔍 特征提取中...[/cyan]", console=self.console, spinner="dots"):
            time.sleep(0.2)
            features, level, engine, flow, score, model = self.runtime.preview_request(prompt)

        self.console.print(
            f"[bold]🔍 特征提取[/bold] -> {', '.join(f'{v:.3f}' for v in features[:4])} ...",
            style="dim",
        )
        self.console.print(f"[bold]🧠 难度分级[/bold] -> Level {level} | 引擎={engine}")
        self.console.print(f"[bold]判定流程[/bold] -> {flow}", style="bright_black")
        self.console.print(f"[bold]🛡️ 安全分计算[/bold] -> score={score:.2f}")

        mstyle = self._model_style(model)
        self.console.print(f"[bold]⚙️ 选模决策[/bold] -> [{mstyle}]{model}[/{mstyle}]")

        with Status("[green]🚀 调用项目主链路执行仲裁...[/green]", console=self.console, spinner="line"):
            result = self.runtime.process_request(prompt=prompt, purpose=purpose)
            time.sleep(0.1)

        final_model = str(result["final_model"])
        fstyle = self._model_style(final_model)
        self.console.print(f"[bold]最终执行模型[/bold] -> [{fstyle}]{final_model}[/{fstyle}]")
        self.console.print(f"[bold]真实调用链路[/bold] -> {result['call_chain']}")
        self.console.print(
            f"[bold]耗时/TPS[/bold] -> {float(result['latency']):.3f}s / {float(result['tps']):.2f}"
        )

        self._stream_answer(str(result["answer"]), style=fstyle)
        return result

    def scenario_showcase(self) -> None:
        scenarios = [
            {
                "title": "[简单任务] Rule 拦截 -> qwen-0.5b",
                "purpose": "验证短文本通过 Rule 防线，并走最低算力模型。",
                "expected": "判定 Level 1，优先 qwen-0.5b，低耗时低能耗。",
                "prompt": "请回复收到",
                "prepare": lambda: None,
            },
            {
                "title": "[复杂逻辑] MLP + RF 级联判定 Level 3",
                "purpose": "展示难题经双引擎复核后被判为高难。",
                "expected": "判定流程体现 MLP->RF 级联，优先较大模型。",
                "prompt": "请证明 O(log n) 并比较快速排序与归并排序复杂度。",
                "prepare": lambda: None,
            },
            {
                "title": "[物理降级] 高温下限制高算力",
                "purpose": "验证高温导致 Safety Score 下降后，上界模型会被限制。",
                "expected": "链路会出现受限降级，无法盲目上探到更大模型。",
                "prompt": "请解释什么是机器学习中的过拟合。",
                "prepare": lambda: self.runtime.inject_temperature(60.0),
            },
            {
                "title": "[质量闭环] 不合格回复优先本地升级",
                "purpose": "验证回复不合格时，系统先尝试端侧升级，只有达到允许上界后才云端兜底。",
                "expected": "若首答不达标，链路应先出现 [升级重试]，而不是直接云端回复。",
                "prompt": "请你简单叙述一下数据结构中链表的原理",
                "prepare": lambda: (
                    self.runtime.inject_temperature(45.0),
                    self.runtime.inject_battery(80.0),
                ),
            },
        ]

        self.console.print("\n[bold cyan]🎬 剧本化场景演示开始[/bold cyan]")
        for idx, scenario in enumerate(scenarios, start=1):
            scenario["prepare"]()
            panel = Panel(
                f"[bold]测试目的[/bold]: {scenario['purpose']}\n"
                f"[bold]预期结果[/bold]: {scenario['expected']}\n"
                f"[bold]输入Prompt[/bold]: {scenario['prompt']}",
                title=f"场景 {idx} - {scenario['title']}",
                border_style="magenta",
            )
            self.console.print(panel)
            self._trace_and_run(
                prompt=str(scenario["prompt"]),
                purpose=f"Scenario-{idx}",
            )
            self.console.print(self.render_status_panel())
            time.sleep(0.25)

        self.console.print("[bold green]✅ 场景演示完成。[/bold green]")

    def _apply_state_command(self, command: str) -> bool:
        cmd = (command or "").strip()
        if not cmd:
            return False

        if cmd.startswith("/temp"):
            parts = cmd.split()
            if len(parts) != 2:
                self.console.print("[red]格式错误：/temp 60[/red]")
                return True
            try:
                self.runtime.inject_temperature(float(parts[1]))
                self.console.print("[green]温度已更新。[/green]")
            except ValueError:
                self.console.print("[red]温度值非法。[/red]")
            self.console.print(self.render_status_panel())
            return True

        if cmd.startswith("/batt"):
            parts = cmd.split()
            if len(parts) != 2:
                self.console.print("[red]格式错误：/batt 10[/red]")
                return True
            try:
                self.runtime.inject_battery(float(parts[1]))
                self.console.print("[green]电量已更新。[/green]")
            except ValueError:
                self.console.print("[red]电量值非法。[/red]")
            self.console.print(self.render_status_panel())
            return True

        return False

    def live_terminal(self) -> None:
        self.console.print(
            Panel(
                "进入自由交互模式。\n"
                "- 输入普通文本执行仲裁\n"
                "- /temp 60  修改温度\n"
                "- /batt 10  修改电量\n"
                "- /back     返回主菜单",
                title="💬 Live Terminal",
                border_style="bright_blue",
            )
        )

        while True:
            user_input = Prompt.ask("[bold cyan]你[/bold cyan]")
            if not user_input.strip():
                continue
            if user_input.strip().lower() == "/back":
                break

            if self._apply_state_command(user_input):
                continue

            self._trace_and_run(prompt=user_input, purpose="Live")
            self.console.print(self.render_status_panel())

    def state_injection_menu(self) -> None:
        self.console.print(
            Panel(
                "输入示例：\n"
                "- /temp 60\n"
                "- /batt 15\n"
                "输入 /back 返回主菜单",
                title="🛠️ 物理状态实时注入",
                border_style="yellow",
            )
        )
        while True:
            cmd = Prompt.ask("[bold yellow]注入命令[/bold yellow]")
            if cmd.strip().lower() == "/back":
                break
            handled = self._apply_state_command(cmd)
            if not handled:
                self.console.print("[red]不支持的命令。请使用 /temp 或 /batt。[/red]")

    def analytics_review(self) -> None:
        metrics = self.runtime.session_metrics()

        summary = Table(title="📊 实验数据实时回溯", box=box.ROUNDED)
        summary.add_column("指标", style="bold")
        summary.add_column("值", justify="right")
        summary.add_row("总请求数", str(int(metrics["total_requests"])))
        summary.add_row("平均TPS", f"{metrics['avg_tps']:.2f}")
        summary.add_row("总电量消耗(%)", f"{metrics['total_battery_drop']:.3f}")
        summary.add_row("云端卸载次数", str(int(metrics["cloud_count"])))
        summary.add_row("云端卸载率(%)", f"{metrics['cloud_offload_rate']:.2f}")
        self.console.print(summary)

        if not self.runtime.records:
            self.console.print("[yellow]当前会话暂无请求记录。[/yellow]")
            return

        detail = Table(title="最近请求明细", box=box.MINIMAL_DOUBLE_HEAD)
        detail.add_column("ID", justify="right")
        detail.add_column("难度")
        detail.add_column("仲裁引擎")
        detail.add_column("最终模型")
        detail.add_column("调用链路")
        detail.add_column("耗时(s)", justify="right")
        detail.add_column("TPS", justify="right")

        for rec in self.runtime.records[-12:]:
            model = str(rec["final_model"])
            mstyle = self._model_style(model)
            detail.add_row(
                str(rec["id"]),
                f"L{rec['difficulty']}",
                str(rec["decision_engine"]),
                f"[{mstyle}]{model}[/{mstyle}]",
                str(rec["call_chain"]),
                f"{float(rec['latency']):.3f}",
                f"{float(rec['tps']):.2f}",
            )
        self.console.print(detail)

    def main_menu(self) -> None:
        while True:
            self.print_header()
            menu = Table(title="主菜单", box=box.SIMPLE)
            menu.add_column("编号", justify="center", style="bold cyan")
            menu.add_column("功能")
            menu.add_row("1", "🎬 剧本化场景演示 (Scenario Showcase)")
            menu.add_row("2", "💬 自由交互模式 (Live Terminal)")
            menu.add_row("3", "🛠️ 物理状态实时注入 (State Injection)")
            menu.add_row("4", "📊 实验数据实时回溯")
            menu.add_row("0", "退出")
            self.console.print(menu)

            choice = Prompt.ask("请选择功能", choices=["1", "2", "3", "4", "0"], default="1")
            if choice == "1":
                self.scenario_showcase()
            elif choice == "2":
                self.live_terminal()
            elif choice == "3":
                self.state_injection_menu()
            elif choice == "4":
                self.analytics_review()
            elif choice == "0":
                self.console.print("[bold green]演示结束，感谢使用。[/bold green]")
                break

            Prompt.ask("\n按回车返回主菜单", default="")


def main() -> None:
    ui = DemoUI()
    ui.main_menu()


if __name__ == "__main__":
    main()
