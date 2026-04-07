"""
experiment_analytics.py
=======================
终极场景可视化与汇总引擎。

功能：
1. 自动遍历 output/scenario_* 目录
2. 对每个场景读取 results_rf.csv / results_mlp.csv
3. 生成对比图：
   - compare_hardware_decay.png
   - compare_energy_scatter.png
   - compare_latency_boxplot.png
   - routing_distribution.png
4. 生成场景汇总表：
   - scenario_summary.md

设计原则：
- 高内聚：每个场景内图表与表格一起产出
- 低耦合：仅依赖标准化 CSV 字段，不与训练过程强绑定
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns


# ===== 全局字体与样式（按需求严格配置）=====
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
    稳健配置中文字体：
    1) 尝试显式注册常见字体文件路径（防止 Matplotlib 缓存未刷新）
    2) 在 seaborn.set_theme 中注入字体 rc（防止被默认样式覆盖）
    3) set_theme 后再次覆盖 rcParams（双保险）
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


class ScenarioVisualizer:
    """
    场景可视化器：负责逐场景生成对比图与摘要表。
    """

    def __init__(self, output_root: str) -> None:
        self.output_root = output_root
        if not os.path.exists(self.output_root):
            raise FileNotFoundError(f"输出目录不存在：{self.output_root}")

    @staticmethod
    def _extract_qid_num(qid: object) -> int:
        match = re.search(r"(\d+)", str(qid))
        return int(match.group(1)) if match else 0

    @staticmethod
    def _extract_final_model(chain: object) -> str:
        matches = re.findall(r"(qwen-0\.5b|llama-1b|qwen-1\.5b)", str(chain or ""))
        return matches[-1] if matches else "unknown"

    @staticmethod
    def _count_keyword(chain: object, keyword: str) -> int:
        return str(chain or "").count(keyword)

    def _load_pair(self, scenario_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rf_path = os.path.join(scenario_dir, "results_rf.csv")
        mlp_path = os.path.join(scenario_dir, "results_mlp.csv")
        if not os.path.exists(rf_path) or not os.path.exists(mlp_path):
            raise FileNotFoundError(f"缺少结果文件：{rf_path} 或 {mlp_path}")

        rf = pd.read_csv(rf_path, encoding="utf-8-sig")
        mlp = pd.read_csv(mlp_path, encoding="utf-8-sig")

        for name, df in [("RF", rf), ("MLP", mlp)]:
            required_cols = [
                "题号",
                "系统预估难度",
                "真实难度标签",
                "真实调用链路",
                "最终总耗时",
                "执行后电量",
                "执行后温度",
            ]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"{name} 结果缺少列：{missing}")

            for col in ["系统预估难度", "真实难度标签", "最终总耗时", "执行后电量", "执行后温度"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["题号数字"] = df["题号"].apply(self._extract_qid_num)
            df["最终调用模型"] = df["真实调用链路"].apply(self._extract_final_model)

        rf["引擎"] = "RF"
        mlp["引擎"] = "MLP"
        return rf, mlp

    @staticmethod
    def _calc_energy_drop(df: pd.DataFrame) -> pd.DataFrame:
        """
        反推单次耗电量：前一题电量 - 本题电量。
        """
        out = df.sort_values("题号数字").copy()
        out["前一题电量"] = out["执行后电量"].shift(1).fillna(100.0)
        out["单次耗电量"] = (out["前一题电量"] - out["执行后电量"]).clip(lower=0.0)
        return out

    def plot_hardware_decay(self, rf: pd.DataFrame, mlp: pd.DataFrame, scenario_dir: str) -> str:
        """
        对比硬件衰减图：
        - 电量（两条实线）
        - 温度（两条虚线）
        """
        rf_sorted = rf.sort_values("题号数字")
        mlp_sorted = mlp.sort_values("题号数字")

        fig, ax1 = plt.subplots(figsize=(10.5, 4.8))
        ax2 = ax1.twinx()

        l1, = ax1.plot(rf_sorted["题号数字"], rf_sorted["执行后电量"], "-", color="#1f77b4", label="RF 电量")
        l2, = ax1.plot(mlp_sorted["题号数字"], mlp_sorted["执行后电量"], "-", color="#ff7f0e", label="MLP 电量")
        l3, = ax2.plot(rf_sorted["题号数字"], rf_sorted["执行后温度"], "--", color="#1f77b4", label="RF 温度")
        l4, = ax2.plot(mlp_sorted["题号数字"], mlp_sorted["执行后温度"], "--", color="#ff7f0e", label="MLP 温度")

        ax1.set_title("RF 与 MLP 温度/电量双Y轴衰减对比")
        ax1.set_xlabel("题号")
        ax1.set_ylabel("执行后电量（%）")
        ax2.set_ylabel("执行后温度（°C）")

        handles = [l1, l2, l3, l4]
        labels = [h.get_label() for h in handles]
        ax1.legend(handles, labels, loc="best", frameon=False)
        ax1.grid(alpha=0.25, linestyle="--")

        out = os.path.join(scenario_dir, "compare_hardware_decay.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_energy_time_scatter(self, rf: pd.DataFrame, mlp: pd.DataFrame, scenario_dir: str) -> str:
        """
        耗时-耗电散点图：
        - 横轴：真实耗时
        - 纵轴：单次耗电量
        - 颜色：最终调用模型
        - 点形：RF / MLP
        """
        rf_energy = self._calc_energy_drop(rf)
        mlp_energy = self._calc_energy_drop(mlp)
        merged = pd.concat([rf_energy, mlp_energy], ignore_index=True)

        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        sns.scatterplot(
            data=merged,
            x="最终总耗时",
            y="单次耗电量",
            hue="最终调用模型",
            style="引擎",
            palette={
                "qwen-0.5b": "#4C72B0",
                "llama-1b": "#55A868",
                "qwen-1.5b": "#C44E52",
                "unknown": "#999999",
            },
            s=52,
            alpha=0.85,
            ax=ax,
        )

        ax.set_title("耗时-耗电散点对比（模型与引擎分组）")
        ax.set_xlabel("真实耗时（秒）")
        ax.set_ylabel("单次耗电量（电量百分比下降）")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(title="模型 / 引擎", frameon=False, ncol=2)

        out = os.path.join(scenario_dir, "compare_energy_scatter.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_latency_boxplot(self, rf: pd.DataFrame, mlp: pd.DataFrame, scenario_dir: str) -> str:
        """
        耗时分布箱线图：
        - 横轴：真实难度
        - 纵轴：总耗时
        - 分组：RF / MLP
        """
        merged = pd.concat([rf, mlp], ignore_index=True).copy()
        merged = merged.dropna(subset=["真实难度标签", "最终总耗时"])
        merged["难度分组"] = merged["真实难度标签"].astype(int).map({1: "简单", 2: "中等", 3: "困难"})

        fig, ax = plt.subplots(figsize=(8.8, 4.9))
        sns.boxplot(
            data=merged,
            x="难度分组",
            y="最终总耗时",
            hue="引擎",
            order=["简单", "中等", "困难"],
            palette={"RF": "#4C72B0", "MLP": "#DD8452"},
            ax=ax,
        )

        ax.set_title("不同难度下 RF 与 MLP 总耗时分布")
        ax.set_xlabel("真实难度等级")
        ax.set_ylabel("总耗时（秒）")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(title="引擎", frameon=False)

        out = os.path.join(scenario_dir, "compare_latency_boxplot.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_routing_distribution(self, rf: pd.DataFrame, mlp: pd.DataFrame, scenario_dir: str) -> str:
        """
        路由分布图（饼图组合）：
        - 左：RF 最终模型占比
        - 右：MLP 最终模型占比
        """
        model_order = ["qwen-0.5b", "llama-1b", "qwen-1.5b"]
        rf_count = rf["最终调用模型"].value_counts().reindex(model_order, fill_value=0)
        mlp_count = mlp["最终调用模型"].value_counts().reindex(model_order, fill_value=0)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].pie(
            rf_count.values,
            labels=rf_count.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#4C72B0", "#55A868", "#C44E52"],
        )
        axes[0].set_title("RF 路由分布")

        axes[1].pie(
            mlp_count.values,
            labels=mlp_count.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#4C72B0", "#55A868", "#C44E52"],
        )
        axes[1].set_title("MLP 路由分布")

        fig.suptitle("最终模型路由占比对比")
        fig.tight_layout()

        out = os.path.join(scenario_dir, "routing_distribution.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    @staticmethod
    def _aggregate_summary(df: pd.DataFrame) -> Dict[str, float]:
        """
        汇总单个引擎在一个场景中的核心指标。
        """
        sorted_df = df.sort_values("题号数字")
        total_time = float(sorted_df["最终总耗时"].sum())
        final_battery = float(sorted_df["执行后电量"].iloc[-1]) if len(sorted_df) > 0 else 100.0
        total_battery_drop = max(0.0, 100.0 - final_battery)
        max_temp = float(sorted_df["执行后温度"].max()) if len(sorted_df) > 0 else 0.0

        throttle_cnt = int(sorted_df["真实调用链路"].astype(str).map(lambda x: x.count("[受限降级]")).sum())
        quality_retry_cnt = int(sorted_df["真实调用链路"].astype(str).map(lambda x: x.count("[质量拦截:")).sum())

        return {
            "总耗时": total_time,
            "总耗电": total_battery_drop,
            "最高温度": max_temp,
            "级联降频触发次数": throttle_cnt,
            "质量重试触发次数": quality_retry_cnt,
        }

    def generate_summary_table(self, rf: pd.DataFrame, mlp: pd.DataFrame, scenario_dir: str) -> str:
        """
        生成场景摘要 Markdown 表。
        """
        rf_sum = self._aggregate_summary(rf)
        mlp_sum = self._aggregate_summary(mlp)

        lines = [
            "| 引擎 | 总耗时(秒) | 总耗电(%) | 最高温度(°C) | 级联降频触发次数 | 质量重试触发次数 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| RF | {rf_sum['总耗时']:.4f} | {rf_sum['总耗电']:.4f} | {rf_sum['最高温度']:.2f} | {int(rf_sum['级联降频触发次数'])} | {int(rf_sum['质量重试触发次数'])} |",
            f"| MLP | {mlp_sum['总耗时']:.4f} | {mlp_sum['总耗电']:.4f} | {mlp_sum['最高温度']:.2f} | {int(mlp_sum['级联降频触发次数'])} | {int(mlp_sum['质量重试触发次数'])} |",
        ]
        content = "\n".join(lines) + "\n"

        out = os.path.join(scenario_dir, "scenario_summary.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write(content)
        return out

    def process_one_scenario(self, scenario_dir: str) -> List[str]:
        """
        处理单个场景目录，返回生成文件路径列表。
        """
        rf, mlp = self._load_pair(scenario_dir)

        outputs = [
            self.plot_hardware_decay(rf, mlp, scenario_dir),
            self.plot_energy_time_scatter(rf, mlp, scenario_dir),
            self.plot_latency_boxplot(rf, mlp, scenario_dir),
            self.plot_routing_distribution(rf, mlp, scenario_dir),
            self.generate_summary_table(rf, mlp, scenario_dir),
        ]
        return outputs

    def run_all(self) -> Dict[str, List[str]]:
        """
        自动遍历 output/scenario_* 并逐一生成结果。
        """
        scenario_dirs = [
            os.path.join(self.output_root, d)
            for d in sorted(os.listdir(self.output_root))
            if d.startswith("scenario_") and os.path.isdir(os.path.join(self.output_root, d))
        ]

        if not scenario_dirs:
            raise RuntimeError("未找到任何 scenario_ 目录，请先运行 run_all_scenarios_benchmark.py。")

        summary: Dict[str, List[str]] = {}
        for sdir in scenario_dirs:
            name = os.path.basename(sdir)
            print(f"[INFO] 正在生成场景图表：{name}")
            summary[name] = self.process_one_scenario(sdir)
        return summary


def main() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.join(project_root, "output")

    visualizer = ScenarioVisualizer(output_root=output_root)
    summary = visualizer.run_all()

    print("\n=== 场景可视化与摘要生成完成 ===")
    for scenario, files in summary.items():
        print(f"\n[{scenario}]")
        for f in files:
            print(f" - {f}")


if __name__ == "__main__":
    main()
