#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/root/LAA/adaptive_arbitration_system")
OUTPUT_ROOT = ROOT / "output"
FIGURE_ROOT = ROOT / "generated_figures"

SCENARIOS = {
    "A": {
        "title_line": "场景A：轻负载",
        "title_pie": "模型调用分布 (场景A)",
        "folder": OUTPUT_ROOT / "scenario_A_light",
    },
    "B": {
        "title_line": "场景B：重负载",
        "title_pie": "模型调用分布 (场景B)",
        "folder": OUTPUT_ROOT / "scenario_B_heavy",
    },
    "C": {
        "title_line": "场景C：幻觉敏感",
        "title_pie": "模型调用分布 (场景C)",
        "folder": OUTPUT_ROOT / "scenario_C_hallucination",
    },
    "D": {
        "title_line": "场景D：冷热脉冲",
        "title_pie": "模型调用分布 (场景D)",
        "folder": OUTPUT_ROOT / "scenario_D_burst",
    },
}

MODEL_ORDER = ["qwen-0.5b", "llama-1b", "qwen-1.5b"]
MODEL_COLORS = {
    "qwen-0.5b": "#D9D9D9",
    "llama-1b": "#CDB7E9",
    "qwen-1.5b": "#5A2A83",
}

COLOR_BATTERY = "#1f77b4"
COLOR_TEMP = "#d62728"
COLOR_HYBRID = "#5A2A83"
COLOR_DIRECT = "#D0D0D0"
REDLINE_COLOR = "#C00000"


def pick_chinese_font() -> str:
    candidate_paths = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]
    for path in candidate_paths:
        if path.exists():
            fm.fontManager.addfont(str(path))
            return fm.FontProperties(fname=str(path)).get_name()

    candidates = [
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
        "WenQuanYi Micro Hei",
        "Droid Sans Fallback",
        "SimHei",
        "SimSun",
        "Microsoft YaHei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


def configure_matplotlib() -> None:
    font_name = pick_chinese_font()
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_name, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
        }
    )


def ensure_output_dir() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def scenario_csv(scenario_key: str, kind: str) -> Path:
    suffix = "results_hybrid.csv" if kind == "hybrid" else "results_qwen15b_direct.csv"
    return SCENARIOS[scenario_key]["folder"] / suffix


def parse_question_numbers(series: Iterable[object]) -> np.ndarray:
    values: list[int] = []
    for idx, item in enumerate(series, start=1):
        text = str(item)
        match = re.search(r"(\d+)$", text)
        values.append(int(match.group(1)) if match else idx)
    return np.asarray(values)


def apply_academic_axes_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E8E8E8", linestyle="-", linewidth=0.8)
    ax.tick_params(axis="both", direction="out", length=4, width=0.8)


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGURE_ROOT / filename, dpi=300)
    plt.close(fig)


def plot_dual_axis_line(
    df: pd.DataFrame,
    title: str,
    filename: str,
    battery_ylim: tuple[float, float] | None = None,
    temp_ylim: tuple[float, float] | None = None,
) -> None:
    x = parse_question_numbers(df["题号"])
    battery = df["执行后电量"].astype(float).to_numpy()
    temp = df["执行后温度"].astype(float).to_numpy()

    fig, ax1 = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        x,
        battery,
        color=COLOR_BATTERY,
        linewidth=2.0,
        linestyle="-",
        solid_capstyle="round",
        solid_joinstyle="round",
        antialiased=True,
        label="电量",
    )[0]
    line2 = ax2.plot(
        x,
        temp,
        color=COLOR_TEMP,
        linewidth=2.0,
        linestyle="-",
        solid_capstyle="round",
        solid_joinstyle="round",
        antialiased=True,
        label="温度",
    )[0]

    ax1.set_title(title, pad=12)
    ax1.set_xlabel("请求序号")
    ax1.set_ylabel("电量 (%)", color=COLOR_BATTERY)
    ax2.set_ylabel("温度 (°C)", color=COLOR_TEMP)

    ax1.set_xlim(1, len(x))
    ax1.set_xticks(np.arange(1, len(x) + 1, 3))

    if battery_ylim:
        ax1.set_ylim(*battery_ylim)
    else:
        b_min, b_max = float(battery.min()), float(battery.max())
        ax1.set_ylim(max(0, b_min - 2), min(100, b_max + 2))

    if temp_ylim:
        ax2.set_ylim(*temp_ylim)
    else:
        t_min, t_max = float(temp.min()), float(temp.max())
        pad = max(2.0, (t_max - t_min) * 0.08)
        ax2.set_ylim(t_min - pad, t_max + pad)

    apply_academic_axes_style(ax1)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", direction="out", length=4, width=0.8, colors=COLOR_TEMP)
    ax1.tick_params(axis="y", colors=COLOR_BATTERY)

    legend = ax1.legend(
        handles=[line1, line2],
        labels=["电量", "温度"],
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#BBBBBB",
    )
    legend.get_frame().set_linewidth(0.8)

    save_figure(fig, filename)


def autopct_generator(values: list[int]) -> callable:
    total = sum(values)

    def inner(pct: float) -> str:
        if pct <= 0 or total == 0:
            return ""
        return f"{pct:.1f}%"

    return inner


def plot_model_distribution_pie(df: pd.DataFrame, title: str, filename: str, external_legend: bool) -> None:
    counts = df["最终调用模型"].value_counts()
    labels = [model for model in MODEL_ORDER if model in counts.index]
    values = [int(counts[label]) for label in labels]
    colors = [MODEL_COLORS[label] for label in labels]

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    wedges, _, autotexts = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=autopct_generator(values),
        pctdistance=0.75,
        wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
        textprops={"fontsize": 10.5, "color": "black"},
    )
    for text in autotexts:
        text.set_fontsize(10.5)
        text.set_color("black")

    ax.set_title(title, pad=14)
    ax.set_aspect("equal")

    legend_labels = [f"{label} ({value}次)" for label, value in zip(labels, values)]
    if external_legend:
        ax.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
    else:
        ax.legend(
            wedges,
            legend_labels,
            loc="lower right",
            frameon=False,
        )

    save_figure(fig, filename)


def compute_summary_values(kind: str, metric: str) -> list[float]:
    values: list[float] = []
    for scenario_key in ["A", "B", "C", "D"]:
        df = read_csv(scenario_csv(scenario_key, kind))
        if metric == "latency":
            values.append(float(df["最终总耗时"].astype(float).sum()))
        elif metric == "latency_mean":
            values.append(float(df["最终总耗时"].astype(float).mean()))
        elif metric == "battery":
            values.append(100.0 - float(df["执行后电量"].astype(float).min()))
        elif metric == "temp_mean":
            values.append(float(df["执行后温度"].astype(float).mean()))
        elif metric == "temp":
            values.append(float(df["执行后温度"].astype(float).max()))
        elif metric == "tps_mean":
            values.append(float(df["平均TPS"].astype(float).mean()))
        else:
            raise ValueError(f"未知指标: {metric}")
    return values


def annotate_bars(ax: plt.Axes, bars: Iterable, fmt: str = "{:.1f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def plot_grouped_bar(
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    redline: float | None = None,
    value_fmt: str = "{:.1f}",
) -> None:
    hybrid = compute_summary_values("hybrid", metric)
    direct = compute_summary_values("direct", metric)

    categories = ["场景A", "场景B", "场景C", "场景D"]
    x = np.arange(len(categories))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    bars1 = ax.bar(x - width / 2, hybrid, width, color=COLOR_HYBRID, label="自适应仲裁")
    bars2 = ax.bar(x + width / 2, direct, width, color=COLOR_DIRECT, label="直连 1.5B 基线")

    ax.set_title(title, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    apply_academic_axes_style(ax)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#BBBBBB")

    ymax = max(max(hybrid), max(direct))
    if redline is not None:
        ymax = max(ymax, redline)
        ax.axhline(redline, color=REDLINE_COLOR, linestyle="--", linewidth=1.6)
        ax.text(
            x[-1] + 0.42,
            redline + max(0.8, ymax * 0.01),
            "85°C 降频红线",
            color=REDLINE_COLOR,
            fontsize=10.5,
            ha="right",
            va="bottom",
        )
    ax.set_ylim(0, ymax * 1.16)

    annotate_bars(ax, bars1, fmt=value_fmt)
    annotate_bars(ax, bars2, fmt=value_fmt)
    save_figure(fig, filename)


def main() -> None:
    configure_matplotlib()
    ensure_output_dir()

    # 第一组：双轴折线图
    plot_dual_axis_line(
        read_csv(scenario_csv("A", "hybrid")),
        title=SCENARIOS["A"]["title_line"],
        filename="01_scenario_A_battery_temperature.png",
        battery_ylim=(90, 100),
        temp_ylim=(20, 40),
    )
    plot_dual_axis_line(
        read_csv(scenario_csv("B", "hybrid")),
        title=SCENARIOS["B"]["title_line"],
        filename="02_scenario_B_battery_temperature.png",
    )
    plot_dual_axis_line(
        read_csv(scenario_csv("C", "hybrid")),
        title=SCENARIOS["C"]["title_line"],
        filename="03_scenario_C_battery_temperature.png",
    )
    plot_dual_axis_line(
        read_csv(scenario_csv("D", "hybrid")),
        title=SCENARIOS["D"]["title_line"],
        filename="04_scenario_D_battery_temperature.png",
    )

    # 第二组：饼图
    plot_model_distribution_pie(
        read_csv(scenario_csv("A", "hybrid")),
        title=SCENARIOS["A"]["title_pie"],
        filename="05_scenario_A_model_distribution.png",
        external_legend=False,
    )
    plot_model_distribution_pie(
        read_csv(scenario_csv("B", "hybrid")),
        title=SCENARIOS["B"]["title_pie"],
        filename="06_scenario_B_model_distribution.png",
        external_legend=False,
    )
    plot_model_distribution_pie(
        read_csv(scenario_csv("C", "hybrid")),
        title=SCENARIOS["C"]["title_pie"],
        filename="07_scenario_C_model_distribution.png",
        external_legend=True,
    )
    plot_model_distribution_pie(
        read_csv(scenario_csv("D", "hybrid")),
        title=SCENARIOS["D"]["title_pie"],
        filename="08_scenario_D_model_distribution.png",
        external_legend=True,
    )

    # 第三组：收益柱状图
    plot_grouped_bar(
        metric="latency",
        title="总时延收益对比",
        ylabel="时延 (秒)",
        filename="09_total_latency_comparison.png",
    )
    plot_grouped_bar(
        metric="battery",
        title="总耗电量收益对比",
        ylabel="耗电量 (%)",
        filename="10_total_battery_comparison.png",
    )
    plot_grouped_bar(
        metric="temp",
        title="峰值温度收益对比",
        ylabel="温度 (°C)",
        filename="11_peak_temperature_comparison.png",
        redline=85.0,
    )
    plot_grouped_bar(
        metric="tps_mean",
        title="平均生成速度 (TPS) 收益对比",
        ylabel="平均生成速度 (TPS)",
        filename="12_average_tps_comparison.png",
    )
    plot_grouped_bar(
        metric="temp_mean",
        title="平均运行温度收益对比",
        ylabel="平均温度 (°C)",
        filename="13_average_temperature_comparison.png",
    )
    plot_grouped_bar(
        metric="latency_mean",
        title="平均单条请求耗时收益对比",
        ylabel="平均单条请求耗时 (秒)",
        filename="14_average_request_latency_comparison.png",
        value_fmt="{:.2f}",
    )

    print(f"图像已生成到: {FIGURE_ROOT}")
    for file in sorted(FIGURE_ROOT.glob("*.png")):
        print(file.name)


if __name__ == "__main__":
    main()
