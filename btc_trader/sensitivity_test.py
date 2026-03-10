"""
sensitivity_test.py — 共识评分阈值敏感性回测

测试 fixed_threshold = 30 / 40 / 50 / 60 四组参数，输出每组：
  - 候选区域总评估次数
  - 通过评分数量 & 通过率
  - 最终开仓数量
  - 胜率 / 平均盈亏比 / 总收益率 / 最大回撤
  - 平均共识评分（通过阈值的区域）

同时输出"无评分过滤"版本作为基准对比。
"""

import os
import sys
import copy
import json
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings("ignore")

# 确保能找到本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from backtest import BacktestEngine, load_klines, resample_klines

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,   # 敏感性测试时只输出警告以上，减少噪音
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_all_data(data_dir: str):
    """加载 15M / 5M / 1H / 4H 数据。"""
    path_15m = os.path.join(data_dir, "BTCUSDT_15m.csv")
    path_5m  = os.path.join(data_dir, "BTCUSDT_5m.csv")
    path_1h  = os.path.join(data_dir, "BTCUSDT_1h.csv")

    if not os.path.exists(path_15m):
        raise FileNotFoundError(f"找不到15M数据文件: {path_15m}")

    df_15m = load_klines(path_15m)
    df_5m  = load_klines(path_5m)  if os.path.exists(path_5m)  else None
    df_1h  = load_klines(path_1h)  if os.path.exists(path_1h)  else None

    # 4H 从 1H 重采样（如果没有独立的4H文件）
    df_4h = resample_klines(df_1h if df_1h is not None else df_15m, "4h")

    print(f"[数据] 15M: {len(df_15m)} 根 | "
          f"5M: {len(df_5m) if df_5m is not None else '重采样'} 根 | "
          f"1H: {len(df_1h) if df_1h is not None else '重采样'} 根 | "
          f"4H: {len(df_4h)} 根")
    print(f"[数据] 时间范围: {df_15m.index[0]} ~ {df_15m.index[-1]}")

    return df_15m, df_5m, df_1h, df_4h


# ---------------------------------------------------------------------------
# 单组回测
# ---------------------------------------------------------------------------
def run_single_test(config: dict, threshold: float,
                    df_15m, df_5m, df_1h, df_4h,
                    use_score: bool = True) -> dict:
    """
    运行单组回测。

    :param threshold:  fixed_threshold 值（use_score=False 时忽略）
    :param use_score:  是否启用共识评分过滤
    """
    cfg = copy.deepcopy(config)
    if use_score:
        cfg["consensus_score"]["fixed_threshold"] = threshold
        label = f"阈值={threshold}"
    else:
        cfg["consensus_score"]["enabled"] = False
        label = "无评分（基准）"

    engine = BacktestEngine(cfg, use_consensus_score=use_score)
    result = engine.run(df_15m, df_5m=df_5m, df_1h=df_1h, df_4h=df_4h)
    result["label"] = label
    result["threshold"] = threshold if use_score else None
    return result


# ---------------------------------------------------------------------------
# 格式化输出
# ---------------------------------------------------------------------------
def print_sensitivity_table(results: list):
    """打印敏感性测试汇总表格。"""
    sep = "=" * 110
    print(f"\n{sep}")
    print("  共识评分阈值敏感性回测报告")
    print(sep)

    header = (
        f"  {'策略':^12} | {'候选区域':^8} | {'通过评分':^8} | {'通过率':^7} | "
        f"{'开仓数':^6} | {'胜率':^7} | {'盈亏比':^7} | {'总收益':^9} | "
        f"{'最大回撤':^9} | {'平均分(通过)':^12}"
    )
    print(header)
    print(f"  {'-'*106}")

    for r in results:
        label = r.get("label", "")
        candidates = r.get("total_candidates_evaluated", 0)
        passed = r.get("total_candidates_passed", 0)
        pass_rate = r.get("pass_rate_pct", 0)
        trades = r.get("total_trades", 0)
        win_rate = r.get("win_rate_pct", 0)
        rr = r.get("avg_rr_ratio", 0)
        ret = r.get("total_return_pct", 0)
        dd = r.get("max_drawdown_pct", 0)
        avg_score = r.get("avg_consensus_score_passed", 0)

        print(
            f"  {label:^12} | {candidates:^8} | {passed:^8} | {pass_rate:^6.1f}% | "
            f"{trades:^6} | {win_rate:^6.1f}% | {rr:^7.3f} | {ret:^8.2f}% | "
            f"{dd:^8.2f}% | {avg_score:^12.1f}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def plot_sensitivity_results(results: list, output_path: str):
    """生成敏感性测试可视化图表。"""
    # 设置中文字体
    font_candidates = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
    font_found = None
    for font_name in font_candidates:
        try:
            fm.findfont(fm.FontProperties(family=font_name), fallback_to_default=False)
            font_found = font_name
            break
        except Exception:
            continue
    if font_found:
        plt.rcParams["font.family"] = font_found
    plt.rcParams["axes.unicode_minus"] = False

    # 分离基准和有评分的结果
    baseline = [r for r in results if r.get("threshold") is None]
    scored   = [r for r in results if r.get("threshold") is not None]
    scored.sort(key=lambda r: r["threshold"])

    thresholds = [r["threshold"] for r in scored]
    labels_all = [r["label"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("共识评分阈值敏感性分析 — BTCUSDT 永续合约", fontsize=15, fontweight="bold")

    # 颜色
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    baseline_color = "#95a5a6"

    def add_baseline(ax, key, label_suffix=""):
        if baseline:
            b_val = baseline[0].get(key, 0)
            ax.axhline(y=b_val, color=baseline_color, linestyle="--", linewidth=1.5,
                       label=f"无评分基准 ({b_val:.2f}{label_suffix})")

    # --- 图1: 开仓数量 vs 候选区域通过数 ---
    ax = axes[0, 0]
    trades_list = [r.get("total_trades", 0) for r in scored]
    passed_list  = [r.get("total_candidates_passed", 0) for r in scored]
    x = range(len(thresholds))
    ax.bar([i - 0.2 for i in x], passed_list, width=0.35, label="通过评分区域数", color="#3498db", alpha=0.7)
    ax.bar([i + 0.2 for i in x], trades_list, width=0.35, label="实际开仓数", color="#e74c3c", alpha=0.7)
    if baseline:
        ax.axhline(y=baseline[0].get("total_trades", 0), color=baseline_color,
                   linestyle="--", linewidth=1.5, label=f"无评分开仓数({baseline[0].get('total_trades',0)})")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"阈值={t}" for t in thresholds])
    ax.set_title("候选区域通过数 vs 实际开仓数")
    ax.set_ylabel("数量")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- 图2: 胜率 ---
    ax = axes[0, 1]
    win_rates = [r.get("win_rate_pct", 0) for r in scored]
    ax.plot(thresholds, win_rates, "o-", color="#2ecc71", linewidth=2, markersize=8, label="有评分胜率")
    add_baseline(ax, "win_rate_pct", "%")
    ax.set_title("胜率 vs 评分阈值")
    ax.set_xlabel("fixed_threshold")
    ax.set_ylabel("胜率 (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- 图3: 平均盈亏比 ---
    ax = axes[0, 2]
    rr_list = [r.get("avg_rr_ratio", 0) for r in scored]
    ax.plot(thresholds, rr_list, "s-", color="#f39c12", linewidth=2, markersize=8, label="有评分盈亏比")
    add_baseline(ax, "avg_rr_ratio")
    ax.set_title("平均盈亏比 vs 评分阈值")
    ax.set_xlabel("fixed_threshold")
    ax.set_ylabel("盈亏比")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- 图4: 总收益率 ---
    ax = axes[1, 0]
    ret_list = [r.get("total_return_pct", 0) for r in scored]
    bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in ret_list]
    ax.bar(thresholds, ret_list, color=bar_colors, alpha=0.8, width=3)
    add_baseline(ax, "total_return_pct", "%")
    ax.set_title("总收益率 vs 评分阈值")
    ax.set_xlabel("fixed_threshold")
    ax.set_ylabel("总收益率 (%)")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- 图5: 最大回撤 ---
    ax = axes[1, 1]
    dd_list = [abs(r.get("max_drawdown_pct", 0)) for r in scored]
    ax.plot(thresholds, dd_list, "D-", color="#e74c3c", linewidth=2, markersize=8, label="有评分最大回撤")
    if baseline:
        b_dd = abs(baseline[0].get("max_drawdown_pct", 0))
        ax.axhline(y=b_dd, color=baseline_color, linestyle="--", linewidth=1.5,
                   label=f"无评分基准 ({b_dd:.2f}%)")
    ax.set_title("最大回撤 vs 评分阈值（越小越好）")
    ax.set_xlabel("fixed_threshold")
    ax.set_ylabel("最大回撤 (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- 图6: 通过率 & 平均评分 ---
    ax = axes[1, 2]
    pass_rates = [r.get("pass_rate_pct", 0) for r in scored]
    avg_scores = [r.get("avg_consensus_score_passed", 0) for r in scored]
    ax2 = ax.twinx()
    ax.bar(thresholds, pass_rates, color="#9b59b6", alpha=0.6, width=3, label="通过率 (%)")
    ax2.plot(thresholds, avg_scores, "o-", color="#e67e22", linewidth=2, markersize=8, label="平均评分（通过）")
    ax.set_title("候选区域通过率 & 平均评分")
    ax.set_xlabel("fixed_threshold")
    ax.set_ylabel("通过率 (%)", color="#9b59b6")
    ax2.set_ylabel("平均评分", color="#e67e22")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图表] 已保存至: {output_path}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  共识评分阈值敏感性回测")
    print(f"  测试阈值: 30 / 40 / 50 / 60")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 加载数据
    df_15m, df_5m, df_1h, df_4h = load_all_data(data_dir)

    results = []

    # --- 基准：无评分过滤 ---
    print("\n[1/5] 运行基准回测（无评分过滤）...")
    r_base = run_single_test(CONFIG, 0, df_15m, df_5m, df_1h, df_4h, use_score=False)
    results.append(r_base)
    print(f"      完成 | 开仓: {r_base['total_trades']} 笔 | 收益: {r_base['total_return_pct']:.2f}%")

    # --- 四组阈值测试 ---
    thresholds = [30, 40, 50, 60]
    for idx, thr in enumerate(thresholds, start=2):
        print(f"\n[{idx}/5] 运行阈值={thr} 回测...")
        r = run_single_test(CONFIG, thr, df_15m, df_5m, df_1h, df_4h, use_score=True)
        results.append(r)
        print(
            f"      完成 | 候选区域: {r['total_candidates_evaluated']} | "
            f"通过: {r['total_candidates_passed']} ({r['pass_rate_pct']:.1f}%) | "
            f"开仓: {r['total_trades']} 笔 | "
            f"收益: {r['total_return_pct']:.2f}%"
        )

    # --- 打印汇总表 ---
    print_sensitivity_table(results)

    # --- 保存 JSON 结果 ---
    json_path = os.path.join(output_dir, "sensitivity_results.json")
    # 去掉 trades 和 equity_curve 等大字段，只保留汇总指标
    summary = []
    for r in results:
        s = {k: v for k, v in r.items() if k not in ("trades", "equity_curve")}
        summary.append(s)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[结果] JSON 汇总已保存至: {json_path}")

    # --- 生成可视化图表 ---
    chart_path = os.path.join(output_dir, "sensitivity_chart.png")
    try:
        plot_sensitivity_results(results, chart_path)
    except Exception as e:
        print(f"[警告] 图表生成失败: {e}")

    # --- 生成 Markdown 报告 ---
    md_path = os.path.join(output_dir, "sensitivity_report.md")
    write_markdown_report(results, md_path)

    print(f"\n[完成] 所有报告已生成。")
    print(f"  - JSON: {json_path}")
    print(f"  - 图表: {chart_path}")
    print(f"  - 报告: {md_path}")


def write_markdown_report(results: list, output_path: str):
    """生成 Markdown 格式的敏感性测试报告。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# 共识评分阈值敏感性回测报告",
        "",
        f"> 生成时间：{now}",
        "",
        "## 一、测试概述",
        "",
        "本报告对共识强度评分模块的 `fixed_threshold` 参数进行了系统性敏感性测试，"
        "分别测试了 **30 / 40 / 50 / 60** 四组阈值，并与**无评分过滤的基准策略**进行对比。",
        "",
        "**评分统计口径说明：**",
        "",
        "| 字段 | 含义 |",
        "| :--- | :--- |",
        "| 候选区域总数 | 每根 K 线循环中，经过区域识别后进入批量评分的候选支撑区总次数 |",
        "| 通过评分数 | 候选区域中评分 ≥ 阈值的次数 |",
        "| 通过率 | 通过评分数 / 候选区域总数 |",
        "| 实际开仓数 | 通过评分后，还需通过 5M 确认、止损计算、仓位计算才能开仓 |",
        "| 平均评分（通过） | 通过阈值的候选区域的平均得分（不含被过滤掉的区域）|",
        "",
        "## 二、汇总对比表",
        "",
        "| 策略 | 候选区域 | 通过评分 | 通过率 | 开仓数 | 胜率 | 盈亏比 | 总收益率 | 最大回撤 | 平均评分(通过) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in results:
        label = r.get("label", "")
        candidates = r.get("total_candidates_evaluated", 0)
        passed = r.get("total_candidates_passed", 0)
        pass_rate = r.get("pass_rate_pct", 0)
        trades = r.get("total_trades", 0)
        win_rate = r.get("win_rate_pct", 0)
        rr = r.get("avg_rr_ratio", 0)
        ret = r.get("total_return_pct", 0)
        dd = r.get("max_drawdown_pct", 0)
        avg_score = r.get("avg_consensus_score_passed", 0)

        ret_str = f"**{ret:+.2f}%**" if ret > 0 else f"{ret:+.2f}%"
        lines.append(
            f"| {label} | {candidates} | {passed} | {pass_rate:.1f}% | "
            f"{trades} | {win_rate:.1f}% | {rr:.3f} | {ret_str} | "
            f"{dd:.2f}% | {avg_score:.1f} |"
        )

    lines += [
        "",
        "## 三、关键发现",
        "",
    ]

    # 自动生成关键发现
    scored = [r for r in results if r.get("threshold") is not None]
    baseline = next((r for r in results if r.get("threshold") is None), None)

    if scored and baseline:
        best_ret = max(scored, key=lambda r: r.get("total_return_pct", -999))
        best_dd  = min(scored, key=lambda r: abs(r.get("max_drawdown_pct", 0)))
        best_wr  = max(scored, key=lambda r: r.get("win_rate_pct", 0))

        lines += [
            f"1. **最优收益率阈值**：`fixed_threshold = {best_ret['threshold']}`，"
            f"总收益率 {best_ret['total_return_pct']:+.2f}%，"
            f"开仓 {best_ret['total_trades']} 笔。",
            "",
            f"2. **最低回撤阈值**：`fixed_threshold = {best_dd['threshold']}`，"
            f"最大回撤 {best_dd['max_drawdown_pct']:.2f}%。",
            "",
            f"3. **最高胜率阈值**：`fixed_threshold = {best_wr['threshold']}`，"
            f"胜率 {best_wr['win_rate_pct']:.1f}%。",
            "",
            f"4. **基准策略（无评分）**：开仓 {baseline['total_trades']} 笔，"
            f"总收益率 {baseline['total_return_pct']:+.2f}%，"
            f"最大回撤 {baseline['max_drawdown_pct']:.2f}%。",
            "",
        ]

        # 判断评分过滤是否有效
        better_than_baseline = [r for r in scored if r.get("total_return_pct", -999) > baseline.get("total_return_pct", 0)]
        if better_than_baseline:
            lines += [
                f"5. **评分过滤有效性**：有 {len(better_than_baseline)} 组阈值的收益率优于无评分基准，"
                f"说明共识评分模块在当前市场环境下具有正向筛选价值。",
                "",
            ]
        else:
            lines += [
                "5. **评分过滤有效性**：当前数据区间内，所有阈值组的收益率均未超过无评分基准，"
                "建议进一步调整评分权重或扩大回测时间窗口。",
                "",
            ]

    lines += [
        "## 四、参数建议",
        "",
        "基于以上测试结果，建议：",
        "",
        "- 若优先**控制风险**：选择较高阈值（50 ~ 60），减少交易次数，提高信号质量。",
        "- 若优先**捕捉机会**：选择较低阈值（30 ~ 40），增加交易频率，但需配合更严格的止损。",
        "- **推荐默认值**：根据回测结果中收益率与回撤的综合表现选取最优阈值，并在实盘中持续监控。",
        "",
        "---",
        "",
        "> 本报告由回测系统自动生成，仅供参考，不构成投资建议。",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[报告] Markdown 报告已保存至: {output_path}")


if __name__ == "__main__":
    main()
