"""
run_backtest.py — 使用真实历史数据执行对比回测并生成可视化报告
"""

import sys
import json
import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime

# 设置中文字体
plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, "/home/ubuntu/btc_trader")
from config import CONFIG
from backtest import BacktestEngine, compare_and_report, load_klines
from logger import setup_logger

setup_logger(CONFIG)
logger = logging.getLogger(__name__)

DATA_DIR = "/home/ubuntu/btc_trader/data"
OUTPUT_DIR = "/home/ubuntu/btc_trader"


def load_all_data():
    print("正在加载历史数据...", flush=True)
    df_15m = load_klines(f"{DATA_DIR}/BTCUSDT_15m.csv", "15m")
    df_1h  = load_klines(f"{DATA_DIR}/BTCUSDT_1h.csv",  "1h")
    df_5m  = load_klines(f"{DATA_DIR}/BTCUSDT_5m.csv",  "5m")
    print(f"  15M: {len(df_15m)} 根 | 1H: {len(df_1h)} 根 | 5M: {len(df_5m)} 根", flush=True)
    # 对齐时间范围：5M 数据只有 90 天，以 5M 最早时间为基准截取 15M 和 1H
    start_time = df_5m.index[0]
    df_15m = df_15m[df_15m.index >= start_time]
    df_1h  = df_1h[df_1h.index  >= start_time]
    print(f"  对齐后 15M: {len(df_15m)} 根 | 1H: {len(df_1h)} 根 | 5M: {len(df_5m)} 根", flush=True)
    return df_15m, df_1h, df_5m


def run_backtest(df_15m, df_1h, df_5m):
    print("\n[回测A] 无共识评分过滤...", flush=True)
    engine_a = BacktestEngine(CONFIG, use_consensus_score=False)
    report_a = engine_a.run(df_15m, df_5m, df_1h)
    print(f"  完成: {report_a['total_trades']} 笔交易 | 胜率={report_a.get('win_rate_pct',0):.1f}%", flush=True)

    print("\n[回测B] 有共识评分过滤...", flush=True)
    engine_b = BacktestEngine(CONFIG, use_consensus_score=True)
    report_b = engine_b.run(df_15m, df_5m, df_1h)
    print(f"  完成: {report_b['total_trades']} 笔交易 | 胜率={report_b.get('win_rate_pct',0):.1f}%", flush=True)

    return report_a, report_b


def plot_equity_curves(report_a, report_b):
    """绘制权益曲线对比图。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("BTCUSDT Backtest: No Score Filter vs. Consensus Score Filter",
                 fontsize=14, fontweight="bold", y=1.01)

    # --- 图1: 权益曲线 ---
    ax1 = axes[0, 0]
    eq_a = report_a.get("equity_curve", [])
    eq_b = report_b.get("equity_curve", [])
    if eq_a:
        df_eq_a = pd.DataFrame(eq_a)
        df_eq_a["time"] = pd.to_datetime(df_eq_a["time"])
        ax1.plot(df_eq_a["time"], df_eq_a["equity"], label="No Score Filter", color="#e74c3c", linewidth=1.5)
    if eq_b:
        df_eq_b = pd.DataFrame(eq_b)
        df_eq_b["time"] = pd.to_datetime(df_eq_b["time"])
        ax1.plot(df_eq_b["time"], df_eq_b["equity"], label="With Score Filter", color="#2ecc71", linewidth=1.5)
    ax1.set_title("Equity Curve Comparison")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Equity (USDT)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=30)

    # --- 图2: 关键指标对比柱状图 ---
    ax2 = axes[0, 1]
    metrics = ["Win Rate (%)", "Avg R:R", "Profit Factor"]
    vals_a = [
        report_a.get("win_rate_pct", 0),
        report_a.get("avg_rr_ratio", 0),
        min(report_a.get("profit_factor", 0), 10),  # 限制显示上限
    ]
    vals_b = [
        report_b.get("win_rate_pct", 0),
        report_b.get("avg_rr_ratio", 0),
        min(report_b.get("profit_factor", 0), 10),
    ]
    x = np.arange(len(metrics))
    width = 0.35
    bars_a = ax2.bar(x - width/2, vals_a, width, label="No Score Filter", color="#e74c3c", alpha=0.85)
    bars_b = ax2.bar(x + width/2, vals_b, width, label="With Score Filter", color="#2ecc71", alpha=0.85)
    ax2.set_title("Key Metrics Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    for bar in bars_a:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_b:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    # --- 图3: 单笔交易盈亏分布 ---
    ax3 = axes[1, 0]
    trades_a = report_a.get("trades", [])
    trades_b = report_b.get("trades", [])
    if trades_a:
        pnls_a = [t["pnl"] for t in trades_a]
        ax3.hist(pnls_a, bins=20, alpha=0.6, color="#e74c3c", label=f"No Filter ({len(pnls_a)} trades)")
    if trades_b:
        pnls_b = [t["pnl"] for t in trades_b]
        ax3.hist(pnls_b, bins=20, alpha=0.6, color="#2ecc71", label=f"With Filter ({len(pnls_b)} trades)")
    ax3.axvline(0, color="black", linestyle="--", linewidth=1)
    ax3.set_title("PnL Distribution per Trade")
    ax3.set_xlabel("PnL (USDT)")
    ax3.set_ylabel("Count")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- 图4: 统计摘要表格 ---
    ax4 = axes[1, 1]
    ax4.axis("off")
    table_data = [
        ["Metric", "No Filter", "With Filter"],
        ["Total Trades",      str(report_a.get("total_trades", 0)),         str(report_b.get("total_trades", 0))],
        ["Win Rate",          f"{report_a.get('win_rate_pct', 0):.1f}%",    f"{report_b.get('win_rate_pct', 0):.1f}%"],
        ["Avg Win (USDT)",    f"{report_a.get('avg_win_usdt', 0):.2f}",     f"{report_b.get('avg_win_usdt', 0):.2f}"],
        ["Avg Loss (USDT)",   f"{report_a.get('avg_loss_usdt', 0):.2f}",    f"{report_b.get('avg_loss_usdt', 0):.2f}"],
        ["Avg R:R",           f"{report_a.get('avg_rr_ratio', 0):.3f}",     f"{report_b.get('avg_rr_ratio', 0):.3f}"],
        ["Profit Factor",     f"{report_a.get('profit_factor', 0):.3f}",    f"{report_b.get('profit_factor', 0):.3f}"],
        ["Total Return",      f"{report_a.get('total_return_pct', 0):.2f}%",f"{report_b.get('total_return_pct', 0):.2f}%"],
        ["Max Drawdown",      f"{report_a.get('max_drawdown_pct', 0):.2f}%",f"{report_b.get('max_drawdown_pct', 0):.2f}%"],
        ["Avg Hold (min)",    f"{report_a.get('avg_hold_minutes', 0):.0f}", f"{report_b.get('avg_hold_minutes', 0):.0f}"],
        ["Final Capital",     f"{report_a.get('final_capital', 0):.0f}",    f"{report_b.get('final_capital', 0):.0f}"],
    ]
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    # 着色表头
    for j in range(3):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # 着色数据行
    for i in range(1, len(table_data)):
        table[i, 0].set_facecolor("#ecf0f1")
        table[i, 1].set_facecolor("#fde8e8")
        table[i, 2].set_facecolor("#e8fde8")
    ax4.set_title("Summary Statistics", pad=20)

    plt.tight_layout()
    chart_path = f"{OUTPUT_DIR}/backtest_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  图表已保存: {chart_path}", flush=True)
    return chart_path


def save_report(report_a, report_b, comparison_text):
    """保存 JSON 报告。"""
    def serialize(obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return str(obj)
        if isinstance(obj, float) and (obj != obj):  # NaN
            return None
        return obj

    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": "Binance BTCUSDT Spot (1 year, 15M/1H/5M)",
        "initial_capital": CONFIG["backtest"]["initial_capital"],
        "without_score": {k: v for k, v in report_a.items() if k not in ("trades", "equity_curve")},
        "with_score":    {k: v for k, v in report_b.items() if k not in ("trades", "equity_curve")},
        "comparison_text": comparison_text,
    }
    json_path = f"{OUTPUT_DIR}/backtest_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=serialize)
    print(f"  JSON 报告已保存: {json_path}", flush=True)
    return json_path


def save_markdown_report(report_a, report_b, comparison_text):
    """生成 Markdown 格式的回测报告。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# BTCUSDT 顺势回踩策略 — 回测报告",
        "",
        f"**生成时间**：{now}",
        f"**数据来源**：Binance BTCUSDT 现货，最近 90 天（15M / 1H / 5M 三周期）",
        f"**初始资金**：{CONFIG['backtest']['initial_capital']:,} USDT",
        f"**每笔风险**：{CONFIG['risk_management']['risk_per_trade_pct']}% 账户净值",
        f"**共识评分阈值**：{CONFIG['consensus_score']['fixed_threshold']} 分",
        "",
        "---",
        "",
        "## 1. 核心指标对比",
        "",
        "| 指标 | 原策略（无评分过滤） | 新策略（有评分过滤） | 变化 |",
        "| :--- | ---: | ---: | ---: |",
    ]

    def delta(a, b, higher_better=True, fmt=".2f"):
        diff = b - a
        sign = "+" if diff >= 0 else ""
        arrow = "↑" if (diff > 0) == higher_better else ("↓" if diff != 0 else "→")
        return f"`{sign}{diff:{fmt}}` {arrow}"

    rows = [
        ("交易次数",       "total_trades",      False, ".0f"),
        ("胜率 (%)",       "win_rate_pct",       True,  ".2f"),
        ("平均盈亏比",     "avg_rr_ratio",       True,  ".3f"),
        ("利润因子",       "profit_factor",      True,  ".3f"),
        ("总收益率 (%)",   "total_return_pct",   True,  ".2f"),
        ("最大回撤 (%)",   "max_drawdown_pct",   False, ".2f"),
        ("平均持仓(分钟)", "avg_hold_minutes",   None,  ".0f"),
        ("最终资金(USDT)", "final_capital",      True,  ".2f"),
    ]
    for label, key, hb, fmt in rows:
        va = report_a.get(key, 0) or 0
        vb = report_b.get(key, 0) or 0
        d = delta(va, vb, hb, fmt) if hb is not None else "—"
        lines.append(f"| {label} | {va:{fmt}} | {vb:{fmt}} | {d} |")

    lines += [
        "",
        "---",
        "",
        "## 2. 详细对比报告",
        "",
        "```",
        comparison_text,
        "```",
        "",
        "---",
        "",
        "## 3. 策略说明",
        "",
        "本次回测使用 Binance BTCUSDT 现货历史数据（最近 90 天），对比了以下两种模式：",
        "",
        "- **原策略（无评分过滤）**：仅使用趋势过滤 + 支撑区识别 + 5M K线确认，不启用共识强度评分模块。",
        "- **新策略（有评分过滤）**：在原策略基础上，增加 7 维度市场共识强度评分，只有评分超过阈值（默认 60 分）的机会才会进入 5M 确认阶段。",
        "",
        "### 退出机制（两种模式相同）",
        "",
        "| 阶段 | 触发条件 | 操作 |",
        "| :--- | :--- | :--- |",
        "| 保本移损 | 盈利达到 1.0R | 止损移至成本价 |",
        "| TP1 | 盈利达到 1.2R | 减仓 35% |",
        "| TP2 | 价格靠近压力区 | 再减仓 35% |",
        "| 跟踪止盈 | TP2 触发后 | 剩余 30% 仓位跟踪 EMA(20) |",
        "| 止损 | 价格跌破结构止损 | 全部平仓 |",
        "",
        "---",
        "",
        "## 4. 使用说明",
        "",
        "如需调整参数重新回测，请修改 `config.py` 中的相关配置，然后运行：",
        "",
        "```bash",
        "python run_backtest.py",
        "```",
        "",
        "> **提示**：回测结果仅供参考，历史表现不代表未来收益。请在充分理解策略逻辑后，结合自身风险承受能力谨慎使用。",
    ]

    md_path = f"{OUTPUT_DIR}/backtest_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Markdown 报告已保存: {md_path}", flush=True)
    return md_path


if __name__ == "__main__":
    df_15m, df_1h, df_5m = load_all_data()
    report_a, report_b = run_backtest(df_15m, df_1h, df_5m)
    comparison_text = compare_and_report(report_a, report_b)
    print("\n" + comparison_text, flush=True)
    chart_path = plot_equity_curves(report_a, report_b)
    save_report(report_a, report_b, comparison_text)
    md_path = save_markdown_report(report_a, report_b, comparison_text)
    print(f"\n✅ 回测完成！报告已保存至:\n  {md_path}\n  {chart_path}", flush=True)
