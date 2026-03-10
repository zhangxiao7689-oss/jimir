"""
sensitivity_test_fast.py — 共识评分阈值敏感性回测（优化版）

优化策略：
  1. 只使用最近 90 天数据（约 8640 根 15M K线）
  2. 区域检测每 3 根 K 线执行一次（而非每根都执行），大幅减少计算量
  3. 区域检测结果缓存，避免重复计算相同数据窗口

测试 fixed_threshold = 30 / 40 / 50 / 60 四组参数 + 无评分基准
"""

import os
import sys
import copy
import json
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from support_zone import SupportZoneDetector, get_nearest_support, get_nearest_resistance
from consensus_score import ConsensusScorer, FakeoutDetector
from entry_signal import EntryConfirmer
from exit_manager import StopLossCalculator, TakeProfitManager
from risk_manager import RiskManager, BinancePrecision
from backtest import load_klines, resample_klines

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 精简版回测引擎（优化计算效率）
# ---------------------------------------------------------------------------

class FastBacktestEngine:
    """
    优化版回测引擎。
    - 区域检测每 zone_detect_interval 根 K 线执行一次
    - 缓存区域检测结果，减少重复计算
    """

    def __init__(self, config: dict, use_consensus_score: bool = True,
                 zone_detect_interval: int = 3):
        self.config = copy.deepcopy(config)
        self.use_consensus_score = use_consensus_score
        self.zone_detect_interval = zone_detect_interval

        if not use_consensus_score:
            self.config["consensus_score"]["enabled"] = False

        bt_cfg = config.get("backtest", {})
        self.initial_capital = bt_cfg.get("initial_capital", 10000)
        self.commission_rate = bt_cfg.get("commission_rate", 0.0005)
        self.slippage_pct    = bt_cfg.get("slippage_pct", 0.0002)

        self.zone_detector   = SupportZoneDetector(self.config)
        self.consensus_scorer = ConsensusScorer(self.config)
        self.fakeout_detector = FakeoutDetector(self.config)
        self.entry_confirmer  = EntryConfirmer(self.config)
        self.sl_calculator    = StopLossCalculator(self.config)
        self.tp_manager       = TakeProfitManager(self.config)
        precision = BinancePrecision()
        self.risk_manager     = RiskManager(self.config, precision)

        self.capital   = self.initial_capital
        self.position  = None
        self.trades    = []
        self.equity_curve = []
        self.all_candidate_scores: list = []
        self.passed_scores: list = []

        # 区域缓存
        self._cached_zones = None
        self._cache_at_idx = -1

    def run(self, df_15m, df_5m=None, df_1h=None, df_4h=None) -> dict:
        if df_1h is None:
            df_1h = resample_klines(df_15m, "1h")
        if df_5m is None:
            df_5m = df_15m
        if df_4h is None:
            df_4h = resample_klines(df_15m, "4h")

        warmup = 100
        trend_cfg = self.config.get("strategy", {})
        ema_slow_p = trend_cfg.get("trend_ema_period", 200)
        ema_fast_p = trend_cfg.get("trend_ema_fast", 50)
        zone_prox  = self.config.get("entry_signal", {}).get("zone_proximity_pct", 0.005)

        for i in range(warmup, len(df_15m)):
            current_time  = df_15m.index[i]
            current_price = df_15m["close"].iloc[i]

            df_15m_slice = df_15m.iloc[:i + 1]
            df_1h_slice  = df_1h[df_1h.index <= current_time]
            df_4h_slice  = df_4h[df_4h.index <= current_time]
            df_5m_slice  = df_5m[df_5m.index <= current_time]

            # 权益曲线（每 5 根记录一次，减少内存占用）
            if i % 5 == 0:
                equity = self._calc_equity(current_price)
                self.equity_curve.append({"time": current_time, "equity": equity, "price": current_price})

            # 更新持仓
            if self.position is not None:
                self._update_position(current_price, df_5m_slice, current_time)

            # 寻找入场（无持仓时）
            if self.position is None:
                # 趋势过滤
                if len(df_1h_slice) >= ema_slow_p:
                    ema200 = df_1h_slice["close"].ewm(span=ema_slow_p, adjust=False).mean().iloc[-1]
                    ema50  = df_1h_slice["close"].ewm(span=ema_fast_p, adjust=False).mean().iloc[-1]
                    if not (current_price > ema200 and ema50 > ema200):
                        continue

                # 区域检测（缓存，每 zone_detect_interval 根 K 线刷新一次）
                if i % self.zone_detect_interval == 0 or self._cached_zones is None:
                    self._cached_zones = self.zone_detector.detect(df_15m_slice, current_price)
                    self._cache_at_idx = i

                zones = self._cached_zones
                support_zones = [z for z in zones if z["zone_type"] == "support"]
                if not support_zones:
                    continue

                candidate_zones = [
                    z for z in support_zones
                    if z["price_start"] * (1 - zone_prox) <= current_price
                ]
                if not candidate_zones:
                    continue

                # 批量评分（传入真实多周期数据）
                self.consensus_scorer.mtf_klines = {"1h": df_1h_slice, "4h": df_4h_slice}
                score_results = self.consensus_scorer.score_batch(candidate_zones, df_15m_slice)

                for sr in score_results:
                    self.all_candidate_scores.append(sr["total_score"])
                passed_results = [r for r in score_results if r["passed_threshold"]]
                for pr in passed_results:
                    self.passed_scores.append(pr["total_score"])

                if not passed_results:
                    continue

                passed_results.sort(key=lambda r: (
                    -r["total_score"],
                    abs(r["zone"]["mid_price"] - current_price)
                ))
                best_result = passed_results[0]
                nearest_support = best_result["zone"]

                # 假突破检测
                fakeout_result = self.fakeout_detector.check(best_result, df_5m_slice)
                if fakeout_result.get("fakeout_type") == "true_breakdown":
                    continue

                # 5M 确认（统一使用 EntryConfirmer）
                confirmed, _, signal_strength = self.entry_confirmer.confirm(
                    df_5m_slice, nearest_support, fakeout_result
                )
                if not confirmed:
                    continue

                # 止损计算
                sl_result = self.sl_calculator.calculate(current_price, nearest_support, df_15m_slice, df_5m_slice)
                if not sl_result["valid"]:
                    continue

                stop_price = sl_result["stop_price"]

                # 仓位计算
                pos_result = self.risk_manager.calculate_position(self.capital, current_price, stop_price, 0)
                if not pos_result["valid"]:
                    continue

                qty = pos_result["qty"]
                actual_entry = current_price * (1 + self.slippage_pct)
                self.capital -= actual_entry * qty * self.commission_rate

                resistance_zones = [z for z in zones if z["zone_type"] == "resistance"]
                nearest_resistance = get_nearest_resistance(resistance_zones, actual_entry)

                self.position = self.tp_manager.init_position(actual_entry, stop_price, qty, nearest_resistance)
                self.position["entry_time"] = current_time
                self.position["score"] = best_result["total_score"]
                self.position["signal_strength"] = signal_strength

        # 强制平仓
        if self.position is not None:
            self._force_close(df_15m["close"].iloc[-1], df_15m.index[-1])

        return self._generate_report()

    def _calc_equity(self, current_price: float) -> float:
        if self.position is None:
            return self.capital
        # 持仓字典中使用 remaining_qty（分批止盈后剩余数量）
        qty = self.position.get("remaining_qty", self.position.get("initial_qty", 0))
        unrealized = (current_price - self.position["entry_price"]) * qty
        return self.capital + unrealized

    def _update_position(self, current_price, df_5m, current_time):
        updated = self.tp_manager.update(self.position, current_price, df_5m)
        actions = updated.pop("actions", [])
        self.position.update(updated)

        for action in actions:
            pnl = action.get("pnl", 0)
            action_type = action.get("type", "")  # exit_manager 返回的是 "type" 键

            # TP1/TP2 减仓：更新资金和手续费，但不关闭持仓
            if action_type in ("TP1", "TP2"):
                self.capital += pnl
                commission = action.get("qty", 0) * current_price * self.commission_rate
                self.capital -= commission
                continue

            # 完全平仓类型：止损 / 跟踪止盈 / 其他关闭
            if action_type in ("STOP_LOSS", "TRAILING_STOP") or self.position.get("phase") == "CLOSED":
                self.capital += pnl
                commission = action.get("qty", 0) * current_price * self.commission_rate
                self.capital -= commission

                hold_minutes = 0
                if "entry_time" in self.position:
                    hold_minutes = (current_time - self.position["entry_time"]).total_seconds() / 60

                self.trades.append({
                    "entry_time": self.position.get("entry_time"),
                    "exit_time": current_time,
                    "entry_price": self.position.get("entry_price"),
                    "exit_price": current_price,
                    "pnl": self.position.get("realized_pnl", pnl),
                    "score": self.position.get("score", 0),
                    "exit_reason": action_type,
                    "hold_minutes": hold_minutes,
                    "tp1_triggered": self.position.get("tp1_triggered", False),
                    "tp2_triggered": self.position.get("tp2_triggered", False),
                })
                self.position = None
                break

        # 额外检查：如果持仓状态已是 CLOSED 但上面没有匹配到关闭事件
        if self.position is not None and self.position.get("phase") == "CLOSED":
            hold_minutes = 0
            if "entry_time" in self.position:
                hold_minutes = (current_time - self.position["entry_time"]).total_seconds() / 60
            self.trades.append({
                "entry_time": self.position.get("entry_time"),
                "exit_time": current_time,
                "entry_price": self.position.get("entry_price"),
                "exit_price": current_price,
                "pnl": self.position.get("realized_pnl", 0),
                "score": self.position.get("score", 0),
                "exit_reason": "CLOSED",
                "hold_minutes": hold_minutes,
                "tp1_triggered": self.position.get("tp1_triggered", False),
                "tp2_triggered": self.position.get("tp2_triggered", False),
            })
            self.position = None

    def _force_close(self, price, time):
        if self.position is None:
            return
        qty = self.position.get("remaining_qty", self.position.get("initial_qty", 0))
        pnl = (price - self.position["entry_price"]) * qty
        self.capital += pnl
        hold_minutes = 0
        if "entry_time" in self.position:
            hold_minutes = (time - self.position["entry_time"]).total_seconds() / 60
        total_pnl = self.position.get("realized_pnl", 0) + pnl
        self.trades.append({
            "entry_time": self.position.get("entry_time"),
            "exit_time": time,
            "entry_price": self.position.get("entry_price"),
            "exit_price": price,
            "pnl": total_pnl,
            "score": self.position.get("score", 0),
            "exit_reason": "FORCE_CLOSE",
            "hold_minutes": hold_minutes,
        })
        self.position = None

    def _generate_report(self) -> dict:
        if not self.trades:
            return {
                "use_consensus_score": self.use_consensus_score,
                "total_trades": 0, "win_count": 0, "loss_count": 0,
                "win_rate_pct": 0, "avg_win_usdt": 0, "avg_loss_usdt": 0,
                "avg_rr_ratio": 0, "profit_factor": 0,
                "total_return_pct": 0, "max_drawdown_pct": 0,
                "avg_hold_minutes": 0,
                "final_capital": round(self.capital, 4),
                "initial_capital": self.initial_capital,
                "avg_consensus_score": 0,
                "avg_consensus_score_all_candidates": round(float(np.mean(self.all_candidate_scores)), 2) if self.all_candidate_scores else 0,
                "avg_consensus_score_passed": round(float(np.mean(self.passed_scores)), 2) if self.passed_scores else 0,
                "total_candidates_evaluated": len(self.all_candidate_scores),
                "total_candidates_passed": len(self.passed_scores),
                "pass_rate_pct": round(len(self.passed_scores) / len(self.all_candidate_scores) * 100, 2) if self.all_candidate_scores else 0,
                "trades": [], "equity_curve": [], "message": "无交易记录",
            }

        df_t = pd.DataFrame(self.trades)
        df_e = pd.DataFrame(self.equity_curve)

        wins   = df_t[df_t["pnl"] > 0]
        losses = df_t[df_t["pnl"] <= 0]

        win_rate = len(wins) / len(df_t) * 100
        avg_win  = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses["pnl"].mean()) if len(losses) > 0 else 0
        avg_rr   = avg_win / avg_loss if avg_loss > 0 else 0
        pf       = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() != 0 else float("inf")
        total_ret = (self.capital - self.initial_capital) / self.initial_capital * 100

        max_dd = 0
        if len(df_e) > 0:
            eq = df_e["equity"]
            max_dd = ((eq - eq.cummax()) / eq.cummax() * 100).min()

        return {
            "use_consensus_score": self.use_consensus_score,
            "total_trades": len(df_t),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "avg_win_usdt": round(avg_win, 4),
            "avg_loss_usdt": round(avg_loss, 4),
            "avg_rr_ratio": round(avg_rr, 3),
            "profit_factor": round(pf, 3),
            "total_return_pct": round(total_ret, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "avg_hold_minutes": round(df_t["hold_minutes"].mean(), 1),
            "final_capital": round(self.capital, 4),
            "initial_capital": self.initial_capital,
            "avg_consensus_score": round(df_t["score"].mean(), 2) if "score" in df_t.columns else 0,
            "avg_consensus_score_all_candidates": round(float(np.mean(self.all_candidate_scores)), 2) if self.all_candidate_scores else 0,
            "avg_consensus_score_passed": round(float(np.mean(self.passed_scores)), 2) if self.passed_scores else 0,
            "total_candidates_evaluated": len(self.all_candidate_scores),
            "total_candidates_passed": len(self.passed_scores),
            "pass_rate_pct": round(len(self.passed_scores) / len(self.all_candidate_scores) * 100, 2) if self.all_candidate_scores else 0,
            "trades": df_t.to_dict("records"),
            "equity_curve": df_e.to_dict("records") if len(df_e) > 0 else [],
        }


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    data_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 65)
    print("  共识评分阈值敏感性回测（优化版）")
    print(f"  测试阈值: 无评分基准 / 30 / 40 / 50 / 60")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # 加载数据
    path_15m = os.path.join(data_dir, "BTCUSDT_15m.csv")
    path_5m  = os.path.join(data_dir, "BTCUSDT_5m.csv")
    path_1h  = os.path.join(data_dir, "BTCUSDT_1h.csv")

    df_15m_full = load_klines(path_15m)
    df_5m_full  = load_klines(path_5m)  if os.path.exists(path_5m)  else None
    df_1h_full  = load_klines(path_1h)  if os.path.exists(path_1h)  else None
    df_4h_full  = resample_klines(df_1h_full if df_1h_full is not None else df_15m_full, "4h")

    # 截取最近 90 天数据
    cutoff = df_15m_full.index[-1] - timedelta(days=90)
    # 保留 cutoff 之前 200 根 1H K线用于趋势指标预热
    warmup_cutoff = cutoff - timedelta(hours=200)

    df_15m = df_15m_full[df_15m_full.index >= warmup_cutoff].copy()
    df_5m  = df_5m_full[df_5m_full.index >= warmup_cutoff].copy() if df_5m_full is not None else None
    df_1h  = df_1h_full[df_1h_full.index >= warmup_cutoff].copy() if df_1h_full is not None else None
    df_4h  = df_4h_full[df_4h_full.index >= warmup_cutoff].copy()

    print(f"\n[数据] 15M: {len(df_15m)} 根 | 5M: {len(df_5m) if df_5m is not None else 'N/A'} 根"
          f" | 1H: {len(df_1h) if df_1h is not None else 'N/A'} 根 | 4H: {len(df_4h)} 根")
    print(f"[数据] 回测区间: {cutoff.strftime('%Y-%m-%d')} ~ {df_15m_full.index[-1].strftime('%Y-%m-%d')}")

    results = []
    thresholds_to_test = [None, 30, 40, 50, 60]  # None = 无评分基准

    for idx, thr in enumerate(thresholds_to_test, start=1):
        use_score = thr is not None
        label = f"阈值={thr}" if use_score else "无评分（基准）"
        print(f"\n[{idx}/{len(thresholds_to_test)}] 运行 {label} 回测...", end="", flush=True)

        cfg = copy.deepcopy(CONFIG)
        if use_score:
            cfg["consensus_score"]["fixed_threshold"] = thr
        else:
            cfg["consensus_score"]["enabled"] = False

        engine = FastBacktestEngine(cfg, use_consensus_score=use_score, zone_detect_interval=3)
        result = engine.run(df_15m, df_5m=df_5m, df_1h=df_1h, df_4h=df_4h)
        result["label"] = label
        result["threshold"] = thr
        results.append(result)

        print(
            f" 完成 | 候选: {result['total_candidates_evaluated']} | "
            f"通过: {result['total_candidates_passed']} ({result['pass_rate_pct']:.1f}%) | "
            f"开仓: {result['total_trades']} 笔 | "
            f"收益: {result['total_return_pct']:+.2f}%"
        )

    # 打印汇总表
    print_table(results)

    # 保存 JSON
    json_path = os.path.join(output_dir, "sensitivity_results.json")
    summary = [{k: v for k, v in r.items() if k not in ("trades", "equity_curve")} for r in results]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # 生成图表
    chart_path = os.path.join(output_dir, "sensitivity_chart.png")
    try:
        plot_results(results, chart_path)
    except Exception as e:
        print(f"[警告] 图表生成失败: {e}")

    # 生成 Markdown 报告
    md_path = os.path.join(output_dir, "sensitivity_report.md")
    write_md_report(results, md_path)

    print(f"\n[完成] 报告已生成:")
    print(f"  JSON: {json_path}")
    print(f"  图表: {chart_path}")
    print(f"  报告: {md_path}")


def print_table(results):
    sep = "=" * 115
    print(f"\n{sep}")
    print("  共识评分阈值敏感性回测报告")
    print(sep)
    print(f"  {'策略':^14} | {'候选区域':^8} | {'通过评分':^8} | {'通过率':^7} | "
          f"{'开仓数':^6} | {'胜率':^7} | {'盈亏比':^7} | {'总收益':^9} | "
          f"{'最大回撤':^9} | {'平均分(通过)':^12}")
    print(f"  {'-'*111}")
    for r in results:
        print(
            f"  {r.get('label',''):^14} | {r.get('total_candidates_evaluated',0):^8} | "
            f"{r.get('total_candidates_passed',0):^8} | {r.get('pass_rate_pct',0):^6.1f}% | "
            f"{r.get('total_trades',0):^6} | {r.get('win_rate_pct',0):^6.1f}% | "
            f"{r.get('avg_rr_ratio',0):^7.3f} | {r.get('total_return_pct',0):^8.2f}% | "
            f"{r.get('max_drawdown_pct',0):^8.2f}% | {r.get('avg_consensus_score_passed',0):^12.1f}"
        )
    print(sep)


def plot_results(results, output_path):
    font_candidates = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
    for fn in font_candidates:
        try:
            fm.findfont(fm.FontProperties(family=fn), fallback_to_default=False)
            plt.rcParams["font.family"] = fn
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    baseline = next((r for r in results if r.get("threshold") is None), None)
    scored   = sorted([r for r in results if r.get("threshold") is not None], key=lambda r: r["threshold"])
    thresholds = [r["threshold"] for r in scored]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("共识评分阈值敏感性分析 — BTCUSDT 永续合约（最近90天）", fontsize=14, fontweight="bold")
    grey = "#95a5a6"

    def add_base(ax, key, suffix=""):
        if baseline:
            v = baseline.get(key, 0)
            ax.axhline(y=v, color=grey, linestyle="--", lw=1.5, label=f"无评分基准({v:.2f}{suffix})")

    # 图1: 开仓数 vs 通过评分数
    ax = axes[0, 0]
    x = range(len(thresholds))
    ax.bar([i-0.2 for i in x], [r.get("total_candidates_passed",0) for r in scored], 0.35, label="通过评分区域", color="#3498db", alpha=0.7)
    ax.bar([i+0.2 for i in x], [r.get("total_trades",0) for r in scored], 0.35, label="实际开仓数", color="#e74c3c", alpha=0.7)
    if baseline:
        ax.axhline(y=baseline.get("total_trades",0), color=grey, linestyle="--", lw=1.5, label=f"基准开仓({baseline.get('total_trades',0)})")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"阈值={t}" for t in thresholds])
    ax.set_title("通过评分区域数 vs 实际开仓数"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # 图2: 胜率
    ax = axes[0, 1]
    ax.plot(thresholds, [r.get("win_rate_pct",0) for r in scored], "o-", color="#2ecc71", lw=2, ms=8, label="有评分胜率")
    add_base(ax, "win_rate_pct", "%"); ax.set_title("胜率 vs 评分阈值"); ax.set_xlabel("fixed_threshold"); ax.set_ylabel("胜率(%)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图3: 盈亏比
    ax = axes[0, 2]
    ax.plot(thresholds, [r.get("avg_rr_ratio",0) for r in scored], "s-", color="#f39c12", lw=2, ms=8, label="有评分盈亏比")
    add_base(ax, "avg_rr_ratio"); ax.set_title("平均盈亏比 vs 评分阈值"); ax.set_xlabel("fixed_threshold"); ax.set_ylabel("盈亏比"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图4: 总收益率
    ax = axes[1, 0]
    rets = [r.get("total_return_pct",0) for r in scored]
    ax.bar(thresholds, rets, color=["#2ecc71" if v>=0 else "#e74c3c" for v in rets], alpha=0.8, width=3)
    add_base(ax, "total_return_pct", "%"); ax.axhline(y=0, color="black", lw=0.8)
    ax.set_title("总收益率 vs 评分阈值"); ax.set_xlabel("fixed_threshold"); ax.set_ylabel("总收益率(%)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # 图5: 最大回撤
    ax = axes[1, 1]
    ax.plot(thresholds, [abs(r.get("max_drawdown_pct",0)) for r in scored], "D-", color="#e74c3c", lw=2, ms=8, label="有评分最大回撤")
    if baseline:
        b = abs(baseline.get("max_drawdown_pct",0))
        ax.axhline(y=b, color=grey, linestyle="--", lw=1.5, label=f"基准({b:.2f}%)")
    ax.set_title("最大回撤 vs 评分阈值（越小越好）"); ax.set_xlabel("fixed_threshold"); ax.set_ylabel("最大回撤(%)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图6: 通过率 & 平均评分
    ax = axes[1, 2]
    ax2 = ax.twinx()
    ax.bar(thresholds, [r.get("pass_rate_pct",0) for r in scored], color="#9b59b6", alpha=0.6, width=3, label="通过率(%)")
    ax2.plot(thresholds, [r.get("avg_consensus_score_passed",0) for r in scored], "o-", color="#e67e22", lw=2, ms=8, label="平均评分(通过)")
    ax.set_title("候选区域通过率 & 平均评分"); ax.set_xlabel("fixed_threshold")
    ax.set_ylabel("通过率(%)", color="#9b59b6"); ax2.set_ylabel("平均评分", color="#e67e22")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图表] 已保存至: {output_path}")


def write_md_report(results, output_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    baseline = next((r for r in results if r.get("threshold") is None), None)
    scored   = sorted([r for r in results if r.get("threshold") is not None], key=lambda r: r["threshold"])

    lines = [
        "# 共识评分阈值敏感性回测报告",
        "",
        f"> 生成时间：{now}  |  回测区间：最近 90 天  |  初始资金：10,000 USDT",
        "",
        "## 一、测试概述",
        "",
        "本报告对共识强度评分模块的 `fixed_threshold` 参数进行系统性敏感性测试，"
        "测试了 **30 / 40 / 50 / 60** 四组阈值，并与无评分过滤的基准策略对比。",
        "",
        "**评分统计口径说明：**",
        "",
        "| 字段 | 含义 |",
        "| :--- | :--- |",
        "| 候选区域总数 | 每根 K 线循环中，进入批量评分的候选支撑区总次数 |",
        "| 通过评分数 | 候选区域中评分 ≥ 阈值的次数 |",
        "| 通过率 | 通过评分数 / 候选区域总数 |",
        "| 实际开仓数 | 通过评分后，还需通过 5M 确认、止损计算、仓位计算才能开仓 |",
        "| 平均评分（通过） | 通过阈值的候选区域的平均得分 |",
        "",
        "## 二、汇总对比表",
        "",
        "| 策略 | 候选区域 | 通过评分 | 通过率 | 开仓数 | 胜率 | 盈亏比 | 总收益率 | 最大回撤 | 平均评分(通过) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in results:
        ret = r.get("total_return_pct", 0)
        ret_str = f"**{ret:+.2f}%**" if ret > 0 else f"{ret:+.2f}%"
        lines.append(
            f"| {r.get('label','')} | {r.get('total_candidates_evaluated',0)} | "
            f"{r.get('total_candidates_passed',0)} | {r.get('pass_rate_pct',0):.1f}% | "
            f"{r.get('total_trades',0)} | {r.get('win_rate_pct',0):.1f}% | "
            f"{r.get('avg_rr_ratio',0):.3f} | {ret_str} | "
            f"{r.get('max_drawdown_pct',0):.2f}% | {r.get('avg_consensus_score_passed',0):.1f} |"
        )

    lines += ["", "## 三、关键发现", ""]

    if scored and baseline:
        best_ret = max(scored, key=lambda r: r.get("total_return_pct", -999))
        best_dd  = min(scored, key=lambda r: abs(r.get("max_drawdown_pct", 0)))
        best_wr  = max(scored, key=lambda r: r.get("win_rate_pct", 0))
        lines += [
            f"1. **最优收益率阈值**：`fixed_threshold = {best_ret['threshold']}`，总收益率 {best_ret['total_return_pct']:+.2f}%，开仓 {best_ret['total_trades']} 笔。",
            f"2. **最低回撤阈值**：`fixed_threshold = {best_dd['threshold']}`，最大回撤 {best_dd['max_drawdown_pct']:.2f}%。",
            f"3. **最高胜率阈值**：`fixed_threshold = {best_wr['threshold']}`，胜率 {best_wr['win_rate_pct']:.1f}%。",
            f"4. **基准策略（无评分）**：开仓 {baseline['total_trades']} 笔，总收益率 {baseline['total_return_pct']:+.2f}%，最大回撤 {baseline['max_drawdown_pct']:.2f}%。",
            "",
        ]
        better = [r for r in scored if r.get("total_return_pct", -999) > baseline.get("total_return_pct", 0)]
        if better:
            lines.append(f"5. **评分过滤有效性**：有 {len(better)} 组阈值的收益率优于无评分基准，共识评分模块具有正向筛选价值。")
        else:
            lines.append("5. **评分过滤有效性**：当前数据区间内，所有阈值组的收益率均未超过无评分基准，建议调整评分权重或扩大回测时间窗口。")

    lines += [
        "", "## 四、参数建议", "",
        "- **控制风险优先**：选择较高阈值（50 ~ 60），减少交易次数，提高信号质量。",
        "- **捕捉机会优先**：选择较低阈值（30 ~ 40），增加交易频率，需配合严格止损。",
        "- **推荐默认值**：根据回测结果中收益率与回撤的综合表现选取最优阈值。",
        "", "---", "", "> 本报告由回测系统自动生成，仅供参考，不构成投资建议。",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[报告] Markdown 报告已保存至: {output_path}")


if __name__ == "__main__":
    main()
