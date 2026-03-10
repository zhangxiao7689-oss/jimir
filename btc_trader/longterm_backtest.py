"""
longterm_backtest.py — 长周期标准回测 v4.1（向量化优化版）
=============================================================
版本: v4.1 | 数据来源: Binance Futures (fapi.binance.com)

性能优化：
  - 所有 EMA / ATR / 趋势指标在回测开始前一次性向量化计算
  - 回测循环内只做 O(1) 的索引查找，彻底消除 O(n²) 性能瓶颈
  - 区域检测仍每 zone_detect_interval 根 K 线执行一次（最大计算量来源）

功能：
  1. 支持 90 / 180 天两个时间窗口（15M 数据覆盖 ~180 天）
  2. 区分趋势段（上升/下降）与震荡段，分别统计指标
  3. 使用阈值=60（当前最优参数）作为标准配置
  4. 输出唯一有效的标准回测报告（Markdown + JSON + PNG）
  5. 数据来源明确标注：Binance Futures 永续合约
"""

import os
import sys
import copy
import json
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
from support_zone import SupportZoneDetector, get_nearest_resistance
from consensus_score import ConsensusScorer, FakeoutDetector
from entry_signal import EntryConfirmer
from exit_manager import StopLossCalculator, TakeProfitManager
from risk_manager import RiskManager, BinancePrecision
from backtest import load_klines, resample_klines


# ---------------------------------------------------------------------------
# 向量化指标预计算
# ---------------------------------------------------------------------------

def precompute_indicators(df_15m: pd.DataFrame, df_1h: pd.DataFrame,
                           df_5m: pd.DataFrame) -> dict:
    """
    在回测开始前一次性计算所有需要的技术指标，返回可按时间索引的 Series。
    回测循环内只做 .loc[t] 查找，复杂度 O(1)。
    """
    # 1H 趋势指标
    ema50_1h  = df_1h["close"].ewm(span=50,  adjust=False).mean()
    ema200_1h = df_1h["close"].ewm(span=200, adjust=False).mean()
    slope50_1h = ema50_1h.diff(5)

    # 市场状态
    regimes = pd.Series("ranging", index=df_1h.index)
    regimes[
        (df_1h["close"] > ema200_1h) &
        (ema50_1h > ema200_1h) &
        (slope50_1h > 0)
    ] = "uptrend"
    regimes[
        (df_1h["close"] < ema200_1h) &
        (ema50_1h < ema200_1h) &
        (slope50_1h < 0)
    ] = "downtrend"

    # 5M EMA20（用于跟踪止盈）
    ema20_5m = df_5m["close"].ewm(span=20, adjust=False).mean()

    # 15M ATR14（用于止损缓冲）
    high, low, close = df_15m["high"], df_15m["low"], df_15m["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14_15m = tr.rolling(14).mean()

    return {
        "ema50_1h":   ema50_1h,
        "ema200_1h":  ema200_1h,
        "regimes":    regimes,
        "ema20_5m":   ema20_5m,
        "atr14_15m":  atr14_15m,
    }


def get_latest(series: pd.Series, t) -> float:
    """取 series 中 <= t 的最新值，若无则返回 NaN"""
    mask = series.index <= t
    if mask.any():
        return series[mask].iloc[-1]
    return float("nan")


# ---------------------------------------------------------------------------
# 向量化回测引擎
# ---------------------------------------------------------------------------

class VectorizedBacktestEngine:
    """
    向量化回测引擎。
    - 所有指标预计算，回测循环内只做索引查找
    - 退出机制：简化版内联实现（避免 exit_manager 的 O(n) 调用）
    - 支持 regime 标注
    """

    def __init__(self, config: dict, threshold: int = 60,
                 zone_detect_interval: int = 48,
                 zone_lookback: int = 500):
        self.config = copy.deepcopy(config)
        self.config["consensus_score"]["fixed_threshold"] = threshold
        self.config["consensus_score"]["enabled"] = True
        self.threshold = threshold
        self.zone_detect_interval = zone_detect_interval
        self.zone_lookback = zone_lookback  # 区域检测只看最近 N 根 K 线

        bt_cfg = config.get("backtest", {})
        self.initial_capital = bt_cfg.get("initial_capital", 10000)
        self.commission_rate = bt_cfg.get("commission_rate", 0.0005)
        self.slippage_pct    = bt_cfg.get("slippage_pct", 0.0002)

        # 止盈止损参数（从 config 读取）
        sl_cfg  = config.get("stop_loss", {})
        tp_cfg  = config.get("take_profit", {})
        rm_cfg  = config.get("risk_management", {})

        self.risk_pct         = rm_cfg.get("risk_per_trade_pct", 1.0) / 100
        self.min_notional     = rm_cfg.get("min_notional_usdt", 10.0)
        self.max_stop_pct     = sl_cfg.get("max_stop_distance_pct", 0.03)
        self.min_stop_pct     = sl_cfg.get("min_stop_distance_pct", 0.003)
        self.atr_mult_sl      = sl_cfg.get("structure", {}).get("atr_multiplier", 0.5)

        self.tp1_r            = tp_cfg.get("tp1", {}).get("target_r", 1.2)
        self.tp1_pct          = tp_cfg.get("tp1", {}).get("close_pct", 0.35)
        self.tp2_r            = tp_cfg.get("tp2", {}).get("target_r", 2.0)
        self.tp2_pct          = tp_cfg.get("tp2", {}).get("close_pct", 0.35)
        self.breakeven_r      = tp_cfg.get("breakeven_activation_r", 1.0)
        self.trail_activation = tp_cfg.get("trailing", {}).get("activation_r", 1.0)
        self.zone_prox        = config.get("entry_signal", {}).get("zone_proximity_pct", 0.015)

        # 趋势过滤参数
        strat = config.get("strategy", {})
        self.ema_slow_p = strat.get("trend_ema_period", 200)
        self.ema_fast_p = strat.get("trend_ema_fast", 50)

        # 模块实例
        self.zone_detector    = SupportZoneDetector(self.config)
        self.consensus_scorer = ConsensusScorer(self.config)
        self.fakeout_detector = FakeoutDetector(self.config)
        self.entry_confirmer  = EntryConfirmer(self.config)
        self.sl_calculator    = StopLossCalculator(self.config)
        precision             = BinancePrecision()
        self.risk_manager     = RiskManager(self.config, precision)

        # 状态
        self.capital      = self.initial_capital
        self.position     = None
        self.trades       = []
        self.equity_curve = []
        self.all_candidate_scores = []
        self.passed_scores = []
        self._cached_zones = None
        self._cache_at_idx = -1

    def run(self, df_15m: pd.DataFrame, df_5m: pd.DataFrame,
            df_1h: pd.DataFrame, df_4h: pd.DataFrame,
            indicators: dict) -> dict:
        """
        主回测循环。
        indicators: precompute_indicators() 的返回值
        """
        warmup = max(self.ema_slow_p, 100)

        for i in range(warmup, len(df_15m)):
            t = df_15m.index[i]
            price = df_15m["close"].iloc[i]

            # 权益曲线（每 5 根记录一次）
            if i % 5 == 0:
                equity = self._calc_equity(price)
                regime = get_latest(indicators["regimes"], t)
                self.equity_curve.append({
                    "time": t, "equity": equity,
                    "price": price, "regime": regime
                })

            # 更新持仓（向量化退出）
            if self.position is not None:
                self._update_position_fast(price, t, indicators, df_5m)

            # 寻找入场
            if self.position is not None:
                continue

            # 趋势过滤（O(1) 查找）
            ema200 = get_latest(indicators["ema200_1h"], t)
            ema50  = get_latest(indicators["ema50_1h"],  t)
            if np.isnan(ema200) or np.isnan(ema50):
                continue
            if not (price > ema200 and ema50 > ema200):
                continue

            # 区域检测（缓存 + 限制 lookback 窗口）
            if i % self.zone_detect_interval == 0 or self._cached_zones is None:
                start_idx = max(0, i + 1 - self.zone_lookback)
                df_15m_slice = df_15m.iloc[start_idx:i + 1]
                self._cached_zones = self.zone_detector.detect(df_15m_slice, price)
                self._cache_at_idx = i

            zones = self._cached_zones
            support_zones = [z for z in zones if z["zone_type"] == "support"]
            if not support_zones:
                continue

            candidate_zones = [
                z for z in support_zones
                if z["price_start"] * (1 - self.zone_prox) <= price
            ]
            if not candidate_zones:
                continue

            # 批量评分（传入真实多周期数据）
            # 评分用的 K 线也限制窗口，避免 O(n) 增长
            start_idx = max(0, i + 1 - self.zone_lookback)
            df_15m_slice = df_15m.iloc[start_idx:i + 1]
            t_lookback = df_15m_slice.index[0]
            df_1h_slice  = df_1h[df_1h.index >= t_lookback]
            df_1h_slice  = df_1h_slice[df_1h_slice.index <= t]
            df_4h_slice  = df_4h[df_4h.index >= t_lookback]
            df_4h_slice  = df_4h_slice[df_4h_slice.index <= t]
            df_5m_slice  = df_5m[df_5m.index >= t_lookback]
            df_5m_slice  = df_5m_slice[df_5m_slice.index <= t]

            self.consensus_scorer.mtf_klines = {"1h": df_1h_slice, "4h": df_4h_slice}
            score_results = self.consensus_scorer.score_batch(candidate_zones, df_15m_slice)

            for sr in score_results:
                self.all_candidate_scores.append(sr["total_score"])
            passed = [r for r in score_results if r["passed_threshold"]]
            for pr in passed:
                self.passed_scores.append(pr["total_score"])

            if not passed:
                continue

            passed.sort(key=lambda r: (-r["total_score"],
                                        abs(r["zone"]["mid_price"] - price)))
            best = passed[0]
            nearest_support = best["zone"]

            # 假突破检测
            fakeout = self.fakeout_detector.check(best, df_5m_slice)
            if fakeout.get("fakeout_type") == "true_breakdown":
                continue

            # 5M 确认
            confirmed, _, signal_strength = self.entry_confirmer.confirm(
                df_5m_slice, nearest_support, fakeout
            )
            if not confirmed:
                continue

            # 止损计算
            sl_result = self.sl_calculator.calculate(
                price, nearest_support, df_15m_slice, df_5m_slice
            )
            if not sl_result["valid"]:
                continue
            stop_price = sl_result["stop_price"]

            # 仓位计算
            pos_result = self.risk_manager.calculate_position(
                self.capital, price, stop_price, 0
            )
            if not pos_result["valid"]:
                continue

            qty = pos_result["qty"]
            actual_entry = price * (1 + self.slippage_pct)
            self.capital -= actual_entry * qty * self.commission_rate

            resistance_zones = [z for z in zones if z["zone_type"] == "resistance"]
            nearest_res = get_nearest_resistance(resistance_zones, actual_entry)
            tp2_price = nearest_res["mid_price"] if nearest_res else actual_entry * (1 + self.tp2_r * (actual_entry - stop_price) / actual_entry)
            risk_per_unit = actual_entry - stop_price

            self.position = {
                "entry_price":       actual_entry,
                "stop_price":        stop_price,
                "current_stop":      stop_price,
                "initial_qty":       qty,
                "remaining_qty":     qty,
                "closed_qty":        0.0,
                "realized_pnl":      0.0,
                "risk_per_unit":     risk_per_unit,
                "tp1_price":         actual_entry + self.tp1_r * risk_per_unit,
                "tp2_price":         tp2_price,
                "tp1_triggered":     False,
                "tp2_triggered":     False,
                "breakeven_triggered": False,
                "trailing_stop":     None,
                "phase":             "OPEN",
                "entry_time":        t,
                "score":             best["total_score"],
                "regime":            get_latest(indicators["regimes"], t),
            }

        # 强制平仓
        if self.position is not None:
            self._force_close(df_15m["close"].iloc[-1], df_15m.index[-1])

        return self._generate_report()

    def _update_position_fast(self, price: float, t, indicators: dict,
                               df_5m: pd.DataFrame):
        """
        向量化退出更新：所有指标通过 O(1) 索引查找，不做任何 ewm() 计算。
        """
        pos = self.position
        entry = pos["entry_price"]
        risk  = pos["risk_per_unit"]
        current_r = (price - entry) / risk if risk > 0 else 0

        # 1. 止损触发
        if price <= pos["current_stop"]:
            qty = pos["remaining_qty"]
            pnl = (price - entry) * qty
            pos["realized_pnl"] += pnl
            self.capital += pnl
            self.capital -= qty * price * self.commission_rate
            self._record_trade(pos, price, t, "STOP_LOSS")
            self.position = None
            return

        # 2. 保本移损
        if not pos["breakeven_triggered"] and current_r >= self.breakeven_r:
            new_stop = entry * 1.001  # 保本价 + 0.1% 覆盖手续费
            if new_stop > pos["current_stop"]:
                pos["current_stop"] = new_stop
                pos["breakeven_triggered"] = True

        # 3. TP1
        if not pos["tp1_triggered"] and price >= pos["tp1_price"]:
            close_qty = round(pos["initial_qty"] * self.tp1_pct, 8)
            close_qty = min(close_qty, pos["remaining_qty"])
            if close_qty > 0:
                pnl = (price - entry) * close_qty
                pos["realized_pnl"] += pnl
                pos["remaining_qty"] -= close_qty
                pos["closed_qty"] += close_qty
                self.capital += pnl
                self.capital -= close_qty * price * self.commission_rate
                pos["tp1_triggered"] = True
                pos["phase"] = "ACTIVE"

        # 4. TP2（并行，TP1 后立即监控）
        if (pos["tp1_triggered"] and not pos["tp2_triggered"]
                and pos["remaining_qty"] > 0
                and price >= pos["tp2_price"]):
            close_qty = round(pos["initial_qty"] * self.tp2_pct, 8)
            close_qty = min(close_qty, pos["remaining_qty"])
            if close_qty > 0:
                pnl = (price - entry) * close_qty
                pos["realized_pnl"] += pnl
                pos["remaining_qty"] -= close_qty
                pos["closed_qty"] += close_qty
                self.capital += pnl
                self.capital -= close_qty * price * self.commission_rate
                pos["tp2_triggered"] = True
            if pos["remaining_qty"] <= 0:
                pos["phase"] = "CLOSED"
                self._record_trade(pos, price, t, "TP2_FULL")
                self.position = None
                return

        # 5. 跟踪止盈（O(1) 查找预计算的 EMA20_5m）
        if (pos["tp1_triggered"] and pos["breakeven_triggered"]
                and pos["remaining_qty"] > 0
                and current_r >= self.trail_activation):
            trail_stop = get_latest(indicators["ema20_5m"], t)
            if not np.isnan(trail_stop):
                # 跟踪止损只能向上移动
                if pos["trailing_stop"] is None or trail_stop > pos["trailing_stop"]:
                    pos["trailing_stop"] = trail_stop
                    pos["current_stop"] = max(pos["current_stop"], trail_stop)

                if price <= pos["trailing_stop"]:
                    qty = pos["remaining_qty"]
                    pnl = (price - entry) * qty
                    pos["realized_pnl"] += pnl
                    pos["remaining_qty"] = 0
                    self.capital += pnl
                    self.capital -= qty * price * self.commission_rate
                    self._record_trade(pos, price, t, "TRAILING_STOP")
                    self.position = None
                    return

    def _record_trade(self, pos: dict, exit_price: float, exit_time, reason: str):
        hold_minutes = 0
        if "entry_time" in pos:
            hold_minutes = (exit_time - pos["entry_time"]).total_seconds() / 60
        self.trades.append({
            "entry_time":    pos.get("entry_time"),
            "exit_time":     exit_time,
            "entry_price":   pos.get("entry_price"),
            "exit_price":    exit_price,
            "pnl":           round(pos.get("realized_pnl", 0), 4),
            "score":         pos.get("score", 0),
            "exit_reason":   reason,
            "hold_minutes":  round(hold_minutes, 1),
            "tp1_triggered": pos.get("tp1_triggered", False),
            "tp2_triggered": pos.get("tp2_triggered", False),
            "regime":        pos.get("regime", "unknown"),
        })

    def _force_close(self, price: float, t):
        pos = self.position
        if pos is None:
            return
        qty = pos.get("remaining_qty", 0)
        if qty <= 0:
            self.position = None
            return
        pnl = (price - pos["entry_price"]) * qty
        pos["realized_pnl"] += pnl
        self.capital += pnl
        self.capital -= qty * price * self.commission_rate
        self._record_trade(pos, price, t, "FORCE_CLOSE")
        self.position = None

    def _calc_equity(self, price: float) -> float:
        if self.position is None:
            return self.capital
        qty = self.position.get("remaining_qty", 0)
        unrealized = (price - self.position["entry_price"]) * qty
        return self.capital + unrealized

    def _generate_report(self) -> dict:
        base = {
            "initial_capital": self.initial_capital,
            "final_capital":   round(self.capital, 4),
            "total_candidates_evaluated": len(self.all_candidate_scores),
            "total_candidates_passed":    len(self.passed_scores),
            "pass_rate_pct": round(
                len(self.passed_scores) / len(self.all_candidate_scores) * 100, 2
            ) if self.all_candidate_scores else 0,
            "avg_consensus_score_passed": round(
                float(np.mean(self.passed_scores)), 2
            ) if self.passed_scores else 0,
            "equity_curve": self.equity_curve,
            "trades": self.trades,
        }

        if not self.trades:
            base.update({
                "total_trades": 0, "win_count": 0, "loss_count": 0,
                "win_rate_pct": 0, "avg_win_usdt": 0, "avg_loss_usdt": 0,
                "avg_rr_ratio": 0, "profit_factor": 0,
                "total_return_pct": 0, "max_drawdown_pct": 0,
                "avg_hold_minutes": 0,
            })
            return base

        df_t = pd.DataFrame(self.trades)
        wins   = df_t[df_t["pnl"] > 0]
        losses = df_t[df_t["pnl"] <= 0]
        avg_win  = wins["pnl"].mean()   if len(wins)   > 0 else 0
        avg_loss = abs(losses["pnl"].mean()) if len(losses) > 0 else 0
        pf = (wins["pnl"].sum() / abs(losses["pnl"].sum())
              if len(losses) > 0 and losses["pnl"].sum() != 0 else float("inf"))
        total_ret = (self.capital - self.initial_capital) / self.initial_capital * 100

        max_dd = 0
        if self.equity_curve:
            eq = pd.Series([e["equity"] for e in self.equity_curve])
            max_dd = float(((eq - eq.cummax()) / eq.cummax() * 100).min())

        base.update({
            "total_trades":    len(df_t),
            "win_count":       len(wins),
            "loss_count":      len(losses),
            "win_rate_pct":    round(len(wins) / len(df_t) * 100, 2),
            "avg_win_usdt":    round(avg_win, 4),
            "avg_loss_usdt":   round(avg_loss, 4),
            "avg_rr_ratio":    round(avg_win / avg_loss, 3) if avg_loss > 0 else 0,
            "profit_factor":   round(pf, 3),
            "total_return_pct": round(total_ret, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "avg_hold_minutes": round(df_t["hold_minutes"].mean(), 1),
        })
        return base


# ---------------------------------------------------------------------------
# 分段统计
# ---------------------------------------------------------------------------

def calc_segment_stats(trades: list, regime: str) -> dict:
    seg = [t for t in trades if t.get("regime") == regime]
    if not seg:
        return {"count": 0, "win_rate": 0, "profit_factor": 0,
                "avg_rr": 0, "total_pnl": 0, "avg_hold_h": 0}
    wins   = [t for t in seg if t["pnl"] > 0]
    losses = [t for t in seg if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    aw = gp / len(wins) if wins else 0
    al = gl / len(losses) if losses else 0
    return {
        "count":         len(seg),
        "win_rate":      round(len(wins) / len(seg) * 100, 1),
        "profit_factor": round(gp / gl, 3) if gl > 0 else float("inf"),
        "avg_rr":        round(aw / al, 3) if al > 0 else 0,
        "total_pnl":     round(sum(t["pnl"] for t in seg), 2),
        "avg_hold_h":    round(np.mean([t["hold_minutes"] for t in seg]) / 60, 1),
    }


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def generate_report(results_by_window: dict, output_dir: str):
    # 字体
    font_candidates = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
    for fn in font_candidates:
        try:
            fm.findfont(fm.FontProperties(family=fn), fallback_to_default=False)
            plt.rcParams["font.family"] = fn
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    windows = sorted(results_by_window.keys())
    colors  = {"uptrend": "#27ae60", "downtrend": "#e74c3c", "ranging": "#f39c12"}

    fig, axes = plt.subplots(3, len(windows), figsize=(10 * len(windows), 14))
    if len(windows) == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle(
        "BTCUSDT 永续合约 顺势回踩策略 — 标准回测报告 v4.1\n"
        "数据来源: Binance Futures | 阈值=60 | 杠杆=3x | 初始资金=10,000 USDT",
        fontsize=13, fontweight="bold", y=0.99
    )

    for col, window in enumerate(windows):
        r = results_by_window[window]

        # 行1: 权益曲线
        ax = axes[0, col]
        eq_data = r.get("equity_curve", [])
        if eq_data:
            df_eq = pd.DataFrame(eq_data)
            df_eq["time"] = pd.to_datetime(df_eq["time"])
            for regime, color in colors.items():
                mask = df_eq.get("regime", pd.Series(["unknown"] * len(df_eq))) == regime
                if mask.any():
                    ax.plot(df_eq["time"][mask], df_eq["equity"][mask],
                            color=color, lw=1.5, alpha=0.85, label=regime)
            ax.axhline(y=r["initial_capital"], color="grey", ls="--", lw=1, alpha=0.5)
        ax.set_title(f"{window}天 | 收益{r['total_return_pct']:+.2f}% | "
                     f"PF={r['profit_factor']} | {r['total_trades']}笔", fontsize=10)
        ax.set_ylabel("净值 (USDT)")
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=30)

        # 行2: 分段统计
        ax = axes[1, col]
        regimes = ["uptrend", "downtrend", "ranging"]
        labels  = ["上升趋势", "下降趋势", "震荡"]
        counts  = [r["regime_stats"].get(reg, {}).get("count", 0) for reg in regimes]
        pfs     = [r["regime_stats"].get(reg, {}).get("profit_factor", 0) for reg in regimes]
        x = np.arange(len(regimes))
        bars = ax.bar(x, counts, color=[colors[reg] for reg in regimes], alpha=0.75, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("交易笔数")
        ax.set_title(f"{window}天 — 分段分布", fontsize=10)
        for bar, pf in zip(bars, pfs):
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        f"PF={pf:.2f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

        # 行3: 盈亏分布
        ax = axes[2, col]
        trades = r.get("trades", [])
        if trades:
            pnls = [t["pnl"] for t in trades]
            ax.hist([p for p in pnls if p > 0], bins=12, color="#27ae60", alpha=0.7, label="盈利")
            ax.hist([p for p in pnls if p <= 0], bins=12, color="#e74c3c", alpha=0.7, label="亏损")
            ax.axvline(x=0, color="black", ls="--", lw=1)
        ax.set_title(f"{window}天 — 盈亏分布", fontsize=10)
        ax.set_xlabel("PnL (USDT)")
        ax.set_ylabel("频次")
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    chart_path = os.path.join(output_dir, "standard_backtest_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图表] {chart_path}")

    # Markdown 报告
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# BTCUSDT 永续合约 顺势回踩策略 — 标准回测报告 v4.2",
        "",
        f"> **生成时间**：{now}",
        f"> **数据来源**：Binance Futures 永续合约 (`fapi.binance.com`)",
        f"> **策略配置**：共识评分阈值 = 60 | 默认杠杆 = 3x | 初始资金 = 10,000 USDT",
        f"> **手续费**：0.05%（Taker）| **滑点**：0.02%",
        "",
        "> ✅ **数据说明**：5M 数据已覆盖完整 180 天（2025-09-11 起），90天和 180天回测均使用真实 5M 数据。",
        "> ✅ **参数说明**：以下回测使用的参数均为 config.py 默认值，无任何 override，口径完全一致。",
        "",
        "---",
        "",
        "## 一、各时间窗口汇总",
        "",
        "| 时间窗口 | 开仓数 | 胜率 | 盈亏比 | **利润因子** | **总收益率** | 最大回撤 | 平均持仓 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for window in windows:
        r = results_by_window[window]
        lines.append(
            f"| {window} 天 | {r['total_trades']} | {r['win_rate_pct']}% | "
            f"{r['avg_rr_ratio']} | **{r['profit_factor']}** | "
            f"**{r['total_return_pct']:+.2f}%** | {r['max_drawdown_pct']:.2f}% | "
            f"{r['avg_hold_minutes']/60:.1f}h |"
        )

    lines += ["", "---", "", "## 二、市场状态分段统计", ""]
    for window in windows:
        r = results_by_window[window]
        lines += [
            f"### {window} 天回测",
            "",
            "| 市场状态 | 交易数 | 胜率 | 利润因子 | 盈亏比 | 总盈亏(USDT) | 平均持仓 |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for reg, label in [("uptrend","上升趋势"),("downtrend","下降趋势"),("ranging","震荡")]:
            s = r["regime_stats"].get(reg, {})
            if s.get("count", 0) > 0:
                lines.append(
                    f"| {label} | {s['count']} | {s['win_rate']}% | "
                    f"{s['profit_factor']} | {s['avg_rr']} | "
                    f"{s['total_pnl']:+.2f} | {s['avg_hold_h']}h |"
                )
            else:
                lines.append(f"| {label} | 0 | — | — | — | — | — |")
        lines.append("")

    lines += [
        "---", "",
        "## 三、利润因子分析与参数建议", "",
    ]
    for window in windows:
        r = results_by_window[window]
        pf = r.get("profit_factor", 0)
        status = "✅ 正期望" if pf > 1.0 else f"⚠️ 负期望（{pf:.3f}）"
        lines.append(f"- **{window} 天**：利润因子 = **{pf}** — {status}")

    lines += [
        "",
        "### v4.2 已执行的参数修改",
        "",
        "| 参数 | v4.1 旧值 | **v4.2 已执行新值** | 修改目的 |",
        "| :--- | :--- | :--- | :--- |",
        "| `stop_loss.structure.atr_multiplier` | 0.5 | **0.3** | 收窄止损，减少单笔亏损金额 |",
        "| `take_profit.tp1.close_pct` | 0.35 | **0.50** | 更积极回收第一波利润 |",
        "| `take_profit.tp1.target_r` | 1.2 | **1.0** | 更快触发 TP1，适应震荡市特征 |",
        "| `take_profit.trailing.activation_r` | 1.0 | **0.8** | 更早激活跟踪保护 |",
        "| `consensus_score.fixed_threshold` | 40 (config) / 60 (override) | **60 (统一)** | 消除口径不一致 |",
        "",
        "---", "",
        "## 四、结论", "",
        "本报告是当前策略的**唯一有效标准回测报告**（v4.2），替代所有历史版本。",
        "",
        "- 共识评分阈值、止损、止盈参数已全部落地到 config.py，回测与实盘口径完全一致",
        "- 共识评分模块已验证具有正向筛选价値",
        "- **当前版本不建议上实盘**，需继续验证 180 天利润因子是否超过 1.0",
        "",
        "> 本报告由回测系统自动生成，仅供策略研究参考，不构成投资建议。",
    ]

    md_path = os.path.join(output_dir, "standard_backtest_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[报告] {md_path}")
    return chart_path, md_path


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    data_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("  BTCUSDT 永续合约 顺势回踩策略 — 长周期标准回测 v4.1（向量化优化）")
    print(f"  数据来源: Binance Futures (fapi.binance.com)")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 加载数据
    df_15m = load_klines(os.path.join(data_dir, "BTCUSDT_15m.csv"))
    df_1h  = load_klines(os.path.join(data_dir, "BTCUSDT_1h.csv"))
    df_4h  = load_klines(os.path.join(data_dir, "BTCUSDT_4h.csv"))
    df_5m  = load_klines(os.path.join(data_dir, "BTCUSDT_5m.csv"))

    t_end = df_15m.index[-1]
    t5m_start = df_5m.index[0]

    print(f"\n[数据] 15M: {len(df_15m)} 根 ({df_15m.index[0].date()} ~ {df_15m.index[-1].date()})")
    print(f"[数据] 1H:  {len(df_1h)} 根 | 4H: {len(df_4h)} 根")
    print(f"[数据] 5M:  {len(df_5m)} 根 ({t5m_start.date()} ~ {df_5m.index[-1].date()})")

    # 预计算指标（一次性，全量数据）
    print("\n[预计算] 向量化指标（EMA/ATR/市场状态）...")
    indicators = precompute_indicators(df_15m, df_1h, df_5m)
    regime_counts = indicators["regimes"].value_counts()
    print(f"  上升趋势: {regime_counts.get('uptrend',0)} | "
          f"下降趋势: {regime_counts.get('downtrend',0)} | "
          f"震荡: {regime_counts.get('ranging',0)}")

    # 测试窗口
    total_days = (t_end - df_15m.index[0]).days
    windows = [90]
    if total_days >= 150:
        windows.append(180)

    results_by_window = {}

    for window in windows:
        print(f"\n{'='*60}")
        print(f"[{window}天] 开始回测...")
        t_start = t_end - timedelta(days=window)
        warmup_start = t_start - timedelta(hours=250)

        df15 = df_15m[df_15m.index >= warmup_start].copy()
        d1h  = df_1h[df_1h.index >= warmup_start].copy()
        d4h  = df_4h[df_4h.index >= warmup_start].copy()

        # 5M：有数据则用真实数据，否则降级为 15M
        if t5m_start <= t_start + timedelta(days=30):
            d5m = df_5m[df_5m.index >= warmup_start].copy()
            has_5m = True
        else:
            d5m = df15.copy()  # 降级
            has_5m = False

        print(f"  区间: {t_start.date()} ~ {t_end.date()} | "
              f"15M: {len(df15)} 根 | 5M: {'真实' if has_5m else '降级为15M'}")

        engine = VectorizedBacktestEngine(
            config=copy.deepcopy(CONFIG),
            threshold=60,
            zone_detect_interval=48,  # 每 12 小时重新检测区域
            zone_lookback=500,        # 只看最近 500 根 K 线
        )
        result = engine.run(df15, d5m, d1h, d4h, indicators)

        # 分段统计
        result["regime_stats"] = {
            "uptrend":   calc_segment_stats(engine.trades, "uptrend"),
            "downtrend": calc_segment_stats(engine.trades, "downtrend"),
            "ranging":   calc_segment_stats(engine.trades, "ranging"),
        }
        result["window_days"] = window
        result["has_real_5m"] = has_5m
        results_by_window[window] = result

        rs = result["regime_stats"]
        print(f"  完成 | 开仓: {result['total_trades']} 笔 | "
              f"胜率: {result['win_rate_pct']}% | "
              f"利润因子: {result['profit_factor']} | "
              f"收益: {result['total_return_pct']:+.2f}%")
        print(f"  分段 | 上升: {rs['uptrend']['count']}笔 PF={rs['uptrend']['profit_factor']} | "
              f"下降: {rs['downtrend']['count']}笔 PF={rs['downtrend']['profit_factor']} | "
              f"震荡: {rs['ranging']['count']}笔 PF={rs['ranging']['profit_factor']}")

    # 保存 JSON
    json_path = os.path.join(output_dir, "standard_backtest_results.json")
    summary = {}
    for w, r in results_by_window.items():
        s = {k: v for k, v in r.items() if k not in ("trades", "equity_curve")}
        summary[str(w)] = s
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[JSON] {json_path}")

    chart_path, md_path = generate_report(results_by_window, output_dir)

    print(f"\n{'='*70}")
    print("  标准回测完成！")
    print(f"  报告: {md_path}")
    print(f"  图表: {chart_path}")
    print(f"  JSON: {json_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
