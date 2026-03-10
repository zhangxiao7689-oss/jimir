"""
funnel_diagnosis.py — 交易漏斗逐层诊断脚本

逐层统计以下每一步的通过/淘汰数量：
  1. 趋势过滤（EMA200 + EMA50）
  2. 候选支撑区数量
  3. 评分通过数量（阈值=40）
  4. 假突破过滤
  5. 5M 确认通过数量（逐条件拆解）
  6. 止损计算有效数量
  7. 仓位计算有效数量
  8. 最终开仓数量

同时检查退出机制：
  - TP1 / TP2 / trailing stop 是否被执行
  - force_close 占比
  - hold_minutes 统计

数据来源说明：
  - 明确输出当前使用的是现货数据还是永续数据
"""

import os
import sys
import copy
import warnings
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from backtest import load_klines, resample_klines
from support_zone import SupportZoneDetector, get_nearest_support, get_nearest_resistance
from consensus_score import ConsensusScorer, FakeoutDetector
from entry_signal import EntryConfirmer
from exit_manager import StopLossCalculator, TakeProfitManager
from risk_manager import RiskManager, BinancePrecision


# ---------------------------------------------------------------------------
# 漏斗诊断引擎
# ---------------------------------------------------------------------------

class FunnelDiagnosticEngine:
    """
    逐层统计交易漏斗各环节的通过/淘汰数量。
    每根 K 线都记录被哪个环节拦截，并记录拦截原因。
    """

    def __init__(self, config: dict, threshold: float = 40):
        self.config = copy.deepcopy(config)
        self.config["consensus_score"]["fixed_threshold"] = threshold
        self.threshold = threshold

        self.zone_detector    = SupportZoneDetector(self.config)
        self.consensus_scorer = ConsensusScorer(self.config)
        self.fakeout_detector = FakeoutDetector(self.config)
        self.entry_confirmer  = EntryConfirmer(self.config)
        self.sl_calculator    = StopLossCalculator(self.config)
        precision = BinancePrecision()
        self.risk_manager     = RiskManager(self.config, precision)
        self.tp_manager       = TakeProfitManager(self.config)

        # 漏斗计数器
        self.counters = {
            "total_bars":             0,
            "has_position":           0,   # 已有持仓，跳过
            "trend_fail":             0,   # 趋势过滤不通过
            "trend_pass":             0,   # 趋势过滤通过
            "no_support_zones":       0,   # 无支撑区
            "no_candidate_zones":     0,   # 无候选区（价格不在区域附近）
            "candidate_zones_total":  0,   # 候选区总数（累计）
            "score_fail":             0,   # 评分不通过
            "score_pass":             0,   # 评分通过（至少1个区域通过）
            "fakeout_blocked":        0,   # 假突破检测拦截
            "confirm_fail_not_in_zone": 0, # 5M确认失败：价格不在区域内
            "confirm_fail_bearish":   0,   # 5M确认失败：阴线
            "confirm_fail_body":      0,   # 5M确认失败：实体比例不足
            "confirm_pass":           0,   # 5M确认通过
            "sl_fail":                0,   # 止损计算无效
            "sl_pass":                0,   # 止损计算有效
            "pos_fail":               0,   # 仓位计算无效
            "pos_pass":               0,   # 仓位计算有效（= 最终开仓）
            "opened":                 0,   # 实际开仓
        }

        # 拦截原因详细记录（采样）
        self.rejection_samples = defaultdict(list)

        # 持仓和交易记录
        self.capital = self.config.get("backtest", {}).get("initial_capital", 10000)
        self.commission_rate = self.config.get("backtest", {}).get("commission_rate", 0.0005)
        self.slippage_pct    = self.config.get("backtest", {}).get("slippage_pct", 0.0002)
        self.position = None
        self.trades = []
        self.equity_curve = []

        # 退出机制统计
        self.exit_stats = {
            "tp1_triggered": 0,
            "tp2_triggered": 0,
            "trailing_triggered": 0,
            "stop_loss_triggered": 0,
            "force_close": 0,
        }

        # 区域缓存
        self._cached_zones = None

    def run(self, df_15m, df_5m=None, df_1h=None, df_4h=None):
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
            self.counters["total_bars"] += 1
            current_time  = df_15m.index[i]
            current_price = df_15m["close"].iloc[i]

            df_15m_slice = df_15m.iloc[:i + 1]
            df_1h_slice  = df_1h[df_1h.index <= current_time]
            df_4h_slice  = df_4h[df_4h.index <= current_time]
            df_5m_slice  = df_5m[df_5m.index <= current_time]

            # 权益曲线
            if i % 5 == 0:
                eq = self._calc_equity(current_price)
                self.equity_curve.append({"time": current_time, "equity": eq})

            # 更新持仓
            if self.position is not None:
                self._update_position(current_price, df_5m_slice, current_time)

            # --- 入场漏斗 ---
            if self.position is not None:
                self.counters["has_position"] += 1
                continue

            # 1. 趋势过滤
            if len(df_1h_slice) >= ema_slow_p:
                ema200 = df_1h_slice["close"].ewm(span=ema_slow_p, adjust=False).mean().iloc[-1]
                ema50  = df_1h_slice["close"].ewm(span=ema_fast_p, adjust=False).mean().iloc[-1]
                trend_ok = (current_price > ema200 and ema50 > ema200)
            else:
                trend_ok = True  # 数据不足时不过滤

            if not trend_ok:
                self.counters["trend_fail"] += 1
                if len(self.rejection_samples["trend_fail"]) < 5:
                    self.rejection_samples["trend_fail"].append({
                        "time": str(current_time),
                        "price": current_price,
                        "ema200": round(ema200, 2),
                        "ema50": round(ema50, 2),
                    })
                continue
            self.counters["trend_pass"] += 1

            # 2. 区域检测（每3根刷新一次）
            if i % 3 == 0 or self._cached_zones is None:
                self._cached_zones = self.zone_detector.detect(df_15m_slice, current_price)

            zones = self._cached_zones
            support_zones = [z for z in zones if z["zone_type"] == "support"]

            if not support_zones:
                self.counters["no_support_zones"] += 1
                continue

            candidate_zones = [
                z for z in support_zones
                if z["price_start"] * (1 - zone_prox) <= current_price
            ]

            if not candidate_zones:
                self.counters["no_candidate_zones"] += 1
                if len(self.rejection_samples["no_candidate"]) < 5:
                    self.rejection_samples["no_candidate"].append({
                        "time": str(current_time),
                        "price": current_price,
                        "support_zones": [(round(z["price_start"],2), round(z["price_end"],2)) for z in support_zones[:3]],
                        "zone_prox_pct": zone_prox,
                    })
                continue

            self.counters["candidate_zones_total"] += len(candidate_zones)

            # 3. 批量评分
            self.consensus_scorer.mtf_klines = {"1h": df_1h_slice, "4h": df_4h_slice}
            score_results = self.consensus_scorer.score_batch(candidate_zones, df_15m_slice)
            passed_results = [r for r in score_results if r["passed_threshold"]]

            if not passed_results:
                self.counters["score_fail"] += 1
                if len(self.rejection_samples["score_fail"]) < 5:
                    scores = [round(r["total_score"], 1) for r in score_results]
                    self.rejection_samples["score_fail"].append({
                        "time": str(current_time),
                        "price": current_price,
                        "scores": scores,
                        "threshold": self.threshold,
                    })
                continue
            self.counters["score_pass"] += 1

            # 选最优区域
            passed_results.sort(key=lambda r: (-r["total_score"], abs(r["zone"]["mid_price"] - current_price)))
            best_result = passed_results[0]
            nearest_support = best_result["zone"]

            # 4. 假突破检测
            fakeout_result = self.fakeout_detector.check(best_result, df_5m_slice)
            if fakeout_result.get("fakeout_type") == "true_breakdown":
                self.counters["fakeout_blocked"] += 1
                continue

            # 5. 5M 确认（逐条件拆解）
            if len(df_5m_slice) < 5:
                continue

            last_5m = df_5m_slice.iloc[-1]
            zone_mid = nearest_support["mid_price"]
            close_c  = last_5m["close"]
            open_c   = last_5m["open"]
            high_c   = last_5m["high"]
            low_c    = last_5m["low"]

            entry_cfg = self.config.get("entry_signal", {})
            zone_proximity = entry_cfg.get("zone_proximity_pct", 0.005)
            min_body_ratio = entry_cfg.get("min_body_ratio", 0.4)

            # 条件1：价格在支撑区内
            in_zone = (
                nearest_support["price_start"] * (1 - zone_proximity) <= close_c <=
                nearest_support["price_end"] * (1 + zone_proximity)
            )
            if not in_zone:
                self.counters["confirm_fail_not_in_zone"] += 1
                if len(self.rejection_samples["not_in_zone"]) < 10:
                    self.rejection_samples["not_in_zone"].append({
                        "time": str(current_time),
                        "price": round(close_c, 2),
                        "zone_start": round(nearest_support["price_start"], 2),
                        "zone_end": round(nearest_support["price_end"], 2),
                        "zone_mid": round(zone_mid, 2),
                        "dist_pct": round(abs(close_c - zone_mid) / zone_mid * 100, 3),
                    })
                continue

            # 条件2：阳线
            is_bullish = close_c > open_c
            if not is_bullish:
                self.counters["confirm_fail_bearish"] += 1
                continue

            # 条件3：实体比例
            total_range = high_c - low_c
            body = abs(close_c - open_c)
            body_ratio = body / total_range if total_range > 0 else 0
            if body_ratio < min_body_ratio:
                self.counters["confirm_fail_body"] += 1
                if len(self.rejection_samples["body_fail"]) < 5:
                    self.rejection_samples["body_fail"].append({
                        "time": str(current_time),
                        "body_ratio": round(body_ratio, 3),
                        "required": min_body_ratio,
                    })
                continue

            self.counters["confirm_pass"] += 1

            # 6. 止损计算
            sl_result = self.sl_calculator.calculate(current_price, nearest_support, df_15m_slice, df_5m_slice)
            if not sl_result["valid"]:
                self.counters["sl_fail"] += 1
                if len(self.rejection_samples["sl_fail"]) < 5:
                    self.rejection_samples["sl_fail"].append({
                        "time": str(current_time),
                        "price": current_price,
                        "reason": sl_result.get("reason", "unknown"),
                        "stop_price": sl_result.get("stop_price"),
                    })
                continue
            self.counters["sl_pass"] += 1
            stop_price = sl_result["stop_price"]

            # 7. 仓位计算
            pos_result = self.risk_manager.calculate_position(self.capital, current_price, stop_price, 0)
            if not pos_result["valid"]:
                self.counters["pos_fail"] += 1
                if len(self.rejection_samples["pos_fail"]) < 5:
                    self.rejection_samples["pos_fail"].append({
                        "time": str(current_time),
                        "price": current_price,
                        "stop_price": stop_price,
                        "stop_dist_pct": round(abs(current_price - stop_price) / current_price * 100, 3),
                        "reason": pos_result.get("reason", "unknown"),
                        "nominal_usdt": pos_result.get("nominal_usdt"),
                    })
                continue
            self.counters["pos_pass"] += 1
            self.counters["opened"] += 1

            # 开仓
            qty = pos_result["qty"]
            actual_entry = current_price * (1 + self.slippage_pct)
            self.capital -= actual_entry * qty * self.commission_rate

            resistance_zones = [z for z in zones if z["zone_type"] == "resistance"]
            nearest_resistance = get_nearest_resistance(resistance_zones, actual_entry)

            self.position = self.tp_manager.init_position(actual_entry, stop_price, qty, nearest_resistance)
            self.position["entry_time"] = current_time
            self.position["score"] = best_result["total_score"]

        # 强制平仓
        if self.position is not None:
            self._force_close(df_15m["close"].iloc[-1], df_15m.index[-1])

        return self._build_report()

    def _calc_equity(self, price):
        if self.position is None:
            return self.capital
        qty = self.position.get("remaining_qty", self.position.get("initial_qty", 0))
        return self.capital + (price - self.position["entry_price"]) * qty

    def _update_position(self, current_price, df_5m, current_time):
        updated = self.tp_manager.update(self.position, current_price, df_5m)
        actions = updated.pop("actions", [])
        self.position.update(updated)

        for action in actions:
            pnl = action.get("pnl", 0)
            self.capital += pnl
            commission = action.get("qty", 0) * current_price * self.commission_rate
            self.capital -= commission

            act = action.get("action", "")
            if act == "TP1":
                self.exit_stats["tp1_triggered"] += 1
            elif act == "TP2":
                self.exit_stats["tp2_triggered"] += 1
            elif act == "TRAILING":
                self.exit_stats["trailing_triggered"] += 1
            elif act == "STOP_LOSS":
                self.exit_stats["stop_loss_triggered"] += 1

            if act in ("CLOSED", "STOP_LOSS"):
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
                    "exit_reason": act,
                    "hold_minutes": hold_minutes,
                    "tp1_triggered": self.position.get("tp1_triggered", False),
                    "tp2_triggered": self.position.get("tp2_triggered", False),
                    "trailing_active": self.position.get("trailing_active", False),
                })
                self.position = None
                break

    def _force_close(self, price, time):
        if self.position is None:
            return
        qty = self.position.get("remaining_qty", self.position.get("initial_qty", 0))
        pnl = (price - self.position["entry_price"]) * qty
        self.capital += pnl
        hold_minutes = 0
        if "entry_time" in self.position:
            hold_minutes = (time - self.position["entry_time"]).total_seconds() / 60
        self.exit_stats["force_close"] += 1
        self.trades.append({
            "entry_time": self.position.get("entry_time"),
            "exit_time": time,
            "entry_price": self.position.get("entry_price"),
            "exit_price": price,
            "pnl": self.position.get("realized_pnl", 0) + pnl,
            "score": self.position.get("score", 0),
            "exit_reason": "FORCE_CLOSE",
            "hold_minutes": hold_minutes,
            "tp1_triggered": self.position.get("tp1_triggered", False),
            "tp2_triggered": self.position.get("tp2_triggered", False),
            "trailing_active": self.position.get("trailing_active", False),
        })
        self.position = None

    def _build_report(self):
        c = self.counters
        bars_no_position = c["total_bars"] - c["has_position"]

        report = {
            "threshold": self.threshold,
            "counters": c,
            "rejection_samples": dict(self.rejection_samples),
            "exit_stats": self.exit_stats,
            "trades": self.trades,
            "funnel_summary": {
                "total_bars":                c["total_bars"],
                "bars_with_position":        c["has_position"],
                "bars_available_for_entry":  bars_no_position,
                "trend_pass":                c["trend_pass"],
                "trend_fail":                c["trend_fail"],
                "no_support_zones":          c["no_support_zones"],
                "no_candidate_zones":        c["no_candidate_zones"],
                "candidate_zones_total":     c["candidate_zones_total"],
                "score_pass_bars":           c["score_pass"],
                "score_fail_bars":           c["score_fail"],
                "fakeout_blocked":           c["fakeout_blocked"],
                "confirm_fail_not_in_zone":  c["confirm_fail_not_in_zone"],
                "confirm_fail_bearish":      c["confirm_fail_bearish"],
                "confirm_fail_body":         c["confirm_fail_body"],
                "confirm_pass":              c["confirm_pass"],
                "sl_fail":                   c["sl_fail"],
                "sl_pass":                   c["sl_pass"],
                "pos_fail":                  c["pos_fail"],
                "pos_pass":                  c["pos_pass"],
                "opened":                    c["opened"],
            },
        }
        return report


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    data_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("  交易漏斗逐层诊断报告")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- 数据来源说明 ---
    print("\n[数据来源]")
    print("  当前使用的是 Binance 现货 BTCUSDT K线数据（通过 api.binance.com 下载）。")
    print("  Binance Futures 永续合约 API（fapi.binance.com）因地区限制无法访问。")
    print("  现货与永续价格高度相关（价差通常 < 0.1%），对策略逻辑验证影响有限，")
    print("  但资金费率、强平机制等期货特性无法模拟。")
    print("  建议：在可访问 Binance Futures API 的环境中运行 download_data.py 获取真实永续数据。")

    # 加载数据
    path_15m = os.path.join(data_dir, "BTCUSDT_15m.csv")
    path_5m  = os.path.join(data_dir, "BTCUSDT_5m.csv")
    path_1h  = os.path.join(data_dir, "BTCUSDT_1h.csv")

    df_15m_full = load_klines(path_15m)
    df_5m_full  = load_klines(path_5m)  if os.path.exists(path_5m)  else None
    df_1h_full  = load_klines(path_1h)  if os.path.exists(path_1h)  else None
    df_4h_full  = resample_klines(df_1h_full if df_1h_full is not None else df_15m_full, "4h")

    # 截取最近 90 天 + 预热
    cutoff = df_15m_full.index[-1] - timedelta(days=90)
    warmup_cutoff = cutoff - timedelta(hours=200)

    df_15m = df_15m_full[df_15m_full.index >= warmup_cutoff].copy()
    df_5m  = df_5m_full[df_5m_full.index >= warmup_cutoff].copy() if df_5m_full is not None else None
    df_1h  = df_1h_full[df_1h_full.index >= warmup_cutoff].copy() if df_1h_full is not None else None
    df_4h  = df_4h_full[df_4h_full.index >= warmup_cutoff].copy()

    print(f"\n[数据] 15M: {len(df_15m)} 根 | 回测区间: {cutoff.strftime('%Y-%m-%d')} ~ {df_15m_full.index[-1].strftime('%Y-%m-%d')}")

    # 运行诊断（阈值=40）
    print("\n[运行] 漏斗诊断（阈值=40）...", flush=True)
    engine = FunnelDiagnosticEngine(CONFIG, threshold=40)
    report = engine.run(df_15m, df_5m=df_5m, df_1h=df_1h, df_4h=df_4h)

    # 打印漏斗报告
    print_funnel_report(report)

    # 打印退出机制报告
    print_exit_report(report)

    # 打印拒绝原因样本
    print_rejection_samples(report)

    # 生成可视化
    chart_path = os.path.join(output_dir, "funnel_chart.png")
    try:
        plot_funnel(report, chart_path)
    except Exception as e:
        print(f"[警告] 图表生成失败: {e}")

    # 生成 Markdown 报告
    md_path = os.path.join(output_dir, "funnel_diagnosis_report.md")
    write_md_report(report, md_path)

    print(f"\n[完成] 报告已生成:")
    print(f"  图表: {chart_path}")
    print(f"  报告: {md_path}")


def print_funnel_report(report):
    fs = report["funnel_summary"]
    c  = report["counters"]

    print("\n" + "=" * 70)
    print("  交易漏斗逐层统计（阈值=40）")
    print("=" * 70)

    def pct(a, b):
        return f"{a/b*100:.1f}%" if b > 0 else "N/A"

    rows = [
        ("总 K 线数（预热后）",   fs["total_bars"],               "—"),
        ("  已有持仓（跳过）",     fs["bars_with_position"],       pct(fs["bars_with_position"], fs["total_bars"])),
        ("  可入场 K 线数",        fs["bars_available_for_entry"], pct(fs["bars_available_for_entry"], fs["total_bars"])),
        ("", "", ""),
        ("[1] 趋势过滤通过",       fs["trend_pass"],               pct(fs["trend_pass"], fs["bars_available_for_entry"])),
        ("    趋势过滤拦截",       fs["trend_fail"],               pct(fs["trend_fail"], fs["bars_available_for_entry"])),
        ("", "", ""),
        ("[2] 无支撑区（拦截）",   fs["no_support_zones"],         pct(fs["no_support_zones"], fs["trend_pass"])),
        ("    无候选区（拦截）",   fs["no_candidate_zones"],       pct(fs["no_candidate_zones"], fs["trend_pass"])),
        ("    候选区总数（累计）", fs["candidate_zones_total"],    "—"),
        ("", "", ""),
        ("[3] 评分通过（K线数）",  fs["score_pass_bars"],          pct(fs["score_pass_bars"], fs["trend_pass"])),
        ("    评分拦截（K线数）",  fs["score_fail_bars"],          pct(fs["score_fail_bars"], fs["trend_pass"])),
        ("", "", ""),
        ("[4] 假突破拦截",         fs["fakeout_blocked"],          pct(fs["fakeout_blocked"], fs["score_pass_bars"])),
        ("", "", ""),
        ("[5] 5M确认：价格不在区域", fs["confirm_fail_not_in_zone"], pct(fs["confirm_fail_not_in_zone"], fs["score_pass_bars"])),
        ("    5M确认：阴线拦截",   fs["confirm_fail_bearish"],     pct(fs["confirm_fail_bearish"], fs["score_pass_bars"])),
        ("    5M确认：实体不足",   fs["confirm_fail_body"],        pct(fs["confirm_fail_body"], fs["score_pass_bars"])),
        ("    5M确认通过",         fs["confirm_pass"],             pct(fs["confirm_pass"], fs["score_pass_bars"])),
        ("", "", ""),
        ("[6] 止损计算有效",       fs["sl_pass"],                  pct(fs["sl_pass"], fs["confirm_pass"])),
        ("    止损计算无效（拦截）", fs["sl_fail"],                 pct(fs["sl_fail"], fs["confirm_pass"])),
        ("", "", ""),
        ("[7] 仓位计算有效",       fs["pos_pass"],                 pct(fs["pos_pass"], fs["sl_pass"])),
        ("    仓位计算无效（拦截）", fs["pos_fail"],                pct(fs["pos_fail"], fs["sl_pass"])),
        ("", "", ""),
        ("[8] 最终开仓数",         fs["opened"],                   pct(fs["opened"], fs["total_bars"])),
    ]

    for label, count, pct_str in rows:
        if label == "":
            print()
            continue
        print(f"  {label:<30} {str(count):>8}   {pct_str:>8}")

    print("=" * 70)


def print_exit_report(report):
    es = report["exit_stats"]
    trades = report["trades"]

    print("\n" + "=" * 70)
    print("  退出机制统计")
    print("=" * 70)
    print(f"  总交易数:           {len(trades)}")
    print(f"  TP1 触发次数:       {es['tp1_triggered']}")
    print(f"  TP2 触发次数:       {es['tp2_triggered']}")
    print(f"  跟踪止盈触发次数:   {es['trailing_triggered']}")
    print(f"  止损触发次数:       {es['stop_loss_triggered']}")
    print(f"  强制平仓（回测结束）: {es['force_close']}")

    if trades:
        df_t = pd.DataFrame(trades)
        print(f"\n  交易详情:")
        for _, t in df_t.iterrows():
            hold_h = t.get("hold_minutes", 0) / 60
            print(f"    入场: {t.get('entry_time')} | 出场: {t.get('exit_time')}")
            print(f"    入场价: {t.get('entry_price', 0):.2f} | 出场价: {t.get('exit_price', 0):.2f}")
            print(f"    盈亏: {t.get('pnl', 0):+.4f} USDT | 持仓: {hold_h:.1f}h | 原因: {t.get('exit_reason')}")
            print(f"    TP1触发: {t.get('tp1_triggered')} | TP2触发: {t.get('tp2_triggered')} | 跟踪: {t.get('trailing_active')}")
            print()

        total_bars = report["counters"]["total_bars"]
        bars_per_15m = 4  # 每小时4根
        total_hours = total_bars / bars_per_15m
        avg_hold = df_t["hold_minutes"].mean() if len(df_t) > 0 else 0
        print(f"  平均持仓时长: {avg_hold:.1f} 分钟 ({avg_hold/60:.1f} 小时)")
        print(f"  回测总时长:   约 {total_hours:.0f} 小时 ({total_hours/24:.0f} 天)")
        if avg_hold > 0 and total_hours > 0:
            hold_pct = avg_hold / 60 / total_hours * 100
            print(f"  持仓时长占回测时长: {hold_pct:.1f}%")

    print("=" * 70)


def print_rejection_samples(report):
    samples = report["rejection_samples"]

    print("\n" + "=" * 70)
    print("  关键拦截原因样本（每类最多5条）")
    print("=" * 70)

    key_map = {
        "not_in_zone": "5M确认失败：价格不在区域内",
        "score_fail":  "评分拦截：分数不足",
        "sl_fail":     "止损计算无效",
        "pos_fail":    "仓位计算无效",
        "trend_fail":  "趋势过滤拦截",
        "body_fail":   "5M确认失败：实体比例不足",
        "no_candidate": "无候选区（价格不在区域附近）",
    }

    for key, label in key_map.items():
        if key in samples and samples[key]:
            print(f"\n  [{label}]")
            for s in samples[key][:5]:
                print(f"    {s}")

    print("=" * 70)


def plot_funnel(report, output_path):
    fs = report["funnel_summary"]

    font_candidates = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
    for fn in font_candidates:
        try:
            fm.findfont(fm.FontProperties(family=fn), fallback_to_default=False)
            plt.rcParams["font.family"] = fn
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("交易漏斗逐层诊断 — BTCUSDT（最近90天，阈值=40）", fontsize=14, fontweight="bold")

    # 左图：漏斗瀑布图
    ax = axes[0]
    stages = [
        ("可入场K线", fs["bars_available_for_entry"]),
        ("趋势通过",  fs["trend_pass"]),
        ("有候选区",  fs["trend_pass"] - fs["no_support_zones"] - fs["no_candidate_zones"]),
        ("评分通过",  fs["score_pass_bars"]),
        ("5M确认",   fs["confirm_pass"]),
        ("止损有效",  fs["sl_pass"]),
        ("仓位有效",  fs["pos_pass"]),
        ("最终开仓",  fs["opened"]),
    ]
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(stages)))

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white", height=0.6)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", fontsize=9)
    ax.set_xlabel("K线数量")
    ax.set_title("交易漏斗（各环节通过数量）")
    ax.set_xlim(0, max(values) * 1.15)
    ax.grid(axis="x", alpha=0.3)

    # 右图：5M确认拦截细分
    ax2 = axes[1]
    confirm_data = {
        "价格不在区域内": fs["confirm_fail_not_in_zone"],
        "阴线拦截":       fs["confirm_fail_bearish"],
        "实体比例不足":   fs["confirm_fail_body"],
        "通过确认":       fs["confirm_pass"],
    }
    total_confirm = sum(confirm_data.values())
    if total_confirm > 0:
        labels2 = list(confirm_data.keys())
        values2 = list(confirm_data.values())
        colors2 = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]
        wedges, texts, autotexts = ax2.pie(
            values2, labels=labels2, colors=colors2,
            autopct=lambda p: f"{p:.1f}%\n({int(p*total_confirm/100)})",
            startangle=90, pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(8)
        ax2.set_title(f"5M确认环节细分\n（总计 {total_confirm} 次评分通过后进入此环节）")
    else:
        ax2.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("5M确认环节细分")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图表] 已保存至: {output_path}")


def write_md_report(report, output_path):
    fs = report["funnel_summary"]
    es = report["exit_stats"]
    trades = report["trades"]
    samples = report["rejection_samples"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def pct(a, b):
        return f"{a/b*100:.1f}%" if b > 0 else "N/A"

    lines = [
        "# 交易漏斗逐层诊断报告",
        "",
        f"> 生成时间：{now}  |  回测区间：最近 90 天  |  评分阈值：40",
        "",
        "## 一、数据来源说明",
        "",
        "**当前使用的是 Binance 现货 BTCUSDT K线数据**（通过 `api.binance.com` 下载）。",
        "",
        "Binance Futures 永续合约 API（`fapi.binance.com`）因地区限制（HTTP 451）在当前沙箱环境中无法直接访问。",
        "现货与永续价格高度相关（价差通常 < 0.1%），对策略逻辑验证影响有限，但资金费率、强平机制等期货特性无法模拟。",
        "",
        "> **建议**：在可访问 Binance Futures API 的网络环境中，运行 `download_data.py` 并修改 `BASE_URL` 为 `https://fapi.binance.com`，即可获取真实永续合约数据。",
        "",
        "## 二、交易漏斗逐层统计",
        "",
        "| 环节 | 数量 | 占可入场K线比例 |",
        "| :--- | ---: | ---: |",
        f"| 总 K 线数（预热后） | {fs['total_bars']:,} | 100% |",
        f"| 已有持仓（跳过） | {fs['bars_with_position']:,} | {pct(fs['bars_with_position'], fs['total_bars'])} |",
        f"| **可入场 K 线数** | **{fs['bars_available_for_entry']:,}** | **100%** |",
        f"| [1] 趋势过滤通过 | {fs['trend_pass']:,} | {pct(fs['trend_pass'], fs['bars_available_for_entry'])} |",
        f"| [1] 趋势过滤拦截 | {fs['trend_fail']:,} | {pct(fs['trend_fail'], fs['bars_available_for_entry'])} |",
        f"| [2] 无支撑区（拦截） | {fs['no_support_zones']:,} | {pct(fs['no_support_zones'], fs['trend_pass'])} |",
        f"| [2] 无候选区（拦截） | {fs['no_candidate_zones']:,} | {pct(fs['no_candidate_zones'], fs['trend_pass'])} |",
        f"| [2] 候选区总数（累计） | {fs['candidate_zones_total']:,} | — |",
        f"| [3] 评分通过（K线数） | {fs['score_pass_bars']:,} | {pct(fs['score_pass_bars'], fs['trend_pass'])} |",
        f"| [3] 评分拦截（K线数） | {fs['score_fail_bars']:,} | {pct(fs['score_fail_bars'], fs['trend_pass'])} |",
        f"| [4] 假突破拦截 | {fs['fakeout_blocked']:,} | {pct(fs['fakeout_blocked'], fs['score_pass_bars'])} |",
        f"| [5] 5M确认：价格不在区域 | {fs['confirm_fail_not_in_zone']:,} | {pct(fs['confirm_fail_not_in_zone'], fs['score_pass_bars'])} |",
        f"| [5] 5M确认：阴线拦截 | {fs['confirm_fail_bearish']:,} | {pct(fs['confirm_fail_bearish'], fs['score_pass_bars'])} |",
        f"| [5] 5M确认：实体不足 | {fs['confirm_fail_body']:,} | {pct(fs['confirm_fail_body'], fs['score_pass_bars'])} |",
        f"| **[5] 5M确认通过** | **{fs['confirm_pass']:,}** | **{pct(fs['confirm_pass'], fs['score_pass_bars'])}** |",
        f"| [6] 止损计算有效 | {fs['sl_pass']:,} | {pct(fs['sl_pass'], fs['confirm_pass'])} |",
        f"| [6] 止损计算无效 | {fs['sl_fail']:,} | {pct(fs['sl_fail'], fs['confirm_pass'])} |",
        f"| [7] 仓位计算有效 | {fs['pos_pass']:,} | {pct(fs['pos_pass'], fs['sl_pass'])} |",
        f"| [7] 仓位计算无效 | {fs['pos_fail']:,} | {pct(fs['pos_fail'], fs['sl_pass'])} |",
        f"| **[8] 最终开仓数** | **{fs['opened']:,}** | **{pct(fs['opened'], fs['total_bars'])}** |",
        "",
        "## 三、退出机制统计",
        "",
        f"| 退出类型 | 次数 |",
        f"| :--- | ---: |",
        f"| TP1 触发 | {es['tp1_triggered']} |",
        f"| TP2 触发 | {es['tp2_triggered']} |",
        f"| 跟踪止盈触发 | {es['trailing_triggered']} |",
        f"| 止损触发 | {es['stop_loss_triggered']} |",
        f"| 强制平仓（回测结束） | {es['force_close']} |",
        f"| **总交易数** | **{len(trades)}** |",
        "",
    ]

    if trades:
        lines += ["### 交易详情", ""]
        df_t = pd.DataFrame(trades)
        for _, t in df_t.iterrows():
            hold_h = t.get("hold_minutes", 0) / 60
            lines.append(
                f"- **入场** {t.get('entry_time')} @ {t.get('entry_price',0):.2f} → "
                f"**出场** {t.get('exit_time')} @ {t.get('exit_price',0):.2f} | "
                f"盈亏: {t.get('pnl',0):+.4f} USDT | 持仓: {hold_h:.1f}h | "
                f"原因: `{t.get('exit_reason')}` | "
                f"TP1: {t.get('tp1_triggered')} | TP2: {t.get('tp2_triggered')} | 跟踪: {t.get('trailing_active')}"
            )
        lines.append("")

    lines += [
        "## 四、关键拦截原因样本",
        "",
    ]

    key_map = {
        "not_in_zone": "5M确认失败：价格不在区域内",
        "score_fail":  "评分拦截：分数不足",
        "sl_fail":     "止损计算无效",
        "pos_fail":    "仓位计算无效",
        "trend_fail":  "趋势过滤拦截",
        "body_fail":   "5M确认失败：实体比例不足",
        "no_candidate": "无候选区（价格不在区域附近）",
    }

    for key, label in key_map.items():
        if key in samples and samples[key]:
            lines.append(f"### {label}")
            lines.append("")
            for s in samples[key][:5]:
                lines.append(f"```")
                lines.append(str(s))
                lines.append(f"```")
            lines.append("")

    lines += [
        "## 五、核心结论与修复建议",
        "",
        "根据以上漏斗数据，主要瓶颈环节为（按影响程度排序）：",
        "",
        "1. **[待填入]** 根据实际运行结果确定主瓶颈",
        "2. **[待填入]** 次要瓶颈",
        "",
        "---",
        "",
        "> 本报告由诊断系统自动生成，仅供参考。",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[报告] Markdown 报告已保存至: {output_path}")


if __name__ == "__main__":
    main()
