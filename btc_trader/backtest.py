"""
backtest.py — 对比回测框架

职责：
  - 在历史K线数据上模拟完整的交易策略执行
  - 支持"有共识评分过滤"和"无共识评分过滤"的双模式对比
  - 输出完整的绩效指标对比报告

支持的对比模式：
  Mode A: use_consensus_score=False  原策略（无评分过滤）
  Mode B: use_consensus_score=True   新策略（有评分过滤）

输出指标：
  - 胜率
  - 平均盈亏比
  - 总收益率
  - 最大回撤
  - 利润因子（Profit Factor）
  - 交易次数
  - 每笔平均持仓时长

使用方式：
  python backtest.py --data data/BTCUSDT_15m.csv --mode compare
"""

import os
import sys
import json
import argparse
import copy
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 本地模块
from config import CONFIG
from support_zone import SupportZoneDetector, get_nearest_support, get_nearest_resistance
from consensus_score import ConsensusScorer, FakeoutDetector
from entry_signal import EntryConfirmer
from exit_manager import StopLossCalculator, TakeProfitManager
from risk_manager import RiskManager, BinancePrecision
from logger import setup_logger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_klines(filepath: str, timeframe: str = "15m") -> pd.DataFrame:
    """
    从 CSV 文件加载K线数据。

    CSV 格式（列名不区分大小写）：
      timestamp, open, high, low, close, volume
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [c.lower() for c in df.columns]

    # 时间列处理
    time_col = next((c for c in df.columns if "time" in c or "date" in c), None)
    if time_col:
        df["time"] = pd.to_datetime(df[time_col])
        df = df.set_index("time").sort_index()
    else:
        df.index = pd.to_datetime(df.index)

    # 确保必要列存在
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"数据文件缺少列: {col}")

    df[required] = df[required].astype(float)
    logger.info(f"加载{timeframe}数据: {len(df)}根K线 | {df.index[0]} ~ {df.index[-1]}")
    return df


def resample_klines(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """将K线数据重采样到更高时间周期。"""
    tf_map = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
    rule = tf_map.get(target_tf, target_tf)

    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return resampled


# ---------------------------------------------------------------------------
# 回测引擎
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    事件驱动回测引擎。

    工作方式：
      - 以15M K线为主循环单位
      - 每根K线结束时检查入场信号
      - 持仓期间每根5M K线更新止盈止损状态
    """

    def __init__(self, config: Dict, use_consensus_score: bool = True):
        self.config = copy.deepcopy(config)
        self.use_consensus_score = use_consensus_score

        # 如果不使用共识评分，临时禁用
        if not use_consensus_score:
            self.config["consensus_score"]["enabled"] = False

        bt_cfg = config.get("backtest", {})
        self.initial_capital = bt_cfg.get("initial_capital", 10000)
        self.commission_rate = bt_cfg.get("commission_rate", 0.0005)
        self.slippage_pct = bt_cfg.get("slippage_pct", 0.0002)

        # 子模块（注意：ConsensusScorer 的 mtf_klines 在 run() 时动态传入）
        self.zone_detector = SupportZoneDetector(self.config)
        self.consensus_scorer = ConsensusScorer(self.config)  # mtf_klines 在每次评分时传入
        self.fakeout_detector = FakeoutDetector(self.config)
        self.entry_confirmer = EntryConfirmer(self.config)
        self.sl_calculator = StopLossCalculator(self.config)
        self.tp_manager = TakeProfitManager(self.config)
        precision = BinancePrecision()
        self.risk_manager = RiskManager(self.config, precision)

        # 回测状态
        self.capital = self.initial_capital
        self.position = None       # 当前持仓状态（None=无持仓）
        self.trades = []           # 已完成交易记录
        self.equity_curve = []     # 权益曲线
        # 评分统计：记录所有候选区域的评分（无论是否开仓）
        self.all_candidate_scores: List[float] = []
        self.passed_scores: List[float] = []

    def run(
        self,
        df_15m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame] = None,
        df_1h: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        执行回测主循环。

        :param df_15m: 15M K线数据（主循环）
        :param df_5m:  5M K线数据（用于入场确认和退出管理）
        :param df_1h:  1H K线数据（用于趋势过滤 + mtf评分）
        :param df_4h:  4H K线数据（用于 mtf 评分）
        :return:       回测结果字典
        """
        score_mode = '开启' if self.use_consensus_score else '关闭'
        logger.info(
            f"\n{'='*60}\n"
            f"  回测开始 | 共识评分={score_mode}\n"
            f"  数据范围: {df_15m.index[0]} ~ {df_15m.index[-1]}\n"
            f"  K线数量: {len(df_15m)} 根\n"
            f"  初始资金: {self.initial_capital} USDT\n"
            f"{'='*60}"
        )

        # 如果没有提供1H和5M数据，从15M重采样
        if df_1h is None:
            df_1h = resample_klines(df_15m, "1h")
        if df_5m is None:
            df_5m = df_15m  # 降级使用15M作为5M
        if df_4h is None:
            df_4h = resample_klines(df_15m, "4h")

        # 预热期：前100根K线用于指标计算，不交易
        warmup = 100

        for i in range(warmup, len(df_15m)):
            current_time = df_15m.index[i]
            current_price = df_15m["close"].iloc[i]

            # 获取当前时间点的历史数据切片
            df_15m_slice = df_15m.iloc[:i + 1]
            df_1h_slice = df_1h[df_1h.index <= current_time]
            df_4h_slice = df_4h[df_4h.index <= current_time]
            df_5m_slice = df_5m[df_5m.index <= current_time]

            # 构建当前时刻的多周期 K线字典（传入 ConsensusScorer）
            mtf_klines_now = {
                "1h": df_1h_slice,
                "4h": df_4h_slice,
            }

            # 记录权益曲线
            equity = self._calc_equity(current_price)
            self.equity_curve.append({
                "time": current_time,
                "equity": equity,
                "price": current_price,
            })

            # --- 更新持仓状态 ---
            if self.position is not None:
                self._update_position(current_price, df_5m_slice, current_time)

            # --- 寻找新入场信号（无持仓时）---
            if self.position is None:
                self._check_entry(
                    df_15m_slice, df_5m_slice, df_1h_slice, current_price, current_time,
                    mtf_klines=mtf_klines_now
                )

        # 强制平仓剩余持仓
        if self.position is not None:
            self._force_close(df_15m["close"].iloc[-1], df_15m.index[-1])

        return self._generate_report()

    def _check_entry(
        self,
        df_15m: pd.DataFrame,
        df_5m: pd.DataFrame,
        df_1h: pd.DataFrame,
        current_price: float,
        current_time: datetime,
        mtf_klines: Optional[Dict] = None,
    ):
        """
        检查入场条件。
        使用批量评分选最佳候选区域，并统一使用 EntryConfirmer 进行 5M 确认。
        """
        mtf_klines = mtf_klines or {}

        # --- Step 1: 趋势过滤 ---
        trend_cfg = self.config.get("strategy", {})
        ema_slow_period = trend_cfg.get("trend_ema_period", 200)
        ema_fast_period = trend_cfg.get("trend_ema_fast", 50)
        if len(df_1h) >= ema_slow_period:
            ema200 = df_1h["close"].ewm(span=ema_slow_period, adjust=False).mean().iloc[-1]
            ema50 = df_1h["close"].ewm(span=ema_fast_period, adjust=False).mean().iloc[-1]
            if not (current_price > ema200 and ema50 > ema200):
                return

        # --- Step 2: 区域识别，筛选候选区域 ---
        zones = self.zone_detector.detect(df_15m, current_price)
        support_zones = [z for z in zones if z["zone_type"] == "support"]
        if not support_zones:
            return

        zone_proximity = self.config.get("entry_signal", {}).get("zone_proximity_pct", 0.005)
        candidate_zones = [
            z for z in support_zones
            if z["price_start"] * (1 - zone_proximity) <= current_price
        ]
        if not candidate_zones:
            return

        # --- Step 3: 批量评分，将多周期数据传入 ConsensusScorer ---
        # 动态更新 mtf_klines，确保每次评分使用当前时刻的真实多周期数据
        self.consensus_scorer.mtf_klines = mtf_klines

        score_results = self.consensus_scorer.score_batch(candidate_zones, df_15m)

        # 记录所有候选区域的评分（用于统计口径修正）
        for sr in score_results:
            self.all_candidate_scores.append(sr["total_score"])
        passed_results = [r for r in score_results if r["passed_threshold"]]
        for pr in passed_results:
            self.passed_scores.append(pr["total_score"])

        if not passed_results:
            return

        # 选取评分最高且最靠近当前价格的候选区域
        passed_results.sort(key=lambda r: (
            -r["total_score"],
            abs(r["zone"]["mid_price"] - current_price)
        ))
        best_score_result = passed_results[0]
        nearest_support = best_score_result["zone"]

        # --- Step 4: 假突破检测 ---
        fakeout_result = self.fakeout_detector.check(best_score_result, df_5m)
        if fakeout_result.get("fakeout_type") == "true_breakdown":
            return

        # --- Step 5: 统一使用 EntryConfirmer 进行 5M 确认 ---
        confirmed, confirm_reason, signal_strength = self.entry_confirmer.confirm(
            df_5m, nearest_support, fakeout_result
        )
        if not confirmed:
            return

        # --- Step 6: 止损计算 ---
        sl_result = self.sl_calculator.calculate(current_price, nearest_support, df_15m, df_5m)
        if not sl_result["valid"]:
            return

        stop_price = sl_result["stop_price"]

        # --- Step 7: 仓位计算 ---
        pos_result = self.risk_manager.calculate_position(
            self.capital, current_price, stop_price, 0
        )
        if not pos_result["valid"]:
            return

        qty = pos_result["qty"]

        # 应用滑点和手续费
        actual_entry = current_price * (1 + self.slippage_pct)
        commission = actual_entry * qty * self.commission_rate
        self.capital -= commission

        # 获取压力区
        resistance_zones = [z for z in zones if z["zone_type"] == "resistance"]
        nearest_resistance = get_nearest_resistance(resistance_zones, actual_entry)

        # 初始化持仓
        self.position = self.tp_manager.init_position(
            actual_entry, stop_price, qty, nearest_resistance
        )
        self.position["entry_time"] = current_time
        self.position["score"] = best_score_result["total_score"]
        self.position["consensus_passed"] = self.use_consensus_score
        self.position["signal_strength"] = signal_strength

        logger.debug(
            f"[回测] 开仓 | 时间={current_time} | 价格={actual_entry:.2f} "
            f"| 止损={stop_price:.2f} | 数量={qty:.6f} "
            f"| 共识评分={best_score_result['total_score']:.1f}"
            f"| 信号强度={signal_strength:.2f}"
        )

    def _update_position(self, current_price: float, df_5m: pd.DataFrame, current_time: datetime):
        """更新持仓状态，处理止盈止损。"""
        updated = self.tp_manager.update(self.position, current_price, df_5m)
        actions = updated.pop("actions", [])
        self.position = updated

        for action in actions:
            action_type = action.get("type")
            if action_type in ("STOP_LOSS", "TRAILING_STOP"):
                # 全仓止损平仓：只加减净盈亏（capital 从未扣过保证金，pnl 已是净盈亏）
                qty = action.get("qty", 0)
                price = action.get("price", current_price)
                actual_price = price * (1 - self.slippage_pct)
                commission = actual_price * qty * self.commission_rate
                pnl = action.get("pnl", 0) - commission
                self.capital += pnl
                self._record_trade(action_type, current_time, pnl)

            elif action_type in ("TP1", "TP2"):
                # 部分止盈：只加减净盈亏
                qty = action.get("qty", 0)
                price = action.get("price", current_price)
                actual_price = price * (1 - self.slippage_pct)
                commission = actual_price * qty * self.commission_rate
                pnl = action.get("pnl", 0) - commission
                self.capital += pnl

        if self.position.get("phase") == "CLOSED":
            # 记录完整交易
            entry_time = self.position.get("entry_time", current_time)
            hold_duration = (current_time - entry_time).total_seconds() / 60  # 分钟
            total_pnl = self.position.get("realized_pnl", 0)
            self.trades.append({
                "entry_time": entry_time,
                "exit_time": current_time,
                "entry_price": self.position["entry_price"],
                "exit_price": current_price,
                "qty": self.position["initial_qty"],
                "pnl": total_pnl,
                "hold_minutes": hold_duration,
                "score": self.position.get("score", 0),
                "consensus_used": self.use_consensus_score,
                "exit_reason": actions[-1].get("type") if actions else "UNKNOWN",
            })
            self.position = None

    def _record_trade(self, exit_type: str, exit_time: datetime, pnl: float):
        """记录部分平仓事件（用于内部追踪）。"""
        pass  # 完整交易在 _update_position 中记录

    def _force_close(self, price: float, time: datetime):
        """强制平仓（回测结束时）。"""
        if self.position and self.position.get("remaining_qty", 0) > 0:
            qty = self.position["remaining_qty"]
            pnl = (price - self.position["entry_price"]) * qty
            self.trades.append({
                "entry_time": self.position.get("entry_time", time),
                "exit_time": time,
                "entry_price": self.position["entry_price"],
                "exit_price": price,
                "qty": qty,
                "pnl": pnl,
                "hold_minutes": 0,
                "score": self.position.get("score", 0),
                "consensus_used": self.use_consensus_score,
                "exit_reason": "FORCE_CLOSE",
            })
            self.position = None

    def _calc_equity(self, current_price: float) -> float:
        """计算当前总权益（含未实现盈亏）。"""
        if self.position:
            unrealized = (current_price - self.position["entry_price"]) * self.position.get("remaining_qty", 0)
            return self.capital + unrealized
        return self.capital

    def _generate_report(self) -> Dict:
        """生成回测绩效报告。"""
        if not self.trades:
            return {
                "use_consensus_score": self.use_consensus_score,
                "total_trades": 0,
                "win_count": 0,
                "loss_count": 0,
                "win_rate_pct": 0,
                "avg_win_usdt": 0,
                "avg_loss_usdt": 0,
                "avg_rr_ratio": 0,
                "profit_factor": 0,
                "total_return_pct": 0,
                "max_drawdown_pct": 0,
                "avg_hold_minutes": 0,
                "final_capital": round(self.capital, 4),
                "initial_capital": self.initial_capital,
                "avg_consensus_score": 0,
                "trades": [],
                "equity_curve": [],
                "message": "无交易记录",
            }

        df_trades = pd.DataFrame(self.trades)
        df_equity = pd.DataFrame(self.equity_curve)

        wins = df_trades[df_trades["pnl"] > 0]
        losses = df_trades[df_trades["pnl"] <= 0]

        win_rate = len(wins) / len(df_trades) * 100 if len(df_trades) > 0 else 0
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses["pnl"].mean()) if len(losses) > 0 else 0
        avg_rr = avg_win / avg_loss if avg_loss > 0 else 0

        total_profit = wins["pnl"].sum() if len(wins) > 0 else 0
        total_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100

        # 最大回撤
        if len(df_equity) > 0:
            equity_series = df_equity["equity"]
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0

        avg_hold = df_trades["hold_minutes"].mean()

        report = {
            "use_consensus_score": self.use_consensus_score,
            "total_trades": len(df_trades),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "avg_win_usdt": round(avg_win, 4),
            "avg_loss_usdt": round(avg_loss, 4),
            "avg_rr_ratio": round(avg_rr, 3),
            "profit_factor": round(profit_factor, 3),
            "total_return_pct": round(total_return, 4),
            "max_drawdown_pct": round(max_drawdown, 4),
            "avg_hold_minutes": round(avg_hold, 1),
            "final_capital": round(self.capital, 4),
            "initial_capital": self.initial_capital,
            # 评分统计口径修正：
            # - avg_consensus_score_all: 所有候选区域的平均分（无论是否开仓）
            # - avg_consensus_score_passed: 通过阈值的候选区域平均分
            # - avg_consensus_score_traded: 实际开仓交易的平均分
            "avg_consensus_score": round(df_trades["score"].mean(), 2) if "score" in df_trades.columns else 0,
            "avg_consensus_score_all_candidates": round(float(np.mean(self.all_candidate_scores)), 2) if self.all_candidate_scores else 0,
            "avg_consensus_score_passed": round(float(np.mean(self.passed_scores)), 2) if self.passed_scores else 0,
            "total_candidates_evaluated": len(self.all_candidate_scores),
            "total_candidates_passed": len(self.passed_scores),
            "pass_rate_pct": round(len(self.passed_scores) / len(self.all_candidate_scores) * 100, 2) if self.all_candidate_scores else 0,
            "trades": df_trades.to_dict("records"),
            "equity_curve": df_equity.to_dict("records") if len(df_equity) > 0 else [],
        }

        return report


# ---------------------------------------------------------------------------
# 对比报告生成
# ---------------------------------------------------------------------------

def compare_and_report(report_a: Dict, report_b: Dict) -> str:
    """
    生成有/无共识评分过滤的对比报告。

    :param report_a: 无共识评分的回测结果
    :param report_b: 有共识评分的回测结果
    :return:         格式化的对比报告字符串
    """
    sep = "=" * 75
    lines = [
        sep,
        "  Binance BTCUSDT 永续合约 — 共识强度评分模块 回测对比报告",
        sep,
        f"  {'指标':<25} {'原策略（无评分）':>18} {'新策略（有评分）':>18} {'变化':>12}",
        f"  {'-'*25} {'-'*18} {'-'*18} {'-'*12}",
    ]

    def fmt_change(a, b, higher_is_better=True, is_pct=False):
        if a == 0:
            return "N/A"
        change = b - a
        suffix = "%" if is_pct else ""
        sign = "+" if change >= 0 else ""
        indicator = "↑" if (change > 0) == higher_is_better else "↓"
        return f"{sign}{change:.2f}{suffix} {indicator}"

    metrics = [
        ("交易次数",          "total_trades",        False, False),
        ("胜率 (%)",          "win_rate_pct",         True,  True),
        ("平均盈亏比",        "avg_rr_ratio",         True,  False),
        ("利润因子",          "profit_factor",        True,  False),
        ("总收益率 (%)",      "total_return_pct",     True,  True),
        ("最大回撤 (%)",      "max_drawdown_pct",     False, True),
        ("平均持仓(分钟)",    "avg_hold_minutes",     None,  False),
        ("平均共识评分",      "avg_consensus_score",  True,  False),
        ("最终资金 (USDT)",   "final_capital",        True,  False),
    ]

    for label, key, higher_better, is_pct in metrics:
        val_a = report_a.get(key, 0)
        val_b = report_b.get(key, 0)
        change_str = fmt_change(val_a, val_b, higher_better, is_pct) if higher_better is not None else "-"
        lines.append(
            f"  {label:<25} {str(val_a):>18} {str(val_b):>18} {change_str:>12}"
        )

    lines.extend([
        sep,
        "  分析结论：",
    ])

    # 自动生成结论
    conclusions = []
    if report_b.get("total_trades", 0) < report_a.get("total_trades", 0):
        reduction = report_a["total_trades"] - report_b["total_trades"]
        conclusions.append(
            f"  ✓ 共识评分过滤减少了 {reduction} 笔低质量交易"
            f"（{reduction/report_a['total_trades']*100:.1f}%）"
        )
    if report_b.get("win_rate_pct", 0) > report_a.get("win_rate_pct", 0):
        conclusions.append(
            f"  ✓ 胜率提升 {report_b['win_rate_pct']-report_a['win_rate_pct']:.2f}%"
        )
    if report_b.get("avg_rr_ratio", 0) > report_a.get("avg_rr_ratio", 0):
        conclusions.append(
            f"  ✓ 平均盈亏比提升 {report_b['avg_rr_ratio']-report_a['avg_rr_ratio']:.3f}"
        )
    dd_a = report_a.get("max_drawdown_pct", 0)
    dd_b = report_b.get("max_drawdown_pct", 0)
    if dd_b > dd_a:  # 回撤值为负数，dd_b > dd_a 意味着 b 的回撤更小
        conclusions.append(
            f"  ✓ 最大回撤降低 {abs(dd_b - dd_a):.2f}%"
        )
    if not conclusions:
        conclusions.append("  - 当前参数下，共识评分过滤效果需进一步调优")

    lines.extend(conclusions)
    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主程序入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BTC顺势回踩策略回测")
    parser.add_argument("--data", type=str, default="data/BTCUSDT_15m.csv", help="15M K线数据路径")
    parser.add_argument("--data-1h", type=str, default=None, help="1H K线数据路径（可选）")
    parser.add_argument("--data-5m", type=str, default=None, help="5M K线数据路径（可选）")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["with_score", "without_score", "compare"],
        default="compare",
        help="回测模式",
    )
    parser.add_argument("--output", type=str, default="backtest_report.json", help="结果输出文件")
    args = parser.parse_args()

    # 初始化日志
    setup_logger(CONFIG)

    # 加载数据
    df_15m = load_klines(args.data, "15m")
    df_1h = load_klines(args.data_1h, "1h") if args.data_1h else None
    df_5m = load_klines(args.data_5m, "5m") if args.data_5m else None

    results = {}

    if args.mode in ("without_score", "compare"):
        engine_a = BacktestEngine(CONFIG, use_consensus_score=False)
        report_a = engine_a.run(df_15m, df_5m, df_1h)
        results["without_score"] = report_a
        logger.info(f"[回测A] 完成 | 交易次数={report_a.get('total_trades', 0)}")

    if args.mode in ("with_score", "compare"):
        engine_b = BacktestEngine(CONFIG, use_consensus_score=True)
        report_b = engine_b.run(df_15m, df_5m, df_1h)
        results["with_score"] = report_b
        logger.info(f"[回测B] 完成 | 交易次数={report_b.get('total_trades', 0)}")

    if args.mode == "compare" and "without_score" in results and "with_score" in results:
        comparison = compare_and_report(results["without_score"], results["with_score"])
        print(comparison)
        results["comparison_text"] = comparison

    # 保存结果
    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        # 移除 trades 和 equity_curve 中的 datetime 对象（JSON不支持）
        def serialize(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return str(obj)
            return obj

        json.dump(results, f, ensure_ascii=False, indent=2, default=serialize)
    logger.info(f"回测结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
