"""
entry_signal.py — 入场信号生成模块

职责：
  - 执行完整的交易流程判断（趋势过滤 -> 区域识别 -> 共识评分 -> 5M确认 -> 入场）
  - 集成 ConsensusScorer 的评分过滤逻辑
  - 集成 FakeoutDetector 的假突破联动逻辑
  - 不负责下单执行，只负责"是否应该开仓"和"以什么参数开仓"

完整交易流程：
  1H 趋势过滤
    -> 15M 支撑区识别（support_zone.py）
    -> 市场共识强度评分（consensus_score.py）
    -> [评分通过] -> 5M 入场确认
    -> [5M确认通过] -> 生成入场信号
    -> risk_manager.py 计算仓位
    -> exit_manager.py 计算止损/止盈
    -> 返回完整的开仓指令
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from support_zone import SupportZoneDetector, get_nearest_support, get_nearest_resistance
from consensus_score import ConsensusScorer, FakeoutDetector
from exit_manager import StopLossCalculator, TakeProfitManager
from risk_manager import RiskManager, BinancePrecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 趋势过滤
# ---------------------------------------------------------------------------

class TrendFilter:
    """
    1H 趋势过滤器。

    判断当前市场是否处于上升趋势（允许做多）。
    逻辑：
      - 价格在 EMA200 上方 -> 上升趋势
      - EMA50 > EMA200 -> 趋势确认
      - 两者同时满足 -> 允许做多
    """

    def __init__(self, config: Dict):
        strat = config.get("strategy", {})
        self.ema_slow = strat.get("trend_ema_period", 200)
        self.ema_fast = strat.get("trend_ema_fast", 50)

    def is_uptrend(self, df_1h: pd.DataFrame) -> Tuple[bool, str]:
        """
        判断1H周期是否为上升趋势。

        :param df_1h: 1H K线数据
        :return:      (是否上升趋势, 原因说明)
        """
        if len(df_1h) < self.ema_slow:
            return False, f"1H数据不足（需要{self.ema_slow}根，当前{len(df_1h)}根）"

        close = df_1h["close"]
        ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1]
        ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean().iloc[-1]
        current_price = close.iloc[-1]

        above_ema_slow = current_price > ema_slow
        ema_aligned = ema_fast > ema_slow

        if above_ema_slow and ema_aligned:
            reason = (
                f"上升趋势确认：价格{current_price:.2f} > EMA{self.ema_slow}({ema_slow:.2f})"
                f"，EMA{self.ema_fast}({ema_fast:.2f}) > EMA{self.ema_slow}({ema_slow:.2f})"
            )
            return True, reason
        elif above_ema_slow and not ema_aligned:
            reason = (
                f"弱上升趋势：价格在EMA{self.ema_slow}上方，但EMA{self.ema_fast}未穿越EMA{self.ema_slow}"
            )
            return False, reason
        else:
            reason = (
                f"非上升趋势：价格{current_price:.2f} <= EMA{self.ema_slow}({ema_slow:.2f})"
            )
            return False, reason


# ---------------------------------------------------------------------------
# 5M 入场确认
# ---------------------------------------------------------------------------

class EntryConfirmer:
    """
    5M 周期入场确认器。

    确认条件（可配置）：
      1. 当前价格在支撑区内或靠近支撑区
      2. 5M K线收盘为阳线（或满足看涨形态）
      3. 实体比例满足最小要求（过滤十字星）
      4. 成交量确认（可选）
      5. 假突破信号增强（可选）
    """

    def __init__(self, config: Dict):
        entry_cfg = config.get("entry_signal", {})
        self.zone_proximity = entry_cfg.get("zone_proximity_pct", 0.005)
        self.confirmation_mode = entry_cfg.get("confirmation_mode", "candle_close")
        self.min_body_ratio = entry_cfg.get("min_body_ratio", 0.3)
        self.vol_confirm_enabled = entry_cfg.get("volume_confirmation_enabled", True)
        self.vol_confirm_mult = entry_cfg.get("volume_confirmation_multiplier", 1.2)
        self.fakeout_relax = entry_cfg.get("fakeout_relax_body_ratio", True)

    def confirm(
        self,
        df_5m: pd.DataFrame,
        zone: Dict,
        fakeout_result: Optional[Dict] = None,
    ) -> Tuple[bool, str, float]:
        """
        执行5M入场确认。

        :param df_5m:           5M K线数据
        :param zone:            候选支撑区域
        :param fakeout_result:  假突破检测结果（可选）
        :return:                (是否确认, 原因说明, 信号强度0~1)
        """
        if len(df_5m) < 5:
            return False, "5M数据不足", 0.0

        last = df_5m.iloc[-1]
        open_c = last["open"]
        close_c = last["close"]
        high_c = last["high"]
        low_c = last["low"]
        volume = last["volume"]

        reasons = []
        signal_strength = 0.0

        # --- 条件1：价格在支撑区内或靠近支撑区 ---
        zone_mid = zone["mid_price"]
        price_dist_pct = abs(close_c - zone_mid) / zone_mid
        in_zone = (
            zone["price_start"] * (1 - self.zone_proximity) <= close_c <=
            zone["price_end"] * (1 + self.zone_proximity)
        )
        if not in_zone:
            return False, f"价格{close_c:.2f}不在支撑区[{zone['price_start']:.2f}~{zone['price_end']:.2f}]内", 0.0
        reasons.append(f"价格在支撑区内（距区域中心{price_dist_pct:.2%}）")
        signal_strength += 0.3

        # --- 条件2：K线方向确认 ---
        is_bullish = close_c > open_c
        if not is_bullish:
            return False, f"5M K线为阴线（收盘{close_c:.2f} < 开盘{open_c:.2f}），不确认入场", 0.0
        reasons.append(f"5M阳线确认（收盘{close_c:.2f} > 开盘{open_c:.2f}）")
        signal_strength += 0.2

        # --- 条件3：实体比例检查 ---
        total_range = high_c - low_c
        body = abs(close_c - open_c)
        body_ratio = body / total_range if total_range > 0 else 0

        # 假突破场景下放宽实体要求
        effective_min_body = self.min_body_ratio
        if fakeout_result and fakeout_result.get("fakeout_detected") and self.fakeout_relax:
            effective_min_body = self.min_body_ratio * 0.6
            reasons.append(f"假突破场景，实体要求放宽至{effective_min_body:.2f}")

        if body_ratio < effective_min_body:
            return False, (
                f"K线实体比例过小（{body_ratio:.2f} < {effective_min_body:.2f}），"
                f"可能是十字星，不确认入场"
            ), 0.0
        reasons.append(f"实体比例{body_ratio:.2f}满足要求（>{effective_min_body:.2f}）")
        signal_strength += 0.2

        # --- 条件4：成交量确认（可选）---
        if self.vol_confirm_enabled:
            avg_vol = df_5m["volume"].iloc[-20:].mean()
            vol_ratio = volume / avg_vol if avg_vol > 0 else 0
            if vol_ratio >= self.vol_confirm_mult:
                reasons.append(f"成交量确认（{vol_ratio:.1f}倍均量）")
                signal_strength += 0.2
            else:
                reasons.append(f"成交量偏低（{vol_ratio:.1f}倍均量，要求>{self.vol_confirm_mult}倍）")
                signal_strength += 0.05

        # --- 条件5：假突破信号增强 ---
        if fakeout_result and fakeout_result.get("fakeout_detected"):
            boost = fakeout_result.get("signal_boost", 1.0)
            signal_strength *= boost
            reasons.append(f"假突破信号增强×{boost}：{fakeout_result.get('reason', '')}")

        signal_strength = min(signal_strength, 1.0)
        reason_str = " | ".join(reasons)
        return True, reason_str, signal_strength


# ---------------------------------------------------------------------------
# 主入场信号生成器
# ---------------------------------------------------------------------------

class EntrySignalGenerator:
    """
    完整交易流程的入场信号生成器。

    集成所有子模块，执行完整的信号生成流程。
    """

    def __init__(
        self,
        config: Dict,
        mtf_klines: Optional[Dict[str, pd.DataFrame]] = None,
        exchange_info: Optional[Dict] = None,
    ):
        self.config = config
        self.trend_filter = TrendFilter(config)
        self.zone_detector = SupportZoneDetector(config)
        self.consensus_scorer = ConsensusScorer(config, mtf_klines)
        self.fakeout_detector = FakeoutDetector(config)
        self.entry_confirmer = EntryConfirmer(config)
        self.sl_calculator = StopLossCalculator(config)
        self.tp_manager = TakeProfitManager(config)
        precision = BinancePrecision(exchange_info) if exchange_info else BinancePrecision()
        self.risk_manager = RiskManager(config, precision)

    def generate(
        self,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_5m: pd.DataFrame,
        account_balance: float,
        available_margin: float,
        current_open_positions: int = 0,
    ) -> Dict:
        """
        执行完整的入场信号生成流程。

        :param df_1h:                   1H K线数据
        :param df_15m:                  15M K线数据
        :param df_5m:                   5M K线数据
        :param account_balance:         账户净值（USDT）
        :param available_margin:        可用保证金（USDT）
        :param current_open_positions:  当前持仓数量
        :return:                        完整的信号结果字典
        """
        current_price = df_5m["close"].iloc[-1]

        result = {
            "signal": False,
            "direction": "long",
            "current_price": current_price,
            "entry_price": None,
            "stop_price": None,
            "tp1_price": None,
            "tp2_price": None,
            "qty": None,
            "notional_usdt": None,
            "risk_usdt": None,
            "consensus_score": None,
            "signal_strength": 0.0,
            "rejection_stage": None,
            "rejection_reason": None,
            "log_details": {},
        }

        # ===================================================================
        # Step 1: 1H 趋势过滤
        # ===================================================================
        is_uptrend, trend_reason = self.trend_filter.is_uptrend(df_1h)
        result["log_details"]["trend"] = trend_reason

        if not is_uptrend:
            result["rejection_stage"] = "trend_filter"
            result["rejection_reason"] = f"趋势过滤未通过：{trend_reason}"
            logger.info(f"[EntrySignal] 趋势过滤 ✗ | {trend_reason}")
            return result

        logger.info(f"[EntrySignal] 趋势过滤 ✓ | {trend_reason}")

        # ===================================================================
        # Step 2: 15M 支撑区识别
        # ===================================================================
        zones = self.zone_detector.detect(df_15m, current_price)
        support_zones = [z for z in zones if z["zone_type"] == "support"]

        result["log_details"]["zones_found"] = len(support_zones)

        if not support_zones:
            result["rejection_stage"] = "zone_detection"
            result["rejection_reason"] = "15M未识别到有效支撑区"
            logger.info(f"[EntrySignal] 区域识别 ✗ | 未找到支撑区")
            return result

        # 过滤出价格附近的候选区域（在当前价格下方且在容差范围内）
        zone_proximity = self.config.get("entry_signal", {}).get("zone_proximity_pct", 0.005)
        candidate_zones = [
            z for z in support_zones
            if z["price_start"] * (1 - zone_proximity) <= current_price
        ]
        if not candidate_zones:
            result["rejection_stage"] = "zone_detection"
            result["rejection_reason"] = "当前价格下方无候选支撑区"
            return result

        logger.info(
            f"[EntrySignal] 区域识别 ✓ | 共{len(support_zones)}个支撑区"
            f" | 候选区域{len(candidate_zones)}个"
        )

        # ===================================================================
        # Step 3: 市场共识强度评分（批量评分所有候选区域，选出最佳）
        # ===================================================================
        score_results = self.consensus_scorer.score_batch(candidate_zones, df_15m)

        # 记录所有候选区域的评分结果（用于日志和统计）
        result["log_details"]["all_zone_scores"] = [
            {"zone": r["zone"], "score": r["total_score"], "passed": r["passed_threshold"]}
            for r in score_results
        ]

        # 只保留通过阈值的区域，按评分降序排序
        passed_results = [r for r in score_results if r["passed_threshold"]]

        if not passed_results:
            # 没有任何区域通过阈值：记录最高分区域供参考
            best_score_result = max(score_results, key=lambda r: r["total_score"])
            result["consensus_score"] = best_score_result
            result["log_details"]["consensus_score"] = best_score_result
            result["rejection_stage"] = "consensus_score"
            result["rejection_reason"] = (
                f"共识评分不足（最高分区域得分={best_score_result['total_score']:.1f} < "
                f"阈值{best_score_result['threshold_value']}），共{len(candidate_zones)}个候选区域均未通过"
            )
            logger.info(
                f"[EntrySignal] 共识评分 ✗ | 候选区域{len(candidate_zones)}个"
                f" | 最高分={best_score_result['total_score']:.1f}"
                f" | 阈值={best_score_result['threshold_value']}"
            )
            return result

        # 选取评分最高且最靠近当前价格的候选区域
        # 排序规则：优先选评分最高的；评分相同时选最近的
        passed_results.sort(key=lambda r: (
            -r["total_score"],
            abs(r["zone"]["mid_price"] - current_price)
        ))
        score_result = passed_results[0]
        nearest_support = score_result["zone"]

        result["consensus_score"] = score_result
        result["log_details"]["consensus_score"] = score_result

        self._log_consensus_score(score_result)

        logger.info(
            f"[EntrySignal] 共识评分 ✓ | 候选区域{len(candidate_zones)}个"
            f" | 通过{len(passed_results)}个"
            f" | 最佳区域得分={score_result['total_score']:.1f}"
            f" | 阈值={score_result['threshold_value']}"
        )

        # ===================================================================
        # Step 4: 假突破检测（联动共识评分）
        # ===================================================================
        fakeout_result = self.fakeout_detector.check(score_result, df_5m)
        result["log_details"]["fakeout"] = fakeout_result

        if fakeout_result.get("fakeout_type") == "true_breakdown":
            result["rejection_stage"] = "fakeout_check"
            result["rejection_reason"] = fakeout_result["reason"]
            logger.info(f"[EntrySignal] 假突破检测 ✗ | {fakeout_result['reason']}")
            return result

        if fakeout_result.get("fakeout_detected"):
            logger.info(f"[EntrySignal] 假突破检测 ✓ | {fakeout_result['reason']}")

        # ===================================================================
        # Step 5: 5M 入场确认
        # ===================================================================
        confirmed, confirm_reason, signal_strength = self.entry_confirmer.confirm(
            df_5m, nearest_support, fakeout_result
        )
        result["signal_strength"] = signal_strength
        result["log_details"]["entry_confirm"] = confirm_reason

        if not confirmed:
            result["rejection_stage"] = "entry_confirm"
            result["rejection_reason"] = f"5M入场确认未通过：{confirm_reason}"
            logger.info(f"[EntrySignal] 5M确认 ✗ | {confirm_reason}")
            return result

        logger.info(f"[EntrySignal] 5M确认 ✓ | {confirm_reason} | 信号强度={signal_strength:.2f}")

        # ===================================================================
        # Step 6: 计算止损
        # ===================================================================
        entry_price = current_price  # 使用当前市价作为入场价（实盘时使用限价单）
        sl_result = self.sl_calculator.calculate(entry_price, nearest_support, df_15m, df_5m)
        result["log_details"]["stop_loss"] = sl_result

        if not sl_result["valid"]:
            result["rejection_stage"] = "stop_loss_calc"
            result["rejection_reason"] = sl_result.get("rejection_reason", "止损计算失败")
            logger.info(f"[EntrySignal] 止损计算 ✗ | {sl_result.get('rejection_reason')}")
            return result

        stop_price = sl_result["stop_price"]
        logger.info(
            f"[EntrySignal] 止损计算 ✓ | 止损价={stop_price:.2f} "
            f"| 方法={sl_result['method']} "
            f"| 距离={sl_result['stop_distance_pct']:.2%}"
        )

        # ===================================================================
        # Step 7: 计算仓位（风险管理）
        # ===================================================================
        pos_result = self.risk_manager.calculate_position(
            account_balance, entry_price, stop_price, current_open_positions
        )
        result["log_details"]["position"] = pos_result

        if not pos_result["valid"]:
            result["rejection_stage"] = "position_calc"
            result["rejection_reason"] = pos_result.get("rejection_reason", "仓位计算失败")
            logger.info(f"[EntrySignal] 仓位计算 ✗ | {pos_result.get('rejection_reason')}")
            return result

        # 检查保证金是否充足
        margin_ok, margin_reason = self.risk_manager.check_margin_sufficient(
            pos_result["qty"], entry_price, available_margin
        )
        if not margin_ok:
            result["rejection_stage"] = "margin_check"
            result["rejection_reason"] = margin_reason
            logger.info(f"[EntrySignal] 保证金检查 ✗ | {margin_reason}")
            return result

        qty = pos_result["qty"]
        logger.info(
            f"[EntrySignal] 仓位计算 ✓ | 数量={qty:.6f} BTC "
            f"| 名义={pos_result['notional_usdt']:.2f} USDT "
            f"| 风险={pos_result['risk_usdt']:.2f} USDT ({pos_result['risk_pct_actual']:.3f}%)"
        )

        # ===================================================================
        # Step 8: 计算止盈目标
        # ===================================================================
        resistance_zones = [z for z in zones if z["zone_type"] == "resistance"]
        nearest_resistance = get_nearest_resistance(resistance_zones, entry_price)

        risk_per_unit = entry_price - stop_price
        tp1_price = entry_price + risk_per_unit * self.config["take_profit"]["tp1"]["target_r"]
        tp2_price = self.tp_manager._calc_tp2_price(entry_price, risk_per_unit, nearest_resistance)

        # ===================================================================
        # 生成最终信号
        # ===================================================================
        result.update({
            "signal": True,
            "direction": "long",
            "entry_price": round(entry_price, 2),
            "stop_price": stop_price,
            "tp1_price": round(tp1_price, 2),
            "tp2_price": round(tp2_price, 2),
            "qty": qty,
            "notional_usdt": pos_result["notional_usdt"],
            "risk_usdt": pos_result["risk_usdt"],
            "risk_pct": pos_result["risk_pct_actual"],
            "zone": nearest_support,
            "resistance_zone": nearest_resistance,
        })

        logger.info(
            f"[EntrySignal] ✅ 入场信号生成 | 方向=LONG "
            f"| 入场={entry_price:.2f} | 止损={stop_price:.2f} "
            f"| TP1={tp1_price:.2f} | TP2={tp2_price:.2f} "
            f"| 数量={qty:.6f} BTC | 共识评分={score_result['total_score']:.1f}"
        )

        return result

    def _log_consensus_score(self, score_result: Dict):
        """输出共识评分的详细日志。"""
        zone = score_result.get("zone", {})
        logger.info(
            f"\n{'='*60}\n"
            f"[共识强度评分] 支撑区 [{zone.get('price_start', 0):.2f} ~ {zone.get('price_end', 0):.2f}]\n"
            f"  总分：{score_result['total_score']:.1f} / 100\n"
            f"  是否通过：{'✓ 通过' if score_result['passed_threshold'] else '✗ 未通过'}"
            f"（阈值={score_result['threshold_value']}）\n"
            f"  分项明细：\n"
        )
        for dim, detail in score_result.get("details", {}).items():
            logger.info(
                f"    [{dim:15s}] 原始分={detail['raw']:5.1f} "
                f"权重={detail['weight']:4.1f}% "
                f"加权分={detail['weighted']:5.2f}"
            )
            for r in detail.get("reasons", []):
                logger.info(f"      ✓ {r}")
            for p in detail.get("penalties", []):
                logger.info(f"      ✗ {p}")

        if score_result.get("penalty_reasons"):
            logger.info(f"  减分原因：")
            for p in score_result["penalty_reasons"]:
                logger.info(f"    - {p}")

        logger.info(
            f"  最终决定：{'允许进入5M确认' if score_result['allow_entry_check'] else '放弃，不进入5M确认'}\n"
            f"{'='*60}"
        )
