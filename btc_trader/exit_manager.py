"""
exit_manager.py — 退出机制管理模块

职责：
  - 计算结构化止损价格（支撑区下沿 + 缓冲，或ATR动态缓冲）
  - 管理分批止盈逻辑（TP1 / TP2 / 剩余仓位跟踪止盈）
  - 管理保本移损逻辑
  - 在回测和实盘中统一使用，确保行为一致

退出流程：
  开仓后
    -> 盈利达到1R  -> 止损移至保本 + 减仓35%（TP1）
    -> 盈利达到压力区 -> 再减仓35%（TP2）
    -> 剩余30%仓位 -> 结构/EMA跟踪止盈，直到信号失效

止损逻辑优先级：
  1. 结构失效止损（优先）：支撑区下沿 + ATR/固定缓冲
  2. ATR止损（备选）
  3. 固定百分比止损（最后备选）
  4. 最大/最小止损距离保护（任何模式下均生效）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 止损计算
# ---------------------------------------------------------------------------

class StopLossCalculator:
    """
    结构化止损计算器。

    支持三种止损模式：
      - structure: 结构失效止损（优先推荐）
      - atr:       ATR动态止损
      - fixed_pct: 固定百分比止损
    """

    def __init__(self, config: Dict):
        self.cfg = config.get("stop_loss", {})
        self.mode = self.cfg.get("mode", "structure")
        self.max_stop_pct = self.cfg.get("max_stop_distance_pct", 0.03)
        self.min_stop_pct = self.cfg.get("min_stop_distance_pct", 0.003)

    def calculate(
        self,
        entry_price: float,
        zone: Dict,
        df_15m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        计算止损价格。

        :param entry_price: 实际入场价格
        :param zone:        触发入场的支撑区域
        :param df_15m:      15M K线数据（用于ATR计算）
        :param df_5m:       5M K线数据（用于确认K线低点止损）
        :return:            止损计算结果字典
        """
        mode = self.mode
        stop_price = None
        method = ""
        reasons = []

        if mode == "structure":
            stop_price, method, reasons = self._calc_structure_stop(
                entry_price, zone, df_15m, df_5m
            )
        elif mode == "atr":
            stop_price, method, reasons = self._calc_atr_stop(entry_price, df_15m)
        elif mode == "fixed_pct":
            stop_price, method, reasons = self._calc_fixed_pct_stop(entry_price)
        else:
            stop_price, method, reasons = self._calc_structure_stop(
                entry_price, zone, df_15m, df_5m
            )

        # --- 最大/最小止损距离保护 ---
        if stop_price is not None:
            stop_distance_pct = abs(entry_price - stop_price) / entry_price
            if stop_distance_pct > self.max_stop_pct:
                # 止损太远，拒绝交易
                return {
                    "stop_price": None,
                    "stop_distance_pct": stop_distance_pct,
                    "valid": False,
                    "method": method,
                    "reasons": reasons,
                    "rejection_reason": (
                        f"止损距离{stop_distance_pct:.2%}超过最大允许值"
                        f"{self.max_stop_pct:.2%}，放弃本次交易"
                    ),
                }
            if stop_distance_pct < self.min_stop_pct:
                # 止损太近，调整到最小距离
                stop_price = entry_price * (1 - self.min_stop_pct)
                reasons.append(
                    f"止损距离过小，调整至最小止损距离{self.min_stop_pct:.2%}"
                )
                stop_distance_pct = self.min_stop_pct

            return {
                "stop_price": round(stop_price, 2),
                "stop_distance_pct": round(stop_distance_pct, 6),
                "risk_per_unit": entry_price - stop_price,  # 1R对应的价格差
                "valid": True,
                "method": method,
                "reasons": reasons,
                "rejection_reason": None,
            }

        return {
            "stop_price": None,
            "valid": False,
            "method": method,
            "reasons": reasons,
            "rejection_reason": "无法计算止损价格",
        }

    def _calc_structure_stop(
        self,
        entry_price: float,
        zone: Dict,
        df_15m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame],
    ) -> Tuple[Optional[float], str, List[str]]:
        """结构失效止损：支撑区下沿 + 缓冲。"""
        struct_cfg = self.cfg.get("structure", {})
        placement = struct_cfg.get("placement", "zone_bottom")
        buffer_mode = struct_cfg.get("buffer_mode", "atr")

        reasons = []

        # 确定止损基准价
        if placement == "zone_bottom":
            base_price = zone["price_start"]  # 支撑区下沿
            reasons.append(f"止损基准：支撑区下沿 {base_price:.2f}")
        elif placement == "entry_candle_low" and df_5m is not None and len(df_5m) > 0:
            # 5M确认K线的最低点
            base_price = df_5m["low"].iloc[-1]
            reasons.append(f"止损基准：5M确认K线低点 {base_price:.2f}")
        else:
            base_price = zone["price_start"]
            reasons.append(f"止损基准：支撑区下沿 {base_price:.2f}（默认）")

        # 计算缓冲
        if buffer_mode == "atr":
            atr_period = struct_cfg.get("atr_period", 14)
            atr_mult = struct_cfg.get("atr_multiplier", 0.5)
            atr_tf = struct_cfg.get("atr_timeframe", "15m")
            atr_val = self._calc_atr(df_15m, atr_period)
            buffer = atr_val * atr_mult
            method = f"结构止损（ATR{atr_period}缓冲×{atr_mult}）"
            reasons.append(
                f"ATR({atr_period})={atr_val:.2f}，缓冲={buffer:.2f}（×{atr_mult}）"
            )
        else:
            buffer_pct = struct_cfg.get("buffer_pct", 0.003)
            buffer = base_price * buffer_pct
            method = f"结构止损（固定{buffer_pct:.2%}缓冲）"
            reasons.append(f"固定缓冲={buffer:.2f}（{buffer_pct:.2%}）")

        stop_price = base_price - buffer
        reasons.append(f"最终止损价：{stop_price:.2f}")
        return stop_price, method, reasons

    def _calc_atr_stop(
        self, entry_price: float, df: pd.DataFrame
    ) -> Tuple[Optional[float], str, List[str]]:
        """ATR动态止损。"""
        atr_cfg = self.cfg.get("atr", {})
        period = atr_cfg.get("period", 14)
        mult = atr_cfg.get("multiplier", 2.0)
        atr_val = self._calc_atr(df, period)
        stop_price = entry_price - atr_val * mult
        reasons = [
            f"ATR({period})={atr_val:.2f}，止损=入场价-{mult}×ATR={stop_price:.2f}"
        ]
        return stop_price, f"ATR止损（×{mult}）", reasons

    def _calc_fixed_pct_stop(
        self, entry_price: float
    ) -> Tuple[Optional[float], str, List[str]]:
        """固定百分比止损。"""
        pct = self.cfg.get("fixed_pct", {}).get("pct", 0.015)
        stop_price = entry_price * (1 - pct)
        reasons = [f"固定止损={pct:.2%}，止损价={stop_price:.2f}"]
        return stop_price, f"固定百分比止损（{pct:.2%}）", reasons

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR（真实波动幅度均值）。"""
        if len(df) < period + 1:
            # 数据不足时，用最近K线的振幅均值代替
            return (df["high"] - df["low"]).mean()
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]


# ---------------------------------------------------------------------------
# 止盈管理
# ---------------------------------------------------------------------------

class TakeProfitManager:
    """
    分批止盈 + 跟踪止盈管理器。

    状态机：
      WAITING_TP1  -> 等待TP1触发（盈利达到1.2R）
      WAITING_TP2  -> TP1已触发，等待TP2触发（靠近压力区）
      TRAILING     -> TP2已触发，剩余仓位跟踪止盈
      CLOSED       -> 全部平仓
    """

    # 并行退出模式说明：
    # - WAITING_TP1 : 等待 TP1 触发（盈利达到 tp1_r）
    # - ACTIVE      : TP1 已触发，TP2 与跟踪止盈并行监控，哪个先到触发哪个
    # - CLOSED      : 全部平仓
    STATES = ["WAITING_TP1", "ACTIVE", "CLOSED"]

    def __init__(self, config: Dict):
        tp_cfg = config.get("take_profit", {})
        self.partial_enabled = tp_cfg.get("partial_tp_enabled", True)

        # TP1配置
        tp1 = tp_cfg.get("tp1", {})
        self.tp1_enabled = tp1.get("enabled", True)
        self.tp1_r = tp1.get("target_r", 1.2)
        self.tp1_close_pct = tp1.get("close_pct", 0.35)

        # TP2配置
        tp2 = tp_cfg.get("tp2", {})
        self.tp2_enabled = tp2.get("enabled", True)
        self.tp2_type = tp2.get("target_type", "resistance_zone")
        self.tp2_r = tp2.get("target_r", 2.0)
        self.tp2_close_pct = tp2.get("close_pct", 0.35)
        self.tp2_proximity = tp2.get("resistance_zone_proximity_pct", 0.005)

        # 跟踪止盈配置
        trail = tp_cfg.get("trailing", {})
        self.trail_enabled = trail.get("enabled", True)
        self.trail_mode = trail.get("mode", "structure")
        self.trail_ema_period = trail.get("ema_period", 20)
        self.trail_ema_tf = trail.get("ema_timeframe", "5m")
        self.trail_struct_lookback = trail.get("structure_lookback_candles", 10)
        self.trail_atr_period = trail.get("atr_period", 14)
        self.trail_atr_mult = trail.get("atr_multiplier", 2.0)
        self.trail_activation_r = trail.get("activation_r", 1.0)

        # 保本配置
        self.breakeven_r = tp_cfg.get("breakeven_activation_r", 1.0)
        self.breakeven_buffer = tp_cfg.get("breakeven_buffer_pct", 0.001)

    def init_position(
        self,
        entry_price: float,
        stop_price: float,
        initial_qty: float,
        resistance_zone: Optional[Dict] = None,
    ) -> Dict:
        """
        初始化持仓状态。

        :param entry_price:      入场价格
        :param stop_price:       初始止损价格
        :param initial_qty:      初始持仓数量（BTC）
        :param resistance_zone:  最近压力区（用于TP2目标）
        :return:                 持仓状态字典
        """
        risk_per_unit = entry_price - stop_price  # 1R = 入场价 - 止损价
        tp1_price = entry_price + risk_per_unit * self.tp1_r
        tp2_price = self._calc_tp2_price(entry_price, risk_per_unit, resistance_zone)

        state = {
            "entry_price": entry_price,
            "stop_price": stop_price,
            "current_stop": stop_price,      # 当前有效止损（会随保本/跟踪移动）
            "risk_per_unit": risk_per_unit,
            "initial_qty": initial_qty,
            "remaining_qty": initial_qty,
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "tp1_triggered": False,
            "tp2_triggered": False,
            "breakeven_triggered": False,
            "trailing_active": False,
            "trailing_stop": None,
            "phase": "WAITING_TP1",
            "closed_qty": 0.0,
            "realized_pnl": 0.0,
            "events": [],
        }

        logger.info(
            f"[ExitManager] 持仓初始化 | 入场={entry_price:.2f} | 止损={stop_price:.2f} "
            f"| 1R={risk_per_unit:.2f} | TP1={tp1_price:.2f} | TP2={tp2_price:.2f} "
            f"| 数量={initial_qty:.6f}"
        )
        return state

    def _calc_tp2_price(
        self,
        entry_price: float,
        risk_per_unit: float,
        resistance_zone: Optional[Dict],
    ) -> float:
        """计算TP2目标价格。"""
        if self.tp2_type == "resistance_zone" and resistance_zone:
            # 使用压力区下沿作为TP2目标（留一点缓冲）
            tp2 = resistance_zone["price_start"] * (1 - self.tp2_proximity)
            # 确保TP2至少达到tp2_r倍R
            min_tp2 = entry_price + risk_per_unit * self.tp2_r
            return max(tp2, min_tp2)
        else:
            return entry_price + risk_per_unit * self.tp2_r

    def update(
        self,
        state: Dict,
        current_price: float,
        df_5m: pd.DataFrame,
        resistance_zone: Optional[Dict] = None,
    ) -> Dict:
        """
        根据当前价格更新持仓状态。

        并行退出设计：
          1. 止损触发 -> 全仓平仓（最高优先级）
          2. 保本移损 -> 盈利达 1R 后止损移至成本价
          3. TP1 -> 盈利达 1.2R 减仓 35%
          4. TP1 触发后，TP2 与跟踪止盈并行监控：
             - TP2：价格靠近压力区时减仓 35%
             - 跟踪止盈：保本触发后立即开始跟踪，不再需要等待 TP2
             - 哪个先到触发哪个，两者互不干扰

        :param state:            当前持仓状态（由 init_position 创建）
        :param current_price:    当前市场价格
        :param df_5m:            5M K线数据（用于跟踪止盈计算）
        :param resistance_zone:  最近压力区（可能随市场变化更新）
        :return:                 更新后的状态 + 操作指令列表
        """
        if state["phase"] == "CLOSED":
            return state

        actions = []
        entry = state["entry_price"]
        risk = state["risk_per_unit"]
        current_r = (current_price - entry) / risk if risk > 0 else 0

        # ── 止损触发（最高优先级） ──────────────────────────────────────────
        if current_price <= state["current_stop"]:
            close_qty = state["remaining_qty"]
            pnl = (current_price - entry) * close_qty
            state["realized_pnl"] += pnl
            state["remaining_qty"] = 0
            state["closed_qty"] += close_qty
            state["phase"] = "CLOSED"
            event = {
                "type": "STOP_LOSS",
                "price": current_price,
                "qty": close_qty,
                "pnl": pnl,
                "reason": f"触发止损（当前止损={state['current_stop']:.2f}）",
            }
            state["events"].append(event)
            actions.append(event)
            logger.info(
                f"[ExitManager] 止损触发 | 价格={current_price:.2f} "
                f"| 止损={state['current_stop']:.2f} | 数量={close_qty:.6f} | PnL={pnl:.2f}"
            )
            return {**state, "actions": actions}

        # ── 保本移损 ────────────────────────────────────────────────────────
        if not state["breakeven_triggered"] and current_r >= self.breakeven_r:
            new_stop = entry * (1 + self.breakeven_buffer)
            if new_stop > state["current_stop"]:
                state["current_stop"] = new_stop
                state["breakeven_triggered"] = True
                event = {
                    "type": "BREAKEVEN",
                    "price": current_price,
                    "new_stop": new_stop,
                    "reason": f"盈利达到{current_r:.2f}R，止损移至保本价{new_stop:.2f}",
                }
                state["events"].append(event)
                actions.append(event)
                logger.info(
                    f"[ExitManager] 保本移损 | 当前价={current_price:.2f} "
                    f"| 新止损={new_stop:.2f} | 盈利={current_r:.2f}R"
                )

        # ── TP1：第一止盈 ──────────────────────────────────────────────────
        if (self.tp1_enabled and not state["tp1_triggered"] and
                current_price >= state["tp1_price"]):
            close_qty = round(state["initial_qty"] * self.tp1_close_pct, 8)
            close_qty = min(close_qty, state["remaining_qty"])
            if close_qty > 0:
                pnl = (current_price - entry) * close_qty
                state["realized_pnl"] += pnl
                state["remaining_qty"] -= close_qty
                state["closed_qty"] += close_qty
                state["tp1_triggered"] = True
                # 并行模式：TP1 触发后进入 ACTIVE 状态，TP2 和跟踪止盈并行开始
                state["phase"] = "ACTIVE"
                event = {
                    "type": "TP1",
                    "price": current_price,
                    "qty": close_qty,
                    "pnl": pnl,
                    "reason": (
                        f"TP1触发（盈利{current_r:.2f}R >= {self.tp1_r}R）"
                        f"，减仓{self.tp1_close_pct:.0%}"
                    ),
                }
                state["events"].append(event)
                actions.append(event)
                logger.info(
                    f"[ExitManager] TP1触发 | 价格={current_price:.2f} "
                    f"| 减仓={close_qty:.6f} | 剩余={state['remaining_qty']:.6f} "
                    f"| PnL={pnl:.2f}"
                )

        # ── TP2：第二止盈（TP1 触发后并行监控） ────────────────────────────
        if (self.tp2_enabled and state["tp1_triggered"] and
                not state["tp2_triggered"] and
                state["remaining_qty"] > 0 and
                current_price >= state["tp2_price"]):
            close_qty = round(state["initial_qty"] * self.tp2_close_pct, 8)
            close_qty = min(close_qty, state["remaining_qty"])
            if close_qty > 0:
                pnl = (current_price - entry) * close_qty
                state["realized_pnl"] += pnl
                state["remaining_qty"] -= close_qty
                state["closed_qty"] += close_qty
                state["tp2_triggered"] = True
                event = {
                    "type": "TP2",
                    "price": current_price,
                    "qty": close_qty,
                    "pnl": pnl,
                    "reason": (
                        f"TP2触发（价格{current_price:.2f} >= TP2目标{state['tp2_price']:.2f}）"
                        f"，再减仓{self.tp2_close_pct:.0%}"
                    ),
                }
                state["events"].append(event)
                actions.append(event)
                logger.info(
                    f"[ExitManager] TP2触发 | 价格={current_price:.2f} "
                    f"| 减仓={close_qty:.6f} | 剩余={state['remaining_qty']:.6f} "
                    f"| PnL={pnl:.2f}"
                )
                # TP2 平仓后如果剩余仓位为零，直接关闭
                if state["remaining_qty"] <= 0:
                    state["phase"] = "CLOSED"
                    return {**state, "actions": actions}

        # ── 跟踪止盈（TP1 触发 + 保本已移后立即并行开始） ──────────────────
        # 并行设计：不再需要等待 TP2 就能开始跟踪，避免僵尸仓位
        trailing_active_condition = (
            self.trail_enabled
            and state["tp1_triggered"]          # TP1 已触发
            and state["breakeven_triggered"]    # 保本已移（锁住最小保护）
            and state["remaining_qty"] > 0
            and current_r >= self.trail_activation_r
        )
        if trailing_active_condition:
            state["trailing_active"] = True
            trail_stop = self._calc_trailing_stop(state, current_price, df_5m)
            if trail_stop is not None:
                # 跟踪止损只能向上移动（对多单）
                if state["trailing_stop"] is None or trail_stop > state["trailing_stop"]:
                    state["trailing_stop"] = trail_stop
                    state["current_stop"] = max(state["current_stop"], trail_stop)

            # 检查跟踪止盈是否触发
            if state["trailing_stop"] and current_price <= state["trailing_stop"]:
                close_qty = state["remaining_qty"]
                pnl = (current_price - entry) * close_qty
                state["realized_pnl"] += pnl
                state["remaining_qty"] = 0
                state["closed_qty"] += close_qty
                state["phase"] = "CLOSED"
                event = {
                    "type": "TRAILING_STOP",
                    "price": current_price,
                    "qty": close_qty,
                    "pnl": pnl,
                    "reason": (
                        f"跟踪止盈触发（{self.trail_mode}模式）"
                        f"，价格{current_price:.2f} <= 跟踪止损{state['trailing_stop']:.2f}"
                    ),
                }
                state["events"].append(event)
                actions.append(event)
                logger.info(
                    f"[ExitManager] 跟踪止盈触发 | 价格={current_price:.2f} "
                    f"| 跟踪止损={state['trailing_stop']:.2f} | 剩余={close_qty:.6f} "
                    f"| PnL={pnl:.2f}"
                )

        return {**state, "actions": actions}

    def _calc_trailing_stop(
        self, state: Dict, current_price: float, df_5m: pd.DataFrame
    ) -> Optional[float]:
        """根据跟踪止盈模式计算当前跟踪止损价格。"""
        if self.trail_mode == "ema":
            if len(df_5m) >= self.trail_ema_period:
                ema = df_5m["close"].ewm(
                    span=self.trail_ema_period, adjust=False
                ).mean().iloc[-1]
                return ema
        elif self.trail_mode == "structure":
            # 最近N根5M K线的最低摆动低点
            lookback = min(self.trail_struct_lookback, len(df_5m))
            recent = df_5m.iloc[-lookback:]
            return recent["low"].min()
        elif self.trail_mode == "atr":
            atr = StopLossCalculator._calc_atr(df_5m, self.trail_atr_period)
            return current_price - atr * self.trail_atr_mult
        return None

    def get_summary(self, state: Dict, current_price: float) -> Dict:
        """获取当前持仓的盈亏摘要。"""
        entry = state["entry_price"]
        risk = state["risk_per_unit"]
        unrealized_pnl = (current_price - entry) * state["remaining_qty"]
        total_pnl = state["realized_pnl"] + unrealized_pnl
        current_r = (current_price - entry) / risk if risk > 0 else 0

        return {
            "phase": state["phase"],
            "entry_price": entry,
            "current_price": current_price,
            "current_stop": state["current_stop"],
            "trailing_stop": state.get("trailing_stop"),
            "current_r": round(current_r, 3),
            "initial_qty": state["initial_qty"],
            "remaining_qty": state["remaining_qty"],
            "closed_qty": state["closed_qty"],
            "realized_pnl": round(state["realized_pnl"], 4),
            "unrealized_pnl": round(unrealized_pnl, 4),
            "total_pnl": round(total_pnl, 4),
            "tp1_triggered": state["tp1_triggered"],
            "tp2_triggered": state["tp2_triggered"],
            "breakeven_triggered": state["breakeven_triggered"],
            "trailing_active": state["trailing_active"],
        }
