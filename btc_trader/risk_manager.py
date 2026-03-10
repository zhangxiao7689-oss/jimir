"""
risk_manager.py — 仓位计算与风险控制模块

职责：
  - 根据账户净值和每笔风险比例计算目标仓位
  - 处理 Binance 最小名义金额（10 USDT）约束
  - 处理精度问题（价格精度、数量精度）
  - 明确定义当风险计算结果与最小名义金额冲突时的处理策略

核心逻辑（最小名义金额与风险控制联动）：
  ┌─────────────────────────────────────────────────────────────────┐
  │ 情况A：按风险计算的名义金额 >= 10 USDT                          │
  │   -> 正常开仓，使用风险计算结果                                 │
  │                                                                 │
  │ 情况B：按风险计算的名义金额 < 10 USDT                           │
  │   -> 取决于 min_notional_breach_action 配置：                   │
  │      "skip"  = 放弃本次交易（默认，保守风控优先）               │
  │      "floor" = 强制使用 10 USDT 开仓，但需额外检查：            │
  │               实际风险 <= 目标风险 × max_risk_multiplier_for_floor│
  │               若超过，则放弃交易                                 │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Binance 精度工具
# ---------------------------------------------------------------------------

class BinancePrecision:
    """
    处理 Binance BTCUSDT 永续合约的精度和最小下单量约束。

    精度信息来源于 Binance exchangeInfo API，此处提供默认值，
    实盘时应从 API 动态获取并更新。
    """

    # BTCUSDT 永续合约默认精度（实盘时从 exchangeInfo 动态获取）
    PRICE_PRECISION = 1        # 价格小数位数（如 65000.1）
    QTY_PRECISION = 3          # 数量小数位数（如 0.001 BTC）
    MIN_QTY = 0.001            # 最小下单数量（BTC）
    MIN_NOTIONAL = 5.0         # Binance 实际最小名义金额（USDT），策略层使用10 USDT

    def __init__(self, exchange_info: Optional[Dict] = None):
        """
        :param exchange_info: 从 Binance exchangeInfo API 获取的交易对信息
                              如果为 None，使用上面的默认值
        """
        if exchange_info:
            self._parse_exchange_info(exchange_info)

    def _parse_exchange_info(self, info: Dict):
        """从 exchangeInfo 解析精度信息。"""
        try:
            filters = info.get("filters", [])
            for f in filters:
                if f["filterType"] == "PRICE_FILTER":
                    tick = float(f["tickSize"])
                    self.PRICE_PRECISION = max(0, -int(math.log10(tick)))
                elif f["filterType"] == "LOT_SIZE":
                    step = float(f["stepSize"])
                    self.QTY_PRECISION = max(0, -int(math.log10(step)))
                    self.MIN_QTY = float(f["minQty"])
                elif f["filterType"] == "MIN_NOTIONAL":
                    self.MIN_NOTIONAL = float(f.get("notional", 5.0))
        except Exception as e:
            logger.warning(f"[RiskManager] 解析 exchangeInfo 失败，使用默认精度: {e}")

    def round_price(self, price: float) -> float:
        """将价格四舍五入到交易所允许的精度。"""
        return round(price, self.PRICE_PRECISION)

    def round_qty(self, qty: float) -> float:
        """将数量向下取整到交易所允许的精度（防止超量）。"""
        factor = 10 ** self.QTY_PRECISION
        return math.floor(qty * factor) / factor

    def is_qty_valid(self, qty: float) -> bool:
        """检查数量是否满足最小下单量要求。"""
        return qty >= self.MIN_QTY

    def is_notional_valid(self, qty: float, price: float) -> bool:
        """检查名义金额是否满足交易所最小要求。"""
        return qty * price >= self.MIN_NOTIONAL


# ---------------------------------------------------------------------------
# 核心风险管理类
# ---------------------------------------------------------------------------

class RiskManager:
    """
    仓位计算与风险控制管理器。

    使用方式：
        rm = RiskManager(config, precision)
        result = rm.calculate_position(
            account_balance=10000,
            entry_price=65000,
            stop_price=64000,
        )
        if result['valid']:
            qty = result['qty']
    """

    def __init__(self, config: Dict, precision: Optional[BinancePrecision] = None):
        rm_cfg = config.get("risk_management", {})
        self.risk_pct = rm_cfg.get("risk_per_trade_pct", 1.0) / 100.0
        self.max_positions = rm_cfg.get("max_open_positions", 1)
        self.min_notional = rm_cfg.get("min_notional_usdt", 10.0)
        self.breach_action = rm_cfg.get("min_notional_breach_action", "skip")
        self.max_risk_mult = rm_cfg.get("max_risk_multiplier_for_floor", 2.0)
        self.max_loss_usdt = rm_cfg.get("max_loss_per_trade_usdt", None)

        exch_cfg = config.get("exchange", {})
        self.leverage = exch_cfg.get("default_leverage", 5)

        self.precision = precision or BinancePrecision()

    def calculate_position(
        self,
        account_balance: float,
        entry_price: float,
        stop_price: float,
        current_open_positions: int = 0,
    ) -> Dict:
        """
        计算本次交易的开仓数量。

        :param account_balance:         账户净值（USDT）
        :param entry_price:             计划入场价格
        :param stop_price:              计算好的止损价格
        :param current_open_positions:  当前已有持仓数量
        :return:                        仓位计算结果字典
        """
        result = {
            "valid": False,
            "qty": 0.0,
            "notional_usdt": 0.0,
            "risk_usdt": 0.0,
            "risk_pct_actual": 0.0,
            "leverage": self.leverage,
            "method": "",
            "rejection_reason": None,
            "warnings": [],
        }

        # --- 前置检查 ---
        if current_open_positions >= self.max_positions:
            result["rejection_reason"] = (
                f"当前持仓数（{current_open_positions}）已达最大限制（{self.max_positions}），"
                f"不开新仓"
            )
            return result

        if entry_price <= 0 or stop_price <= 0:
            result["rejection_reason"] = "入场价或止损价无效"
            return result

        stop_distance = entry_price - stop_price
        if stop_distance <= 0:
            result["rejection_reason"] = f"止损价（{stop_price}）高于入场价（{entry_price}），逻辑错误"
            return result

        # --- Step 1：按账户风险计算目标亏损金额 ---
        target_risk_usdt = account_balance * self.risk_pct

        # 绝对值保护
        if self.max_loss_usdt is not None:
            target_risk_usdt = min(target_risk_usdt, self.max_loss_usdt)

        # --- Step 2：根据止损距离计算目标数量 ---
        # 公式：qty = 目标风险金额 / 每单位止损距离
        # 注意：这里计算的是"不考虑杠杆"的实际BTC数量
        # 因为止损是按BTC价格计算的，与杠杆无关
        target_qty = target_risk_usdt / stop_distance

        # --- Step 3：精度处理 ---
        target_qty = self.precision.round_qty(target_qty)

        # --- Step 4：计算对应的名义金额 ---
        target_notional = target_qty * entry_price

        logger.debug(
            f"[RiskManager] 风险计算 | 账户={account_balance:.2f} USDT "
            f"| 目标风险={target_risk_usdt:.2f} USDT ({self.risk_pct:.1%}) "
            f"| 止损距离={stop_distance:.2f} | 目标数量={target_qty:.6f} BTC "
            f"| 名义金额={target_notional:.2f} USDT"
        )

        # --- Step 5：最小名义金额检查 ---
        if target_notional < self.min_notional:
            return self._handle_min_notional_breach(
                result, account_balance, entry_price, stop_price,
                stop_distance, target_risk_usdt, target_qty, target_notional
            )

        # --- Step 6：检查数量有效性 ---
        if not self.precision.is_qty_valid(target_qty):
            result["rejection_reason"] = (
                f"计算数量{target_qty:.6f} BTC 低于最小下单量"
                f"{self.precision.MIN_QTY} BTC，放弃交易"
            )
            return result

        # --- Step 7：检查交易所最小名义金额 ---
        if not self.precision.is_notional_valid(target_qty, entry_price):
            result["rejection_reason"] = (
                f"名义金额{target_notional:.2f} USDT 低于交易所最小要求"
                f"{self.precision.MIN_NOTIONAL} USDT"
            )
            return result

        # --- 成功 ---
        actual_risk = stop_distance * target_qty
        result.update({
            "valid": True,
            "qty": target_qty,
            "notional_usdt": round(target_notional, 4),
            "risk_usdt": round(actual_risk, 4),
            "risk_pct_actual": round(actual_risk / account_balance * 100, 4),
            "method": "risk_based",
            "margin_required": round(target_notional / self.leverage, 4),
        })

        logger.info(
            f"[RiskManager] 仓位确认 | 数量={target_qty:.6f} BTC "
            f"| 名义={target_notional:.2f} USDT "
            f"| 实际风险={actual_risk:.2f} USDT ({result['risk_pct_actual']:.3f}%) "
            f"| 所需保证金={result['margin_required']:.2f} USDT"
        )
        return result

    def _handle_min_notional_breach(
        self,
        result: Dict,
        account_balance: float,
        entry_price: float,
        stop_price: float,
        stop_distance: float,
        target_risk_usdt: float,
        target_qty: float,
        target_notional: float,
    ) -> Dict:
        """
        处理名义金额低于最小要求的情况。

        情况B详细处理逻辑：
          - "skip"  模式：直接放弃，记录原因
          - "floor" 模式：强制使用最小名义金额，但检查实际风险是否可接受
        """
        logger.warning(
            f"[RiskManager] 名义金额不足 | 按风险计算={target_notional:.2f} USDT "
            f"< 最小要求={self.min_notional:.2f} USDT"
        )

        if self.breach_action == "skip":
            result["rejection_reason"] = (
                f"按账户风险{self.risk_pct:.1%}计算的名义金额"
                f"（{target_notional:.2f} USDT）低于最小开仓金额"
                f"（{self.min_notional:.2f} USDT）。"
                f"当前账户规模（{account_balance:.2f} USDT）或止损距离"
                f"（{stop_distance:.2f} USDT/BTC）不支持此次交易。"
                f"建议：增加账户资金，或等待止损距离更小的入场机会。"
                f"[action=skip，放弃本次交易]"
            )
            return result

        elif self.breach_action == "floor":
            # 强制使用最小名义金额
            floor_qty = self.precision.round_qty(self.min_notional / entry_price)
            floor_notional = floor_qty * entry_price
            floor_risk = stop_distance * floor_qty

            # 检查强制开仓后的实际风险是否在可接受范围内
            risk_multiplier = floor_risk / target_risk_usdt if target_risk_usdt > 0 else float("inf")

            if risk_multiplier > self.max_risk_mult:
                result["rejection_reason"] = (
                    f"强制使用最小名义金额（{floor_notional:.2f} USDT）会导致实际风险"
                    f"（{floor_risk:.2f} USDT）超过目标风险的"
                    f"{risk_multiplier:.1f}倍（允许最大{self.max_risk_mult}倍）。"
                    f"当前止损距离（{stop_distance:.2f} USDT/BTC）过大，"
                    f"强制开仓风险不可接受，放弃本次交易。"
                    f"[action=floor，风险超限，放弃]"
                )
                return result

            if not self.precision.is_qty_valid(floor_qty):
                result["rejection_reason"] = (
                    f"强制最小名义金额对应数量{floor_qty:.6f} BTC "
                    f"低于最小下单量{self.precision.MIN_QTY} BTC，放弃交易"
                )
                return result

            # 强制开仓通过
            result.update({
                "valid": True,
                "qty": floor_qty,
                "notional_usdt": round(floor_notional, 4),
                "risk_usdt": round(floor_risk, 4),
                "risk_pct_actual": round(floor_risk / account_balance * 100, 4),
                "method": "floor_notional",
                "margin_required": round(floor_notional / self.leverage, 4),
                "warnings": [
                    f"按风险计算的名义金额（{target_notional:.2f} USDT）低于最小要求，"
                    f"已强制使用最小名义金额（{floor_notional:.2f} USDT）。"
                    f"实际风险（{floor_risk:.2f} USDT）为目标风险的{risk_multiplier:.1f}倍，"
                    f"在可接受范围内（<{self.max_risk_mult}倍）。"
                ],
            })

            logger.warning(
                f"[RiskManager] 强制最小名义金额 | 数量={floor_qty:.6f} BTC "
                f"| 名义={floor_notional:.2f} USDT "
                f"| 实际风险={floor_risk:.2f} USDT "
                f"（目标风险的{risk_multiplier:.1f}倍）"
            )
            return result

        # 未知模式，默认skip
        result["rejection_reason"] = f"未知的 min_notional_breach_action: {self.breach_action}"
        return result

    def check_margin_sufficient(
        self, qty: float, entry_price: float, available_margin: float
    ) -> Tuple[bool, str]:
        """
        检查可用保证金是否足够开仓。

        :param qty:              计划开仓数量
        :param entry_price:      入场价格
        :param available_margin: 账户可用保证金（USDT）
        :return:                 (是否足够, 原因说明)
        """
        required_margin = (qty * entry_price) / self.leverage
        # 保留20%安全缓冲，避免开仓后立即触发强平
        required_with_buffer = required_margin * 1.2

        if available_margin >= required_with_buffer:
            return True, f"可用保证金{available_margin:.2f} USDT >= 所需{required_with_buffer:.2f} USDT"
        else:
            return False, (
                f"可用保证金不足：需要{required_with_buffer:.2f} USDT（含20%缓冲），"
                f"实际可用{available_margin:.2f} USDT"
            )
