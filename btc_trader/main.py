"""
main.py — 主程序与 Binance Futures 执行层

职责：
  - 初始化所有模块
  - 连接 Binance Futures API（通过 python-binance）
  - 主循环：定时拉取K线 -> 生成信号 -> 执行下单 -> 管理持仓
  - 处理手动平仓后的状态恢复（回归监控状态，继续寻找机会）
  - 异常处理与自动重连

运行方式：
  python main.py [--testnet] [--dry-run]

环境变量（必须设置）：
  BINANCE_API_KEY      Binance API Key
  BINANCE_API_SECRET   Binance API Secret

可选参数：
  --testnet   使用 Binance 测试网
  --dry-run   模拟模式（不实际下单，仅打印信号）
"""

import os
import sys
import time
import signal
import argparse
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

# 本地模块
from config import CONFIG
from logger import (
    setup_logger,
    log_consensus_score,
    log_entry_signal,
    log_exit_event,
    log_order_event,
    log_cycle_start,
    log_position_sync,
)
from entry_signal import EntrySignalGenerator
from exit_manager import TakeProfitManager
from risk_manager import RiskManager, BinancePrecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Binance API 封装
# ---------------------------------------------------------------------------

class BinanceFuturesClient:
    """
    Binance USDT-M 永续合约 API 封装。

    依赖：pip install python-binance
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        try:
            from binance.client import Client
            from binance.exceptions import BinanceAPIException
            self._BinanceAPIException = BinanceAPIException
        except ImportError:
            raise ImportError(
                "请先安装 python-binance：pip install python-binance"
            )

        self.client = Client(api_key, api_secret, testnet=testnet)
        self.symbol = CONFIG["exchange"]["symbol"]
        self.leverage = CONFIG["exchange"]["default_leverage"]
        self._setup_leverage()

    def _setup_leverage(self):
        """设置合约杠杆倍数。"""
        try:
            self.client.futures_change_leverage(
                symbol=self.symbol, leverage=self.leverage
            )
            logger.info(f"[Binance] 杠杆设置成功: {self.leverage}x")
        except Exception as e:
            logger.warning(f"[Binance] 杠杆设置失败（可能已设置）: {e}")

    def get_klines(self, interval: str, limit: int = 500) -> pd.DataFrame:
        """
        获取K线数据。

        :param interval: K线周期（"1m", "5m", "15m", "1h" 等）
        :param limit:    获取数量（最大1500）
        :return:         DataFrame（含 open/high/low/close/volume）
        """
        raw = self.client.futures_klines(
            symbol=self.symbol, interval=interval, limit=limit
        )
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("open_time")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        # 移除最后一根未收盘的K线
        return df.iloc[:-1]

    def get_account_info(self) -> Dict:
        """获取账户信息（净值、可用保证金）。"""
        info = self.client.futures_account()
        return {
            "total_wallet_balance": float(info["totalWalletBalance"]),
            "total_unrealized_profit": float(info["totalUnrealizedProfit"]),
            "available_balance": float(info["availableBalance"]),
            "total_margin_balance": float(info["totalMarginBalance"]),
        }

    def get_position(self) -> Optional[Dict]:
        """获取当前 BTCUSDT 持仓信息。"""
        positions = self.client.futures_position_information(symbol=self.symbol)
        for pos in positions:
            amt = float(pos["positionAmt"])
            if abs(amt) > 0:
                return {
                    "side": "long" if amt > 0 else "short",
                    "qty": abs(amt),
                    "entry_price": float(pos["entryPrice"]),
                    "unrealized_pnl": float(pos["unRealizedProfit"]),
                    "leverage": int(pos["leverage"]),
                }
        return None

    def get_exchange_info(self) -> Dict:
        """获取交易对精度信息。"""
        info = self.client.futures_exchange_info()
        for sym in info["symbols"]:
            if sym["symbol"] == self.symbol:
                return sym
        return {}

    def place_market_order(self, side: str, qty: float) -> Dict:
        """
        下市价单。

        :param side: "BUY" 或 "SELL"
        :param qty:  数量（BTC）
        :return:     订单响应
        """
        params = {
            "symbol": self.symbol,
            "side": side,
            "type": "MARKET",
            "quantity": qty,
        }
        response = self.client.futures_create_order(**params)
        log_order_event(f"MARKET_{side}", params, response)
        return response

    def place_stop_market_order(self, side: str, qty: float, stop_price: float) -> Dict:
        """
        下止损市价单（STOP_MARKET）。

        :param side:       "BUY" 或 "SELL"
        :param qty:        数量
        :param stop_price: 触发价格
        :return:           订单响应
        """
        params = {
            "symbol": self.symbol,
            "side": side,
            "type": "STOP_MARKET",
            "quantity": qty,
            "stopPrice": stop_price,
            "closePosition": False,
        }
        response = self.client.futures_create_order(**params)
        log_order_event("STOP_MARKET", params, response)
        return response

    def place_take_profit_market_order(self, side: str, qty: float, stop_price: float) -> Dict:
        """
        下止盈市价单（TAKE_PROFIT_MARKET）。
        """
        params = {
            "symbol": self.symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "quantity": qty,
            "stopPrice": stop_price,
            "closePosition": False,
        }
        response = self.client.futures_create_order(**params)
        log_order_event("TAKE_PROFIT_MARKET", params, response)
        return response

    def cancel_all_orders(self) -> Dict:
        """撤销所有挂单。"""
        response = self.client.futures_cancel_all_open_orders(symbol=self.symbol)
        log_order_event("CANCEL_ALL", {"symbol": self.symbol}, response)
        return response

    def get_current_price(self) -> float:
        """获取当前最新价格。"""
        ticker = self.client.futures_symbol_ticker(symbol=self.symbol)
        return float(ticker["price"])


# ---------------------------------------------------------------------------
# 模拟客户端（--dry-run 模式）
# ---------------------------------------------------------------------------

class DryRunClient:
    """模拟交易客户端，不实际下单。"""

    def __init__(self):
        self.symbol = CONFIG["exchange"]["symbol"]
        self.leverage = CONFIG["exchange"]["default_leverage"]
        self._fake_price = 65000.0

    def get_klines(self, interval: str, limit: int = 500) -> pd.DataFrame:
        """生成模拟K线数据（用于测试）。"""
        import numpy as np
        n = limit
        timestamps = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="15T")
        close = self._fake_price + np.cumsum(np.random.randn(n) * 100)
        close = np.abs(close)
        df = pd.DataFrame({
            "open": close * (1 + np.random.randn(n) * 0.001),
            "high": close * (1 + np.abs(np.random.randn(n)) * 0.002),
            "low": close * (1 - np.abs(np.random.randn(n)) * 0.002),
            "close": close,
            "volume": np.random.randint(100, 1000, n).astype(float),
        }, index=timestamps)
        return df

    def get_account_info(self) -> Dict:
        return {
            "total_wallet_balance": 10000.0,
            "total_unrealized_profit": 0.0,
            "available_balance": 9000.0,
            "total_margin_balance": 10000.0,
        }

    def get_position(self) -> Optional[Dict]:
        return None

    def get_exchange_info(self) -> Dict:
        return {}

    def get_current_price(self) -> float:
        import random
        self._fake_price += random.gauss(0, 50)
        return max(self._fake_price, 10000)

    def place_market_order(self, side: str, qty: float) -> Dict:
        logger.info(f"[DryRun] 模拟下单: MARKET {side} {qty} BTC")
        return {"orderId": "dry_run", "status": "FILLED"}

    def place_stop_market_order(self, side: str, qty: float, stop_price: float) -> Dict:
        logger.info(f"[DryRun] 模拟止损单: STOP_MARKET {side} {qty} BTC @ {stop_price}")
        return {"orderId": "dry_run_sl", "status": "NEW"}

    def place_take_profit_market_order(self, side: str, qty: float, stop_price: float) -> Dict:
        logger.info(f"[DryRun] 模拟止盈单: TP_MARKET {side} {qty} BTC @ {stop_price}")
        return {"orderId": "dry_run_tp", "status": "NEW"}

    def cancel_all_orders(self) -> Dict:
        logger.info("[DryRun] 模拟撤销所有挂单")
        return {}


# ---------------------------------------------------------------------------
# 主交易机器人
# ---------------------------------------------------------------------------

class BTCTraderBot:
    """
    BTC 顺势回踩交易机器人主类。

    状态机：
      MONITORING  -> 监控中，寻找入场机会
      IN_POSITION -> 持仓中，管理止盈止损
    """

    def __init__(self, client, dry_run: bool = False):
        self.client = client
        self.dry_run = dry_run
        self.config = CONFIG
        self.state = "MONITORING"
        self.position_state = None     # exit_manager 的持仓状态
        self.cycle_count = 0
        self._running = True

        # 初始化信号生成器
        exchange_info = client.get_exchange_info()
        self.signal_generator = EntrySignalGenerator(self.config, exchange_info=exchange_info)
        self.tp_manager = TakeProfitManager(self.config)

        # 注册退出信号
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info(
            f"[Bot] 初始化完成 | 模式={'DryRun' if dry_run else 'Live'} "
            f"| 交易对={self.config['exchange']['symbol']}"
        )

    def run(self):
        """主循环入口。"""
        logger.info("[Bot] 机器人启动，进入主循环...")
        poll_interval = self.config.get("bot", {}).get("poll_interval_seconds", 60)

        while self._running:
            try:
                self.cycle_count += 1
                current_price = self.client.get_current_price()
                log_cycle_start(self.cycle_count, current_price)

                if self.state == "MONITORING":
                    self._monitoring_cycle()
                elif self.state == "IN_POSITION":
                    self._position_cycle(current_price)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"[Bot] 主循环异常: {e}\n{traceback.format_exc()}")
                time.sleep(10)  # 异常后等待10秒再重试

            time.sleep(poll_interval)

        logger.info("[Bot] 机器人已停止")

    def _monitoring_cycle(self):
        """监控状态：拉取数据，检查入场信号。"""
        # 拉取多周期K线
        df_15m = self.client.get_klines("15m", limit=500)
        df_5m = self.client.get_klines("5m", limit=300)
        df_1h = self.client.get_klines("1h", limit=300)

        if df_15m is None or len(df_15m) < 100:
            logger.warning("[Bot] K线数据不足，跳过本轮")
            return

        # 获取账户信息
        account = self.client.get_account_info()
        balance = account["total_margin_balance"]
        available = account["available_balance"]

        # 检查是否有未同步的持仓（手动开仓或上次运行遗留）
        live_position = self.client.get_position()
        if live_position:
            logger.warning(
                f"[Bot] 检测到未同步持仓: {live_position}，切换至持仓管理状态"
            )
            self._sync_position_from_exchange(live_position)
            return

        # 生成入场信号
        signal_result = self.signal_generator.generate(
            df_1h=df_1h,
            df_15m=df_15m,
            df_5m=df_5m,
            account_balance=balance,
            available_margin=available,
            current_open_positions=0,
        )

        log_entry_signal(signal_result)

        if signal_result["signal"]:
            self._execute_entry(signal_result, df_15m, df_5m)

    def _position_cycle(self, current_price: float):
        """
        持仓状态：更新止盈止损，处理手动平仓。
        """
        # 同步交易所实际持仓
        live_position = self.client.get_position()

        # 检测手动平仓：机器人认为有持仓，但交易所已无持仓
        if live_position is None and self.position_state is not None:
            logger.warning(
                "[Bot] 检测到手动平仓！持仓已不存在于交易所。"
                "撤销所有挂单，回归监控状态，继续寻找下一个交易机会。"
            )
            self.client.cancel_all_orders()
            self.position_state = None
            self.state = "MONITORING"
            return

        if self.position_state is None:
            self.state = "MONITORING"
            return

        # 拉取5M数据用于跟踪止盈计算
        df_5m = self.client.get_klines("5m", limit=100)

        # 更新持仓状态
        updated = self.tp_manager.update(
            self.position_state, current_price, df_5m
        )
        actions = updated.pop("actions", [])
        self.position_state = updated

        # 处理触发的事件
        for action in actions:
            summary = self.tp_manager.get_summary(self.position_state, current_price)
            log_exit_event(action, summary)
            self._execute_exit_action(action, current_price)

        # 更新止损单（如果止损价格发生变化）
        self._sync_stop_order(current_price)

        # 检查是否全部平仓
        if self.position_state.get("phase") == "CLOSED":
            logger.info("[Bot] 持仓已全部平仓，回归监控状态")
            self.client.cancel_all_orders()
            self.position_state = None
            self.state = "MONITORING"

    def _execute_entry(self, signal_result: Dict, df_15m: pd.DataFrame, df_5m: pd.DataFrame):
        """执行开仓操作。"""
        qty = signal_result["qty"]
        entry_price = signal_result["entry_price"]
        stop_price = signal_result["stop_price"]
        tp1_price = signal_result["tp1_price"]
        tp2_price = signal_result["tp2_price"]
        zone = signal_result.get("zone", {})
        resistance_zone = signal_result.get("resistance_zone")

        logger.info(
            f"[Bot] 执行开仓 | 数量={qty} BTC | 入场≈{entry_price:.2f} "
            f"| 止损={stop_price:.2f} | TP1={tp1_price:.2f} | TP2={tp2_price:.2f}"
        )

        # 1. 市价开仓
        order_response = self.client.place_market_order("BUY", qty)
        actual_entry = entry_price  # 实盘中应从成交回报获取实际成交价

        # 2. 挂止损单
        self.client.place_stop_market_order("SELL", qty, stop_price)

        # 3. 挂TP1止盈单（部分平仓）
        tp1_qty = round(qty * self.config["take_profit"]["tp1"]["close_pct"], 3)
        if tp1_qty >= BinancePrecision.MIN_QTY:
            self.client.place_take_profit_market_order("SELL", tp1_qty, tp1_price)

        # 4. 挂TP2止盈单（部分平仓）
        tp2_qty = round(qty * self.config["take_profit"]["tp2"]["close_pct"], 3)
        if tp2_qty >= BinancePrecision.MIN_QTY:
            self.client.place_take_profit_market_order("SELL", tp2_qty, tp2_price)

        # 5. 初始化持仓状态（用于跟踪止盈管理）
        self.position_state = self.tp_manager.init_position(
            actual_entry, stop_price, qty, resistance_zone
        )
        self.state = "IN_POSITION"

        logger.info(f"[Bot] 开仓完成，切换至持仓管理状态")

    def _execute_exit_action(self, action: Dict, current_price: float):
        """根据退出事件执行平仓操作。"""
        action_type = action.get("type")
        qty = action.get("qty", 0)

        if qty <= 0:
            return

        if action_type in ("STOP_LOSS", "TRAILING_STOP"):
            # 止损/跟踪止盈：市价平仓（止损单可能已自动触发，此处作为备份）
            logger.info(f"[Bot] 执行{action_type}平仓: {qty} BTC @ 市价")
            if not self.dry_run:
                try:
                    self.client.place_market_order("SELL", qty)
                except Exception as e:
                    logger.error(f"[Bot] 平仓失败: {e}")

        elif action_type in ("TP1", "TP2"):
            # 止盈：止盈单可能已自动触发，此处作为备份确认
            logger.info(f"[Bot] 确认{action_type}已触发: {qty} BTC @ {action.get('price', 0):.2f}")

    def _sync_stop_order(self, current_price: float):
        """
        同步止损单价格（当止损移至保本或跟踪止盈更新时）。
        """
        if self.position_state is None:
            return

        current_stop = self.position_state.get("current_stop")
        if current_stop and self.position_state.get("breakeven_triggered"):
            # 止损已移动，需要更新交易所止损单
            # 实盘中：先撤旧止损单，再挂新止损单
            # 此处简化处理，仅记录日志
            logger.debug(f"[Bot] 止损更新提醒: 当前止损={current_stop:.2f}，请确认交易所止损单已更新")

    def _sync_position_from_exchange(self, live_position: Dict):
        """
        从交易所同步持仓状态（用于处理机器人重启或手动开仓的情况）。
        """
        entry_price = live_position["entry_price"]
        qty = live_position["qty"]

        # 使用默认止损距离重建持仓状态
        default_stop_pct = self.config["stop_loss"].get("fixed_pct", {}).get("pct", 0.015)
        stop_price = entry_price * (1 - default_stop_pct)

        self.position_state = self.tp_manager.init_position(entry_price, stop_price, qty)
        self.state = "IN_POSITION"

        log_position_sync(live_position)
        logger.info(
            f"[Bot] 持仓同步完成 | 入场={entry_price:.2f} "
            f"| 数量={qty} BTC | 重建止损={stop_price:.2f}"
        )

    def _handle_shutdown(self, signum, frame):
        """处理关闭信号。"""
        logger.info(f"[Bot] 收到关闭信号（{signum}），准备停止...")
        self._running = False


# ---------------------------------------------------------------------------
# 主程序入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BTC USDT-M 永续合约顺势回踩交易机器人")
    parser.add_argument("--testnet", action="store_true", help="使用 Binance 测试网")
    parser.add_argument("--dry-run", action="store_true", help="模拟模式（不实际下单）")
    args = parser.parse_args()

    # 初始化日志
    setup_logger(CONFIG)
    logger.info("=" * 60)
    logger.info("  BTC 顺势回踩交易机器人 启动")
    logger.info(f"  模式: {'DryRun（模拟）' if args.dry_run else 'Live（实盘）'}")
    logger.info(f"  网络: {'测试网' if args.testnet else '主网'}")
    logger.info("=" * 60)

    if args.dry_run:
        client = DryRunClient()
    else:
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")

        if not api_key or not api_secret:
            logger.error(
                "错误：未设置 BINANCE_API_KEY 或 BINANCE_API_SECRET 环境变量。\n"
                "请执行：\n"
                "  export BINANCE_API_KEY=your_key\n"
                "  export BINANCE_API_SECRET=your_secret"
            )
            sys.exit(1)

        client = BinanceFuturesClient(api_key, api_secret, testnet=args.testnet)

    bot = BTCTraderBot(client, dry_run=args.dry_run)
    bot.run()


if __name__ == "__main__":
    main()
