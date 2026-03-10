"""
logger.py — 结构化日志系统

职责：
  - 初始化全局日志配置（文件轮转 + 控制台输出）
  - 提供专用的结构化日志函数（共识评分、入场信号、退出事件）
  - 确保所有关键决策过程都有完整的可追溯日志

日志格式：
  [时间] [级别] [模块] 消息内容

日志文件：
  logs/btc_trader.log（自动轮转，最大10MB，保留10个备份）
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# 日志初始化
# ---------------------------------------------------------------------------

def setup_logger(config: Dict) -> logging.Logger:
    """
    初始化全局日志系统。

    :param config: 全局配置字典
    :return:       根日志记录器
    """
    log_cfg = config.get("logging", {})
    level_str = log_cfg.get("level", "INFO")
    log_dir = log_cfg.get("log_dir", "logs")
    log_filename = log_cfg.get("log_filename", "btc_trader.log")
    max_bytes = log_cfg.get("max_bytes", 10 * 1024 * 1024)
    backup_count = log_cfg.get("backup_count", 10)
    console_output = log_cfg.get("console_output", True)

    level = getattr(logging, level_str.upper(), logging.INFO)

    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    # 日志格式
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除已有 handlers（避免重复添加）
    root_logger.handlers.clear()

    # 文件 Handler（轮转）
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)

    # 控制台 Handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    root_logger.info(
        f"日志系统初始化完成 | 级别={level_str} | 文件={log_path} "
        f"| 最大{max_bytes//1024//1024}MB × {backup_count}个备份"
    )
    return root_logger


# ---------------------------------------------------------------------------
# 专用结构化日志函数
# ---------------------------------------------------------------------------

_trade_logger = logging.getLogger("trade")


def log_consensus_score(score_result: Dict, config: Optional[Dict] = None):
    """
    输出共识强度评分的完整结构化日志。

    日志内容：
      - 支撑区价格范围
      - 总评分
      - 是否通过阈值
      - 各子项评分明细（原始分、权重、加权分、加分原因、减分原因）
      - 最终是否允许进入5M确认
    """
    if config and not config.get("logging", {}).get("log_consensus_score", True):
        return

    zone = score_result.get("zone", {})
    total = score_result.get("total_score", 0)
    passed = score_result.get("passed_threshold", False)
    threshold = score_result.get("threshold_value", 0)
    details = score_result.get("details", {})

    sep = "=" * 70
    lines = [
        sep,
        f"  【共识强度评分报告】",
        f"  支撑区范围  : {zone.get('price_start', 0):.2f} ~ {zone.get('price_end', 0):.2f} USDT",
        f"  区域类型    : {zone.get('zone_type', 'N/A')} | 来源: {zone.get('source', 'N/A')}",
        f"  区域形成时间: {zone.get('formed_at_time', 'N/A')}",
        f"  历史触碰次数: {zone.get('touch_count', 0)} 次",
        f"  ─────────────────────────────────────────────────────────",
        f"  总评分      : {total:.1f} / 100",
        f"  阈值        : {threshold} ({score_result.get('threshold_type', 'fixed')} 模式)",
        f"  是否通过    : {'✅ 通过' if passed else '❌ 未通过'}",
        f"  ─────────────────────────────────────────────────────────",
        f"  分项明细：",
    ]

    dim_names = {
        "structure":    "结构显著性",
        "retest":       "测试次数质量",
        "mtf":          "多周期共振",
        "round_number": "整数/心理关口",
        "confluence":   "技术重合度",
        "volume":       "成交量结构",
        "freshness":    "近期新鲜度",
    }

    for dim_key, detail in details.items():
        dim_name = dim_names.get(dim_key, dim_key)
        lines.append(
            f"  [{dim_name:10s}] "
            f"原始分={detail.get('raw', 0):5.1f} | "
            f"权重={detail.get('weight', 0):4.1f}% | "
            f"加权分={detail.get('weighted', 0):5.2f}"
        )
        for r in detail.get("reasons", []):
            lines.append(f"      ✓ {r}")
        for p in detail.get("penalties", []):
            lines.append(f"      ✗ {p}")

    if score_result.get("penalty_reasons"):
        lines.append(f"  ─────────────────────────────────────────────────────────")
        lines.append(f"  综合减分原因：")
        for p in score_result["penalty_reasons"]:
            lines.append(f"    ✗ {p}")

    lines.append(f"  ─────────────────────────────────────────────────────────")
    lines.append(
        f"  最终决定    : "
        f"{'✅ 允许进入5M确认阶段' if score_result.get('allow_entry_check') else '❌ 放弃，不进入5M确认'}"
    )
    lines.append(sep)

    _trade_logger.info("\n" + "\n".join(lines))


def log_entry_signal(signal_result: Dict):
    """
    输出入场信号的完整结构化日志。
    """
    sep = "=" * 70
    if signal_result.get("signal"):
        lines = [
            sep,
            f"  【✅ 入场信号触发】",
            f"  方向        : {signal_result.get('direction', 'N/A').upper()}",
            f"  入场价格    : {signal_result.get('entry_price', 0):.2f} USDT",
            f"  止损价格    : {signal_result.get('stop_price', 0):.2f} USDT",
            f"  第一止盈    : {signal_result.get('tp1_price', 0):.2f} USDT",
            f"  第二止盈    : {signal_result.get('tp2_price', 0):.2f} USDT",
            f"  开仓数量    : {signal_result.get('qty', 0):.6f} BTC",
            f"  名义金额    : {signal_result.get('notional_usdt', 0):.2f} USDT",
            f"  风险金额    : {signal_result.get('risk_usdt', 0):.2f} USDT ({signal_result.get('risk_pct', 0):.3f}%)",
            f"  信号强度    : {signal_result.get('signal_strength', 0):.2f}",
            f"  共识评分    : {signal_result.get('consensus_score', {}).get('total_score', 0):.1f}",
            sep,
        ]
    else:
        lines = [
            sep,
            f"  【❌ 信号未触发】",
            f"  拒绝阶段    : {signal_result.get('rejection_stage', 'N/A')}",
            f"  拒绝原因    : {signal_result.get('rejection_reason', 'N/A')}",
            sep,
        ]

    _trade_logger.info("\n" + "\n".join(lines))


def log_exit_event(event: Dict, position_summary: Dict):
    """
    输出退出事件（止损/止盈/跟踪止盈）的结构化日志。
    """
    if not event:
        return

    event_type = event.get("type", "UNKNOWN")
    type_labels = {
        "STOP_LOSS":     "🛑 止损触发",
        "TP1":           "💰 第一止盈",
        "TP2":           "💰 第二止盈",
        "TRAILING_STOP": "🎯 跟踪止盈",
        "BREAKEVEN":     "🔒 保本移损",
    }
    label = type_labels.get(event_type, event_type)

    sep = "-" * 60
    lines = [
        sep,
        f"  【{label}】",
        f"  触发价格    : {event.get('price', 0):.2f} USDT",
        f"  平仓数量    : {event.get('qty', 0):.6f} BTC" if event.get("qty") else "",
        f"  本次盈亏    : {event.get('pnl', 0):.4f} USDT" if event.get("pnl") is not None else "",
        f"  原因        : {event.get('reason', 'N/A')}",
        f"  ─────────────────────────────────────────────",
        f"  当前阶段    : {position_summary.get('phase', 'N/A')}",
        f"  当前盈亏(R) : {position_summary.get('current_r', 0):.3f}R",
        f"  已实现盈亏  : {position_summary.get('realized_pnl', 0):.4f} USDT",
        f"  剩余数量    : {position_summary.get('remaining_qty', 0):.6f} BTC",
        sep,
    ]

    _trade_logger.info("\n" + "\n".join([l for l in lines if l]))


def log_order_event(order_type: str, order_params: Dict, response: Optional[Dict] = None):
    """
    输出下单/撤单事件的结构化日志。
    """
    sep = "-" * 60
    lines = [
        sep,
        f"  【订单事件: {order_type}】",
        f"  参数        : {json.dumps(order_params, ensure_ascii=False)}",
    ]
    if response:
        lines.append(f"  响应        : orderId={response.get('orderId', 'N/A')} | status={response.get('status', 'N/A')}")
    lines.append(sep)
    _trade_logger.info("\n" + "\n".join(lines))


def log_cycle_start(cycle_num: int, current_price: float):
    """记录每次主循环的开始。"""
    _trade_logger.debug(
        f"[主循环 #{cycle_num}] 开始 | 当前价格={current_price:.2f} USDT "
        f"| 时间={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


def log_position_sync(position_info: Dict):
    """记录持仓同步结果。"""
    _trade_logger.info(
        f"[持仓同步] 当前持仓: {json.dumps(position_info, ensure_ascii=False, default=str)}"
    )
