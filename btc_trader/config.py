"""
config.py — 全局配置文件
Binance BTCUSDT 永续合约顺势回踩 + 市场共识强度评分策略

所有参数均在此处集中管理，禁止在其他模块中硬编码任何策略参数。
修改此文件即可调整策略行为，无需改动任何逻辑代码。
"""

CONFIG = {

    # =========================================================================
    # 一、交易所与账户基础配置
    # =========================================================================
    "exchange": {
        "name": "binance",
        "symbol": "BTCUSDT",
        "contract_type": "perpetual",          # 永续合约
        "position_mode": "one_way",            # one_way=单向持仓, hedge=双向持仓
        "margin_type": "cross",                # cross=全仓, isolated=逐仓
        "default_leverage": 3,                 # 默认杠杆倍数（第一版保守，可调整至5x）
        "testnet": True,                       # True=使用测试网, False=实盘
        "api_key": "",                         # 从环境变量读取，此处留空
        "api_secret": "",                      # 从环境变量读取，此处留空
    },

    # =========================================================================
    # 二、策略主框架配置
    # =========================================================================
    "strategy": {
        "trend_timeframe": "1h",               # 趋势过滤周期
        "zone_timeframe": "15m",               # 支撑区识别周期
        "entry_timeframe": "5m",               # 入场确认周期
        "trend_ema_period": 200,               # 趋势过滤使用的 EMA 周期
        "trend_ema_fast": 50,                  # 快速EMA（辅助趋势判断）
        "kline_limit": 500,                    # 每次拉取K线数量
    },

    # =========================================================================
    # 三、支撑/压力区识别配置 (support_zone.py)
    # =========================================================================
    "support_zone": {
        "lookback_candles": 100,               # 向前回溯K线数量以识别区域
        "zone_merge_threshold_pct": 0.003,     # 区域合并阈值（价格差 < 0.3% 则合并）
        "swing_prominence_pct": 0.005,         # 摆动高低点最小显著性（相对价格的0.5%）
        "min_zone_width_pct": 0.001,           # 区域最小宽度（0.1%）
        "max_zone_width_pct": 0.015,           # 区域最大宽度（1.5%）
        "zone_touch_tolerance_pct": 0.002,     # 触碰区域的价格容差（0.2%）
    },

    # =========================================================================
    # 四、市场共识强度评分配置 (consensus_score.py)
    # =========================================================================
    "consensus_score": {

        # --- 总开关 ---
        "enabled": True,                       # False=跳过评分，所有区域直接进入5M确认

        # --- 过滤模式 ---
        # "fixed"     = 总分必须 >= fixed_threshold 才通过
        # "percentile"= 总分必须在当前所有候选区域中排名前 top_percentile% 才通过
        # "both"      = 两个条件都要满足（更严格）
        "filtering_mode": "fixed",
        "fixed_threshold": 60,                 # 固定阈值（满分100分）
                                               # v4.1 标准回测参数：60（与 longterm_backtest.py 统一）
        "top_percentile": 30,                  # 百分位模式：取前30%的区域

        # --- 各维度权重（总权重应为100，或归一化处理）---
        "weights": {
            "structure":     20,               # 结构显著性
            "retest":        15,               # 测试次数质量
            "mtf":           20,               # 多周期共振
            "round_number":  15,               # 整数/心理关口
            "confluence":    15,               # 技术重合度
            "volume":        10,               # 成交量结构
            "freshness":      5,               # 近期新鲜度
        },

        # --- 各维度开关（False=该维度不参与评分，权重自动重新分配）---
        "dimension_enabled": {
            "structure":     True,
            "retest":        True,
            "mtf":           True,
            "round_number":  True,
            "confluence":    True,
            "volume":        True,
            "freshness":     True,
        },

        # --- 结构显著性评分参数 ---
        "structure": {
            "lookback_candles": 100,           # 识别前高前低的回溯K线数
            "prominence_pct": 0.01,            # 高低点显著性阈值（1%）
            "box_detection_enabled": True,     # 是否识别大箱体边界
            "box_min_width_candles": 10,       # 箱体最小宽度（K线数）
        },

        # --- 测试次数评分参数 ---
        "retest": {
            "window_candles": 80,              # 统计测试次数的K线窗口
            "optimal_retest_count": 3,         # 最优测试次数（加分峰值点）
            "max_retest_before_penalty": 5,    # 超过此次数开始减分
            "min_reaction_pct": 0.005,         # 有效测试的最小反应幅度（0.5%）
            "strong_reaction_pct": 0.015,      # 强反应阈值（1.5%），额外加分
        },

        # --- 多周期共振评分参数 ---
        "mtf": {
            "enabled": True,
            "timeframes": ["1h", "4h"],        # 检查共振的更高周期
            "proximity_pct": 0.008,            # 判断"靠近"的价格容差（0.8%）
            "ema_periods": [50, 200],          # 检查与哪些EMA共振
        },

        # --- 整数/心理关口评分参数 ---
        "round_number": {
            "enabled": True,
            "major_levels": [1000],            # 主要整数关口间距（1000 USDT）
            "minor_levels": [500],             # 次要整数关口间距（500 USDT）
            "proximity_pct": 0.005,            # 判断"靠近"整数位的容差（0.5%）
            "overlap_bonus_multiplier": 1.5,   # 与其他结构重叠时的加分倍数
        },

        # --- 技术重合度评分参数 ---
        "confluence": {
            "fibonacci_enabled": True,
            "fib_levels": [0.236, 0.382, 0.5, 0.618, 0.786],  # Fibonacci 回撤位
            "fib_swing_lookback": 100,         # 计算Fib的摆动高低点回溯K线数
            "ema_enabled": True,
            "ema_periods": [20, 50, 100, 200], # 检查与哪些EMA重合
            "proximity_pct": 0.005,            # 重合判断的价格容差（0.5%）
            "score_per_overlap": 5,            # 每重合一个技术因素的加分
            "max_score": 100,                  # 该维度最高原始分（归一化前）
        },

        # --- 成交量结构评分参数 ---
        "volume": {
            "enabled": True,
            "lookback_candles": 100,           # 成交量分析回溯K线数
            "volume_spike_multiplier": 2.0,    # 成交量放大倍数阈值（>2倍均量为放量）
            "poc_proximity_pct": 0.005,        # 判断"靠近"成交密集区的容差（0.5%）
        },

        # --- 近期新鲜度评分参数 ---
        "freshness": {
            "max_age_candles": 80,             # 超过此K线数的区域视为"过期"，得0分
            "decay_start_candles": 20,         # 超过此K线数开始衰减
            "decay_factor": 0.95,              # 每根K线的衰减因子（指数衰减）
        },

        # --- 假突破/假跌破联动配置 ---
        "fakeout": {
            "enabled": True,
            "min_score_for_fakeout_bonus": 65, # 只有高于此分数的区域才识别假突破加成
            "fakeout_recovery_candles": 3,     # 跌破后N根K线内收回才算假突破
            "fakeout_max_breach_pct": 0.008,   # 最大允许跌破幅度（0.8%），超过则视为真跌破
            "fakeout_signal_boost": 1.2,       # 假突破确认后，入场信号强度乘数
        },
    },

    # =========================================================================
    # 五、风险管理配置 (risk_manager.py)
    # =========================================================================
    "risk_management": {
        "risk_per_trade_pct": 1.0,             # 每笔交易风险占账户净值的百分比（1%）
        "max_open_positions": 1,               # 最大同时持仓数（第一版：单仓）
        "min_notional_usdt": 10.0,             # Binance 最小名义开仓金额（USDT）

        # 当按风险计算出的名义金额 < min_notional_usdt 时的处理策略：
        # "skip"  = 放弃本次交易（推荐：保守风控优先）
        # "floor" = 强制使用 min_notional_usdt 开仓（需额外检查风险是否可接受）
        "min_notional_breach_action": "skip",

        # 当 floor 模式下，强制使用最小名义金额导致实际风险超过以下倍数时，放弃交易
        # 例如：设定 2.0 表示实际风险不超过目标风险的2倍，否则放弃
        "max_risk_multiplier_for_floor": 2.0,

        # 最大单笔允许亏损（USDT绝对值），作为额外保护层
        # 设为 None 则不启用绝对值保护
        "max_loss_per_trade_usdt": None,
    },

    # =========================================================================
    # 六、止损配置 (exit_manager.py)
    # =========================================================================
    "stop_loss": {
        # 止损模式：
        # "structure" = 结构失效止损（优先，放在支撑区下沿 + 缓冲）
        # "atr"       = ATR动态止损（备选，基于ATR计算止损距离）
        # "fixed_pct" = 固定百分比止损（最后备选，不推荐作为主模式）
        "mode": "structure",

        # --- 结构止损参数（mode = "structure" 时生效）---
        "structure": {
            # 止损放置位置：
            # "zone_bottom" = 支撑区下沿 + 缓冲（默认）
            # "entry_candle_low" = 5M确认K线低点 + 缓冲
            "placement": "zone_bottom",

            # 缓冲模式：
            # "pct"  = 固定百分比缓冲
            # "atr"  = ATR动态缓冲（推荐，更适应市场波动）
            "buffer_mode": "atr",

            # 固定百分比缓冲参数（buffer_mode = "pct" 时生效）
            "buffer_pct": 0.003,               # 支撑区下沿再往下0.3%

            # ATR缓冲参数（buffer_mode = "atr" 时生效）
            "atr_period": 14,                  # ATR计算周期
            "atr_multiplier": 0.3,             # 止损 = 支撑区下沿 - 0.3 * ATR(14)
                                               # v4.1 优化：从 0.5 收窄至 0.3，减少单笔亏损金额
            "atr_timeframe": "15m",            # 计算ATR使用的K线周期
        },

        # --- ATR止损参数（mode = "atr" 时生效）---
        "atr": {
            "period": 14,
            "multiplier": 2.0,                 # 止损 = 入场价 - 2.0 * ATR(14)
            "timeframe": "15m",
        },

        # --- 固定百分比止损参数（mode = "fixed_pct" 时生效）---
        "fixed_pct": {
            "pct": 0.015,                      # 入场价下方1.5%
        },

        # --- 最大止损距离保护（任何模式下均生效）---
        # 如果计算出的止损距离超过此值，放弃本次交易（防止止损太宽）
        "max_stop_distance_pct": 0.03,         # 最大允许止损距离3%
        "min_stop_distance_pct": 0.003,        # 最小允许止损距离0.3%（防止止损太近）
    },

    # =========================================================================
    # 七、止盈配置 (exit_manager.py)
    # =========================================================================
    "take_profit": {

        # --- 分批止盈配置 ---
        "partial_tp_enabled": True,

        # 第一止盈：到达1.0R时减仓（v4.1 优化：更快锁定利润）
        "tp1": {
            "enabled": True,
            "target_r": 1.0,                   # 盈利达到1.0倍风险（1.0R）时触发
                                               # v4.1 优化：从 1.2R 降至 1.0R，更快锁定第一波利润
            "close_pct": 0.50,                 # 减仓50%的持仓
                                               # v4.1 优化：从 35% 提升至 50%，更积极回收利润
        },

        # 第二止盈：到达最近15M压力区时减仓
        "tp2": {
            "enabled": True,
            "target_type": "resistance_zone",  # "resistance_zone" 或 "fixed_r"
            "target_r": 2.0,                   # 当无法识别压力区时，退化为2R
            "close_pct": 0.35,                 # 再减仓35%的持仓（此时总计减仓70%）
            "resistance_zone_proximity_pct": 0.005,  # 靠近压力区的容差
        },

        # 剩余仓位：跟踪止盈
        "trailing": {
            "enabled": True,
            # 跟踪止盈模式：
            # "ema"       = 价格跌破指定EMA时平仓剩余仓位
            # "structure" = 价格跌破5M最近摆动低点时平仓
            # "atr"       = 使用ATR跟踪止盈
            "mode": "structure",

            # EMA跟踪止盈参数（mode = "ema" 时生效）
            "ema_period": 20,
            "ema_timeframe": "5m",

            # 结构跟踪止盈参数（mode = "structure" 时生效）
            "structure_lookback_candles": 10,  # 向前查找最近摆动低点的K线数
            "structure_timeframe": "5m",

            # ATR跟踪止盈参数（mode = "atr" 时生效）
            "atr_period": 14,
            "atr_multiplier": 2.0,
            "atr_timeframe": "5m",

            # 跟踪止盈激活条件：只有盈利超过此R值后，才启动跟踪止盈
            "activation_r": 0.8,               # 盈利达到0.8R后才开始跟踪
                                               # v4.1 优化：从 1.0R 降至 0.8R，更早激活跟踪保护
        },

        # --- 强制止盈保护（防止利润全部回吐）---
        # 当持仓盈利曾达到此R值后，如果价格回撤至入场价，则强制平仓剩余仓位
        "breakeven_activation_r": 1.0,         # 盈利达到1R后，将止损移至保本价
        "breakeven_buffer_pct": 0.001,         # 保本价 = 入场价 + 0.1%（覆盖手续费）
    },

    # =========================================================================
    # 八、入场信号配置 (entry_signal.py)
    # =========================================================================
    "entry_signal": {
        # 5M确认条件
        "confirmation_timeframe": "5m",
        "ema_fast": 9,
        "ema_slow": 21,

        # 确认模式：
        # "candle_close" = 等待5M K线收盘确认
        # "candle_pattern" = 识别5M看涨K线形态（锤子线、吞没等）
        "confirmation_mode": "candle_close",

        # 价格必须在支撑区内（或靠近支撑区）才触发确认
        # 诊断修复：原值 0.005 (0.5%) 过严，导致 5M 确认几乎全部失败
        # 放宽至 0.015 (1.5%)，给价格在 5M 周期内足够的波动容忍空间
        "zone_proximity_pct": 0.015,

        # 确认K线的最小实体比例（实体/总振幅），过小的十字星不算确认
        # 诊断修复：原值 0.3 偏严，放宽至 0.2 以减少无效拦截
        "min_body_ratio": 0.2,

        # 是否要求成交量确认（确认K线成交量 > N倍均量）
        "volume_confirmation_enabled": True,
        "volume_confirmation_multiplier": 1.2,

        # 假突破确认后的额外宽松条件
        "fakeout_relax_body_ratio": True,      # 假突破确认时，放宽实体比例要求
    },

    # =========================================================================
    # 九、日志配置 (logger.py)
    # =========================================================================
    "logging": {
        "level": "INFO",                       # DEBUG / INFO / WARNING / ERROR
        "log_dir": "logs",
        "log_filename": "btc_trader.log",
        "max_bytes": 10 * 1024 * 1024,         # 单个日志文件最大10MB
        "backup_count": 10,                    # 保留10个历史日志文件
        "console_output": True,                # 是否同时输出到控制台
        "log_consensus_score": True,           # 是否记录共识评分详情
        "log_exit_events": True,               # 是否记录每次止盈/止损事件
        "log_order_events": True,              # 是否记录每次下单/撤单事件
    },

    # =========================================================================
    # 十、回测配置 (backtest.py)
    # =========================================================================
    "backtest": {
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "initial_capital": 10000,              # 初始资金（USDT）
        "commission_rate": 0.0005,             # 手续费率（0.05%，Binance Futures Taker）
        "slippage_pct": 0.0002,                # 滑点（0.02%）
        "data_source": "csv",                  # "csv" 或 "api"
        "data_dir": "data",

        # 对比回测：自动运行两次（有/无共识评分过滤），输出对比报告
        "compare_mode": True,
    },
}
