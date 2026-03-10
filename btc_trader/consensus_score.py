"""
consensus_score.py — 市场共识强度评分模块 (Consensus Strength Score)

职责：
  - 接收候选支撑/压力区域（来自 support_zone.py）
  - 从7个独立维度对每个区域进行量化评分
  - 输出总分、是否通过阈值、分项明细及评分理由
  - 不做任何开仓决策，只负责"这个区域值不值得继续观察"

评分维度（共7个，总分100分，各维度权重可配置）：
  1. structure_score    — 结构显著性
  2. retest_score       — 测试次数质量
  3. mtf_score          — 多周期共振
  4. round_number_score — 整数/心理关口
  5. confluence_score   — 技术重合度
  6. volume_score       — 成交量结构
  7. freshness_score    — 近期新鲜度

输出结构示例：
  {
    "total_score": 78.5,
    "passed_threshold": True,
    "threshold_type": "fixed",
    "threshold_value": 60,
    "details": {
      "structure": {"raw": 85, "weighted": 17.0, "reasons": ["近期显著前低 +85"]},
      "retest":    {"raw": 70, "weighted": 10.5, "reasons": ["首次有效测试 +70"]},
      ...
    },
    "bonus_reasons": ["靠近整数关口 $65000"],
    "penalty_reasons": ["测试次数过多，支撑衰减"],
    "allow_entry_check": True,
    "zone": {...}
  }
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 核心评分类
# ---------------------------------------------------------------------------

class ConsensusScorer:
    """
    市场共识强度评分器。

    使用方式：
        scorer = ConsensusScorer(config, mtf_klines)
        result = scorer.score(zone, df_15m)
        if result['passed_threshold']:
            # 进入5M确认逻辑
    """

    def __init__(self, config: Dict, mtf_klines: Optional[Dict[str, pd.DataFrame]] = None):
        """
        :param config: 全局配置字典（CONFIG）
        :param mtf_klines: 多周期K线数据，格式 {"1h": df_1h, "4h": df_4h}
        """
        self.cfg = config.get("consensus_score", {})
        self.weights = self._normalize_weights(self.cfg.get("weights", {}))
        self.dim_enabled = self.cfg.get("dimension_enabled", {})
        self.mtf_klines = mtf_klines or {}
        self.filtering_mode = self.cfg.get("filtering_mode", "fixed")
        self.fixed_threshold = self.cfg.get("fixed_threshold", 60)
        self.top_percentile = self.cfg.get("top_percentile", 30)

    def _normalize_weights(self, raw_weights: Dict) -> Dict:
        """
        将权重归一化，使各维度权重之和为100。
        若某维度被禁用，其权重会重新分配给其他维度。
        """
        dim_enabled = self.cfg.get("dimension_enabled", {})
        active = {k: v for k, v in raw_weights.items() if dim_enabled.get(k, True)}
        total = sum(active.values())
        if total == 0:
            return {}
        return {k: v / total * 100 for k, v in active.items()}

    # -----------------------------------------------------------------------
    # 主评分入口
    # -----------------------------------------------------------------------

    def score(self, zone: Dict, df_primary: pd.DataFrame) -> Dict:
        """
        对单个区域进行全维度评分。

        :param zone:       候选区域字典（来自 support_zone.py）
        :param df_primary: 主识别周期（15M）的K线 DataFrame
        :return:           完整评分结果字典
        """
        if not self.cfg.get("enabled", True):
            # 评分模块被全局禁用，直接通过
            return self._bypass_result(zone)

        details = {}
        bonus_reasons = []
        penalty_reasons = []

        # --- 逐维度评分 ---
        if self.dim_enabled.get("structure", True):
            raw, reasons, penalties = self._score_structure(zone, df_primary)
            details["structure"] = self._make_detail(raw, "structure", reasons, penalties)
            bonus_reasons.extend(reasons)
            penalty_reasons.extend(penalties)

        if self.dim_enabled.get("retest", True):
            raw, reasons, penalties = self._score_retest(zone, df_primary)
            details["retest"] = self._make_detail(raw, "retest", reasons, penalties)
            bonus_reasons.extend(reasons)
            penalty_reasons.extend(penalties)

        if self.dim_enabled.get("mtf", True):
            raw, reasons, penalties = self._score_mtf(zone, df_primary)
            details["mtf"] = self._make_detail(raw, "mtf", reasons, penalties)
            bonus_reasons.extend(reasons)
            penalty_reasons.extend(penalties)

        if self.dim_enabled.get("round_number", True):
            raw, reasons, penalties = self._score_round_number(zone)
            details["round_number"] = self._make_detail(raw, "round_number", reasons, penalties)
            bonus_reasons.extend(reasons)
            penalty_reasons.extend(penalties)

        if self.dim_enabled.get("confluence", True):
            raw, reasons, penalties = self._score_confluence(zone, df_primary)
            details["confluence"] = self._make_detail(raw, "confluence", reasons, penalties)
            bonus_reasons.extend(reasons)
            penalty_reasons.extend(penalties)

        if self.dim_enabled.get("volume", True):
            raw, reasons, penalties = self._score_volume(zone, df_primary)
            details["volume"] = self._make_detail(raw, "volume", reasons, penalties)
            bonus_reasons.extend(reasons)
            penalty_reasons.extend(penalties)

        if self.dim_enabled.get("freshness", True):
            raw, reasons, penalties = self._score_freshness(zone, df_primary)
            details["freshness"] = self._make_detail(raw, "freshness", reasons, penalties)
            bonus_reasons.extend(reasons)
            penalty_reasons.extend(penalties)

        # --- 计算加权总分 ---
        total_score = sum(d["weighted"] for d in details.values())
        total_score = round(min(max(total_score, 0), 100), 2)

        # --- 判断是否通过阈值 ---
        passed = self._check_threshold(total_score)

        result = {
            "total_score": total_score,
            "passed_threshold": passed,
            "threshold_type": self.filtering_mode,
            "threshold_value": self.fixed_threshold,
            "details": details,
            "bonus_reasons": bonus_reasons,
            "penalty_reasons": penalty_reasons,
            "allow_entry_check": passed,
            "zone": zone,
        }

        logger.debug(
            f"[ConsensusScore] Zone {zone['zone_type']} "
            f"[{zone['price_start']:.1f} ~ {zone['price_end']:.1f}] "
            f"| Total={total_score:.1f} | Passed={passed}"
        )

        return result

    def score_batch(self, zones: List[Dict], df_primary: pd.DataFrame) -> List[Dict]:
        """
        对多个候选区域批量评分，并在百分位模式下进行相对排名过滤。

        :param zones:      候选区域列表
        :param df_primary: 主识别周期K线
        :return:           评分结果列表（每个元素对应一个区域的完整评分）
        """
        results = [self.score(z, df_primary) for z in zones]

        if self.filtering_mode in ("percentile", "both") and len(results) > 1:
            scores = [r["total_score"] for r in results]
            threshold_score = np.percentile(scores, 100 - self.top_percentile)
            for r in results:
                percentile_passed = r["total_score"] >= threshold_score
                if self.filtering_mode == "percentile":
                    r["passed_threshold"] = percentile_passed
                elif self.filtering_mode == "both":
                    r["passed_threshold"] = r["passed_threshold"] and percentile_passed
                r["allow_entry_check"] = r["passed_threshold"]

        return results

    # -----------------------------------------------------------------------
    # 维度1：结构显著性评分
    # -----------------------------------------------------------------------

    def _score_structure(self, zone: Dict, df: pd.DataFrame) -> Tuple[float, List[str], List[str]]:
        """
        评估区域的结构显著性：越显眼的位置，越容易形成市场共识。

        评分逻辑：
          - 区域显著性（prominence）越高，得分越高
          - 是否是近期N根K线内的最高/最低点（额外加分）
          - 是否是箱体边界（额外加分）
        """
        cfg = self.cfg.get("structure", {})
        lookback = cfg.get("lookback_candles", 100)

        raw_score = 0.0
        reasons = []
        penalties = []

        prominence = zone.get("prominence", 0.0)
        zone_type = zone["zone_type"]
        mid = zone["mid_price"]

        # 基础分：基于显著性
        if prominence >= 0.02:
            raw_score += 50
            reasons.append(f"极高显著性摆动点（prominence={prominence:.2%}）+50")
        elif prominence >= 0.01:
            raw_score += 35
            reasons.append(f"高显著性摆动点（prominence={prominence:.2%}）+35")
        elif prominence >= 0.005:
            raw_score += 20
            reasons.append(f"中等显著性摆动点（prominence={prominence:.2%}）+20")
        else:
            raw_score += 5
            reasons.append(f"低显著性摆动点（prominence={prominence:.2%}）+5")

        # 是否是近期最高/最低点
        recent_df = df.iloc[-lookback:]
        if zone_type == "support":
            period_low = recent_df["low"].min()
            if abs(mid - period_low) / period_low < 0.005:
                raw_score += 30
                reasons.append(f"近期{lookback}根K线内最低点 +30")
            elif abs(mid - period_low) / period_low < 0.015:
                raw_score += 15
                reasons.append(f"接近近期{lookback}根K线最低点 +15")
        else:
            period_high = recent_df["high"].max()
            if abs(mid - period_high) / period_high < 0.005:
                raw_score += 30
                reasons.append(f"近期{lookback}根K线内最高点 +30")
            elif abs(mid - period_high) / period_high < 0.015:
                raw_score += 15
                reasons.append(f"接近近期{lookback}根K线最高点 +15")

        # 箱体边界加分
        if zone.get("source") == "box":
            raw_score += 20
            reasons.append("大箱体边界结构 +20")

        return min(raw_score, 100.0), reasons, penalties

    # -----------------------------------------------------------------------
    # 维度2：测试次数质量评分
    # -----------------------------------------------------------------------

    def _score_retest(self, zone: Dict, df: pd.DataFrame) -> Tuple[float, List[str], List[str]]:
        """
        评估区域的测试次数质量。

        评分逻辑（非线性，不是"越多越好"）：
          - 首次测试：基础分
          - 2~3次有效测试（每次都有明显反应）：加分峰值
          - 4~5次测试：开始减分（支撑开始衰减）
          - >5次测试：明显减分（支撑大概率已被削弱）
          - 每次测试后的反应幅度影响评分
        """
        cfg = self.cfg.get("retest", {})
        optimal = cfg.get("optimal_retest_count", 3)
        max_before_penalty = cfg.get("max_retest_before_penalty", 5)
        min_reaction = cfg.get("min_reaction_pct", 0.005)
        strong_reaction = cfg.get("strong_reaction_pct", 0.015)

        touch_count = zone.get("touch_count", 0)
        reactions = zone.get("reactions", [])

        raw_score = 0.0
        reasons = []
        penalties = []

        if touch_count == 0:
            raw_score = 10
            reasons.append("区域尚未被测试（原始区域）+10")
            return raw_score, reasons, penalties

        # 有效测试次数（反应幅度 >= min_reaction 的触碰）
        valid_touches = sum(1 for r in reactions if r >= min_reaction)
        strong_touches = sum(1 for r in reactions if r >= strong_reaction)

        # 基础分：基于有效测试次数（非线性曲线）
        if valid_touches == 1:
            raw_score += 40
            reasons.append(f"首次有效测试 +40")
        elif valid_touches == 2:
            raw_score += 60
            reasons.append(f"2次有效测试，支撑得到验证 +60")
        elif valid_touches <= optimal:
            raw_score += 75
            reasons.append(f"{valid_touches}次有效测试，支撑强度良好 +75")
        elif valid_touches <= max_before_penalty:
            raw_score += 50
            reasons.append(f"{valid_touches}次测试，支撑开始衰减 +50")
            penalties.append(f"测试次数偏多（{valid_touches}次），支撑可能衰减")
        else:
            raw_score += 20
            penalties.append(f"测试次数过多（{valid_touches}次），支撑大概率已被削弱 -55")

        # 强反应加分
        if strong_touches >= 2:
            raw_score += 20
            reasons.append(f"{strong_touches}次强反应（>{strong_reaction:.1%}）+20")
        elif strong_touches == 1:
            raw_score += 10
            reasons.append(f"1次强反应（>{strong_reaction:.1%}）+10")

        # 无效测试（触碰但无反应）减分
        invalid_touches = touch_count - valid_touches
        if invalid_touches > 0:
            penalty = min(invalid_touches * 10, 30)
            raw_score -= penalty
            penalties.append(f"{invalid_touches}次无效触碰（无明显反应）-{penalty}")

        return min(max(raw_score, 0), 100.0), reasons, penalties

    # -----------------------------------------------------------------------
    # 维度3：多周期共振评分
    # -----------------------------------------------------------------------

    def _score_mtf(self, zone: Dict, df_primary: pd.DataFrame) -> Tuple[float, List[str], List[str]]:
        """
        评估区域是否与更高周期的关键结构位共振。

        评分逻辑：
          - 每个更高周期的结构位（前高/前低/EMA）与该区域重叠，加分
          - 共振的周期越多，分数越高
        """
        cfg = self.cfg.get("mtf", {})
        if not cfg.get("enabled", True):
            return 50.0, ["多周期共振评分已禁用，给予中性分 50"], []

        timeframes = cfg.get("timeframes", ["1h", "4h"])
        proximity = cfg.get("proximity_pct", 0.008)
        ema_periods = cfg.get("ema_periods", [50, 200])

        mid = zone["mid_price"]
        raw_score = 0.0
        reasons = []
        penalties = []
        max_possible = len(timeframes) * 50 + len(ema_periods) * 25

        for tf in timeframes:
            df_tf = self.mtf_klines.get(tf)
            if df_tf is None or len(df_tf) < 20:
                continue

            # 检查与更高周期前高/前低的共振
            tf_high = df_tf["high"].iloc[-100:].max()
            tf_low = df_tf["low"].iloc[-100:].min()
            recent_swing_highs = self._find_recent_swings(df_tf, "high", n=5)
            recent_swing_lows = self._find_recent_swings(df_tf, "low", n=5)

            for sh in recent_swing_highs:
                if abs(mid - sh) / mid <= proximity:
                    raw_score += 30
                    reasons.append(f"与{tf}前高共振（{sh:.1f}）+30")
                    break

            for sl in recent_swing_lows:
                if abs(mid - sl) / mid <= proximity:
                    raw_score += 30
                    reasons.append(f"与{tf}前低共振（{sl:.1f}）+30")
                    break

            # 检查与更高周期EMA的共振
            for period in ema_periods:
                if len(df_tf) >= period:
                    ema_val = df_tf["close"].ewm(span=period, adjust=False).mean().iloc[-1]
                    if abs(mid - ema_val) / mid <= proximity:
                        raw_score += 20
                        reasons.append(f"与{tf} EMA{period}共振（{ema_val:.1f}）+20")

        # 归一化到100分
        if max_possible > 0:
            normalized = min(raw_score / max_possible * 100, 100)
        else:
            normalized = 50.0

        if not reasons:
            penalties.append("未发现多周期共振结构")
            normalized = max(normalized, 0)

        return normalized, reasons, penalties

    def _find_recent_swings(self, df: pd.DataFrame, swing_type: str, n: int = 5) -> List[float]:
        """找出最近N个摆动高点或低点的价格。"""
        prices = []
        window = 3
        col = "high" if swing_type == "high" else "low"
        compare = (lambda a, b: a > b) if swing_type == "high" else (lambda a, b: a < b)

        for i in range(window, len(df) - window):
            val = df.iloc[i][col]
            neighbors = list(df.iloc[i - window:i][col]) + list(df.iloc[i + 1:i + window + 1][col])
            if all(compare(val, nb) for nb in neighbors):
                prices.append(val)
                if len(prices) >= n:
                    break

        return prices

    # -----------------------------------------------------------------------
    # 维度4：整数/心理关口评分
    # -----------------------------------------------------------------------

    def _score_round_number(self, zone: Dict) -> Tuple[float, List[str], List[str]]:
        """
        评估区域是否靠近整数/心理关口。

        评分逻辑：
          - 靠近主要整数关口（如1000的倍数）：高分
          - 靠近次要整数关口（如500的倍数）：中分
          - 与其他结构重叠时额外加分（在 confluence 维度体现）
        """
        cfg = self.cfg.get("round_number", {})
        if not cfg.get("enabled", True):
            return 50.0, ["整数关口评分已禁用，给予中性分 50"], []

        major_levels = cfg.get("major_levels", [1000])
        minor_levels = cfg.get("minor_levels", [500])
        proximity = cfg.get("proximity_pct", 0.005)

        mid = zone["mid_price"]
        raw_score = 0.0
        reasons = []
        penalties = []

        # 检查主要整数关口
        for level_step in major_levels:
            nearest_major = round(mid / level_step) * level_step
            dist_pct = abs(mid - nearest_major) / mid
            if dist_pct <= proximity:
                score_add = max(0, (1 - dist_pct / proximity)) * 80
                raw_score += score_add
                reasons.append(
                    f"靠近主要整数关口 ${nearest_major:,.0f}（距离{dist_pct:.2%}）+{score_add:.0f}"
                )

        # 检查次要整数关口（避免与主要重复计算）
        for level_step in minor_levels:
            nearest_minor = round(mid / level_step) * level_step
            # 如果这个次要关口已经是主要关口，跳过
            is_major = any(nearest_minor % m == 0 for m in major_levels)
            if is_major:
                continue
            dist_pct = abs(mid - nearest_minor) / mid
            if dist_pct <= proximity:
                score_add = max(0, (1 - dist_pct / proximity)) * 40
                raw_score += score_add
                reasons.append(
                    f"靠近次要整数关口 ${nearest_minor:,.0f}（距离{dist_pct:.2%}）+{score_add:.0f}"
                )

        if not reasons:
            penalties.append("不靠近任何整数/心理关口")

        return min(raw_score, 100.0), reasons, penalties

    # -----------------------------------------------------------------------
    # 维度5：技术重合度评分
    # -----------------------------------------------------------------------

    def _score_confluence(self, zone: Dict, df: pd.DataFrame) -> Tuple[float, List[str], List[str]]:
        """
        评估区域是否有多个技术因素重叠。

        评分逻辑：
          - 每重合一个技术因素（EMA、Fibonacci回撤位等），加分
          - 重合因素越多，分数越高（但有上限）
        """
        cfg = self.cfg.get("confluence", {})
        proximity = cfg.get("proximity_pct", 0.005)
        score_per_overlap = cfg.get("score_per_overlap", 5)

        mid = zone["mid_price"]
        raw_score = 0.0
        reasons = []
        penalties = []
        overlap_count = 0

        # --- EMA重合检查 ---
        if cfg.get("ema_enabled", True):
            ema_periods = cfg.get("ema_periods", [20, 50, 100, 200])
            for period in ema_periods:
                if len(df) >= period:
                    ema_val = df["close"].ewm(span=period, adjust=False).mean().iloc[-1]
                    dist_pct = abs(mid - ema_val) / mid
                    if dist_pct <= proximity:
                        add = score_per_overlap * (1 + (200 - period) / 200)  # 长周期EMA权重更高
                        raw_score += add
                        overlap_count += 1
                        reasons.append(f"与EMA{period}重合（{ema_val:.1f}，距离{dist_pct:.2%}）+{add:.1f}")

        # --- Fibonacci回撤位重合检查 ---
        if cfg.get("fibonacci_enabled", True):
            fib_levels = cfg.get("fib_levels", [0.236, 0.382, 0.5, 0.618, 0.786])
            fib_lookback = cfg.get("fib_swing_lookback", 100)
            fib_zones = self._calc_fibonacci_levels(df, fib_lookback, fib_levels)

            for fib_ratio, fib_price in fib_zones:
                dist_pct = abs(mid - fib_price) / mid
                if dist_pct <= proximity:
                    # 0.618和0.5是最重要的Fibonacci位
                    importance = 1.5 if fib_ratio in [0.5, 0.618] else 1.0
                    add = score_per_overlap * importance
                    raw_score += add
                    overlap_count += 1
                    reasons.append(
                        f"与Fibonacci {fib_ratio:.3f}回撤位重合（{fib_price:.1f}）+{add:.1f}"
                    )

        if overlap_count == 0:
            penalties.append("无技术因素重合")
        elif overlap_count >= 3:
            bonus = 15
            raw_score += bonus
            reasons.append(f"多重技术因素重合（{overlap_count}个）额外加分 +{bonus}")

        return min(raw_score, 100.0), reasons, penalties

    def _calc_fibonacci_levels(
        self, df: pd.DataFrame, lookback: int, ratios: List[float]
    ) -> List[Tuple[float, float]]:
        """计算最近摆动高低点之间的Fibonacci回撤位。"""
        recent = df.iloc[-lookback:]
        swing_high = recent["high"].max()
        swing_low = recent["low"].min()
        diff = swing_high - swing_low

        if diff <= 0:
            return []

        # 判断当前趋势方向（用于确定回撤方向）
        last_close = df["close"].iloc[-1]
        is_uptrend = last_close > (swing_high + swing_low) / 2

        levels = []
        for ratio in ratios:
            if is_uptrend:
                # 上升趋势中，Fibonacci回撤从高点往下
                fib_price = swing_high - diff * ratio
            else:
                # 下降趋势中，Fibonacci回撤从低点往上
                fib_price = swing_low + diff * ratio
            levels.append((ratio, fib_price))

        return levels

    # -----------------------------------------------------------------------
    # 维度6：成交量结构评分
    # -----------------------------------------------------------------------

    def _score_volume(self, zone: Dict, df: pd.DataFrame) -> Tuple[float, List[str], List[str]]:
        """
        评估区域是否伴随明显的量价特征。

        评分逻辑：
          - 区域附近是否有放量K线（成交量 > N倍均量）
          - 是否是放量突破后的回踩区
          - 是否是前期放量反转区（成交密集区）
        """
        cfg = self.cfg.get("volume", {})
        if not cfg.get("enabled", True):
            return 50.0, ["成交量评分已禁用，给予中性分 50"], []

        lookback = cfg.get("lookback_candles", 100)
        spike_mult = cfg.get("volume_spike_multiplier", 2.0)
        poc_proximity = cfg.get("poc_proximity_pct", 0.005)

        mid = zone["mid_price"]
        raw_score = 0.0
        reasons = []
        penalties = []

        recent_df = df.iloc[-lookback:].copy()
        if len(recent_df) < 10:
            return 30.0, ["数据不足，给予基础分 30"], []

        avg_volume = recent_df["volume"].mean()
        if avg_volume == 0:
            return 30.0, ["成交量数据异常，给予基础分 30"], []

        # --- 检查区域附近是否有放量K线 ---
        zone_candles = recent_df[
            (recent_df["low"] <= zone["price_end"] * 1.01) &
            (recent_df["high"] >= zone["price_start"] * 0.99)
        ]

        if len(zone_candles) > 0:
            max_vol_ratio = zone_candles["volume"].max() / avg_volume
            if max_vol_ratio >= spike_mult * 2:
                raw_score += 60
                reasons.append(f"区域内存在极强放量K线（{max_vol_ratio:.1f}倍均量）+60")
            elif max_vol_ratio >= spike_mult:
                raw_score += 40
                reasons.append(f"区域内存在放量K线（{max_vol_ratio:.1f}倍均量）+40")
            else:
                raw_score += 10
                reasons.append(f"区域内成交量正常（{max_vol_ratio:.1f}倍均量）+10")

        # --- 检查是否是放量突破后的回踩区 ---
        # 逻辑：在区域形成后，是否出现过放量突破，然后价格回踩到该区域
        formed_idx = zone.get("formed_at_index", 0)
        post_zone_df = recent_df.iloc[formed_idx:]

        if len(post_zone_df) >= 5:
            # 找放量突破K线
            breakout_candles = post_zone_df[post_zone_df["volume"] > avg_volume * spike_mult]
            if len(breakout_candles) > 0:
                # 检查突破后价格是否回踩到区域
                last_close = df["close"].iloc[-1]
                if abs(last_close - mid) / mid <= poc_proximity * 3:
                    raw_score += 30
                    reasons.append("放量突破后回踩至该区域 +30")

        # --- 成交密集区（POC）检测 ---
        # 简化版：统计每个价格区间的成交量，找成交最密集的区域
        poc_price = self._calc_poc(recent_df)
        if poc_price and abs(mid - poc_price) / mid <= poc_proximity:
            raw_score += 25
            reasons.append(f"靠近成交密集区POC（{poc_price:.1f}）+25")

        if not reasons:
            penalties.append("区域附近无明显成交量特征")
            raw_score = 15

        return min(raw_score, 100.0), reasons, penalties

    def _calc_poc(self, df: pd.DataFrame, bins: int = 20) -> Optional[float]:
        """计算成交量加权价格（简化版POC）。"""
        if len(df) < 5:
            return None
        try:
            price_min = df["low"].min()
            price_max = df["high"].max()
            if price_max <= price_min:
                return None

            bin_edges = np.linspace(price_min, price_max, bins + 1)
            vol_by_bin = np.zeros(bins)

            for _, row in df.iterrows():
                # 将每根K线的成交量按价格范围均匀分配到各价格区间
                candle_bins = np.searchsorted(bin_edges, [row["low"], row["high"]])
                start_bin = max(candle_bins[0] - 1, 0)
                end_bin = min(candle_bins[1], bins - 1)
                if end_bin >= start_bin:
                    n_bins = end_bin - start_bin + 1
                    vol_by_bin[start_bin:end_bin + 1] += row["volume"] / n_bins

            poc_bin = np.argmax(vol_by_bin)
            poc_price = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2
            return poc_price
        except Exception:
            return None

    # -----------------------------------------------------------------------
    # 维度7：近期新鲜度评分
    # -----------------------------------------------------------------------

    def _score_freshness(self, zone: Dict, df: pd.DataFrame) -> Tuple[float, List[str], List[str]]:
        """
        评估区域的时效性：历史上重要但已过期的位置，不应给高权重。

        评分逻辑：
          - 区域形成时间越近，得分越高
          - 超过 max_age_candles 的区域得0分
          - 在 decay_start_candles 到 max_age_candles 之间，指数衰减
          - 最近有市场反应（触碰后反弹）的区域额外加分
        """
        cfg = self.cfg.get("freshness", {})
        max_age = cfg.get("max_age_candles", 80)
        decay_start = cfg.get("decay_start_candles", 20)
        decay_factor = cfg.get("decay_factor", 0.95)

        formed_idx = zone.get("formed_at_index", 0)
        current_idx = len(df) - 1
        age = current_idx - formed_idx

        reasons = []
        penalties = []

        if age <= 0:
            return 100.0, ["刚刚形成的结构，新鲜度最高 +100"], []

        if age > max_age:
            penalties.append(f"区域形成时间过久（{age}根K线前），新鲜度为0")
            return 0.0, reasons, penalties

        if age <= decay_start:
            raw_score = 100.0
            reasons.append(f"近期形成的结构（{age}根K线前），新鲜度高 +100")
        else:
            # 指数衰减
            decay_candles = age - decay_start
            raw_score = 100.0 * (decay_factor ** decay_candles)
            reasons.append(
                f"结构形成于{age}根K线前，新鲜度衰减至 {raw_score:.0f}"
            )

        # 最近是否有市场反应（最近10根K线内有触碰且反应明显）
        reactions = zone.get("reactions", [])
        touch_count = zone.get("touch_count", 0)
        if touch_count > 0 and reactions:
            last_reaction = reactions[-1] if reactions else 0
            if last_reaction >= 0.005:
                bonus = min(20, raw_score * 0.2)
                raw_score = min(raw_score + bonus, 100)
                reasons.append(f"近期仍有市场反应（{last_reaction:.2%}），额外加分 +{bonus:.0f}")

        return min(max(raw_score, 0), 100.0), reasons, penalties

    # -----------------------------------------------------------------------
    # 辅助方法
    # -----------------------------------------------------------------------

    def _make_detail(
        self, raw_score: float, dim_name: str, reasons: List[str], penalties: List[str]
    ) -> Dict:
        """将原始分数转换为加权分数，并组织输出结构。"""
        weight = self.weights.get(dim_name, 0)
        weighted = raw_score * weight / 100
        return {
            "raw": round(raw_score, 2),
            "weight": round(weight, 2),
            "weighted": round(weighted, 2),
            "reasons": reasons,
            "penalties": penalties,
        }

    def _check_threshold(self, total_score: float) -> bool:
        """根据过滤模式判断总分是否通过阈值。"""
        if self.filtering_mode == "fixed":
            return total_score >= self.fixed_threshold
        elif self.filtering_mode == "percentile":
            # 百分位模式需要在 score_batch 中处理，单次评分默认通过
            return True
        elif self.filtering_mode == "both":
            return total_score >= self.fixed_threshold
        return True

    def _bypass_result(self, zone: Dict) -> Dict:
        """当评分模块被禁用时，返回默认通过结果。"""
        return {
            "total_score": 100.0,
            "passed_threshold": True,
            "threshold_type": "disabled",
            "threshold_value": 0,
            "details": {},
            "bonus_reasons": ["评分模块已禁用，自动通过"],
            "penalty_reasons": [],
            "allow_entry_check": True,
            "zone": zone,
        }


# ---------------------------------------------------------------------------
# 假突破/假跌破联动检测
# ---------------------------------------------------------------------------

class FakeoutDetector:
    """
    假突破/假跌破检测器。

    与 ConsensusScorer 联动使用：
      - 只对高共识区域（score >= min_score_for_fakeout_bonus）检测假突破
      - 假突破确认后，返回信号增强标志
    """

    def __init__(self, config: Dict):
        fk_cfg = config.get("consensus_score", {}).get("fakeout", {})
        self.enabled = fk_cfg.get("enabled", True)
        self.min_score = fk_cfg.get("min_score_for_fakeout_bonus", 65)
        self.recovery_candles = fk_cfg.get("fakeout_recovery_candles", 3)
        self.max_breach_pct = fk_cfg.get("fakeout_max_breach_pct", 0.008)
        self.signal_boost = fk_cfg.get("fakeout_signal_boost", 1.2)

    def check(self, score_result: Dict, df_5m: pd.DataFrame) -> Dict:
        """
        检查高共识支撑区是否出现假跌破信号。

        :param score_result: ConsensusScorer.score() 的返回值
        :param df_5m:        5M周期K线数据
        :return: 包含假突破检测结果的字典
        """
        result = {
            "fakeout_detected": False,
            "fakeout_type": None,       # "fakeout_long" / "fakeout_short"
            "signal_boost": 1.0,
            "reason": "",
        }

        if not self.enabled:
            return result

        total_score = score_result.get("total_score", 0)
        if total_score < self.min_score:
            result["reason"] = f"共识评分{total_score:.1f}低于假突破检测门槛{self.min_score}"
            return result

        zone = score_result.get("zone", {})
        if not zone:
            return result

        zone_type = zone.get("zone_type", "support")
        zone_bottom = zone["price_start"]
        zone_top = zone["price_end"]

        if len(df_5m) < self.recovery_candles + 2:
            return result

        # 检查最近K线是否出现假跌破（针对支撑区）
        if zone_type == "support":
            recent = df_5m.iloc[-(self.recovery_candles + 2):]
            # 寻找跌破支撑区下沿的K线
            breach_candles = recent[recent["low"] < zone_bottom * (1 - 0.0001)]
            if len(breach_candles) == 0:
                return result

            # 检查跌破幅度是否在允许范围内
            min_low = breach_candles["low"].min()
            breach_pct = (zone_bottom - min_low) / zone_bottom
            if breach_pct > self.max_breach_pct:
                result["reason"] = (
                    f"跌破幅度过大（{breach_pct:.2%} > {self.max_breach_pct:.2%}），"
                    f"视为真跌破，降低做多意愿"
                )
                result["fakeout_type"] = "true_breakdown"
                return result

            # 检查是否在 recovery_candles 内收回区域
            last_close = df_5m["close"].iloc[-1]
            if last_close >= zone_bottom:
                result["fakeout_detected"] = True
                result["fakeout_type"] = "fakeout_long"
                result["signal_boost"] = self.signal_boost
                result["reason"] = (
                    f"高共识支撑区（评分{total_score:.1f}）假跌破后迅速收回，"
                    f"跌破幅度{breach_pct:.2%}，入场信号增强×{self.signal_boost}"
                )

        return result
