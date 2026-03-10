"""
support_zone.py — 支撑/压力区识别模块

职责：
  - 从指定周期的K线数据中识别候选支撑区和压力区
  - 输出标准化的区域对象列表，供 consensus_score.py 进行评分
  - 不做任何评分判断，只负责"找出候选区域"

区域识别逻辑：
  1. 识别显著的摆动高点（Swing High）和摆动低点（Swing Low）
  2. 将相邻且价格接近的高/低点合并为"区域"（Zone）
  3. 识别大箱体边界（可选）
  4. 输出标准化的区域字典列表
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# 数据结构定义
# ---------------------------------------------------------------------------

def make_zone(
    price_start: float,
    price_end: float,
    zone_type: str,
    formed_at_index: int,
    formed_at_time: Optional[datetime],
    touch_count: int = 1,
    source: str = "swing",
    prominence: float = 0.0,
) -> Dict:
    """
    创建标准化的区域字典。

    :param price_start: 区域下沿价格
    :param price_end:   区域上沿价格
    :param zone_type:   "support" 或 "resistance"
    :param formed_at_index: 区域形成时的K线索引（用于新鲜度计算）
    :param formed_at_time:  区域形成时间
    :param touch_count:     历史触碰次数（初始为1）
    :param source:          区域来源标签（swing / box / manual）
    :param prominence:      显著性得分（用于结构显著性评分维度）
    :return: 标准化区域字典
    """
    mid = (price_start + price_end) / 2.0
    return {
        "price_start": min(price_start, price_end),
        "price_end": max(price_start, price_end),
        "mid_price": mid,
        "zone_type": zone_type,
        "formed_at_index": formed_at_index,
        "formed_at_time": formed_at_time,
        "touch_count": touch_count,
        "source": source,
        "prominence": prominence,
        "reactions": [],   # 每次触碰后的价格反应记录，供 retest 评分使用
    }


# ---------------------------------------------------------------------------
# 核心识别类
# ---------------------------------------------------------------------------

class SupportZoneDetector:
    """
    支撑/压力区识别器。

    使用方式：
        detector = SupportZoneDetector(config)
        zones = detector.detect(df_15m)
    """

    def __init__(self, config: Dict):
        sz_cfg = config.get("support_zone", {})
        self.lookback = sz_cfg.get("lookback_candles", 100)
        self.merge_threshold = sz_cfg.get("zone_merge_threshold_pct", 0.003)
        self.prominence_pct = sz_cfg.get("swing_prominence_pct", 0.005)
        self.min_width_pct = sz_cfg.get("min_zone_width_pct", 0.001)
        self.max_width_pct = sz_cfg.get("max_zone_width_pct", 0.015)
        self.touch_tolerance = sz_cfg.get("zone_touch_tolerance_pct", 0.002)

        # 箱体识别参数
        struct_cfg = config.get("consensus_score", {}).get("structure", {})
        self.box_enabled = struct_cfg.get("box_detection_enabled", True)
        self.box_min_width = struct_cfg.get("box_min_width_candles", 10)

    def detect(self, df: pd.DataFrame, current_price: Optional[float] = None) -> List[Dict]:
        """
        主入口：对输入的K线 DataFrame 进行区域识别。

        :param df: 包含 open/high/low/close/volume 列的 DataFrame，
                   index 为时间序列（DatetimeIndex）
        :param current_price: 当前价格（用于过滤距离过远的区域）
        :return: 标准化区域字典列表
        """
        if len(df) < 20:
            return []

        # 只取最近 lookback 根K线
        df_slice = df.iloc[-self.lookback:].copy().reset_index(drop=False)
        if "index" in df_slice.columns:
            df_slice = df_slice.rename(columns={"index": "time"})

        zones = []

        # Step 1: 识别摆动高低点
        swing_highs, swing_lows = self._find_swing_points(df_slice)

        # Step 2: 将摆动点转换为区域
        for sh in swing_highs:
            z = self._swing_to_zone(df_slice, sh, "resistance")
            if z:
                zones.append(z)

        for sl in swing_lows:
            z = self._swing_to_zone(df_slice, sl, "support")
            if z:
                zones.append(z)

        # Step 3: 识别箱体边界（可选）
        if self.box_enabled:
            box_zones = self._detect_box_zones(df_slice)
            zones.extend(box_zones)

        # Step 4: 合并相近区域
        zones = self._merge_zones(zones)

        # Step 5: 统计每个区域的历史触碰次数和反应
        zones = self._count_touches_and_reactions(df_slice, zones)

        # Step 6: 过滤距离当前价格过远的区域（可选）
        if current_price:
            zones = self._filter_by_proximity(zones, current_price, max_pct=0.15)

        # Step 7: 按区域中心价格排序
        zones.sort(key=lambda z: z["mid_price"])

        return zones

    # -----------------------------------------------------------------------
    # 私有方法：摆动高低点识别
    # -----------------------------------------------------------------------

    def _find_swing_points(self, df: pd.DataFrame):
        """
        使用左右各 N 根K线的比较方法识别摆动高低点。
        显著性过滤：摆动点与左右邻近K线的价格差必须超过 prominence_pct。
        """
        n = len(df)
        window = 3  # 左右各3根K线确认摆动点

        swing_highs = []
        swing_lows = []

        for i in range(window, n - window):
            high_i = df.iloc[i]["high"]
            low_i = df.iloc[i]["low"]

            # 检查是否为摆动高点
            left_highs = df.iloc[i - window:i]["high"].values
            right_highs = df.iloc[i + 1:i + window + 1]["high"].values

            if high_i > max(left_highs) and high_i > max(right_highs):
                # 计算显著性：与左右最高点的差值
                prominence = min(
                    high_i - max(left_highs),
                    high_i - max(right_highs)
                ) / high_i
                if prominence >= self.prominence_pct:
                    swing_highs.append({
                        "index": i,
                        "price": high_i,
                        "prominence": prominence,
                        "time": df.iloc[i].get("time", None),
                    })

            # 检查是否为摆动低点
            left_lows = df.iloc[i - window:i]["low"].values
            right_lows = df.iloc[i + 1:i + window + 1]["low"].values

            if low_i < min(left_lows) and low_i < min(right_lows):
                prominence = min(
                    min(left_lows) - low_i,
                    min(right_lows) - low_i
                ) / low_i
                if prominence >= self.prominence_pct:
                    swing_lows.append({
                        "index": i,
                        "price": low_i,
                        "prominence": prominence,
                        "time": df.iloc[i].get("time", None),
                    })

        return swing_highs, swing_lows

    def _swing_to_zone(self, df: pd.DataFrame, swing_point: Dict, zone_type: str) -> Optional[Dict]:
        """
        将单个摆动点扩展为区域（使用对应K线的实体范围作为区域宽度）。
        """
        idx = swing_point["index"]
        candle = df.iloc[idx]
        price = swing_point["price"]

        open_c = candle["open"]
        close_c = candle["close"]
        body_top = max(open_c, close_c)
        body_bottom = min(open_c, close_c)

        if zone_type == "resistance":
            # 压力区：从实体顶部到最高点
            p_start = body_top
            p_end = price  # high
        else:
            # 支撑区：从最低点到实体底部
            p_start = price  # low
            p_end = body_bottom

        # 宽度校验
        width_pct = abs(p_end - p_start) / price
        if width_pct < self.min_width_pct:
            # 区域太窄，用最小宽度扩展
            half = price * self.min_width_pct / 2
            p_start = price - half
            p_end = price + half
        elif width_pct > self.max_width_pct:
            # 区域太宽，截断
            if zone_type == "resistance":
                p_start = price * (1 - self.max_width_pct)
            else:
                p_end = price * (1 + self.max_width_pct)

        time_val = swing_point.get("time", None)
        if hasattr(time_val, "to_pydatetime"):
            time_val = time_val.to_pydatetime()

        return make_zone(
            price_start=p_start,
            price_end=p_end,
            zone_type=zone_type,
            formed_at_index=idx,
            formed_at_time=time_val,
            prominence=swing_point.get("prominence", 0.0),
            source="swing",
        )

    # -----------------------------------------------------------------------
    # 私有方法：箱体识别
    # -----------------------------------------------------------------------

    def _detect_box_zones(self, df: pd.DataFrame) -> List[Dict]:
        """
        识别大箱体结构：寻找价格在一定范围内横盘震荡的区间，
        提取其上沿和下沿作为区域。
        """
        box_zones = []
        n = len(df)
        if n < self.box_min_width:
            return box_zones

        # 滑动窗口检测箱体
        for start in range(0, n - self.box_min_width, 5):
            end = min(start + self.box_min_width * 2, n)
            window_df = df.iloc[start:end]

            box_high = window_df["high"].max()
            box_low = window_df["low"].min()
            box_range_pct = (box_high - box_low) / box_low

            # 箱体条件：价格范围在 1% ~ 8% 之间
            if 0.01 <= box_range_pct <= 0.08:
                # 检查是否真的是横盘（大多数K线在箱体内）
                in_box = window_df[
                    (window_df["high"] <= box_high * 1.002) &
                    (window_df["low"] >= box_low * 0.998)
                ]
                if len(in_box) / len(window_df) >= 0.75:
                    # 箱体上沿 -> 压力区
                    box_zones.append(make_zone(
                        price_start=box_high * (1 - self.min_width_pct),
                        price_end=box_high,
                        zone_type="resistance",
                        formed_at_index=start,
                        formed_at_time=df.iloc[start].get("time", None),
                        source="box",
                        prominence=box_range_pct,
                    ))
                    # 箱体下沿 -> 支撑区
                    box_zones.append(make_zone(
                        price_start=box_low,
                        price_end=box_low * (1 + self.min_width_pct),
                        zone_type="support",
                        formed_at_index=start,
                        formed_at_time=df.iloc[start].get("time", None),
                        source="box",
                        prominence=box_range_pct,
                    ))

        return box_zones

    # -----------------------------------------------------------------------
    # 私有方法：区域合并
    # -----------------------------------------------------------------------

    def _merge_zones(self, zones: List[Dict]) -> List[Dict]:
        """
        将价格中心相近的同类型区域合并，避免重复区域。
        合并条件：两个区域的中心价格差 < merge_threshold_pct。
        """
        if not zones:
            return zones

        # 按类型分组处理
        merged = []
        for zone_type in ["support", "resistance"]:
            type_zones = [z for z in zones if z["zone_type"] == zone_type]
            type_zones.sort(key=lambda z: z["mid_price"])

            if not type_zones:
                continue

            current_group = [type_zones[0]]
            for z in type_zones[1:]:
                last = current_group[-1]
                price_diff_pct = abs(z["mid_price"] - last["mid_price"]) / last["mid_price"]
                if price_diff_pct <= self.merge_threshold:
                    current_group.append(z)
                else:
                    merged.append(self._merge_group(current_group))
                    current_group = [z]
            merged.append(self._merge_group(current_group))

        return merged

    def _merge_group(self, group: List[Dict]) -> Dict:
        """将一组相近区域合并为一个区域，取价格范围的并集。"""
        if len(group) == 1:
            return group[0]

        p_start = min(z["price_start"] for z in group)
        p_end = max(z["price_end"] for z in group)
        # 取最早形成的时间
        earliest = min(group, key=lambda z: z["formed_at_index"])
        # 取最高显著性
        max_prominence = max(z["prominence"] for z in group)

        return make_zone(
            price_start=p_start,
            price_end=p_end,
            zone_type=group[0]["zone_type"],
            formed_at_index=earliest["formed_at_index"],
            formed_at_time=earliest["formed_at_time"],
            touch_count=sum(z["touch_count"] for z in group),
            source=group[0]["source"],
            prominence=max_prominence,
        )

    # -----------------------------------------------------------------------
    # 私有方法：触碰次数统计
    # -----------------------------------------------------------------------

    def _count_touches_and_reactions(self, df: pd.DataFrame, zones: List[Dict]) -> List[Dict]:
        """
        统计每个区域在K线历史中被触碰的次数，
        以及每次触碰后的价格反应幅度（用于 retest 评分维度）。
        """
        for zone in zones:
            touches = []
            reactions = []

            for i in range(len(df)):
                candle = df.iloc[i]
                low_i = candle["low"]
                high_i = candle["high"]
                close_i = candle["close"]

                # 判断是否触碰区域
                touched = False
                if zone["zone_type"] == "support":
                    if low_i <= zone["price_end"] * (1 + self.touch_tolerance) and \
                       low_i >= zone["price_start"] * (1 - self.touch_tolerance):
                        touched = True
                else:
                    if high_i >= zone["price_start"] * (1 - self.touch_tolerance) and \
                       high_i <= zone["price_end"] * (1 + self.touch_tolerance):
                        touched = True

                if touched:
                    touches.append(i)
                    # 计算触碰后的反应（后续3根K线的最大涨幅/跌幅）
                    if i + 3 < len(df):
                        future_slice = df.iloc[i + 1:i + 4]
                        if zone["zone_type"] == "support":
                            reaction = (future_slice["high"].max() - close_i) / close_i
                        else:
                            reaction = (close_i - future_slice["low"].min()) / close_i
                        reactions.append(max(reaction, 0.0))

            zone["touch_count"] = len(touches)
            zone["reactions"] = reactions

        return zones

    # -----------------------------------------------------------------------
    # 私有方法：按距离过滤
    # -----------------------------------------------------------------------

    def _filter_by_proximity(self, zones: List[Dict], current_price: float, max_pct: float = 0.15) -> List[Dict]:
        """过滤掉距离当前价格过远的区域。"""
        return [
            z for z in zones
            if abs(z["mid_price"] - current_price) / current_price <= max_pct
        ]


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def get_nearest_support(zones: List[Dict], current_price: float) -> Optional[Dict]:
    """获取当前价格下方最近的支撑区。"""
    supports = [z for z in zones if z["zone_type"] == "support" and z["mid_price"] < current_price]
    if not supports:
        return None
    return max(supports, key=lambda z: z["mid_price"])


def get_nearest_resistance(zones: List[Dict], current_price: float) -> Optional[Dict]:
    """获取当前价格上方最近的压力区。"""
    resistances = [z for z in zones if z["zone_type"] == "resistance" and z["mid_price"] > current_price]
    if not resistances:
        return None
    return min(resistances, key=lambda z: z["mid_price"])


def price_in_zone(price: float, zone: Dict, tolerance_pct: float = 0.002) -> bool:
    """判断价格是否在区域内（含容差）。"""
    return (
        zone["price_start"] * (1 - tolerance_pct) <= price <=
        zone["price_end"] * (1 + tolerance_pct)
    )
