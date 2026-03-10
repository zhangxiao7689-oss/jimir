"""
download_okx_data.py — 从 OKX 公开 API 下载 BTC-USDT-SWAP 永续合约历史 K 线数据

OKX API 文档: https://www.okx.com/docs-v5/en/#order-book-trading-market-data-get-candlesticks-history
接口: GET /api/v5/market/history-candles
限制: 每次最多 100 根，通过翻页获取全量数据

输出文件:
  data/BTCUSDT_15m.csv   — 15分钟 K线（约 6 个月）
  data/BTCUSDT_1h.csv    — 1小时 K线
  data/BTCUSDT_5m.csv    — 5分钟 K线（约 3 个月）
  data/BTCUSDT_4h.csv    — 4小时 K线
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

BASE_URL = "https://www.okx.com"
INST_ID  = "BTC-USDT-SWAP"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# OKX bar 参数映射
BAR_MAP = {
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1H",
    "4h":  "4H",
}

# 目标下载天数
DAYS_MAP = {
    "5m":  90,
    "15m": 180,
    "1h":  365,
    "4h":  365,
}


def fetch_candles(bar: str, after_ms: int = None, before_ms: int = None, limit: int = 100):
    """获取一页 K 线数据"""
    url = f"{BASE_URL}/api/v5/market/history-candles"
    params = {
        "instId": INST_ID,
        "bar": BAR_MAP[bar],
        "limit": limit,
    }
    if after_ms:
        params["after"] = str(after_ms)
    if before_ms:
        params["before"] = str(before_ms)

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            if data.get("code") == "0":
                return data.get("data", [])
            else:
                print(f"  [警告] API 返回错误: {data.get('msg')} (尝试 {attempt+1}/3)")
                time.sleep(2)
        except Exception as e:
            print(f"  [错误] 请求失败: {e} (尝试 {attempt+1}/3)")
            time.sleep(3)
    return []


def download_bar(bar: str, days: int) -> pd.DataFrame:
    """下载指定周期的全量历史数据"""
    print(f"\n[下载] {INST_ID} {bar} K线 (最近 {days} 天)...")

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    target_ms = now_ms - days * 24 * 3600 * 1000

    all_rows = []
    after_ms = None  # 从最新开始往前翻页
    page = 0

    while True:
        page += 1
        rows = fetch_candles(bar, after_ms=after_ms, limit=100)

        if not rows:
            print(f"  第 {page} 页: 无数据，停止")
            break

        # OKX 返回格式: [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
        # 按时间降序排列（最新在前）
        all_rows.extend(rows)

        oldest_ts = int(rows[-1][0])
        newest_ts = int(rows[0][0])

        if page % 10 == 0:
            oldest_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=timezone.utc)
            print(f"  第 {page} 页: {len(all_rows)} 根，最早: {oldest_dt.strftime('%Y-%m-%d %H:%M')}")

        if oldest_ts <= target_ms:
            print(f"  已达到目标时间范围，停止")
            break

        # 翻页：after 设为当前页最旧的时间戳
        after_ms = oldest_ts
        time.sleep(0.2)  # 避免触发频率限制

    if not all_rows:
        print(f"  [错误] 未获取到任何数据")
        return pd.DataFrame()

    # 转换为 DataFrame
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume", "volCcy", "volQuote", "confirm"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df.sort_index()

    # 只保留目标时间范围内的数据
    target_dt = pd.Timestamp.fromtimestamp(target_ms / 1000, tz="UTC")
    df = df[df.index >= target_dt]

    # 去重
    df = df[~df.index.duplicated(keep="last")]

    print(f"  完成: {len(df)} 根 | {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    return df


def main():
    print("=" * 60)
    print(f"  OKX {INST_ID} 历史 K 线数据下载")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 下载各周期数据
    results = {}
    for bar, days in DAYS_MAP.items():
        df = download_bar(bar, days)
        if not df.empty:
            results[bar] = df

    # 保存文件（文件名与现有代码兼容）
    file_map = {
        "5m":  "BTCUSDT_5m.csv",
        "15m": "BTCUSDT_15m.csv",
        "1h":  "BTCUSDT_1h.csv",
        "4h":  "BTCUSDT_4h.csv",
    }

    print("\n[保存]")
    for bar, filename in file_map.items():
        if bar in results:
            path = os.path.join(DATA_DIR, filename)
            results[bar].to_csv(path)
            print(f"  {filename}: {len(results[bar])} 根 -> {path}")

    print("\n[完成] 所有数据已保存至 data/ 目录")
    print("  数据来源: OKX BTC-USDT-SWAP 永续合约（真实期货数据）")


if __name__ == "__main__":
    main()
