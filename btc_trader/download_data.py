"""
download_data.py — 从 Binance 现货公开 API 下载 BTCUSDT 历史 K 线数据

使用现货 API（api.binance.com），无需 API Key。
BTCUSDT 现货与永续合约价格高度相关，适合用于策略回测。
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timezone

BASE_URL = "https://api.binance.com"
SYMBOL = "BTCUSDT"
DATA_DIR = "/home/ubuntu/btc_trader/data"
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """分页拉取 K 线数据（每次最多 1000 根）。"""
    all_klines = []
    current_start = start_ms
    while current_start < end_ms:
        url = f"{BASE_URL}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_klines.extend(data)
        current_start = data[-1][6] + 1
        print(f"  [{interval}] 已下载 {len(all_klines)} 根 K 线...", flush=True)
        time.sleep(0.2)
        if len(data) < 1000:
            break
    return all_klines


def klines_to_df(klines: list) -> pd.DataFrame:
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def download(interval: str, days: int) -> str:
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000
    print(f"\n正在下载 {SYMBOL} {interval} 数据（最近 {days} 天）...", flush=True)
    klines = fetch_klines(SYMBOL, interval, start_ms, end_ms)
    df = klines_to_df(klines)
    filepath = os.path.join(DATA_DIR, f"BTCUSDT_{interval}.csv")
    df.to_csv(filepath)
    print(f"  ✅ 保存完成: {filepath} ({len(df)} 根 K 线)", flush=True)
    return filepath


if __name__ == "__main__":
    download("15m", days=365)
    download("1h",  days=365)
    download("5m",  days=90)
    print("\n所有数据下载完成！")
