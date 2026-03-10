"""
binance0310_180d.py — Binance Futures BTCUSDT 永续合约历史 K 线数据下载脚本
（180 天完整版）

使用方法:
  1. 确保已安装依赖: pip install requests pandas
  2. 在您的本地电脑上运行: python binance0310_180d.py
  3. 运行完成后，将生成的 4 个 CSV 文件上传给 Manus

输出文件（保存在脚本同目录下）:
  BTCUSDT_5m.csv   — 5分钟 K线，最近 180 天
  BTCUSDT_15m.csv  — 15分钟 K线，最近 180 天
  BTCUSDT_1h.csv   — 1小时 K线，最近 365 天
  BTCUSDT_4h.csv   — 4小时 K线，最近 365 天

数据来源: Binance Futures API (fapi.binance.com)
合约品种: BTCUSDT 永续合约 (BTCUSDT PERP)
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# ─────────────────────────────────────────
#  配置区（如需修改下载范围，在此调整）
# ─────────────────────────────────────────
SYMBOL = "BTCUSDT"
BASE_URL = "https://fapi.binance.com"

DOWNLOAD_TASKS = [
    {"interval": "5m",  "days": 180, "filename": "BTCUSDT_5m.csv"},
    {"interval": "15m", "days": 180, "filename": "BTCUSDT_15m.csv"},
    {"interval": "1h",  "days": 365, "filename": "BTCUSDT_1h.csv"},
    {"interval": "4h",  "days": 365, "filename": "BTCUSDT_4h.csv"},
]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
# ─────────────────────────────────────────


def fetch_klines(interval: str, start_ms: int, end_ms: int, limit: int = 1500):
    """从 Binance Futures API 获取一批 K 线数据"""
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  [警告] 请求失败 (尝试 {attempt + 1}/3): {e}")
            time.sleep(3)
    return []


def download(interval: str, days: int, filename: str):
    """下载指定周期的完整历史数据"""
    print(f"\n{'='*55}")
    print(f"  下载: {SYMBOL} {interval} K线 (最近 {days} 天)")
    print(f"{'='*55}")

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000
    end_ms = now_ms

    all_rows = []
    current_start = start_ms
    page = 0

    while current_start < end_ms:
        page += 1
        rows = fetch_klines(interval, current_start, end_ms, limit=1500)

        if not rows:
            print(f"  第 {page} 页: 无数据，停止")
            break

        all_rows.extend(rows)
        last_ts = int(rows[-1][0])

        if page % 5 == 0:
            dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
            print(f"  第 {page} 页: 累计 {len(all_rows)} 根，当前进度至 {dt.strftime('%Y-%m-%d %H:%M')}")

        # 翻页：下一页从当前最后一根的下一毫秒开始
        current_start = last_ts + 1

        # 如果返回数量少于 limit，说明已到最新，停止
        if len(rows) < 1500:
            break

        time.sleep(0.1)  # 避免触发频率限制

    if not all_rows:
        print(f"  [错误] 未获取到任何数据，请检查网络或 API 是否可访问")
        return

    # 转换为 DataFrame
    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype(int), unit="ms", utc=True
    )
    df = df.set_index("timestamp")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    # 去重 & 排序
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # 只保留目标时间范围
    target_start = pd.Timestamp.fromtimestamp(start_ms / 1000, tz="UTC")
    df = df[df.index >= target_start]

    # 保存
    output_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(output_path)

    start_str = df.index[0].strftime("%Y-%m-%d")
    end_str   = df.index[-1].strftime("%Y-%m-%d")
    print(f"\n  完成: {len(df)} 根 K线")
    print(f"  时间范围: {start_str} ~ {end_str}")
    print(f"  已保存至: {output_path}")


def main():
    print("\n" + "=" * 55)
    print("  Binance Futures BTCUSDT 历史 K 线数据下载工具")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    print(f"\n  数据来源: Binance Futures API (fapi.binance.com)")
    print(f"  合约品种: {SYMBOL} 永续合约")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"\n  本次下载范围：")
    for task in DOWNLOAD_TASKS:
        print(f"    {task['interval']:>4s}  最近 {task['days']:>3d} 天  →  {task['filename']}")

    for task in DOWNLOAD_TASKS:
        download(
            interval=task["interval"],
            days=task["days"],
            filename=task["filename"],
        )

    print("\n" + "=" * 55)
    print("  全部下载完成！")
    print("  请将以下 4 个文件上传给 Manus：")
    for task in DOWNLOAD_TASKS:
        print(f"    - {task['filename']}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
