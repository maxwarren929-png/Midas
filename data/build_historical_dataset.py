#!/usr/bin/env python3
"""
Build canonical historical market dataset (2003 - 2023) across a multi-asset sector basket:
SPY (S&P 500), QQQ (Nasdaq), TLT (20+ Yr Treasuries), IWM (Russell 2000), XLE (Energy).
Schema: date,asset,open,high,low,close,adjusted_close,volume
"""

import csv
import datetime
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "historical"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "market_2003_2023.csv"

TICKERS = ["SPY", "QQQ", "TLT", "IWM", "XLE"]

def fetch_ticker_data(ticker: str):
    print(f"Fetching {ticker} from Yahoo Finance...")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=25y"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    
    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])
    volumes = quote.get("volume", [])
    
    bars = {}
    for i, ts in enumerate(timestamps):
        date_str = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%d")
        if "2003-01-01" <= date_str <= "2023-12-31":
            o = opens[i] if i < len(opens) else None
            h = highs[i] if i < len(highs) else None
            l = lows[i] if i < len(lows) else None
            c = closes[i] if i < len(closes) else None
            v = volumes[i] if i < len(volumes) else None
            
            if None not in (o, h, l, c, v) and min(o, h, l, c) > 0 and v > 0:
                bars[date_str] = {
                    "open": round(o, 4),
                    "high": round(h, 4),
                    "low": round(l, 4),
                    "close": round(c, 4),
                    "adjusted_close": round(c, 4),
                    "volume": round(float(v), 2),
                }
    print(f"  --> {len(bars)} valid trading days for {ticker}")
    return bars

def main():
    ticker_bars = {}
    for t in TICKERS:
        ticker_bars[t] = fetch_ticker_data(t)
        time.sleep(1)
        
    # Find common dates across all assets
    common_dates = sorted(set.intersection(*[set(b.keys()) for b in ticker_bars.values()]))
    print(f"\nTotal aligned trading days across all {len(TICKERS)} assets: {len(common_dates)}")
    print(f"Start: {common_dates[0]} | End: {common_dates[-1]}")
    
    rows = []
    for d in common_dates:
        for t in TICKERS:
            b = ticker_bars[t][d]
            rows.append({
                "date": d,
                "asset": t,
                "open": b["open"],
                "high": b["high"],
                "low": b["low"],
                "close": b["close"],
                "adjusted_close": b["adjusted_close"],
                "volume": b["volume"]
            })
            
    with OUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "asset", "open", "high", "low", "close", "adjusted_close", "volume"])
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Successfully wrote {len(rows)} canonical rows to {OUT_FILE}")

if __name__ == "__main__":
    main()
