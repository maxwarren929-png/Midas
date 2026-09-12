#!/usr/bin/env python3
"""
Fetch 16-Asset Macro Universe (Equities, Leveraged 3x Long/Short, Commodities, Crypto, Fixed Income):
- Core Equities: SPY, QQQ, IWM
- Mega-Cap Leaders: NVDA, AAPL, MSFT
- 3x Leveraged Growth: TQQQ, SOXL, UPRO
- 3x Leveraged Inverse (Crash Hunters): SQQQ, SPXU
- Commodities & Metals: GLD, SLV, XLE
- Fixed Income: TLT
- Crypto: BTC-USD
"""

import csv
import datetime
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "macro"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "macro_16_assets_2015_2026.csv"

TICKERS = [
    "SPY", "QQQ", "IWM",
    "NVDA", "AAPL", "MSFT",
    "TQQQ", "SOXL", "UPRO",
    "SQQQ", "SPXU",
    "GLD", "SLV", "XLE",
    "TLT", "BTC-USD"
]

def fetch_ticker_data(ticker: str):
    print(f"Fetching {ticker} from Yahoo Finance...")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=15y"
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
        if date_str >= "2015-01-01":
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
    print(f"  --> {len(bars)} valid daily bars for {ticker} (Last: {sorted(bars.keys())[-1] if bars else 'None'})")
    return bars

def main():
    ticker_bars = {}
    for t in TICKERS:
        ticker_bars[t] = fetch_ticker_data(t)
        time.sleep(0.5)
        
    common_dates = sorted(set.intersection(*[set(b.keys()) for b in ticker_bars.values()]))
    print(f"\nTotal aligned trading days across all {len(TICKERS)} assets: {len(common_dates)}")
    print(f"Start Date: {common_dates[0]} | Most Recent Date: {common_dates[-1]}")
    
    # Check for BTC ticker rename (Midas prefers clean alphanumeric like BTC or BTCUSD)
    clean_ticker_map = {t: ("BTC" if t == "BTC-USD" else t) for t in TICKERS}
    
    rows = []
    for d in common_dates:
        for t in TICKERS:
            b = ticker_bars[t][d]
            rows.append({
                "date": d,
                "asset": clean_ticker_map[t],
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
