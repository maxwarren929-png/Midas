#!/usr/bin/env python3
"""Generate the frozen, offline Prompt 3B multi-regime market."""

from __future__ import annotations

import csv
import hashlib
import math
import random
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "prompt3b"
N_DAYS = 1800
SEED = 3_025_071
ASSETS = ("SPY", "AAA", "BBB", "CCC")
START = date(2010, 1, 4)


def business_days(start: date, count: int) -> list[date]:
    result: list[date] = []
    current = start
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return result


def regime_for(index: int) -> tuple[str, float, float, float]:
    if index < 300:
        return "broad_rising", 0.00035, 0.004, 0.55
    if index < 600:
        return "broad_falling", -0.00030, 0.006, 0.65
    if index < 900:
        return "sideways_low_volatility", 0.0, 0.002, 0.10
    if index < 1200:
        return "sideways_high_volatility", 0.0, 0.012, -0.20
    if index < 1500:
        return "asset_momentum", 0.00010, 0.007, 0.25
    return "asset_mean_reversion", 0.0, 0.005, 0.15


def idiosyncratic_regime_bias(regime: str, asset_index: int) -> float:
    # The falling and high-volatility regimes deliberately separate asset
    # paths. This creates a realistic cross-sectional survival test without
    # exposing regime labels to policies.
    if regime == "broad_falling":
        return (0.0018, -0.0040, 0.0025, -0.0012)[asset_index]
    if regime == "sideways_high_volatility":
        return (0.0004, -0.0010, 0.0008, -0.0005)[asset_index]
    return 0.0


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    dates = business_days(START, N_DAYS)
    prices = [100.0, 82.0, 126.0, 64.0]
    momentum = [0.0 for _ in ASSETS]
    rows: list[dict[str, object]] = []
    regimes: list[dict[str, object]] = []
    asset_bias = [0.0, 0.00026, -0.00020, 0.00010]
    mean_reversion_targets = [100.0, 82.0, 126.0, 64.0]

    for t, trading_date in enumerate(dates):
        regime, common_drift, volatility, correlation = regime_for(t)
        regimes.append({"date": trading_date.isoformat(), "regime": regime, "index": t})
        common_shock = rng.gauss(0.0, volatility)
        for asset_index, asset in enumerate(ASSETS):
            open_noise = rng.gauss(0.0, volatility * 0.18)
            opening = prices[asset_index] * math.exp(open_noise)
            if regime == "asset_momentum":
                momentum[asset_index] = 0.82 * momentum[asset_index] + rng.gauss(
                    asset_bias[asset_index], volatility * 0.55
                )
                idiosyncratic = momentum[asset_index]
            elif regime == "asset_mean_reversion":
                gap = math.log(mean_reversion_targets[asset_index] / prices[asset_index])
                idiosyncratic = 0.0045 * gap + rng.gauss(0.0, volatility * 0.8)
            else:
                idiosyncratic = rng.gauss(0.0, volatility * 0.75)
            idiosyncratic += idiosyncratic_regime_bias(regime, asset_index)
            log_return = common_drift + correlation * common_shock + idiosyncratic
            closing = opening * math.exp(log_return)
            high = max(opening, closing) * (1.0 + abs(rng.gauss(0.0, volatility * 0.35)))
            low = min(opening, closing) * (1.0 - abs(rng.gauss(0.0, volatility * 0.35)))
            low = max(low, 0.01)
            volume = 1_000_000.0 * (1.0 + abs(rng.gauss(0.0, 0.20)))
            rows.append(
                {
                    "date": trading_date.isoformat(),
                    "asset": asset,
                    "open": f"{opening:.8f}",
                    "high": f"{high:.8f}",
                    "low": f"{low:.8f}",
                    "close": f"{closing:.8f}",
                    "adjusted_close": f"{closing:.8f}",
                    "volume": f"{volume:.2f}",
                }
            )
            prices[asset_index] = closing

    with (DATA / "market.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (DATA / "regimes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "regime", "index"])
        writer.writeheader()
        writer.writerows(regimes)

    digest = hashlib.sha256((DATA / "market.csv").read_bytes()).hexdigest()
    (DATA / "market.sha256").write_text(f"{digest}  market.csv\n", encoding="ascii")
    print(f"rows={len(rows)} dates={len(dates)} assets={len(ASSETS)} sha256={digest}")


if __name__ == "__main__":
    main()
