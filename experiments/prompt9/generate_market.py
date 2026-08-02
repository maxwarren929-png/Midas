#!/usr/bin/env python3
"""Generate a deterministic synthetic market from a versioned JSON specification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any


CSV_FIELDS = ("date", "asset", "open", "high", "low", "close", "adjusted_close", "volume")


def business_days(start: date, count: int) -> list[date]:
    result: list[date] = []
    current = start
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return result


def require_number(value: Any, name: str, minimum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("version") != 1:
        raise ValueError("generator version must be 1")
    for key in ("name", "regime_family", "generation_timestamp"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if not isinstance(spec.get("seed"), int) or isinstance(spec["seed"], bool):
        raise ValueError("seed must be an integer")
    try:
        date.fromisoformat(spec["start_date"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("start_date must be an ISO date") from error
    days = spec.get("days")
    if not isinstance(days, int) or isinstance(days, bool) or days < 2:
        raise ValueError("days must be an integer of at least 2")
    assets = spec.get("assets")
    if not isinstance(assets, list) or len(assets) < 2:
        raise ValueError("assets must contain at least two entries")
    symbols: set[str] = set()
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise ValueError(f"assets[{index}] must be an object")
        symbol = asset.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip() or symbol in symbols:
            raise ValueError("asset symbols must be unique non-empty strings")
        symbols.add(symbol)
        require_number(asset.get("start_price"), f"assets[{index}].start_price", 0.01)
        require_number(asset.get("drift_offset", 0.0), f"assets[{index}].drift_offset")
    volume = spec.get("volume")
    if not isinstance(volume, dict):
        raise ValueError("volume must be an object")
    require_number(volume.get("base"), "volume.base", 0.01)
    require_number(volume.get("noise"), "volume.noise", 0.0)
    require_number(volume.get("volatility_sensitivity"), "volume.volatility_sensitivity", 0.0)
    regimes = spec.get("regimes")
    if not isinstance(regimes, list) or not regimes:
        raise ValueError("regimes must be a non-empty list")
    expected_start = 0
    for index, regime in enumerate(regimes):
        if not isinstance(regime, dict):
            raise ValueError(f"regimes[{index}] must be an object")
        if regime.get("start") != expected_start:
            raise ValueError("regimes must be contiguous and start at zero")
        end = regime.get("end")
        if not isinstance(end, int) or isinstance(end, bool) or end <= expected_start or end > days:
            raise ValueError(f"regimes[{index}].end is invalid")
        expected_start = end
        for name in ("drift", "volatility", "correlation", "mean_reversion"):
            require_number(regime.get(name, 0.0), f"regimes[{index}].{name}")
        if regime.get("volatility", 0.0) <= 0.0:
            raise ValueError("regime volatility must be positive")
        if not -1.0 <= regime.get("correlation", 0.0) <= 1.0:
            raise ValueError("regime correlation must be between -1 and 1")
        clustering = require_number(
            regime.get("volatility_clustering", 0.0),
            f"regimes[{index}].volatility_clustering",
            0.0,
        )
        if clustering >= 1.0:
            raise ValueError("volatility_clustering must be less than 1")
        probability = require_number(
            regime.get("shock_probability", 0.0),
            f"regimes[{index}].shock_probability",
            0.0,
        )
        if probability > 1.0:
            raise ValueError("shock_probability must not exceed 1")
        require_number(regime.get("shock_mean", 0.0), f"regimes[{index}].shock_mean")
        require_number(regime.get("shock_volatility", 0.0), f"regimes[{index}].shock_volatility", 0.0)
        offsets = regime.get("asset_drift_offsets", [0.0] * len(assets))
        if not isinstance(offsets, list) or len(offsets) != len(assets):
            raise ValueError("asset_drift_offsets must match the asset count")
        for offset in offsets:
            require_number(offset, "asset_drift_offsets entry")
        period = regime.get("leadership_rotation_period", 0)
        if not isinstance(period, int) or isinstance(period, bool) or period < 0:
            raise ValueError("leadership_rotation_period must be a non-negative integer")
    if expected_start != days:
        raise ValueError("regimes must cover every generated day")


def regime_at(spec: dict[str, Any], index: int) -> tuple[int, dict[str, Any]]:
    for regime_index, regime in enumerate(spec["regimes"]):
        if regime["start"] <= index < regime["end"]:
            return regime_index, regime
    raise AssertionError("validated regimes cover every day")


def generate_rows(spec: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    validate_spec(spec)
    rng = random.Random(spec["seed"])
    dates = business_days(date.fromisoformat(spec["start_date"]), spec["days"])
    assets = spec["assets"]
    prices = [float(asset["start_price"]) for asset in assets]
    anchors = prices.copy()
    conditional_volatility = [float(spec["regimes"][0]["volatility"])] * len(assets)
    rows: list[dict[str, str]] = []
    regime_rows: list[dict[str, object]] = []

    for day_index, trading_date in enumerate(dates):
        regime_index, regime = regime_at(spec, day_index)
        base_volatility = float(regime["volatility"])
        correlation = float(regime.get("correlation", 0.0))
        common = rng.gauss(0.0, base_volatility)
        shock = 0.0
        if rng.random() < float(regime.get("shock_probability", 0.0)):
            shock = rng.gauss(
                float(regime.get("shock_mean", 0.0)),
                float(regime.get("shock_volatility", 0.0)),
            )
        regime_rows.append(
            {
                "date": trading_date.isoformat(),
                "day_index": day_index,
                "regime_index": regime_index,
                "regime": regime.get("label", f"regime_{regime_index}"),
            }
        )
        offsets = [float(value) for value in regime.get("asset_drift_offsets", [0.0] * len(assets))]
        rotation_period = int(regime.get("leadership_rotation_period", 0))
        if rotation_period:
            rotation = ((day_index - int(regime["start"])) // rotation_period) % len(assets)
            offsets = offsets[-rotation:] + offsets[:-rotation] if rotation else offsets
        for asset_index, asset in enumerate(assets):
            clustering = float(regime.get("volatility_clustering", 0.0))
            conditional_volatility[asset_index] = (
                (1.0 - clustering) * base_volatility
                + clustering
                * conditional_volatility[asset_index]
                * (0.85 + min(abs(common) / base_volatility, 3.0) * 0.15)
            )
            sigma = max(conditional_volatility[asset_index], base_volatility * 0.25)
            opening = prices[asset_index] * math.exp(rng.gauss(0.0, sigma * 0.15))
            idiosyncratic = rng.gauss(0.0, sigma)
            correlated = correlation * common + math.sqrt(max(0.0, 1.0 - correlation**2)) * idiosyncratic
            reversion = float(regime.get("mean_reversion", 0.0)) * math.log(
                anchors[asset_index] / prices[asset_index]
            )
            log_return = (
                float(regime.get("drift", 0.0))
                + float(asset.get("drift_offset", 0.0))
                + offsets[asset_index]
                + reversion
                + correlated
                + shock
            )
            closing = max(0.01, opening * math.exp(log_return))
            intraday = abs(rng.gauss(0.0, sigma * 0.35))
            high = max(opening, closing) * (1.0 + intraday)
            low = max(0.01, min(opening, closing) * (1.0 - intraday))
            volume_config = spec["volume"]
            volume_multiplier = math.exp(rng.gauss(0.0, float(volume_config["noise"])))
            volume_multiplier *= 1.0 + float(volume_config["volatility_sensitivity"]) * abs(log_return)
            volume = float(volume_config["base"]) * volume_multiplier
            rows.append(
                {
                    "date": trading_date.isoformat(),
                    "asset": asset["symbol"],
                    "open": f"{opening:.8f}",
                    "high": f"{high:.8f}",
                    "low": f"{low:.8f}",
                    "close": f"{closing:.8f}",
                    "adjusted_close": f"{closing:.8f}",
                    "volume": f"{volume:.2f}",
                }
            )
            prices[asset_index] = closing
    return rows, regime_rows


def write_csv(path: Path, fieldnames: tuple[str, ...] | tuple[str, str, str, str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate(spec_path: Path, output_directory: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    rows, regime_rows = generate_rows(spec)
    output_directory.mkdir(parents=True, exist_ok=True)
    market_path = output_directory / "market.csv"
    regimes_path = output_directory / "regimes.csv"
    write_csv(market_path, CSV_FIELDS, rows)
    write_csv(regimes_path, ("date", "day_index", "regime_index", "regime"), regime_rows)
    market_sha256 = hashlib.sha256(market_path.read_bytes()).hexdigest()
    regimes_sha256 = hashlib.sha256(regimes_path.read_bytes()).hexdigest()
    canonical_spec = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("ascii")
    metadata = {
        "generator_version": spec["version"],
        "name": spec["name"],
        "regime_family": spec["regime_family"],
        "seed": spec["seed"],
        "generation_timestamp": spec["generation_timestamp"],
        "dates": spec["days"],
        "rows": len(rows),
        "assets": [asset["symbol"] for asset in spec["assets"]],
        "start_date": regime_rows[0]["date"],
        "end_date": regime_rows[-1]["date"],
        "market_sha256": market_sha256,
        "regimes_sha256": regimes_sha256,
        "generator_config_sha256": hashlib.sha256(canonical_spec).hexdigest(),
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    (output_directory / "generator-config.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = generate(args.config, args.output)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
