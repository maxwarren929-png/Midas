#!/usr/bin/env python3
"""Materialize the frozen Prompt 9 market suite, folds, and condition templates."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from generate_market import business_days, generate


ROOT = Path(__file__).resolve().parents[2]
ASSETS = [
    {"symbol": "SPY", "start_price": 100.0, "drift_offset": 0.00000},
    {"symbol": "AAA", "start_price": 80.0, "drift_offset": 0.00004},
    {"symbol": "BBB", "start_price": 120.0, "drift_offset": -0.00003},
    {"symbol": "CCC", "start_price": 60.0, "drift_offset": 0.00002},
]
GENERATION_TIMESTAMP = "2026-08-02T00:00:00Z"
START_DATE = "2010-01-04"
DAYS = 1600


def regime(
    label: str,
    start: int,
    end: int,
    drift: float,
    volatility: float,
    correlation: float,
    *,
    mean_reversion: float = 0.0,
    clustering: float = 0.25,
    shock_probability: float = 0.0,
    shock_mean: float = 0.0,
    shock_volatility: float = 0.0,
    offsets: list[float] | None = None,
    rotation: int = 0,
) -> dict[str, Any]:
    return {
        "label": label,
        "start": start,
        "end": end,
        "drift": drift,
        "volatility": volatility,
        "correlation": correlation,
        "mean_reversion": mean_reversion,
        "volatility_clustering": clustering,
        "shock_probability": shock_probability,
        "shock_mean": shock_mean,
        "shock_volatility": shock_volatility,
        "asset_drift_offsets": offsets or [0.0, 0.0, 0.0, 0.0],
        "leadership_rotation_period": rotation,
    }


def family_regimes(family: str) -> list[dict[str, Any]]:
    if family == "uptrend":
        return [regime("uptrend_with_pullbacks", 0, DAYS, 0.00045, 0.009, 0.45, shock_probability=0.012, shock_mean=-0.018, shock_volatility=0.006, offsets=[0.00012, 0.00005, -0.00006, -0.00011])]
    if family == "downtrend":
        return [regime("downtrend_with_relief", 0, DAYS, -0.00038, 0.010, 0.55, shock_probability=0.012, shock_mean=0.016, shock_volatility=0.006, offsets=[0.00008, -0.00002, -0.00008, 0.00002])]
    if family == "sideways":
        return [regime("sideways_mean_reverting", 0, DAYS, 0.0, 0.010, 0.30, mean_reversion=0.006, clustering=0.35, offsets=[0.00003, -0.00003, 0.00002, -0.00002])]
    if family == "high-volatility":
        return [regime("clustered_high_volatility", 0, DAYS, 0.00005, 0.022, 0.55, clustering=0.85, shock_probability=0.018, shock_mean=-0.012, shock_volatility=0.020)]
    if family == "low-volatility":
        return [regime("low_signal_to_cost", 0, DAYS, 0.00004, 0.0022, 0.35, clustering=0.20, offsets=[0.00002, -0.00002, 0.00001, -0.00001])]
    if family == "regime-switching":
        return [
            regime("low_volatility_uptrend", 0, 400, 0.00045, 0.005, 0.35),
            regime("high_volatility_downtrend", 400, 800, -0.00055, 0.020, 0.72, clustering=0.82, shock_probability=0.02, shock_mean=-0.025, shock_volatility=0.012),
            regime("sideways_recovery", 800, 1200, 0.00005, 0.011, 0.40, mean_reversion=0.004),
            regime("renewed_trend", 1200, DAYS, 0.00035, 0.009, 0.48),
        ]
    if family == "cross-sectional":
        return [regime("rotating_leadership", 0, DAYS, 0.00008, 0.010, 0.28, offsets=[0.00045, 0.00015, -0.00020, -0.00040], rotation=100)]
    if family == "correlated-stress":
        return [regime("correlation_stress", 0, DAYS, 0.00012, 0.013, 0.88, clustering=0.78, shock_probability=0.025, shock_mean=-0.028, shock_volatility=0.016, offsets=[0.00008, 0.00002, -0.00004, -0.00008])]
    raise ValueError(f"unknown regime family {family}")


def specification(market_id: str, family: str, seed: int) -> dict[str, Any]:
    return {
        "version": 1,
        "name": market_id,
        "regime_family": family,
        "seed": seed,
        "generation_timestamp": GENERATION_TIMESTAMP,
        "start_date": START_DATE,
        "days": DAYS,
        "assets": ASSETS,
        "volume": {"base": 2_000_000.0, "noise": 0.22, "volatility_sensitivity": 8.0},
        "regimes": family_regimes(family),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_conditions() -> None:
    output = ROOT / "configs/prompt9/conditions"
    output.mkdir(parents=True, exist_ok=True)
    neural_sources = {
        "combined-ppo": ROOT / "configs/prompt6/combined-ppo.toml",
        "combined-ppo-evolution": ROOT / "configs/prompt6/combined-ppo-evolution.toml",
    }
    for name, source in neural_sources.items():
        (output / f"{name}.toml").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    base = neural_sources["combined-ppo"].read_text(encoding="utf-8")
    for name, policy in (
        ("cash", "cash"),
        ("equal-weight", "equal_weight"),
        ("buy-and-hold", "buy_and_hold"),
        ("random", "random"),
        ("momentum", "momentum"),
    ):
        text = re.sub(r"population_size = \d+", "population_size = 1", base, count=1)
        text = re.sub(r'policy = "[^"]+"', f'policy = "{policy}"', text, count=1)
        text = re.sub(r'implementation = "ppo"', 'implementation = "no_learning"', text, count=1)
        text = re.sub(r"enabled = false", "enabled = false", text, count=1)
        text = re.sub(r"every_steps = \d+", "every_steps = 0", text, count=1)
        if name == "momentum":
            text = text.replace(
                'policy = "momentum"\n',
                'policy = "momentum"\nmomentum = { signal_feature = "return20", minimum_signal = 0.0 }\n',
                1,
            )
        (output / f"{name}.toml").write_text(text, encoding="utf-8")
    rows = [
        {"condition": name, "kind": "neural" if name.startswith("combined") else "benchmark", "template": f"configs/prompt9/conditions/{name}.toml"}
        for name in ("combined-ppo", "combined-ppo-evolution", "cash", "equal-weight", "buy-and-hold", "random", "momentum")
    ]
    write_csv(ROOT / "data/prompt9/manifests/conditions.csv", rows)


def write_folds() -> None:
    dates = business_days(__import__("datetime").date.fromisoformat(START_DATE), DAYS)
    windows = (("fold-1", 0, 599, 600, 799, 800, 999), ("fold-2", 0, 899, 900, 1099, 1100, 1299), ("fold-3", 0, 1199, 1200, 1399, 1400, 1599))
    rows = []
    for name, train_start, train_end, validation_start, validation_end, test_start, test_end in windows:
        rows.append({
            "fold": name,
            "training_start": dates[train_start].isoformat(),
            "training_end": dates[train_end].isoformat(),
            "validation_start": dates[validation_start].isoformat(),
            "validation_end": dates[validation_end].isoformat(),
            "test_start": dates[test_start].isoformat(),
            "test_end": dates[test_end].isoformat(),
        })
    write_csv(ROOT / "data/prompt9/manifests/folds.csv", rows)
    write_csv(ROOT / "data/prompt9/manifests/final-fold.csv", [rows[-1]])


def main() -> None:
    families = ("uptrend", "downtrend", "sideways", "high-volatility", "low-volatility", "regime-switching", "cross-sectional", "correlated-stress")
    markets = []
    for family_index, family in enumerate(families, start=1):
        for instance in range(1, 4):
            market_id = f"{family}-{instance:02d}"
            markets.append((market_id, family, 90_000 + family_index * 100 + instance, False))
    for instance, family in enumerate(("uptrend", "sideways", "regime-switching", "correlated-stress"), start=1):
        markets.append((f"holdout-{instance:02d}-{family}", family, 99_000 + instance, True))
    manifest_rows = []
    for market_id, family, seed, holdout in markets:
        spec_path = ROOT / f"configs/prompt9/generator/{market_id}.json"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(specification(market_id, family, seed), indent=2, sort_keys=True) + "\n", encoding="ascii")
        output = ROOT / f"data/prompt9/{'holdout' if holdout else family}/{market_id}"
        metadata = generate(spec_path, output)
        manifest_rows.append({
            "market_id": market_id,
            "regime_family": family,
            "holdout": str(holdout).lower(),
            "seed": seed,
            "dataset": (output / "market.csv").relative_to(ROOT).as_posix(),
            "metadata": (output / "metadata.json").relative_to(ROOT).as_posix(),
            "generator_config": spec_path.relative_to(ROOT).as_posix(),
            "sha256": metadata["market_sha256"],
            "generation_timestamp": metadata["generation_timestamp"],
            "rows": metadata["rows"],
            "dates": metadata["dates"],
            "assets": ";".join(metadata["assets"]),
            "start_date": metadata["start_date"],
            "end_date": metadata["end_date"],
        })
    write_csv(ROOT / "data/prompt9/manifests/markets.csv", manifest_rows)
    write_folds()
    write_conditions()
    print(json.dumps({"markets": len(markets), "development": 24, "holdout": 4}, sort_keys=True))


if __name__ == "__main__":
    main()
