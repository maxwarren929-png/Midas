#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_market import generate, generate_rows, validate_spec  # noqa: E402


def specification(seed: int = 17) -> dict:
    return {
        "version": 1,
        "name": "test-market",
        "regime_family": "test",
        "seed": seed,
        "generation_timestamp": "2026-08-02T00:00:00Z",
        "start_date": "2020-01-02",
        "days": 12,
        "assets": [
            {"symbol": "AAA", "start_price": 100.0, "drift_offset": 0.0001},
            {"symbol": "BBB", "start_price": 80.0, "drift_offset": -0.0001},
        ],
        "volume": {"base": 1000000.0, "noise": 0.1, "volatility_sensitivity": 4.0},
        "regimes": [
            {
                "label": "first",
                "start": 0,
                "end": 6,
                "drift": 0.0002,
                "volatility": 0.005,
                "correlation": 0.4,
                "mean_reversion": 0.0,
                "volatility_clustering": 0.3,
                "shock_probability": 0.0,
                "shock_mean": 0.0,
                "shock_volatility": 0.0,
                "asset_drift_offsets": [0.0002, -0.0002],
                "leadership_rotation_period": 0,
            },
            {
                "label": "second",
                "start": 6,
                "end": 12,
                "drift": -0.0001,
                "volatility": 0.01,
                "correlation": 0.8,
                "mean_reversion": 0.001,
                "volatility_clustering": 0.6,
                "shock_probability": 0.1,
                "shock_mean": -0.02,
                "shock_volatility": 0.01,
                "asset_drift_offsets": [0.0, 0.0],
                "leadership_rotation_period": 3,
            },
        ],
    }


class GeneratorTests(unittest.TestCase):
    def test_same_specification_produces_identical_bytes_at_different_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps(specification()), encoding="utf-8")
            first = root / "first"
            second = root / "second"
            first_metadata = generate(config, first)
            second_metadata = generate(config, second)
            for filename in ("market.csv", "regimes.csv", "metadata.json", "generator-config.json"):
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())
            self.assertEqual(first_metadata, second_metadata)

    def test_different_seeds_produce_different_market_data(self) -> None:
        first, _ = generate_rows(specification(17))
        second, _ = generate_rows(specification(18))
        self.assertNotEqual(first, second)

    def test_regimes_must_be_contiguous_and_cover_the_market(self) -> None:
        invalid = copy.deepcopy(specification())
        invalid["regimes"][1]["start"] = 7
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_spec(invalid)
        invalid = copy.deepcopy(specification())
        invalid["regimes"][-1]["end"] = 11
        with self.assertRaisesRegex(ValueError, "cover every generated day"):
            validate_spec(invalid)

    def test_regime_labels_are_not_written_to_market_rows(self) -> None:
        rows, regimes = generate_rows(specification())
        self.assertEqual(tuple(rows[0]), (
            "date", "asset", "open", "high", "low", "close", "adjusted_close", "volume"
        ))
        self.assertIn("regime", regimes[0])
        self.assertNotIn("regime", rows[0])


if __name__ == "__main__":
    unittest.main()
