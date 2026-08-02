#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aggregate  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def envelope(total_return: float) -> dict:
    return {
        "report": {
            "agents": [[0, {"starting_net_worth": 10000.0, "ending_net_worth": 10000.0 * (1 + total_return), "total_return": total_return, "annualised_return": total_return, "maximum_drawdown": max(0.0, -total_return), "volatility": 0.01, "sharpe_like": total_return / 0.01, "sortino_like": total_return / 0.01, "turnover": 10.0, "total_fees": 1.0, "spread_cost": 2.0, "slippage_cost": 3.0, "trades": 2}]],
            "deaths": 0,
            "births": 1,
            "learning_updates": 0,
            "dataset_identity": "dataset",
            "observation_schema_identity": "schema",
            "failure_reason": None,
        },
        "agents": [{"portfolio": {"positions": [{"quantity": 1.0}]}}],
        "diagnostics": {"mean_cash_allocation": 0.2, "mean_concentration": 0.8, "report_trace_count": 2},
        "metrics": {"policy_failures": 0, "reproductions": 0},
        "evolution": {"metrics": {}},
        "config_hash": "config",
    }


class AggregationTests(unittest.TestCase):
    def test_reports_are_reproducible_and_paired_by_shared_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market_manifest = root / "markets.csv"
            write_csv(market_manifest, [{"market_id": "m1", "regime_family": "uptrend", "holdout": "false", "sha256": "abc"}])
            rows = []
            for condition, value in (("combined-ppo", 0.02), ("cash", -0.01), ("equal-weight", 0.01), ("buy-and-hold", 0.005), ("momentum", 0.015)):
                output = root / f"{condition}.json"
                output.write_text(json.dumps(envelope(value)), encoding="utf-8")
                rows.append({"task": f"screening/m1/fold-1/{condition}/1/validation", "status": "passed", "output": output.name})
            run_manifest = root / "runs.csv"
            write_csv(run_manifest, rows)
            first = root / "first"
            second = root / "second"
            aggregate.aggregate([run_manifest], market_manifest, first, root)
            aggregate.aggregate([run_manifest], market_manifest, second, root)
            self.assertEqual(
                {path.name: path.read_bytes() for path in first.iterdir()},
                {path.name: path.read_bytes() for path in second.iterdir()},
            )
            paired = aggregate.read_csv(first / "paired-comparisons.csv")
            cash = next(row for row in paired if row["strategy"] == "combined-ppo" and row["comparator"] == "cash")
            self.assertEqual(cash["wins"], "1")
            self.assertAlmostEqual(float(cash["mean_paired_difference"]), 0.03)


if __name__ == "__main__":
    unittest.main()
