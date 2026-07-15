#!/usr/bin/env python3
"""Create compact, deterministic tables from Prompt 6 raw CLI reports."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/prompt6"
CONDITIONS = (
    "default-control",
    "default-ppo",
    "momentum-ppo",
    "volatility-trend-ppo",
    "combined-ppo",
    "combined-ppo-evolution",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(items: list[float]) -> float:
    return statistics.fmean(items) if items else 0.0


def report_stats(report: dict) -> dict:
    agents = [row[1] for row in report["agents"]]
    return {
        "final_nw": mean([row["ending_net_worth"] for row in agents]),
        "drawdown": mean([row["maximum_drawdown"] for row in agents]),
        "turnover": mean([row["turnover"] for row in agents]),
        "costs": mean([row["total_fees"] for row in agents]),
        "survival_days": mean([row["survival_days"] for row in agents]),
        "deaths": report["deaths"],
        "births": report["births"],
        "updates": report["learning_updates"],
        "active_positions": report["neural_mean_active_positions"],
        "concentration": report["neural_mean_concentration"],
        "cash": report["neural_mean_cash_allocation"],
        "schema": report["observation_schema_identity"],
        "lifecycle": report["lifecycle"],
    }


def main() -> None:
    manifest = list(csv.DictReader((REPORTS / "run-manifest.csv").open(encoding="utf-8")))
    seeds = sorted({int(row["seed"]) for row in manifest if row["status"] == "passed"})
    rows: list[dict] = []
    for condition in CONDITIONS:
        for seed in seeds:
            paths = {
                split: REPORTS / f"{condition}-seed-{seed}{'' if split == 'training' else '-' + split}.txt"
                for split in ("training", "validation", "test")
            }
            if not all(path.exists() for path in paths.values()):
                continue
            stats = {split: report_stats(load(path)) for split, path in paths.items()}
            rows.append({
                "condition": condition,
                "seed": seed,
                "status": stats["training"]["lifecycle"],
                "schema_hash": stats["training"]["schema"],
                "train_nw": stats["training"]["final_nw"],
                "validation_nw": stats["validation"]["final_nw"],
                "test_nw": stats["test"]["final_nw"],
                "test_drawdown": stats["test"]["drawdown"],
                "test_turnover": stats["test"]["turnover"],
                "test_costs": stats["test"]["costs"],
                "test_survival_days": stats["test"]["survival_days"],
                "deaths": stats["training"]["deaths"],
                "births": stats["training"]["births"],
                "ppo_updates": stats["training"]["updates"],
                "active_positions": stats["training"]["active_positions"],
                "concentration": stats["training"]["concentration"],
                "cash_allocation": stats["training"]["cash"],
            })
    with (REPORTS / "per-seed-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fields = [
        "condition", "seeds", "mean_train_nw", "mean_validation_nw", "mean_test_nw",
        "median_test_nw", "std_test_nw", "mean_test_drawdown", "mean_test_turnover",
        "mean_test_costs", "mean_test_survival_days", "mean_deaths", "mean_births",
        "mean_ppo_updates", "mean_active_positions", "mean_concentration", "mean_cash_allocation",
        "schema_hash",
    ]
    summaries = []
    for condition in CONDITIONS:
        group = [row for row in rows if row["condition"] == condition]
        if not group:
            continue
        tests = [float(row["test_nw"]) for row in group]
        summaries.append({
            "condition": condition,
            "seeds": len(group),
            "mean_train_nw": mean([float(row["train_nw"]) for row in group]),
            "mean_validation_nw": mean([float(row["validation_nw"]) for row in group]),
            "mean_test_nw": mean(tests),
            "median_test_nw": statistics.median(tests),
            "std_test_nw": statistics.stdev(tests) if len(tests) > 1 else 0.0,
            "mean_test_drawdown": mean([float(row["test_drawdown"]) for row in group]),
            "mean_test_turnover": mean([float(row["test_turnover"]) for row in group]),
            "mean_test_costs": mean([float(row["test_costs"]) for row in group]),
            "mean_test_survival_days": mean([float(row["test_survival_days"]) for row in group]),
            "mean_deaths": mean([float(row["deaths"]) for row in group]),
            "mean_births": mean([float(row["births"]) for row in group]),
            "mean_ppo_updates": mean([float(row["ppo_updates"]) for row in group]),
            "mean_active_positions": mean([float(row["active_positions"]) for row in group]),
            "mean_concentration": mean([float(row["concentration"]) for row in group]),
            "mean_cash_allocation": mean([float(row["cash_allocation"]) for row in group]),
            "schema_hash": group[0]["schema_hash"],
        })
    with (REPORTS / "condition-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
