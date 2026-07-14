#!/usr/bin/env python3
"""Build compact Prompt 4 summaries from the frozen CLI JSON outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "prompt4"
CONDITIONS = ("control", "ppo", "evolution-only", "ppo-evolution")
SEEDS = tuple(range(41, 51))


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def split_path(condition: str, seed: int, split: str) -> Path:
    suffix = {"training": "", "validation": "-validation", "test": "-test"}[split]
    return REPORTS / f"{condition}-seed-{seed}{suffix}.txt"


def mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else 0.0


def report_stats(report: dict) -> dict:
    agents = [entry[1] for entry in report.get("agents", [])]
    return {
        "lifecycle": report.get("lifecycle", "unknown"),
        "failure_reason": report.get("failure_reason") or "",
        "steps": report.get("steps", 0),
        "mean_final_net_worth": mean(agents, "ending_net_worth"),
        "median_final_net_worth": statistics.median(
            [float(row["ending_net_worth"]) for row in agents]
        )
        if agents
        else 0.0,
        "mean_total_return": mean(agents, "total_return"),
        "mean_drawdown": mean(agents, "maximum_drawdown"),
        "mean_turnover": mean(agents, "turnover"),
        "mean_trades": mean(agents, "trades"),
        "mean_total_fees": mean(agents, "total_fees"),
        "mean_survival_days": mean(agents, "survival_days"),
        "bankruptcy_rate": statistics.fmean(
            [1.0 if row.get("bankrupt") else 0.0 for row in agents]
        )
        if agents
        else 0.0,
        "deaths": report.get("deaths", 0),
        "births": report.get("births", 0),
        "learning_updates": report.get("learning_updates", 0),
        "neural_mean_abs_trade_impulse": report.get(
            "neural_mean_abs_trade_impulse", 0.0
        ),
        "neural_small_trade_impulse_fraction": report.get(
            "neural_small_trade_impulse_fraction", 0.0
        ),
        "neural_saturated_trade_impulse_fraction": report.get(
            "neural_saturated_trade_impulse_fraction", 0.0
        ),
        "neural_mean_active_positions": report.get("neural_mean_active_positions", 0.0),
        "neural_mean_concentration": report.get("neural_mean_concentration", 0.0),
        "neural_mean_cash_allocation": report.get("neural_mean_cash_allocation", 0.0),
        "active_lineages": report.get("active_lineages", 0),
    }


def main() -> None:
    rows: list[dict] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            split_reports = {
                split: report_stats(read_json(split_path(condition, seed, split)))
                for split in ("training", "validation", "test")
            }
            training = split_reports["training"]
            row = {
                "condition": condition,
                "seed": seed,
                "status": training["lifecycle"],
                "failure_reason": training["failure_reason"],
                "train_steps": training["steps"],
                "train_final_nw": training["mean_final_net_worth"],
                "validation_final_nw": split_reports["validation"]["mean_final_net_worth"],
                "test_final_nw": split_reports["test"]["mean_final_net_worth"],
                "test_median_nw": split_reports["test"]["median_final_net_worth"],
                "test_return": split_reports["test"]["mean_total_return"],
                "test_drawdown": split_reports["test"]["mean_drawdown"],
                "test_turnover": split_reports["test"]["mean_turnover"],
                "test_costs": split_reports["test"]["mean_total_fees"],
                "test_survival_days": split_reports["test"]["mean_survival_days"],
                "test_bankruptcy_rate": split_reports["test"]["bankruptcy_rate"],
                "deaths": training["deaths"],
                "births": training["births"],
                "ppo_updates": training["learning_updates"],
                "active_lineages": training["active_lineages"],
                "mean_abs_trade_impulse": training["neural_mean_abs_trade_impulse"],
                "small_impulse_fraction": training["neural_small_trade_impulse_fraction"],
                "saturated_impulse_fraction": training[
                    "neural_saturated_trade_impulse_fraction"
                ],
                "mean_active_positions": training["neural_mean_active_positions"],
                "mean_concentration": training["neural_mean_concentration"],
                "mean_cash_allocation": training["neural_mean_cash_allocation"],
            }
            rows.append(row)

    fields = list(rows[0])
    with (REPORTS / "per-seed-results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_fields = [
        "condition",
        "seeds",
        "mean_train_nw",
        "mean_validation_nw",
        "mean_test_nw",
        "median_test_nw",
        "std_test_nw",
        "mean_test_drawdown",
        "mean_test_turnover",
        "mean_test_costs",
        "mean_test_survival_days",
        "mean_deaths",
        "mean_births",
        "mean_ppo_updates",
        "mean_abs_trade_impulse",
        "mean_active_positions",
        "mean_concentration",
        "mean_cash_allocation",
        "mean_active_lineages",
    ]
    summaries: list[dict] = []
    for condition in CONDITIONS:
        group = [row for row in rows if row["condition"] == condition]
        tests = [float(row["test_final_nw"]) for row in group]
        summaries.append(
            {
                "condition": condition,
                "seeds": len(group),
                "mean_train_nw": mean(group, "train_final_nw"),
                "mean_validation_nw": mean(group, "validation_final_nw"),
                "mean_test_nw": statistics.fmean(tests),
                "median_test_nw": statistics.median(tests),
                "std_test_nw": statistics.stdev(tests) if len(tests) > 1 else 0.0,
                "mean_test_drawdown": mean(group, "test_drawdown"),
                "mean_test_turnover": mean(group, "test_turnover"),
                "mean_test_costs": mean(group, "test_costs"),
                "mean_test_survival_days": mean(group, "test_survival_days"),
                "mean_deaths": mean(group, "deaths"),
                "mean_births": mean(group, "births"),
                "mean_ppo_updates": mean(group, "ppo_updates"),
                "mean_abs_trade_impulse": mean(group, "mean_abs_trade_impulse"),
                "mean_active_positions": mean(group, "mean_active_positions"),
                "mean_concentration": mean(group, "mean_concentration"),
                "mean_cash_allocation": mean(group, "mean_cash_allocation"),
                "mean_active_lineages": mean(group, "active_lineages"),
            }
        )
    with (REPORTS / "condition-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dataset = ROOT / "data" / "prompt3b" / "market.csv"
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    with (REPORTS / "run-manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "seed", "commit", "config", "checkpoint", "training", "validation", "test"])
        for condition in CONDITIONS:
            for seed in SEEDS:
                writer.writerow(
                    [
                        condition,
                        seed,
                        commit,
                        f"configs/prompt3b/runs/{condition}-seed-{seed}.toml",
                        f"checkpoints/prompt4/{condition}-seed-{seed}.bin",
                        split_path(condition, seed, "training"),
                        split_path(condition, seed, "validation"),
                        split_path(condition, seed, "test"),
                    ]
                )
    (REPORTS / "dataset-hash.txt").write_text(
        f"sha256  {dataset_hash}  data/prompt3b/market.csv\n"
    )


if __name__ == "__main__":
    main()
