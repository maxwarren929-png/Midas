#!/usr/bin/env python3
"""Aggregate Prompt 3B CLI reports without changing simulation behavior."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "prompt3b"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def report_stats(report: dict) -> dict[str, float]:
    agents = [item[1] for item in report.get("agents", [])]
    nw = [float(agent["ending_net_worth"]) for agent in agents]
    drawdown = [float(agent["maximum_drawdown"]) for agent in agents]
    turnover = [float(agent["turnover"]) for agent in agents]
    return {
        "mean_final_net_worth": mean(nw) if nw else 0.0,
        "median_final_net_worth": median(nw) if nw else 0.0,
        "best_final_net_worth": max(nw) if nw else 0.0,
        "worst_final_net_worth": min(nw) if nw else 0.0,
        "mean_maximum_drawdown": mean(drawdown) if drawdown else 0.0,
        "mean_turnover": mean(turnover) if turnover else 0.0,
        "survival_rate": sum(not bool(agent.get("bankrupt", False)) for agent in agents)
        / len(agents)
        if agents
        else 0.0,
        "population_at_end": float(len(agents)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    manifest = REPORTS / "run-manifest.csv"
    runs = list(csv.DictReader(manifest.open()))
    training = {}
    validation = {}
    test = {}
    for row in runs:
        if row["status"] != "passed":
            continue
        key = (row["condition"], int(row["seed"]))
        path = ROOT / row["report_file"]
        if not path.exists() or not path.suffix == ".txt":
            continue
        try:
            value = read_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if row["split"] == "training":
            training[key] = value
        elif row["split"] == "validation":
            validation[key] = value
        elif row["split"] == "test":
            test[key] = value

    per_seed = []
    for key in sorted(training):
        condition, seed = key
        train = training[key]
        val = validation.get(key, {})
        test_report = test.get(key, {})
        train_stats = report_stats(train)
        val_stats = report_stats(val)
        test_stats = report_stats(test_report)
        diagnostics = REPORTS / f"{condition}-seed-{seed}-diagnostics.json"
        diag = read_json(diagnostics) if diagnostics.exists() else {}
        learning = diag.get("learning", {})
        evolution_report_path = REPORTS / f"{condition}-seed-{seed}-evolution-report.json"
        evolution_metrics = {}
        if evolution_report_path.exists():
            evolution_metrics = read_json(evolution_report_path).get("evolution", {}).get("metrics", {})
        per_seed.append(
            {
                "condition": condition,
                "seed": seed,
                "train_steps": train.get("steps", 0),
                "train_mean_final_nw": train_stats["mean_final_net_worth"],
                "validation_mean_final_nw": val_stats["mean_final_net_worth"],
                "test_mean_final_nw": test_stats["mean_final_net_worth"],
                "test_median_final_nw": test_stats["median_final_net_worth"],
                "test_mean_drawdown": test_stats["mean_maximum_drawdown"],
                "test_mean_turnover": test_stats["mean_turnover"],
                "test_survival_rate": test_stats["survival_rate"],
                "deaths": train.get("deaths", 0),
                "births": train.get("births", 0),
                "reproductions": train.get("evolutionary_births", 0),
                "max_generation_depth": evolution_metrics.get("maximum_generation_depth", train.get("maximum_generation_depth", 0)),
                "active_lineages": evolution_metrics.get("active_lineages", train.get("active_lineages", 0)),
                "ppo_updates": learning.get("ppo_updates", train.get("learning_updates", 0)),
                "mean_kl": learning.get("mean_approximate_kl", 0.0),
                "clip_fraction": learning.get("mean_clip_fraction", 0.0),
                "actor_delta": learning.get("actor_head_delta_norm", 0.0),
                "critic_delta": learning.get("critic_head_delta_norm", 0.0),
                "entropy": learning.get("mean_entropy", 0.0),
                "gradient_norm": learning.get("mean_gradient_norm", 0.0),
                "concentration": train.get("neural_mean_concentration", 0.0),
                "cash_allocation": train.get("neural_mean_cash_allocation", 0.0),
            }
        )
    write_csv(REPORTS / "per-seed-results.csv", per_seed)

    summary = []
    for condition in sorted({row["condition"] for row in per_seed}):
        rows = [row for row in per_seed if row["condition"] == condition]
        def agg(field: str) -> tuple[float, float, float, float]:
            values = [float(row[field]) for row in rows]
            return (mean(values), median(values), pstdev(values) if len(values) > 1 else 0.0, min(values))
        nw = agg("test_mean_final_nw")
        summary.append(
            {
                "condition": condition,
                "seeds": len(rows),
                "mean_test_nw": nw[0],
                "median_test_nw": nw[1],
                "std_test_nw": nw[2],
                "min_test_nw": nw[3],
                "max_test_nw": max(float(row["test_mean_final_nw"]) for row in rows),
                "mean_validation_nw": mean(float(row["validation_mean_final_nw"]) for row in rows),
                "mean_drawdown": mean(float(row["test_mean_drawdown"]) for row in rows),
                "mean_turnover": mean(float(row["test_mean_turnover"]) for row in rows),
                "mean_survival": mean(float(row["test_survival_rate"]) for row in rows),
                "mean_ppo_updates": mean(float(row["ppo_updates"]) for row in rows),
                "mean_deaths": mean(float(row["deaths"]) for row in rows),
                "mean_reproductions": mean(float(row["reproductions"]) for row in rows),
                "max_generation": max(float(row["max_generation_depth"]) for row in rows),
            }
        )
    write_csv(REPORTS / "condition-summary.csv", summary)


if __name__ == "__main__":
    main()
