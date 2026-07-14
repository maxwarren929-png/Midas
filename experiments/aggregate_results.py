#!/usr/bin/env python3
"""Aggregate deterministic CLI reports without changing simulation state."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

CONDITIONS = ("control", "ppo", "evolution-only", "ppo-evolution")
SEEDS = (41, 42, 43, 44, 45)
REPORTS = Path("reports")


def json_report(path: Path):
    text = path.read_text()
    return json.loads(text[text.find("{") :])


def load_run(condition, seed):
    return json_report(REPORTS / f"{condition}-seed-{seed}-diagnostics.txt")


def mean(values):
    return statistics.fmean(values) if values else 0.0


def run_row(condition, seed):
    diagnostic = load_run(condition, seed)
    report = diagnostic["report"]
    agents = [item[1] for item in report["agents"]]
    steps = max(1, report["steps"])
    evolution_path = REPORTS / f"{condition}-seed-{seed}-evolution.txt"
    evolution = json_report(evolution_path)["evolution"]["metrics"] if evolution_path.exists() else {}
    final = [agent["ending_net_worth"] for agent in agents]
    returns = [agent["total_return"] for agent in agents]
    drawdowns = [agent["maximum_drawdown"] for agent in agents]
    turnovers = [agent["turnover"] for agent in agents]
    survival = [agent["survival_days"] / steps for agent in agents]
    bankrupt = [agent["bankrupt"] for agent in agents]
    return {
        "condition": condition,
        "seed": seed,
        "steps": report["steps"],
        "lifecycle": report["lifecycle"],
        "mean_final_net_worth": mean(final),
        "median_final_net_worth": statistics.median(final) if final else 0.0,
        "best_final_net_worth": max(final) if final else 0.0,
        "worst_final_net_worth": min(final) if final else 0.0,
        "mean_return": mean(returns),
        "median_return": statistics.median(returns) if returns else 0.0,
        "mean_maximum_drawdown": mean(drawdowns),
        "mean_turnover": mean(turnovers),
        "survival_rate": mean(survival),
        "bankruptcy_rate": mean([float(value) for value in bankrupt]),
        "deaths": report["deaths"],
        "births": report["births"],
        "reproductions": report["evolutionary_births"],
        "ppo_updates": diagnostic["learning"]["ppo_updates"],
        "policy_failures": diagnostic["mechanics"]["policy_failures"],
        "recoverable_fallbacks": diagnostic["mechanics"]["recoverable_fallbacks"],
        "masked_rollout_entries": diagnostic["mechanics"]["masked_rollout_entries"],
        "actor_ineligible_entries": diagnostic["mechanics"]["actor_ineligible_entries"],
        "critic_ineligible_entries": diagnostic["mechanics"]["critic_ineligible_entries"],
        "invalid_action_penalty_total": diagnostic["mechanics"]["invalid_action_penalty_total"],
        "evolution_fallbacks": diagnostic["mechanics"]["evolution_fallbacks"],
        "max_generation_depth": evolution.get("maximum_generation_depth", diagnostic["evolution"]["maximum_generation_depth"]),
        "active_lineages": evolution.get("active_lineages", diagnostic["evolution"]["active_lineages"]),
        "mean_genetic_distance": evolution.get("mean_genetic_distance", diagnostic["evolution"]["mean_genetic_distance"]),
        "minimum_genetic_distance": evolution.get("minimum_genetic_distance", diagnostic["evolution"]["minimum_genetic_distance"]),
        "mean_lifespan": evolution.get("mean_lifespan", 0.0),
        "crossovers": evolution.get("crossover_count", diagnostic["evolution"]["crossovers_observed"]),
        "mutations": evolution.get("mutation_count", diagnostic["evolution"]["mutations_observed"]),
        "random_fallback_births": evolution.get("random_fallback_births", 0),
        "diversity_relaxations": evolution.get("diversity_constraint_relaxations", 0),
        "actor_delta": diagnostic["learning"]["actor_head_delta_norm"],
        "critic_delta": diagnostic["learning"]["critic_head_delta_norm"],
        "hidden_delta": diagnostic["learning"]["hidden_delta_norm"],
        "recurrent_delta": diagnostic["learning"]["recurrent_delta_norm"],
        "genome_matches": diagnostic["learning"]["birth_genome_matches_policy"],
        "optimizer_steps": diagnostic["learning"]["optimizer_steps"],
        "mean_policy_loss": diagnostic["learning"]["mean_policy_loss"],
        "mean_value_loss": diagnostic["learning"]["mean_value_loss"],
        "mean_entropy": diagnostic["learning"]["mean_entropy"],
        "mean_gradient_norm": diagnostic["learning"]["mean_gradient_norm"],
    }


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = [run_row(condition, seed) for condition in CONDITIONS for seed in SEEDS]
    write_csv(REPORTS / "run-level-results.csv", rows, list(rows[0]))
    ppo_fields = [
        "condition", "seed", "ppo_updates", "optimizer_steps", "actor_delta", "critic_delta",
        "hidden_delta", "recurrent_delta", "genome_matches", "mean_policy_loss",
        "mean_value_loss", "mean_entropy", "mean_gradient_norm", "masked_rollout_entries",
        "actor_ineligible_entries", "critic_ineligible_entries",
    ]
    write_csv(REPORTS / "ppo-diagnostics.csv", [{key: row[key] for key in ppo_fields} for row in rows], ppo_fields)
    evolution_fields = [
        "condition", "seed", "deaths", "births", "reproductions", "crossovers", "mutations",
        "random_fallback_births", "max_generation_depth", "active_lineages", "mean_genetic_distance",
        "minimum_genetic_distance", "mean_lifespan", "diversity_relaxations",
    ]
    write_csv(REPORTS / "evolution-diagnostics.csv", [{key: row[key] for key in evolution_fields} for row in rows], evolution_fields)

    summary = []
    for condition in CONDITIONS:
        group = [row for row in rows if row["condition"] == condition]
        summary.append({
            "condition": condition,
            "seeds": len(group),
            "mean_final_net_worth": mean([row["mean_final_net_worth"] for row in group]),
            "median_final_net_worth": statistics.median([row["median_final_net_worth"] for row in group]),
            "best_final_net_worth": max(row["best_final_net_worth"] for row in group),
            "worst_final_net_worth": min(row["worst_final_net_worth"] for row in group),
            "mean_return": mean([row["mean_return"] for row in group]),
            "mean_maximum_drawdown": mean([row["mean_maximum_drawdown"] for row in group]),
            "survival_rate": mean([row["survival_rate"] for row in group]),
            "bankruptcy_rate": mean([row["bankruptcy_rate"] for row in group]),
            "ppo_updates": mean([row["ppo_updates"] for row in group]),
            "deaths": mean([row["deaths"] for row in group]),
            "reproductions": mean([row["reproductions"] for row in group]),
            "max_generation_depth": max(row["max_generation_depth"] for row in group),
            "active_lineages": mean([row["active_lineages"] for row in group]),
            "seed_mean_std": statistics.stdev([row["mean_final_net_worth"] for row in group]),
        })
    fields = list(summary[0])
    write_csv(REPORTS / "experiment-summary.csv", summary, fields)

    benchmark_rows = []
    for seed in SEEDS:
        for split in ("training", "validation", "test"):
            path = REPORTS / f"benchmark-seed-{seed}-{split}.txt"
            for line in path.read_text().splitlines():
                if ": ending_net_worth=" in line:
                    policy, value = line.split(": ending_net_worth=")
                    benchmark_rows.append({"seed": seed, "split": split, "policy": policy, "ending_net_worth": float(value)})
    write_csv(REPORTS / "benchmark-results.csv", benchmark_rows, ["seed", "split", "policy", "ending_net_worth"])

    control = {row["seed"]: row for row in rows if row["condition"] == "control"}
    comparisons = []
    for condition in CONDITIONS:
        if condition == "control":
            continue
        group = [row for row in rows if row["condition"] == condition]
        differences = [row["mean_final_net_worth"] - control[row["seed"]]["mean_final_net_worth"] for row in group]
        comparisons.append({
            "condition": condition,
            "vs_control_mean_net_worth_delta": mean(differences),
            "seeds_beating_control": sum(value > 0.0 for value in differences),
            "seeds": len(differences),
        })
    write_csv(REPORTS / "condition-comparisons.csv", comparisons, list(comparisons[0]))


if __name__ == "__main__":
    main()
