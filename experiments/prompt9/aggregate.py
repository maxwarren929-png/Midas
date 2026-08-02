#!/usr/bin/env python3
"""Create deterministic Prompt 9 run, aggregate, and paired-comparison tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("cash", "equal-weight", "buy-and-hold", "momentum")
PAIRS = (
    ("combined-ppo", "cash"),
    ("combined-ppo", "equal-weight"),
    ("combined-ppo", "momentum"),
    ("combined-ppo-evolution", "combined-ppo"),
    ("combined-ppo-evolution", "cash"),
    ("combined-ppo-evolution", "equal-weight"),
    ("combined-ppo-evolution", "momentum"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return statistics.fmean(items) if items else 0.0


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def sign_test_p_value(wins: int, losses: int) -> float:
    count = wins + losses
    if count == 0:
        return 1.0
    tail = sum(math.comb(count, value) for value in range(min(wins, losses) + 1)) / (2**count)
    return min(1.0, 2.0 * tail)


def bootstrap_ci(values: list[float], label: str) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    estimates = [mean(rng.choice(values) for _ in values) for _ in range(2000)]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def run_identity(row: dict[str, Any]) -> tuple[str, str, str, int, str]:
    return row["market_id"], row["fold"], row["split"], int(row["seed"]), row["condition"]


def parse_task(task: str) -> dict[str, str]:
    stage, market_id, fold, condition, seed, split = task.split("/")
    return {
        "stage": stage,
        "market_id": market_id,
        "fold": fold,
        "condition": condition,
        "seed": seed,
        "split": split,
    }


def extract_result(manifest: dict[str, str], market: dict[str, str], root: Path) -> dict[str, Any]:
    identity = parse_task(manifest["task"])
    envelope = json.loads((root / manifest["output"]).read_text(encoding="utf-8"))
    report = envelope["report"]
    summaries = [value for _, value in report["agents"]]
    agents = envelope.get("agents", [])
    diagnostics = envelope.get("diagnostics", {})
    evolution = envelope.get("evolution", {}).get("metrics", {})
    starting = mean(value["starting_net_worth"] for value in summaries)
    ending = mean(value["ending_net_worth"] for value in summaries)
    costs = mean(
        value["total_fees"] + value["spread_cost"] + value["slippage_cost"]
        for value in summaries
    )
    checkpoint_hash = ""
    provenance_path = Path(manifest["output"]).with_suffix(".provenance.json")
    if (root / provenance_path).exists():
        checkpoint_hash = json.loads((root / provenance_path).read_text(encoding="utf-8")).get(
            "checkpoint_sha256", ""
        ) or ""
    result: dict[str, Any] = {
        **identity,
        "regime_family": market["regime_family"],
        "holdout": market["holdout"],
        "starting_net_worth": starting,
        "ending_net_worth": ending,
        "absolute_profit": ending - starting,
        "total_return": mean(value["total_return"] for value in summaries),
        "annualized_return": mean(value["annualised_return"] for value in summaries),
        "maximum_drawdown": mean(value["maximum_drawdown"] for value in summaries),
        "worst_agent_drawdown": max((value["maximum_drawdown"] for value in summaries), default=0.0),
        "volatility": mean(value["volatility"] for value in summaries),
        "sharpe_like": mean(value["sharpe_like"] for value in summaries),
        "sortino_like": mean(value["sortino_like"] for value in summaries),
        "turnover": mean(value["turnover"] for value in summaries),
        "transaction_costs": costs,
        "trades": sum(value["trades"] for value in summaries),
        "profitable": ending > starting,
        "deaths": report["deaths"],
        "replacements": report["births"] - len(summaries),
        "reproductions": envelope.get("metrics", {}).get("reproductions", 0),
        "ppo_updates": report["learning_updates"],
        "final_active_positions": mean(
            sum(abs(position["quantity"]) > 1e-12 for position in agent["portfolio"]["positions"])
            for agent in agents
        ),
        "mean_cash_allocation": diagnostics.get("mean_cash_allocation", 0.0),
        "mean_concentration": diagnostics.get("mean_concentration", 0.0),
        "policy_failures": envelope.get("metrics", {}).get("policy_failures", 0),
        "checkpoint_identity": checkpoint_hash,
        "dataset_identity": report["dataset_identity"],
        "dataset_sha256": market["sha256"],
        "schema_identity": report["observation_schema_identity"],
        "configuration_hash": envelope["config_hash"],
        "report_trace_count": diagnostics.get("report_trace_count", 0),
        "crossover_count": evolution.get("crossover_count", 0),
        "mutation_count": evolution.get("mutation_count", 0),
        "failure_reason": report.get("failure_reason") or "",
    }
    return result


def add_benchmark_flags(rows: list[dict[str, Any]]) -> None:
    lookup = {run_identity(row): row for row in rows}
    for row in rows:
        base = (row["market_id"], row["fold"], row["split"], int(row["seed"]))
        for benchmark in BENCHMARKS:
            peer = lookup.get((*base, benchmark))
            row[f"beat_{benchmark.replace('-', '_')}"] = bool(
                peer and row["total_return"] > peer["total_return"]
            )


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity, group in sorted(groups.items()):
        returns = [row["total_return"] for row in group]
        output.append(
            {
                **dict(zip(keys, identity)),
                "runs": len(group),
                "mean_return": mean(returns),
                "median_return": statistics.median(returns),
                "return_stddev": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
                "return_iqr": percentile(returns, 0.75) - percentile(returns, 0.25),
                "best_return": max(returns),
                "worst_return": min(returns),
                "profitable_percentage": 100.0 * mean(row["profitable"] for row in group),
                "beat_cash_percentage": 100.0 * mean(row["beat_cash"] for row in group),
                "beat_equal_weight_percentage": 100.0
                * mean(row["beat_equal_weight"] for row in group),
                "beat_buy_and_hold_percentage": 100.0
                * mean(row["beat_buy_and_hold"] for row in group),
                "beat_momentum_percentage": 100.0 * mean(row["beat_momentum"] for row in group),
                "mean_maximum_drawdown": mean(row["maximum_drawdown"] for row in group),
                "worst_maximum_drawdown": max(row["worst_agent_drawdown"] for row in group),
                "mean_transaction_cost": mean(row["transaction_costs"] for row in group),
                "mean_turnover": mean(row["turnover"] for row in group),
            }
        )
    return output


def paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {run_identity(row): row for row in rows}
    output = []
    for left, right in PAIRS:
        differences = []
        for key, row in sorted(lookup.items()):
            if key[-1] != left:
                continue
            peer = lookup.get((*key[:-1], right))
            if peer:
                differences.append(row["total_return"] - peer["total_return"])
        wins = sum(value > 1e-15 for value in differences)
        losses = sum(value < -1e-15 for value in differences)
        ties = len(differences) - wins - losses
        low, high = bootstrap_ci(differences, f"{left}:{right}")
        output.append(
            {
                "strategy": left,
                "comparator": right,
                "pairs": len(differences),
                "mean_paired_difference": mean(differences),
                "median_paired_difference": statistics.median(differences) if differences else 0.0,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "paired_win_percentage": 100.0 * wins / len(differences) if differences else 0.0,
                "mean_difference_ci_95_low": low,
                "mean_difference_ci_95_high": high,
                "two_sided_sign_test_p": sign_test_p_value(wins, losses),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    run_manifests: list[Path], market_manifest: Path, output_dir: Path, root: Path = ROOT
) -> dict[str, int]:
    markets = {row["market_id"]: row for row in read_csv(market_manifest)}
    manifest_rows = [row for path in run_manifests for row in read_csv(path)]
    failures = [row for row in manifest_rows if row["status"] not in ("passed", "skipped")]
    completed = [
        row
        for row in manifest_rows
        if row["status"] in ("passed", "skipped") and parse_task(row["task"])["split"] != "training"
    ]
    rows = [extract_result(row, markets[parse_task(row["task"])["market_id"]], root) for row in completed]
    rows.sort(key=run_identity)
    add_benchmark_flags(rows)
    write_csv(output_dir / "per-run-results.csv", rows)
    write_csv(output_dir / "per-market-summary.csv", summarize(rows, ("condition", "split", "market_id")))
    write_csv(output_dir / "per-regime-summary.csv", summarize(rows, ("condition", "split", "regime_family")))
    write_csv(output_dir / "all-markets-summary.csv", summarize(rows, ("condition", "split")))
    write_csv(output_dir / "paired-comparisons.csv", paired_comparisons(rows))
    write_csv(output_dir / "holdout-summary.csv", summarize([row for row in rows if row["holdout"] == "true"], ("condition", "split")))
    write_csv(output_dir / "failures.csv", failures)
    reproducibility = {
        "completed_evaluations": len(rows),
        "failed_tasks": len(failures),
        "input_manifests": {
            path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted([*run_manifests, market_manifest])
        },
        "output_sha256": {},
    }
    for path in sorted(output_dir.glob("*.csv")):
        reproducibility["output_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"completed": len(rows), "failed": len(failures)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=Path, action="append", required=True)
    parser.add_argument("--market-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.run_manifest, args.market_manifest, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
