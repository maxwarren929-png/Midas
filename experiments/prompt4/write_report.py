#!/usr/bin/env python3
"""Write the human-readable Prompt 4 experiment report from frozen outputs."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "prompt4"


def rows(path: str) -> list[dict]:
    with (REPORTS / path).open() as handle:
        return list(csv.DictReader(handle))


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    summary = rows("condition-summary.csv")
    per_seed = rows("per-seed-results.csv")
    by_condition = {row["condition"]: row for row in summary}
    control = {int(row["seed"]): float(row["test_final_nw"]) for row in per_seed if row["condition"] == "control"}
    benchmark = (REPORTS / "benchmark-test.txt").read_text().strip()
    diagnostics = json.loads((REPORTS / "ppo-seed-42-diagnostics.json").read_text())
    evolution_diagnostics = json.loads((REPORTS / "ppo-evolution-seed-42-diagnostics.json").read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dataset_hash = (REPORTS / "dataset-hash.txt").read_text().strip().split()[1]

    lines = [
        "# Prompt 4 Persistent Incremental Trade Experiment",
        "",
        "## Executive Summary",
        "",
        "The neural action interface now emits persistent incremental trade impulses rather than target portfolios. All 40 frozen Prompt 3B condition/seed runs (control, PPO, evolution-only, PPO plus evolution; seeds 41-50) completed with checkpoint format 11, and every validation/test replay completed. The action-path regression suite and checkpoint-resume comparison passed.",
        "",
        "On this frozen synthetic market, incremental trades reduced test turnover substantially relative to the Prompt 3B target-allocation implementation. PPO test mean net worth was above the new static neural control, but results remain preliminary and do not establish profitability or generalisation.",
        "",
        "## Commit and Dataset",
        "",
        f"Implementation revision used for the rerun: `{commit}`.",
        f"Dataset: `data/prompt3b/market.csv`, SHA-256 `{dataset_hash}`, 1,800 business dates, 1,259 training steps, 269 validation steps, and 269 test steps. The market, costs, survival pressure, architecture, reward, and seeds were unchanged from Prompt 3B.",
        "",
        "## Action Semantics",
        "",
        "Each active asset has an independent Beta latent `u`; the PPO action is the latent vector and the executed impulse is `2u - 1`. The configured `maximum_trade_fraction = 0.05` scales the impulse by current close net worth. Cash is implicit and holdings persist. The validator masks unavailable assets and enforces the configured fraction; execution sells first, then cost-aware proportionally scales buys. Beta log probability and entropy are exact, and deterministic evaluation uses Beta means without consuming RNG or decaying temperature.",
        "",
        "## Mechanical Results",
        "",
        "| Condition | Seeds | Train NW | Validation NW | Test NW | Test turnover | Test costs | PPO updates | Deaths | Births |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in ("control", "ppo", "evolution-only", "ppo-evolution"):
        row = by_condition[condition]
        lines.append(
            f"| {condition} | {row['seeds']} | {fmt(float(row['mean_train_nw']))} | {fmt(float(row['mean_validation_nw']))} | {fmt(float(row['mean_test_nw']))} | {fmt(float(row['mean_test_turnover']))} | {fmt(float(row['mean_test_costs']))} | {fmt(float(row['mean_ppo_updates']), 1)} | {fmt(float(row['mean_deaths']), 1)} | {fmt(float(row['mean_births']), 1)} |"
        )

    lines += [
        "",
        "Mean test turnover fell from the Prompt 3B values of 17,461 (control), 17,681 (PPO), 17,416 (evolution-only), and 17,536 (PPO plus evolution) to 1,730, 2,506, 1,528, and 2,378 respectively. That is an approximately 90%, 86%, 91%, and 86% reduction. The new PPO policy had mean absolute impulse 0.463, mean active positions 3.61, mean concentration 0.496, and mean cash allocation 0.198 in training.",
        "",
        "PPO beat the paired static control on 6/10 test seeds; PPO plus evolution beat it on 8/10. The mean test differences were +46.89 and +68.16 net-worth units respectively. These paired results are encouraging for the action-interface objective but are not a profitability claim, and evolution is not isolated from the changed action semantics in this comparison.",
        "",
        "## Prompt 3B Comparison",
        "",
        "| Condition | Prompt 3B test NW | Prompt 4 test NW | Prompt 3B turnover | Prompt 4 turnover |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| control | 9260.26 | 9858.22 | 17461 | 1729.52 |",
        "| PPO | 9267.50 | 9905.10 | 17681 | 2506.41 |",
        "| evolution-only | 9258.11 | 9851.87 | 17416 | 1528.48 |",
        "| PPO + evolution | 9274.80 | 9926.38 | 17536 | 2378.06 |",
        "",
        "The historical and current runs use the same frozen market and protocol; the difference is the intended action-interface change plus the required checkpoint/validator compatibility updates. The large turnover reduction is therefore consistent with persistent holdings, but performance changes should not be attributed to PPO alone.",
        "",
        "## PPO and Evolution Evidence",
        "",
        f"Seed 42 PPO diagnostics: {diagnostics['learning']['ppo_updates']} updates, actor delta {diagnostics['learning']['actor_head_delta_norm']:.6f}, critic delta {diagnostics['learning']['critic_head_delta_norm']:.6f}, hidden delta {diagnostics['learning']['hidden_delta_norm']:.1f}, recurrent delta {diagnostics['learning']['recurrent_delta_norm']:.1f}, birth genomes unchanged `{diagnostics['learning']['birth_genome_matches_policy']}`, mean KL {diagnostics['learning']['mean_approximate_kl']:.3e}, clip fraction {diagnostics['learning']['mean_clip_fraction']:.3f}.",
        f"Seed 42 PPO plus evolution diagnostics: {evolution_diagnostics['mechanics']['deaths']} deaths, {evolution_diagnostics['mechanics']['reproductions']} reproductions, {evolution_diagnostics['evolution']['crossovers_observed']} crossovers, {evolution_diagnostics['evolution']['mutations_observed']} mutation reports, maximum generation {evolution_diagnostics['evolution']['maximum_generation_depth']}, and {evolution_diagnostics['evolution']['active_lineages']} active lineages. Birth lifetime state is built fresh from birth genomes; learned parent state is not inherited.",
        "",
        "## Benchmarks",
        "",
        "The test-split benchmark command produced:",
        "",
        "```text",
        benchmark,
        "```",
        "",
        "Buy-and-hold, equal-weight, and cash continue to use their documented benchmark semantics and are not constrained by neural concentration caps. The neural conditions exceed equal-weight and buy-and-hold in this particular test replay, while PPO is only modestly above cash; this is not evidence of robust trading skill.",
        "",
        "## Checkpoint Determinism",
        "",
        "For PPO seed 42, an uninterrupted run and a 600-step checkpoint followed by resume produced byte-identical JSON reports. The full test suite also compares domain histories, policy/lifetime state, optimizer state, rollouts, and evolution state. Checkpoint format 11 rejects format 10 because the new impulse metrics changed the serialized state layout; no unsafe migration is attempted.",
        "",
        "## Verification",
        "",
        "- `cargo fmt --all --check` passed.",
        "- `cargo check --workspace --all-targets` passed.",
        "- `cargo clippy --workspace --all-targets --all-features -- -D warnings` passed.",
        "- `cargo test --workspace --all-features` passed: 33 core unit tests, 53 foundation regressions, and 7 Prompt 2 integration tests; **93 passed, 0 failed, 0 ignored**.",
        "- Release CLI validation, 40 training runs, 80 validation/test replays, benchmark, diagnostics, and checkpoint-resume comparison passed.",
        "",
        "## Limitations",
        "",
        "The action distribution factorises independent per-asset Beta impulses and retains stable asset ordering. Hidden/recurrent layers remain frozen; full recurrent BPTT, turnover/reward redesign, GPU support, and Lamarckian inheritance remain outside scope. The experiment uses one synthetic market and ten seeds. No profitability or intelligence claim is supported.",
        "",
        "Raw run logs and checkpoints are kept under `reports/prompt4/` and `checkpoints/prompt4/`; compact tables are `condition-summary.csv`, `per-seed-results.csv`, and `run-manifest.csv`.",
    ]
    (ROOT / "reports" / "PROMPT4_INCREMENTAL_TRADE_EXPERIMENT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
