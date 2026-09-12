#!/usr/bin/env python3
"""
Matrix V12: Genetic Trading Cadence Battle Royale (Evolvable Churn & Conviction Traits)

In this experiment, trading frequency and risk parameters are NOT manually hardcoded.
Instead, they are unlocked as EVOLVABLE GENETIC CHROMOSOMES:
  1. [brain.traits.turnover_sensitivity]: Evolves between 0.00001 (hyper-active) and 0.10 (patient whale)
  2. [brain.traits.drawdown_sensitivity]: Evolves between 0.0 (unconstrained growth) and 0.50 (defensive shield)
  3. [brain.traits.maximum_allocation]: Evolves between 0.20 and 1.0 (adaptive leverage capacity)
  4. [brain.traits.action_temperature]: Evolves between 0.1 and 3.0 (impulse sharpness)

Models Evaluated:
  1. g64_deep_macro_gru (64-Unit Deep GRU, [128, 64] layers, 16d BPTT)
  2. m16_momentum_rotator (16-Unit Memory, [32, 16] layers, 15d BPTT)
  3. g1_aggressive_mega_15d (8-Unit Memory, [32, 16] layers, 15d BPTT)
  4. g32_conviction_gru (32-Unit GRU, [64, 32] layers, 16d BPTT)

Universe: 20-Asset Hyper Universe (SOXL, TQQQ, AMD, META, NVDA, BTC, GLD, SLV, etc.)
Training: 2015-01-05 to 2022-12-30 (8 Years)
Testing: 2024-08-12 to 2026-08-14 (Past 2 Years)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "matrix_v12_evolvable_cadence"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "matrix_v12_evolvable_cadence"
REPORTS_DIR = ROOT / "reports" / "matrix_v12_evolvable_cadence"
BIN = ROOT / "target" / "release" / "midas-cli"
BASE_TEMPLATE_PATH = ROOT / "configs" / "prompt3b" / "experiment-ppo.toml"

ASSETS_20 = [
    "AAPL", "AMD", "BTC", "GLD", "IWM", "META", "MSFT", "NVDA", "PLTR", "QQQ",
    "SLV", "SOXL", "SPXU", "SPY", "SQQQ", "TLT", "TQQQ", "TSLA", "UPRO", "XLE"
]

MODELS = {
    "g64_deep_macro_gru": {
        "description": "64-Unit Deep GRU ([128, 64], memory=64, 16d BPTT)",
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 16, "rollout": 64, "minibatch": 16,
    },
    "m16_momentum_rotator": {
        "description": "16-Unit Momentum Rotator ([32, 16], memory=16, 15d BPTT)",
        "memory_size": 16, "hidden": "[32, 16]", "seq_len": 15, "rollout": 60, "minibatch": 15,
    },
    "g1_aggressive_mega_15d": {
        "description": "8-Unit Aggressive Distilled ([32, 16], memory=8, 15d BPTT)",
        "memory_size": 8, "hidden": "[32, 16]", "seq_len": 15, "rollout": 60, "minibatch": 15,
    },
    "g32_conviction_gru": {
        "description": "32-Unit GRU ([64, 32], memory=32, 16d BPTT)",
        "memory_size": 32, "hidden": "[64, 32]", "seq_len": 16, "rollout": 64, "minibatch": 16,
    },
}


def build_config_text(seed: int, mod_key: str) -> str:
    template = BASE_TEMPLATE_PATH.read_text(encoding="utf-8")
    mod = MODELS[mod_key]
    
    # 1. 20-Asset Universe & Dataset Path
    template = re.sub(r'assets\s*=\s*\[.*\]', f'assets = {json.dumps(ASSETS_20)}', template)
    template = re.sub(r'(?m)^path\s*=\s*"data/.*"$', 'path = "data/hyper/hyper_20_assets_2016_2026.csv"', template)
    
    # 2. Splits (2015 to 2026)
    template = re.sub(r'(?m)^start\s*=\s*"2010-01-04"$', 'start = "2015-01-05"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2014-10-31"$', 'end = "2022-12-30"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2014-11-03"$', 'start = "2023-01-03"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2015-11-13"$', 'end = "2024-08-09"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2015-11-16"$', 'start = "2024-08-12"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2016-11-25"$', 'end = "2026-08-14"', template)

    # 3. Seed and Checkpoint Target
    template = re.sub(r'(?m)^seed\s*=.*$', f"seed = {seed}", template)
    template = re.sub(
        r'(?m)^path\s*=\s*"checkpoints/.*"$',
        f'path = "checkpoints/matrix_v12_evolvable_cadence/{mod_key}-seed-{seed}.bin"',
        template
    )

    # 4. Neural Architecture
    template = re.sub(r'(?m)^hidden_layers\s*=.*$', f"hidden_layers = {mod['hidden']}", template)
    template = re.sub(r'(?m)^memory_size\s*=.*$', f"memory_size = {mod['memory_size']}", template)
    template = re.sub(r'(?m)^sequence_length\s*=.*$', f"sequence_length = {mod['seq_len']}", template)
    template = re.sub(r'(?m)^rollout_length\s*=.*$', f"rollout_length = {mod['rollout']}", template)
    template = re.sub(r'(?m)^minibatch_size\s*=.*$', f"minibatch_size = {mod['minibatch']}", template)

    # 5. Sizing & High-Conviction Allocation
    template = re.sub(r'(?m)^maximum_trade_fraction\s*=.*$', "maximum_trade_fraction = 0.50", template)

    # 6. Reward
    template = template.replace('[reward]\nkind = "risk_adjusted_return"', '[reward]\nkind = "log_return"')
    template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.0", template)
    template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.0001", template)

    # 7. Corporate Actions Tolerance
    template = re.sub(r'(?m)^discontinuity_threshold\s*=.*$', "discontinuity_threshold = 0.85", template)

    # 8. 🧬 UNLOCK EVOLVABLE GENETIC TRAITS (Turnover / Cadence / Risk / Allocation)
    traits_block = """
[brain.traits.learning_rate]
initial = 0.0003
minimum = 0.00001
maximum = 0.01
mutation_scale = 0.05
evolvable = true

[brain.traits.entropy_coefficient]
initial = 0.01
minimum = 0.001
maximum = 0.05
mutation_scale = 0.05
evolvable = true

[brain.traits.action_temperature]
initial = 1.0
minimum = 0.2
maximum = 2.5
mutation_scale = 0.1
evolvable = true

[brain.traits.temperature_decay]
initial = 0.999
minimum = 0.95
maximum = 1.0
mutation_scale = 0.01
evolvable = true

[brain.traits.drawdown_sensitivity]
initial = 0.05
minimum = 0.0
maximum = 0.50
mutation_scale = 0.05
evolvable = true

[brain.traits.turnover_sensitivity]
initial = 0.001
minimum = 0.00001
maximum = 0.05
mutation_scale = 0.05
evolvable = true

[brain.traits.maximum_allocation]
initial = 0.95
minimum = 0.20
maximum = 1.0
mutation_scale = 0.05
evolvable = true
"""
    # Replace existing traits block with full evolvable block
    template = re.sub(r'\[brain\.traits\.learning_rate\][\s\S]*?(?=\[training\])', traits_block + "\n", template)

    # 9. Distilled High-Signal Features (60 Clean Inputs)
    obs_block = """
[observations]
schema = "default"
providers = ["market", "portfolio", "agent_state", "rolling_market_stats"]

[observations.rolling_market_stats]
enabled = true
return_windows = [20, 60, 120]
volatility_windows = [20]
sma_windows = [50, 200]
relative_strength_windows = [20]
market_summary = true
market_summary_window = 20
minimum_history = "zero_fill"
"""
    template += obs_block

    return template


def generate_all_configs(seeds: List[int] = [41]) -> List[Path]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for mod_key in MODELS.keys():
        for seed in seeds:
            cfg_text = build_config_text(seed, mod_key)
            cfg_path = CONFIGS_DIR / f"{mod_key}-seed-{seed}.toml"
            cfg_path.write_text(cfg_text, encoding="utf-8")
            generated.append(cfg_path)
    print(f"Generated {len(generated)} experiment configs in {CONFIGS_DIR}")
    return generated


def run_single_job(mod_key: str, seed: int) -> Dict[str, Any]:
    cfg_path = CONFIGS_DIR / f"{mod_key}-seed-{seed}.toml"
    cp_path = CHECKPOINTS_DIR / f"{mod_key}-seed-{seed}.bin"
    rep_path = REPORTS_DIR / f"{mod_key}-seed-{seed}.json"
    
    t0 = time.time()
    
    # 1. Train on 2015-2022 with Darwinian Evolution & Mutating Traits
    sim_proc = subprocess.run([
        str(BIN), "simulate",
        "--config", str(cfg_path),
        "--split", "training",
        "--checkpoint", str(cp_path)
    ], capture_output=True, text=True)
    
    if sim_proc.returncode != 0:
        return {"model": mod_key, "seed": seed, "error": sim_proc.stderr}
        
    t_train = time.time() - t0
    
    # 2. Evaluate on Past 2 Years (2024-2026)
    eval_proc = subprocess.run([
        str(BIN), "evaluate",
        "--config", str(cfg_path),
        "--checkpoint", str(cp_path),
        "--split", "test"
    ], capture_output=True, text=True)
    
    if eval_proc.returncode != 0:
        return {"model": mod_key, "seed": seed, "error": eval_proc.stderr}
        
    try:
        data = json.loads(eval_proc.stdout)
        agents = data.get("agents", [])
        if agents:
            rets = [a[1]["total_return"] * 100 for a in agents]
            dds = [a[1]["maximum_drawdown"] * 100 for a in agents]
            tos = [a[1]["turnover"] for a in agents]
            fees = [a[1]["total_fees"] + a[1]["spread_cost"] + a[1]["slippage_cost"] for a in agents]
            
            res = {
                "model": mod_key,
                "seed": seed,
                "return_%": sum(rets) / len(rets),
                "max_drawdown_%": sum(dds) / len(dds),
                "turnover_$": sum(tos) / len(tos),
                "friction_$": sum(fees) / len(fees),
                "train_time_s": t_train,
            }
            print(f"[✓] {mod_key} -> Return: {res['return_%']:>+7.2f}%, Max DD: {res['max_drawdown_%']:>6.2f}%, Turnover: ${res['turnover_$']:.2f}")
            return res
    except Exception as e:
        return {"model": mod_key, "seed": seed, "error": str(e)}

    return {"model": mod_key, "seed": seed, "error": "No agent data"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[41])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_all_configs(args.seeds)

    tasks = []
    for mod_key in MODELS.keys():
        for seed in args.seeds:
            tasks.append((mod_key, seed))

    print(f"\n=== Running {len(tasks)} Matrix V12 Genetic Cadence Models across {args.workers} CPU Workers ===")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_job, m, s): (m, s) for m, s in tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if "error" not in res:
                results.append(res)
            else:
                print(f"[!] Error in {res.get('model')}: {res.get('error')}")

    print("\n" + "="*115)
    print(f"  MATRIX V12: GENETIC TRADING CADENCE & EVOLVED TRAITS LEADERBOARD (2024-08-12 TO 2026-08-14)")
    print("  Setup: 20-Asset Universe + Evolvable Turnover Sensitivity + Evolvable Drawdown Sensitivity + 60 Signals")
    print("="*115)
    print(f"{'Model Configuration':<36} | {'Test Return (%)':<16} | {'Max DD (%)':<11} | {'Turnover ($)':<12} | {'Friction ($)':<12}")
    print("-"*115)
    
    results.sort(key=lambda x: x["return_%"], reverse=True)
    for r in results:
        print(f"{r['model']:<36} | {r['return_%']:>+15.2f}% | {r['max_drawdown_%']:>10.2f}% | ${r['turnover_$']:>10.2f} | ${r['friction_$']:>10.2f}")
    print("="*115)


if __name__ == "__main__":
    main()
