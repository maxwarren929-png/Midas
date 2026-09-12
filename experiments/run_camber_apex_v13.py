#!/usr/bin/env python3
"""
Matrix V13 (Camber Cloud & Local Apex Suite):
Corrected 64-Unit Variants & 128-Unit Titan Deep Brain Gauntlet.

Fixes all config bugs:
  1. g64_base_control: Canonical baseline (64 Units, [128, 64], 16d BPTT, 50% Sizing, Pure Log Return)
  2. g64_macro_horizon_30d: 30-Day BPTT Sequence Memory (seq_len=30, rollout=60, minibatch=15)
  3. g64_hyper_burst_70: 70% Max Sizing + Sharp Action Impulse (Beta min_shape=0.5, temp=2.0)
  4. g64_ath_breakout_sensor: Multi-horizon 252-Day & 5-Day Sensors (return=[5, 20, 60, 120, 252], vol=[10, 20, 60])
  5. g64_sortino_shield: Active Asymmetric Sortino Brake (kind='risk_adjusted_return', dd_penalty=0.05)
  6. g128_titan_deep_gru: 128-Unit Titan Brain ([256, 128, 64], seq_len=20, entropy=0.03)

Dispatches across Camber Cloud (or local multi-core workers).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "matrix_v13_camber"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "matrix_v13_camber"
REPORTS_DIR = ROOT / "reports" / "matrix_v13_camber"
BIN = ROOT / "target" / "release" / "midas-cli"
DATA_FILE = ROOT / "data" / "hyper" / "hyper_20_assets_2016_2026.csv"
BASE_TEMPLATE_PATH = ROOT / "configs" / "prompt3b" / "experiment-ppo.toml"

ASSETS_20 = [
    "AAPL", "AMD", "BTC", "GLD", "IWM", "META", "MSFT", "NVDA", "PLTR", "QQQ",
    "SLV", "SOXL", "SPXU", "SPY", "SQQQ", "TLT", "TQQQ", "TSLA", "UPRO", "XLE"
]

MODELS = {
    "g64_base_control": {
        "description": "Canonical 64-Unit Baseline Control",
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 16, "rollout": 64, "minibatch": 16,
        "trade_fraction": 0.50, "max_alloc": 0.95, "reward_kind": "log_return", "dd_penalty": 0.0,
        "entropy": 0.01, "temp": 1.0, "beta_min_shape": 0.1,
        "ret_windows": "[20, 60, 120]", "vol_windows": "[20]", "sma_windows": "[50, 200]",
    },
    "g64_macro_horizon_30d": {
        "description": "64-Unit Brain with 30-Day BPTT Sequence Memory",
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 30, "rollout": 60, "minibatch": 15,
        "trade_fraction": 0.50, "max_alloc": 0.95, "reward_kind": "log_return", "dd_penalty": 0.0,
        "entropy": 0.01, "temp": 1.0, "beta_min_shape": 0.1,
        "ret_windows": "[20, 60, 120]", "vol_windows": "[20]", "sma_windows": "[50, 200]",
    },
    "g64_hyper_burst_70": {
        "description": "64-Unit Brain with 70% Single-Trade Concentration & Sharp Impulse",
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 16, "rollout": 64, "minibatch": 16,
        "trade_fraction": 0.70, "max_alloc": 0.99, "reward_kind": "log_return", "dd_penalty": 0.0,
        "entropy": 0.015, "temp": 2.0, "beta_min_shape": 0.5,
        "ret_windows": "[20, 60, 120]", "vol_windows": "[20]", "sma_windows": "[50, 200]",
    },
    "g64_ath_breakout_sensor": {
        "description": "64-Unit Brain with 252-Day Secular ATH Breakout Sensors",
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 16, "rollout": 64, "minibatch": 16,
        "trade_fraction": 0.50, "max_alloc": 0.95, "reward_kind": "log_return", "dd_penalty": 0.0,
        "entropy": 0.01, "temp": 1.0, "beta_min_shape": 0.1,
        "ret_windows": "[5, 20, 60, 120, 252]", "vol_windows": "[10, 20, 60]", "sma_windows": "[20, 50, 200]",
    },
    "g64_sortino_shield": {
        "description": "64-Unit Brain with Active Quadratic Sortino Downside Penalty",
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 16, "rollout": 64, "minibatch": 16,
        "trade_fraction": 0.50, "max_alloc": 0.95, "reward_kind": "risk_adjusted_return", "dd_penalty": 0.05,
        "entropy": 0.01, "temp": 1.0, "beta_min_shape": 0.1,
        "ret_windows": "[20, 60, 120]", "vol_windows": "[20]", "sma_windows": "[50, 200]",
    },
    "g128_titan_deep_gru": {
        "description": "128-Unit Titan Deep Brain ([256, 128, 64], seq_len=20, entropy=0.03)",
        "memory_size": 128, "hidden": "[256, 128, 64]", "seq_len": 20, "rollout": 60, "minibatch": 15,
        "trade_fraction": 0.50, "max_alloc": 0.95, "reward_kind": "log_return", "dd_penalty": 0.0,
        "entropy": 0.03, "temp": 1.2, "beta_min_shape": 0.1,
        "ret_windows": "[20, 60, 120]", "vol_windows": "[20]", "sma_windows": "[50, 200]",
    },
}


def build_config_text(mod_key: str, seed: int = 41) -> str:
    template = BASE_TEMPLATE_PATH.read_text(encoding="utf-8")
    mod = MODELS[mod_key]
    
    # 1. 20-Asset Universe
    template = re.sub(r'assets\s*=\s*\[.*\]', f'assets = {json.dumps(ASSETS_20)}', template)
    template = re.sub(r'(?m)^path\s*=\s*"data/.*"$', 'path = "data/hyper/hyper_20_assets_2016_2026.csv"', template)
    
    # 2. Splits (2015 to 2026)
    template = re.sub(r'(?m)^start\s*=\s*"2010-01-04"$', 'start = "2015-01-05"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2014-10-31"$', 'end = "2022-12-30"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2014-11-03"$', 'start = "2023-01-03"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2015-11-13"$', 'end = "2024-08-09"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2015-11-16"$', 'start = "2024-08-12"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2016-11-25"$', 'end = "2026-08-14"', template)

    # 3. Seed and Checkpoint
    template = re.sub(r'(?m)^seed\s*=.*$', f"seed = {seed}", template)
    template = re.sub(
        r'(?m)^path\s*=\s*"checkpoints/.*"$',
        f'path = "checkpoints/matrix_v13_camber/{mod_key}-seed-{seed}.bin"',
        template
    )

    # 4. Neural Architecture
    template = re.sub(r'(?m)^hidden_layers\s*=.*$', f"hidden_layers = {mod['hidden']}", template)
    template = re.sub(r'(?m)^memory_size\s*=.*$', f"memory_size = {mod['memory_size']}", template)
    template = re.sub(r'(?m)^sequence_length\s*=.*$', f"sequence_length = {mod['seq_len']}", template)
    template = re.sub(r'(?m)^rollout_length\s*=.*$', f"rollout_length = {mod['rollout']}", template)
    template = re.sub(r'(?m)^minibatch_size\s*=.*$', f"minibatch_size = {mod['minibatch']}", template)

    # 5. Action Distribution & Sizing
    template = re.sub(r'(?m)^maximum_trade_fraction\s*=.*$', f"maximum_trade_fraction = {mod['trade_fraction']}", template)
    template = template.replace(
        "[brain.traits.maximum_allocation]\ninitial = 0.5\nminimum = 0.01\nmaximum = 1.0",
        f"[brain.traits.maximum_allocation]\ninitial = {mod['max_alloc']}\nminimum = 0.10\nmaximum = 1.0"
    )
    template = template.replace(
        "[brain.traits.action_temperature]\ninitial = 1.0\nminimum = 0.1\nmaximum = 3.0",
        f"[brain.traits.action_temperature]\ninitial = {mod['temp']}\nminimum = 0.1\nmaximum = 3.0"
    )

    # 6. Reward & Downside Penalty
    if mod['reward_kind'] == "log_return":
        template = template.replace('[reward]\nkind = "risk_adjusted_return"', '[reward]\nkind = "log_return"')
    else:
        template = template.replace('[reward]\nkind = "log_return"', '[reward]\nkind = "risk_adjusted_return"')
        
    template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', f"drawdown_penalty_weight = {mod['dd_penalty']}", template)
    template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.0001", template)
    template = re.sub(r'(?m)^entropy_coefficient\s*=.*$', f"entropy_coefficient = {mod['entropy']}", template)

    # 7. Corporate Actions Tolerance
    template = re.sub(r'(?m)^discontinuity_threshold\s*=.*$', "discontinuity_threshold = 0.85", template)

    # 8. Distinct Sensors
    obs_block = f"""
[observations]
schema = "default"
providers = ["market", "portfolio", "agent_state", "rolling_market_stats"]

[observations.rolling_market_stats]
enabled = true
return_windows = {mod['ret_windows']}
volatility_windows = {mod['vol_windows']}
sma_windows = {mod['sma_windows']}
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
            cfg_text = build_config_text(mod_key, seed)
            cfg_path = CONFIGS_DIR / f"{mod_key}-seed-{seed}.toml"
            cfg_path.write_text(cfg_text, encoding="utf-8")
            generated.append(cfg_path)
    print(f"Generated {len(generated)} experiment configs in {CONFIGS_DIR}")
    return generated


def run_local_job(mod_key: str, seed: int = 41) -> Dict[str, Any]:
    cfg_path = CONFIGS_DIR / f"{mod_key}-seed-{seed}.toml"
    cp_path = CHECKPOINTS_DIR / f"{mod_key}-seed-{seed}.bin"
    
    t0 = time.time()
    
    # 1. Train
    sim_proc = subprocess.run([
        str(BIN), "simulate",
        "--config", str(cfg_path),
        "--split", "training",
        "--checkpoint", str(cp_path)
    ], capture_output=True, text=True)
    
    if sim_proc.returncode != 0:
        return {"model": mod_key, "error": sim_proc.stderr}
        
    t_train = time.time() - t0
    
    # 2. Evaluate
    eval_proc = subprocess.run([
        str(BIN), "evaluate",
        "--config", str(cfg_path),
        "--checkpoint", str(cp_path),
        "--split", "test"
    ], capture_output=True, text=True)
    
    if eval_proc.returncode != 0:
        return {"model": mod_key, "error": eval_proc.stderr}
        
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
                "description": MODELS[mod_key]["description"],
                "return_%": sum(rets) / len(rets),
                "max_drawdown_%": sum(dds) / len(dds),
                "turnover_$": sum(tos) / len(tos),
                "friction_$": sum(fees) / len(fees),
                "train_time_s": t_train,
            }
            print(f"[✓] {mod_key:<26} -> Return: {res['return_%']:>+7.2f}%, Max DD: {res['max_drawdown_%']:>6.2f}%, Turnover: ${res['turnover_$']:.2f}")
            return res
    except Exception as e:
        return {"model": mod_key, "error": str(e)}

    return {"model": mod_key, "error": "No agent data"}


def check_camber_auth() -> bool:
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    res = subprocess.run(["camber", "me"], capture_output=True, text=True, env=env)
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cloud", action="store_true", help="Force Camber Cloud dispatch")
    args = parser.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_all_configs([41])

    is_camber_ready = check_camber_auth()
    print(f"\nCamber Cloud Status: {'READY / AUTHENTICATED' if is_camber_ready else 'NOT AUTHENTICATED (Run `camber login` or set CAMBER_API_KEY)'}")

    if args.cloud and not is_camber_ready:
        print("[!] Cannot dispatch to Camber Cloud: Camber login session is expired or CAMBER_API_KEY is not set.")
        print("[!] Please run `camber login` in your terminal or provide your API key.")
        return

    print(f"\n=== Running Matrix V13 Corrected Gauntlet across {args.workers} Parallel Workers ===")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_local_job, m): m for m in MODELS.keys()}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if "error" not in res:
                results.append(res)
            else:
                print(f"[!] Error in {res.get('model')}: {res.get('error')}")

    print("\n" + "="*120)
    print(f"  MATRIX V13 (CORRECTED GAUNTLET): APEX 64-UNIT VARIANTS & 128-UNIT TITAN LEADERBOARD")
    print("  Setup: 20-Asset Universe | Distinct Sensors | Sortino Brake | 70% Burst | 30d Memory")
    print("="*120)
    print(f"{'Model Configuration':<26} | {'2-Year Return (%)':<18} | {'Max DD (%)':<11} | {'Turnover ($)':<12} | {'Friction ($)':<12}")
    print("-"*120)
    
    results.sort(key=lambda x: x["return_%"], reverse=True)
    for r in results:
        print(f"{r['model']:<26} | {r['return_%']:>+17.2f}% | {r['max_drawdown_%']:>10.2f}% | ${r['turnover_$']:>10.2f} | ${r['friction_$']:>10.2f}")
    print("="*120)


if __name__ == "__main__":
    main()
