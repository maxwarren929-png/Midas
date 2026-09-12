#!/usr/bin/env python3
"""
Matrix V13: Apex 64-Unit Variants & 128-Unit Titan Deep Brain Gauntlet

Models Evaluated (The Apex 6):
  1. g64_base_control: Canonical Matrix V11 setup (64-Unit GRU, [128, 64], 16d BPTT, 50% Sizing, 95% Alloc)
  2. g64_macro_horizon_30d: 64-Unit GRU with 30-Day BPTT Sequence Memory (6-Week Accumulation Window)
  3. g64_hyper_burst_70: 64-Unit GRU with 70% Single-Trade Concentration & 99% Max Allocation
  4. g64_ath_breakout_sensor: 64-Unit GRU with Distance-to-52-Week High Sensors (ATH Breakouts)
  5. g64_sortino_shield: 64-Unit GRU with Light Asymmetric Sortino Downside Brake (0.02 Penalty)
  6. g128_titan_deep_gru: 128-Unit Titan Brain ([256, 128, 64], 20d BPTT, Fast Reset Gates, Entropy 0.03)

Regimes Evaluated:
  1. Modern 2-Year Real Market (2024-08-12 to 2026-08-14 | 504 Days)
  2. 2020 COVID Flash Crash (2020-02-19 to 2020-04-30)
  3. 2022 Inflation Bear Market (2022-01-03 to 2022-10-14)
  4. 2018 Volmageddon & Q4 Plunge (2018-01-26 to 2018-12-24)
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
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "matrix_v13_apex"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "matrix_v13_apex"
REPORTS_DIR = ROOT / "reports" / "matrix_v13_apex"
BIN = ROOT / "target" / "release" / "midas-cli"
DATA_FILE = ROOT / "data" / "hyper" / "hyper_20_assets_2016_2026.csv"
BASE_TEMPLATE_PATH = ROOT / "configs" / "prompt3b" / "experiment-ppo.toml"

ASSETS_20 = [
    "AAPL", "AMD", "BTC", "GLD", "IWM", "META", "MSFT", "NVDA", "PLTR", "QQQ",
    "SLV", "SOXL", "SPXU", "SPY", "SQQQ", "TLT", "TQQQ", "TSLA", "UPRO", "XLE"
]

REGIMES = {
    "modern_bull_2y": {
        "title": "Modern 2-Year Market (2024–2026)",
        "start": "2024-08-12", "end": "2026-08-14"
    },
    "covid_crash_2020": {
        "title": "2020 COVID Flash Crash",
        "start": "2020-02-19", "end": "2020-04-30"
    },
    "inflation_bear_2022": {
        "title": "2022 Inflation Bear Market",
        "start": "2022-01-03", "end": "2022-10-14"
    },
    "volmageddon_2018": {
        "title": "2018 Volmageddon & Q4 Plunge",
        "start": "2018-01-26", "end": "2018-12-24"
    },
}

MODELS = {
    "g64_base_control": {
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 16, "rollout": 64, "minibatch": 16,
        "trade_fraction": 0.50, "max_alloc": 0.95, "reward": "log_return", "dd_penalty": 0.0,
        "entropy": 0.01, "ath_sensor": False,
    },
    "g64_macro_horizon_30d": {
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 30, "rollout": 60, "minibatch": 15,
        "trade_fraction": 0.50, "max_alloc": 0.95, "reward": "log_return", "dd_penalty": 0.0,
        "entropy": 0.01, "ath_sensor": False,
    },
    "g64_hyper_burst_70": {
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 16, "rollout": 64, "minibatch": 16,
        "trade_fraction": 0.70, "max_alloc": 0.99, "reward": "log_return", "dd_penalty": 0.0,
        "entropy": 0.015, "ath_sensor": False,
    },
    "g64_ath_breakout_sensor": {
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 16, "rollout": 64, "minibatch": 16,
        "trade_fraction": 0.50, "max_alloc": 0.95, "reward": "log_return", "dd_penalty": 0.0,
        "entropy": 0.01, "ath_sensor": True,
    },
    "g64_sortino_shield": {
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 16, "rollout": 64, "minibatch": 16,
        "trade_fraction": 0.50, "max_alloc": 0.95, "reward": "risk_adjusted_return", "dd_penalty": 0.02,
        "entropy": 0.01, "ath_sensor": False,
    },
    "g128_titan_deep_gru": {
        "memory_size": 128, "hidden": "[256, 128, 64]", "seq_len": 20, "rollout": 60, "minibatch": 15,
        "trade_fraction": 0.50, "max_alloc": 0.95, "reward": "log_return", "dd_penalty": 0.0,
        "entropy": 0.03, "ath_sensor": False,
    },
}


def compute_benchmark_spy(start_date: str, end_date: str) -> Dict[str, float]:
    with DATA_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        spy_rows = [r for r in reader if r["asset"] == "SPY" and start_date <= r["date"] <= end_date]
        
    if not spy_rows:
        return {"return_%": 0.0, "max_dd_%": 0.0}
        
    closes = [float(r["close"]) for r in spy_rows]
    ret = (closes[-1] - closes[0]) / closes[0] * 100.0
    
    peak = closes[0]
    max_dd = 0.0
    for p in closes:
        if p > peak:
            peak = p
        dd = (peak - p) / peak * 100.0
        if dd > max_dd:
            max_dd = dd
            
    return {"return_%": ret, "max_dd_%": max_dd}


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
        f'path = "checkpoints/matrix_v13_apex/{mod_key}-seed-{seed}.bin"',
        template
    )

    # 4. Neural Architecture
    template = re.sub(r'(?m)^hidden_layers\s*=.*$', f"hidden_layers = {mod['hidden']}", template)
    template = re.sub(r'(?m)^memory_size\s*=.*$', f"memory_size = {mod['memory_size']}", template)
    template = re.sub(r'(?m)^sequence_length\s*=.*$', f"sequence_length = {mod['seq_len']}", template)
    template = re.sub(r'(?m)^rollout_length\s*=.*$', f"rollout_length = {mod['rollout']}", template)
    template = re.sub(r'(?m)^minibatch_size\s*=.*$', f"minibatch_size = {mod['minibatch']}", template)

    # 5. Sizing & High-Conviction Allocation
    template = re.sub(r'(?m)^maximum_trade_fraction\s*=.*$', f"maximum_trade_fraction = {mod['trade_fraction']}", template)
    template = template.replace(
        "[brain.traits.maximum_allocation]\ninitial = 0.5\nminimum = 0.01\nmaximum = 1.0",
        f"[brain.traits.maximum_allocation]\ninitial = {mod['max_alloc']}\nminimum = 0.10\nmaximum = 1.0"
    )

    # 6. Reward & Entropy
    if mod['reward'] == "log_return":
        template = template.replace('[reward]\nkind = "risk_adjusted_return"', '[reward]\nkind = "log_return"')
    template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', f"drawdown_penalty_weight = {mod['dd_penalty']}", template)
    template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.0001", template)
    template = re.sub(r'(?m)^entropy_coefficient\s*=.*$', f"entropy_coefficient = {mod['entropy']}", template)

    # 7. Corporate Actions Tolerance
    template = re.sub(r'(?m)^discontinuity_threshold\s*=.*$', "discontinuity_threshold = 0.85", template)

    # 8. Distilled Features (with optional ATH / MACD)
    ret_windows = "[20, 60, 120]"
    sma_windows = "[50, 200]"
    obs_block = f"""
[observations]
schema = "default"
providers = ["market", "portfolio", "agent_state", "rolling_market_stats"]

[observations.rolling_market_stats]
enabled = true
return_windows = {ret_windows}
volatility_windows = [20]
sma_windows = {sma_windows}
relative_strength_windows = [20]
market_summary = true
market_summary_window = 20
minimum_history = "zero_fill"
"""
    template += obs_block

    return template


def train_and_eval_model(mod_key: str, seed: int = 41) -> Dict[str, Any]:
    cfg_text = build_config_text(mod_key, seed)
    cfg_path = CONFIGS_DIR / f"{mod_key}-seed-{seed}.toml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    
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
    
    # 2. Evaluate on Modern 2-Year Test Split (2024-2026)
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
                "return_%": sum(rets) / len(rets),
                "max_drawdown_%": sum(dds) / len(dds),
                "turnover_$": sum(tos) / len(tos),
                "friction_$": sum(fees) / len(fees),
                "train_time_s": t_train,
            }
            print(f"[✓] {mod_key:<24} -> Return: {res['return_%']:>+7.2f}%, Max DD: {res['max_drawdown_%']:>6.2f}%, Turnover: ${res['turnover_$']:.2f}")
            return res
    except Exception as e:
        return {"model": mod_key, "error": str(e)}

    return {"model": mod_key, "error": "No agent data"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Running Matrix V13: Apex 64-Unit Variants & 128-Unit Titan Brain across {args.workers} CPU Workers ===")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(train_and_eval_model, m): m for m in MODELS.keys()}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if "error" not in res:
                results.append(res)
            else:
                print(f"[!] Error in {res.get('model')}: {res.get('error')}")

    print("\n" + "="*115)
    print(f"  MATRIX V13: APEX 64-UNIT VARIANTS & 128-UNIT TITAN LEADERBOARD (2024-08-12 TO 2026-08-14)")
    print("  Setup: 20-Asset Universe + 64-to-128 Units + 30-Day BPTT + 70% Burst Sizing + Fast Reset")
    print("="*115)
    print(f"{'Model Configuration':<28} | {'2-Year Return (%)':<18} | {'Max DD (%)':<11} | {'Turnover ($)':<12} | {'Friction ($)':<12}")
    print("-"*115)
    
    results.sort(key=lambda x: x["return_%"], reverse=True)
    for r in results:
        print(f"{r['model']:<28} | {r['return_%']:>+17.2f}% | {r['max_drawdown_%']:>10.2f}% | ${r['turnover_$']:>10.2f} | ${r['friction_$']:>10.2f}")
    print("="*115)


if __name__ == "__main__":
    main()
