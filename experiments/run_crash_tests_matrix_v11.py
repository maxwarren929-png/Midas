#!/usr/bin/env python3
"""
Historical Crash Stress Tests for Matrix V11 Models (20-Asset Hyper Universe)

Walk-Forward Training & Blind Crash Evaluation across 3 Historic Market Panics:
  1. 💥 Crash 1: 2018 Volmageddon & Q4 Market Plunge
       - Trained on: 2015-01-05 to 2017-12-29 (3 Years Prior)
       - Evaluated on: 2018-01-26 to 2018-12-24 (The Crash)
  2. 🦠 Crash 2: 2020 COVID Flash Crash
       - Trained on: 2015-01-05 to 2019-12-31 (5 Years Prior)
       - Evaluated on: 2020-02-19 to 2020-04-30 (The COVID Crash)
  3. 📉 Crash 3: 2022 Inflation & Federal Reserve Bear Market
       - Trained on: 2015-01-05 to 2021-12-31 (7 Years Prior)
       - Evaluated on: 2022-01-03 to 2022-10-14 (The Bear Market)

Models Evaluated:
  1. g64_deep_macro_gru (+72.15% All-Time Champion)
  2. g1_aggressive_mega_15d
  3. m16_momentum_rotator
  4. g32_conviction_gru
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
CONFIGS_DIR = ROOT / "configs" / "matrix_v11_crashes"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "matrix_v11_crashes"
REPORTS_DIR = ROOT / "reports" / "matrix_v11_crashes"
BIN = ROOT / "target" / "release" / "midas-cli"
DATA_FILE = ROOT / "data" / "hyper" / "hyper_20_assets_2016_2026.csv"
BASE_TEMPLATE_PATH = ROOT / "configs" / "prompt3b" / "experiment-ppo.toml"

ASSETS_20 = [
    "AAPL", "AMD", "BTC", "GLD", "IWM", "META", "MSFT", "NVDA", "PLTR", "QQQ",
    "SLV", "SOXL", "SPXU", "SPY", "SQQQ", "TLT", "TQQQ", "TSLA", "UPRO", "XLE"
]

CRASHES = {
    "2018_volmageddon": {
        "title": "2018 Volmageddon & Q4 Plunge",
        "t_start": "2015-01-05", "t_end": "2017-12-29",
        "v_start": "2018-01-02", "v_end": "2018-01-25",
        "test_start": "2018-01-26", "test_end": "2018-12-24",
    },
    "2020_covid": {
        "title": "2020 COVID Flash Crash",
        "t_start": "2015-01-05", "t_end": "2019-12-31",
        "v_start": "2020-01-02", "v_end": "2020-02-18",
        "test_start": "2020-02-19", "test_end": "2020-04-30",
    },
    "2022_inflation": {
        "title": "2022 Inflation & Rate Hike Bear",
        "t_start": "2015-01-05", "t_end": "2021-12-24",
        "v_start": "2021-12-27", "v_end": "2021-12-31",
        "test_start": "2022-01-03", "test_end": "2022-10-14",
    },
}

MODELS = {
    "g64_deep_macro_gru": {
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 16, "rollout": 64, "minibatch": 16,
    },
    "g1_aggressive_mega_15d": {
        "memory_size": 8, "hidden": "[32, 16]", "seq_len": 15, "rollout": 60, "minibatch": 15,
    },
    "m16_momentum_rotator": {
        "memory_size": 16, "hidden": "[32, 16]", "seq_len": 15, "rollout": 60, "minibatch": 15,
    },
    "g32_conviction_gru": {
        "memory_size": 32, "hidden": "[64, 32]", "seq_len": 16, "rollout": 64, "minibatch": 16,
    },
}


def compute_benchmark_spy_stats(start_date: str, end_date: str) -> Dict[str, float]:
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


def build_config_text(crash_key: str, mod_key: str, seed: int = 41) -> str:
    template = BASE_TEMPLATE_PATH.read_text(encoding="utf-8")
    mod = MODELS[mod_key]
    crash = CRASHES[crash_key]
    
    # 1. 20-Asset Universe & Path
    template = re.sub(r'assets\s*=\s*\[.*\]', f'assets = {json.dumps(ASSETS_20)}', template)
    template = re.sub(r'(?m)^path\s*=\s*"data/.*"$', 'path = "data/hyper/hyper_20_assets_2016_2026.csv"', template)
    
    # 2. Splits
    template = re.sub(r'(?m)^start\s*=\s*"2010-01-04"$', f'start = "{crash["t_start"]}"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2014-10-31"$', f'end = "{crash["t_end"]}"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2014-11-03"$', f'start = "{crash["v_start"]}"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2015-11-13"$', f'end = "{crash["v_end"]}"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2015-11-16"$', f'start = "{crash["test_start"]}"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2016-11-25"$', f'end = "{crash["test_end"]}"', template)

    # 3. Seed & Checkpoint
    template = re.sub(r'(?m)^seed\s*=.*$', f"seed = {seed}", template)
    template = re.sub(
        r'(?m)^path\s*=\s*"checkpoints/.*"$',
        f'path = "checkpoints/matrix_v11_crashes/{crash_key}__{mod_key}-seed-{seed}.bin"',
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
    template = template.replace(
        "[brain.traits.maximum_allocation]\ninitial = 0.5\nminimum = 0.01\nmaximum = 1.0",
        "[brain.traits.maximum_allocation]\ninitial = 0.95\nminimum = 0.10\nmaximum = 1.0"
    )

    # 6. Reward
    template = template.replace('[reward]\nkind = "risk_adjusted_return"', '[reward]\nkind = "log_return"')
    template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.0", template)
    template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.0001", template)

    # 7. Corporate Actions Tolerance
    template = re.sub(r'(?m)^discontinuity_threshold\s*=.*$', "discontinuity_threshold = 0.85", template)

    # 8. Distilled Features
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


def run_crash_job(crash_key: str, mod_key: str, seed: int = 41) -> Dict[str, Any]:
    cfg_text = build_config_text(crash_key, mod_key, seed)
    cfg_path = CONFIGS_DIR / f"{crash_key}__{mod_key}-seed-{seed}.toml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    
    cp_path = CHECKPOINTS_DIR / f"{crash_key}__{mod_key}-seed-{seed}.bin"
    
    # 1. Train up to crash date
    sim_proc = subprocess.run([
        str(BIN), "simulate",
        "--config", str(cfg_path),
        "--split", "training",
        "--checkpoint", str(cp_path)
    ], capture_output=True, text=True)
    
    if sim_proc.returncode != 0:
        return {"crash": crash_key, "model": mod_key, "error": sim_proc.stderr}
        
    # 2. Evaluate blind on the crash
    eval_proc = subprocess.run([
        str(BIN), "evaluate",
        "--config", str(cfg_path),
        "--checkpoint", str(cp_path),
        "--split", "test"
    ], capture_output=True, text=True)
    
    if eval_proc.returncode != 0:
        return {"crash": crash_key, "model": mod_key, "error": eval_proc.stderr}
        
    try:
        data = json.loads(eval_proc.stdout)
        agents = data.get("agents", [])
        if agents:
            rets = [a[1]["total_return"] * 100 for a in agents]
            dds = [a[1]["maximum_drawdown"] * 100 for a in agents]
            tos = [a[1]["turnover"] for a in agents]
            fees = [a[1]["total_fees"] + a[1]["spread_cost"] + a[1]["slippage_cost"] for a in agents]
            
            res = {
                "crash": crash_key,
                "crash_title": CRASHES[crash_key]["title"],
                "model": mod_key,
                "return_%": sum(rets) / len(rets),
                "max_drawdown_%": sum(dds) / len(dds),
                "turnover_$": sum(tos) / len(tos),
                "friction_$": sum(fees) / len(fees),
            }
            print(f"[✓] {CRASHES[crash_key]['title']} | {mod_key} -> Return: {res['return_%']:>+7.2f}%, Max DD: {res['max_drawdown_%']:>6.2f}%")
            return res
    except Exception as e:
        return {"crash": crash_key, "model": mod_key, "error": str(e)}

    return {"crash": crash_key, "model": mod_key, "error": "No agent data"}


def main():
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    for c_key in CRASHES.keys():
        for m_key in MODELS.keys():
            tasks.append((c_key, m_key))
            
    print(f"=== Training and Stress-Testing 12 Models across 3 Historic Market Crashes (6 CPU Workers) ===")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_crash_job, c, m): (c, m) for c, m in tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if "error" not in res:
                results.append(res)
            else:
                print(f"[!] Error in {res.get('crash')} | {res.get('model')}: {res.get('error')}")

    for c_key, c_info in CRASHES.items():
        spy_stats = compute_benchmark_spy_stats(c_info["test_start"], c_info["test_end"])
        crash_results = [r for r in results if r["crash"] == c_key]
        crash_results.sort(key=lambda x: x["return_%"], reverse=True)
        
        print("\n" + "="*115)
        print(f"  CRASH EVENT: {c_info['title'].upper()} ({c_info['test_start']} to {c_info['test_end']})")
        print(f"  MARKET BASELINE: S&P 500 (SPY) Return: {spy_stats['return_%']:>+.2f}% | Max Drawdown: {spy_stats['max_dd_%']:.2f}%")
        print("="*115)
        print(f"{'Model Configuration':<32} | {'Return (%)':<12} | {'Max DD (%)':<11} | {'Net Alpha vs SPY':<18} | {'Turnover ($)':<12}")
        print("-"*115)
        
        for r in crash_results:
            alpha = r["return_%"] - spy_stats["return_%"]
            print(f"{r['model']:<32} | {r['return_%']:>+11.2f}% | {r['max_drawdown_%']:>10.2f}% | {alpha:>+16.2f}% | ${r['turnover_$']:>10.2f}")
        print("="*115)


if __name__ == "__main__":
    main()
