#!/usr/bin/env python3
"""
Run Matrix V9: 5-Way Ablation & Synergy Experiment across Parallel Local CPU Workers.
Models:
  1. m16_momentum_rotator
  2. g2_high_conviction_gru
  3. g1_aggressive_mega_15d
  4. m2_mega_combo_15d

Runs:
  - Run 1: Setting 1 ON (Asymmetric Sortino Reward)
  - Run 2: Setting 2 ON (High-Conviction Action Concentration: 50% Trade Size, 95% Max Allocation)
  - Run 3: Setting 3 ON (Trend Strength & Full SMA Divergence Sensors: SMA 10/20/50/100/200)
  - Run 4: Setting 4 ON (20-Asset Hyper-Growth Universe: NVDL 2x, PLTR, TSLA, ETH, etc.)
  - Run 5: ALL 4 SETTINGS ON COMBINED ("God-Mode / All-Engines-Fire")
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from experiments.generate_matrix_v9_ablation import MODELS, RUNS, generate_all_configs

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "matrix_v9_ablation"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "matrix_v9_ablation"
REPORTS_DIR = ROOT / "reports" / "matrix_v9_ablation"
BIN = ROOT / "target" / "release" / "midas-cli"


def run_single_ablation(run_key: str, mod_key: str, seed: int) -> Dict[str, Any]:
    cfg_path = CONFIGS_DIR / f"{run_key}__{mod_key}-seed-{seed}.toml"
    cp_path = CHECKPOINTS_DIR / f"{run_key}__{mod_key}-seed-{seed}.bin"
    rep_path = REPORTS_DIR / f"{run_key}__{mod_key}-seed-{seed}.json"
    
    t0 = time.time()
    
    # 1. Train
    sim_proc = subprocess.run([
        str(BIN), "simulate",
        "--config", str(cfg_path),
        "--split", "training",
        "--checkpoint", str(cp_path)
    ], capture_output=True, text=True)
    
    if sim_proc.returncode != 0:
        return {"run": run_key, "model": mod_key, "seed": seed, "error": sim_proc.stderr}
        
    t_train = time.time() - t0
    
    # 2. Evaluate on Past 2 Years
    eval_proc = subprocess.run([
        str(BIN), "evaluate",
        "--config", str(cfg_path),
        "--checkpoint", str(cp_path),
        "--split", "test"
    ], capture_output=True, text=True)
    
    if eval_proc.returncode != 0:
        return {"run": run_key, "model": mod_key, "seed": seed, "error": eval_proc.stderr}
        
    try:
        data = json.loads(eval_proc.stdout)
        agents = data.get("agents", [])
        if agents:
            rets = [a[1]["total_return"] * 100 for a in agents]
            dds = [a[1]["maximum_drawdown"] * 100 for a in agents]
            tos = [a[1]["turnover"] for a in agents]
            fees = [a[1]["total_fees"] + a[1]["spread_cost"] + a[1]["slippage_cost"] for a in agents]
            
            res = {
                "run": run_key,
                "model": mod_key,
                "seed": seed,
                "return_%": sum(rets) / len(rets),
                "max_drawdown_%": sum(dds) / len(dds),
                "turnover_$": sum(tos) / len(tos),
                "friction_$": sum(fees) / len(fees),
                "train_time_s": t_train,
            }
            print(f"[✓] {run_key} | {mod_key} -> Return: {res['return_%']:>+7.2f}%, Max DD: {res['max_drawdown_%']:>6.2f}%, Turnover: ${res['turnover_$']:.2f}")
            return res
    except Exception as e:
        return {"run": run_key, "model": mod_key, "seed": seed, "error": str(e)}

    return {"run": run_key, "model": mod_key, "seed": seed, "error": "No agent data"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[41])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_all_configs(args.seeds)

    tasks = []
    for run_key in RUNS.keys():
        for mod_key in MODELS.keys():
            for seed in args.seeds:
                tasks.append((run_key, mod_key, seed))

    print(f"\n=== Running {len(tasks)} Ablation & Synergy Runs across {args.workers} CPU Workers ===")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_ablation, r, m, s): (r, m, s) for r, m, s in tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if "error" not in res:
                results.append(res)
            else:
                print(f"[!] Error in {res.get('run')} | {res.get('model')}: {res.get('error')}")

    # Group and print by Run setting
    for run_key, run_desc in RUNS.items():
        run_rows = [r for r in results if r["run"] == run_key]
        if not run_rows:
            continue
            
        print("\n" + "="*115)
        print(f"  ABLATION RUN: {run_desc.upper()}")
        print("="*115)
        print(f"{'Model Configuration':<36} | {'Test Return (%)':<16} | {'Max DD (%)':<11} | {'Turnover ($)':<12} | {'Friction ($)':<12}")
        print("-"*115)
        
        run_rows.sort(key=lambda x: x["return_%"], reverse=True)
        for r in run_rows:
            print(f"{r['model']:<36} | {r['return_%']:>+15.2f}% | {r['max_drawdown_%']:>10.2f}% | ${r['turnover_$']:>10.2f} | ${r['friction_$']:>10.2f}")
        print("="*115)

    # Master Comparative Summary Across All 5 Runs
    print("\n" + "="*115)
    print("  🏆 MASTER COMPARATIVE SUMMARY: TOP MODEL PER RUN")
    print("="*115)
    for run_key, run_desc in RUNS.items():
        run_rows = [r for r in results if r["run"] == run_key]
        if run_rows:
            best = max(run_rows, key=lambda x: x["return_%"])
            print(f"{run_desc:<65} -> Best: {best['model']:<24} ({best['return_%']:>+7.2f}%, Max DD: {best['max_drawdown_%']:.2f}%)")
    print("="*115)


if __name__ == "__main__":
    main()
