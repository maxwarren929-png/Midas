#!/usr/bin/env python3
"""
Run 16-Asset Macro Universe Experiments across Parallel Local CPU Workers:
Assets: AAPL, BTC, GLD, IWM, MSFT, NVDA, QQQ, SLV, SOXL, SPXU, SPY, SQQQ, TLT, TQQQ, UPRO, XLE
Trains on 2015-2022 (8 years).
Evaluates on the Past 2 Years (2024-08-12 to 2026-08-14, 504 trading days).
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

from experiments.generate_matrix_macro16 import MODELS, generate_all_configs

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "camber_matrix_macro16"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "camber_matrix_macro16"
REPORTS_DIR = ROOT / "reports" / "camber_matrix_macro16"
BIN = ROOT / "target" / "release" / "midas-cli"


def train_and_eval_macro(mod_key: str, seed: int) -> Dict[str, Any]:
    cfg_path = CONFIGS_DIR / f"{mod_key}-seed-{seed}.toml"
    cp_path = CHECKPOINTS_DIR / f"{mod_key}-seed-{seed}.bin"
    rep_path = REPORTS_DIR / f"{mod_key}-seed-{seed}.json"
    
    print(f"[*] Starting 16-Asset training for {mod_key} (seed {seed})...")
    t0 = time.time()
    
    # 1. Train on 2015-2022 (8 years)
    sim_proc = subprocess.run([
        str(BIN), "simulate",
        "--config", str(cfg_path),
        "--split", "training",
        "--checkpoint", str(cp_path)
    ], capture_output=True, text=True)
    
    if sim_proc.returncode != 0:
        print(f"[!] Error training {mod_key}: {sim_proc.stderr}")
        return {"model": mod_key, "seed": seed, "error": sim_proc.stderr}
        
    t_train = time.time() - t0
    print(f"[+] Finished training {mod_key} (seed {seed}) in {t_train:.1f}s. Evaluating on Past 2 Years (16 Assets)...")
    
    # 2. Evaluate on Past 2 Years (2024-2026)
    eval_proc = subprocess.run([
        str(BIN), "evaluate",
        "--config", str(cfg_path),
        "--checkpoint", str(cp_path),
        "--split", "test"
    ], capture_output=True, text=True)
    
    if eval_proc.returncode != 0:
        print(f"[!] Error evaluating {mod_key}: {eval_proc.stderr}")
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

    print(f"\n=== Running {len(tasks)} 16-Asset Macro Models across {args.workers} CPU Workers ===")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(train_and_eval_macro, mod, seed): (mod, seed) for mod, seed in tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if "error" not in res:
                results.append(res)
            else:
                print(f"[!] Error in {res['model']}: {res.get('error')}")

    # Leaderboard
    print("\n" + "="*115)
    print(f"  16-ASSET MACRO LEADERBOARD: PAST 2 YEARS (2024-08-12 TO 2026-08-14 | 504 TRADING DAYS)")
    print("  Includes: TQQQ (3x Tech), SOXL (3x Semis), UPRO (3x S&P), SQQQ (3x Short), SPXU (3x Short), BTC, GLD")
    print("="*115)
    print(f"{'Model Configuration':<36} | {'Test Return (%)':<16} | {'Max DD (%)':<11} | {'Turnover ($)':<12} | {'Friction ($)':<12}")
    print("-"*115)
    
    results.sort(key=lambda x: x["return_%"], reverse=True)
    for r in results:
        print(f"{r['model']:<36} | {r['return_%']:>+15.2f}% | {r['max_drawdown_%']:>10.2f}% | ${r['turnover_$']:>10.2f} | ${r['friction_$']:>10.2f}")
    print("="*115)


if __name__ == "__main__":
    main()
