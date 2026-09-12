#!/usr/bin/env python3
"""
Run Growth Historical Stress Tests across 4 Real-World Historical Crash Events:
1. 2008 Global Financial Crisis (Oct 2007 - Mar 2009)
2. 2018 Volmageddon & Q4 Sell-off (Feb 2018 - Dec 2018)
3. 2020 COVID Flash Crash & Rebound (Feb 2020 - Aug 2020)
4. 2022 Inflation Bear Market (Jan 2022 - Dec 2022)
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

from experiments.generate_growth_historical_stress import EVENTS, GROWTH_MODELS, generate_all_configs

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "growth_historical_stress"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "growth_historical_stress"
REPORTS_DIR = ROOT / "reports" / "growth_historical_stress"
BIN = ROOT / "target" / "release" / "midas-cli"


def run_single_job(ev_key: str, mod_key: str, seed: int) -> Dict[str, Any]:
    cfg_path = CONFIGS_DIR / f"{ev_key}__{mod_key}-seed-{seed}.toml"
    cp_path = CHECKPOINTS_DIR / f"{ev_key}__{mod_key}-seed-{seed}.bin"
    rep_path = REPORTS_DIR / f"{ev_key}__{mod_key}-seed-{seed}.json"
    
    t0 = time.time()
    
    # 1. Train
    sim_proc = subprocess.run([
        str(BIN), "simulate",
        "--config", str(cfg_path),
        "--split", "training",
        "--checkpoint", str(cp_path)
    ], capture_output=True, text=True)
    
    if sim_proc.returncode != 0:
        return {"event": ev_key, "model": mod_key, "seed": seed, "error": sim_proc.stderr}
        
    t_train = time.time() - t0
    
    # 2. Evaluate on target historical crash event
    eval_proc = subprocess.run([
        str(BIN), "evaluate",
        "--config", str(cfg_path),
        "--checkpoint", str(cp_path),
        "--split", "test"
    ], capture_output=True, text=True)
    
    if eval_proc.returncode != 0:
        return {"event": ev_key, "model": mod_key, "seed": seed, "error": eval_proc.stderr}
        
    try:
        data = json.loads(eval_proc.stdout)
        agents = data.get("agents", [])
        if agents:
            rets = [a[1]["total_return"] * 100 for a in agents]
            dds = [a[1]["maximum_drawdown"] * 100 for a in agents]
            tos = [a[1]["turnover"] for a in agents]
            fees = [a[1]["total_fees"] + a[1]["spread_cost"] + a[1]["slippage_cost"] for a in agents]
            
            res = {
                "event": ev_key,
                "model": mod_key,
                "seed": seed,
                "return_%": sum(rets) / len(rets),
                "max_drawdown_%": sum(dds) / len(dds),
                "turnover_$": sum(tos) / len(tos),
                "friction_$": sum(fees) / len(fees),
                "train_time_s": t_train,
            }
            print(f"[✓] {ev_key} | {mod_key} -> Return: {res['return_%']:>+7.2f}%, Max DD: {res['max_drawdown_%']:>6.2f}%, Turnover: ${res['turnover_$']:.2f}")
            return res
    except Exception as e:
        return {"event": ev_key, "model": mod_key, "seed": seed, "error": str(e)}

    return {"event": ev_key, "model": mod_key, "seed": seed, "error": "No agent data"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[41])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_all_configs(args.seeds)

    tasks = []
    for ev_key in EVENTS.keys():
        for mod_key in GROWTH_MODELS.keys():
            for seed in args.seeds:
                tasks.append((ev_key, mod_key, seed))

    print(f"\n=== Running {len(tasks)} Growth Historical Stress Tests across {args.workers} CPU Workers ===")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_job, ev, mod, seed): (ev, mod, seed) for ev, mod, seed in tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if "error" not in res:
                results.append(res)
            else:
                print(f"[!] Error in {res['event']} | {res['model']}: {res.get('error')}")

    # Group and print by historical event
    for ev_key in EVENTS.keys():
        ev_desc = EVENTS[ev_key]["description"]
        ev_rows = [r for r in results if r["event"] == ev_key]
        if not ev_rows:
            continue
            
        print("\n" + "="*115)
        print(f"  HISTORICAL CRASH EVENT: {ev_desc.upper()}")
        print("="*115)
        print(f"{'Model Configuration':<36} | {'Test Return (%)':<16} | {'Max DD (%)':<11} | {'Turnover ($)':<12} | {'Friction ($)':<12}")
        print("-"*115)
        
        ev_rows.sort(key=lambda x: x["return_%"], reverse=True)
        for r in ev_rows:
            print(f"{r['model']:<36} | {r['return_%']:>+15.2f}% | {r['max_drawdown_%']:>10.2f}% | ${r['turnover_$']:>10.2f} | ${r['friction_$']:>10.2f}")
        print("="*115)


if __name__ == "__main__":
    main()
