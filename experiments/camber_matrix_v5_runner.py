#!/usr/bin/env python3
"""
Camber Matrix V5 Runner:
Trains and evaluates the 5 Advanced Models across 4 Real-World Historical Crash Events:
  1. 2008 Global Financial Crisis (Oct 2007 - Mar 2009)
  2. 2018 Volmageddon & Q4 Sell-off (Feb 2018 - Dec 2018)
  3. 2020 COVID Flash Crash & Rebound (Feb 2020 - Aug 2020)
  4. 2022 Inflation Bear Market (Jan 2022 - Dec 2022)
"""

from __future__ import annotations

import argparse
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

from experiments.generate_matrix_v5 import EVENTS, MODELS, generate_all_configs

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "camber_matrix_v5"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "camber_matrix_v5"
REPORTS_DIR = ROOT / "reports" / "camber_matrix_v5"
BIN = ROOT / "target" / "release" / "midas-cli"


def run_camber_cmd(cmd: List[str]) -> str:
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def sync_configs_to_stash():
    print("Syncing Matrix V5 configs to Camber Stash...")
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    cmd = [
        "camber", "stash", "cp", "-r",
        str(CONFIGS_DIR),
        "stash://kohosting/projects/midas/configs/camber_matrix_v5"
    ]
    subprocess.run(cmd, check=True, env=env)
    print("Configs synced to Stash successfully!")


@dataclass
class JobRecord:
    event: str
    model: str
    seed: int
    config_rel_path: str
    checkpoint_rel_path: str
    report_rel_path: str
    job_id: Optional[str] = None
    status: str = "PENDING"


def dispatch_all_jobs(seeds: List[int], size: str = "xxsmall") -> List[JobRecord]:
    records: List[JobRecord] = []
    
    for ev_key in EVENTS.keys():
        for mod_key in MODELS.keys():
            for seed in seeds:
                config_rel = f"configs/camber_matrix_v5/{ev_key}__{mod_key}-seed-{seed}.toml"
                checkpoint_rel = f"checkpoints/camber_matrix_v5/{ev_key}__{mod_key}-seed-{seed}.bin"
                report_rel = f"reports/camber_matrix_v5/{ev_key}__{mod_key}-seed-{seed}.json"
                
                records.append(JobRecord(
                    event=ev_key,
                    model=mod_key,
                    seed=seed,
                    config_rel_path=config_rel,
                    checkpoint_rel_path=checkpoint_rel,
                    report_rel_path=report_rel
                ))
                
    print(f"\nDispatching {len(records)} jobs to Camber Cloud (Size: {size})...")
    for rec in records:
        cmd_str = (
            f"mkdir -p checkpoints/camber_matrix_v5 reports/camber_matrix_v5 && "
            f"chmod +x ./midas-cli && "
            f"./midas-cli simulate --config {rec.config_rel_path} --split training --checkpoint {rec.checkpoint_rel_path} && "
            f"./midas-cli evaluate --config {rec.config_rel_path} --checkpoint {rec.checkpoint_rel_path} --split test > {rec.report_rel_path}"
        )
        
        camber_cmd = [
            "camber", "job", "create",
            "--engine", "base",
            "--size", size,
            "--path", "stash://kohosting/projects/midas/",
            "--cmd", cmd_str
        ]
        
        for attempt in range(5):
            try:
                out = run_camber_cmd(camber_cmd)
                match = re.search(r"Job ID:\s*(\d+)", out)
                if match:
                    rec.job_id = match.group(1)
                    rec.status = "SUBMITTED"
                    print(f"  [+] Dispatched {rec.event} | {rec.model} (seed {rec.seed}) -> Job ID: {rec.job_id}")
                    break
            except Exception as e:
                time.sleep(2 * (attempt + 1))
        time.sleep(0.5)
            
    return records


def monitor_jobs(records: List[JobRecord], poll_interval: int = 15) -> None:
    print("\nMonitoring parallel historical stress test jobs on Camber Cloud...")
    pending = [r for r in records if r.job_id]
    
    while pending:
        time.sleep(poll_interval)
        still_running = []
        for r in pending:
            try:
                out = run_camber_cmd(["camber", "job", "get", r.job_id])
                status_match = re.search(r"Status:\s*([A-Z_]+)", out)
                if status_match:
                    st = status_match.group(1)
                    r.status = st
                    if st in ["COMPLETED", "FAILED", "CANCELLED"]:
                        print(f"  [*] Job {r.job_id} ({r.event} | {r.model}) reached status: {st}")
                    else:
                        still_running.append(r)
                else:
                    still_running.append(r)
            except Exception:
                still_running.append(r)
                
        pending = still_running
        if pending:
            print(f"  --> {len(pending)} / {len(records)} jobs still executing in the cloud...")


def sync_and_evaluate(records: List[JobRecord]):
    print("\nSyncing all historical stress test checkpoints from Camber Stash...")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    
    subprocess.run([
        "camber", "stash", "cp", "-r",
        "stash://kohosting/projects/midas/checkpoints/camber_matrix_v5/",
        str(CHECKPOINTS_DIR)
    ], check=False, env=env)
    
    # Evaluate every event
    summary = []
    for r in records:
        cp = CHECKPOINTS_DIR / f"{r.event}__{r.model}-seed-{r.seed}.bin"
        cfg = CONFIGS_DIR / f"{r.event}__{r.model}-seed-{r.seed}.toml"
        if not cp.exists() or not cfg.exists():
            continue
            
        proc = subprocess.run([str(BIN), "evaluate", "--config", str(cfg), "--checkpoint", str(cp), "--split", "test"], capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"Error evaluating {r.event}__{r.model}: {proc.stderr}")
            continue
            
        try:
            data = json.loads(proc.stdout)
            agents = data.get("agents", [])
            if agents:
                rets = [a[1]["total_return"] * 100 for a in agents]
                dds = [a[1]["maximum_drawdown"] * 100 for a in agents]
                tos = [a[1]["turnover"] for a in agents]
                fees = [a[1]["total_fees"] + a[1]["spread_cost"] + a[1]["slippage_cost"] for a in agents]
                
                summary.append({
                    "event": r.event,
                    "model": r.model,
                    "seed": r.seed,
                    "return_%": sum(rets) / len(rets),
                    "max_drawdown_%": sum(dds) / len(dds),
                    "turnover_$": sum(tos) / len(tos),
                    "friction_$": sum(fees) / len(fees),
                })
        except Exception as e:
            print(f"Error parsing JSON for {r.event}__{r.model}: {e}")

    # Print results grouped by historical crash event
    for ev_key in EVENTS.keys():
        ev_desc = EVENTS[ev_key]["description"]
        ev_rows = [r for r in summary if r["event"] == ev_key]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[41])
    parser.add_argument("--size", default="xxsmall")
    args = parser.parse_args()

    print(f"=== Starting Camber Cloud Matrix V5 (All 4 Historical Crash Events) ===")
    generate_all_configs(args.seeds)
    sync_configs_to_stash()
    records = dispatch_all_jobs(args.seeds, size=args.size)
    monitor_jobs(records)
    sync_and_evaluate(records)


if __name__ == "__main__":
    main()
