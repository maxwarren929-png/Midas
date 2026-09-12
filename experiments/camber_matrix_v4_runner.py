#!/usr/bin/env python3
"""
Camber Cloud Matrix V4 Runner:
Runs the 5 Advanced Models across 3 Non-Bull Market Scenarios (15 total parallel runs).
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

from experiments.generate_matrix_v4 import SCENARIOS, MODELS, generate_all_configs

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "camber_matrix_v4"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "camber_matrix_v4"
REPORTS_DIR = ROOT / "reports" / "camber_matrix_v4"
BIN = ROOT / "target" / "release" / "midas-cli"


def run_camber_cmd(cmd: List[str]) -> str:
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def sync_configs_to_stash():
    print("Syncing Matrix V4 configs to Camber Stash...")
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    cmd = [
        "camber", "stash", "cp", "-r",
        str(CONFIGS_DIR),
        "stash://kohosting/projects/midas/configs/camber_matrix_v4"
    ]
    subprocess.run(cmd, check=True, env=env)
    print("Configs synced to Stash successfully!")


@dataclass
class JobRecord:
    scenario: str
    model: str
    seed: int
    config_rel_path: str
    checkpoint_rel_path: str
    report_rel_path: str
    job_id: Optional[str] = None
    status: str = "PENDING"


def dispatch_all_jobs(seeds: List[int], size: str = "xxsmall") -> List[JobRecord]:
    records: List[JobRecord] = []
    
    for sc_key in SCENARIOS.keys():
        for mod_key in MODELS.keys():
            for seed in seeds:
                config_rel = f"configs/camber_matrix_v4/{sc_key}__{mod_key}-seed-{seed}.toml"
                checkpoint_rel = f"checkpoints/camber_matrix_v4/{sc_key}__{mod_key}-seed-{seed}.bin"
                report_rel = f"reports/camber_matrix_v4/{sc_key}__{mod_key}-seed-{seed}.json"
                
                records.append(JobRecord(
                    scenario=sc_key,
                    model=mod_key,
                    seed=seed,
                    config_rel_path=config_rel,
                    checkpoint_rel_path=checkpoint_rel,
                    report_rel_path=report_rel
                ))
                
    print(f"\nDispatching {len(records)} jobs to Camber Cloud (Size: {size})...")
    for rec in records:
        cmd_str = (
            f"mkdir -p checkpoints/camber_matrix_v4 reports/camber_matrix_v4 && "
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
        
        # Retry loop for rate-limits
        for attempt in range(5):
            try:
                out = run_camber_cmd(camber_cmd)
                match = re.search(r"Job ID:\s*(\d+)", out)
                if match:
                    rec.job_id = match.group(1)
                    rec.status = "SUBMITTED"
                    print(f"  [+] Dispatched {rec.scenario} | {rec.model} (seed {rec.seed}) -> Job ID: {rec.job_id}")
                    break
            except Exception as e:
                time.sleep(2 * (attempt + 1))
        time.sleep(0.5)
            
    return records


def monitor_jobs(records: List[JobRecord], poll_interval: int = 15) -> None:
    print("\nMonitoring parallel non-bull training jobs on Camber Cloud...")
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
                        print(f"  [*] Job {r.job_id} ({r.scenario} | {r.model}) reached status: {st}")
                    else:
                        still_running.append(r)
                else:
                    still_running.append(r)
            except Exception:
                still_running.append(r)
                
        pending = still_running
        if pending:
            print(f"  --> {len(pending)} / {len(records)} jobs still training in the cloud...")


def sync_and_evaluate(records: List[JobRecord]):
    print("\nSyncing trained non-bull checkpoints from Camber Stash...")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    
    subprocess.run([
        "camber", "stash", "cp", "-r",
        "stash://kohosting/projects/midas/checkpoints/camber_matrix_v4/",
        str(CHECKPOINTS_DIR)
    ], check=False, env=env)
    
    # Evaluate on respective test splits
    summary = []
    for r in records:
        cp = CHECKPOINTS_DIR / f"{r.scenario}__{r.model}-seed-{r.seed}.bin"
        cfg = CONFIGS_DIR / f"{r.scenario}__{r.model}-seed-{r.seed}.toml"
        if not cp.exists() or not cfg.exists():
            continue
            
        proc = subprocess.run([str(BIN), "evaluate", "--config", str(cfg), "--checkpoint", str(cp), "--split", "test"], capture_output=True, text=True)
        if proc.returncode != 0:
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
                    "scenario": r.scenario,
                    "model": r.model,
                    "seed": r.seed,
                    "return_%": sum(rets) / len(rets),
                    "max_drawdown_%": sum(dds) / len(dds),
                    "turnover_$": sum(tos) / len(tos),
                    "friction_$": sum(fees) / len(fees),
                })
        except Exception as e:
            print(f"Error parsing {r.scenario}__{r.model}: {e}")

    # Print results grouped by scenario
    for sc_key in SCENARIOS.keys():
        sc_desc = SCENARIOS[sc_key]["description"]
        sc_rows = [r for r in summary if r["scenario"] == sc_key]
        if not sc_rows:
            continue
            
        print("\n" + "="*115)
        print(f"  {sc_desc.upper()}")
        print("="*115)
        print(f"{'Model Configuration':<36} | {'Test Return (%)':<16} | {'Max DD (%)':<11} | {'Turnover ($)':<12} | {'Friction ($)':<12}")
        print("-"*115)
        
        sc_rows.sort(key=lambda x: x["return_%"], reverse=True)
        for r in sc_rows:
            print(f"{r['model']:<36} | {r['return_%']:>+15.2f}% | {r['max_drawdown_%']:>10.2f}% | ${r['turnover_$']:>10.2f} | ${r['friction_$']:>10.2f}")
        print("="*115)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[41])
    parser.add_argument("--size", default="xxsmall")
    args = parser.parse_args()

    print(f"=== Starting Camber Cloud Matrix V4 (Non-Bull Market Scenarios) ===")
    generate_all_configs(args.seeds)
    sync_configs_to_stash()
    records = dispatch_all_jobs(args.seeds, size=args.size)
    monitor_jobs(records)
    sync_and_evaluate(records)


if __name__ == "__main__":
    main()
