#!/usr/bin/env python3
"""
Camber Cloud Matrix V3 Runner:
Dispatches and evaluates the Combos (Risk Reward + BPTT 8d, Mega Combo with Rich Sensors, 15d Ultra BPTT, and 15d Ultra Mega Combo).
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

from experiments.generate_matrix_v3 import CONDITIONS, generate_all_configs

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "camber_matrix_v3"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "camber_matrix_v3"
REPORTS_DIR = ROOT / "reports" / "camber_matrix_v3"
BIN = ROOT / "target" / "release" / "midas-cli"


def run_camber_cmd(cmd: List[str]) -> str:
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def sync_configs_to_stash():
    print("Syncing Matrix V3 configs to Camber Stash...")
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    cmd = [
        "camber", "stash", "cp", "-r",
        str(CONFIGS_DIR),
        "stash://kohosting/projects/midas/configs/camber_matrix_v3"
    ]
    subprocess.run(cmd, check=True, env=env)
    print("Configs synced to Stash successfully!")


@dataclass
class JobRecord:
    condition: str
    seed: int
    config_rel_path: str
    checkpoint_rel_path: str
    report_rel_path: str
    job_id: Optional[str] = None
    status: str = "PENDING"


def dispatch_all_jobs(seeds: List[int], size: str = "xxsmall") -> List[JobRecord]:
    records: List[JobRecord] = []
    
    for cond_key in CONDITIONS.keys():
        for seed in seeds:
            config_rel = f"configs/camber_matrix_v3/{cond_key}-seed-{seed}.toml"
            checkpoint_rel = f"checkpoints/camber_matrix_v3/{cond_key}-seed-{seed}.bin"
            report_rel = f"reports/camber_matrix_v3/{cond_key}-seed-{seed}.json"
            
            records.append(JobRecord(
                condition=cond_key,
                seed=seed,
                config_rel_path=config_rel,
                checkpoint_rel_path=checkpoint_rel,
                report_rel_path=report_rel
            ))
            
    print(f"\nDispatching {len(records)} jobs to Camber Cloud (Size: {size})...")
    for rec in records:
        cmd_str = (
            f"mkdir -p checkpoints/camber_matrix_v3 reports/camber_matrix_v3 && "
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
                    print(f"  [+] Dispatched {rec.condition} (seed {rec.seed}) -> Job ID: {rec.job_id}")
                    break
            except Exception as e:
                time.sleep(2 * (attempt + 1))
        time.sleep(0.5)
            
    return records


def monitor_jobs(records: List[JobRecord], poll_interval: int = 15) -> None:
    print("\nMonitoring parallel jobs on Camber Cloud...")
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
                        print(f"  [*] Job {r.job_id} ({r.condition}, seed {r.seed}) finished with status: {st}")
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
    print("\nSyncing trained checkpoints from Camber Stash...")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    
    subprocess.run([
        "camber", "stash", "cp", "-r",
        "stash://kohosting/projects/midas/checkpoints/camber_matrix_v3/",
        str(CHECKPOINTS_DIR)
    ], check=False, env=env)
    
    # Evaluate locally on the held-out test split
    summary = []
    for r in records:
        cp = CHECKPOINTS_DIR / f"{r.condition}-seed-{r.seed}.bin"
        cfg = CONFIGS_DIR / f"{r.condition}-seed-{r.seed}.toml"
        if not cp.exists() or not cfg.exists():
            continue
            
        proc = subprocess.run([str(BIN), "evaluate", "--config", str(cfg), "--checkpoint", str(cp), "--split", "test"], capture_output=True, text=True)
        if proc.returncode != 0:
            continue
            
        try:
            data = json.loads(proc.stdout)
            agents = data.get("agents", [])
            if agents:
                returns = [a[1]["total_return"] * 100 for a in agents]
                drawdowns = [a[1]["maximum_drawdown"] * 100 for a in agents]
                sharpes = [a[1]["sharpe_like"] for a in agents]
                tos = [a[1]["turnover"] for a in agents]
                trades = [a[1]["trades"] for a in agents]
                fees = [a[1]["total_fees"] + a[1]["spread_cost"] + a[1]["slippage_cost"] for a in agents]
                bankruptcies = sum(1 for a in agents if a[1]["bankrupt"])
                
                summary.append({
                    "condition": r.condition,
                    "seed": r.seed,
                    "return_%": sum(returns) / len(returns),
                    "max_drawdown_%": sum(drawdowns) / len(drawdowns),
                    "sharpe_like": sum(sharpes) / len(sharpes),
                    "turnover_$": sum(tos) / len(tos),
                    "friction_$": sum(fees) / len(fees),
                    "trades": sum(trades) / len(trades),
                    "bankruptcies": bankruptcies
                })
        except Exception as e:
            print(f"Error parsing {r.condition}-{r.seed}: {e}")

    # Group by condition
    cond_groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in summary:
        cond_groups.setdefault(row["condition"], []).append(row)
        
    print("\n" + "="*115)
    print(f"{'Condition':<36} | {'Seeds':<5} | {'Test Return (%)':<16} | {'Max DD (%)':<11} | {'Turnover ($)':<12} | {'Friction ($)':<12}")
    print("="*115)
    
    leaderboard_rows = []
    for cond, rows in sorted(cond_groups.items()):
        n = len(rows)
        avg_ret = sum(r["return_%"] for r in rows) / n
        avg_dd = sum(r["max_drawdown_%"] for r in rows) / n
        avg_to = sum(r["turnover_$"] for r in rows) / n
        avg_fric = sum(r["friction_$"] for r in rows) / n
        leaderboard_rows.append((avg_ret, cond, n, avg_dd, avg_to, avg_fric))
        
    # Sort leaderboard by return (best to worst)
    leaderboard_rows.sort(key=lambda x: x[0], reverse=True)
    for avg_ret, cond, n, avg_dd, avg_to, avg_fric in leaderboard_rows:
        print(f"{cond:<36} | {n:<5} | {avg_ret:>+15.2f}% | {avg_dd:>10.2f}% | ${avg_to:>10.2f} | ${avg_fric:>10.2f}")
    print("="*115)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42])
    parser.add_argument("--size", default="xxsmall")
    args = parser.parse_args()

    print(f"=== Starting Camber Cloud Matrix V3 (Combos + 15d Ultra BPTT) ===")
    generate_all_configs(args.seeds)
    sync_configs_to_stash()
    records = dispatch_all_jobs(args.seeds, size=args.size)
    monitor_jobs(records)
    sync_and_evaluate(records)


if __name__ == "__main__":
    main()
