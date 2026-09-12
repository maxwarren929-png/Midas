#!/usr/bin/env python3
"""
Camber Cloud Matrix V13 Dispatcher and Real-Time Monitor

Dispatches all 6 distinct Apex 64-Unit Variants & 128-Unit Titan Brain to Camber Cloud:
  1. g64_base_control
  2. g64_macro_horizon_30d
  3. g64_hyper_burst_70
  4. g64_ath_breakout_sensor
  5. g64_sortino_shield
  6. g128_titan_deep_gru
"""

from __future__ import annotations

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
REPORTS_DIR = ROOT / "reports" / "matrix_v13_camber"
API_KEY = "5fa7e652ed5d57bc7f62d098e7c18afd653a348b"

MODELS = [
    "g64_base_control",
    "g64_macro_horizon_30d",
    "g64_hyper_burst_70",
    "g64_ath_breakout_sensor",
    "g64_sortino_shield",
    "g128_titan_deep_gru",
]


@dataclass
class CamberJob:
    model: str
    seed: int
    job_id: Optional[str] = None
    status: str = "PENDING"


def run_camber(cmd: List[str]) -> str:
    env = os.environ.copy()
    env["CAMBER_API_KEY"] = API_KEY
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Camber command error ({' '.join(cmd)}): {res.stderr.strip() or res.stdout.strip()}")
    return res.stdout.strip()


def dispatch_all(size: str = "small") -> List[CamberJob]:
    jobs: List[CamberJob] = []
    print("\n🚀 Dispatching 6 Matrix V13 Models to Camber Cloud...")
    
    for mod in MODELS:
        cfg_rel = f"configs/matrix_v13_camber/{mod}-seed-41.toml"
        cp_rel = f"checkpoints/matrix_v13_camber/{mod}-seed-41.bin"
        rep_rel = f"reports/matrix_v13_camber/{mod}-seed-41.json"
        
        cmd_str = (
            f"mkdir -p checkpoints/matrix_v13_camber reports/matrix_v13_camber && "
            f"chmod +x ./midas-cli && "
            f"./midas-cli simulate --config {cfg_rel} --split training --checkpoint {cp_rel} && "
            f"./midas-cli evaluate --config {cfg_rel} --checkpoint {cp_rel} --split test > {rep_rel}"
        )
        
        camber_cmd = [
            "camber", "job", "create",
            "--engine", "base",
            "--size", size,
            "--path", "stash://kohosting/projects/midas/",
            "--cmd", cmd_str
        ]
        
        out = run_camber(camber_cmd)
        match = re.search(r"Job ID:\s*(\d+)", out)
        if match:
            jid = match.group(1)
            job = CamberJob(model=mod, seed=41, job_id=jid, status="SUBMITTED")
            jobs.append(job)
            print(f"  [+] Dispatched {mod:<26} -> Camber Job ID: {jid}")
        else:
            print(f"  [!] Failed to parse Job ID for {mod}: {out}")
            
    return jobs


def monitor_and_collect(jobs: List[CamberJob], poll_interval: int = 12) -> None:
    print("\n⏳ Monitoring Camber Cloud Jobs (Live Cloud Execution)...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    active = [j for j in jobs if j.job_id]
    completed: List[CamberJob] = []
    
    while active:
        time.sleep(poll_interval)
        still_active = []
        for j in active:
            try:
                out = run_camber(["camber", "job", "get", str(j.job_id)])
                status_match = re.search(r"Status:\s*([A-Z_]+)", out)
                if status_match:
                    status = status_match.group(1)
                    if status != j.status:
                        j.status = status
                        print(f"  [Status Update] Job {j.job_id} ({j.model}) -> {status}")
                        
                    if status in ["COMPLETED", "SUCCESS", "FINISHED"]:
                        print(f"  [✓] Job {j.job_id} ({j.model}) COMPLETED on Camber Cloud!")
                        completed.append(j)
                    elif status in ["FAILED", "ERROR", "CANCELLED"]:
                        print(f"  [✗] Job {j.job_id} ({j.model}) {status} on Camber Cloud.")
                        completed.append(j)
                    else:
                        still_active.append(j)
                else:
                    still_active.append(j)
            except Exception as e:
                print(f"  [!] Error polling job {j.job_id}: {e}")
                still_active.append(j)
                
        active = still_active

    print("\n📥 All Camber Cloud jobs finished! Syncing report artifacts from Stash...")
    env = os.environ.copy()
    env["CAMBER_API_KEY"] = API_KEY
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    
    sync_cmd = [
        "camber", "stash", "cp", "-r",
        "stash://kohosting/projects/midas/reports/matrix_v13_camber",
        str(ROOT / "reports")
    ]
    subprocess.run(sync_cmd, env=env)
    
    # Parse and display leaderboard
    results = []
    for j in jobs:
        rep_file = REPORTS_DIR / f"{j.model}-seed-41.json"
        if rep_file.exists():
            try:
                data = json.loads(rep_file.read_text(encoding="utf-8"))
                agents = data.get("agents", [])
                if agents:
                    rets = [a[1]["total_return"] * 100 for a in agents]
                    dds = [a[1]["maximum_drawdown"] * 100 for a in agents]
                    tos = [a[1]["turnover"] for a in agents]
                    fees = [a[1]["total_fees"] + a[1]["spread_cost"] + a[1]["slippage_cost"] for a in agents]
                    results.append({
                        "model": j.model,
                        "job_id": j.job_id,
                        "return_%": sum(rets) / len(rets),
                        "max_drawdown_%": sum(dds) / len(dds),
                        "turnover_$": sum(tos) / len(tos),
                        "friction_$": sum(fees) / len(fees),
                    })
            except Exception as e:
                print(f"[!] Error parsing report for {j.model}: {e}")

    if results:
        results.sort(key=lambda x: x["return_%"], reverse=True)
        print("\n" + "="*120)
        print(f"  CAMBER CLOUD MATRIX V13 LEADERBOARD (2024-08-12 TO 2026-08-14 | 504 TRADING DAYS)")
        print("="*120)
        print(f"{'Model Configuration':<26} | {'Camber Job ID':<14} | {'2-Year Return (%)':<18} | {'Max DD (%)':<11} | {'Turnover ($)':<12}")
        print("-"*120)
        for r in results:
            print(f"{r['model']:<26} | {r['job_id']:<14} | {r['return_%']:>+17.2f}% | {r['max_drawdown_%']:>10.2f}% | ${r['turnover_$']:>10.2f}")
        print("="*120)


def main():
    jobs = dispatch_all(size="small")
    monitor_and_collect(jobs)


if __name__ == "__main__":
    main()
