#!/usr/bin/env python3
"""
Matrix V15 (Re-Run of Failed Models on Camber Cloud):
  1. g64_4x_deep_ppo: 1 Pass (16 PPO Epochs) on 'medium' node
  2. g64_2_pass_curriculum: 2 Sequential Passes (simulate -> resume -> evaluate)
  3. g64_4_pass_curriculum: 4 Sequential Passes (simulate -> resume x3 -> evaluate)
"""

from __future__ import annotations

import argparse
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
CONFIGS_DIR = ROOT / "configs" / "matrix_v15_training_depth"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "matrix_v15_training_depth"
REPORTS_DIR = ROOT / "reports" / "matrix_v15_training_depth"
API_KEY = "5fa7e652ed5d57bc7f62d098e7c18afd653a348b"

MODELS_RERUN = {
    "g64_4x_deep_ppo": {
        "description": "1 Pass, 16 PPO Epochs (~500 Gradient Steps)",
        "passes": 1,
        "size": "medium",
    },
    "g64_2_pass_curriculum": {
        "description": "2 Sequential 8-Year Passes (simulate -> resume -> evaluate)",
        "passes": 2,
        "size": "small",
    },
    "g64_4_pass_curriculum": {
        "description": "4 Sequential 8-Year Passes (simulate -> resume x3 -> evaluate)",
        "passes": 4,
        "size": "small",
    },
}


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


def dispatch_reruns() -> List[CamberJob]:
    jobs: List[CamberJob] = []
    print("\n🚀 Dispatching 3 Re-Run Models to Camber Cloud...")
    
    for mod, spec in MODELS_RERUN.items():
        cfg_rel = f"configs/matrix_v15_training_depth/{mod}-seed-41.toml"
        cp_rel = f"checkpoints/matrix_v15_training_depth/{mod}-seed-41.bin"
        rep_rel = f"reports/matrix_v15_training_depth/{mod}-seed-41.json"
        
        sim_cmds = [f"./midas-cli simulate --config {cfg_rel} --split training --checkpoint {cp_rel}"]
        for _ in range(spec["passes"] - 1):
            sim_cmds.append(f"./midas-cli resume --config {cfg_rel} --checkpoint {cp_rel}")
            
        full_sim = " && ".join(sim_cmds)
        cmd_str = (
            f"mkdir -p checkpoints/matrix_v15_training_depth reports/matrix_v15_training_depth && "
            f"chmod +x ./midas-cli && "
            f"{full_sim} && "
            f"./midas-cli evaluate --config {cfg_rel} --checkpoint {cp_rel} --split test > {rep_rel}"
        )
        
        camber_cmd = [
            "camber", "job", "create",
            "--engine", "base",
            "--size", spec["size"],
            "--path", "stash://kohosting/projects/midas/",
            "--cmd", cmd_str
        ]
        
        out = run_camber(camber_cmd)
        match = re.search(r"Job ID:\s*(\d+)", out)
        if match:
            jid = match.group(1)
            job = CamberJob(model=mod, seed=41, job_id=jid, status="SUBMITTED")
            jobs.append(job)
            print(f"  [+] Dispatched {mod:<32} -> Camber Job ID: {jid} (size: {spec['size']})")
        else:
            print(f"  [!] Failed to parse Job ID for {mod}: {out}")
            
    return jobs


def monitor_and_collect(jobs: List[CamberJob], poll_interval: int = 12) -> None:
    print("\n⏳ Monitoring Camber Cloud Re-Run Jobs...")
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

    print("\n📥 All re-run jobs finished! Syncing report artifacts from Stash...")
    env = os.environ.copy()
    env["CAMBER_API_KEY"] = API_KEY
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    
    sync_cmd = [
        "camber", "stash", "cp", "-r",
        "stash://kohosting/projects/midas/reports/matrix_v15_training_depth",
        str(ROOT / "reports")
    ]
    subprocess.run(sync_cmd, env=env)


def main():
    jobs = dispatch_reruns()
    monitor_and_collect(jobs)


if __name__ == "__main__":
    main()
