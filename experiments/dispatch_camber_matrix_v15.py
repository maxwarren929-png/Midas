#!/usr/bin/env python3
"""
Matrix V15: Training Depth & Epoch Intensity Sweep on Camber Cloud

Investigates:
  1. g64_1x_standard: 1 Pass (4 PPO Epochs, ~125 Gradient Updates) -> Benchmark (+72.15%)
  2. g64_2x_deep_ppo: 1 Pass (8 PPO Epochs, ~250 Gradient Updates)
  3. g64_4x_deep_ppo: 1 Pass (16 PPO Epochs, ~500 Gradient Updates)
  4. g64_2_pass_curriculum: 2 Sequential 8-Year Passes (~250 Gradient Updates)
  5. g64_4_pass_curriculum: 4 Sequential 8-Year Passes (~500 Gradient Updates)

Tests whether deeper training unlocks higher alpha or triggers overfitting.
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
BASE_TEMPLATE_PATH = ROOT / "configs" / "prompt3b" / "experiment-ppo.toml"
API_KEY = "5fa7e652ed5d57bc7f62d098e7c18afd653a348b"

ASSETS_20 = [
    "AAPL", "AMD", "BTC", "GLD", "IWM", "META", "MSFT", "NVDA", "PLTR", "QQQ",
    "SLV", "SOXL", "SPXU", "SPY", "SQQQ", "TLT", "TQQQ", "TSLA", "UPRO", "XLE"
]

MODELS = {
    "g64_1x_standard": {
        "description": "1 Pass, 4 PPO Epochs (~125 Gradient Steps) - Benchmark Control",
        "opt_epochs": 4,
        "passes": 1,
    },
    "g64_2x_deep_ppo": {
        "description": "1 Pass, 8 PPO Epochs (~250 Gradient Steps)",
        "opt_epochs": 8,
        "passes": 1,
    },
    "g64_4x_deep_ppo": {
        "description": "1 Pass, 16 PPO Epochs (~500 Gradient Steps)",
        "opt_epochs": 16,
        "passes": 1,
    },
    "g64_2_pass_curriculum": {
        "description": "2 Sequential 8-Year Passes (Pass 1 -> Pass 2 -> Evaluate)",
        "opt_epochs": 4,
        "passes": 2,
    },
    "g64_4_pass_curriculum": {
        "description": "4 Sequential 8-Year Passes (Pass 1 -> 2 -> 3 -> 4 -> Evaluate)",
        "opt_epochs": 4,
        "passes": 4,
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

    # 3. Seed and Checkpoint Target
    template = re.sub(r'(?m)^seed\s*=.*$', f"seed = {seed}", template)
    template = re.sub(
        r'(?m)^path\s*=\s*"checkpoints/.*"$',
        f'path = "checkpoints/matrix_v15_training_depth/{mod_key}-seed-{seed}.bin"',
        template
    )

    # 4. Neural Architecture (Canonical 64-Unit Champion Setup)
    template = re.sub(r'(?m)^hidden_layers\s*=.*$', 'hidden_layers = [128, 64]', template)
    template = re.sub(r'(?m)^memory_size\s*=.*$', 'memory_size = 64', template)
    template = re.sub(r'(?m)^sequence_length\s*=.*$', 'sequence_length = 16', template)
    template = re.sub(r'(?m)^rollout_length\s*=.*$', 'rollout_length = 64', template)
    template = re.sub(r'(?m)^minibatch_size\s*=.*$', 'minibatch_size = 16', template)

    # 5. Sizing & High-Conviction Allocation
    template = re.sub(r'(?m)^maximum_trade_fraction\s*=.*$', 'maximum_trade_fraction = 0.50', template)
    template = template.replace(
        "[brain.traits.maximum_allocation]\ninitial = 0.5\nminimum = 0.01\nmaximum = 1.0",
        "[brain.traits.maximum_allocation]\ninitial = 0.95\nminimum = 0.10\nmaximum = 1.0"
    )

    # 6. Training Intensity (PPO Inner Optimization Epochs)
    template = re.sub(r'(?m)^optimisation_epochs\s*=.*$', f"optimisation_epochs = {mod['opt_epochs']}", template)

    # 7. Reward & Friction
    template = template.replace('[reward]\nkind = "risk_adjusted_return"', '[reward]\nkind = "log_return"')
    template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', 'drawdown_penalty_weight = 0.0', template)
    template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', 'turnover_penalty_weight = 0.0001', template)
    template = re.sub(r'(?m)^entropy_coefficient\s*=.*$', 'entropy_coefficient = 0.01', template)
    template = re.sub(r'(?m)^discontinuity_threshold\s*=.*$', 'discontinuity_threshold = 0.85', template)

    # 8. Distilled 60-Feature Observations
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


def generate_all_configs(seeds: List[int] = [41]) -> List[Path]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for mod_key in MODELS.keys():
        for seed in seeds:
            cfg_text = build_config_text(mod_key, seed)
            cfg_path = CONFIGS_DIR / f"{mod_key}-seed-{seed}.toml"
            cfg_path.write_text(cfg_text, encoding="utf-8")
            generated.append(cfg_path)
    print(f"Generated {len(generated)} experiment configs in {CONFIGS_DIR}")
    return generated


def sync_to_stash():
    print("Syncing Matrix V15 configs to Camber Stash...")
    env = os.environ.copy()
    env["CAMBER_API_KEY"] = API_KEY
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    
    cmd = [
        "camber", "stash", "cp", "-r",
        str(CONFIGS_DIR),
        "stash://kohosting/projects/midas/configs/matrix_v15_training_depth"
    ]
    subprocess.run(cmd, check=True, env=env)
    print("Configs successfully synced to Camber Stash!")


def dispatch_all(size: str = "small") -> List[CamberJob]:
    jobs: List[CamberJob] = []
    print("\n🚀 Dispatching Matrix V15 Training Intensity Models to Camber Cloud...")
    
    for mod, spec in MODELS.items():
        cfg_rel = f"configs/matrix_v15_training_depth/{mod}-seed-41.toml"
        cp_rel = f"checkpoints/matrix_v15_training_depth/{mod}-seed-41.bin"
        rep_rel = f"reports/matrix_v15_training_depth/{mod}-seed-41.json"
        
        sim_cmds = []
        for _ in range(spec["passes"]):
            sim_cmds.append(f"./midas-cli simulate --config {cfg_rel} --split training --checkpoint {cp_rel}")
            
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
            print(f"  [+] Dispatched {mod:<32} -> Camber Job ID: {jid}")
        else:
            print(f"  [!] Failed to parse Job ID for {mod}: {out}")
            
    return jobs


def monitor_and_collect(jobs: List[CamberJob], poll_interval: int = 12) -> None:
    print("\n⏳ Monitoring Camber Cloud Training Depth Jobs...")
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

    print("\n📥 All Camber Cloud training jobs finished! Syncing report artifacts from Stash...")
    env = os.environ.copy()
    env["CAMBER_API_KEY"] = API_KEY
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    
    sync_cmd = [
        "camber", "stash", "cp", "-r",
        "stash://kohosting/projects/midas/reports/matrix_v15_training_depth",
        str(ROOT / "reports")
    ]
    subprocess.run(sync_cmd, env=env)
    
    # Render final leaderboard
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
                        "description": MODELS[j.model]["description"],
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
        print("\n" + "="*125)
        print(f"  CAMBER CLOUD MATRIX V15: TRAINING INTENSITY LEADERBOARD (2024-08-12 TO 2026-08-14 | 504 TRADING DAYS)")
        print("="*125)
        print(f"{'Model Configuration':<32} | {'Camber Job ID':<14} | {'2-Year Return (%)':<18} | {'Max DD (%)':<11} | {'Turnover ($)':<12}")
        print("-"*125)
        for r in results:
            print(f"{r['model']:<32} | {r['job_id']:<14} | {r['return_%']:>+17.2f}% | {r['max_drawdown_%']:>10.2f}% | ${r['turnover_$']:>10.2f}")
        print("="*125)


def main():
    generate_all_configs([41])
    sync_to_stash()
    jobs = dispatch_all(size="small")
    monitor_and_collect(jobs)


if __name__ == "__main__":
    main()
