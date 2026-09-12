#!/usr/bin/env python3
"""
Matrix V14: The 100%+ Super-Alpha Gauntlet on Camber Cloud

Innovations Tested:
  1. g64_base_champion: Canonical +72.15% Benchmark Control (64 Units, [128, 64], 16d BPTT)
  2. g64_multi_scale_momentum: Full 5d to 252d Multi-Scale Momentum (Returns 5/10/20/60/120/252, SMAs 20/50/200, Vols 10/20)
  3. g64_high_temperature_conviction: Sharp Beta Action Impulse (temp=2.5, max_trade=0.60, max_alloc=0.98)
  4. g64_low_churn_compounder: Ultra-low Turnover Drag (turnover_penalty=0.00005, rewarding multi-week trend holding)
  5. g64_godmode_apex_synergy: ALL 4 ALPHA INNOVATIONS COMBINED (Multi-Scale Sensors + Sharp Sizing + Low Churn + 98% Allocation)

Dispatched 100% to Camber Cloud with live monitoring.
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

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "matrix_v14_super_alpha"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "matrix_v14_super_alpha"
REPORTS_DIR = ROOT / "reports" / "matrix_v14_super_alpha"
BIN = ROOT / "target" / "release" / "midas-cli"
DATA_FILE = ROOT / "data" / "hyper" / "hyper_20_assets_2016_2026.csv"
BASE_TEMPLATE_PATH = ROOT / "configs" / "prompt3b" / "experiment-ppo.toml"
API_KEY = "5fa7e652ed5d57bc7f62d098e7c18afd653a348b"

ASSETS_20 = [
    "AAPL", "AMD", "BTC", "GLD", "IWM", "META", "MSFT", "NVDA", "PLTR", "QQQ",
    "SLV", "SOXL", "SPXU", "SPY", "SQQQ", "TLT", "TQQQ", "TSLA", "UPRO", "XLE"
]

MODELS = {
    "g64_base_champion": {
        "description": "Canonical 64-Unit Benchmark (+72.15% Control)",
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 16, "rollout": 64, "minibatch": 16,
        "trade_fraction": 0.50, "max_alloc": 0.95, "reward_kind": "log_return", "dd_penalty": 0.0,
        "turnover_penalty": 0.0001, "entropy": 0.01, "temp": 1.0, "beta_min_shape": 0.1,
        "ret_windows": "[20, 60, 120]", "vol_windows": "[20]", "sma_windows": "[50, 200]",
    },
    "g64_multi_scale_momentum": {
        "description": "Multi-Scale Momentum Sensors (5d to 252d Full Horizon)",
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 16, "rollout": 64, "minibatch": 16,
        "trade_fraction": 0.50, "max_alloc": 0.95, "reward_kind": "log_return", "dd_penalty": 0.0,
        "turnover_penalty": 0.0001, "entropy": 0.01, "temp": 1.0, "beta_min_shape": 0.1,
        "ret_windows": "[5, 10, 20, 60, 120, 252]", "vol_windows": "[10, 20]", "sma_windows": "[20, 50, 200]",
    },
    "g64_high_temperature_conviction": {
        "description": "Sharp Action Impulse & Concentrated Compounding (temp=2.5, 60% Sizing, 98% Alloc)",
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 16, "rollout": 64, "minibatch": 16,
        "trade_fraction": 0.60, "max_alloc": 0.98, "reward_kind": "log_return", "dd_penalty": 0.0,
        "turnover_penalty": 0.0001, "entropy": 0.015, "temp": 2.5, "beta_min_shape": 0.3,
        "ret_windows": "[20, 60, 120]", "vol_windows": "[20]", "sma_windows": "[50, 200]",
    },
    "g64_low_churn_compounder": {
        "description": "Ultra-Low Churn Drag & Long Trend Holding (turnover_penalty=0.00005)",
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 16, "rollout": 64, "minibatch": 16,
        "trade_fraction": 0.50, "max_alloc": 0.96, "reward_kind": "log_return", "dd_penalty": 0.0,
        "turnover_penalty": 0.00005, "entropy": 0.01, "temp": 1.0, "beta_min_shape": 0.1,
        "ret_windows": "[20, 60, 120]", "vol_windows": "[20]", "sma_windows": "[50, 200]",
    },
    "g64_godmode_apex_synergy": {
        "description": "GOD-MODE SYNERGY: Multi-Scale Sensors + Sharp Sizing + Low Churn + 98% Allocation",
        "memory_size": 64, "hidden": "[128, 64]", "seq_len": 16, "rollout": 64, "minibatch": 16,
        "trade_fraction": 0.60, "max_alloc": 0.98, "reward_kind": "log_return", "dd_penalty": 0.0,
        "turnover_penalty": 0.00005, "entropy": 0.015, "temp": 2.2, "beta_min_shape": 0.3,
        "ret_windows": "[5, 10, 20, 60, 120, 252]", "vol_windows": "[10, 20]", "sma_windows": "[20, 50, 200]",
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
        f'path = "checkpoints/matrix_v14_super_alpha/{mod_key}-seed-{seed}.bin"',
        template
    )

    # 4. Architecture
    template = re.sub(r'(?m)^hidden_layers\s*=.*$', f"hidden_layers = {mod['hidden']}", template)
    template = re.sub(r'(?m)^memory_size\s*=.*$', f"memory_size = {mod['memory_size']}", template)
    template = re.sub(r'(?m)^sequence_length\s*=.*$', f"sequence_length = {mod['seq_len']}", template)
    template = re.sub(r'(?m)^rollout_length\s*=.*$', f"rollout_length = {mod['rollout']}", template)
    template = re.sub(r'(?m)^minibatch_size\s*=.*$', f"minibatch_size = {mod['minibatch']}", template)

    # 5. Sizing & High-Conviction Allocation
    template = re.sub(r'(?m)^maximum_trade_fraction\s*=.*$', f"maximum_trade_fraction = {mod['trade_fraction']}", template)
    template = template.replace(
        "[brain.traits.maximum_allocation]\ninitial = 0.5\nminimum = 0.01\nmaximum = 1.0",
        f"[brain.traits.maximum_allocation]\ninitial = {mod['max_alloc']}\nminimum = 0.10\nmaximum = 1.0"
    )
    template = template.replace(
        "[brain.traits.action_temperature]\ninitial = 1.0\nminimum = 0.1\nmaximum = 3.0",
        f"[brain.traits.action_temperature]\ninitial = {mod['temp']}\nminimum = 0.1\nmaximum = 3.0"
    )

    # 6. Reward & Churn Friction
    if mod['reward_kind'] == "log_return":
        template = template.replace('[reward]\nkind = "risk_adjusted_return"', '[reward]\nkind = "log_return"')
    else:
        template = template.replace('[reward]\nkind = "log_return"', '[reward]\nkind = "risk_adjusted_return"')
        
    template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', f"drawdown_penalty_weight = {mod['dd_penalty']}", template)
    template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', f"turnover_penalty_weight = {mod['turnover_penalty']}", template)
    template = re.sub(r'(?m)^entropy_coefficient\s*=.*$', f"entropy_coefficient = {mod['entropy']}", template)

    # 7. Corporate Actions Tolerance
    template = re.sub(r'(?m)^discontinuity_threshold\s*=.*$', "discontinuity_threshold = 0.85", template)

    # 8. Distinct Sensor Configuration
    obs_block = f"""
[observations]
schema = "default"
providers = ["market", "portfolio", "agent_state", "rolling_market_stats"]

[observations.rolling_market_stats]
enabled = true
return_windows = {mod['ret_windows']}
volatility_windows = {mod['vol_windows']}
sma_windows = {mod['sma_windows']}
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
    print("Syncing Matrix V14 configs to Camber Stash...")
    env = os.environ.copy()
    env["CAMBER_API_KEY"] = API_KEY
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    
    cmd = [
        "camber", "stash", "cp", "-r",
        str(CONFIGS_DIR),
        "stash://kohosting/projects/midas/configs/matrix_v14_super_alpha"
    ]
    subprocess.run(cmd, check=True, env=env)
    print("Configs successfully synced to Camber Stash!")


def dispatch_all(size: str = "small") -> List[CamberJob]:
    jobs: List[CamberJob] = []
    print("\n🚀 Dispatching Matrix V14 Super-Alpha Models to Camber Cloud...")
    
    for mod in MODELS.keys():
        cfg_rel = f"configs/matrix_v14_super_alpha/{mod}-seed-41.toml"
        cp_rel = f"checkpoints/matrix_v14_super_alpha/{mod}-seed-41.bin"
        rep_rel = f"reports/matrix_v14_super_alpha/{mod}-seed-41.json"
        
        cmd_str = (
            f"mkdir -p checkpoints/matrix_v14_super_alpha reports/matrix_v14_super_alpha && "
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
            print(f"  [+] Dispatched {mod:<32} -> Camber Job ID: {jid}")
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
        "stash://kohosting/projects/midas/reports/matrix_v14_super_alpha",
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
        print(f"  CAMBER CLOUD MATRIX V14: SUPER-ALPHA LEADERBOARD (2024-08-12 TO 2026-08-14 | 504 TRADING DAYS)")
        print("="*125)
        print(f"{'Model Configuration':<34} | {'Camber Job ID':<14} | {'2-Year Return (%)':<18} | {'Max DD (%)':<11} | {'Turnover ($)':<12}")
        print("-"*125)
        for r in results:
            print(f"{r['model']:<34} | {r['job_id']:<14} | {r['return_%']:>+17.2f}% | {r['max_drawdown_%']:>10.2f}% | ${r['turnover_$']:>10.2f}")
        print("="*125)


def main():
    generate_all_configs([41])
    sync_to_stash()
    jobs = dispatch_all(size="small")
    monitor_and_collect(jobs)


if __name__ == "__main__":
    main()
