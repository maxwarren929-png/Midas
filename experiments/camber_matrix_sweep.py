#!/usr/bin/env python3
"""
Camber Cloud Experiment Matrix Runner for Midas.
Generates experimental conditions across multiple seeds, dispatches them in parallel to Camber Cloud,
monitors execution, downloads artifacts, and builds a comprehensive comparison matrix.
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
CONFIGS_DIR = ROOT / "configs" / "camber_matrix"
REPORTS_DIR = ROOT / "reports" / "camber_matrix"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "camber_matrix"

BASE_TEMPLATE_PATH = ROOT / "configs" / "prompt3b" / "experiment-ppo.toml"

# Define conditions including the new Winning Combination (07_winning_combo)
CONDITIONS: Dict[str, Dict[str, Any]] = {
    "01_baseline_ppo": {
        "description": "Baseline PPO with Momentum (lr=3e-4), [32, 16] hidden, memory=8",
        "overrides": {}
    },
    "02_adamw_optimizer": {
        "description": "PPO with AdamW optimizer (calibrated lr=3.5e-6, weight_decay=0.01)",
        "overrides": {
            "learning.learning_rate": 0.0000035,
            "optimizer": {
                "kind": "adamw",
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "weight_decay": 0.01,
            }
        }
    },
    "03_deep_recurrent": {
        "description": "Deep Architecture [64, 32] with enlarged GRU memory=16 and AdamW",
        "overrides": {
            "brain.hidden_layers": [64, 32],
            "brain.memory_size": 16,
            "learning.learning_rate": 0.0000035,
            "optimizer": {
                "kind": "adamw",
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "weight_decay": 0.01,
            }
        }
    },
    "04_drawdown_penalty": {
        "description": "Risk-Aware Reward with Drawdown Penalty (weight=0.5)",
        "overrides": {
            "reward.drawdown_penalty_weight": 0.5,
        }
    },
    "05_turnover_penalty": {
        "description": "Cost-Aware Reward with Turnover Penalty (weight=0.01)",
        "overrides": {
            "reward.turnover_penalty_weight": 0.01,
        }
    },
    "06_ppo_evolution": {
        "description": "Hybrid PPO + Genetic Evolution (Tournament=2, Fitness Sharing)",
        "overrides": {
            "evolution.enabled": True,
            "evolution.selection.tournament_size": 2,
        }
    },
    "07_winning_combo": {
        "description": "Winning Combination: Turnover Penalty (0.01) + Drawdown Penalty (0.5) + AdamW (lr=3.5e-6)",
        "overrides": {
            "reward.turnover_penalty_weight": 0.01,
            "reward.drawdown_penalty_weight": 0.5,
            "learning.learning_rate": 0.0000035,
            "optimizer": {
                "kind": "adamw",
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "weight_decay": 0.01,
            }
        }
    }
}


def build_config_text(seed: int, overrides: Dict[str, Any], condition_name: str) -> str:
    template = BASE_TEMPLATE_PATH.read_text(encoding="utf-8")
    
    # Update seed and paths
    template = re.sub(r'(?m)^seed\s*=.*$', f"seed = {seed}", template)
    template = re.sub(
        r'(?m)^path\s*=\s*"checkpoints/.*"$',
        f'path = "checkpoints/camber_matrix/{condition_name}-seed-{seed}.bin"',
        template
    )
    
    # Apply learning_rate override
    if "learning.learning_rate" in overrides:
        lr = overrides["learning.learning_rate"]
        template = re.sub(r'(?m)^learning_rate\s*=.*$', f"learning_rate = {lr}", template)
    
    # Apply hidden layers & memory size
    if "brain.hidden_layers" in overrides:
        hl = json.dumps(overrides["brain.hidden_layers"])
        template = re.sub(r'(?m)^hidden_layers\s*=.*$', f"hidden_layers = {hl}", template)
    if "brain.memory_size" in overrides:
        ms = overrides["brain.memory_size"]
        template = re.sub(r'(?m)^memory_size\s*=.*$', f"memory_size = {ms}", template)
        
    # Apply reward penalties
    if "reward.drawdown_penalty_weight" in overrides:
        dp = overrides["reward.drawdown_penalty_weight"]
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', f"drawdown_penalty_weight = {dp}", template)
    if "reward.turnover_penalty_weight" in overrides:
        tp = overrides["reward.turnover_penalty_weight"]
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', f"turnover_penalty_weight = {tp}", template)
        
    # Apply evolution settings
    if "evolution.enabled" in overrides:
        ev_enabled = "true" if overrides["evolution.enabled"] else "false"
        template = re.sub(r'(?m)^enabled\s*=\s*(true|false)$', f"enabled = {ev_enabled}", template)
    if "evolution.selection.tournament_size" in overrides:
        ts = overrides["evolution.selection.tournament_size"]
        template = re.sub(r'(?m)^tournament_size\s*=.*$', f"tournament_size = {ts}", template)

    # Append optimizer section if defined
    if "optimizer" in overrides:
        opt = overrides["optimizer"]
        opt_text = f"""
[optimizer]
kind = "{opt['kind']}"
beta1 = {opt['beta1']}
beta2 = {opt['beta2']}
epsilon = {opt['epsilon']}
weight_decay = {opt['weight_decay']}
"""
        template += opt_text

    return template


def generate_all_configs(seeds: List[int]) -> List[Path]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for cond_name, spec in CONDITIONS.items():
        for seed in seeds:
            cfg_text = build_config_text(seed, spec["overrides"], cond_name)
            cfg_path = CONFIGS_DIR / f"{cond_name}-seed-{seed}.toml"
            cfg_path.write_text(cfg_text, encoding="utf-8")
            generated.append(cfg_path)
    print(f"Generated {len(generated)} experiment configs in {CONFIGS_DIR}")
    return generated


def run_camber_cmd(cmd: List[str]) -> str:
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def sync_to_stash():
    print("Syncing generated configs to Camber Stash...")
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    cmd = [
        "camber", "stash", "cp", "-r",
        str(CONFIGS_DIR),
        "stash://kohosting/projects/midas/configs/camber_matrix"
    ]
    subprocess.run(cmd, check=True, env=env)
    print("Configs successfully synced to Stash!")


@dataclass
class JobRecord:
    condition: str
    seed: int
    config_rel_path: str
    checkpoint_rel_path: str
    report_rel_path: str
    job_id: Optional[str] = None
    status: str = "PENDING"


def dispatch_jobs(seeds: List[int], size: str = "xxsmall") -> List[JobRecord]:
    records: List[JobRecord] = []
    
    for cond_name in CONDITIONS.keys():
        for seed in seeds:
            config_rel = f"configs/camber_matrix/{cond_name}-seed-{seed}.toml"
            checkpoint_rel = f"checkpoints/camber_matrix/{cond_name}-seed-{seed}.bin"
            report_rel = f"reports/camber_matrix/{cond_name}-seed-{seed}.json"
            
            records.append(JobRecord(
                condition=cond_name,
                seed=seed,
                config_rel_path=config_rel,
                checkpoint_rel_path=checkpoint_rel,
                report_rel_path=report_rel
            ))
            
    print(f"\nDispatching {len(records)} jobs to Camber Cloud...")
    for rec in records:
        # Command runs training on training split and directly writes evaluate JSON to reports
        cmd_str = (
            f"mkdir -p checkpoints/camber_matrix reports/camber_matrix && "
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
        
        out = run_camber_cmd(camber_cmd)
        match = re.search(r"Job ID:\s*(\d+)", out)
        if match:
            rec.job_id = match.group(1)
            rec.status = "SUBMITTED"
            print(f"  [+] Dispatched {rec.condition} (seed {rec.seed}) -> Job ID: {rec.job_id}")
        else:
            print(f"  [!] Failed to parse Job ID for {rec.condition} (seed {rec.seed}): {out}")
            
    return records


def monitor_jobs(records: List[JobRecord], poll_interval: int = 10) -> None:
    print("\nMonitoring running jobs on Camber Cloud...")
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
                        print(f"  [*] Job {r.job_id} ({r.condition}, seed {r.seed}) reached status: {st}")
                    else:
                        still_running.append(r)
                else:
                    still_running.append(r)
            except Exception as e:
                print(f"  [!] Error checking job {r.job_id}: {e}")
                still_running.append(r)
                
        pending = still_running
        if pending:
            print(f"  --> {len(pending)} / {len(records)} jobs still executing in the cloud...")


def sync_artifacts_down():
    print("\nSyncing checkpoints and test reports down from Camber Stash...")
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "camber", "stash", "cp", "-r",
        "stash://kohosting/projects/midas/checkpoints/camber_matrix/",
        str(CHECKPOINTS_DIR)
    ], check=False, env=env)
    subprocess.run([
        "camber", "stash", "cp", "-r",
        "stash://kohosting/projects/midas/reports/camber_matrix/",
        str(REPORTS_DIR)
    ], check=False, env=env)
    print("Download complete!")


def main():
    parser = argparse.ArgumentParser(description="Run Midas Comparison Matrix on Camber Cloud")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42], help="Random seeds to evaluate")
    parser.add_argument("--size", default="xxsmall", choices=["xxsmall", "xsmall", "small", "medium", "large"], help="Camber node size")
    args = parser.parse_args()

    print(f"=== Starting Midas Comparison Matrix (with Winning Combo) on Camber Cloud ===")
    print(f"Conditions ({len(CONDITIONS)}): {list(CONDITIONS.keys())}")
    print(f"Seeds: {args.seeds}")
    print(f"Instance Size: {args.size}")
    
    # 1. Generate configs
    generate_all_configs(args.seeds)
    
    # 2. Sync to Stash
    sync_to_stash()
    
    # 3. Dispatch parallel jobs
    records = dispatch_jobs(args.seeds, size=args.size)
    
    # 4. Monitor jobs
    monitor_jobs(records)
    
    # 5. Sync artifacts down
    sync_artifacts_down()


if __name__ == "__main__":
    main()
