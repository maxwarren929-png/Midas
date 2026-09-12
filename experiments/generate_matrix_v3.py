#!/usr/bin/env python3
"""
Matrix V3 Configuration Generator:
1. Combo: Risk Adjusted Reward + Long BPTT (8d)
2. Mega Combo: Risk Adjusted Reward + Long BPTT (8d) + Rich Sensors
3. Ultra Long BPTT (15d): sequence_length = 15, rollout = 60, minibatch = 15
4. Ultra Mega Combo (15d): 15d BPTT + Risk Adjusted Reward + Rich Sensors
5. Reference Controls: Baseline PPO & 8d Long BPTT Control
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "camber_matrix_v3"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "camber_matrix_v3"
REPORTS_DIR = ROOT / "reports" / "camber_matrix_v3"
BASE_TEMPLATE_PATH = ROOT / "configs" / "prompt3b" / "experiment-ppo.toml"

CONDITIONS = {
    "01_baseline_control": {
        "description": "Control: Baseline PPO (2d BPTT, log_return)",
        "kind": "baseline"
    },
    "02_long_bptt_8d_control": {
        "description": "Control: Previous Winner (8d BPTT, seq=8, rollout=32, minibatch=8)",
        "kind": "bptt_8d"
    },
    "03_combo_reward_and_bptt_8d": {
        "description": "Combo: Risk-Adjusted Reward (DD=0.5, TO=0.01) + 8d Long BPTT",
        "kind": "combo_reward_bptt"
    },
    "04_mega_combo_sensors_reward_bptt_8d": {
        "description": "Mega Combo: Rich Sensors + Risk-Adjusted Reward + 8d Long BPTT",
        "kind": "mega_combo_8d"
    },
    "05_ultra_long_bptt_15d": {
        "description": "Ultra Long BPTT: 15-Day Recurrent Horizon (seq=15, rollout=60, minibatch=15)",
        "kind": "bptt_15d"
    },
    "06_ultra_mega_combo_15d": {
        "description": "Ultra Mega Combo: Rich Sensors + Risk-Adjusted Reward + 15d Long BPTT",
        "kind": "ultra_mega_15d"
    }
}

def build_config_text(seed: int, cond_key: str) -> str:
    template = BASE_TEMPLATE_PATH.read_text(encoding="utf-8")
    
    # Update seed and paths
    template = re.sub(r'(?m)^seed\s*=.*$', f"seed = {seed}", template)
    template = re.sub(
        r'(?m)^path\s*=\s*"checkpoints/.*"$',
        f'path = "checkpoints/camber_matrix_v3/{cond_key}-seed-{seed}.bin"',
        template
    )

    kind = CONDITIONS[cond_key]["kind"]

    if kind == "baseline":
        pass

    elif kind == "bptt_8d":
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 8", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 32", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 8", template)

    elif kind == "combo_reward_bptt":
        # 8d BPTT
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 8", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 32", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 8", template)
        # Risk adjusted reward
        template = re.sub(r'(?m)^kind\s*=\s*"log_return"$', 'kind = "risk_adjusted_return"', template)
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.5", template)
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.01", template)

    elif kind == "mega_combo_8d":
        # 8d BPTT
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 8", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 32", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 8", template)
        # Risk adjusted reward
        template = re.sub(r'(?m)^kind\s*=\s*"log_return"$', 'kind = "risk_adjusted_return"', template)
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.5", template)
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.01", template)
        # Rich sensors
        obs_block = """
[observations]
schema = "default"
providers = ["market", "portfolio", "agent_state", "rolling_market_stats"]

[observations.rolling_market_stats]
enabled = true
return_windows = [5, 20, 60, 120]
volatility_windows = [20, 60]
sma_windows = [20, 60]
relative_strength_windows = [20, 60]
market_summary = true
market_summary_window = 20
minimum_history = "zero_fill"
"""
        template += obs_block

    elif kind == "bptt_15d":
        # 15d BPTT
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 15", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 60", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 15", template)

    elif kind == "ultra_mega_15d":
        # 15d BPTT
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 15", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 60", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 15", template)
        # Risk adjusted reward
        template = re.sub(r'(?m)^kind\s*=\s*"log_return"$', 'kind = "risk_adjusted_return"', template)
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.5", template)
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.01", template)
        # Rich sensors
        obs_block = """
[observations]
schema = "default"
providers = ["market", "portfolio", "agent_state", "rolling_market_stats"]

[observations.rolling_market_stats]
enabled = true
return_windows = [5, 20, 60, 120]
volatility_windows = [20, 60]
sma_windows = [20, 60]
relative_strength_windows = [20, 60]
market_summary = true
market_summary_window = 20
minimum_history = "zero_fill"
"""
        template += obs_block

    return template

def generate_all_configs(seeds: List[int]) -> List[Path]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for cond_key in CONDITIONS.keys():
        for seed in seeds:
            cfg_text = build_config_text(seed, cond_key)
            cfg_path = CONFIGS_DIR / f"{cond_key}-seed-{seed}.toml"
            cfg_path.write_text(cfg_text, encoding="utf-8")
            generated.append(cfg_path)
    print(f"Generated {len(generated)} experiment configs in {CONFIGS_DIR}")
    return generated

if __name__ == "__main__":
    generate_all_configs([41, 42])
