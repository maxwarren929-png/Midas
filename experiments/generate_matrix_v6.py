#!/usr/bin/env python3
"""
Matrix V6 Generator: The Last 30 Days Real-Time Market Experiment
Dataset: SPY, QQQ, TLT, IWM, XLE (2003 - 2026)

Splits:
  - Training: 2003-01-02 to 2022-12-30 (20-year multi-regime historical grounding)
  - Validation: 2023-01-03 to 2026-07-10
  - Out-of-Sample Test (Last 30 Days): 2026-07-15 to 2026-08-14

Models (The Final 4):
  1. m2_mega_combo_15d: 15-day memory + rich sensors + risk reward
  2. m4_heavyweight_gru: 32-unit GRU + [64, 32] + 16-day memory + rich sensors + risk reward
  3. m5_mega_8d_plus_evolvable: 8-day memory + rich sensors + evolvable traits
  4. 08_risk_adjusted_reward: Standard PPO + explicit drawdown and turnover penalty
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "camber_matrix_v6"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "camber_matrix_v6"
REPORTS_DIR = ROOT / "reports" / "camber_matrix_v6"
BASE_TEMPLATE_PATH = ROOT / "configs" / "prompt3b" / "experiment-ppo.toml"

MODELS = {
    "m2_mega_combo_15d": {
        "description": "Mega Combo 15d (15d BPTT + Rich Sensors + Risk Reward)",
        "kind": "mega_15d",
    },
    "m4_heavyweight_gru": {
        "description": "Heavyweight GRU (32-unit GRU + [64, 32] + 16d BPTT + Rich Sensors + Risk Reward)",
        "kind": "heavyweight_gru",
    },
    "m5_mega_8d_plus_evolvable": {
        "description": "Mega Combo 8d + Evolvable Risk Traits",
        "kind": "mega_8d_evolvable",
    },
    "08_risk_adjusted_reward": {
        "description": "Risk-Adjusted Reward Baseline (PPO + DD=0.5, TO=0.01 penalties)",
        "kind": "risk_reward_light",
    },
}


def build_config_text(seed: int, mod_key: str) -> str:
    template = BASE_TEMPLATE_PATH.read_text(encoding="utf-8")
    mod = MODELS[mod_key]
    
    # 1. Multi-asset universe and dataset path
    template = re.sub(r'assets\s*=\s*\[.*\]', 'assets = ["SPY", "QQQ", "TLT", "IWM", "XLE"]', template)
    template = re.sub(r'(?m)^path\s*=\s*"data/.*"$', 'path = "data/current/current_market_2003_2026.csv"', template)
    
    # 2. Splits: 2003-2022 Training, 2023-2026/07 Validation, 2026-07-15 to 2026-08-14 Last 30 Days Test
    template = re.sub(r'(?m)^start\s*=\s*"2010-01-04"$', 'start = "2003-01-02"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2014-10-31"$', 'end = "2022-12-30"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2014-11-03"$', 'start = "2023-01-03"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2015-11-13"$', 'end = "2026-07-10"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2015-11-16"$', 'start = "2026-07-15"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2016-11-25"$', 'end = "2026-08-14"', template)

    # 3. Seed and checkpoint target
    template = re.sub(r'(?m)^seed\s*=.*$', f"seed = {seed}", template)
    template = re.sub(
        r'(?m)^path\s*=\s*"checkpoints/.*"$',
        f'path = "checkpoints/camber_matrix_v6/{mod_key}-seed-{seed}.bin"',
        template
    )

    kind = mod["kind"]

    if kind == "mega_15d":
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 15", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 60", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 15", template)
        template = re.sub(r'(?m)^kind\s*=\s*"log_return"$', 'kind = "risk_adjusted_return"', template)
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.5", template)
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.01", template)
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

    elif kind == "heavyweight_gru":
        template = re.sub(r'(?m)^hidden_layers\s*=.*$', "hidden_layers = [64, 32]", template)
        template = re.sub(r'(?m)^memory_size\s*=.*$', "memory_size = 32", template)
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 16", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 64", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 16", template)
        template = re.sub(r'(?m)^kind\s*=\s*"log_return"$', 'kind = "risk_adjusted_return"', template)
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.5", template)
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.01", template)
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

    elif kind == "mega_8d_evolvable":
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 8", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 32", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 8", template)
        template = re.sub(r'(?m)^kind\s*=\s*"log_return"$', 'kind = "risk_adjusted_return"', template)
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.5", template)
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.01", template)
        template = re.sub(r'(?m)^enabled\s*=\s*false$', "enabled = true", template)
        template = template.replace(
            "[brain.traits.drawdown_sensitivity]\ninitial = 0.1\nminimum = 0.0\nmaximum = 2.0\nmutation_scale = 0.0\nevolvable = false",
            "[brain.traits.drawdown_sensitivity]\ninitial = 0.5\nminimum = 0.0\nmaximum = 2.0\nmutation_scale = 0.2\nevolvable = true"
        )
        template = template.replace(
            "[brain.traits.turnover_sensitivity]\ninitial = 0.01\nminimum = 0.0\nmaximum = 1.0\nmutation_scale = 0.0\nevolvable = false",
            "[brain.traits.turnover_sensitivity]\ninitial = 0.01\nminimum = 0.0\nmaximum = 1.0\nmutation_scale = 0.1\nevolvable = true"
        )
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

    elif kind == "risk_reward_light":
        template = re.sub(r'(?m)^kind\s*=\s*"log_return"$', 'kind = "risk_adjusted_return"', template)
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.5", template)
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.01", template)

    return template


def generate_all_configs(seeds: List[int] = [41, 42]) -> List[Path]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for mod_key in MODELS.keys():
        for seed in seeds:
            cfg_text = build_config_text(seed, mod_key)
            cfg_path = CONFIGS_DIR / f"{mod_key}-seed-{seed}.toml"
            cfg_path.write_text(cfg_text, encoding="utf-8")
            generated.append(cfg_path)
    print(f"Generated {len(generated)} experiment configs in {CONFIGS_DIR}")
    return generated


if __name__ == "__main__":
    generate_all_configs([41, 42])
