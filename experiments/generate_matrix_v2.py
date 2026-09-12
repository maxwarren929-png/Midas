#!/usr/bin/env python3
"""
Camber Cloud Matrix Runner for the 5 New Advanced Settings + Baseline & Loser Formula.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "camber_matrix_v2"
REPORTS_DIR = ROOT / "reports" / "camber_matrix_v2"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "camber_matrix_v2"

BASE_TEMPLATE_PATH = ROOT / "configs" / "prompt3b" / "experiment-ppo.toml"

CONDITIONS: Dict[str, Dict[str, Any]] = {
    "01_baseline_ppo": {
        "description": "Baseline Momentum PPO (hidden=[32, 16], memory=8, log_return)",
        "type": "baseline",
    },
    "02_drawdown_penalty": {
        "description": "Standard PPO with Drawdown Penalty (weight=0.5)",
        "type": "baseline_penalty",
    },
    "03_turnover_penalty": {
        "description": "Standard PPO with Turnover Penalty (weight=0.01)",
        "type": "baseline_penalty",
    },
    "04_rich_sensors": {
        "description": "Setting 1: Multi-horizon Rolling Market Stats (5, 20, 60, 120-day momenta, volatilities, SMAs)",
        "type": "rich_sensors",
    },
    "05_tight_trade_limit": {
        "description": "Setting 2: Tight Defensive Trade Limit (max 2% daily trade fraction)",
        "type": "tight_trade",
    },
    "06_long_bptt_horizon": {
        "description": "Setting 3: Long Recurrent Sequence (seq=8, rollout=32, minibatch=8)",
        "type": "long_bptt",
    },
    "07_evolvable_risk_traits": {
        "description": "Setting 4: Genetically Evolvable Risk Aversion (evolvable drawdown & turnover traits)",
        "type": "evolvable_traits",
    },
    "08_risk_adjusted_reward": {
        "description": "Setting 5: Risk-Adjusted Reward (calibrated drawdown & turnover penalty formula)",
        "type": "risk_adjusted_reward",
    },
    "09_loser_formula": {
        "description": "Loser Formula Combo: Deep Recurrent [64, 32] + AdamW + PPO Evolution",
        "type": "loser_formula",
    },
}


def build_config_text(seed: int, cond_key: str) -> str:
    template = BASE_TEMPLATE_PATH.read_text(encoding="utf-8")
    
    # Update seed and paths
    template = re.sub(r'(?m)^seed\s*=.*$', f"seed = {seed}", template)
    template = re.sub(
        r'(?m)^path\s*=\s*"checkpoints/.*"$',
        f'path = "checkpoints/camber_matrix_v2/{cond_key}-seed-{seed}.bin"',
        template
    )

    cond_type = CONDITIONS[cond_key]["type"]

    if cond_type == "baseline_penalty":
        if "drawdown" in cond_key:
            template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.5", template)
        elif "turnover" in cond_key:
            template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.01", template)

    elif cond_type == "rich_sensors":
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

    elif cond_type == "tight_trade":
        template = re.sub(r'(?m)^maximum_trade_fraction\s*=.*$', "maximum_trade_fraction = 0.02", template)

    elif cond_type == "long_bptt":
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 8", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 32", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 8", template)

    elif cond_type == "evolvable_traits":
        template = re.sub(r'(?m)^enabled\s*=\s*false$', "enabled = true", template)
        template = template.replace(
            "[brain.traits.drawdown_sensitivity]\ninitial = 0.1\nminimum = 0.0\nmaximum = 2.0\nmutation_scale = 0.0\nevolvable = false",
            "[brain.traits.drawdown_sensitivity]\ninitial = 0.5\nminimum = 0.0\nmaximum = 2.0\nmutation_scale = 0.2\nevolvable = true"
        )
        template = template.replace(
            "[brain.traits.turnover_sensitivity]\ninitial = 0.01\nminimum = 0.0\nmaximum = 1.0\nmutation_scale = 0.0\nevolvable = false",
            "[brain.traits.turnover_sensitivity]\ninitial = 0.01\nminimum = 0.0\nmaximum = 1.0\nmutation_scale = 0.1\nevolvable = true"
        )

    elif cond_type == "risk_adjusted_reward":
        template = re.sub(r'(?m)^kind\s*=\s*"log_return"$', 'kind = "risk_adjusted_return"', template)
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.5", template)
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.01", template)

    elif cond_type == "loser_formula":
        # Deep Recurrent + AdamW + PPO Evolution
        template = re.sub(r'(?m)^hidden_layers\s*=.*$', "hidden_layers = [64, 32]", template)
        template = re.sub(r'(?m)^memory_size\s*=.*$', "memory_size = 16", template)
        template = re.sub(r'(?m)^enabled\s*=\s*false$', "enabled = true", template)
        template = re.sub(r'(?m)^learning_rate\s*=.*$', "learning_rate = 0.0000035", template)
        opt_text = """
[optimizer]
kind = "adamw"
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
weight_decay = 0.01
"""
        template += opt_text

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
