#!/usr/bin/env python3
"""
Matrix Growth Generator: 4 Aggressive High-Alpha Configurations
Dataset: SPY, QQQ, TLT, IWM, XLE (2003 - 2026)

Splits:
  - Training: 2003-01-02 to 2022-12-30 (20-year multi-regime historical grounding)
  - Validation: 2023-01-03 to 2024-08-09
  - Out-of-Sample Test (Past 2 Years): 2024-08-12 to 2026-08-14 (504 trading days)

Configs:
  1. g1_aggressive_mega_15d: 15d BPTT + Rich Sensors + Pure Log Return + 25% Trade Fraction + 85% Max Allocation
  2. g2_high_conviction_gru: 32-Unit GRU + 16d BPTT + [64, 32] + Light DD Penalty (0.05) + 35% Trade Fraction + 90% Max Allocation
  3. g3_momentum_rotator: 15d BPTT + Expanded Sensors + Pure Log Return + 40% Trade Fraction + 95% Max Allocation
  4. g4_balanced_growth_shield: 15d BPTT + [48, 24] + Moderate DD Penalty (0.10) + 20% Trade Fraction + 75% Max Allocation
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "camber_matrix_growth"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "camber_matrix_growth"
REPORTS_DIR = ROOT / "reports" / "camber_matrix_growth"
BASE_TEMPLATE_PATH = ROOT / "configs" / "prompt3b" / "experiment-ppo.toml"

GROWTH_MODELS = {
    "g1_aggressive_mega_15d": {
        "description": "Aggressive Mega 15d (Pure Log Return, 25% Trade Size, 85% Max Allocation)",
        "kind": "g1_aggressive",
    },
    "g2_high_conviction_gru": {
        "description": "High Conviction 32-Unit GRU (35% Trade Size, 90% Max Allocation, Light DD=0.05)",
        "kind": "g2_conviction",
    },
    "g3_momentum_rotator": {
        "description": "Momentum Rotator (Pure Log Return, 40% Trade Size, 95% Max Allocation)",
        "kind": "g3_rotator",
    },
    "g4_balanced_growth_shield": {
        "description": "Balanced Growth Shield (20% Trade Size, 75% Max Allocation, Moderate DD=0.10)",
        "kind": "g4_balanced",
    },
}


def build_config_text(seed: int, mod_key: str) -> str:
    template = BASE_TEMPLATE_PATH.read_text(encoding="utf-8")
    mod = GROWTH_MODELS[mod_key]
    
    # 1. Multi-asset universe and dataset path
    template = re.sub(r'assets\s*=\s*\[.*\]', 'assets = ["SPY", "QQQ", "TLT", "IWM", "XLE"]', template)
    template = re.sub(r'(?m)^path\s*=\s*"data/.*"$', 'path = "data/current/current_market_2003_2026.csv"', template)
    
    # 2. Splits: 2003-2022 Training, 2023-2024 Validation, 2024-2026 Past 2 Years Test
    template = re.sub(r'(?m)^start\s*=\s*"2010-01-04"$', 'start = "2003-01-02"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2014-10-31"$', 'end = "2022-12-30"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2014-11-03"$', 'start = "2023-01-03"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2015-11-13"$', 'end = "2024-08-09"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2015-11-16"$', 'start = "2024-08-12"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2016-11-25"$', 'end = "2026-08-14"', template)

    # 3. Seed and checkpoint target
    template = re.sub(r'(?m)^seed\s*=.*$', f"seed = {seed}", template)
    template = re.sub(
        r'(?m)^path\s*=\s*"checkpoints/.*"$',
        f'path = "checkpoints/camber_matrix_growth/{mod_key}-seed-{seed}.bin"',
        template
    )

    kind = mod["kind"]

    if kind == "g1_aggressive":
        # 15d BPTT
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 15", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 60", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 15", template)
        # Sizing
        template = re.sub(r'(?m)^maximum_trade_fraction\s*=.*$', "maximum_trade_fraction = 0.25", template)
        template = template.replace(
            "[brain.traits.maximum_allocation]\ninitial = 0.5\nminimum = 0.01\nmaximum = 1.0",
            "[brain.traits.maximum_allocation]\ninitial = 0.85\nminimum = 0.10\nmaximum = 1.0"
        )
        # Reward: Pure Log Return, zero DD penalty
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.0", template)
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.0005", template)
        # Rich Sensors
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

    elif kind == "g2_conviction":
        # 32-unit GRU, 16d BPTT
        template = re.sub(r'(?m)^hidden_layers\s*=.*$', "hidden_layers = [64, 32]", template)
        template = re.sub(r'(?m)^memory_size\s*=.*$', "memory_size = 32", template)
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 16", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 64", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 16", template)
        # Sizing
        template = re.sub(r'(?m)^maximum_trade_fraction\s*=.*$', "maximum_trade_fraction = 0.35", template)
        template = template.replace(
            "[brain.traits.maximum_allocation]\ninitial = 0.5\nminimum = 0.01\nmaximum = 1.0",
            "[brain.traits.maximum_allocation]\ninitial = 0.90\nminimum = 0.10\nmaximum = 1.0"
        )
        # Reward: Risk Adjusted with very light DD penalty (0.05)
        template = template.replace('[reward]\nkind = "log_return"', '[reward]\nkind = "risk_adjusted_return"')
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.05", template)
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.001", template)
        # Rich Sensors
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

    elif kind == "g3_rotator":
        # 15d BPTT, memory=16
        template = re.sub(r'(?m)^memory_size\s*=.*$', "memory_size = 16", template)
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 15", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 60", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 15", template)
        # Sizing
        template = re.sub(r'(?m)^maximum_trade_fraction\s*=.*$', "maximum_trade_fraction = 0.40", template)
        template = template.replace(
            "[brain.traits.maximum_allocation]\ninitial = 0.5\nminimum = 0.01\nmaximum = 1.0",
            "[brain.traits.maximum_allocation]\ninitial = 0.95\nminimum = 0.10\nmaximum = 1.0"
        )
        # Reward: Pure Log Return
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.0", template)
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.0001", template)
        # Rich Sensors
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

    elif kind == "g4_balanced":
        # 15d BPTT, [48, 24], memory=16
        template = re.sub(r'(?m)^hidden_layers\s*=.*$', "hidden_layers = [48, 24]", template)
        template = re.sub(r'(?m)^memory_size\s*=.*$', "memory_size = 16", template)
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 15", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 60", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 15", template)
        # Sizing
        template = re.sub(r'(?m)^maximum_trade_fraction\s*=.*$', "maximum_trade_fraction = 0.20", template)
        template = template.replace(
            "[brain.traits.maximum_allocation]\ninitial = 0.5\nminimum = 0.01\nmaximum = 1.0",
            "[brain.traits.maximum_allocation]\ninitial = 0.75\nminimum = 0.10\nmaximum = 1.0"
        )
        # Reward: Risk Adjusted with moderate DD penalty (0.10)
        template = template.replace('[reward]\nkind = "log_return"', '[reward]\nkind = "risk_adjusted_return"')
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.10", template)
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.002", template)
        # Rich Sensors
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


def generate_all_configs(seeds: List[int] = [41, 42]) -> List[Path]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for mod_key in GROWTH_MODELS.keys():
        for seed in seeds:
            cfg_text = build_config_text(seed, mod_key)
            cfg_path = CONFIGS_DIR / f"{mod_key}-seed-{seed}.toml"
            cfg_path.write_text(cfg_text, encoding="utf-8")
            generated.append(cfg_path)
    print(f"Generated {len(generated)} experiment configs in {CONFIGS_DIR}")
    return generated


if __name__ == "__main__":
    generate_all_configs([41, 42])
