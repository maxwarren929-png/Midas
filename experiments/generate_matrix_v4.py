#!/usr/bin/env python3
"""
Matrix V4 Generator: Testing 3 Non-Bull Market Scenarios across 5 Advanced Models.
Scenarios:
  1. s1_downtrend: Pure Bear / Downtrend Market (Prompt 9 Downtrend)
  2. s2_sideways: Pure Sideways / Chop Market (Prompt 9 Sideways)
  3. s3_regime_switching: Balanced Multi-Regime Switching Market (Prompt 9 Regime-Switching)

Models:
  1. m1_mega_combo_8d: Rich Sensors + Risk-Adjusted Reward + 8d BPTT
  2. m2_mega_combo_15d: Rich Sensors + Risk-Adjusted Reward + 15d BPTT
  3. m3_evolvable_traits: PPO + Genetic Evolvable Risk Traits
  4. m4_heavyweight_gru: GRU memory 32 + [64, 32] + seq=16, rollout=64 + Rich Sensors + Risk Reward
  5. m5_mega_8d_plus_evolvable: Mega Combo 8d + Genetic Evolvable Traits
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "camber_matrix_v4"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "camber_matrix_v4"
REPORTS_DIR = ROOT / "reports" / "camber_matrix_v4"
BASE_TEMPLATE_PATH = ROOT / "configs" / "prompt3b" / "experiment-ppo.toml"

SCENARIOS = {
    "s1_downtrend": {
        "description": "Scenario 1: Pure Bear / Downtrend Market",
        "data_path": "data/prompt9/downtrend/downtrend-01/market.csv",
        "training": ("2010-01-04", "2014-08-08"),
        "validation": ("2014-08-11", "2015-05-15"),
        "test": ("2015-05-18", "2016-02-19"),
    },
    "s2_sideways": {
        "description": "Scenario 2: Pure Sideways / Chop Mean-Reversion Market",
        "data_path": "data/prompt9/sideways/sideways-01/market.csv",
        "training": ("2010-01-04", "2014-08-08"),
        "validation": ("2014-08-11", "2015-05-15"),
        "test": ("2015-05-18", "2016-02-19"),
    },
    "s3_regime_switching": {
        "description": "Scenario 3: Balanced Multi-Regime Switching Market",
        "data_path": "data/prompt9/regime-switching/regime-switching-01/market.csv",
        "training": ("2010-01-04", "2014-08-08"),
        "validation": ("2014-08-11", "2015-05-15"),
        "test": ("2015-05-18", "2016-02-19"),
    },
}

MODELS = {
    "m1_mega_combo_8d": {
        "description": "Mega Combo 8d (Rich Sensors + Risk Reward + 8d BPTT)",
        "kind": "mega_8d",
    },
    "m2_mega_combo_15d": {
        "description": "Mega Combo 15d (Rich Sensors + Risk Reward + 15d BPTT)",
        "kind": "mega_15d",
    },
    "m3_evolvable_traits": {
        "description": "Evolvable Risk Traits (Genetically evolvable drawdown/turnover aversion)",
        "kind": "evolvable_traits",
    },
    "m4_heavyweight_gru": {
        "description": "Heavyweight GRU Combo (Memory 32, [64, 32], seq=16, rollout=64 + Rich Sensors + Risk Reward)",
        "kind": "heavyweight_gru",
    },
    "m5_mega_8d_plus_evolvable": {
        "description": "Mega Combo 8d + Evolvable Traits (Fusion of 8d Mega Combo and Genetic Evolution)",
        "kind": "mega_8d_evolvable",
    },
}


def build_config_text(seed: int, sc_key: str, mod_key: str) -> str:
    template = BASE_TEMPLATE_PATH.read_text(encoding="utf-8")
    sc = SCENARIOS[sc_key]
    mod = MODELS[mod_key]
    
    # 1. Update data path & splits
    template = re.sub(r'(?m)^path\s*=\s*"data/.*"$', f'path = "{sc["data_path"]}"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2010-01-04"$', f'start = "{sc["training"][0]}"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2014-10-31"$', f'end = "{sc["training"][1]}"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2014-11-03"$', f'start = "{sc["validation"][0]}"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2015-11-13"$', f'end = "{sc["validation"][1]}"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2015-11-16"$', f'start = "{sc["test"][0]}"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2016-11-25"$', f'end = "{sc["test"][1]}"', template)

    # 2. Update seed and checkpoint target
    template = re.sub(r'(?m)^seed\s*=.*$', f"seed = {seed}", template)
    template = re.sub(
        r'(?m)^path\s*=\s*"checkpoints/.*"$',
        f'path = "checkpoints/camber_matrix_v4/{sc_key}__{mod_key}-seed-{seed}.bin"',
        template
    )

    kind = mod["kind"]

    if kind == "mega_8d":
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 8", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 32", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 8", template)
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

    elif kind == "mega_15d":
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

    elif kind == "evolvable_traits":
        template = re.sub(r'(?m)^enabled\s*=\s*false$', "enabled = true", template)
        template = template.replace(
            "[brain.traits.drawdown_sensitivity]\ninitial = 0.1\nminimum = 0.0\nmaximum = 2.0\nmutation_scale = 0.0\nevolvable = false",
            "[brain.traits.drawdown_sensitivity]\ninitial = 0.5\nminimum = 0.0\nmaximum = 2.0\nmutation_scale = 0.2\nevolvable = true"
        )
        template = template.replace(
            "[brain.traits.turnover_sensitivity]\ninitial = 0.01\nminimum = 0.0\nmaximum = 1.0\nmutation_scale = 0.0\nevolvable = false",
            "[brain.traits.turnover_sensitivity]\ninitial = 0.01\nminimum = 0.0\nmaximum = 1.0\nmutation_scale = 0.1\nevolvable = true"
        )

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

    return template


def generate_all_configs(seeds: List[int] = [41]) -> List[Path]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for sc_key in SCENARIOS.keys():
        for mod_key in MODELS.keys():
            for seed in seeds:
                cfg_text = build_config_text(seed, sc_key, mod_key)
                cfg_path = CONFIGS_DIR / f"{sc_key}__{mod_key}-seed-{seed}.toml"
                cfg_path.write_text(cfg_text, encoding="utf-8")
                generated.append(cfg_path)
    print(f"Generated {len(generated)} experiment configs in {CONFIGS_DIR}")
    return generated


if __name__ == "__main__":
    generate_all_configs([41])
