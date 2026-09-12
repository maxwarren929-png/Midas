#!/usr/bin/env python3
"""
Matrix V5 Generator: Multi-Asset Historical Real-World Stress Testing (2003 - 2023)
Assets: SPY, QQQ, TLT, IWM, XLE

Events:
  1. ev1_2008_gfc: 2008 Global Financial Crisis (Oct 2007 - Mar 2009) [Trained on 2003-2006 Bull]
  2. ev2_2018_volmageddon: 2018 Volmageddon & Q4 Sell-off (Feb 2018 - Dec 2018) [Trained on 2003-2014 Multi-Regime]
  3. ev3_2020_covid: 2020 COVID Flash Crash & Rebound (Feb 2020 - Aug 2020) [Trained on 2003-2014 Multi-Regime]
  4. ev4_2022_inflation_bear: 2022 Inflation Bear Market (Jan 2022 - Dec 2022) [Trained on 2003-2014 Multi-Regime]

Models:
  1. m1_mega_combo_8d
  2. m2_mega_combo_15d
  3. m4_heavyweight_gru
  4. m5_mega_8d_plus_evolvable
  5. 08_risk_adjusted_reward
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "camber_matrix_v5"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "camber_matrix_v5"
REPORTS_DIR = ROOT / "reports" / "camber_matrix_v5"
BASE_TEMPLATE_PATH = ROOT / "configs" / "prompt3b" / "experiment-ppo.toml"

EVENTS = {
    "ev1_2008_gfc": {
        "description": "2008 Global Financial Crisis (Oct 2007 - Mar 2009)",
        "training": ("2003-01-02", "2006-12-29"),
        "validation": ("2007-01-03", "2007-09-28"),
        "test": ("2007-10-01", "2009-03-31"),
    },
    "ev2_2018_volmageddon": {
        "description": "2018 Volmageddon & Q4 Sell-Off (Feb 2018 - Dec 2018)",
        "training": ("2003-01-02", "2014-12-31"),
        "validation": ("2015-01-02", "2016-12-30"),
        "test": ("2018-02-01", "2018-12-31"),
    },
    "ev3_2020_covid": {
        "description": "2020 COVID Flash Crash & Rebound (Feb 2020 - Aug 2020)",
        "training": ("2003-01-02", "2014-12-31"),
        "validation": ("2015-01-02", "2016-12-30"),
        "test": ("2020-02-03", "2020-08-31"),
    },
    "ev4_2022_inflation_bear": {
        "description": "2022 Inflation Bear Market (Jan 2022 - Dec 2022)",
        "training": ("2003-01-02", "2014-12-31"),
        "validation": ("2015-01-02", "2016-12-30"),
        "test": ("2022-01-03", "2022-12-30"),
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
    "m4_heavyweight_gru": {
        "description": "Heavyweight GRU (Memory 32, [64, 32], seq=16, rollout=64 + Rich Sensors + Risk Reward)",
        "kind": "heavyweight_gru",
    },
    "m5_mega_8d_plus_evolvable": {
        "description": "Mega Combo 8d + Evolvable Traits",
        "kind": "mega_8d_evolvable",
    },
    "08_risk_adjusted_reward": {
        "description": "Risk-Adjusted Reward (PPO + DD=0.5, TO=0.01 penalties)",
        "kind": "risk_reward_light",
    },
}


def build_config_text(seed: int, ev_key: str, mod_key: str) -> str:
    template = BASE_TEMPLATE_PATH.read_text(encoding="utf-8")
    ev = EVENTS[ev_key]
    mod = MODELS[mod_key]
    
    # 1. Update multi-asset universe and dataset path
    template = re.sub(r'assets\s*=\s*\[.*\]', 'assets = ["SPY", "QQQ", "TLT", "IWM", "XLE"]', template)
    template = re.sub(r'(?m)^path\s*=\s*"data/.*"$', 'path = "data/historical/market_2003_2023.csv"', template)
    
    # 2. Update splits
    template = re.sub(r'(?m)^start\s*=\s*"2010-01-04"$', f'start = "{ev["training"][0]}"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2014-10-31"$', f'end = "{ev["training"][1]}"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2014-11-03"$', f'start = "{ev["validation"][0]}"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2015-11-13"$', f'end = "{ev["validation"][1]}"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2015-11-16"$', f'start = "{ev["test"][0]}"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2016-11-25"$', f'end = "{ev["test"][1]}"', template)

    # 3. Update seed and checkpoint target
    template = re.sub(r'(?m)^seed\s*=.*$', f"seed = {seed}", template)
    template = re.sub(
        r'(?m)^path\s*=\s*"checkpoints/.*"$',
        f'path = "checkpoints/camber_matrix_v5/{ev_key}__{mod_key}-seed-{seed}.bin"',
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


def generate_all_configs(seeds: List[int] = [41]) -> List[Path]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for ev_key in EVENTS.keys():
        for mod_key in MODELS.keys():
            for seed in seeds:
                cfg_text = build_config_text(seed, ev_key, mod_key)
                cfg_path = CONFIGS_DIR / f"{ev_key}__{mod_key}-seed-{seed}.toml"
                cfg_path.write_text(cfg_text, encoding="utf-8")
                generated.append(cfg_path)
    print(f"Generated {len(generated)} experiment configs in {CONFIGS_DIR}")
    return generated


if __name__ == "__main__":
    generate_all_configs([41])
