#!/usr/bin/env python3
"""
Generate Growth Historical Stress Testing Configs:
4 Hyper-Growth Models across 4 Real-World Historical Events.

Events:
  1. ev1_2008_gfc: 2008 Global Financial Crisis (Oct 2007 - Mar 2009)
  2. ev2_2018_volmageddon: 2018 Volmageddon & Q4 Sell-off (Feb 2018 - Dec 2018)
  3. ev3_2020_covid: 2020 COVID Flash Crash & Rebound (Feb 2020 - Aug 2020)
  4. ev4_2022_inflation_bear: 2022 Inflation Bear Market (Jan 2022 - Dec 2022)

Growth Models:
  1. g1_aggressive_mega_15d: 15d BPTT + Pure Log Return + 25% Trade Size + 85% Max Allocation
  2. g2_high_conviction_gru: 32-Unit GRU + 16d BPTT + 35% Trade Size + 90% Max Allocation + Light DD=0.05
  3. g3_momentum_rotator: 15d BPTT + Pure Log Return + 40% Trade Size + 95% Max Allocation
  4. g4_balanced_growth_shield: 15d BPTT + 20% Trade Size + 75% Max Allocation + Moderate DD=0.10
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "growth_historical_stress"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "growth_historical_stress"
REPORTS_DIR = ROOT / "reports" / "growth_historical_stress"
BASE_TEMPLATE_PATH = ROOT / "configs" / "prompt3b" / "experiment-ppo.toml"

EVENTS = {
    "ev1_2008_gfc": {
        "description": "2008 Global Financial Crisis (Oct 2007 - Mar 2009)",
        "training": ("2003-01-02", "2006-12-29"),
        "validation": ("2007-01-03", "2007-09-28"),
        "test": ("2007-10-01", "2009-03-31"),
    },
    "ev2_2018_volmageddon": {
        "description": "2018 Volmageddon & Q4 Sell-off (Feb 2018 - Dec 2018)",
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


def build_config_text(seed: int, ev_key: str, mod_key: str) -> str:
    template = BASE_TEMPLATE_PATH.read_text(encoding="utf-8")
    ev = EVENTS[ev_key]
    mod = GROWTH_MODELS[mod_key]
    
    # 1. Multi-asset universe and dataset path
    template = re.sub(r'assets\s*=\s*\[.*\]', 'assets = ["SPY", "QQQ", "TLT", "IWM", "XLE"]', template)
    template = re.sub(r'(?m)^path\s*=\s*"data/.*"$', 'path = "data/current/current_market_2003_2026.csv"', template)
    
    # 2. Splits
    template = re.sub(r'(?m)^start\s*=\s*"2010-01-04"$', f'start = "{ev["training"][0]}"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2014-10-31"$', f'end = "{ev["training"][1]}"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2014-11-03"$', f'start = "{ev["validation"][0]}"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2015-11-13"$', f'end = "{ev["validation"][1]}"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2015-11-16"$', f'start = "{ev["test"][0]}"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2016-11-25"$', f'end = "{ev["test"][1]}"', template)

    # 3. Seed and checkpoint target
    template = re.sub(r'(?m)^seed\s*=.*$', f"seed = {seed}", template)
    template = re.sub(
        r'(?m)^path\s*=\s*"checkpoints/.*"$',
        f'path = "checkpoints/growth_historical_stress/{ev_key}__{mod_key}-seed-{seed}.bin"',
        template
    )

    kind = mod["kind"]

    if kind == "g1_aggressive":
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 15", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 60", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 15", template)
        template = re.sub(r'(?m)^maximum_trade_fraction\s*=.*$', "maximum_trade_fraction = 0.25", template)
        template = template.replace(
            "[brain.traits.maximum_allocation]\ninitial = 0.5\nminimum = 0.01\nmaximum = 1.0",
            "[brain.traits.maximum_allocation]\ninitial = 0.85\nminimum = 0.10\nmaximum = 1.0"
        )
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.0", template)
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.0005", template)
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
        template = re.sub(r'(?m)^hidden_layers\s*=.*$', "hidden_layers = [64, 32]", template)
        template = re.sub(r'(?m)^memory_size\s*=.*$', "memory_size = 32", template)
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 16", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 64", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 16", template)
        template = re.sub(r'(?m)^maximum_trade_fraction\s*=.*$', "maximum_trade_fraction = 0.35", template)
        template = template.replace(
            "[brain.traits.maximum_allocation]\ninitial = 0.5\nminimum = 0.01\nmaximum = 1.0",
            "[brain.traits.maximum_allocation]\ninitial = 0.90\nminimum = 0.10\nmaximum = 1.0"
        )
        template = template.replace('[reward]\nkind = "log_return"', '[reward]\nkind = "risk_adjusted_return"')
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.05", template)
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.001", template)
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
        template = re.sub(r'(?m)^memory_size\s*=.*$', "memory_size = 16", template)
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 15", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 60", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 15", template)
        template = re.sub(r'(?m)^maximum_trade_fraction\s*=.*$', "maximum_trade_fraction = 0.40", template)
        template = template.replace(
            "[brain.traits.maximum_allocation]\ninitial = 0.5\nminimum = 0.01\nmaximum = 1.0",
            "[brain.traits.maximum_allocation]\ninitial = 0.95\nminimum = 0.10\nmaximum = 1.0"
        )
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.0", template)
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.0001", template)
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
        template = re.sub(r'(?m)^hidden_layers\s*=.*$', "hidden_layers = [48, 24]", template)
        template = re.sub(r'(?m)^memory_size\s*=.*$', "memory_size = 16", template)
        template = re.sub(r'(?m)^sequence_length\s*=.*$', "sequence_length = 15", template)
        template = re.sub(r'(?m)^rollout_length\s*=.*$', "rollout_length = 60", template)
        template = re.sub(r'(?m)^minibatch_size\s*=.*$', "minibatch_size = 15", template)
        template = re.sub(r'(?m)^maximum_trade_fraction\s*=.*$', "maximum_trade_fraction = 0.20", template)
        template = template.replace(
            "[brain.traits.maximum_allocation]\ninitial = 0.5\nminimum = 0.01\nmaximum = 1.0",
            "[brain.traits.maximum_allocation]\ninitial = 0.75\nminimum = 0.10\nmaximum = 1.0"
        )
        template = template.replace('[reward]\nkind = "log_return"', '[reward]\nkind = "risk_adjusted_return"')
        template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', "drawdown_penalty_weight = 0.10", template)
        template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.002", template)
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
    for ev_key in EVENTS.keys():
        for mod_key in GROWTH_MODELS.keys():
            for seed in seeds:
                cfg_text = build_config_text(seed, ev_key, mod_key)
                cfg_path = CONFIGS_DIR / f"{ev_key}__{mod_key}-seed-{seed}.toml"
                cfg_path.write_text(cfg_text, encoding="utf-8")
                generated.append(cfg_path)
    print(f"Generated {len(generated)} experiment configs in {CONFIGS_DIR}")
    return generated


if __name__ == "__main__":
    generate_all_configs([41])
