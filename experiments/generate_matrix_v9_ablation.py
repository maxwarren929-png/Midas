#!/usr/bin/env python3
"""
Matrix V9: 5-Way Ablation & Synergy Experiment
Models Tested:
  1. m16_momentum_rotator
  2. g2_high_conviction_gru
  3. g1_aggressive_mega_15d
  4. m2_mega_combo_15d

Runs Tested:
  - Run 1: Setting 1 ON (Asymmetric Sortino Reward)
  - Run 2: Setting 2 ON (High-Conviction Action Concentration: 50% Trade Size, 95% Max Allocation)
  - Run 3: Setting 3 ON (Trend Strength & SMA Divergence Sensors: SMA 10/20/50/100/200 + Multi-Horizon RS)
  - Run 4: Setting 4 ON (20-Asset Hyper-Growth Universe: NVDL, PLTR, TSLA, ETH, SOXL, TQQQ, SQQQ, BTC, GLD, etc.)
  - Run 5: ALL 4 SETTINGS ON COMBINED ("God-Mode / All-Engines-Fire")
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "matrix_v9_ablation"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "matrix_v9_ablation"
REPORTS_DIR = ROOT / "reports" / "matrix_v9_ablation"
BASE_TEMPLATE_PATH = ROOT / "configs" / "prompt3b" / "experiment-ppo.toml"

ASSETS_16 = [
    "AAPL", "BTC", "GLD", "IWM", "MSFT", "NVDA", "QQQ",
    "SLV", "SOXL", "SPXU", "SPY", "SQQQ", "TLT", "TQQQ", "UPRO", "XLE"
]

ASSETS_20 = [
    "AAPL", "AMD", "BTC", "GLD", "IWM", "META", "MSFT", "NVDA", "PLTR", "QQQ",
    "SLV", "SOXL", "SPXU", "SPY", "SQQQ", "TLT", "TQQQ", "TSLA", "UPRO", "XLE"
]

MODELS = {
    "m16_momentum_rotator": {
        "description": "Momentum Rotator (15d BPTT, 40% Trade Size, 95% Max Allocation)",
        "memory_size": 16, "hidden": "[32, 16]", "seq_len": 15, "rollout": 60, "minibatch": 15,
        "max_trade": 0.40, "max_alloc": 0.95, "reward_kind": "log_return", "dd_weight": 0.0,
    },
    "g2_high_conviction_gru": {
        "description": "High Conviction 32-Unit GRU (16d BPTT, 35% Trade Size, 90% Max Allocation)",
        "memory_size": 32, "hidden": "[64, 32]", "seq_len": 16, "rollout": 64, "minibatch": 16,
        "max_trade": 0.35, "max_alloc": 0.90, "reward_kind": "risk_adjusted_return", "dd_weight": 0.05,
    },
    "g1_aggressive_mega_15d": {
        "description": "Aggressive Mega 15d (15d BPTT, 25% Trade Size, 85% Max Allocation)",
        "memory_size": 8, "hidden": "[32, 16]", "seq_len": 15, "rollout": 60, "minibatch": 15,
        "max_trade": 0.25, "max_alloc": 0.85, "reward_kind": "log_return", "dd_weight": 0.0,
    },
    "m2_mega_combo_15d": {
        "description": "Mega Combo 15d (15d BPTT, 5% Trade Size, 50% Max Allocation)",
        "memory_size": 8, "hidden": "[32, 16]", "seq_len": 15, "rollout": 60, "minibatch": 15,
        "max_trade": 0.05, "max_alloc": 0.50, "reward_kind": "risk_adjusted_return", "dd_weight": 0.5,
    },
}

RUNS = {
    "run1_asymmetric_sortino": "Setting 1 ON: Asymmetric Sortino Convex Reward",
    "run2_top_k_concentration": "Setting 2 ON: High-Conviction Action Concentration (50% Trade, 95% Alloc)",
    "run3_trend_sma_sensors": "Setting 3 ON: Trend Strength & Full SMA Divergence Sensors (10/20/50/100/200)",
    "run4_hyper_20_assets": "Setting 4 ON: 20-Asset Hyper Universe (NVDL 2x, PLTR, TSLA, ETH, etc.)",
    "run5_all_combined_godmode": "ALL 4 SETTINGS ON: Asymmetric Sortino + Concentration + Trend Sensors + 20 Assets",
}


def build_config_text(seed: int, run_key: str, mod_key: str) -> str:
    template = BASE_TEMPLATE_PATH.read_text(encoding="utf-8")
    mod = MODELS[mod_key]
    
    # Determine Universe & Dataset Path based on run_key
    if run_key in ["run4_hyper_20_assets", "run5_all_combined_godmode"]:
        assets = ASSETS_20
        data_path = "data/hyper/hyper_20_assets_2016_2026.csv"
        # Dates: 2015-01-05 to 2026-08-14
        t_start, t_end = "2015-01-05", "2022-12-30"
        v_start, v_end = "2023-01-03", "2024-08-09"
        test_start, test_end = "2024-08-12", "2026-08-14"
    else:
        assets = ASSETS_16
        data_path = "data/macro/macro_16_assets_2015_2026.csv"
        # Dates: 2015-01-02 to 2026-08-14
        t_start, t_end = "2015-01-02", "2022-12-30"
        v_start, v_end = "2023-01-03", "2024-08-09"
        test_start, test_end = "2024-08-12", "2026-08-14"

    template = re.sub(r'assets\s*=\s*\[.*\]', f'assets = {json.dumps(assets)}', template)
    template = re.sub(r'(?m)^path\s*=\s*"data/.*"$', f'path = "{data_path}"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2010-01-04"$', f'start = "{t_start}"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2014-10-31"$', f'end = "{t_end}"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2014-11-03"$', f'start = "{v_start}"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2015-11-13"$', f'end = "{v_end}"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2015-11-16"$', f'start = "{test_start}"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2016-11-25"$', f'end = "{test_end}"', template)
    template = re.sub(r'(?m)^seed\s*=.*$', f"seed = {seed}", template)
    template = re.sub(
        r'(?m)^path\s*=\s*"checkpoints/.*"$',
        f'path = "checkpoints/matrix_v9_ablation/{run_key}__{mod_key}-seed-{seed}.bin"',
        template
    )

    # Architecture
    template = re.sub(r'(?m)^hidden_layers\s*=.*$', f"hidden_layers = {mod['hidden']}", template)
    template = re.sub(r'(?m)^memory_size\s*=.*$', f"memory_size = {mod['memory_size']}", template)
    template = re.sub(r'(?m)^sequence_length\s*=.*$', f"sequence_length = {mod['seq_len']}", template)
    template = re.sub(r'(?m)^rollout_length\s*=.*$', f"rollout_length = {mod['rollout']}", template)
    template = re.sub(r'(?m)^minibatch_size\s*=.*$', f"minibatch_size = {mod['minibatch']}", template)

    # Base Sizing & Allocation
    max_trade = mod["max_trade"]
    max_alloc = mod["max_alloc"]
    dd_penalty = mod["dd_weight"]
    reward_kind = mod["reward_kind"]

    # Apply Setting 1: Asymmetric Sortino Reward
    if run_key in ["run1_asymmetric_sortino", "run5_all_combined_godmode"]:
        reward_kind = "asymmetric_sortino"
        dd_penalty = 0.05

    # Apply Setting 2: High-Conviction Action Concentration
    if run_key in ["run2_top_k_concentration", "run5_all_combined_godmode"]:
        max_trade = 0.50
        max_alloc = 0.95

    # Apply Sizing Replacements
    template = re.sub(r'(?m)^maximum_trade_fraction\s*=.*$', f"maximum_trade_fraction = {max_trade}", template)
    template = template.replace(
        "[brain.traits.maximum_allocation]\ninitial = 0.5\nminimum = 0.01\nmaximum = 1.0",
        f"[brain.traits.maximum_allocation]\ninitial = {max_alloc}\nminimum = 0.10\nmaximum = 1.0"
    )

    # Apply Reward Replacements
    template = template.replace('[reward]\nkind = "log_return"', f'[reward]\nkind = "{reward_kind}"')
    template = re.sub(r'(?m)^drawdown_penalty_weight\s*=.*$', f"drawdown_penalty_weight = {dd_penalty}", template)
    template = re.sub(r'(?m)^turnover_penalty_weight\s*=.*$', "turnover_penalty_weight = 0.0005", template)

    # Determine Sensor Provider Configuration
    if run_key in ["run3_trend_sma_sensors", "run5_all_combined_godmode"]:
        # Expanded Trend Strength & SMA Divergence Sensors
        obs_block = """
[observations]
schema = "default"
providers = ["market", "portfolio", "agent_state", "rolling_market_stats"]

[observations.rolling_market_stats]
enabled = true
return_windows = [5, 10, 20, 60, 120, 250]
volatility_windows = [10, 20, 60]
sma_windows = [10, 20, 50, 100, 200]
relative_strength_windows = [10, 20, 60]
market_summary = true
market_summary_window = 20
minimum_history = "zero_fill"
"""
    else:
        # Standard Rich Sensors
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
    template = re.sub(r'(?m)^discontinuity_threshold\s*=.*$', 'discontinuity_threshold = 0.85', template)

    # Observations Block
    template += obs_block

    return template


def generate_all_configs(seeds: List[int] = [41]) -> List[Path]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for run_key in RUNS.keys():
        for mod_key in MODELS.keys():
            for seed in seeds:
                cfg_text = build_config_text(seed, run_key, mod_key)
                cfg_path = CONFIGS_DIR / f"{run_key}__{mod_key}-seed-{seed}.toml"
                cfg_path.write_text(cfg_text, encoding="utf-8")
                generated.append(cfg_path)
    print(f"Generated {len(generated)} experiment configs in {CONFIGS_DIR}")
    return generated


if __name__ == "__main__":
    generate_all_configs([41])
