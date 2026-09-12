#!/usr/bin/env python3
"""
Matrix Macro 16 Generator:
Full 16-Asset Macro Universe (Equities, 3x Bull, 3x Bear Inverses, Commodities, Crypto, Treasuries):
Assets: AAPL, BTC, GLD, IWM, MSFT, NVDA, QQQ, SLV, SOXL, SPXU, SPY, SQQQ, TLT, TQQQ, UPRO, XLE

Splits:
  - Training: 2015-01-02 to 2022-12-30 (8 years: 2018 drop, 2020 COVID crash, 2022 inflation bear)
  - Validation: 2023-01-03 to 2024-08-09
  - Out-of-Sample Test (Past 2 Years): 2024-08-12 to 2026-08-14 (504 trading days)
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "camber_matrix_macro16"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "camber_matrix_macro16"
REPORTS_DIR = ROOT / "reports" / "camber_matrix_macro16"
BASE_TEMPLATE_PATH = ROOT / "configs" / "prompt3b" / "experiment-ppo.toml"

ASSETS = [
    "AAPL", "BTC", "GLD", "IWM", "MSFT", "NVDA", "QQQ",
    "SLV", "SOXL", "SPXU", "SPY", "SQQQ", "TLT", "TQQQ", "UPRO", "XLE"
]

MODELS = {
    "m16_alpha_gru_32": {
        "description": "Macro 32-Unit GRU (35% Trade Size, 90% Max Allocation, Light DD=0.05)",
        "kind": "alpha_gru_32",
    },
    "m16_mega_combo_15d": {
        "description": "Macro Mega Combo 15d (Pure Log Return, 25% Trade Size, 85% Max Allocation)",
        "kind": "mega_15d",
    },
    "m16_momentum_rotator": {
        "description": "Macro Momentum Rotator (Pure Log Return, 40% Trade Size, 95% Max Allocation)",
        "kind": "momentum_rotator",
    },
    "m16_balanced_shield": {
        "description": "Macro Balanced Shield (20% Trade Size, 75% Max Allocation, Moderate DD=0.10)",
        "kind": "balanced_shield",
    },
}


def build_config_text(seed: int, mod_key: str) -> str:
    template = BASE_TEMPLATE_PATH.read_text(encoding="utf-8")
    mod = MODELS[mod_key]
    
    # 1. 16-Asset Macro Universe & Data Path
    asset_str = json.dumps(ASSETS)
    template = re.sub(r'assets\s*=\s*\[.*\]', f'assets = {asset_str}', template)
    template = re.sub(r'(?m)^path\s*=\s*"data/.*"$', 'path = "data/macro/macro_16_assets_2015_2026.csv"', template)
    
    # 2. Splits
    template = re.sub(r'(?m)^start\s*=\s*"2010-01-04"$', 'start = "2015-01-02"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2014-10-31"$', 'end = "2022-12-30"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2014-11-03"$', 'start = "2023-01-03"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2015-11-13"$', 'end = "2024-08-09"', template)
    template = re.sub(r'(?m)^start\s*=\s*"2015-11-16"$', 'start = "2024-08-12"', template)
    template = re.sub(r'(?m)^end\s*=\s*"2016-11-25"$', 'end = "2026-08-14"', template)

    # 3. Seed and checkpoint target
    template = re.sub(r'(?m)^seed\s*=.*$', f"seed = {seed}", template)
    template = re.sub(
        r'(?m)^path\s*=\s*"checkpoints/.*"$',
        f'path = "checkpoints/camber_matrix_macro16/{mod_key}-seed-{seed}.bin"',
        template
    )

    kind = mod["kind"]

    if kind == "alpha_gru_32":
        # 32-unit GRU, 16d BPTT
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

    elif kind == "mega_15d":
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

    elif kind == "momentum_rotator":
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

    elif kind == "balanced_shield":
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
    for mod_key in MODELS.keys():
        for seed in seeds:
            cfg_text = build_config_text(seed, mod_key)
            cfg_path = CONFIGS_DIR / f"{mod_key}-seed-{seed}.toml"
            cfg_path.write_text(cfg_text, encoding="utf-8")
            generated.append(cfg_path)
    print(f"Generated {len(generated)} experiment configs in {CONFIGS_DIR}")
    return generated


if __name__ == "__main__":
    generate_all_configs([41])
