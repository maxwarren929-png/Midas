#!/usr/bin/env python3
"""Materialize Prompt 3B configs from the checked-in Prompt 2 schema."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = (ROOT / "configs" / "prompt2.toml").read_text(encoding="utf-8")
OUT = ROOT / "configs" / "prompt3b"
OUT.mkdir(parents=True, exist_ok=True)

TRAINING = ("2010-01-04", "2014-10-31")
VALIDATION = ("2014-11-03", "2015-11-13")
TEST = ("2015-11-16", "2016-11-25")


def replace_line(text: str, key: str, value: str) -> str:
    pattern = rf"(?m)^{re.escape(key)}\s*=.*$"
    result, count = re.subn(pattern, f"{key} = {value}", text, count=1)
    if count != 1:
        raise RuntimeError(f"could not replace {key}")
    return result


def replace_section_value(text: str, section: str, key: str, value: str) -> str:
    match = re.search(rf"(?ms)^\[{re.escape(section)}\]\n(.*?)(?=^\[|\Z)", text)
    if not match:
        raise RuntimeError(f"missing section {section}")
    body = match.group(1)
    body = replace_line(body, key, value)
    return text[: match.start(1)] + body + text[match.end(1) :]


def base_config(seed: int, order: str = "stable") -> str:
    text = TEMPLATE
    text = replace_line(text, "assets", '["SPY", "AAA", "BBB", "CCC"]')
    text = replace_line(text, "benchmark_asset", '"SPY"')
    text = replace_line(text, "path", '"data/prompt3b/market.csv"')
    text = replace_line(text, "population_size", "8")
    text = replace_line(text, "starting_cash", "10000.0")
    text = replace_line(text, "bankruptcy_threshold", "8950.0")
    text = replace_line(text, "daily_survival_cost", "0.5")
    text = replace_line(text, "seed", str(seed))
    text = text.replace('start = "2020-01-02"', f'start = "{TRAINING[0]}"', 1)
    text = text.replace('end = "2020-01-13"', f'end = "{TRAINING[1]}"', 1)
    text = text.replace('start = "2020-01-14"', f'start = "{VALIDATION[0]}"', 1)
    text = text.replace('end = "2020-01-23"', f'end = "{VALIDATION[1]}"', 1)
    text = text.replace('start = "2020-01-24"', f'start = "{TEST[0]}"', 1)
    text = text.replace('end = "2020-02-05"', f'end = "{TEST[1]}"', 1)
    text = text.replace('asset_order = "stable"', f'asset_order = "{order}"', 1)
    text = text.replace('every_steps = 0', 'every_steps = 128', 1)
    text = text.replace('path = "checkpoints/latest.bin"', 'path = "checkpoints/prompt3b/automatic.bin"', 1)
    # Make the pressure and policy-failure behavior explicit in every copy.
    text = text.replace('maximum_consecutive = 3', 'maximum_consecutive = 3', 1)
    return text


def set_learning(text: str, implementation: str, *, rate: str = "0.0003", rollout: str = "4", minibatch: str = "2", sequence: str = "2", epochs: str = "4", entropy: str = "0.01") -> str:
    text = replace_section_value(text, "learning", "implementation", f'"{implementation}"')
    for key, value in {
        "learning_rate": rate,
        "rollout_length": rollout,
        "minibatch_size": minibatch,
        "sequence_length": sequence,
        "optimisation_epochs": epochs,
        "entropy_coefficient": entropy,
    }.items():
        text = replace_section_value(text, "learning", key, value)
    return text


def set_evolution(text: str, enabled: bool) -> str:
    return replace_section_value(text, "evolution", "enabled", str(enabled).lower())


def set_reward(text: str, kind: str = "log_return", turnover: str = "0.0") -> str:
    # The base template has no reward table; append one after the evolution
    # table so all variants use the same explicit decomposition contract.
    text += (
        "\n[reward]\n"
        f'kind = "{kind}"\n'
        "drawdown_penalty_weight = 0.0\n"
        f"turnover_penalty_weight = {turnover}\n"
        "invalid_action_penalty = 0.05\n"
        "bankruptcy_penalty = 1.0\n"
        "clip_min = -5.0\n"
        "clip_max = 5.0\n"
    )
    return text


def set_trait_initial(text: str, trait: str, value: str) -> str:
    return replace_section_value(text, f"brain.traits.{trait}", "initial", value)


def write(name: str, text: str) -> None:
    (OUT / name).write_text(text, encoding="utf-8")


def main() -> None:
    # These four are the primary paired conditions. All non-intended fields
    # are generated from the same base function above.
    primary = {
        "control": ("no_learning", False, {}),
        "ppo": ("ppo", False, {}),
        "evolution-only": ("no_learning", True, {}),
        "ppo-evolution": ("ppo", True, {}),
    }
    for name, (learning, evolution, kwargs) in primary.items():
        text = base_config(42)
        text = set_learning(text, learning, **kwargs)
        text = set_evolution(text, evolution)
        write(f"experiment-{name}.toml", set_reward(text))

    conservative = base_config(42)
    conservative = set_learning(conservative, "ppo", rate="0.0003", rollout="64", minibatch="32", sequence="16", epochs="2")
    conservative = set_evolution(conservative, False)
    write("experiment-conservative.toml", set_reward(conservative))
    low_rate = set_learning(conservative, "ppo", rate="0.0001", rollout="64", minibatch="32", sequence="16", epochs="2")
    low_rate = set_trait_initial(low_rate, "learning_rate", "0.0001")
    write("experiment-low-rate.toml", set_reward(low_rate))
    low_entropy = set_learning(conservative, "ppo", rate="0.0003", rollout="64", minibatch="32", sequence="16", epochs="2", entropy="0.001")
    low_entropy = set_trait_initial(low_entropy, "entropy_coefficient", "0.001")
    write("experiment-low-entropy.toml", set_reward(low_entropy))
    rotated = base_config(42, order="episode_rotated")
    rotated = set_learning(rotated, "ppo")
    rotated = set_evolution(rotated, False)
    write("experiment-rotated-order.toml", set_reward(rotated))
    for name, coefficient in (("mild-turnover", "0.0005"), ("moderate-turnover", "0.001")):
        text = set_learning(conservative, "ppo", rate="0.0003", rollout="64", minibatch="32", sequence="16", epochs="2")
        write(f"experiment-{name}.toml", set_reward(text, "risk_adjusted_return", coefficient))

    runs = OUT / "runs"
    runs.mkdir(exist_ok=True)
    for condition in [
        "control",
        "ppo",
        "evolution-only",
        "ppo-evolution",
        "conservative",
        "low-rate",
        "low-entropy",
        "rotated-order",
        "mild-turnover",
        "moderate-turnover",
    ]:
        source = (OUT / f"experiment-{condition}.toml").read_text(encoding="utf-8")
        for seed in range(41, 51):
            text = replace_section_value(source, "simulation", "seed", str(seed))
            text = replace_section_value(
                text,
                "checkpoint",
                "path",
                f'"checkpoints/prompt3b/{condition}-seed-{seed}.bin"',
            )
            write_path = runs / f"{condition}-seed-{seed}.toml"
            write_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
