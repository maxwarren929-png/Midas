#!/usr/bin/env python3
"""Run the frozen Prompt 6 observation-schema ablation matrix."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get("MIDAS_BIN", str(ROOT / "target" / "release" / "midas-cli")))
TEMPLATES = {
    "default-control": ROOT / "configs/prompt6/default-control.toml",
    "default-ppo": ROOT / "configs/prompt6/default-ppo.toml",
    "momentum-ppo": ROOT / "configs/prompt6/momentum-ppo.toml",
    "volatility-trend-ppo": ROOT / "configs/prompt6/volatility-trend-ppo.toml",
    "combined-ppo": ROOT / "configs/prompt6/combined-ppo.toml",
    "combined-ppo-evolution": ROOT / "configs/prompt6/combined-ppo-evolution.toml",
}
RUN_CONFIGS = ROOT / "configs/prompt6/runs"
REPORTS = ROOT / "reports/prompt6"
CHECKPOINTS = ROOT / "checkpoints/prompt6"


def write_seed_config(condition: str, seed: int) -> Path:
    source = TEMPLATES[condition].read_text(encoding="utf-8")
    source = source.replace("seed = 41", f"seed = {seed}", 1)
    checkpoint = f"checkpoints/prompt6/{condition}-seed-{seed}.bin"
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("path = ") and index > 100:
            lines[index] = f'path = "{checkpoint}"'
            break
    else:
        raise RuntimeError(f"checkpoint path missing from {condition}")
    target = RUN_CONFIGS / f"{condition}-seed-{seed}.toml"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def run(command: list[str], output: Path) -> tuple[int, str, str]:
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    output.write_text(result.stdout + result.stderr, encoding="utf-8")
    return result.returncode, started, dt.datetime.now(dt.timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", choices=sorted(TEMPLATES), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    RUN_CONFIGS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = REPORTS / "run-manifest.csv"
    fields = [
        "condition", "seed", "split", "commit", "config", "checkpoint", "report_file",
        "start_time", "end_time", "status",
    ]
    with manifest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if handle.tell() == 0:
            writer.writeheader()
        for condition in args.conditions:
            for seed in args.seeds:
                config = write_seed_config(condition, seed)
                checkpoint = CHECKPOINTS / f"{condition}-seed-{seed}.bin"
                for split in ("training", "validation", "test"):
                    report = REPORTS / f"{condition}-seed-{seed}{'' if split == 'training' else '-' + split}.txt"
                    if split == "training":
                        command = [
                            str(BIN), "simulate", "--config", str(config), "--split", split,
                            "--checkpoint", str(checkpoint),
                        ]
                    else:
                        command = [
                            str(BIN), "evaluate", "--config", str(config), "--checkpoint",
                            str(checkpoint), "--split", split,
                        ]
                    status, started, ended = run(command, report)
                    writer.writerow({
                        "condition": condition,
                        "seed": seed,
                        "split": split,
                        "commit": commit,
                        "config": str(config.relative_to(ROOT)),
                        "checkpoint": str(checkpoint.relative_to(ROOT)),
                        "report_file": str(report.relative_to(ROOT)),
                        "start_time": started,
                        "end_time": ended,
                        "status": "passed" if status == 0 else f"failed:{status}",
                    })
                    handle.flush()
                    if status != 0:
                        raise SystemExit(f"{condition} seed {seed} {split} failed; see {report}")


if __name__ == "__main__":
    main()
