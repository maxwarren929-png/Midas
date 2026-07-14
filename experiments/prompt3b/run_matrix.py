#!/usr/bin/env python3
"""Run frozen Prompt 3B conditions and preserve raw CLI artifacts."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get("MIDAS_BIN", str(ROOT / "target" / "release" / "midas-cli")))
CONFIG_ROOT = ROOT / "configs" / "prompt3b" / "runs"
REPORT_ROOT = ROOT / "reports" / "prompt3b"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "prompt3b"


def run(command: list[str], output: Path) -> int:
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    output.write_text(result.stdout + result.stderr, encoding="utf-8")
    finished = dt.datetime.now(dt.timezone.utc).isoformat()
    return result.returncode, started, finished


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--trace", action="append", default=[])
    args = parser.parse_args()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = REPORT_ROOT / "run-manifest.csv"
    fieldnames = [
        "condition",
        "seed",
        "split",
        "commit",
        "config",
        "checkpoint",
        "report_file",
        "start_time",
        "end_time",
        "status",
    ]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with manifest.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if handle.tell() == 0:
            writer.writeheader()
        for condition in args.conditions:
            for seed in args.seeds:
                config = CONFIG_ROOT / f"{condition}-seed-{seed}.toml"
                checkpoint = CHECKPOINT_ROOT / f"{condition}-seed-{seed}.bin"
                report = REPORT_ROOT / f"{condition}-seed-{seed}.txt"
                command = [
                    str(BIN),
                    "simulate",
                    "--config",
                    str(config),
                    "--split",
                    "training",
                    "--checkpoint",
                    str(checkpoint),
                ]
                code, started, finished = run(command, report)
                writer.writerow(
                    {
                        "condition": condition,
                        "seed": seed,
                        "split": "training",
                        "commit": commit,
                        "config": str(config.relative_to(ROOT)),
                        "checkpoint": str(checkpoint.relative_to(ROOT)),
                        "report_file": str(report.relative_to(ROOT)),
                        "start_time": started,
                        "end_time": finished,
                        "status": "passed" if code == 0 else f"failed:{code}",
                    }
                )
                handle.flush()
                if code != 0:
                    raise SystemExit(f"{condition} seed {seed} failed; see {report}")
                for split in ("validation", "test"):
                    eval_report = REPORT_ROOT / f"{condition}-seed-{seed}-{split}.txt"
                    eval_command = [
                        str(BIN),
                        "evaluate",
                        "--config",
                        str(config),
                        "--checkpoint",
                        str(checkpoint),
                        "--split",
                        split,
                    ]
                    eval_code, eval_start, eval_end = run(eval_command, eval_report)
                    writer.writerow(
                        {
                            "condition": condition,
                            "seed": seed,
                            "split": split,
                            "commit": commit,
                            "config": str(config.relative_to(ROOT)),
                            "checkpoint": str(checkpoint.relative_to(ROOT)),
                            "report_file": str(eval_report.relative_to(ROOT)),
                            "start_time": eval_start,
                            "end_time": eval_end,
                            "status": "passed" if eval_code == 0 else f"failed:{eval_code}",
                        }
                    )
                    handle.flush()
                    if eval_code != 0:
                        raise SystemExit(f"{condition} seed {seed} {split} failed; see {eval_report}")
                if f"{condition}:{seed}" in args.trace:
                    trace_dir = REPORT_ROOT / f"traces-{condition}-seed-{seed}"
                    trace_report = REPORT_ROOT / f"{condition}-seed-{seed}-diagnostics.json"
                    trace_command = [
                        str(ROOT / "target" / "release" / "diagnostics"),
                        "--config",
                        str(config),
                        "--split",
                        "training",
                        "--trace-dir",
                        str(trace_dir),
                    ]
                    trace_code, trace_start, trace_end = run(trace_command, trace_report)
                    writer.writerow(
                        {
                            "condition": condition,
                            "seed": seed,
                            "split": "training-diagnostics",
                            "commit": commit,
                            "config": str(config.relative_to(ROOT)),
                            "checkpoint": str(checkpoint.relative_to(ROOT)),
                            "report_file": str(trace_report.relative_to(ROOT)),
                            "start_time": trace_start,
                            "end_time": trace_end,
                            "status": "passed" if trace_code == 0 else f"failed:{trace_code}",
                        }
                    )
                    handle.flush()
                    if trace_code != 0:
                        raise SystemExit(f"diagnostics failed; see {trace_report}")


if __name__ == "__main__":
    main()
