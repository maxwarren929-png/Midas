#!/usr/bin/env python3
"""Run deterministic, resumable walk-forward experiment matrices."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKER = ROOT / "target" / "release" / "prompt9-worker"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def replace_toml_value(text: str, section: str, key: str, value: str) -> str:
    lines = text.splitlines()
    current = ""
    found = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
        elif current == section and stripped.startswith(f"{key} ="):
            lines[index] = f"{key} = {value}"
            found = True
            break
    if not found:
        raise ValueError(f"missing [{section}] {key} in configuration template")
    return "\n".join(lines) + "\n"


def quote_toml(value: str) -> str:
    return json.dumps(value)


def materialize_config(
    template: Path,
    target: Path,
    dataset: str,
    fold: dict[str, str],
    seed: int,
    checkpoint: str,
) -> bytes:
    text = template.read_text(encoding="utf-8")
    replacements = [
        ("data", "path", quote_toml(dataset)),
        ("splits.training", "start", quote_toml(fold["training_start"])),
        ("splits.training", "end", quote_toml(fold["training_end"])),
        ("splits.validation", "start", quote_toml(fold["validation_start"])),
        ("splits.validation", "end", quote_toml(fold["validation_end"])),
        ("splits.test", "start", quote_toml(fold["test_start"])),
        ("splits.test", "end", quote_toml(fold["test_end"])),
        ("simulation", "seed", str(seed)),
        ("checkpoint", "path", quote_toml(checkpoint)),
    ]
    for section, key, value in replacements:
        text = replace_toml_value(text, section, key, value)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return text.encode("utf-8")


@dataclass(frozen=True)
class Task:
    stage: str
    market: dict[str, str]
    condition: dict[str, str]
    fold: dict[str, str]
    seed: int
    split: str
    implementation_commit: str

    @property
    def key(self) -> str:
        return "/".join(
            (
                self.stage,
                self.market["market_id"],
                self.fold["fold"],
                self.condition["condition"],
                str(self.seed),
                self.split,
            )
        )


def task_paths(task: Task) -> dict[str, Path]:
    base = Path(task.stage) / task.market["market_id"] / task.fold["fold"] / task.condition["condition"]
    stem = f"seed-{task.seed}"
    return {
        "config": ROOT / "configs" / "prompt9" / "runs" / base / f"{stem}.toml",
        "checkpoint": ROOT / "checkpoints" / "prompt9" / base / f"{stem}.bin",
        "output": ROOT / "reports" / "prompt9" / "raw" / base / f"{stem}-{task.split}.json",
        "stderr": ROOT / "reports" / "prompt9" / "logs" / base / f"{stem}-{task.split}.log",
        "provenance": ROOT / "reports" / "prompt9" / "raw" / base / f"{stem}-{task.split}.provenance.json",
    }


def provenance_digest(task: Task, config_bytes: bytes, checkpoint_hash: str | None) -> str:
    value = {
        "task": task.key,
        "implementation_commit": task.implementation_commit,
        "dataset_sha256": task.market["sha256"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "checkpoint_sha256": checkpoint_hash,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("ascii")).hexdigest()


def valid_completed_output(output: Path, provenance: Path, digest: str) -> bool:
    try:
        value = json.loads(output.read_text(encoding="utf-8"))
        saved = json.loads(provenance.read_text(encoding="utf-8"))
        return isinstance(value.get("report"), dict) and saved.get("input_digest") == digest
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def execute_task(task: Task, worker: Path, timeout: int) -> dict[str, Any]:
    paths = task_paths(task)
    for name in ("output", "stderr", "provenance", "checkpoint"):
        paths[name].parent.mkdir(parents=True, exist_ok=True)
    template = ROOT / task.condition["template"]
    checkpoint_relative = paths["checkpoint"].relative_to(ROOT).as_posix()
    config_bytes = materialize_config(
        template,
        paths["config"],
        task.market["dataset"],
        task.fold,
        task.seed,
        checkpoint_relative,
    )
    is_neural = task.condition["kind"] == "neural"
    checkpoint_hash = sha256(paths["checkpoint"]) if is_neural and paths["checkpoint"].exists() else None
    digest = provenance_digest(task, config_bytes, checkpoint_hash)
    if valid_completed_output(paths["output"], paths["provenance"], digest):
        return {"task": task.key, "status": "skipped", "output": paths["output"].relative_to(ROOT).as_posix()}

    started = dt.datetime.now(dt.timezone.utc).isoformat()
    if is_neural and task.split != "training" and not paths["checkpoint"].exists():
        return {
            "task": task.key,
            "status": "failed:missing_checkpoint",
            "started_at": started,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    mode = "train" if is_neural and task.split == "training" else "evaluate" if is_neural else "simulate"
    command = [
        str(worker),
        "--config",
        str(paths["config"]),
        "--mode",
        mode,
        "--split",
        task.split,
    ]
    if is_neural:
        command.extend(("--checkpoint", str(paths["checkpoint"])))
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "RUST_LOG": "warn"},
        )
        paths["stderr"].write_text(result.stderr, encoding="utf-8")
        status = "passed" if result.returncode == 0 else f"failed:exit_{result.returncode}"
        if result.returncode == 0:
            json.loads(result.stdout)
            paths["output"].write_text(result.stdout, encoding="utf-8")
            checkpoint_hash = sha256(paths["checkpoint"]) if is_neural else None
            digest = provenance_digest(task, config_bytes, checkpoint_hash)
            paths["provenance"].write_text(
                json.dumps(
                    {
                        "input_digest": digest,
                        "implementation_commit": task.implementation_commit,
                        "checkpoint_sha256": checkpoint_hash,
                        "command": [str(item) for item in command],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    except subprocess.TimeoutExpired as error:
        status = "failed:timeout"
        paths["stderr"].write_text(str(error), encoding="utf-8")
    except (OSError, json.JSONDecodeError) as error:
        status = "failed:invalid_output"
        paths["stderr"].write_text(str(error), encoding="utf-8")
    return {
        "task": task.key,
        "status": status,
        "started_at": started,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": paths["config"].relative_to(ROOT).as_posix(),
        "checkpoint": checkpoint_relative if is_neural else "",
        "output": paths["output"].relative_to(ROOT).as_posix(),
        "stderr": paths["stderr"].relative_to(ROOT).as_posix(),
    }


def write_run_manifest(stage: str, rows: list[dict[str, Any]]) -> None:
    path = ROOT / "reports" / "prompt9" / f"{stage}-run-manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        {row["task"]: row for row in read_manifest(path) if row.get("task")}
        if path.exists()
        else {}
    )
    existing.update({row["task"]: {key: str(value) for key, value in row.items()} for row in rows})
    fieldnames = sorted({key for row in existing.values() for key in row})
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing[key] for key in sorted(existing))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-manifest", type=Path, required=True)
    parser.add_argument("--condition-manifest", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--splits", choices=("validation", "test"), nargs="+", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--worker", type=Path, default=DEFAULT_WORKER)
    parser.add_argument("--holdout", choices=("include", "exclude", "only"), default="exclude")
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout < 1:
        raise SystemExit("jobs and timeout must be positive")
    markets = read_manifest(args.market_manifest)
    if args.holdout == "exclude":
        markets = [market for market in markets if market["holdout"] == "false"]
    elif args.holdout == "only":
        markets = [market for market in markets if market["holdout"] == "true"]
    conditions = read_manifest(args.condition_manifest)
    folds = read_manifest(args.fold_manifest)
    tasks: list[Task] = []
    for market in markets:
        for condition in conditions:
            for fold in folds:
                for seed in args.seeds:
                    if condition["kind"] == "neural":
                        tasks.append(Task(args.stage, market, condition, fold, seed, "training", args.implementation_commit))
                    for split in args.splits:
                        tasks.append(Task(args.stage, market, condition, fold, seed, split, args.implementation_commit))
    # Training must finish before its evaluations. Running phase groups retains
    # parallelism across independent market/condition/fold/seed combinations.
    results: list[dict[str, Any]] = []
    for split in ("training", *args.splits):
        phase = [task for task in tasks if task.split == split]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(execute_task, task, args.worker, args.timeout) for task in phase]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
        write_run_manifest(args.stage, results)
    failures = [row for row in results if not row["status"] in ("passed", "skipped")]
    if failures:
        raise SystemExit(f"{len(failures)} experiment tasks failed")


if __name__ == "__main__":
    main()
