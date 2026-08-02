#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment  # noqa: E402


TEMPLATE = """[data]
path = "old.csv"
[splits.training]
start = "2020-01-01"
end = "2020-01-02"
[splits.validation]
start = "2020-01-03"
end = "2020-01-04"
[splits.test]
start = "2020-01-05"
end = "2020-01-06"
[simulation]
seed = 1
[checkpoint]
path = "old.bin"
every_steps = 0
"""


class RunnerTests(unittest.TestCase):
    def test_materialized_configuration_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.toml"
            template.write_text(TEMPLATE, encoding="utf-8")
            fold = {
                "training_start": "2021-01-01",
                "training_end": "2021-02-01",
                "validation_start": "2021-02-02",
                "validation_end": "2021-03-01",
                "test_start": "2021-03-02",
                "test_end": "2021-04-01",
            }
            first = run_experiment.materialize_config(
                template, root / "first.toml", "market.csv", fold, 42, "checkpoint.bin"
            )
            second = run_experiment.materialize_config(
                template, root / "second.toml", "market.csv", fold, 42, "checkpoint.bin"
            )
            self.assertEqual(first, second)
            self.assertIn(b"seed = 42", first)
            self.assertIn(b'path = "market.csv"', first)

    def test_completed_valid_task_is_skipped_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs/prompt9/conditions").mkdir(parents=True)
            (root / "configs/prompt9/conditions/neural.toml").write_text(TEMPLATE, encoding="utf-8")
            dataset = root / "market.csv"
            dataset.write_text("market\n", encoding="ascii")
            worker = root / "worker"
            worker.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args=sys.argv\n"
                "checkpoint=pathlib.Path(args[args.index('--checkpoint')+1])\n"
                "checkpoint.parent.mkdir(parents=True, exist_ok=True)\n"
                "checkpoint.write_bytes(b'checkpoint')\n"
                "count=pathlib.Path('worker-count')\n"
                "count.write_text(str(int(count.read_text())+1) if count.exists() else '1')\n"
                "print(json.dumps({'report': {}}))\n",
                encoding="utf-8",
            )
            worker.chmod(0o755)
            task = run_experiment.Task(
                stage="screening",
                market={
                    "market_id": "market-1",
                    "dataset": "market.csv",
                    "sha256": run_experiment.sha256(dataset),
                },
                condition={
                    "condition": "neural",
                    "kind": "neural",
                    "template": "configs/prompt9/conditions/neural.toml",
                },
                fold={
                    "fold": "fold-1",
                    "training_start": "2021-01-01",
                    "training_end": "2021-02-01",
                    "validation_start": "2021-02-02",
                    "validation_end": "2021-03-01",
                    "test_start": "2021-03-02",
                    "test_end": "2021-04-01",
                },
                seed=42,
                split="training",
                implementation_commit="abc123",
            )
            with patch.object(run_experiment, "ROOT", root):
                first = run_experiment.execute_task(task, worker, 10)
                second = run_experiment.execute_task(task, worker, 10)
                output = run_experiment.task_paths(task)["output"]
            self.assertEqual(first["status"], "passed")
            self.assertEqual(second["status"], "skipped")
            self.assertEqual((root / "worker-count").read_text(), "1")
            self.assertEqual(json.loads(output.read_text())["report"], {})

    def test_manifest_recovery_ignores_incomplete_rows_and_replaces_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "reports/prompt9/screening-run-manifest.csv"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("status,task\npassed,complete\npassed\n", encoding="utf-8")
            with patch.object(run_experiment, "ROOT", root):
                run_experiment.write_run_manifest(
                    "screening",
                    [{"task": "new", "status": "passed", "output": "new.json"}],
                )
            rows = run_experiment.read_manifest(manifest)
            self.assertEqual([row["task"] for row in rows], ["complete", "new"])
            self.assertFalse(manifest.with_suffix(".tmp").exists())


if __name__ == "__main__":
    unittest.main()
