from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


def _load_compare_module():
    module_path = Path(__file__).resolve().parents[1] / "eval" / "compare_experiment_results.py"
    spec = importlib.util.spec_from_file_location("compare_experiment_results", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CompareExperimentResultsTests(unittest.TestCase):
    def test_load_model_stats_includes_bara_records(self) -> None:
        module = _load_compare_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "toy-model"
            model_dir.mkdir(parents=True)
            reward_path = model_dir / "toy-model.reward_table.jsonl"
            bara_path = model_dir / "toy-model_bara_A0-A1-A2.jsonl"

            reward_rows = [
                {"task_id": "t1", "arm_id": "CD", "pass_at_k": 1.0, "completion_tokens": 10},
                {"task_id": "t1", "arm_id": "CF", "pass_at_k": 0.0, "completion_tokens": 20},
                {"task_id": "t1", "arm_id": "PH", "pass_at_k": 0.0, "completion_tokens": 30},
                {"task_id": "t2", "arm_id": "CD", "pass_at_k": 0.0, "completion_tokens": 11},
                {"task_id": "t2", "arm_id": "CF", "pass_at_k": 1.0, "completion_tokens": 21},
                {"task_id": "t2", "arm_id": "PH", "pass_at_k": 0.0, "completion_tokens": 31},
            ]
            bara_rows = [
                {"task_id": "t1", "arm_id": "CD", "pass_at_k": 1.0, "completion_tokens": 10},
                {"task_id": "t2", "arm_id": "CF", "pass_at_k": 1.0, "completion_tokens": 21},
            ]

            reward_path.write_text(
                "".join(json.dumps(row) + "\n" for row in reward_rows),
                encoding="utf-8",
            )
            bara_path.write_text(
                "".join(json.dumps(row) + "\n" for row in bara_rows),
                encoding="utf-8",
            )

            stats = module.load_model_stats(reward_path, "bigcodebench-v0.1.4")

        self.assertIsNotNone(stats.bara_stats)
        self.assertEqual(2, stats.bara_stats.num_tasks)
        self.assertEqual(2, stats.bara_stats.passed_count)
        self.assertEqual({"CD": 1, "CF": 1}, stats.bara_stats.arm_pull_counts)

    def test_print_model_summary_renders_bara_section(self) -> None:
        module = _load_compare_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "toy-model"
            model_dir.mkdir(parents=True)
            reward_path = model_dir / "toy-model.reward_table.jsonl"
            bara_path = model_dir / "toy-model_bara_A0-A1-A2.jsonl"

            reward_path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in [
                        {"task_id": "t1", "arm_id": "CD", "pass_at_k": 1.0, "completion_tokens": 10},
                        {"task_id": "t1", "arm_id": "CF", "pass_at_k": 0.0, "completion_tokens": 20},
                        {"task_id": "t2", "arm_id": "CD", "pass_at_k": 0.0, "completion_tokens": 11},
                        {"task_id": "t2", "arm_id": "CF", "pass_at_k": 1.0, "completion_tokens": 21},
                    ]
                ),
                encoding="utf-8",
            )
            bara_path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in [
                        {"task_id": "t1", "arm_id": "CD", "pass_at_k": 1.0, "completion_tokens": 10},
                        {"task_id": "t2", "arm_id": "CF", "pass_at_k": 1.0, "completion_tokens": 21},
                    ]
                ),
                encoding="utf-8",
            )

            stats = module.load_model_stats(reward_path, "bigcodebench-v0.1.4")

        output = io.StringIO()
        with redirect_stdout(output):
            module.print_model_summary(stats, "union_desc")

        rendered = output.getvalue()
        self.assertIn("[BARA]", rendered)
        self.assertIn("bara", rendered)
        self.assertIn("bara_vs_best_single", rendered)


if __name__ == "__main__":
    unittest.main()
