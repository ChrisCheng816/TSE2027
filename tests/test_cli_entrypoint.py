from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBPROCESS_CWD = REPO_ROOT / "codegen_agent"
SUBPROCESS_BOOTSTRAP = "import os, sys; sys.path.insert(0, os.environ['CODEGEN_AGENT_REPO_ROOT']); "
DATASET_STUB = """
import sys, types
stub = types.ModuleType('datasets')
stub.Dataset = type('Dataset', (), {})
stub.IterableDataset = type('IterableDataset', (), {})
stub.concatenate_datasets = lambda *args, **kwargs: None
stub.load_dataset = lambda *args, **kwargs: None
sys.modules['datasets'] = stub
"""


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["CODEGEN_AGENT_REPO_ROOT"] = str(REPO_ROOT)
    return env


class CliEntrypointTests(unittest.TestCase):
    def test_eval_flag_is_not_supported(self) -> None:
        env = _subprocess_env()
        snippet = SUBPROCESS_BOOTSTRAP + DATASET_STUB + """
from codegen_agent import cli
parser = cli.build_parser()
try:
    parser.parse_args(['--run-mode', 'all_arms', '--' + 'eval'])
except SystemExit as exc:
    assert exc.code != 0
else:
    raise AssertionError('expected removed eval flag to be rejected')
"""
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=SUBPROCESS_CWD,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, msg=result.stderr)

    def test_repeated_sampling_config_defaults(self) -> None:
        env = _subprocess_env()
        snippet = SUBPROCESS_BOOTSTRAP + DATASET_STUB + """
from codegen_agent import cli
parser = cli.build_parser()
gpu_config = cli._resolve_config(parser.parse_args(['--cuda-visible-devices', '2,3']))
repeated = cli._resolve_config(parser.parse_args(['--run-mode', 'repeated_sampling', '--arm-id', 'A2', '--budget', '20000']))
opus_repeated = cli._resolve_config(parser.parse_args(['--run-mode', 'repeated_sampling', '--arm-id', 'A2', '--budget', '20000', '--model-name', 'claude-opus-4.8']))
repeated_t = cli._resolve_config(parser.parse_args(['--run-mode', 'repeated_sampling', '--arm-id', 'A2', '--budget', '20000', '--t', '0.7']))
multi_model_args = parser.parse_args(['--run-mode', 'repeated_sampling', '--arm-id', 'A2', '--budget', '20000', '--model-name', 'qwen2.5-32b', 'qwen3-30b'])
adaptive = cli._resolve_config(parser.parse_args(['--run-mode', 'adaptive']))
assert gpu_config.vllm.cuda_visible_devices == '2,3'
assert repeated.run_mode == 'repeated_sampling'
assert repeated.repeated_sampling_arm_id == 'A2'
assert repeated.arm_ids == ('A2',)
assert repeated.adaptive.order_run is True
assert repeated.adaptive.bigcodebench_per_task_completion_token_budget == 20000
assert cli._requested_model_names(multi_model_args) == ['qwen2.5-32b', 'qwen3-30b']
assert cli._resolve_config(multi_model_args, model_name='qwen3-30b').model.model_name == 'qwen3-30b'
assert opus_repeated.model.model_name == 'claude-opus-4.8'
assert repeated.generation.temperature == 1
assert repeated.generation.top_p == 0.95
assert repeated_t.generation.temperature == 0.7
assert repeated_t.generation.top_p == 0.95
assert adaptive.run_mode == 'adaptive'
"""
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=SUBPROCESS_CWD,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, msg=result.stderr)

    def test_t_flag_is_rejected_outside_repeated_sampling(self) -> None:
        env = _subprocess_env()
        snippet = SUBPROCESS_BOOTSTRAP + DATASET_STUB + """
from codegen_agent import cli
parser = cli.build_parser()
try:
    cli._resolve_config(parser.parse_args(['--run-mode', 'all_arms', '--t', '0.7']))
except ValueError as exc:
    assert '--t is only supported' in str(exc)
else:
    raise AssertionError('expected --t to be rejected outside repeated_sampling')
"""
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=SUBPROCESS_CWD,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, msg=result.stderr)

    def test_repeated_sampling_config_path_and_guardrails(self) -> None:
        env = _subprocess_env()
        with tempfile.TemporaryDirectory() as tmpdir:
            snippet = SUBPROCESS_BOOTSTRAP + DATASET_STUB + """
from pathlib import Path
from codegen_agent import cli
from codegen_agent.config import PathConfig
parser = cli.build_parser()
config = cli._resolve_config(parser.parse_args(['--run-mode', 'repeated_sampling', '--arm-id', 'A2', '--budget', '20000']))
assert config.arm_ids == ('A2',)
assert config.repeated_sampling_arm_id == 'A2'
assert config.adaptive.order_run is True
paths = PathConfig(repeated_sampling_results_dir=Path(sys.argv[1]))
output_path = paths.repeated_sampling_output_path('qwen2.5-32b', 'A2', 'bigcodebench-v0.1.4', budget_label='budget-20000')
assert output_path.name == 'qwen2.5-32b_A2_budget-20000.jsonl'
assert output_path.parent == Path(sys.argv[1]) / 'pass_only' / 'qwen2.5-32b'
order_cfg = cli._resolve_config(parser.parse_args(['--run-mode', 'adaptive', '--order_run', '--budget', '20000']))
random_cfg = cli._resolve_config(parser.parse_args(['--run-mode', 'adaptive', '--order_run', '--random_run', '--budget', '20000']))
fix_cfg = cli._resolve_config(parser.parse_args(['--run-mode', 'adaptive', '--order_run', '--fix_run', '--budget', '20000']))
allocator_cfg = cli._resolve_config(parser.parse_args(['--run-mode', 'adaptive', '--order_run', '--budget', '20000', '--budget-allocator', 'equal_split']))
assert cli._online_order_run_output_path(config=order_cfg, budget_label='budget-20000').name == 'qwen2.5-32b_order_budget-20000.jsonl'
assert cli._online_order_run_output_path(config=random_cfg, budget_label='budget-20000').name == 'qwen2.5-32b_random_budget-20000.jsonl'
assert cli._online_order_run_output_path(config=fix_cfg, budget_label='budget-20000').name == 'qwen2.5-32b_fix_budget-20000.jsonl'
assert cli._online_order_run_output_path(config=allocator_cfg, budget_label='budget-20000').name == 'qwen2.5-32b_allocator-equal_split_order_budget-20000.jsonl'
for argv, expected in [
    (['--run-mode', 'repeated_sampling'], '--arm-id'),
    (['--run-mode', 'repeated_sampling', '--arm-id', 'A2'], '--budget'),
    (['--run-mode', 'adaptive', '--arm-id', 'A2'], '--arm-id is only supported'),
    (['--run-mode', 'adaptive', '--arm-ids', 'A0'], '--arm-ids is only supported'),
]:
    try:
        cli._resolve_config(parser.parse_args(argv))
    except ValueError as exc:
        assert expected in str(exc)
    else:
        raise AssertionError(f'expected ValueError for {argv}')
"""
            result = subprocess.run(
                [sys.executable, "-c", snippet, tmpdir],
                cwd=SUBPROCESS_CWD,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(0, result.returncode, msg=result.stderr)

    def test_python_m_codegene_agent_cli_invokes_main(self) -> None:
        env = _subprocess_env()
        snippet = SUBPROCESS_BOOTSTRAP + DATASET_STUB + """
import runpy, sys
sys.argv = ['codegen_agent.cli', '--help']
runpy.run_module('codegen_agent.cli', run_name='__main__')
"""
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=SUBPROCESS_CWD,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode)
        self.assertIn("usage:", result.stdout.lower())
        self.assertIn("--run-mode", result.stdout)

    def test_repo_root_can_resolve_codegen_agent_module_without_parent_pythonpath(self) -> None:
        env = _subprocess_env()
        snippet = SUBPROCESS_BOOTSTRAP + DATASET_STUB + """
import runpy, sys
sys.argv = ['codegen_agent.cli', '--help']
runpy.run_module('codegen_agent.cli', run_name='__main__')
"""
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=SUBPROCESS_CWD,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode)
        self.assertIn("usage:", result.stdout.lower())
        self.assertIn("--run-mode", result.stdout)


if __name__ == "__main__":
    unittest.main()
