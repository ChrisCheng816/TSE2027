from __future__ import annotations

import subprocess
import sys
import unittest
from unittest.mock import patch

from codegen_agent.evaluator import build_executable_code, evaluate_completion, evaluate_self_test_completion
from codegen_agent.types import CodeGenTask


class EvaluatorTests(unittest.TestCase):
    def test_evaluate_self_test_completion_records_per_test_summary_for_named_self_tests(self) -> None:
        task = CodeGenTask(
            dataset_name="toy",
            task_id="self-test-summary",
            prompt="Implement task_func.",
            prompt_source="prompt",
            code_prompt="def task_func(value):\n",
            entry_point="task_func",
        )

        result = evaluate_self_test_completion(
            task,
            "def task_func(value):\n    return value\n",
            (
                "def test_1(candidate):\n"
                "    assert candidate(1) == 1\n\n"
                "def test_2(candidate):\n"
                "    assert candidate(2) == 2\n\n"
                "def test_3(candidate):\n"
                "    assert candidate(3) == 4\n"
            ),
            timeout_sec=2,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(3, result["num_tests"])
        self.assertEqual(2, result["passed_tests"])
        self.assertEqual(1, result["failed_tests"])
        self.assertEqual(["test_1", "test_2", "test_3"], [test["name"] for test in result["tests"]])
        self.assertEqual([True, True, False], [test["passed"] for test in result["tests"]])
        self.assertEqual("AssertionError", result["tests"][2]["error_type"])

    def test_evaluate_self_test_completion_treats_legacy_check_as_single_test(self) -> None:
        task = CodeGenTask(
            dataset_name="toy",
            task_id="legacy-self-test-check",
            prompt="Implement task_func.",
            prompt_source="prompt",
            code_prompt="def task_func(value):\n",
            entry_point="task_func",
        )

        result = evaluate_self_test_completion(
            task,
            "def task_func(value):\n    return value\n",
            "def check(candidate):\n    assert candidate(1) == 2\n",
            timeout_sec=2,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(1, result["num_tests"])
        self.assertEqual(0, result["passed_tests"])
        self.assertEqual(1, result["failed_tests"])
        self.assertEqual("check", result["tests"][0]["name"])
        self.assertEqual("AssertionError", result["tests"][0]["error_type"])

    def test_evaluate_self_test_completion_uses_legacy_candidate_call_by_default(self) -> None:
        task = CodeGenTask(
            dataset_name="toy",
            task_id="zero-arg-named-self-test-default",
            prompt="Implement task_func.",
            prompt_source="prompt",
            code_prompt="def task_func(value):\n",
            entry_point="task_func",
        )

        result = evaluate_self_test_completion(
            task,
            "def task_func(value):\n    return value\n",
            (
                "def test_1():\n"
                "    assert task_func(1) == 1\n\n"
                "def test_2():\n"
                "    assert candidate(2) == 2\n"
            ),
            timeout_sec=2,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(2, result["num_tests"])
        self.assertEqual(0, result["passed_tests"])
        self.assertEqual("TypeError", result["tests"][0]["error_type"])

    def test_evaluate_self_test_completion_can_execute_zero_arg_named_self_tests_when_enabled(self) -> None:
        task = CodeGenTask(
            dataset_name="toy",
            task_id="zero-arg-named-self-test-enabled",
            prompt="Implement task_func.",
            prompt_source="prompt",
            code_prompt="def task_func(value):\n",
            entry_point="task_func",
        )

        result = evaluate_self_test_completion(
            task,
            "def task_func(value):\n    return value\n",
            (
                "def test_1():\n"
                "    assert task_func(1) == 1\n\n"
                "def test_2():\n"
                "    assert candidate(2) == 2\n"
            ),
            timeout_sec=2,
            signature_compatible_named_tests=True,
        )

        self.assertTrue(result["passed"], msg=result["stderr"])
        self.assertEqual(2, result["num_tests"])
        self.assertEqual(2, result["passed_tests"])

    def test_evaluate_self_test_completion_guards_process_kill_calls(self) -> None:
        task = CodeGenTask(
            dataset_name="bigcodebench-v0.1.4",
            task_id="process-kill-self-test",
            prompt="Implement task_func.",
            prompt_source="prompt",
            code_prompt="def task_func():\n",
            entry_point="task_func",
        )

        result = evaluate_self_test_completion(
            task,
            (
                "import os\n"
                "import signal\n"
                "import subprocess\n\n"
                "def task_func():\n"
                "    os.kill(os.getpid(), signal.SIGTERM)\n"
                "    try:\n"
                "        subprocess.run(['pgrep', ''], check=True, capture_output=True, text=True)\n"
                "    except subprocess.CalledProcessError:\n"
                "        return 'guarded'\n"
                "    return 'unguarded'\n"
            ),
            "def test_1(candidate):\n    assert candidate() == 'guarded'\n",
            timeout_sec=2,
        )

        self.assertTrue(result["passed"], msg=result["stderr"])
        self.assertEqual(1, result["num_tests"])
        self.assertEqual(1, result["passed_tests"])

    def test_evaluate_completion_does_not_guard_official_evaluation(self) -> None:
        task = CodeGenTask(
            dataset_name="bigcodebench-v0.1.4",
            task_id="official-eval-unguarded-kill",
            prompt="Implement task_func.",
            prompt_source="prompt",
            test="def check(candidate):\n    candidate()\n",
            entry_point="task_func",
        )

        result = evaluate_completion(
            task,
            (
                "import os\n"
                "import signal\n\n"
                "def task_func():\n"
                "    os.kill(os.getpid(), signal.SIGTERM)\n"
                "    return 'alive'\n"
            ),
            timeout_sec=2,
        )

        self.assertFalse(result["passed"])
        self.assertNotEqual(0, result["exit_code"])

    def test_evaluate_completion_tolerates_non_utf8_process_output(self) -> None:
        task = CodeGenTask(
            dataset_name="bigcodebench-v0.1.4",
            task_id="official-eval-binary-stdout",
            prompt="Implement task_func.",
            prompt_source="prompt",
            test="def check(candidate):\n    candidate()\n",
            entry_point="task_func",
        )

        result = evaluate_completion(
            task,
            (
                "import sys\n\n"
                "def task_func():\n"
                "    sys.stdout.buffer.write(bytes([0x1f, 0x8b]))\n"
            ),
            timeout_sec=2,
        )

        self.assertTrue(result["passed"], msg=result["stderr"])
        self.assertIn("\ufffd", result["stdout"])

    def test_build_executable_code_inserts_body_only_completion_into_function_stub(self) -> None:
        task = CodeGenTask(
            dataset_name="toy",
            task_id="body-only",
            prompt="Implement task_func.",
            prompt_source="prompt",
            code_prompt="import math\n\ndef task_func(value):\n",
            entry_point="task_func",
        )

        executable = build_executable_code(
            task,
            "    squared = value * value\n    return math.floor(squared)\n",
        )

        self.assertIn("import math", executable)
        self.assertIn("def task_func(value):", executable)
        self.assertIn("    squared = value * value", executable)
        self.assertIn("    return math.floor(squared)", executable)

    def test_build_executable_code_preserves_starter_imports_and_helpers(self) -> None:
        task = CodeGenTask(
            dataset_name="toy",
            task_id="helper-imports",
            prompt="Implement task_func.",
            prompt_source="prompt",
            code_prompt=(
                "import math\n\n"
                "def helper(value):\n"
                "    return value + 1\n\n"
                "def task_func(value):\n"
                "    pass\n"
            ),
            entry_point="task_func",
        )

        executable = build_executable_code(
            task,
            "def task_func(value):\n    return math.floor(helper(value))\n",
        )

        self.assertIn("import math", executable)
        self.assertIn("def helper(value):", executable)
        self.assertIn("def task_func(value):", executable)
        self.assertEqual(1, executable.count("def task_func(value):"))

    def test_build_executable_code_drops_incomplete_stub_when_completion_contains_full_function(self) -> None:
        task = CodeGenTask(
            dataset_name="toy",
            task_id="full-function-with-stub",
            prompt="Implement task_func.",
            prompt_source="prompt",
            code_prompt="import math\n\ndef task_func(value):\n",
            entry_point="task_func",
        )

        executable = build_executable_code(
            task,
            "def task_func(value):\n    return math.floor(value)\n",
        )

        self.assertIn("import math", executable)
        self.assertEqual(1, executable.count("def task_func(value):"))
        self.assertNotIn("def task_func(value):\n\ndef task_func(value):", executable)

    def test_build_executable_code_deduplicates_support_imports_when_completion_has_full_function(self) -> None:
        task = CodeGenTask(
            dataset_name="toy",
            task_id="dedupe-full-function-imports",
            prompt="Implement task_func.",
            prompt_source="prompt",
            code_prompt=(
                "import math\n"
                "from collections import Counter\n\n"
                "def helper(value):\n"
                "    return value + 1\n\n"
                "def task_func(value):\n"
            ),
            entry_point="task_func",
        )

        executable = build_executable_code(
            task,
            "import math\n\n"
            "def task_func(value):\n"
            "    return helper(value)\n",
        )

        self.assertEqual(1, executable.count("import math"))
        self.assertEqual(1, executable.count("from collections import Counter"))
        self.assertEqual(1, executable.count("def helper(value):"))
        self.assertEqual(1, executable.count("def task_func(value):"))

    def test_build_executable_code_extracts_leading_imports_from_body_only_completion(self) -> None:
        task = CodeGenTask(
            dataset_name="toy",
            task_id="body-with-imports",
            prompt="Implement task_func.",
            prompt_source="prompt",
            code_prompt="import math\n\ndef task_func(value):\n",
            entry_point="task_func",
        )

        executable = build_executable_code(
            task,
            "import math\n"
            "from collections import Counter\n\n"
            "    counts = Counter([value])\n"
            "    return math.floor(counts[value])\n",
        )

        self.assertEqual(1, executable.count("import math"))
        self.assertEqual(1, executable.count("from collections import Counter"))
        self.assertEqual(1, executable.count("def task_func(value):"))
        self.assertIn("    counts = Counter([value])", executable)
        self.assertIn("    return math.floor(counts[value])", executable)




    def test_evaluate_completion_uses_current_python_interpreter(self) -> None:
        task = CodeGenTask(
            dataset_name="toy",
            task_id="python-exec",
            prompt="Implement task_func.",
            prompt_source="prompt",
            test="def check(candidate):\n    assert candidate(1) == 1\n",
            entry_point="task_func",
        )

        with patch("codegen_agent.evaluator.subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(
                args=[sys.executable, "run_eval.py"],
                returncode=0,
                stdout="",
                stderr="",
            )
            evaluate_completion(
                task,
                "def task_func(value):\n    return value\n",
                timeout_sec=1,
            )

        self.assertEqual(sys.executable, run_mock.call_args.args[0][0])




    def test_evaluate_self_test_completion_executes_generated_callable_tests(self) -> None:
        task = CodeGenTask(
            dataset_name="toy",
            task_id="self-tests-callable",
            prompt="Implement task_func.",
            prompt_source="prompt",
            test="def check(candidate):\n    assert candidate(1) == 1\n",
            entry_point="task_func",
        )

        result = evaluate_self_test_completion(
            task,
            "def task_func(value):\n    return value\n",
            "def check(candidate):\n    assert candidate(1) == 1\n    assert candidate(3) == 3\n",
            timeout_sec=2,
        )

        self.assertTrue(result["passed"], msg=result["stderr"])



if __name__ == "__main__":
    unittest.main()
