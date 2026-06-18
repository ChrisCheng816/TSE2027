from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import CodeGenTask

_SELF_TEST_SUMMARY_PREFIX = "__SELF_TEST_SUMMARY__="


@dataclass(frozen=True)
class _TopLevelSegment:
    kind: str
    key: str | None
    text: str


def build_executable_code(task: CodeGenTask, cleaned_output: str) -> str:
    code_prefix = (task.code_prompt or "").rstrip()
    raw_output = cleaned_output.strip("\n")
    code_body = raw_output.strip()
    entry_point = (task.entry_point or "").strip()

    if entry_point and (
        f"def {entry_point}" in code_body
        or f"class {entry_point}" in code_body
    ):
        support_code = _extract_support_code(code_prefix, entry_point)
        return _merge_module_text(support_code=support_code, generated_module=code_body)

    if _should_insert_body_into_stub(code_prefix=code_prefix, entry_point=entry_point, raw_output=raw_output):
        support_code = _extract_support_code(code_prefix, entry_point)
        entry_stub = _extract_entry_stub(code_prefix, entry_point)
        leading_module, body_only = _split_leading_module_preamble(raw_output)
        merged_support = _merge_module_text(support_code=support_code, generated_module=leading_module).strip()
        parts = [part for part in (merged_support, entry_stub.strip(), _indent_body_for_entry_stub(entry_stub, body_only)) if part]
        return "\n\n".join(parts).rstrip() + "\n"

    if code_prefix and code_body:
        return code_prefix + "\n" + code_body + "\n"
    if code_prefix:
        return code_prefix + "\n"
    return code_body + ("\n" if code_body else "")


def evaluate_completion(
    task: CodeGenTask,
    cleaned_output: str,
    *,
    timeout_sec: int,
) -> dict[str, Any]:
    if task.task_kind == "stdin_stdout":
        return _evaluate_stdio_completion(task, cleaned_output, timeout_sec=timeout_sec)

    executable_code = build_executable_code(task, cleaned_output)
    harness_code = _build_harness(task, executable_code)
    return _run_python_harness(
        cleaned_output=cleaned_output,
        executable_code=executable_code,
        harness_code=harness_code,
        timeout_sec=timeout_sec,
    )


def evaluate_self_test_completion(
    task: CodeGenTask,
    cleaned_output: str,
    self_test_code: str,
    *,
    timeout_sec: int,
    signature_compatible_named_tests: bool = False,
) -> dict[str, Any]:
    executable_code = build_executable_code(task, cleaned_output)
    harness_code = _build_self_test_harness(
        task,
        executable_code,
        self_test_code,
        signature_compatible_named_tests=signature_compatible_named_tests,
    )
    result = _run_python_harness(
        cleaned_output=cleaned_output,
        executable_code=executable_code,
        harness_code=harness_code,
        timeout_sec=timeout_sec,
    )
    summary, stdout = _extract_self_test_summary(result.get("stdout") or "")
    result["stdout"] = stdout
    if summary is not None:
        result["num_tests"] = int(summary.get("num_tests") or 0)
        result["passed_tests"] = int(summary.get("passed_tests") or 0)
        result["failed_tests"] = int(summary.get("failed_tests") or 0)
        result["tests"] = list(summary.get("tests") or [])
    return result


def estimate_pass_at_k(num_samples: int, num_correct: int, pass_k: int) -> float:
    if num_samples <= 0:
        return 0.0
    pass_k = max(1, min(pass_k, num_samples))
    if num_samples - num_correct < pass_k:
        return 1.0
    numerator = math.comb(num_samples - num_correct, pass_k)
    denominator = math.comb(num_samples, pass_k)
    return 1.0 - (numerator / denominator)


def _build_harness(task: CodeGenTask, executable_code: str) -> str:
    entry_point = task.entry_point or ""
    test_code = task.test or ""
    return "\n".join(
        [
            _compatibility_prelude().rstrip(),
            "",
            executable_code.rstrip(),
            "",
            _build_alias_block(entry_point),
            "",
            test_code.rstrip(),
            "",
            _runner_block(),
            "",
        ]
    )


def _build_self_test_harness(
    task: CodeGenTask,
    executable_code: str,
    self_test_code: str,
    *,
    signature_compatible_named_tests: bool = False,
) -> str:
    entry_point = task.entry_point or ""
    task_kind = task.task_kind or ""
    leading_sections = [_compatibility_prelude().rstrip(), ""]
    if task_kind != "stdin_stdout":
        leading_sections.extend(
            [
                executable_code.rstrip(),
                "",
                _build_alias_block(entry_point),
                "",
            ]
        )
    return "\n".join(
        leading_sections
        + [
            _build_run_io_helper(executable_code),
            "",
            self_test_code.rstrip(),
            "",
            _self_test_runner_block(
                task_kind,
                signature_compatible_named_tests=signature_compatible_named_tests,
            ),
            "",
        ]
    )


def _evaluate_stdio_completion(
    task: CodeGenTask,
    cleaned_output: str,
    *,
    timeout_sec: int,
) -> dict[str, Any]:
    executable_code = build_executable_code(task, cleaned_output)
    test_spec = task.test_spec or {}
    raw_inputs = test_spec.get("inputs") or []
    expected_outputs = test_spec.get("outputs") or []
    start = time.perf_counter()
    failure_stderr = ""
    try:
        with tempfile.TemporaryDirectory(prefix="bigcodebench_eval_") as tmp_dir:
            script_path = Path(tmp_dir) / "run_eval.py"
            script_path.write_text(executable_code, encoding="utf-8")
            for raw_stdin, expected in zip(raw_inputs, expected_outputs):
                completed = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=tmp_dir,
                    input=str(raw_stdin),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_sec,
                )
                if completed.returncode != 0:
                    duration_sec = time.perf_counter() - start
                    return {
                        "cleaned_output": cleaned_output,
                        "executable_code": executable_code,
                        "passed": False,
                        "exit_code": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                        "duration_sec": duration_sec,
                    }
                actual = _normalize_stdio_text(completed.stdout)
                expected_text = _normalize_stdio_text(str(expected))
                if not _stdio_outputs_match(expected_text, actual):
                    failure_stderr = f"AssertionError: expected {expected_text!r}, got {actual!r}"
                    duration_sec = time.perf_counter() - start
                    return {
                        "cleaned_output": cleaned_output,
                        "executable_code": executable_code,
                        "passed": False,
                        "exit_code": 1,
                        "stdout": completed.stdout,
                        "stderr": failure_stderr,
                        "duration_sec": duration_sec,
                    }
        duration_sec = time.perf_counter() - start
        return {
            "cleaned_output": cleaned_output,
            "executable_code": executable_code,
            "passed": True,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "duration_sec": duration_sec,
        }
    except subprocess.TimeoutExpired as exc:
        duration_sec = time.perf_counter() - start
        stdout = _normalize_subprocess_stream(exc.stdout)
        stderr = _normalize_subprocess_stream(exc.stderr)
        return {
            "cleaned_output": cleaned_output,
            "executable_code": executable_code,
            "passed": False,
            "exit_code": -1,
            "stdout": stdout,
            "stderr": stderr + f"\nTimeoutExpired: evaluation exceeded {timeout_sec} seconds.",
            "duration_sec": duration_sec,
        }


def _compatibility_prelude() -> str:
    return textwrap.dedent(
        """
        from typing import Any, DefaultDict, Deque, Dict, List, Optional, Set, Tuple
        """
    ).strip()


def _normalize_stdio_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _stdio_outputs_match(expected_text: str, actual_text: str) -> bool:
    if expected_text == actual_text:
        return True

    expected_tokens = expected_text.split()
    actual_tokens = actual_text.split()
    if len(expected_tokens) != len(actual_tokens) or not expected_tokens:
        return False

    expected_numbers = [_parse_numeric_token(token) for token in expected_tokens]
    actual_numbers = [_parse_numeric_token(token) for token in actual_tokens]
    if any(value is None for value in expected_numbers) or any(value is None for value in actual_numbers):
        return False

    return all(
        abs(expected - actual) <= 1e-6 * max(1.0, abs(expected))
        for expected, actual in zip(expected_numbers, actual_numbers)
    )


def _parse_numeric_token(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def _build_alias_block(entry_point: str) -> str:
    escaped_entry = repr(entry_point)
    return textwrap.dedent(
        f"""
        _ENTRY_POINT = {escaped_entry}
        if _ENTRY_POINT:
            _candidate = globals().get(_ENTRY_POINT)
            if _candidate is None:
                _solution_cls = globals().get("Solution")
                if isinstance(_solution_cls, type) and hasattr(_solution_cls, _ENTRY_POINT):
                    try:
                        _candidate = getattr(_solution_cls(), _ENTRY_POINT)
                    except TypeError:
                        _candidate = None
            if _candidate is None:
                for _value in list(globals().values()):
                    if not isinstance(_value, type) or not hasattr(_value, _ENTRY_POINT):
                        continue
                    try:
                        _candidate = getattr(_value(), _ENTRY_POINT)
                    except TypeError:
                        _candidate = None
                    if _candidate is not None:
                        break
            if _candidate is not None:
                globals().setdefault("task_func", _candidate)
                globals().setdefault("candidate", _candidate)
        """
    ).strip()


def _extract_support_code(code_prefix: str, entry_point: str) -> str:
    if not code_prefix.strip():
        return ""
    try:
        tree = ast.parse(code_prefix)
    except SyntaxError:
        return _support_code_before_incomplete_entry_stub(code_prefix, entry_point)

    segments: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == entry_point:
            continue
        segment = ast.get_source_segment(code_prefix, node)
        if segment is None and getattr(node, "lineno", None) is not None and getattr(node, "end_lineno", None) is not None:
            lines = code_prefix.splitlines()
            segment = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        if segment:
            segments.append(segment.strip())
    return "\n\n".join(segment for segment in segments if segment)


def _support_code_before_incomplete_entry_stub(code_prefix: str, entry_point: str) -> str:
    if not code_prefix.strip():
        return ""
    if not entry_point:
        return code_prefix.strip()

    lines = code_prefix.splitlines()
    enclosing_class_index: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if line.lstrip() != line:
            if (
                enclosing_class_index is not None
                and (
                    stripped.startswith(f"def {entry_point}(")
                    or stripped.startswith(f"async def {entry_point}(")
                )
            ):
                return "\n".join(lines[:enclosing_class_index]).strip()
            continue
        if stripped.startswith("class "):
            enclosing_class_index = index
        elif stripped:
            enclosing_class_index = None
        if (
            stripped.startswith(f"def {entry_point}(")
            or stripped.startswith(f"async def {entry_point}(")
            or stripped.startswith(f"class {entry_point}(")
            or stripped == f"class {entry_point}:"
        ):
            return "\n".join(lines[:index]).strip()
    return code_prefix.strip()


def _extract_entry_stub(code_prefix: str, entry_point: str) -> str:
    if not code_prefix.strip():
        return ""
    if not entry_point:
        return code_prefix.strip()
    try:
        tree = ast.parse(code_prefix)
    except SyntaxError:
        return _incomplete_entry_stub(code_prefix, entry_point)

    lines = code_prefix.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == entry_point:
            start_index = max(node.lineno - 1, 0)
            if getattr(node, "body", None):
                end_index = max(node.body[0].lineno - 1, start_index + 1)
            else:
                end_index = max(getattr(node, "end_lineno", node.lineno), start_index + 1)
            return "\n".join(lines[start_index:end_index]).rstrip()
    return code_prefix.strip()


def _incomplete_entry_stub(code_prefix: str, entry_point: str) -> str:
    lines = code_prefix.splitlines()
    enclosing_class_index: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if line.lstrip() == line:
            if stripped.startswith("class "):
                enclosing_class_index = index
            elif stripped:
                enclosing_class_index = None
        if (
            stripped.startswith(f"def {entry_point}(")
            or stripped.startswith(f"async def {entry_point}(")
            or stripped.startswith(f"class {entry_point}(")
            or stripped == f"class {entry_point}:"
        ):
            if line.lstrip() != line and enclosing_class_index is not None:
                return "\n".join(lines[enclosing_class_index:]).rstrip()
            return "\n".join(lines[index:]).rstrip()
    return code_prefix.strip()


def _split_leading_module_preamble(raw_output: str) -> tuple[str, str]:
    lines = raw_output.splitlines()
    preamble_lines: list[str] = []
    body_start = 0
    seen_top_level_import = False

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if seen_top_level_import:
                preamble_lines.append("")
            continue
        if line[:1].isspace():
            body_start = index
            break
        if stripped.startswith("import ") or stripped.startswith("from "):
            preamble_lines.append(line.rstrip())
            seen_top_level_import = True
            continue
        body_start = index
        break
    else:
        body_start = len(lines)

    preamble = "\n".join(preamble_lines).strip()
    body = "\n".join(lines[body_start:]).strip("\n")
    if not preamble:
        body = raw_output.strip("\n")
    return preamble, body


def _merge_module_text(*, support_code: str, generated_module: str) -> str:
    support_text = support_code.strip()
    generated_text = generated_module.strip()
    if not support_text:
        return generated_text + ("\n" if generated_text else "")
    if not generated_text:
        return support_text + ("\n" if support_text else "")

    try:
        support_segments = _collect_top_level_segments(support_text)
        generated_segments = _collect_top_level_segments(generated_text)
    except SyntaxError:
        return _fallback_merge_module_text(support_text, generated_text)

    generated_import_keys = {segment.key for segment in generated_segments if segment.kind == "import" and segment.key is not None}
    generated_named_keys = {segment.key for segment in generated_segments if segment.kind == "named" and segment.key is not None}
    merged_support_segments: list[str] = []

    for segment in support_segments:
        if segment.kind == "import" and segment.key in generated_import_keys:
            continue
        if segment.kind == "named" and segment.key in generated_named_keys:
            continue
        merged_support_segments.append(segment.text)

    parts = [segment for segment in merged_support_segments if segment.strip()]
    parts.append(generated_text)
    return "\n\n".join(parts).rstrip() + "\n"


def _collect_top_level_segments(source: str) -> list[_TopLevelSegment]:
    tree = ast.parse(source)
    lines = source.splitlines()
    segments: list[_TopLevelSegment] = []
    for node in tree.body:
        segment = ast.get_source_segment(source, node)
        if segment is None and getattr(node, "lineno", None) is not None and getattr(node, "end_lineno", None) is not None:
            segment = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        text = (segment or "").strip()
        if not text:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            segments.append(_TopLevelSegment(kind="import", key=ast.unparse(node).strip(), text=text))
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            segments.append(_TopLevelSegment(kind="named", key=node.name, text=text))
            continue
        segments.append(_TopLevelSegment(kind="other", key=None, text=text))
    return segments


def _fallback_merge_module_text(support_text: str, generated_text: str) -> str:
    support_lines = [line.rstrip() for line in support_text.splitlines() if line.strip()]
    generated_lines = [line.rstrip() for line in generated_text.splitlines() if line.strip()]
    generated_line_set = set(generated_lines)
    merged_lines = [line for line in support_lines if line not in generated_line_set]
    if merged_lines:
        return "\n".join(merged_lines + ["", generated_text]).rstrip() + "\n"
    return generated_text.rstrip() + "\n"


def _should_insert_body_into_stub(*, code_prefix: str, entry_point: str, raw_output: str) -> bool:
    if not code_prefix or not entry_point or not raw_output.strip():
        return False
    if f"def {entry_point}" not in code_prefix and f"class {entry_point}" not in code_prefix:
        return False
    lowered_output = raw_output.lstrip()
    return not (
        lowered_output.startswith(f"def {entry_point}")
        or lowered_output.startswith(f"class {entry_point}")
    )


def _indent_body_for_entry_stub(entry_stub: str, raw_output: str) -> str:
    body = textwrap.dedent(raw_output).strip("\n")
    if not body:
        body = "pass"

    indent_width = 4
    stub_lines = [line for line in entry_stub.splitlines() if line.strip()]
    if stub_lines:
        last_line = stub_lines[-1]
        indent_width = len(last_line) - len(last_line.lstrip()) + 4
    return textwrap.indent(body, " " * indent_width)


def _runner_block() -> str:
    return textwrap.dedent(
        """
        if __name__ == "__main__":
            import sys
            import unittest

            _candidate = globals().get("task_func") or globals().get("candidate")

            if "check" in globals():
                if _candidate is None:
                    raise RuntimeError("Missing task_func/candidate binding for check(...).")
                check(_candidate)
                raise SystemExit(0)

            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(sys.modules[__name__])
            result = unittest.TextTestRunner(verbosity=0).run(suite)
            raise SystemExit(0 if result.wasSuccessful() else 1)
        """
    ).strip()


def _build_run_io_helper(executable_code: str) -> str:
    escaped_code = repr(executable_code)
    return textwrap.dedent(
        f"""
        _SELF_TEST_EXECUTABLE_CODE = {escaped_code}

        def _run_io(_stdin: str) -> str:
            import subprocess
            import sys
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory(prefix="bigcodebench_selftest_") as _tmp_dir:
                _script_path = Path(_tmp_dir) / "candidate.py"
                _script_path.write_text(_SELF_TEST_EXECUTABLE_CODE, encoding="utf-8")
                _completed = subprocess.run(
                    [sys.executable, str(_script_path)],
                    cwd=_tmp_dir,
                    input=str(_stdin),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            if _completed.returncode != 0:
                raise AssertionError(
                    f"run_io failed with exit_code={{_completed.returncode}}\\n"
                    f"[Stdout]\\n{{_completed.stdout}}\\n"
                    f"[Stderr]\\n{{_completed.stderr}}"
                )
            return _completed.stdout
        """
    ).strip()


def _self_test_runner_block(
    task_kind: str,
    *,
    signature_compatible_named_tests: bool = False,
) -> str:
    run_target = "_run_io" if task_kind == "stdin_stdout" else "(globals().get('task_func') or globals().get('candidate'))"
    missing_error = (
        "Missing run_io binding for check(...)." if task_kind == "stdin_stdout" else "Missing task_func/candidate binding for check(...)."
    )
    return textwrap.dedent(
        f"""
        if __name__ == "__main__":
            import json
            import inspect
            import os
            import subprocess
            import sys
            import unittest

            _target = {run_target}
            _summary_prefix = {_SELF_TEST_SUMMARY_PREFIX!r}
            _signature_compatible_named_tests = {signature_compatible_named_tests!r}

            def _install_process_guard():
                _real_run = subprocess.run
                _real_check_output = subprocess.check_output
                _real_call = subprocess.call
                _real_popen = subprocess.Popen
                _real_system = os.system
                _guarded_names = {{"pgrep", "pkill", "kill", "killall"}}

                def _command_name(args):
                    if isinstance(args, (list, tuple)) and args:
                        return os.path.basename(str(args[0]))
                    if isinstance(args, str):
                        return args.strip().split(maxsplit=1)[0] if args.strip() else ""
                    return ""

                def _is_guarded(args):
                    return _command_name(args) in _guarded_names

                def _guarded_run(args, *run_args, **run_kwargs):
                    if _is_guarded(args):
                        if run_kwargs.get("check"):
                            raise subprocess.CalledProcessError(1, args, output="", stderr="")
                        return subprocess.CompletedProcess(args, 1, "", "")
                    return _real_run(args, *run_args, **run_kwargs)

                def _guarded_check_output(args, *check_args, **check_kwargs):
                    if _is_guarded(args):
                        raise subprocess.CalledProcessError(1, args, output=b"")
                    return _real_check_output(args, *check_args, **check_kwargs)

                def _guarded_call(args, *call_args, **call_kwargs):
                    if _is_guarded(args):
                        return 1
                    return _real_call(args, *call_args, **call_kwargs)

                class _GuardedProcess:
                    returncode = 1

                    def __init__(self, args):
                        self.args = args

                    def communicate(self, input=None, timeout=None):
                        return b"", b""

                    def wait(self, timeout=None):
                        return self.returncode

                    def poll(self):
                        return self.returncode

                    def kill(self):
                        return None

                    def terminate(self):
                        return None

                def _guarded_popen(args, *popen_args, **popen_kwargs):
                    if _is_guarded(args):
                        return _GuardedProcess(args)
                    return _real_popen(args, *popen_args, **popen_kwargs)

                def _guarded_system(command):
                    if isinstance(command, str) and _is_guarded(command):
                        return 1
                    return _real_system(command)

                os.kill = lambda pid, sig: None
                os.system = _guarded_system
                subprocess.run = _guarded_run
                subprocess.check_output = _guarded_check_output
                subprocess.call = _guarded_call
                subprocess.Popen = _guarded_popen

            _install_process_guard()

            def _emit_summary(_results):
                _num_tests = len(_results)
                _passed_tests = sum(1 for _result in _results if _result.get("passed"))
                _failed_tests = _num_tests - _passed_tests
                print(
                    _summary_prefix
                    + json.dumps(
                        {{
                            "num_tests": _num_tests,
                            "passed_tests": _passed_tests,
                            "failed_tests": _failed_tests,
                            "tests": _results,
                        }},
                        ensure_ascii=True,
                    )
                )

            def _run_named_tests(_tests):
                _results = []
                for _name, _fn in _tests:
                    try:
                        if _signature_compatible_named_tests:
                            _signature = inspect.signature(_fn)
                            _parameters = list(_signature.parameters.values())
                            _has_varargs = any(
                                _param.kind == inspect.Parameter.VAR_POSITIONAL
                                for _param in _parameters
                            )
                            _positional_params = [
                                _param
                                for _param in _parameters
                                if _param.kind
                                in (
                                    inspect.Parameter.POSITIONAL_ONLY,
                                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                )
                            ]
                            _keyword_candidate = _signature.parameters.get("candidate")
                            if _has_varargs or _positional_params:
                                _fn(_target)
                            elif (
                                _keyword_candidate is not None
                                and _keyword_candidate.kind == inspect.Parameter.KEYWORD_ONLY
                            ):
                                _fn(candidate=_target)
                            else:
                                _fn()
                        else:
                            _fn(_target)
                    except Exception as _exc:
                        _results.append(
                            {{
                                "name": _name,
                                "passed": False,
                                "error_type": type(_exc).__name__,
                                "error_message": str(_exc),
                            }}
                        )
                    else:
                        _results.append({{"name": _name, "passed": True}})
                _emit_summary(_results)
                _failed = [_result for _result in _results if not _result.get("passed")]
                if _failed:
                    for _result in _failed:
                        print(
                            f"{{_result['name']}}: {{_result.get('error_type')}}: {{_result.get('error_message', '')}}",
                            file=sys.stderr,
                        )
                    raise SystemExit(1)
                raise SystemExit(0)

            _named_tests = [
                (_name, _value)
                for _name, _value in sorted(globals().items())
                if _name.startswith("test_") and callable(_value)
            ]

            if _named_tests:
                if _target is None:
                    raise RuntimeError({missing_error!r})
                _run_named_tests(_named_tests)

            if "check" in globals():
                if _target is None:
                    raise RuntimeError({missing_error!r})
                try:
                    check(_target)
                except Exception as _exc:
                    _emit_summary(
                        [
                            {{
                                "name": "check",
                                "passed": False,
                                "error_type": type(_exc).__name__,
                                "error_message": str(_exc),
                            }}
                        ]
                    )
                    raise
                _emit_summary([{{"name": "check", "passed": True}}])
                raise SystemExit(0)

            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(sys.modules[__name__])
            result = unittest.TextTestRunner(verbosity=0).run(suite)
            raise SystemExit(0 if result.wasSuccessful() else 1)
        """
    ).strip()


def _run_python_harness(
    *,
    cleaned_output: str,
    executable_code: str,
    harness_code: str,
    timeout_sec: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="bigcodebench_eval_") as tmp_dir:
            script_path = Path(tmp_dir) / "run_eval.py"
            script_path.write_text(harness_code, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
            )
        duration_sec = time.perf_counter() - start
        return {
            "cleaned_output": cleaned_output,
            "executable_code": executable_code,
            "passed": completed.returncode == 0,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "duration_sec": duration_sec,
        }
    except subprocess.TimeoutExpired as exc:
        duration_sec = time.perf_counter() - start
        stdout = _normalize_subprocess_stream(exc.stdout)
        stderr = _normalize_subprocess_stream(exc.stderr)
        return {
            "cleaned_output": cleaned_output,
            "executable_code": executable_code,
            "passed": False,
            "exit_code": -1,
            "stdout": stdout,
            "stderr": stderr + f"\nTimeoutExpired: evaluation exceeded {timeout_sec} seconds.",
            "duration_sec": duration_sec,
        }


def _normalize_subprocess_stream(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _extract_self_test_summary(stdout: str) -> tuple[dict[str, Any] | None, str]:
    if not stdout:
        return None, ""
    visible_lines: list[str] = []
    summary: dict[str, Any] | None = None
    for line in stdout.splitlines():
        if line.startswith(_SELF_TEST_SUMMARY_PREFIX):
            payload = line[len(_SELF_TEST_SUMMARY_PREFIX) :].strip()
            try:
                summary = json.loads(payload)
            except json.JSONDecodeError:
                summary = None
                visible_lines.append(line)
            continue
        visible_lines.append(line)
    visible_stdout = "\n".join(visible_lines).strip()
    return summary, visible_stdout
