from __future__ import annotations

import ast
from dataclasses import dataclass

from ..dataset_families import is_bigcodebench_family_dataset
from ..output_parsing import (
    extract_fenced_code_blocks,
    extract_first_fenced_code_block,
    remove_standalone_code_fence_lines as _shared_remove_standalone_code_fence_lines,
    normalize_structured_label,
    split_structured_sections,
    strip_wrapping_code_fences as _shared_strip_wrapping_code_fences,
)
from ..types import CodeGenTask

_LIVECODEBENCH_DATASET = "livecodebench-release_v6"
_APPS_DATASET = "apps"


@dataclass(frozen=True)
class A2DraftBundle:
    code: str
    self_tests: str
    tests: str


def extract_a2_draft_bundle(text: str, *, task: CodeGenTask | None = None) -> A2DraftBundle:
    source = str(text or "").strip()
    is_livecodebench = task is not None and task.dataset_name in {_LIVECODEBENCH_DATASET, _APPS_DATASET}
    sections = split_structured_sections(
        source,
        aliases={
            "code": (),
            "tests": ("test", "testcase", "testcases"),
            "self_tests": ("selftest", "selftests", "selftestcase", "selftestcases"),
        },
    )
    code_section: str | None = None
    raw_tests: str | None = None
    resolved_tests_name: str | None = None
    for index, section in enumerate(sections):
        if section.name != "code":
            continue
        for trailing_section in sections[index + 1 :]:
            if trailing_section.name not in {"tests", "self_tests"}:
                continue
            code_section = section.content
            raw_tests = trailing_section.content
            resolved_tests_name = trailing_section.name
            break
        if code_section is not None:
            break
    if code_section is None or raw_tests is None or resolved_tests_name is None:
        code_section, raw_tests, resolved_tests_name = _extract_leading_code_before_tests(
            source,
            sections,
        )
    if code_section is None or raw_tests is None or resolved_tests_name is None:
        raise ValueError("Missing [CODE] or [TESTS]/[SELF_TESTS] section in A2 draft output.")

    code = _sanitize_livecodebench_python_section(code_section) if is_livecodebench else _sanitize_python_section(
        code_section,
        section_name="code",
    )
    if resolved_tests_name == "self_tests":
        tests = _sanitize_python_section(raw_tests, section_name="self-tests")
        self_tests = tests
    else:
        tests = (
            _sanitize_livecodebench_tests_section(raw_tests)
            if is_livecodebench
            else _sanitize_text_section(raw_tests, section_name="tests")
        )
        if is_livecodebench:
            self_tests = _compile_livecodebench_example_tests(task, tests)
        elif task is None or is_bigcodebench_family_dataset(task.dataset_name):
            self_tests = _sanitize_python_section(raw_tests, section_name="tests")
        else:
            self_tests = _sanitize_python_section(raw_tests, section_name="tests")
    if not code or not self_tests or not tests:
        raise ValueError("A2 draft output must contain non-empty code and self-test sections.")
    return A2DraftBundle(code=code, self_tests=self_tests, tests=tests)


def _extract_leading_code_before_tests(
    source: str,
    sections: list,
) -> tuple[str | None, str | None, str | None]:
    for section in sections:
        if section.name not in {"tests", "self_tests"}:
            continue
        marker_start = source.find(section.marker_text)
        if marker_start <= 0:
            continue
        leading_code = source[:marker_start].strip()
        if leading_code:
            return leading_code, section.content, section.name
    return None, None, None


def _sanitize_python_section(text: str, *, section_name: str) -> str:
    fenced_code = extract_first_fenced_code_block(text)
    sanitized = fenced_code if fenced_code is not None else _strip_code_fence(text)
    sanitized = _remove_standalone_code_fence_lines(sanitized)
    if not sanitized:
        raise ValueError(f"A2 draft output must contain non-empty {section_name} section.")
    try:
        ast.parse(sanitized)
    except SyntaxError as exc:
        raise ValueError(f"A2 draft {section_name} section is not valid Python: {exc}") from exc
    return sanitized


def _sanitize_livecodebench_python_section(text: str) -> str:
    candidates: list[str] = []
    fenced_blocks = extract_fenced_code_blocks(text)
    python_blocks = [body for language, body in fenced_blocks if language in {"python", "py"}]
    candidates.extend(reversed(python_blocks))
    candidates.extend(body for language, body in reversed(fenced_blocks) if language not in {"python", "py"})
    unclosed_python_block = _extract_unclosed_python_code_block(text)
    if unclosed_python_block:
        candidates.append(unclosed_python_block)
    stripped_section = _remove_standalone_code_fence_lines(_strip_code_fence(text))
    if stripped_section:
        candidates.append(stripped_section)

    for candidate in candidates:
        sanitized = _remove_standalone_code_fence_lines(_strip_code_fence(candidate))
        if not sanitized or not _has_python_code_signal(sanitized):
            continue
        try:
            ast.parse(sanitized)
        except SyntaxError:
            continue
        return sanitized
    raise ValueError("A2 draft code section is not valid LiveCodeBench Python.")


def _extract_unclosed_python_code_block(text: str) -> str | None:
    lines = str(text or "").splitlines()
    start_index: int | None = None
    for index, line in enumerate(lines):
        marker = line.strip().lower()
        if marker.startswith("```python") or marker.startswith("```py") or marker.startswith("``python") or marker.startswith("``py"):
            start_index = index
    if start_index is None:
        return None
    body_lines: list[str] = []
    for line in lines[start_index + 1 :]:
        if line.strip().startswith("```") or line.strip().startswith("``"):
            break
        body_lines.append(line)
    body = "\n".join(body_lines).strip()
    return body or None


def _sanitize_livecodebench_tests_section(text: str) -> str:
    sanitized = _remove_standalone_code_fence_lines(_strip_code_fence(text)).strip()
    if not sanitized:
        raise ValueError("A2 draft output must contain non-empty tests section.")
    return sanitized


def _has_python_code_signal(text: str) -> bool:
    try:
        module = ast.parse(str(text or "").strip())
    except SyntaxError:
        return False
    signal_nodes = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.If,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Return,
    )
    for node in module.body:
        if isinstance(node, signal_nodes):
            return True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return True
    return False


def _sanitize_text_section(text: str, *, section_name: str) -> str:
    fenced_text = extract_first_fenced_code_block(text)
    sanitized = fenced_text if fenced_text is not None else _strip_code_fence(text)
    sanitized = _remove_standalone_code_fence_lines(sanitized).strip()
    if not sanitized:
        raise ValueError(f"A2 draft output must contain non-empty {section_name} section.")
    return sanitized


def _compile_livecodebench_example_tests(task: CodeGenTask, tests: str) -> str:
    test_cases = _parse_example_test_cases(tests)
    if task.task_kind == "stdin_stdout":
        return _compile_stdio_example_tests(test_cases)
    return _compile_callable_example_tests(test_cases)


def _parse_example_test_cases(tests: str) -> list[tuple[str, str]]:
    lines = str(tests or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cases: list[tuple[str, str]] = []
    state: str | None = None
    input_lines: list[str] = []
    output_lines: list[str] = []

    def _finish_case() -> None:
        nonlocal input_lines, output_lines
        input_text = "\n".join(input_lines).strip()
        output_text = "\n".join(output_lines).strip()
        if input_text and output_text:
            cases.append((input_text, output_text))
        input_lines = []
        output_lines = []

    for line in lines:
        marker = line.strip()
        marker_name = ""
        if marker.startswith("[") and marker.endswith("]"):
            marker_name = normalize_structured_label(marker[1:-1])
        if marker_name == "input":
            if state == "output":
                _finish_case()
            elif state == "input" and input_lines:
                raise ValueError("Malformed [TESTS] section: encountered [Input] before finishing [Output].")
            state = "input"
            continue
        if marker_name == "output":
            if state != "input":
                raise ValueError("Malformed [TESTS] section: encountered [Output] before [Input].")
            state = "output"
            continue
        if state == "input":
            input_lines.append(line)
        elif state == "output":
            output_lines.append(line)
        elif marker:
            raise ValueError("Malformed [TESTS] section: content must appear under [Input] or [Output].")

    if state == "output":
        _finish_case()
    elif state is not None and not cases:
        raise ValueError("Malformed [TESTS] section: missing [Output] for final case.")

    if not cases:
        raise ValueError("Malformed [TESTS] section: no [Input]/[Output] cases were found.")
    return cases


def _compile_callable_example_tests(test_cases: list[tuple[str, str]]) -> str:
    rendered_tests: list[str] = []
    for index, (call_text, expected_text) in enumerate(test_cases, 1):
        call_expr = _rewrite_candidate_call(call_text)
        expected_expr = _normalize_expected_expression(expected_text)
        rendered_tests.append(
            "\n".join(
                [
                    f"def test_{index}(candidate):",
                    f"    expected = ({expected_expr})",
                    f"    actual = ({call_expr})",
                    '    assert actual == expected, f"expected {expected!r}, got {actual!r}"',
                ]
            )
        )
    return "\n\n".join(rendered_tests).strip()


def _compile_stdio_example_tests(test_cases: list[tuple[str, str]]) -> str:
    rendered_tests: list[str] = []
    for index, (stdin_text, stdout_text) in enumerate(test_cases, 1):
        rendered_tests.append(
            "\n".join(
                [
                    f"def test_{index}(run_io):",
                    f"    expected = {stdout_text!r}",
                    f"    actual = run_io({stdin_text!r})",
                    '    assert actual.strip() == expected.strip(), f"expected {expected!r}, got {actual!r}"',
                ]
            )
        )
    return "\n\n".join(rendered_tests).strip()


def _rewrite_candidate_call(call_text: str) -> str:
    source = str(call_text or "").strip()
    if not source:
        raise ValueError("Callable [TESTS] input must contain a concrete function call.")
    try:
        expression = ast.parse(source, mode="eval").body
    except SyntaxError as exc:
        expression = _extract_final_callable_expression(source)
        if expression is None:
            raise ValueError(f"Callable [TESTS] input is not a valid Python expression: {exc}") from exc
    if not isinstance(expression, ast.Call):
        raise ValueError("Callable [TESTS] input must be a single function call.")
    expression.func = ast.Name(id="candidate", ctx=ast.Load())
    return ast.unparse(expression)


def _extract_final_callable_expression(source: str) -> ast.Call | None:
    try:
        module = ast.parse(source, mode="exec")
    except SyntaxError:
        return None
    if not module.body:
        return None
    for statement in module.body[:-1]:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            return None
    final_statement = module.body[-1]
    if not isinstance(final_statement, ast.Expr) or not isinstance(final_statement.value, ast.Call):
        return None
    return final_statement.value


def _normalize_expected_expression(expected_text: str) -> str:
    source = str(expected_text or "").strip()
    if not source:
        raise ValueError("Callable [TESTS] output must contain an expected return value.")
    try:
        ast.parse(source, mode="eval")
    except SyntaxError:
        return repr(source)
    return source


def _strip_code_fence(text: str) -> str:
    return _shared_strip_wrapping_code_fences(text)


def _remove_standalone_code_fence_lines(text: str) -> str:
    return _shared_remove_standalone_code_fence_lines(text)
