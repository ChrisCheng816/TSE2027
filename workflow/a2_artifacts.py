from __future__ import annotations

import ast
from dataclasses import dataclass

from ..output_parsing import (
    extract_first_fenced_code_block,
    remove_standalone_code_fence_lines as _shared_remove_standalone_code_fence_lines,
    split_structured_sections,
    strip_wrapping_code_fences as _shared_strip_wrapping_code_fences,
)
from ..types import CodeGenTask


@dataclass(frozen=True)
class A2DraftBundle:
    code: str
    self_tests: str
    tests: str


def extract_a2_draft_bundle(text: str, *, task: CodeGenTask | None = None) -> A2DraftBundle:
    source = str(text or "").strip()
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

    code = _sanitize_python_section(code_section, section_name="code")
    if resolved_tests_name == "self_tests":
        tests = _sanitize_python_section(raw_tests, section_name="self-tests")
        self_tests = tests
    else:
        tests = _sanitize_text_section(raw_tests, section_name="tests")
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


def _sanitize_text_section(text: str, *, section_name: str) -> str:
    fenced_text = extract_first_fenced_code_block(text)
    sanitized = fenced_text if fenced_text is not None else _strip_code_fence(text)
    sanitized = _remove_standalone_code_fence_lines(sanitized).strip()
    if not sanitized:
        raise ValueError(f"A2 draft output must contain non-empty {section_name} section.")
    return sanitized


def _strip_code_fence(text: str) -> str:
    return _shared_strip_wrapping_code_fences(text)


def _remove_standalone_code_fence_lines(text: str) -> str:
    return _shared_remove_standalone_code_fence_lines(text)
