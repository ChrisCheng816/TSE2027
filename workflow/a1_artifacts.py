from __future__ import annotations

import ast
from dataclasses import dataclass

from ..output_parsing import (
    extract_first_fenced_code_block,
    remove_standalone_code_fence_lines as _shared_remove_standalone_code_fence_lines,
    split_structured_sections,
    strip_wrapping_code_fences as _shared_strip_wrapping_code_fences,
)


@dataclass(frozen=True)
class A1FinalBundle:
    plan: str
    code: str


def extract_a1_final_bundle(text: str) -> A1FinalBundle:
    source = str(text or "").strip()
    sections = split_structured_sections(
        source,
        aliases={
            "plan": (),
            "code": (),
        },
    )
    plan_section: str | None = None
    code_section: str | None = None
    for index, section in enumerate(sections):
        if section.name != "plan":
            continue
        for trailing_section in sections[index + 1 :]:
            if trailing_section.name != "code":
                continue
            plan_section = section.content
            code_section = trailing_section.content
            break
        if plan_section is not None:
            break
    if plan_section is None or code_section is None:
        fallback_code = _extract_fallback_python_code(source)
        if fallback_code is not None:
            return A1FinalBundle(
                plan="A1 output did not include parseable [PLAN]/[CODE] sections; extracted the first Python code block.",
                code=fallback_code,
            )
        raise ValueError("Missing [PLAN] or [CODE] section in A1 output.")

    plan = _sanitize_plan_section(plan_section)
    code = _sanitize_python_section(code_section, section_name="code")
    if not plan or not code:
        raise ValueError("A1 output must contain non-empty plan and code sections.")
    return A1FinalBundle(plan=plan, code=code)


def _sanitize_plan_section(text: str) -> str:
    sanitized = _strip_code_fence(text)
    sanitized = _remove_standalone_code_fence_lines(sanitized).strip()
    if not sanitized:
        raise ValueError("A1 output must contain a non-empty plan section.")
    return sanitized


def _sanitize_python_section(text: str, *, section_name: str) -> str:
    fenced_code = extract_first_fenced_code_block(text)
    sanitized = fenced_code if fenced_code is not None else _strip_code_fence(text)
    sanitized = _remove_standalone_code_fence_lines(sanitized)
    if not sanitized:
        raise ValueError(f"A1 output must contain non-empty {section_name} section.")
    try:
        ast.parse(sanitized)
    except SyntaxError as exc:
        raise ValueError(f"A1 {section_name} section is not valid Python: {exc}") from exc
    return sanitized


def _extract_fallback_python_code(text: str) -> str | None:
    fenced_code = extract_first_fenced_code_block(text)
    if fenced_code is None:
        return None
    try:
        return _sanitize_python_section(fenced_code, section_name="fallback code")
    except ValueError:
        return None


def _strip_code_fence(text: str) -> str:
    return _shared_strip_wrapping_code_fences(text)


def _remove_standalone_code_fence_lines(text: str) -> str:
    return _shared_remove_standalone_code_fence_lines(text)
