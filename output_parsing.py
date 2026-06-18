from __future__ import annotations

import re
from dataclasses import dataclass


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_STRUCTURED_TAG_RE = re.compile(
    r"(?im)^[ \t]*`?\[(?P<label>[A-Za-z][A-Za-z0-9 _-]{0,40})\]`?[ \t]*"
)
_CODE_FENCE_LINE_RE = re.compile(
    r"^[ \t]*(?:`{2,}(?:[ \t]*[A-Za-z0-9_.+-]+)?|[A-Za-z0-9_.+-]+`{2,})[ \t]*$"
)


@dataclass(frozen=True)
class StructuredSection:
    name: str
    marker_text: str
    content: str


def normalize_structured_label(label: str) -> str:
    return _NON_ALNUM_RE.sub("", str(label or "").strip().lower())


def split_structured_sections(
    text: str,
    *,
    aliases: dict[str, tuple[str, ...] | list[str] | set[str]],
) -> list[StructuredSection]:
    source = str(text or "")
    alias_to_name: dict[str, str] = {}
    for canonical_name, raw_aliases in aliases.items():
        for raw_alias in {canonical_name, *(raw_aliases or ())}:
            normalized = normalize_structured_label(raw_alias)
            if normalized:
                alias_to_name[normalized] = canonical_name

    markers: list[tuple[str, str, int, int]] = []
    for match in _STRUCTURED_TAG_RE.finditer(source):
        canonical_name = alias_to_name.get(normalize_structured_label(match.group("label")))
        if canonical_name is None:
            continue
        markers.append((canonical_name, match.group(0), match.start(), match.end()))

    sections: list[StructuredSection] = []
    for index, (canonical_name, marker_text, _, marker_end) in enumerate(markers):
        next_start = markers[index + 1][2] if index + 1 < len(markers) else len(source)
        sections.append(
            StructuredSection(
                name=canonical_name,
                marker_text=marker_text,
                content=source[marker_end:next_start],
            )
        )
    return sections


def is_code_fence_line(line: str) -> bool:
    return bool(_CODE_FENCE_LINE_RE.fullmatch(str(line or "").strip()))


def strip_wrapping_code_fences(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return ""
    lines = stripped.splitlines()
    if len(lines) >= 2 and is_code_fence_line(lines[0]) and is_code_fence_line(lines[-1]):
        return "\n".join(lines[1:-1]).strip()
    return stripped


def remove_standalone_code_fence_lines(text: str) -> str:
    return "\n".join(
        line for line in str(text or "").splitlines() if not is_code_fence_line(line)
    ).strip()


def extract_first_fenced_code_block(text: str) -> str | None:
    blocks = extract_fenced_code_blocks(text)
    if not blocks:
        return None
    return blocks[0][1]


def extract_fenced_code_blocks(text: str) -> list[tuple[str, str]]:
    lines = str(text or "").splitlines()
    blocks: list[tuple[str, str]] = []
    start_index: int | None = None
    start_language = ""
    for index, line in enumerate(lines):
        if is_code_fence_line(line):
            start_index = index
            start_language = _code_fence_language(line)
            break
    if start_index is None:
        return blocks
    for index in range(start_index + 1, len(lines)):
        if is_code_fence_line(lines[index]):
            blocks.append((start_language, "\n".join(lines[start_index + 1 : index]).strip()))
            nested = extract_fenced_code_blocks("\n".join(lines[index + 1 :]))
            blocks.extend(nested)
            return blocks
    return blocks


def _code_fence_language(line: str) -> str:
    stripped = str(line or "").strip()
    match = re.fullmatch(r"`{2,}\s*([A-Za-z0-9_.+-]+)?", stripped)
    if match:
        return str(match.group(1) or "").strip().lower()
    match = re.fullmatch(r"([A-Za-z0-9_.+-]+)`{2,}", stripped)
    if match:
        return str(match.group(1) or "").strip().lower()
    return ""
