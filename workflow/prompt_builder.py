from __future__ import annotations

import json
import re

from ..dataset_families import is_bigcodebench_family_dataset
from .arm_catalog import PipelineArm
from .prompt_loader import prompt_template_path, render_prompt_template
from ..types import CodeGenTask, PromptBundle, PromptMessage


LIVECODEBENCH_DATASET = "livecodebench-release_v6"


def build_prompt_bundle(
    task: CodeGenTask,
    arm: PipelineArm,
    *,
    model_name: str | None = None,
    phase: str | None = None,
    artifacts: dict[str, str] | None = None,
) -> PromptBundle:
    resolved_phase = "draft_bundle" if arm.arm_id == "A2" and phase is None else (phase or "final_code")
    task_prompt = _build_task_prompt(task, arm_id=arm.arm_id, phase=resolved_phase)
    system_prompt = _build_system_prompt(task, arm.arm_id, model_name=model_name)
    messages = _build_messages(
        task=task,
        task_prompt=task_prompt,
        system_prompt=system_prompt,
        dataset_name=task.dataset_name,
        arm=arm,
        phase=resolved_phase,
        artifacts=artifacts or {},
    )
    user_prompt = next(message.content for message in reversed(messages) if message.role == "user")
    raw_prompt = _serialize_messages(messages)
    return PromptBundle(
        system=system_prompt,
        user=user_prompt,
        arm_id=arm.arm_id,
        raw_prompt=raw_prompt,
        phase=resolved_phase,
        messages=messages,
        model_name=model_name or "",
        dataset_name=task.dataset_name,
        task_kind=task.task_kind,
    )


def _build_task_prompt(task: CodeGenTask, *, arm_id: str, phase: str) -> str:
    if arm_id in {"A0", "A1", "A2"} and _is_livecodebench_style_dataset(task.dataset_name):
        return _extract_livecodebench_question_text(task)
    if is_bigcodebench_family_dataset(task.dataset_name) and (task.instruct_prompt or "").strip():
        base_prompt = task.instruct_prompt.strip()
    else:
        base_prompt = (task.prompt or "").strip()
    if arm_id == "A2" and phase == "draft_bundle":
        doc_examples = _render_doc_examples(task.doc_struct)
        if doc_examples:
            return f"[Doc Examples]\n{doc_examples}\n\n[Task]\n{base_prompt}"
    return base_prompt


def _render_doc_examples(doc_struct: dict[str, object] | str | None) -> str:
    parsed_doc_struct = _parse_doc_struct(doc_struct)
    if not isinstance(parsed_doc_struct, dict):
        return ""

    raw_examples = parsed_doc_struct.get("examples")
    if isinstance(raw_examples, str):
        examples = [raw_examples]
    elif isinstance(raw_examples, list):
        examples = [item for item in raw_examples]
    else:
        return ""

    rendered_examples: list[str] = []
    for example in examples:
        if isinstance(example, str):
            text = example.strip()
        else:
            text = json.dumps(example, ensure_ascii=False).strip()
        if text:
            rendered_examples.append(text)
    return "\n".join(rendered_examples).strip()


def _parse_doc_struct(doc_struct: dict[str, object] | str | None) -> dict[str, object] | None:
    if isinstance(doc_struct, dict):
        return doc_struct
    if not isinstance(doc_struct, str):
        return None
    text = doc_struct.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _build_messages(
    *,
    task: CodeGenTask,
    task_prompt: str,
    system_prompt: str,
    dataset_name: str,
    arm: PipelineArm,
    phase: str,
    artifacts: dict[str, str],
) -> tuple[PromptMessage, ...]:
    messages: list[PromptMessage] = []
    if system_prompt:
        messages.append(PromptMessage(role="system", content=system_prompt))

    if arm.arm_id == "A0":
        if phase != "final_code":
            raise KeyError(f"Unsupported phase={phase} for arm_id={arm.arm_id}")
        prompt_template = _resolve_a0_final_template(dataset_name, task_kind=task.task_kind)
        render_values = (
            _build_livecodebench_render_values(task, task_prompt=task_prompt, phase=phase)
            if _is_livecodebench_style_dataset(dataset_name)
            else {"task_prompt": task_prompt}
        )
        messages.append(PromptMessage(role="user", content=render_prompt_template(prompt_template, **render_values)))
        return tuple(messages)

    if arm.arm_id == "A1":
        if phase != "final_code":
            raise KeyError(f"Unsupported phase={phase} for arm_id={arm.arm_id}")
        prompt_template = _resolve_a1_final_template(dataset_name, task_kind=task.task_kind)
        if _is_livecodebench_style_dataset(dataset_name):
            render_values = _build_livecodebench_render_values(task, task_prompt=task_prompt, phase=phase)
        else:
            render_values = {"task_prompt": task_prompt}
            if is_bigcodebench_family_dataset(dataset_name):
                render_values["starter_code"] = (task.code_prompt or "").strip()
        messages.append(PromptMessage(role="user", content=render_prompt_template(prompt_template, **render_values)))
        return tuple(messages)

    if arm.arm_id == "A2":
        if phase == "draft_bundle":
            prompt_template = _resolve_a2_draft_template(dataset_name, task_kind=task.task_kind)
            render_values = (
                _build_livecodebench_render_values(task, task_prompt=task_prompt, phase=phase)
                if _is_livecodebench_style_dataset(dataset_name)
                else {"task_prompt": task_prompt}
            )
            messages.append(PromptMessage(role="user", content=render_prompt_template(prompt_template, **render_values)))
            return tuple(messages)
        if phase == "repair_code":
            prompt_template = _resolve_a2_repair_template(dataset_name, task_kind=task.task_kind)
            if _is_livecodebench_style_dataset(dataset_name):
                render_values = _build_livecodebench_render_values(
                    task,
                    task_prompt=task_prompt,
                    phase=phase,
                    artifacts=artifacts,
                )
            else:
                render_values = {
                    "task_prompt": task_prompt,
                    "previous_code": artifacts.get("previous_code", "").strip(),
                    "self_tests": artifacts.get("self_tests", "").strip(),
                    "error_log": artifacts.get("error_log", "").strip(),
                }
            messages.append(PromptMessage(role="user", content=render_prompt_template(prompt_template, **render_values)))
            return tuple(messages)
        raise KeyError(f"Unsupported phase={phase} for arm_id={arm.arm_id}")

    raise KeyError(f"Unsupported arm_id: {arm.arm_id}")


def _serialize_messages(messages: tuple[PromptMessage, ...]) -> str:
    chunks: list[str] = []
    for message in messages:
        chunks.extend((f"[{message.role.upper()}]", message.content.strip()))
    return "\n\n".join(chunk for chunk in chunks if chunk)


def _build_system_prompt(task: CodeGenTask, arm_id: str, *, model_name: str | None = None) -> str:
    if not _should_include_system_prompt(task=task, arm_id=arm_id, model_name=model_name):
        return ""
    relative_path = f"shared/{arm_id.lower()}.py"
    if not prompt_template_path(relative_path).is_file():
        return ""
    return render_prompt_template(relative_path)


def _should_include_system_prompt(*, task: CodeGenTask, arm_id: str, model_name: str | None) -> bool:
    del arm_id
    if not is_bigcodebench_family_dataset(task.dataset_name):
        return False
    normalized_model_name = str(model_name or "").strip().lower()
    if not normalized_model_name:
        return False
    return "deepseek" in normalized_model_name


def _resolve_a2_draft_template(dataset_name: str, *, task_kind: str | None = None) -> str:
    if dataset_name == LIVECODEBENCH_DATASET:
        return _resolve_livecodebench_template("A2", "draft_bundle", task_kind)
    return "a2/draft_bundle.py"


def _resolve_a0_final_template(dataset_name: str, *, task_kind: str | None = None) -> str:
    if dataset_name == LIVECODEBENCH_DATASET:
        return _resolve_livecodebench_template("A0", "final_code", task_kind)
    return "a0/final_code.py"


def _resolve_a1_final_template(dataset_name: str, *, task_kind: str | None = None) -> str:
    if dataset_name == LIVECODEBENCH_DATASET:
        return _resolve_livecodebench_template("A1", "final_code", task_kind)
    del task_kind
    if is_bigcodebench_family_dataset(dataset_name):
        return "a1/final_code.py"
    raise KeyError(f"Unsupported dataset for A1 prompt: {dataset_name}")


def _resolve_a2_repair_template(dataset_name: str, *, task_kind: str | None = None) -> str:
    if dataset_name == LIVECODEBENCH_DATASET:
        return _resolve_livecodebench_template("A2", "repair_code", task_kind)
    return "a2/repair_code.py"


def _resolve_livecodebench_template(arm_id: str, phase: str, task_kind: str | None) -> str:
    normalized_task_kind = "stdio" if task_kind == "stdin_stdout" else "callable"
    file_name = f"{normalized_task_kind}_{phase}.py"
    return f"{arm_id.lower()}_livecodebench/{file_name}"


def _is_livecodebench_style_dataset(dataset_name: str) -> bool:
    return dataset_name in {LIVECODEBENCH_DATASET}


def _build_livecodebench_render_values(
    task: CodeGenTask,
    *,
    task_prompt: str,
    phase: str,
    artifacts: dict[str, str] | None = None,
) -> dict[str, str]:
    artifacts = artifacts or {}
    starter_code = _extract_livecodebench_starter_code(task)
    full_task_prompt = _build_livecodebench_a0_task_prompt(task)
    previous_code = artifacts.get("previous_code", artifacts.get("Draft", "")).strip()
    self_tests = artifacts.get("self_tests", artifacts.get("compiled_tests", "")).strip()
    tests = artifacts.get("tests", self_tests).strip()
    error_log = artifacts.get("error_log", "").strip()
    if phase == "draft_bundle":
        guidance, examples = _build_livecodebench_a1_self_test_guidance(task)
    else:
        guidance, examples = _build_livecodebench_a0_final_guidance(task)
    return {
        "content": task_prompt,
        "task_prompt": full_task_prompt,
        "task": task_prompt,
        "Task": task_prompt,
        "question_content": task_prompt,
        "problem": task_prompt,
        "Problem": task_prompt,
        "starter_code": starter_code,
        "StarterCode": starter_code,
        "task_shape_guidance": guidance,
        "task_shape_examples": examples,
        "previous_code": previous_code,
        "PreviousCode": previous_code,
        "Draft": previous_code,
        "draft": previous_code,
        "tests": tests,
        "Tests": tests,
        "self_tests": self_tests,
        "SelfTests": self_tests,
        "compiled_tests": self_tests,
        "error_log": error_log,
        "ErrorLog": error_log,
        "error": error_log,
    }


def _build_livecodebench_a0_final_guidance(task: CodeGenTask) -> tuple[str, str]:
    code_prompt = (task.code_prompt or "").strip()
    entry_point = (task.entry_point or "task_func").strip() or "task_func"
    if task.task_kind == "stdin_stdout":
        return (
            "\n".join(
                [
                    "- This is a stdin/stdout LiveCodeBench task; produce a complete executable Python script.",
                    "- When the judge executes the file, it must read from stdin and write the final answer to stdout.",
                    "- Do not rely on the harness to call candidate(), solve(), or main() for you.",
                    "- If you define solve() or main(), the final file must end with a top-level call.",
                    "- Preserve the exact input and output protocol required by [Task].",
                    "- Do not rewrite stdin/stdout tasks as normal Python function signatures.",
                ]
            ),
            "\n".join(
                [
                    "# Example: executable stdin/stdout script",
                    "import sys",
                    "",
                    "def solve() -> None:",
                    "    data = sys.stdin.buffer.read().split()",
                    '    sys.stdout.write(data[0].decode() if data else "")',
                    "",
                    'if __name__ == "__main__":',
                    "    solve()",
                ]
            ),
        )

    if "class Solution" in code_prompt:
        return (
            "\n".join(
                [
                    f"- Preserve `class Solution` and the method `{entry_point}` exactly.",
                    "- The judge will call the bound method directly.",
                    "- Do not rewrite this task as a stdin/stdout script.",
                    "- Do not print the final answer unless [Task] explicitly requires stdout behavior.",
                    "- Do not instantiate `Solution` yourself.",
                    "- Implement only the requested method body and any helpers it needs.",
                ]
            ),
            "\n".join(
                [
                    "# Good: keep the required class/method shape",
                    "class Solution:",
                    f"    def {entry_point}(self, values, k):",
                    "        return 0",
                ]
            ),
        )

    return (
        "\n".join(
            [
                "- Preserve the function name, signature, imports, and starter scaffold exactly.",
                "- Return the requested value directly.",
                "- Do not read from stdin unless [Task] explicitly requires it.",
                "- Do not print the final answer unless [Task] explicitly requires stdout behavior.",
                "- Do not wrap the solution in `class Solution`.",
                "- Do not turn a callable task into a script entry point.",
            ]
        ),
        "\n".join(
            [
                "# Good: keep the callable interface",
                f"def {entry_point}(values, k):",
                "    return 0",
            ]
        ),
    )


def _build_livecodebench_a0_task_prompt(task: CodeGenTask) -> str:
    question_text = _extract_livecodebench_question_text(task)
    starter_code = _extract_livecodebench_starter_code(task)
    sections: list[str] = []
    sections.extend(("[Problem]", question_text))
    if starter_code:
        sections.extend(("[Starter Code]", "```python\n" + starter_code + "\n```"))
    return "\n\n".join(section for section in sections if section)


def _extract_livecodebench_question_text(task: CodeGenTask) -> str:
    raw_question = str(task.raw_record.get("question_content") or task.raw_record.get("question") or "").strip()
    if not raw_question:
        raw_question = _strip_appended_starter_code((task.prompt or "").strip(), (task.code_prompt or "").strip())
    normalized = raw_question.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


def _extract_livecodebench_starter_code(task: CodeGenTask) -> str:
    starter_code = str(task.code_prompt or task.raw_record.get("starter_code") or "").strip()
    if starter_code:
        return starter_code
    if task.task_kind == "stdin_stdout":
        return "\n".join(
            [
                "import sys",
                "",
                "def solve() -> None:",
                "    pass",
                "",
                'if __name__ == "__main__":',
                "    solve()",
            ]
        )
    return ""


def _strip_appended_starter_code(prompt_text: str, starter_code: str) -> str:
    if not prompt_text or not starter_code:
        return prompt_text
    starter_block = "Starter code:\n\n```python\n" + starter_code.strip() + "\n```"
    if prompt_text.endswith(starter_block):
        return prompt_text[: -len(starter_block)].rstrip()
    return prompt_text


def _question_has_explicit_examples(question_text: str) -> bool:
    return bool(
        re.search(
            r"(sample\s+input|sample\s+output|\bexample\b|\bexamples\b)",
            question_text,
            flags=re.IGNORECASE,
        )
    )


def _build_livecodebench_a1_self_test_guidance(task: CodeGenTask) -> tuple[str, str]:
    code_prompt = (task.code_prompt or "").strip()
    entry_point = (task.entry_point or "task_func").strip() or "task_func"
    if task.task_kind == "stdin_stdout":
        return (
            "\n".join(
                [
                    "- The [CODE] section must be a complete executable Python script.",
                    "- It must read from stdin and write to stdout when the file is executed by the judge.",
                    "- Do not rely on the harness to call candidate(), solve(), or main() for you.",
                    "- If you define solve() or main(), call it at top level in the generated code.",
                    "- Each self-test must accept exactly one argument named `run_io`.",
                    "- `run_io` accepts one stdin string and returns the captured stdout text.",
                    "- Build the full stdin string exactly as the problem input format requires, including newline separators.",
                    "- Compare the returned stdout text after `.strip()` unless spacing itself is part of the requirement.",
                    # "- Avoid tests whose expected stdout is long, combinatorial, or dozens of lines; keep expected outputs short.",
                    # "- Prefer small concrete inputs whose full expected stdout fits in a few lines.",
                    "- Do not rewrite stdin/stdout tasks as normal Python call signatures.",
                    "- Do not call candidate(...) for stdin/stdout tasks.",
                ]
            ),
            "\n".join(
                [
                    "def test_1(run_io):",
                    '    assert run_io("2\\n").strip() == "4"',
                    "",
                    "def test_2(run_io):",
                    '    assert run_io("3 5\\n").strip() == "8"',
                ]
            ),
        )

    if "class Solution" in code_prompt:
        lowered_entry = entry_point.lower()
        return (
            "\n".join(
                [
                    "- Each self-test must accept exactly one argument named `candidate`.",
                    f"- `candidate` is already the bound entry method `{entry_point}` from `Solution`.",
                    "- Call candidate(args...) directly.",
                    "- Do not instantiate Solution inside the self-tests.",
                    "- Do not write candidate().",
                    f"- Do not write candidate.{lowered_entry}(...).",
                    f"- Do not write Solution().{entry_point}(...).",
                ]
            ),
            "\n".join(
                [
                    "def test_1(candidate):",
                    "    assert candidate([1, 2, 3], 4) == 2",
                    "",
                    "def test_2(candidate):",
                    "    assert candidate([0, 0, 0], 1) == 0",
                ]
            ),
        )

    return (
        "\n".join(
            [
                "- Each self-test must accept exactly one argument named `candidate`.",
                "- `candidate` is the callable entry point from the task.",
                "- Call candidate(args...) directly.",
                "- Do not instantiate Solution for plain callable tasks.",
                "- Do not use run_io for plain callable tasks.",
            ]
        ),
        "\n".join(
            [
                "def test_1(candidate):",
                "    assert candidate([1, 2, 3], 2) == 2",
                "",
                "def test_2(candidate):",
                "    assert candidate([], 0) == 0",
            ]
        ),
    )

