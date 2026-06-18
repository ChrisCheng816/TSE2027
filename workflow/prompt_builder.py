from __future__ import annotations

import json

from ..dataset_families import is_bigcodebench_family_dataset
from .arm_catalog import PipelineArm
from .prompt_loader import prompt_template_path, render_prompt_template
from ..types import CodeGenTask, PromptBundle, PromptMessage


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
        render_values = {"task_prompt": task_prompt}
        messages.append(
            PromptMessage(
                role="user",
                content=render_prompt_template(prompt_template, **render_values),
            )
        )
        return tuple(messages)

    if arm.arm_id == "A1":
        if phase != "final_code":
            raise KeyError(f"Unsupported phase={phase} for arm_id={arm.arm_id}")
        prompt_template = _resolve_a1_final_template(dataset_name, task_kind=task.task_kind)
        render_values = {"task_prompt": task_prompt}
        if is_bigcodebench_family_dataset(dataset_name):
            render_values["starter_code"] = (task.code_prompt or "").strip()
        messages.append(
            PromptMessage(
                role="user",
                content=render_prompt_template(prompt_template, **render_values),
            )
        )
        return tuple(messages)

    if arm.arm_id == "A2":
        if phase == "draft_bundle":
            prompt_template = _resolve_a2_draft_template(dataset_name, task_kind=task.task_kind)
            render_values = {"task_prompt": task_prompt}
            messages.append(
                PromptMessage(role="user", content=render_prompt_template(prompt_template, **render_values))
            )
            return tuple(messages)
        if phase == "repair_code":
            prompt_template = _resolve_a2_repair_template(dataset_name, task_kind=task.task_kind)
            render_values = {
                "task_prompt": task_prompt,
                "previous_code": artifacts.get("previous_code", "").strip(),
                "self_tests": artifacts.get("self_tests", "").strip(),
                "error_log": artifacts.get("error_log", "").strip(),
            }
            messages.append(
                PromptMessage(
                    role="user",
                    content=render_prompt_template(prompt_template, **render_values),
                )
            )
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
    del dataset_name, task_kind
    return "a2/draft_bundle.py"


def _resolve_a0_final_template(dataset_name: str, *, task_kind: str | None = None) -> str:
    del dataset_name, task_kind
    return "a0/final_code.py"


def _resolve_a1_final_template(dataset_name: str, *, task_kind: str | None = None) -> str:
    del task_kind
    if is_bigcodebench_family_dataset(dataset_name):
        return "a1/final_code.py"
    raise KeyError(f"Unsupported dataset for A1 prompt: {dataset_name}")


def _resolve_a2_repair_template(dataset_name: str, *, task_kind: str | None = None) -> str:
    del dataset_name, task_kind
    return "a2/repair_code.py"
