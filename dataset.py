from __future__ import annotations

import ast
import base64
import json
import logging
import pickle
import sys
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .record_compaction import write_compact_jsonl

from datasets import Dataset, IterableDataset, load_dataset

from .types import CodeGenTask

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_CACHE_ROOT = PROJECT_ROOT / ".cache" / "datasets"
LIVECODEBENCH_RELEASE_FILES = {
    "release_v6": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"),
}
DATASET_ALIASES = {
    "livecodebench": "livecodebench-release_v6",
}

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_dataset_path: str
    hf_split: str
    prompt_fields: tuple[str, ...]
    loader_kind: str = "hf"
    task_id_field: str = "task_id"
    complete_prompt_field: str | None = "complete_prompt"
    instruct_prompt_field: str | None = "instruct_prompt"
    canonical_solution_field: str | None = "canonical_solution"
    code_prompt_field: str | None = "code_prompt"
    test_field: str | None = "test"
    entry_point_field: str | None = "entry_point"
    doc_struct_field: str | None = "doc_struct"
    libs_field: str | None = "libs"


DEFAULT_DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="bigcodebench-v0.1.4",
        hf_dataset_path="bigcode/bigcodebench",
        hf_split="v0.1.4",
        prompt_fields=("instruct_prompt", "complete_prompt"),
    ),
    DatasetSpec(
        name="bigcodebench-hard-v0.1.4",
        hf_dataset_path="bigcode/bigcodebench-hard",
        hf_split="v0.1.4",
        prompt_fields=("instruct_prompt", "complete_prompt"),
    ),
    DatasetSpec(
        name="livecodebench-release_v6",
        hf_dataset_path="livecodebench/code_generation_lite",
        hf_split="release_v6",
        prompt_fields=("question_content",),
        loader_kind="livecodebench",
    ),
)


def get_dataset_choices(dataset_specs: tuple[DatasetSpec, ...] = DEFAULT_DATASET_SPECS) -> tuple[str, ...]:
    return tuple(spec.name for spec in dataset_specs) + tuple(DATASET_ALIASES)


def normalize_dataset_name(dataset_name: str) -> str:
    normalized = str(dataset_name or "").strip()
    return DATASET_ALIASES.get(normalized, normalized)


def resolve_dataset_spec(
    dataset_name: str,
    dataset_specs: tuple[DatasetSpec, ...] = DEFAULT_DATASET_SPECS,
) -> DatasetSpec:
    dataset_name = normalize_dataset_name(dataset_name)
    dataset_spec_map = {spec.name: spec for spec in dataset_specs}
    if dataset_name not in dataset_spec_map:
        available = ", ".join(get_dataset_choices(dataset_specs))
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available datasets: {available}")
    return dataset_spec_map[dataset_name]


def load_codegen_tasks(
    dataset_name: str,
    *,
    limit: int | None = None,
    dataset_specs: tuple[DatasetSpec, ...] = DEFAULT_DATASET_SPECS,
) -> list[CodeGenTask]:
    dataset_spec = resolve_dataset_spec(dataset_name, dataset_specs=dataset_specs)
    if dataset_spec.loader_kind == "livecodebench":
        return _load_livecodebench_tasks(dataset_spec, limit=limit)

    records = _load_records_from_hub(dataset_spec)
    tasks: list[CodeGenTask] = []
    for index, record in enumerate(records):
        task = _parse_task_record(dict(record), dataset_spec=dataset_spec, fallback_index=index)
        tasks.append(task)
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_bigcodebench_tasks(limit: int | None = None) -> list[CodeGenTask]:
    return load_codegen_tasks("bigcodebench-v0.1.4", limit=limit)


def load_bigcodebench_hard_tasks(limit: int | None = None) -> list[CodeGenTask]:
    return load_codegen_tasks("bigcodebench-hard-v0.1.4", limit=limit)


def normalize_livecodebench_record(
    record: dict[str, Any],
    *,
    fallback_index: int,
    dataset_name: str = "livecodebench-release_v6",
) -> CodeGenTask:
    question_id = str(record.get("question_id") or "").strip()
    if not question_id:
        raise ValueError(f"Missing LiveCodeBench question_id at index {fallback_index}.")

    question_content = str(record.get("question_content") or "").strip()
    if not question_content:
        raise ValueError(f"Missing LiveCodeBench question_content at index {fallback_index}.")

    starter_code = str(record.get("starter_code") or "").strip()
    public_tests = _parse_livecodebench_public_tests(str(record.get("public_test_cases") or ""))
    private_tests = _decode_livecodebench_private_tests(str(record.get("private_test_cases") or ""))
    all_tests = public_tests + private_tests
    if not all_tests:
        raise ValueError(f"Missing LiveCodeBench tests at index {fallback_index}.")
    test_types = {str(test_case.get("testtype") or "").strip() for test_case in all_tests}

    metadata = {
        "question_title": record.get("question_title"),
        "platform": record.get("platform"),
        "contest_id": record.get("contest_id"),
        "contest_date": record.get("contest_date"),
        "difficulty": record.get("difficulty"),
        "metadata": _parse_livecodebench_metadata(record.get("metadata")),
    }
    prompt = _build_prompt_with_starter(question=question_content, starter_code=starter_code)
    task_id = f"LiveCodeBench/{question_id}"

    if test_types == {"stdin"}:
        inputs, outputs = _normalize_livecodebench_stdio_test_cases(all_tests, fallback_index=fallback_index)
        return CodeGenTask(
            dataset_name=dataset_name,
            task_id=task_id,
            prompt=prompt,
            prompt_source="official_livecodebench",
            task_kind="stdin_stdout",
            code_prompt=starter_code or None,
            test_spec={"inputs": inputs, "outputs": outputs},
            raw_record=dict(record),
            metadata=metadata,
        )

    if test_types == {"functional"}:
        entry_point = _extract_single_public_callable_entry(starter_code)
        if not entry_point:
            raise ValueError(f"Missing LiveCodeBench callable entry_point at index {fallback_index}.")
        inputs, outputs = _normalize_livecodebench_functional_test_cases(all_tests, fallback_index=fallback_index)
        test_spec = {"fn_name": entry_point, "inputs": inputs, "outputs": outputs}
        return CodeGenTask(
            dataset_name=dataset_name,
            task_id=task_id,
            prompt=prompt,
            prompt_source="official_livecodebench",
            task_kind="callable",
            code_prompt=starter_code or None,
            test=_build_callable_check_function(test_spec),
            entry_point=entry_point,
            test_spec=test_spec,
            raw_record=dict(record),
            metadata=metadata,
        )

    raise ValueError(
        f"Unsupported LiveCodeBench test types at index {fallback_index}: {sorted(test_types)!r}"
    )



def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    write_compact_jsonl(path, rows)


def _load_records_from_hub(dataset_spec: DatasetSpec) -> Dataset | IterableDataset:
    dataset = load_dataset(dataset_spec.hf_dataset_path, split=dataset_spec.hf_split)
    if not isinstance(dataset, (Dataset, IterableDataset)):
        raise TypeError(f"Unsupported dataset object: {type(dataset)!r}")
    return dataset


def _load_livecodebench_tasks(dataset_spec: DatasetSpec, *, limit: int | None) -> list[CodeGenTask]:
    tasks: list[CodeGenTask] = []
    for index, record in enumerate(_iter_livecodebench_release_records(dataset_spec)):
        tasks.append(normalize_livecodebench_record(dict(record), fallback_index=index, dataset_name=dataset_spec.name))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks



def _iter_livecodebench_release_records(dataset_spec: DatasetSpec) -> Iterable[dict[str, Any]]:
    try:
        shard_names = LIVECODEBENCH_RELEASE_FILES[dataset_spec.hf_split]
    except KeyError as exc:
        raise KeyError(f"Unsupported LiveCodeBench release {dataset_spec.hf_split!r}.") from exc

    base_url = f"https://huggingface.co/datasets/{dataset_spec.hf_dataset_path}/resolve/main"
    for shard_name in shard_names:
        local_path = DATASET_CACHE_ROOT / "livecodebench" / dataset_spec.hf_split / shard_name
        if local_path.exists():
            logging.info("Reading LiveCodeBench shard from local cache -> %s", local_path)
            with local_path.open("r", encoding="utf-8") as handle:
                yield from _iter_jsonl_records(handle)
            continue

        shard_url = f"{base_url}/{shard_name}"
        logging.info("Streaming LiveCodeBench shard from remote -> %s", shard_url)
        with urllib.request.urlopen(shard_url) as response:
            yield from _iter_jsonl_records(response)


def _iter_jsonl_records(lines: Iterable[str | bytes]) -> Iterable[dict[str, Any]]:
    for raw_line in lines:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8").strip()
        else:
            line = raw_line.strip()
        if not line:
            continue
        yield json.loads(line)



def _parse_task_record(
    record: dict[str, Any],
    *,
    dataset_spec: DatasetSpec,
    fallback_index: int,
) -> CodeGenTask:
    prompt_value, prompt_source = _first_present(record, dataset_spec.prompt_fields)
    if prompt_value is None:
        raise ValueError(
            f"Missing prompt fields {dataset_spec.prompt_fields} in dataset record at index {fallback_index}."
        )

    reserved_keys = {
        dataset_spec.task_id_field,
        dataset_spec.complete_prompt_field,
        dataset_spec.instruct_prompt_field,
        dataset_spec.canonical_solution_field,
        dataset_spec.code_prompt_field,
        dataset_spec.test_field,
        dataset_spec.entry_point_field,
        dataset_spec.doc_struct_field,
        dataset_spec.libs_field,
    }
    metadata = {
        key: value
        for key, value in record.items()
        if key not in reserved_keys and key not in dataset_spec.prompt_fields
    }

    return CodeGenTask(
        dataset_name=dataset_spec.name,
        task_id=str(record.get(dataset_spec.task_id_field, fallback_index)),
        prompt=str(prompt_value),
        prompt_source=prompt_source or dataset_spec.prompt_fields[0],
        complete_prompt=_optional_string(_get_optional(record, dataset_spec.complete_prompt_field)),
        instruct_prompt=_optional_string(_get_optional(record, dataset_spec.instruct_prompt_field)),
        canonical_solution=_optional_string(_get_optional(record, dataset_spec.canonical_solution_field)),
        code_prompt=_optional_string(_get_optional(record, dataset_spec.code_prompt_field)),
        test=_optional_string(_get_optional(record, dataset_spec.test_field)),
        entry_point=_optional_string(_get_optional(record, dataset_spec.entry_point_field)),
        doc_struct=_get_optional(record, dataset_spec.doc_struct_field),
        libs=_get_optional(record, dataset_spec.libs_field),
        raw_record=record,
        metadata=metadata,
    )


def _build_prompt_with_starter(*, question: str, starter_code: str) -> str:
    sections = [question.strip()]
    if starter_code.strip():
        sections.append("Starter code:")
        sections.append("```python\n" + starter_code.strip() + "\n```")
    return "\n\n".join(section for section in sections if section)


def _build_callable_check_function(test_spec: dict[str, Any]) -> str:
    inputs_repr = _safe_repr(test_spec["inputs"])
    outputs_repr = _safe_repr(test_spec["outputs"])
    lines: list[str] = [
        f"_TEST_INPUTS = {inputs_repr}",
        f"_TEST_OUTPUTS = {outputs_repr}",
        "",
        "def check(candidate):",
        "    for raw_args, expected in zip(_TEST_INPUTS, _TEST_OUTPUTS):",
        "        if isinstance(raw_args, (list, tuple)):",
        "            result = candidate(*raw_args)",
        "        else:",
        "            result = candidate(raw_args)",
        "        assert result == expected, f'expected {expected!r}, got {result!r}'",
    ]
    return "\n".join(lines)


def _parse_livecodebench_public_tests(value: str) -> list[dict[str, Any]]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid LiveCodeBench public_test_cases payload: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError("LiveCodeBench public_test_cases must decode to a list.")
    return [dict(item) for item in parsed]


def _decode_livecodebench_private_tests(value: str) -> list[dict[str, Any]]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        decoded_bytes = base64.b64decode(text)
        decompressed = zlib.decompress(decoded_bytes)
        payload = pickle.loads(decompressed)
        parsed = json.loads(str(payload))
    except (ValueError, zlib.error, pickle.PickleError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid LiveCodeBench private_test_cases payload: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError("LiveCodeBench private_test_cases must decode to a list.")
    return [dict(item) for item in parsed]


def _normalize_livecodebench_stdio_test_cases(
    test_cases: list[dict[str, Any]],
    *,
    fallback_index: int,
) -> tuple[list[str], list[str]]:
    inputs: list[str] = []
    outputs: list[str] = []
    for test_case in test_cases:
        test_type = str(test_case.get("testtype") or "").strip()
        if test_type != "stdin":
            raise ValueError(
                f"Unsupported LiveCodeBench testtype {test_type!r} at index {fallback_index}; only 'stdin' is supported."
            )
        if "input" not in test_case:
            raise ValueError(f"Missing LiveCodeBench test input at index {fallback_index}.")
        if "output" not in test_case:
            raise ValueError(f"Missing LiveCodeBench test output at index {fallback_index}.")
        inputs.append(str(test_case["input"]))
        outputs.append(str(test_case["output"]))
    return inputs, outputs


def _normalize_livecodebench_functional_test_cases(
    test_cases: list[dict[str, Any]],
    *,
    fallback_index: int,
) -> tuple[list[list[Any]], list[Any]]:
    inputs: list[list[Any]] = []
    outputs: list[Any] = []
    for test_case in test_cases:
        test_type = str(test_case.get("testtype") or "").strip()
        if test_type != "functional":
            raise ValueError(
                f"Unsupported LiveCodeBench testtype {test_type!r} at index {fallback_index}; only 'functional' is supported."
            )
        if "input" not in test_case:
            raise ValueError(f"Missing LiveCodeBench test input at index {fallback_index}.")
        if "output" not in test_case:
            raise ValueError(f"Missing LiveCodeBench test output at index {fallback_index}.")
        inputs.append(_parse_livecodebench_functional_args(str(test_case["input"])))
        outputs.append(_parse_livecodebench_literal(str(test_case["output"])))
    return inputs, outputs


def _parse_livecodebench_functional_args(value: str) -> list[Any]:
    text = str(value or "").strip()
    if not text:
        return []
    whole_value = _try_parse_livecodebench_literal(text)
    if whole_value is not None:
        return [whole_value]
    return [_parse_livecodebench_literal(line) for line in text.splitlines() if line.strip()]


def _parse_livecodebench_literal(value: str) -> Any:
    parsed = _try_parse_livecodebench_literal(value)
    if parsed is not None:
        return parsed
    return str(value).strip()


def _try_parse_livecodebench_literal(value: str) -> Any | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def _parse_livecodebench_metadata(value: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text



def _safe_repr(value: Any) -> str:
    limit_getter = getattr(sys, "get_int_max_str_digits", None)
    limit_setter = getattr(sys, "set_int_max_str_digits", None)
    previous_limit = limit_getter() if callable(limit_getter) and callable(limit_setter) else None
    try:
        if callable(limit_setter):
            limit_setter(0)
        return repr(value)
    finally:
        if callable(limit_setter) and previous_limit is not None:
            limit_setter(previous_limit)


def _extract_single_public_callable_entry(starter_code: str) -> str | None:
    if not starter_code.strip():
        return None
    try:
        tree = ast.parse(starter_code)
    except SyntaxError:
        return _extract_callable_entry_from_incomplete_stub(starter_code)

    public_functions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
    ]
    public_class_methods: list[str] = []
    class_defs = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    for class_node in class_defs:
        public_class_methods.extend(
            child.name
            for child in class_node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_")
        )

    non_definition_nodes = [
        node
        for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if non_definition_nodes:
        return None
    if public_functions and class_defs:
        return None
    if len(public_functions) == 1:
        return public_functions[0]
    if not public_functions and len(class_defs) == 1 and len(public_class_methods) == 1:
        return public_class_methods[0]
    return None


def _extract_callable_entry_from_incomplete_stub(starter_code: str) -> str | None:
    top_level_functions: list[str] = []
    class_methods: list[str] = []
    inside_top_level_class = False

    for raw_line in starter_code.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        if indent == 0:
            inside_top_level_class = stripped.startswith("class ")
            if stripped.startswith("def ") or stripped.startswith("async def "):
                name = _extract_signature_name_from_line(stripped)
                if name and not name.startswith("_"):
                    top_level_functions.append(name)
            continue
        if inside_top_level_class and (stripped.startswith("def ") or stripped.startswith("async def ")):
            name = _extract_signature_name_from_line(stripped)
            if name and not name.startswith("_"):
                class_methods.append(name)

    if top_level_functions and class_methods:
        return None
    if len(top_level_functions) == 1:
        return top_level_functions[0]
    if len(class_methods) == 1:
        return class_methods[0]
    return None


def _extract_signature_name_from_line(line: str) -> str | None:
    stripped = line.strip()
    if stripped.startswith("async def "):
        stripped = stripped[len("async def ") :]
    elif stripped.startswith("def "):
        stripped = stripped[len("def ") :]
    else:
        return None
    name, _, _ = stripped.partition("(")
    name = name.strip()
    return name or None



def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> tuple[Any | None, str | None]:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key], key
    return None, None


def _get_optional(record: dict[str, Any], key: str | None) -> Any:
    if key is None:
        return None
    return record.get(key)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
