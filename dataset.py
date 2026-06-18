from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .record_compaction import write_compact_jsonl

from datasets import Dataset, IterableDataset, load_dataset

from .types import CodeGenTask

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
)


def get_dataset_choices(dataset_specs: tuple[DatasetSpec, ...] = DEFAULT_DATASET_SPECS) -> tuple[str, ...]:
    return tuple(spec.name for spec in dataset_specs)


def resolve_dataset_spec(
    dataset_name: str,
    dataset_specs: tuple[DatasetSpec, ...] = DEFAULT_DATASET_SPECS,
) -> DatasetSpec:
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


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    write_compact_jsonl(path, rows)


def _load_records_from_hub(dataset_spec: DatasetSpec) -> Dataset | IterableDataset:
    dataset = load_dataset(dataset_spec.hf_dataset_path, split=dataset_spec.hf_split)
    if not isinstance(dataset, (Dataset, IterableDataset)):
        raise TypeError(f"Unsupported dataset object: {type(dataset)!r}")
    return dataset


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
