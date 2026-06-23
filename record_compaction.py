from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = PROJECT_ROOT / "results" / "main"
INFO_ROOT = PROJECT_ROOT / "other" / "info"

SINGLE_ARM_PROMPT_FIELDS = ("raw_prompt",)
SINGLE_ARM_PREDICTION_FIELDS = ("raw_output", "stdout", "stderr", "prediction_index", "phase")

IDENTITY_FIELDS = ("task_id", "arm_id", "model_name", "dataset_name")
VECTOR_FIELDS = ("feature_vector", "decision_feature_vector")
LEGACY_DEBUG_FIELD = "debug_" + "artifact"


def sanitize_result_paths(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_result_string(value)
    if isinstance(value, list):
        return [sanitize_result_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_result_paths(item) for item in value)
    if isinstance(value, dict):
        return {key: sanitize_result_paths(item) for key, item in value.items()}
    return value


def _sanitize_result_string(text: str) -> str:
    sanitized = text
    replacements: list[tuple[str, str]] = []
    repo_parent = PROJECT_ROOT.parent

    for local_root in (PROJECT_ROOT, repo_parent / "codegen_agent"):
        root_text = local_root.as_posix()
        replacements.append((root_text + "/", "./"))
        replacements.append((root_text, "."))

    parent_text = repo_parent.as_posix()
    replacements.append((parent_text + "/", "./"))
    replacements.append((parent_text, "."))

    home_text = Path.home().as_posix()
    replacements.append((home_text + "/", "~/"))
    replacements.append((home_text, "~"))

    for source, target in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        if source and source in sanitized:
            sanitized = sanitized.replace(source, target)

    user_name = Path.home().name
    if user_name and user_name in sanitized:
        sanitized = sanitized.replace(user_name, "user")
    return sanitized


def relative_display_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def debug_sidecar_path_for_result(result_path: str | Path) -> Path:
    source_path = Path(result_path)
    try:
        relative_path = source_path.resolve().relative_to(RESULT_ROOT.resolve())
    except ValueError:
        try:
            relative_path = source_path.resolve().relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            relative_path = Path(*[part for part in source_path.parts if part not in {"/", ""}])
        return INFO_ROOT / relative_path.parent / f"{source_path.stem}.debug.jsonl"

    parts = relative_path.parts
    if parts and parts[0] == "pass_only":
        info_relative = Path("pass_only_info", *parts[1:-1])
    elif parts and parts[0] == "token_aware":
        info_relative = Path("token_aware_info", *parts[1:-1])
    else:
        info_relative = relative_path.parent
    return INFO_ROOT / info_relative / f"{source_path.stem}.debug.jsonl"


def write_compact_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target_path = Path(path)
    debug_path = debug_sidecar_path_for_result(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    compact_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    for row in rows:
        compact_row, debug_row = compact_result_row(
            row,
            result_path=target_path,
            debug_index=len(debug_rows),
        )
        compact_rows.append(compact_row)
        debug_rows.append(debug_row)

    _write_jsonl_atomic(target_path, compact_rows)
    _write_jsonl_atomic(debug_path, debug_rows)


def append_compact_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    target_path = Path(path)
    debug_path = debug_sidecar_path_for_result(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    debug_index = _line_count(debug_path)
    compact_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    for row in rows:
        compact_row, debug_row = compact_result_row(
            row,
            result_path=target_path,
            debug_index=debug_index + len(debug_rows),
        )
        compact_rows.append(compact_row)
        debug_rows.append(debug_row)

    with target_path.open("a", encoding="utf-8") as handle:
        for row in compact_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with debug_path.open("a", encoding="utf-8") as handle:
        for row in debug_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_result_row(
    row: dict[str, Any],
    *,
    result_path: str | Path,
    debug_index: int,
    existing_debug_row: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    full_row = sanitize_result_paths(_rehydrate_row(copy.deepcopy(row), existing_debug_row))
    result_schema = _classify_result_schema(full_row, result_path=Path(result_path))

    debug_payload = _base_debug_payload(
        full_row,
        result_path=Path(result_path),
        debug_index=debug_index,
        result_schema=result_schema,
        existing_debug_row=existing_debug_row,
    )

    compact_row = _base_result_row(full_row)
    _copy_feature_dimensions(full_row, compact_row)

    if result_schema == "single_arm":
        _apply_single_arm_schema(full_row, compact_row, debug_payload)
    elif result_schema == "bara":
        _apply_bara_schema(full_row, compact_row, debug_payload)
    else:
        _apply_selector_schema(full_row, compact_row, debug_payload)

    return compact_row, debug_payload


def compact_jsonl_file(path: str | Path) -> tuple[int, int, Path]:
    target_path = Path(path)
    rows = _load_jsonl(target_path)
    debug_path = debug_sidecar_path_for_result(target_path)
    existing_debug_rows = _load_jsonl(debug_path) if debug_path.exists() else []
    if existing_debug_rows:
        _validate_debug_alignment(target_path=target_path, rows=rows, debug_rows=existing_debug_rows)

    compact_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        existing_debug = existing_debug_rows[index] if index < len(existing_debug_rows) else None
        compact_row, debug_row = compact_result_row(
            row,
            result_path=target_path,
            debug_index=index,
            existing_debug_row=existing_debug,
        )
        compact_rows.append(compact_row)
        debug_rows.append(debug_row)

    original_size = target_path.stat().st_size
    target_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl_atomic(target_path, compact_rows)
    _write_jsonl_atomic(debug_path, debug_rows)
    return original_size, target_path.stat().st_size, debug_path


def _apply_single_arm_schema(
    full_row: dict[str, Any],
    compact_row: dict[str, Any],
    debug_payload: dict[str, Any],
) -> None:
    prompt = full_row.get("prompt")
    if isinstance(prompt, dict):
        compact_prompt, debug_prompt = _split_dict(prompt, keep_fields=SINGLE_ARM_PROMPT_FIELDS)
        compact_row["prompt"] = compact_prompt
        if debug_prompt:
            debug_payload["prompt"] = debug_prompt

    predictions = full_row.get("predictions")
    if isinstance(predictions, list):
        compact_predictions: list[Any] = []
        debug_predictions: list[Any] = []
        for prediction_index, prediction in enumerate(predictions):
            if not isinstance(prediction, dict):
                compact_predictions.append(prediction)
                debug_predictions.append(prediction)
                continue
            enriched_prediction = dict(prediction)
            enriched_prediction.setdefault("prediction_index", prediction_index)
            compact_prediction, debug_prediction = _split_dict(
                enriched_prediction,
                keep_fields=SINGLE_ARM_PREDICTION_FIELDS,
            )
            compact_predictions.append(compact_prediction)
            if debug_prediction:
                debug_prediction["prediction_index"] = prediction_index
                debug_prediction.setdefault("phase", enriched_prediction.get("phase"))
                debug_predictions.append(debug_prediction)
        compact_row["predictions"] = compact_predictions
        if debug_predictions:
            debug_payload["predictions"] = debug_predictions

    _move_features_to_debug(full_row, debug_payload)
    _move_selection_reason_to_debug(full_row, debug_payload)


def _apply_selector_schema(
    full_row: dict[str, Any],
    compact_row: dict[str, Any],
    debug_payload: dict[str, Any],
) -> None:
    _move_if_present(full_row, debug_payload, "prompt")
    _move_if_present(full_row, debug_payload, "predictions")
    _move_features_to_debug(full_row, debug_payload)
    selection_reason = _move_selection_reason_to_debug(full_row, debug_payload)
    compact_row["selector_summary"] = _selector_summary(selection_reason)


def _apply_bara_schema(
    full_row: dict[str, Any],
    compact_row: dict[str, Any],
    debug_payload: dict[str, Any],
) -> None:
    _move_if_present(full_row, debug_payload, "prompt")
    _move_if_present(full_row, debug_payload, "predictions")
    _move_features_to_debug(full_row, debug_payload)
    selection_reason = _move_selection_reason_to_debug(full_row, debug_payload)
    compact_row["selector_summary"] = _selector_summary(selection_reason)
    compact_row["bara_summary"] = _bara_summary(selection_reason, full_row)


def _base_result_row(row: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "prompt",
        "predictions",
        "feature_names",
        "feature_vector",
        "selection_reason",
        "selector_summary",
        "bara_summary",
        "order_run_summary",
        LEGACY_DEBUG_FIELD,
    }
    return {key: copy.deepcopy(value) for key, value in row.items() if key not in excluded}


def _base_debug_payload(
    row: dict[str, Any],
    *,
    result_path: Path,
    debug_index: int,
    result_schema: str,
    existing_debug_row: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_result_path": relative_display_path(result_path),
        "source_debug_index": int(debug_index),
        "result_schema": result_schema,
    }
    for field in IDENTITY_FIELDS:
        payload[field] = row.get(field)
    if existing_debug_row is not None and existing_debug_row.get("source_result_path") is not None:
        payload["previous_source_result_path"] = existing_debug_row.get("source_result_path")
    return payload


def _classify_result_schema(row: dict[str, Any], *, result_path: Path) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    selection_reason = row.get("selection_reason") if isinstance(row.get("selection_reason"), dict) else {}
    filename = result_path.name
    if bool(metadata.get("bara_rank_and_allocate")) or bool(selection_reason.get("bara")):
        return "bara"
    if bool(metadata.get("adaptive_order_run")) or bool(selection_reason.get("order_run")):
        return "bara"
    bara_markers = ("_bara_", "_uniform_", "_static_order", "_budget_")
    legacy_markers = ("_order_", "_order_random_", "_fix_run_")
    if any(marker in filename for marker in (*bara_markers, *legacy_markers)):
        return "bara"
    return "single_arm"


def _rehydrate_row(row: dict[str, Any], debug_row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(debug_row, dict):
        return row

    if isinstance(debug_row.get("prompt"), dict):
        prompt = row.get("prompt")
        if not isinstance(prompt, dict):
            prompt = {}
        row["prompt"] = _merge_missing(dict(prompt), debug_row["prompt"])

    if isinstance(debug_row.get("predictions"), list):
        row_predictions = row.get("predictions")
        if not isinstance(row_predictions, list):
            row_predictions = []
        merged_predictions = [dict(item) if isinstance(item, dict) else item for item in row_predictions]
        for debug_prediction_index, debug_prediction in enumerate(debug_row["predictions"]):
            if not isinstance(debug_prediction, dict):
                continue
            target_index = _safe_int(debug_prediction.get("prediction_index"), debug_prediction_index)
            while len(merged_predictions) <= target_index:
                merged_predictions.append({})
            if isinstance(merged_predictions[target_index], dict):
                merged_predictions[target_index] = _merge_missing(
                    dict(merged_predictions[target_index]),
                    debug_prediction,
                )
        row["predictions"] = merged_predictions

    if isinstance(debug_row.get("features"), dict):
        features = debug_row["features"]
        for field in ("feature_names", "feature_vector"):
            if field in features and field not in row:
                row[field] = copy.deepcopy(features[field])

    if isinstance(debug_row.get("selection_reason"), dict):
        selection_reason = row.get("selection_reason")
        if not isinstance(selection_reason, dict):
            selection_reason = {}
        row["selection_reason"] = _merge_missing(dict(selection_reason), debug_row["selection_reason"])

    return row


def _merge_missing(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if key not in base:
            base[key] = copy.deepcopy(value)
        elif isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _merge_missing(dict(base[key]), value)
    return base


def _split_dict(payload: dict[str, Any], *, keep_fields: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    keep_set = set(keep_fields)
    kept = {key: copy.deepcopy(value) for key, value in payload.items() if key in keep_set}
    moved = {key: copy.deepcopy(value) for key, value in payload.items() if key not in keep_set}
    return kept, moved


def _move_if_present(source: dict[str, Any], debug_payload: dict[str, Any], field: str) -> Any:
    if field in source:
        value = copy.deepcopy(source[field])
        debug_payload[field] = value
        return value
    return None


def _move_features_to_debug(row: dict[str, Any], debug_payload: dict[str, Any]) -> None:
    features: dict[str, Any] = {}
    if "feature_names" in row:
        features["feature_names"] = copy.deepcopy(row["feature_names"])
    if "feature_vector" in row:
        features["feature_vector"] = copy.deepcopy(row["feature_vector"])
    if features:
        debug_payload["features"] = features


def _move_selection_reason_to_debug(row: dict[str, Any], debug_payload: dict[str, Any]) -> dict[str, Any]:
    selection_reason = row.get("selection_reason")
    if not isinstance(selection_reason, dict):
        selection_reason = {}
    debug_payload["selection_reason"] = copy.deepcopy(selection_reason)
    return selection_reason


def _copy_feature_dimensions(row: dict[str, Any], compact_row: dict[str, Any]) -> None:
    feature_names = row.get("feature_names")
    feature_vector = row.get("feature_vector")
    if isinstance(feature_names, list):
        compact_row["feature_name_count"] = len(feature_names)
    compact_row["feature_vector_dim"] = len(feature_vector) if isinstance(feature_vector, list) else int(compact_row.get("feature_vector_dim") or 0)


def _selector_summary(selection_reason: dict[str, Any]) -> dict[str, Any]:
    selector = selection_reason.get("selector") if isinstance(selection_reason.get("selector"), dict) else {}
    details = selector.get("details") if isinstance(selector.get("details"), dict) else {}
    summary = {
        "selected_arm": selector.get("arm_id") or selector.get("selected_arm"),
        "ranked_arms": copy.deepcopy(details.get("ranked_arms")),
        "scores": copy.deepcopy(selector.get("scores")),
        "expected_rewards": copy.deepcopy(selector.get("expected_rewards")),
        "uncertainty_bonuses": copy.deepcopy(selector.get("uncertainty_bonuses")),
        "neighbor_count": details.get("neighbor_count"),
        "forced_exploration": details.get("forced_exploration"),
        "cold_start_randomized": details.get("cold_start_randomized"),
        "forced_arm_id": details.get("forced_arm_id"),
        "spent_completion_tokens_before_decision": selection_reason.get("spent_completion_tokens_before_decision"),
        "remaining_completion_budget_before_decision": selection_reason.get("remaining_completion_budget_before_decision"),
    }
    return {key: value for key, value in summary.items() if value is not None}


def _bara_summary(selection_reason: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    trace = selection_reason.get("bara_execution_trace")
    if not isinstance(trace, list):
        trace = selection_reason.get("order_run_execution_trace")
    if not isinstance(trace, list):
        trace = []
    compact_trace = [_compact_trace_item(item) for item in trace if isinstance(item, dict)]
    attempted_arms = [item["arm_id"] for item in compact_trace if item.get("arm_id") is not None]
    solved_rank = next((item.get("rank") for item in compact_trace if item.get("passed") is True), None)
    return {
        "enabled": bool(
            selection_reason.get("bara")
            or selection_reason.get("order_run")
            or row.get("metadata", {}).get("bara_rank_and_allocate")
            or row.get("metadata", {}).get("adaptive_order_run")
        ),
        "uniform": bool(
            selection_reason.get("bara_uniform_run")
            or row.get("metadata", {}).get("bara_uniform_run")
            or row.get("metadata", {}).get("adaptive_random_run")
        ),
        "static": bool(
            selection_reason.get("bara_static_run")
            or row.get("metadata", {}).get("bara_static_run")
            or row.get("metadata", {}).get("adaptive_fix_run")
        ),
        "static_budget": bool(
            selection_reason.get("bara_static_budget")
            or row.get("metadata", {}).get("bara_static_budget")
            or row.get("metadata", {}).get("adaptive_fix_budget")
        ),
        "free_run": bool(
            selection_reason.get("bara_free_run")
            or row.get("metadata", {}).get("bara_free_run")
            or row.get("metadata", {}).get("adaptive_free_run")
        ),
        "task_completion_token_budget": (
            selection_reason.get("bara_task_completion_token_budget")
            or selection_reason.get("order_run_task_completion_token_budget")
            or row.get("completion_token_budget")
        ),
        "attempted_arms": attempted_arms,
        "num_attempted_arms": len(attempted_arms),
        "solved_rank": solved_rank,
        "trace": compact_trace,
    }


def _compact_trace_item(item: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "arm_id": item.get("arm_id"),
        "rank": item.get("rank"),
        "allocated_completion_token_cap": item.get("allocated_completion_token_cap"),
        "actual_completion_tokens": item.get("actual_completion_tokens"),
        "actual_accounted_completion_tokens": item.get("actual_accounted_completion_tokens"),
        "actual_total_tokens": item.get("actual_total_tokens"),
        "actual_knn_total_tokens": item.get("actual_knn_total_tokens"),
        "charged_token_usage": item.get("charged_token_usage"),
        "remaining_budget_before_arm": item.get("remaining_budget_before_arm"),
        "remaining_budget_after_arm": item.get("remaining_budget_after_arm"),
        "over_allocated_token_cap": item.get("over_allocated_token_cap"),
        "passed": item.get("passed"),
        "reward": item.get("reward"),
        "generation_duration_sec": item.get("generation_duration_sec"),
    }
    allocation = item.get("allocation") if isinstance(item.get("allocation"), dict) else {}
    if allocation.get("allocation_mode") is not None:
        compact["allocation_mode"] = allocation.get("allocation_mode")
    return {key: value for key, value in compact.items() if value is not None}


def _validate_debug_alignment(
    *,
    target_path: Path,
    rows: list[dict[str, Any]],
    debug_rows: list[dict[str, Any]],
) -> None:
    if len(rows) != len(debug_rows):
        raise ValueError(
            f"Debug line count mismatch for {target_path}: result_rows={len(rows)} debug_rows={len(debug_rows)}"
        )
    for index, (row, debug_row) in enumerate(zip(rows, debug_rows)):
        for field in ("task_id", "model_name", "dataset_name"):
            left = row.get(field)
            right = debug_row.get(field)
            if right is not None and left != right:
                raise ValueError(
                    f"Debug alignment mismatch for {target_path} line {index}: field={field} result={left!r} debug={right!r}"
                )
        result_arm = row.get("arm_id")
        debug_arm = debug_row.get("arm_id")
        if debug_arm is not None and result_arm is not None and result_arm != debug_arm:
            raise ValueError(
                f"Debug alignment mismatch for {target_path} line {index}: field=arm_id result={result_arm!r} debug={debug_arm!r}"
            )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            parsed = json.loads(raw_line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)
