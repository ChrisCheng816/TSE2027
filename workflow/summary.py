from __future__ import annotations

from typing import Any

from .arm_catalog import DEFAULT_PIPELINE_ARMS, PipelineArm
from ..types import GenerationRecord


def build_experiment_summary(
    records: list[GenerationRecord],
    *,
    arms: tuple[PipelineArm, ...] | None = None,
) -> dict[str, object]:
    def _maybe_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    def _attempted_arm_count(record: Any) -> int:
        attempted_arms = _order_run_attempted_arms(record)
        if attempted_arms:
            return len(attempted_arms)
        return 1

    def _order_run_attempted_arms(record: Any) -> list[str]:
        metadata = getattr(record, "metadata", None)
        if isinstance(metadata, dict):
            attempted_arms = metadata.get("adaptive_attempted_arms")
            if isinstance(attempted_arms, list) and attempted_arms:
                return [str(arm_id) for arm_id in attempted_arms]
        selection_reason = getattr(record, "selection_reason", None)
        if isinstance(selection_reason, dict):
            attempted_arms = selection_reason.get("order_run_execution_trace")
            if isinstance(attempted_arms, list) and attempted_arms:
                return [
                    str(item.get("arm_id"))
                    for item in attempted_arms
                    if isinstance(item, dict) and item.get("arm_id") is not None
                ]
        return []

    reward_values = [record.reward for record in records if record.reward is not None]
    completion_token_values = [
        float(record.completion_tokens)
        for record in records
        if record.completion_tokens is not None
    ]
    prompt_token_values = [
        value
        for record in records
        if (value := _maybe_float(getattr(record, "prompt_tokens", None))) is not None
    ]
    total_token_values = [
        value
        for record in records
        if (value := _maybe_float(getattr(record, "total_tokens", None))) is not None
    ]
    attempted_arm_counts = [_attempted_arm_count(record) for record in records]
    num_tasks = len(records)
    passed_tasks = sum(1 for record in records if (record.pass_at_k or 0.0) > 0.0)
    ordered_arm_ids: list[str] = []
    for arm in arms or DEFAULT_PIPELINE_ARMS:
        if arm.arm_id not in ordered_arm_ids:
            ordered_arm_ids.append(arm.arm_id)
    for record in records:
        if record.arm_id not in ordered_arm_ids:
            ordered_arm_ids.append(record.arm_id)

    arm_pull_counts: dict[str, int] = {arm_id: 0 for arm_id in ordered_arm_ids}
    arm_reward_sums: dict[str, float] = {arm_id: 0.0 for arm_id in ordered_arm_ids}
    arm_prompt_token_sums: dict[str, float] = {arm_id: 0.0 for arm_id in ordered_arm_ids}
    arm_completion_token_sums: dict[str, float] = {arm_id: 0.0 for arm_id in ordered_arm_ids}
    arm_total_token_sums: dict[str, float] = {arm_id: 0.0 for arm_id in ordered_arm_ids}
    order_run_arm_attempt_position_counts: dict[str, dict[str, int]] = {
        arm_id: {"first": 0, "second": 0, "third": 0}
        for arm_id in ordered_arm_ids
    }
    has_order_run_attempts = False
    for record in records:
        arm_pull_counts[record.arm_id] += 1
        attempted_arms = _order_run_attempted_arms(record)
        if attempted_arms:
            has_order_run_attempts = True
            for position_name, arm_id in zip(("first", "second", "third"), attempted_arms):
                if arm_id not in order_run_arm_attempt_position_counts:
                    order_run_arm_attempt_position_counts[arm_id] = {
                        "first": 0,
                        "second": 0,
                        "third": 0,
                    }
                order_run_arm_attempt_position_counts[arm_id][position_name] += 1
        if record.reward is not None:
            arm_reward_sums[record.arm_id] += record.reward
        prompt_token_value = _maybe_float(getattr(record, "prompt_tokens", None))
        if prompt_token_value is not None:
            arm_prompt_token_sums[record.arm_id] += prompt_token_value
        if record.completion_tokens is not None:
            arm_completion_token_sums[record.arm_id] += float(record.completion_tokens)
        total_token_value = _maybe_float(getattr(record, "total_tokens", None))
        if total_token_value is not None:
            arm_total_token_sums[record.arm_id] += total_token_value
    arm_mean_rewards = {
        arm_id: (arm_reward_sums[arm_id] / count) if count else None
        for arm_id, count in arm_pull_counts.items()
    }
    arm_mean_prompt_tokens = {
        arm_id: (arm_prompt_token_sums[arm_id] / count) if count else None
        for arm_id, count in arm_pull_counts.items()
    }
    arm_mean_completion_tokens = {
        arm_id: (arm_completion_token_sums[arm_id] / count) if count else None
        for arm_id, count in arm_pull_counts.items()
    }
    arm_mean_total_tokens = {
        arm_id: (arm_total_token_sums[arm_id] / count) if count else None
        for arm_id, count in arm_pull_counts.items()
    }
    summary = {
        "num_tasks": num_tasks,
        "reward_count": len(reward_values),
        "mean_reward": (sum(reward_values) / len(reward_values)) if reward_values else None,
        "prompt_token_count": len(prompt_token_values),
        "mean_prompt_tokens": (
            sum(prompt_token_values) / len(prompt_token_values)
        ) if prompt_token_values else None,
        "completion_token_count": len(completion_token_values),
        "mean_completion_tokens": (
            sum(completion_token_values) / len(completion_token_values)
        ) if completion_token_values else None,
        "total_token_count": len(total_token_values),
        "mean_total_tokens": (
            sum(total_token_values) / len(total_token_values)
        ) if total_token_values else None,
        "num_passed_tasks": passed_tasks,
        "pass_rate": (passed_tasks / num_tasks) if num_tasks else None,
        "mean_arms_used_per_task": (
            sum(attempted_arm_counts) / len(attempted_arm_counts)
        ) if attempted_arm_counts else None,
        "arm_pull_counts": arm_pull_counts,
        "arm_mean_rewards": arm_mean_rewards,
        "arm_mean_prompt_tokens": arm_mean_prompt_tokens,
        "arm_mean_completion_tokens": arm_mean_completion_tokens,
        "arm_mean_total_tokens": arm_mean_total_tokens,
    }
    if has_order_run_attempts:
        summary["order_run_arm_attempt_position_counts"] = order_run_arm_attempt_position_counts
    return summary
