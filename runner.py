from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from .arms import DEFAULT_PIPELINE_ARMS, PipelineArm, build_prompt_bundle
from .bara import selector_apply_update
from .config import BARAConfig, RewardConfig, bara_per_task_completion_token_budget
from .evaluator import estimate_pass_at_k, evaluate_completion, evaluate_self_test_completion
from .features import BigCodeBenchFeatureExtractor
from .models import ModelClient
from .types import BanditDecision, CodeGenTask, GenerationConfig, GenerationRecord, ModelOutput, PromptBundle
from .workflow.a2_artifacts import extract_a2_draft_bundle
from .workflow.a1_artifacts import extract_a1_final_bundle
from .workflow.rewarding import (
    accounted_completion_tokens as _accounted_completion_tokens,
    accounted_total_tokens as _accounted_total_tokens,
    compute_repair_decision_reward as _compute_repair_decision_reward,
    compute_reward_components as _compute_reward_components,
    knn_total_tokens as _knn_total_tokens,
    remaining_raw_completion_token_cap as _remaining_raw_completion_token_cap,
    reward_formula_description as _reward_formula_description,
    reward_weight_dict as _reward_weight_dict,
    task_completion_token_budget as _task_completion_token_budget,
    visible_task_budget_reference as _visible_task_budget_reference,
)
from .workflow.summary import build_experiment_summary

_A2_SELF_TEST_BYPASS_TASK_IDS = frozenset({"BigCodeBench/17"})


@dataclass(frozen=True)
class AttemptStep:
    phase: str
    arm_id: str
    prompt: PromptBundle
    output: ModelOutput
    artifact_type: str
    evaluation: dict[str, Any] | None = None
    generation_duration_sec: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRunner:
    model_client: ModelClient
    selector: Any
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    bara_config: BARAConfig = field(default_factory=BARAConfig)
    feature_extractor: BigCodeBenchFeatureExtractor | None = field(default_factory=BigCodeBenchFeatureExtractor)
    budget_allocator: Any | None = None
    compute_features_and_reward: bool = True
    arms: tuple[PipelineArm, ...] = field(default_factory=lambda: DEFAULT_PIPELINE_ARMS)
    dataset_name: str = "bigcodebench-v0.1.4"
    batch_size: int = 4
    generation_seed: int | None = 42
    progress_interval: int = 50
    evaluator: Callable[[CodeGenTask, str], dict[str, Any]] | Callable[..., dict[str, Any]] = evaluate_completion
    self_test_signature_compatible: bool = False

    def run_all_arms(
        self,
        tasks: list[CodeGenTask],
        *,
        num_samples_per_task: int,
        pass_k: int,
        evaluation_timeout_sec: int,
        on_batch_complete: Callable[[PipelineArm, list[GenerationRecord], list[GenerationRecord]], None] | None = None,
        on_arm_complete: Callable[[PipelineArm, list[GenerationRecord]], None] | None = None,
    ) -> dict[str, list[GenerationRecord]]:
        self._validate_sampling_request(num_samples_per_task=num_samples_per_task, pass_k=pass_k)
        arm_records: dict[str, list[GenerationRecord]] = {arm.arm_id: [] for arm in self.arms}
        for arm in self.arms:
            logging.info("all_arms start arm=%s total_tasks=%s", arm.arm_id, len(tasks))
            resolved_batch_size = max(self.batch_size, 1)
            for batch_number, batch_start in enumerate(range(0, len(tasks), resolved_batch_size), 1):
                batch_tasks = tasks[batch_start : batch_start + resolved_batch_size]
                batch_end = batch_start + len(batch_tasks)
                logging.info(
                    "all_arms batch_start arm=%s batch=%s task_range=%s-%s batch_tasks=%s task_ids=%s",
                    arm.arm_id,
                    batch_number,
                    batch_start + 1,
                    batch_end,
                    len(batch_tasks),
                    _task_ids_for_log(batch_tasks),
                )
                batch_steps = self._run_pipeline_batch(
                    batch_tasks,
                    arm=arm,
                    evaluation_timeout_sec=evaluation_timeout_sec,
                )
                batch_records: list[GenerationRecord] = []
                for batch_offset, (task, steps) in enumerate(zip(batch_tasks, batch_steps), 1):
                    task_index = batch_start + batch_offset
                    feature_result = (
                        self.feature_extractor.extract(task)
                        if self.compute_features_and_reward and self.feature_extractor is not None
                        else None
                    )
                    metadata = {
                        "arm_hypothesis": arm.hypothesis,
                        "pipeline_stage": arm.stage,
                    }
                    if feature_result is not None:
                        metadata["feature_signals"] = feature_result.signals
                    if self.compute_features_and_reward:
                        metadata["reward_formula"] = _reward_formula_description(self.reward_config)
                        metadata["reward_weights"] = _reward_weight_dict(self.reward_config)
                    record = self._build_record(
                        task=task,
                        final_arm=arm,
                        feature_result=feature_result,
                        steps=steps,
                        pass_k=pass_k,
                        selection_reason={},
                        metadata=metadata,
                        compute_reward=self.compute_features_and_reward,
                    )
                    arm_records[arm.arm_id].append(record)
                    batch_records.append(record)
                    if self.compute_features_and_reward:
                        logging.info(
                            "all_arms arm=%s task=%s/%s task_id=%s pass@%s=%.4f reward=%.4f completion_tokens=%s reference_tokens=%s budget=%s",
                            arm.arm_id,
                            task_index,
                            len(tasks),
                            task.task_id,
                            pass_k,
                            float(record.pass_at_k or 0.0),
                            float(record.reward or 0.0),
                            record.completion_tokens,
                            record.reference_completion_tokens,
                            record.completion_token_budget,
                        )
                    else:
                        logging.info(
                            "all_arms arm=%s task=%s/%s task_id=%s pass@%s=%.4f completion_tokens=%s",
                            arm.arm_id,
                            task_index,
                            len(tasks),
                            task.task_id,
                            pass_k,
                            float(record.pass_at_k or 0.0),
                            record.completion_tokens,
                        )
                if on_batch_complete is not None and batch_records:
                    on_batch_complete(arm, batch_records, arm_records[arm.arm_id])
                logging.info(
                    "all_arms batch_end arm=%s batch=%s task_range=%s-%s batch_tasks=%s task_ids=%s",
                    arm.arm_id,
                    batch_number,
                    batch_start + 1,
                    batch_end,
                    len(batch_tasks),
                    _task_ids_for_log(batch_tasks),
                )
            if on_arm_complete is not None:
                on_arm_complete(arm, arm_records[arm.arm_id])
        return arm_records

    def run_repeated_sampling(
        self,
        tasks: list[CodeGenTask],
        *,
        arm: PipelineArm,
        num_samples_per_task: int,
        pass_k: int,
        evaluation_timeout_sec: int,
        on_batch_complete: Callable[[list[GenerationRecord], list[GenerationRecord]], None] | None = None,
    ) -> list[GenerationRecord]:
        self._validate_sampling_request(num_samples_per_task=num_samples_per_task, pass_k=pass_k)
        records: list[GenerationRecord] = []
        batch_records: list[GenerationRecord] = []
        resolved_batch_size = max(self.batch_size, 1)

        for task_index, task in enumerate(tasks, 1):
            task_budget = self._task_completion_budget(
                reference_completion_tokens=self.model_client.count_tokens(
                    _visible_task_budget_reference(task) or ""
                ),
            )
            remaining_budget = int(task_budget)
            combined_steps: list[AttemptStep] = []
            execution_trace: list[dict[str, Any]] = []
            attempt_index = 0

            while remaining_budget > 0:
                attempt_index += 1
                allocated_cap = remaining_budget
                arm_steps = self._run_arm_once_with_completion_cap(
                    task=task,
                    arm=arm,
                    evaluation_timeout_sec=evaluation_timeout_sec,
                    completion_token_cap=allocated_cap,
                )
                actual_completion_tokens = sum(step.output.completion_tokens for step in arm_steps)
                actual_accounted_completion_tokens = sum(
                    self._step_accounted_completion_tokens(step) for step in arm_steps
                )
                actual_total_tokens = self._steps_total_token_cost(arm_steps)
                actual_knn_total_tokens = self._steps_knn_total_token_cost(arm_steps)
                over_allocated_cap = actual_total_tokens > allocated_cap
                charged_token_usage = int(allocated_cap) if over_allocated_cap else actual_total_tokens
                if over_allocated_cap:
                    arm_steps = _mark_bara_steps_over_cap_failure(
                        arm_steps,
                        allocated_cap=allocated_cap,
                        actual_total_tokens=actual_total_tokens,
                    )
                solved = not over_allocated_cap and bool((arm_steps[-1].evaluation or {}).get("passed"))
                annotated_steps = [
                    replace(
                        step,
                        details={
                            **dict(step.details),
                            "repeated_sampling": True,
                            "repeated_sampling_attempt_index": attempt_index,
                            "repeated_sampling_allocated_token_cap": allocated_cap,
                            "repeated_sampling_actual_completion_tokens": actual_completion_tokens,
                            "repeated_sampling_actual_accounted_completion_tokens": actual_accounted_completion_tokens,
                            "repeated_sampling_actual_total_tokens": actual_total_tokens,
                            "repeated_sampling_actual_knn_total_tokens": actual_knn_total_tokens,
                            "repeated_sampling_charged_token_usage": charged_token_usage,
                            "repeated_sampling_over_allocated_token_cap": over_allocated_cap,
                            "repeated_sampling_remaining_budget_before": remaining_budget,
                            "repeated_sampling_remaining_budget_after": max(remaining_budget - charged_token_usage, 0),
                        },
                    )
                    for step in arm_steps
                ]
                combined_steps.extend(annotated_steps)
                execution_trace.append(
                    {
                        "arm_id": arm.arm_id,
                        "attempt_index": attempt_index,
                        "allocated_token_cap": allocated_cap,
                        "actual_completion_tokens": actual_completion_tokens,
                        "actual_accounted_completion_tokens": actual_accounted_completion_tokens,
                        "actual_total_tokens": actual_total_tokens,
                        "actual_knn_total_tokens": actual_knn_total_tokens,
                        "charged_token_usage": charged_token_usage,
                        "over_allocated_token_cap": over_allocated_cap,
                        "passed": solved,
                        "generation_duration_sec": sum(step.generation_duration_sec for step in arm_steps),
                    }
                )
                remaining_budget = max(remaining_budget - charged_token_usage, 0)
                if solved:
                    break

            record = self._build_record(
                task=task,
                final_arm=arm,
                feature_result=None,
                steps=combined_steps,
                pass_k=pass_k,
                selection_reason={},
                metadata={
                    "arm_hypothesis": arm.hypothesis,
                    "pipeline_stage": arm.stage,
                    "reward_formula": _reward_formula_description(self.reward_config),
                    "reward_weights": _reward_weight_dict(self.reward_config),
                    "repeated_sampling": True,
                    "repeated_sampling_execution_trace": execution_trace,
                    "repeated_sampling_attempt_count": attempt_index,
                    "repeated_sampling_remaining_budget": remaining_budget,
                    "repeated_sampling_over_allocated_token_cap": any(
                        e["over_allocated_token_cap"] for e in execution_trace
                    ),
                },
                completion_token_budget_override=task_budget,
            )
            logging.info(
                "repeated_sampling task=%s/%s task_id=%s arm=%s attempts=%s pass@%s=%.4f",
                task_index,
                len(tasks),
                task.task_id,
                arm.arm_id,
                attempt_index,
                pass_k,
                float(record.pass_at_k or 0.0),
            )
            records.append(record)
            batch_records.append(record)
            if on_batch_complete is not None and (
                len(batch_records) >= resolved_batch_size or task_index == len(tasks)
            ):
                on_batch_complete(list(batch_records), records)
                batch_records.clear()

        return records

    def run_bara(
        self,
        tasks: list[CodeGenTask],
        *,
        num_samples_per_task: int,
        pass_k: int,
        evaluation_timeout_sec: int,
        initial_completed_tasks: int = 0,
        initial_spent_completion_tokens: int = 0,
        on_batch_complete: Callable[[list[GenerationRecord], list[GenerationRecord]], None] | None = None,
    ) -> list[GenerationRecord]:
        self._validate_sampling_request(num_samples_per_task=num_samples_per_task, pass_k=pass_k)
        arm_map = {arm.arm_id: arm for arm in self.arms}
        if not arm_map:
            raise ValueError("BARA runs require at least one arm.")
        a0_arm = arm_map.get("A0") or next(iter(arm_map.values()))
        a2_arm = arm_map.get("A2")
        records: list[GenerationRecord] = []
        spent_token_budget = int(initial_spent_completion_tokens)
        total_budget = getattr(self.bara_config, "total_completion_token_budget", None)
        total_tasks = initial_completed_tasks + len(tasks)
        batch_records: list[GenerationRecord] = []
        resolved_batch_size = max(self.batch_size, 1)
        if self.feature_extractor is None:
            raise RuntimeError("BARA runs require a feature extractor.")
        uniform_ranker = random.Random(int(getattr(self.bara_config, "task_order_seed", 0) or 0))

        for relative_task_index, task in enumerate(tasks, 1):
            task_index = initial_completed_tasks + relative_task_index
            task_features = self.feature_extractor.extract(task)
            remaining_tasks_after_current = len(tasks) - relative_task_index
            remaining_budget = _remaining_completion_budget(
                total_completion_token_budget=total_budget,
                spent_completion_tokens=spent_token_budget,
                reserved_for_future_tasks=remaining_tasks_after_current,
            )
            decision = self.selector.select(task_features.values)
            spent_before_task = spent_token_budget
            if getattr(self.bara_config, "rank_and_allocate", False):
                steps, selected_arm, bara_trace, task_level_budget = self._run_bara_task(
                    task=task,
                    task_features=task_features,
                    decision=decision,
                    arm_map=arm_map,
                    evaluation_timeout_sec=evaluation_timeout_sec,
                    pass_k=pass_k,
                    remaining_completion_budget=remaining_budget,
                    task_index=task_index,
                    uniform_ranker=uniform_ranker,
                )
            else:
                selected_arm = arm_map.get(decision.arm_id, a0_arm)
                if selected_arm.arm_id == "A2" and a2_arm is not None:
                    steps = self._run_a2_workflow_batch(
                        [task],
                        arm=a2_arm,
                        evaluation_timeout_sec=evaluation_timeout_sec,
                        max_completion_token_cap=remaining_budget,
                    )[0]
                    selected_arm = a2_arm
                else:
                    artifact_type = "final_code" if selected_arm.arm_id == "A0" else "final_bundle"
                    steps = self._run_generation_batch(
                        [task],
                        arm=selected_arm,
                        phase="final_code",
                        artifact_type=artifact_type,
                        evaluate_output=True,
                        evaluation_timeout_sec=evaluation_timeout_sec,
                        max_new_tokens_cap=remaining_budget,
                    )[0]
                task_level_budget = None
                bara_trace = []
            spent_token_budget += (
                sum(
                    int(item.get("charged_token_usage") or item.get("actual_knn_total_tokens") or 0)
                    for item in bara_trace
                )
                if bara_trace
                else self._steps_total_token_cost(steps)
            )

            selection_reason: dict[str, Any] = {
                "selector": _decision_to_dict(decision),
                "decision_feature_names": list(task_features.names),
                "decision_feature_vector": list(task_features.values),
                "spent_completion_tokens_before_decision": spent_before_task,
                "spent_token_budget_before_decision": spent_before_task,
                "remaining_completion_budget_before_decision": remaining_budget,
                "remaining_token_budget_before_decision": remaining_budget,
            }
            if bara_trace:
                selection_reason.update(
                    {
                        "bara": True,
                        "bara_task_completion_token_budget": task_level_budget,
                        "bara_execution_trace": bara_trace,
                        "bara_uniform_run": bool(getattr(self.bara_config, "uniform_run", False)),
                        "bara_static_run": bool(getattr(self.bara_config, "static_run", False)),
                        "bara_static_budget": bool(getattr(self.bara_config, "static_budget", False)),
                        "bara_free_run": bool(getattr(self.bara_config, "free_run", False)),
                        "task_order_seed": getattr(self.bara_config, "task_order_seed", None),
                        "static_order": str(getattr(self.bara_config, "static_order", "A0-A1-A2") or "A0-A1-A2"),
                    }
                )
            feature_result = task_features

            record = self._build_record(
                task=task,
                final_arm=selected_arm,
                feature_result=feature_result,
                steps=steps,
                pass_k=pass_k,
                selection_reason=selection_reason,
                metadata={
                    "feature_signals": feature_result.signals,
                    "arm_hypothesis": selected_arm.hypothesis,
                    "pipeline_stage": selected_arm.stage,
                    "reward_formula": _reward_formula_description(self.reward_config),
                    "reward_weights": _reward_weight_dict(self.reward_config),
                    "bara_total_completion_token_budget": total_budget,
                    "spent_completion_tokens_after_task": spent_token_budget,
                    "spent_token_budget_after_task": spent_token_budget,
                    "bara_selected_arm": selected_arm.arm_id,
                    "bara_rank_and_allocate": bool(bara_trace),
                    "bara_attempted_arms": [item["arm_id"] for item in bara_trace] or [selected_arm.arm_id],
                    "bara_uniform_run": bool(getattr(self.bara_config, "uniform_run", False)),
                    "bara_static_run": bool(getattr(self.bara_config, "static_run", False)),
                    "bara_static_budget": bool(getattr(self.bara_config, "static_budget", False)),
                    "bara_free_run": bool(getattr(self.bara_config, "free_run", False)),
                    "bara_task_order_seed": getattr(self.bara_config, "task_order_seed", None),
                    "bara_static_order": str(getattr(self.bara_config, "static_order", "A0-A1-A2") or "A0-A1-A2"),
                },
                completion_token_budget_override=task_level_budget,
            )
            if not bara_trace:
                feedback = _selector_feedback(record, metric="reward")
                selector_apply_update(
                    self.selector,
                    arm_id=selected_arm.arm_id,
                    context=task_features.values,
                    reward=feedback,
                    solved=bool(record.pass_at_k and record.pass_at_k > 0.0),
                    metadata={
                        "task_id": task.task_id,
                        "record": record.to_dict(),
                    },
                )
            logging.info(
                "BARA task=%s/%s task_id=%s final_arm=%s pass@%s=%.4f reward=%.4f attempts=%s",
                task_index,
                total_tasks,
                task.task_id,
                record.arm_id,
                pass_k,
                float(record.pass_at_k or 0.0),
                float(record.reward or 0.0),
                ",".join(item["arm_id"] for item in bara_trace) if bara_trace else record.arm_id,
            )
            records.append(record)
            batch_records.append(record)
            if on_batch_complete is not None and (
                len(batch_records) >= resolved_batch_size or relative_task_index == len(tasks)
            ):
                on_batch_complete(list(batch_records), records)
                batch_records.clear()
        return records

    def _validate_sampling_request(self, *, num_samples_per_task: int, pass_k: int) -> None:
        if num_samples_per_task != 1:
            raise ValueError(
                "This workflow runner currently supports num_samples_per_task=1 only. "
                f"Got {num_samples_per_task}."
            )
        if pass_k != 1:
            raise ValueError(
                "This workflow runner currently supports pass_k=1 only. "
                f"Got {pass_k}."
            )

    def _task_completion_budget(self, *, reference_completion_tokens: int) -> int:
        return _task_completion_token_budget(
            reference_completion_tokens=reference_completion_tokens,
            budget_multiplier=self.reward_config.completion_budget_multiplier,
            fixed_budget=bara_per_task_completion_token_budget(
                self.bara_config,
                self.dataset_name,
            ),
        )

    def _count_prompt_tokens(self, prompt: PromptBundle) -> int:
        count_prompt_tokens = getattr(self.model_client, "count_prompt_tokens", None)
        if callable(count_prompt_tokens):
            return max(int(count_prompt_tokens(prompt)), 0)
        return max(int(self.model_client.count_tokens(prompt.raw_prompt or "")), 0)

    def _step_accounted_completion_tokens(self, step: AttemptStep) -> int:
        return _accounted_completion_tokens(
            completion_tokens=step.output.completion_tokens,
            reward_config=self.reward_config,
        )

    def _step_total_token_cost(self, step: AttemptStep) -> int:
        return _accounted_total_tokens(
            prompt_tokens=step.output.prompt_tokens,
            completion_tokens=step.output.completion_tokens,
            reward_config=self.reward_config,
        )

    def _steps_total_token_cost(self, steps: list[AttemptStep]) -> int:
        return sum(self._step_total_token_cost(step) for step in steps)

    def _step_knn_prompt_tokens(self, step: AttemptStep) -> int:
        return max(int(self.model_client.count_tokens(step.prompt.user or "")), 0)

    def _steps_knn_prompt_tokens(self, steps: list[AttemptStep]) -> int:
        return sum(self._step_knn_prompt_tokens(step) for step in steps)

    def _steps_knn_total_token_cost(self, steps: list[AttemptStep]) -> int:
        return _knn_total_tokens(
            prompt_tokens_excluding_system=self._steps_knn_prompt_tokens(steps),
            completion_tokens=sum(step.output.completion_tokens for step in steps),
            reward_config=self.reward_config,
        )

    def _run_bara_task(
        self,
        *,
        task: CodeGenTask,
        task_features,
        decision: BanditDecision,
        arm_map: dict[str, PipelineArm],
        evaluation_timeout_sec: int,
        pass_k: int,
        remaining_completion_budget: int | None,
        task_index: int,
        uniform_ranker: random.Random,
    ) -> tuple[list[AttemptStep], PipelineArm, list[dict[str, Any]], int]:
        if self.budget_allocator is None:
            raise RuntimeError("BARA requires a budget allocator.")
        available_arm_ids = list(arm_map)
        ranked_arms = _decision_ranked_arms(decision, available_arm_ids=available_arm_ids)
        if not ranked_arms:
            ranked_arms = list(available_arm_ids)
        for arm_id in available_arm_ids:
            if arm_id not in ranked_arms:
                ranked_arms.append(arm_id)
        if getattr(self.bara_config, "uniform_run", False):
            ranked_arms = list(available_arm_ids)
            uniform_ranker.shuffle(ranked_arms)
        if getattr(self.bara_config, "static_run", False):
            ranked_arms = _static_arm_order(
                tuple(available_arm_ids),
                str(getattr(self.bara_config, "static_order", "A0-A1-A2") or "A0-A1-A2"),
            )
            if ranked_arms:
                rotation_offset = (int(task_index) - 1) % len(ranked_arms)
                ranked_arms = ranked_arms[rotation_offset:] + ranked_arms[:rotation_offset]
        task_budget = self._task_completion_budget(
            reference_completion_tokens=self.model_client.count_tokens(_visible_task_budget_reference(task) or ""),
        )
        if remaining_completion_budget is not None:
            task_budget = max(0, min(task_budget, int(remaining_completion_budget)))

        combined_steps: list[AttemptStep] = []
        execution_trace: list[dict[str, Any]] = []
        final_arm = arm_map[ranked_arms[0]]
        remaining_budget = int(task_budget)

        for arm_index, arm_id in enumerate(ranked_arms, 1):
            if remaining_budget <= 0:
                break
            arm = arm_map[arm_id]
            if getattr(self.bara_config, "static_budget", False):
                remaining_ranked_arms = list(ranked_arms[arm_index - 1 :])
                prior_weights = dict(
                    getattr(self.bara_config, "budget_prior_weights", None)
                    or {candidate: 1.0 for candidate in available_arm_ids}
                )
                allocated_cap, fixed_arm_prior_weight, fixed_prior_denominator = _static_prior_budget_cap(
                    arm_id=arm_id,
                    remaining_ranked_arms=remaining_ranked_arms,
                    remaining_budget=remaining_budget,
                    prior_weights=prior_weights,
                )
                allocation = {
                    "arm_id": arm_id,
                    "remaining_budget_before_arm": int(remaining_budget),
                    "allocated_completion_token_cap": allocated_cap,
                    "allocation_mode": "fixed_prior_budget",
                    "fixed_budget_prior_weights": dict(prior_weights),
                    "fixed_budget_arm_prior_weight": fixed_arm_prior_weight,
                    "fixed_budget_prior_denominator_weight": fixed_prior_denominator,
                    "fixed_budget_remaining_arms": remaining_ranked_arms,
                }
            else:
                allocation = self.budget_allocator.allocate(
                    arm_id=arm_id,
                    context=task_features.values,
                    ranked_arms=ranked_arms[arm_index - 1 :],
                    remaining_budget=remaining_budget,
                )
            allocated_cap = int(allocation.get("allocated_completion_token_cap") or 0)
            if allocated_cap <= 0:
                break
            arm_steps = self._run_arm_once_with_completion_cap(
                task=task,
                arm=arm,
                evaluation_timeout_sec=evaluation_timeout_sec,
                completion_token_cap=allocated_cap,
            )
            actual_completion_tokens = sum(step.output.completion_tokens for step in arm_steps)
            actual_accounted_completion_tokens = sum(
                self._step_accounted_completion_tokens(step) for step in arm_steps
            )
            actual_total_tokens = self._steps_total_token_cost(arm_steps)
            actual_knn_total_tokens = self._steps_knn_total_token_cost(arm_steps)
            over_allocated_cap = actual_total_tokens > allocated_cap
            charged_token_usage = int(allocated_cap) if over_allocated_cap else actual_total_tokens
            if over_allocated_cap:
                arm_steps = _mark_bara_steps_over_cap_failure(
                    arm_steps,
                    allocated_cap=allocated_cap,
                    actual_total_tokens=actual_total_tokens,
                )
            solved = False if over_allocated_cap else bool((arm_steps[-1].evaluation or {}).get("passed"))
            annotated_steps = [
                replace(
                    step,
                    details={
                        **dict(step.details),
                        "bara": True,
                        "bara_rank": arm_index,
                        "bara_allocated_completion_token_cap": allocated_cap,
                        "bara_actual_completion_tokens": actual_completion_tokens,
                        "bara_actual_accounted_completion_tokens": actual_accounted_completion_tokens,
                        "bara_actual_total_tokens": actual_total_tokens,
                        "bara_actual_knn_total_tokens": actual_knn_total_tokens,
                        "bara_charged_token_usage": charged_token_usage,
                        "bara_over_allocated_token_cap": over_allocated_cap,
                        "bara_remaining_budget_before_arm": remaining_budget,
                        "bara_remaining_budget_after_arm": max(remaining_budget - charged_token_usage, 0),
                        "bara_allocation": dict(allocation),
                    },
                )
                for step in arm_steps
            ]
            combined_steps.extend(annotated_steps)
            attempt_record = self._observe_bara_attempt(
                task=task,
                task_features=task_features,
                arm=arm,
                steps=annotated_steps,
                pass_k=pass_k,
                allocated_completion_token_cap=allocated_cap,
                token_usage_for_budget=charged_token_usage,
                observe_budget_allocator=True,
            )
            execution_trace.append(
                {
                    "arm_id": arm_id,
                    "rank": arm_index,
                    "allocated_completion_token_cap": allocated_cap,
                    "actual_completion_tokens": actual_completion_tokens,
                    "actual_accounted_completion_tokens": actual_accounted_completion_tokens,
                    "actual_total_tokens": actual_total_tokens,
                    "actual_knn_total_tokens": actual_knn_total_tokens,
                    "charged_token_usage": charged_token_usage,
                    "over_allocated_token_cap": over_allocated_cap,
                    "passed": solved,
                    "reward": attempt_record.reward,
                    "generation_duration_sec": attempt_record.generation_duration_sec,
                    "allocation": dict(allocation),
                }
            )
            remaining_budget = max(remaining_budget - charged_token_usage, 0)
            final_arm = arm
            if solved:
                break
        if not combined_steps:
            fallback_arm = arm_map[ranked_arms[0]]
            fallback_cap = 1
            fallback_steps = self._run_arm_once_with_completion_cap(
                task=task,
                arm=fallback_arm,
                evaluation_timeout_sec=evaluation_timeout_sec,
                completion_token_cap=fallback_cap,
            )
            combined_steps.extend(
                [
                    replace(
                        step,
                        details={
                            **dict(step.details),
                            "bara": True,
                            "bara_rank": 1,
                            "bara_allocated_completion_token_cap": fallback_cap,
                            "bara_actual_completion_tokens": sum(
                                candidate_step.output.completion_tokens for candidate_step in fallback_steps
                            ),
                            "bara_actual_accounted_completion_tokens": sum(
                                self._step_accounted_completion_tokens(candidate_step)
                                for candidate_step in fallback_steps
                            ),
                            "bara_actual_total_tokens": self._steps_total_token_cost(fallback_steps),
                            "bara_remaining_budget_before_arm": 0,
                            "bara_remaining_budget_after_arm": 0,
                            "bara_allocation": {
                                "arm_id": fallback_arm.arm_id,
                                "allocated_completion_token_cap": fallback_cap,
                                "allocation_mode": "fallback_min_cap",
                            },
                        },
                    )
                    for step in fallback_steps
                ]
            )
            attempt_record = self._observe_bara_attempt(
                task=task,
                task_features=task_features,
                arm=fallback_arm,
                steps=list(combined_steps),
                pass_k=pass_k,
                allocated_completion_token_cap=fallback_cap,
            )
            execution_trace.append(
                {
                    "arm_id": fallback_arm.arm_id,
                    "rank": 1,
                    "allocated_completion_token_cap": fallback_cap,
                    "actual_completion_tokens": sum(step.output.completion_tokens for step in fallback_steps),
                    "actual_accounted_completion_tokens": sum(
                        self._step_accounted_completion_tokens(step) for step in fallback_steps
                    ),
                    "actual_total_tokens": self._steps_total_token_cost(fallback_steps),
                    "passed": bool((fallback_steps[-1].evaluation or {}).get("passed")),
                    "reward": attempt_record.reward,
                    "generation_duration_sec": attempt_record.generation_duration_sec,
                    "allocation": {
                        "arm_id": fallback_arm.arm_id,
                        "allocated_completion_token_cap": fallback_cap,
                        "allocation_mode": "fallback_min_cap",
                    },
                }
            )
            final_arm = fallback_arm
        return combined_steps, final_arm, execution_trace, int(task_budget)

    def _run_arm_once_with_completion_cap(
        self,
        *,
        task: CodeGenTask,
        arm: PipelineArm,
        evaluation_timeout_sec: int,
        completion_token_cap: int | None,
    ) -> list[AttemptStep]:
        if arm.arm_id == "A2":
            return self._run_a2_workflow_batch(
                [task],
                arm=arm,
                evaluation_timeout_sec=evaluation_timeout_sec,
                max_completion_token_cap=completion_token_cap,
            )[0]
        artifact_type = "final_code" if arm.arm_id == "A0" else "final_bundle"
        return self._run_generation_batch(
            [task],
            arm=arm,
            phase="final_code",
            artifact_type=artifact_type,
            evaluate_output=True,
            evaluation_timeout_sec=evaluation_timeout_sec,
            max_new_tokens_cap=completion_token_cap,
        )[0]

    def _observe_bara_attempt(
        self,
        *,
        task: CodeGenTask,
        task_features,
        arm: PipelineArm,
        steps: list[AttemptStep],
        pass_k: int,
        allocated_completion_token_cap: int,
        token_usage_for_budget: int | None = None,
        observe_budget_allocator: bool = True,
    ) -> GenerationRecord:
        feature_result = task_features
        attempt_record = self._build_record(
            task=task,
            final_arm=arm,
            feature_result=feature_result,
            steps=steps,
            pass_k=pass_k,
            selection_reason={},
            metadata={
                "feature_signals": feature_result.signals,
                "arm_hypothesis": arm.hypothesis,
                "pipeline_stage": arm.stage,
                "reward_formula": _reward_formula_description(self.reward_config),
                "reward_weights": _reward_weight_dict(self.reward_config),
                "bara_rank_and_allocate_attempt": True,
            },
            completion_token_budget_override=allocated_completion_token_cap,
        )
        feedback = _selector_feedback(attempt_record, metric="reward")
        selector_apply_update(
            self.selector,
            arm_id=arm.arm_id,
            context=task_features.values,
            reward=feedback,
            solved=bool(attempt_record.pass_at_k and attempt_record.pass_at_k > 0.0),
            metadata={
                "task_id": task.task_id,
                "record": attempt_record.to_dict(),
                "bara": True,
            },
        )
        if observe_budget_allocator and self.budget_allocator is not None:
            self.budget_allocator.observe(
                arm.arm_id,
                task_features.values,
                token_usage=(
                    int(token_usage_for_budget)
                    if token_usage_for_budget is not None
                    else self._steps_total_token_cost(steps)
                ),
                token_cap=allocated_completion_token_cap,
                metadata={
                    "task_id": task.task_id,
                    "record": attempt_record.to_dict(),
                    "bara": True,
                },
            )
        return attempt_record

    def _run_pipeline(
        self,
        task: CodeGenTask,
        *,
        arm: PipelineArm,
        evaluation_timeout_sec: int,
    ) -> list[AttemptStep]:
        return self._run_pipeline_batch([task], arm=arm, evaluation_timeout_sec=evaluation_timeout_sec)[0]

    def _run_pipeline_batch(
        self,
        tasks: list[CodeGenTask],
        *,
        arm: PipelineArm,
        evaluation_timeout_sec: int,
    ) -> list[list[AttemptStep]]:
        if arm.arm_id == "A0":
            return self._run_generation_batch(
                tasks,
                arm=arm,
                phase="final_code",
                artifact_type="final_code",
                evaluate_output=True,
                evaluation_timeout_sec=evaluation_timeout_sec,
            )

        if arm.arm_id == "A2":
            return self._run_a2_workflow_batch(
                tasks,
                arm=arm,
                evaluation_timeout_sec=evaluation_timeout_sec,
            )

        if arm.arm_id == "A1":
            return self._run_generation_batch(
                tasks,
                arm=arm,
                phase="final_code",
                artifact_type="final_bundle",
                evaluate_output=True,
                evaluation_timeout_sec=evaluation_timeout_sec,
            )

        raise KeyError(f"Unsupported arm_id: {arm.arm_id}")

    def _run_generation_batch(
        self,
        tasks: list[CodeGenTask],
        *,
        arm: PipelineArm,
        phase: str,
        artifact_type: str,
        evaluate_output: bool,
        evaluation_timeout_sec: int,
        config: GenerationConfig | None = None,
        artifacts_by_task: list[dict[str, str]] | None = None,
        max_new_tokens_cap: int | None = None,
    ) -> list[list[AttemptStep]]:
        prompts = [
            build_prompt_bundle(
                task,
                arm,
                model_name=self.model_client.model_name,
                phase=phase,
                artifacts=(artifacts_by_task[index] if artifacts_by_task is not None else None),
            )
            for index, task in enumerate(tasks)
        ]
        resolved_max_new_tokens_cap = max_new_tokens_cap
        if max_new_tokens_cap is not None and prompts:
            prompt_token_cap_candidates = [
                _remaining_raw_completion_token_cap(
                    token_budget=max_new_tokens_cap,
                    prompt_tokens=self._count_prompt_tokens(prompt),
                    reward_config=self.reward_config,
                )
                for prompt in prompts
            ]
            resolved_max_new_tokens_cap = min(prompt_token_cap_candidates)
            if resolved_max_new_tokens_cap <= 0:
                step_lists: list[list[AttemptStep]] = []
                for task, prompt in zip(tasks, prompts):
                    empty_output = ModelOutput(
                        raw_output="",
                        cleaned_output="",
                        completion_tokens=0,
                        prompt_tokens=0,
                    )
                    evaluation = (
                        self.evaluator(task, "", timeout_sec=evaluation_timeout_sec)
                        if evaluate_output
                        else None
                    )
                    step_lists.append(
                        [
                            AttemptStep(
                                phase=phase,
                                arm_id=arm.arm_id,
                                prompt=prompt,
                                output=empty_output,
                                artifact_type=artifact_type,
                                evaluation=evaluation,
                                generation_duration_sec=0.0,
                                details={
                                    "skipped_due_to_budget": True,
                                    "remaining_token_budget_before_call": int(max_new_tokens_cap),
                                },
                            )
                        ]
                    )
                return step_lists
        generation_config = _phase_generation_config(
            config or self.generation_config,
            tasks=tasks,
            arm=arm,
            phase=phase,
            max_new_tokens_cap=resolved_max_new_tokens_cap,
        )
        generation_start = time.perf_counter()
        candidate_outputs = self.model_client.generate_batch(
            prompts,
            generation_config,
            n=1,
            seed=self.generation_seed,
        )
        generation_duration_sec = time.perf_counter() - generation_start
        per_prompt_generation_duration_sec = (
            generation_duration_sec / len(prompts)
            if prompts
            else 0.0
        )
        step_lists: list[list[AttemptStep]] = []
        for task, prompt, prompt_candidates in zip(tasks, prompts, candidate_outputs):
            candidate_output = (
                prompt_candidates[0]
                if prompt_candidates
                else ModelOutput(raw_output="", cleaned_output="", completion_tokens=0)
            )
            candidate_output, details = _normalize_structured_output(
                arm=arm,
                phase=phase,
                output=candidate_output,
            )
            evaluation = (
                self.evaluator(
                    task,
                    candidate_output.cleaned_output,
                    timeout_sec=evaluation_timeout_sec,
                )
                if evaluate_output
                else None
            )
            step_lists.append(
                [
                    AttemptStep(
                        phase=phase,
                        arm_id=arm.arm_id,
                        prompt=prompt,
                        output=candidate_output,
                        artifact_type=artifact_type,
                        evaluation=evaluation,
                        generation_duration_sec=per_prompt_generation_duration_sec,
                        details=details,
                    )
                ]
            )
        return step_lists

    def _run_a2_workflow_batch(
        self,
        tasks: list[CodeGenTask],
        *,
        arm: PipelineArm,
        evaluation_timeout_sec: int,
        max_completion_token_cap: int | None = None,
    ) -> list[list[AttemptStep]]:
        if len(tasks) <= 1 or max_completion_token_cap is not None:
            return self._run_a2_workflow_batch_sequential(
                tasks,
                arm=arm,
                evaluation_timeout_sec=evaluation_timeout_sec,
                max_completion_token_cap=max_completion_token_cap,
            )

        batch_task_ids = _task_ids_for_log(tasks)
        logging.info(
            "a2_batch draft_generate_start arm=%s batch_tasks=%s task_ids=%s",
            arm.arm_id,
            len(tasks),
            batch_task_ids,
        )
        draft_step_lists = self._run_generation_batch(
            tasks,
            arm=arm,
            phase="draft_bundle",
            artifact_type="draft_bundle",
            evaluate_output=False,
            evaluation_timeout_sec=evaluation_timeout_sec,
        )
        logging.info(
            "a2_batch draft_generate_end arm=%s batch_tasks=%s task_ids=%s",
            arm.arm_id,
            len(tasks),
            batch_task_ids,
        )
        step_lists: list[list[AttemptStep]] = [[] for _ in tasks]
        repair_indices: list[int] = []
        repair_tasks: list[CodeGenTask] = []
        repair_artifacts: list[dict[str, str]] = []
        draft_pass_count = 0
        parse_error_count = 0

        logging.info(
            "a2_batch draft_eval_start arm=%s batch_tasks=%s task_ids=%s",
            arm.arm_id,
            len(tasks),
            batch_task_ids,
        )
        for index, (task, draft_steps) in enumerate(zip(tasks, draft_step_lists)):
            draft_step = draft_steps[0]
            skip_self_test = _should_skip_a2_self_test(task)
            skip_reason = "task_id_self_test_bypass" if skip_self_test else None
            if skip_self_test:
                logging.info(
                    "a2_batch draft_self_test_skipped arm=%s task_id=%s reason=%s",
                    arm.arm_id,
                    task.task_id,
                    skip_reason,
                )
                draft_code, self_tests, tests, parse_error = self._extract_a2_draft_bundle(task, draft_step)
                draft_self_eval = _draft_failure_evaluation(parse_error) if parse_error else None
            else:
                draft_code, self_tests, tests, draft_self_eval, parse_error = self._evaluate_a2_draft_step(
                    task,
                    draft_step,
                    evaluation_timeout_sec=evaluation_timeout_sec,
                )
            details: dict[str, Any] = _a2_test_artifact_details(
                task,
                compiled_tests=self_tests,
                tests=tests,
            )
            if skip_reason is not None:
                details["self_test_skipped_reason"] = skip_reason
            elif draft_self_eval is not None:
                details["self_test_evaluation"] = dict(draft_self_eval)
                details["self_test_summary"] = _self_test_summary(draft_self_eval)
            draft_step = replace(
                draft_step,
                output=ModelOutput(
                    raw_output=draft_step.output.raw_output,
                    cleaned_output=draft_code,
                    completion_tokens=draft_step.output.completion_tokens,
                    prompt_tokens=draft_step.output.prompt_tokens,
                ),
                evaluation=draft_self_eval,
                details=details,
            )
            if parse_error:
                draft_step.details["bundle_parse_error"] = parse_error
                parse_error_count += 1
            step_lists[index].append(draft_step)

            if skip_self_test and not parse_error:
                official_eval = self.evaluator(task, draft_code, timeout_sec=evaluation_timeout_sec)
                draft_step = replace(
                    draft_step,
                    evaluation=official_eval,
                    details={
                        **draft_step.details,
                        "final_official_evaluation": dict(official_eval),
                    },
                )
                step_lists[index][0] = draft_step
                if official_eval.get("passed"):
                    draft_pass_count += 1
                    continue
                repair_evaluation = official_eval
            else:
                repair_evaluation = draft_self_eval or {}

            if (draft_self_eval or {}).get("passed"):
                draft_pass_count += 1
                official_eval = self.evaluator(task, draft_code, timeout_sec=evaluation_timeout_sec)
                step_lists[index][0] = replace(
                    draft_step,
                    evaluation=official_eval,
                    details={
                        **draft_step.details,
                        "final_official_evaluation": dict(official_eval),
                    },
                )
                continue

            repair_indices.append(index)
            repair_tasks.append(task)
            repair_artifacts.append(
                {
                    "previous_code": draft_code,
                    "self_tests": self_tests,
                    "compiled_tests": self_tests,
                    "tests": tests,
                    "error_log": _evaluation_error_log(
                        repair_evaluation,
                        failed_tests_heading=_failed_tests_heading(task),
                    ),
                }
            )

        logging.info(
            "a2_batch draft_eval_end arm=%s batch_tasks=%s draft_passed=%s repair_candidates=%s parse_errors=%s repair_task_ids=%s",
            arm.arm_id,
            len(tasks),
            draft_pass_count,
            len(repair_tasks),
            parse_error_count,
            _task_ids_for_log(repair_tasks),
        )
        if not repair_tasks:
            logging.info(
                "a2_batch repair_generate_skipped arm=%s batch_tasks=%s repair_candidates=0",
                arm.arm_id,
                len(tasks),
            )
            return step_lists

        repair_task_ids = _task_ids_for_log(repair_tasks)
        logging.info(
            "a2_batch repair_generate_start arm=%s repair_candidates=%s task_ids=%s",
            arm.arm_id,
            len(repair_tasks),
            repair_task_ids,
        )
        repair_step_lists = self._run_generation_batch(
            repair_tasks,
            arm=arm,
            phase="repair_code",
            artifact_type="repair_code",
            evaluate_output=False,
            evaluation_timeout_sec=evaluation_timeout_sec,
            artifacts_by_task=repair_artifacts,
        )
        logging.info(
            "a2_batch repair_generate_end arm=%s repair_candidates=%s task_ids=%s",
            arm.arm_id,
            len(repair_tasks),
            repair_task_ids,
        )
        logging.info(
            "a2_batch repair_eval_start arm=%s repair_candidates=%s task_ids=%s",
            arm.arm_id,
            len(repair_tasks),
            repair_task_ids,
        )
        for task_index, task, repair_steps in zip(repair_indices, repair_tasks, repair_step_lists):
            repair_step = repair_steps[0]
            official_eval = self.evaluator(
                task,
                repair_step.output.cleaned_output,
                timeout_sec=evaluation_timeout_sec,
            )
            repair_step = replace(
                repair_step,
                evaluation=official_eval,
                details={
                    **_a2_test_artifact_details(
                        task,
                        compiled_tests=_compiled_tests_from_details(step_lists[task_index][0].details),
                        tests=str(step_lists[task_index][0].details.get("tests") or ""),
                    ),
                    "final_official_evaluation": dict(official_eval),
                },
            )
            step_lists[task_index].append(repair_step)
        logging.info(
            "a2_batch repair_eval_end arm=%s repair_candidates=%s task_ids=%s",
            arm.arm_id,
            len(repair_tasks),
            repair_task_ids,
        )
        return step_lists

    def _run_a2_workflow_batch_sequential(
        self,
        tasks: list[CodeGenTask],
        *,
        arm: PipelineArm,
        evaluation_timeout_sec: int,
        max_completion_token_cap: int | None = None,
    ) -> list[list[AttemptStep]]:
        remaining_budget = max_completion_token_cap
        step_lists: list[list[AttemptStep]] = []
        for task in tasks:
            steps = self._run_a2_workflow(
                task,
                arm=arm,
                evaluation_timeout_sec=evaluation_timeout_sec,
                max_completion_token_cap=remaining_budget,
            )
            step_lists.append(steps)
            if remaining_budget is not None:
                remaining_budget -= self._steps_total_token_cost(steps)
        return step_lists

    def _run_a2_workflow(
        self,
        task: CodeGenTask,
        *,
        arm: PipelineArm,
        evaluation_timeout_sec: int,
        max_completion_token_cap: int | None = None,
    ) -> list[AttemptStep]:
        remaining_budget = max_completion_token_cap
        draft_step = self._run_generation_batch(
            [task],
            arm=arm,
            phase="draft_bundle",
            artifact_type="draft_bundle",
            evaluate_output=False,
            evaluation_timeout_sec=evaluation_timeout_sec,
            max_new_tokens_cap=remaining_budget,
        )[0][0]
        if remaining_budget is not None:
            remaining_budget -= self._step_total_token_cost(draft_step)

        skip_self_test = _should_skip_a2_self_test(task)
        skip_reason = "task_id_self_test_bypass" if skip_self_test else None
        if skip_self_test:
            draft_code, self_tests, tests, parse_error = self._extract_a2_draft_bundle(task, draft_step)
            draft_self_eval = _draft_failure_evaluation(parse_error) if parse_error else None
        else:
            draft_code, self_tests, tests, draft_self_eval, parse_error = self._evaluate_a2_draft_step(
                task,
                draft_step,
                evaluation_timeout_sec=evaluation_timeout_sec,
            )
        details: dict[str, Any] = _a2_test_artifact_details(
            task,
            compiled_tests=self_tests,
            tests=tests,
        )
        if skip_reason is not None:
            details["self_test_skipped_reason"] = skip_reason
        elif draft_self_eval is not None:
            details["self_test_evaluation"] = dict(draft_self_eval)
            details["self_test_summary"] = _self_test_summary(draft_self_eval)
        draft_step = replace(
            draft_step,
            output=ModelOutput(
                raw_output=draft_step.output.raw_output,
                cleaned_output=draft_code,
                completion_tokens=draft_step.output.completion_tokens,
                prompt_tokens=draft_step.output.prompt_tokens,
            ),
            evaluation=draft_self_eval,
            details=details,
        )
        if parse_error:
            draft_step.details["bundle_parse_error"] = parse_error

        if skip_self_test and not parse_error:
            official_eval = self.evaluator(task, draft_code, timeout_sec=evaluation_timeout_sec)
            draft_step = replace(
                draft_step,
                evaluation=official_eval,
                details={
                    **draft_step.details,
                    "final_official_evaluation": dict(official_eval),
                },
            )
            repair_evaluation = official_eval
        else:
            repair_evaluation = draft_self_eval or {}
            if not (skip_self_test and parse_error) and not bool((draft_self_eval or {}).get("passed")):
                repair_evaluation = draft_self_eval or {}
        can_attempt_repair = bool(
            not bool((draft_step.evaluation or {}).get("passed")) and (remaining_budget is None or remaining_budget > 0)
        )
        if not can_attempt_repair:
            return [draft_step]

        repair_step = self._run_generation_batch(
            [task],
            arm=arm,
            phase="repair_code",
            artifact_type="repair_code",
            evaluate_output=False,
            evaluation_timeout_sec=evaluation_timeout_sec,
            artifacts_by_task=[
                {
                    "previous_code": draft_code,
                    "self_tests": self_tests,
                    "compiled_tests": self_tests,
                    "tests": tests,
                    "error_log": _evaluation_error_log(
                        repair_evaluation,
                        failed_tests_heading=_failed_tests_heading(task),
                    ),
                }
            ],
            max_new_tokens_cap=remaining_budget,
        )[0][0]
        official_eval = self.evaluator(
            task,
            repair_step.output.cleaned_output,
            timeout_sec=evaluation_timeout_sec,
        )
        repair_step = replace(
            repair_step,
            evaluation=official_eval,
            details={
                **_a2_test_artifact_details(
                    task,
                    compiled_tests=self_tests,
                    tests=tests,
                ),
                "final_official_evaluation": dict(official_eval),
            },
        )
        return [draft_step, repair_step]

    def _evaluate_a2_draft_step(
        self,
        task: CodeGenTask,
        step: AttemptStep,
        *,
        evaluation_timeout_sec: int,
    ) -> tuple[str, str, str, dict[str, Any], str | None]:
        draft_code, self_tests, tests, parse_error = self._extract_a2_draft_bundle(task, step)
        if parse_error:
            return draft_code, self_tests, tests, _draft_failure_evaluation(parse_error), parse_error
        self_eval = evaluate_self_test_completion(
            task,
            draft_code,
            self_tests,
            timeout_sec=evaluation_timeout_sec,
            signature_compatible_named_tests=self.self_test_signature_compatible,
        )
        return draft_code, self_tests, tests, self_eval, None

    def _extract_a2_draft_bundle(
        self,
        task: CodeGenTask,
        step: AttemptStep,
    ) -> tuple[str, str, str, str | None]:
        raw_output = step.output.raw_output or ""
        try:
            bundle = extract_a2_draft_bundle(raw_output, task=task)
            return bundle.code, bundle.self_tests, bundle.tests, None
        except Exception as exc:
            return "", "", "", f"{type(exc).__name__}: {exc}"

    def _build_record(
        self,
        *,
        task: CodeGenTask,
        final_arm: PipelineArm,
        feature_result,
        steps: list[AttemptStep],
        pass_k: int,
        selection_reason: dict[str, Any],
        metadata: dict[str, Any],
        compute_reward: bool = True,
        completion_token_budget_override: int | None = None,
    ) -> GenerationRecord:
        final_step = steps[-1]
        final_evaluation = final_step.evaluation or {}
        prompt_tokens = sum(step.output.prompt_tokens for step in steps)
        completion_tokens = sum(step.output.completion_tokens for step in steps)
        accounted_completion_tokens = sum(self._step_accounted_completion_tokens(step) for step in steps)
        total_tokens = self._steps_total_token_cost(steps)
        knn_prompt_tokens = self._steps_knn_prompt_tokens(steps)
        knn_total_tokens = _knn_total_tokens(
            prompt_tokens_excluding_system=knn_prompt_tokens,
            completion_tokens=completion_tokens,
            reward_config=self.reward_config,
        )
        generation_duration_sec = sum(step.generation_duration_sec for step in steps)
        generation_durations_by_phase_sec: dict[str, float] = {}
        for step in steps:
            generation_durations_by_phase_sec[step.phase] = (
                generation_durations_by_phase_sec.get(step.phase, 0.0) + step.generation_duration_sec
            )
        generation_inference_durations_sec = [step.generation_duration_sec for step in steps]
        output_length = len(final_step.output.cleaned_output or "")
        passed = bool(final_evaluation.get("passed"))
        pass_at_k = estimate_pass_at_k(
            num_samples=1,
            num_correct=1 if passed else 0,
            pass_k=pass_k,
        )
        reference_completion_tokens = 0
        completion_token_budget = None
        reward_components: dict[str, Any] = {}
        reward_value: float | None = None
        predictions = [_attempt_to_prediction(step, reward_config=self.reward_config) for step in steps]
        metadata = dict(metadata)
        metadata["attempt_count"] = len(steps)
        metadata["workflow_phases"] = [step.phase for step in steps]
        metadata["workflow_major_calls"] = final_arm.max_major_calls
        if compute_reward:
            reference_source = _visible_task_budget_reference(task)
            reference_completion_tokens = self.model_client.count_tokens(reference_source or "")
            if completion_token_budget_override is not None:
                completion_token_budget = int(completion_token_budget_override)
            else:
                completion_token_budget = self._task_completion_budget(
                    reference_completion_tokens=reference_completion_tokens,
                )
            reward_components = _compute_reward_components(
                pass_at_k=pass_at_k,
                reference_completion_tokens=reference_completion_tokens,
                completion_tokens=completion_tokens,
                completion_token_budget=completion_token_budget,
                reward_config=self.reward_config,
                prompt_tokens=prompt_tokens,
                cost_tokens=total_tokens,
            )
            reward_value = reward_components["total_reward"]
            metadata["reward_version"] = reward_components["reward_version"]
            metadata["reward_reference_source"] = "visible_task_and_starter_code"
        metadata["token_accounting"] = {
            "include_prompt_tokens_in_cost": bool(getattr(self.reward_config, "include_prompt_tokens_in_cost", True)),
            "prompt_token_cost_weight": float(getattr(self.reward_config, "prompt_token_cost_weight", 1.0)),
            "completion_token_cost_weight": float(getattr(self.reward_config, "completion_token_cost_weight", 1.0)),
            "knn_prompt_tokens_excluding_system": knn_prompt_tokens,
            "knn_total_tokens": knn_total_tokens,
        }
        return GenerationRecord(
            task_id=task.task_id,
            arm_id=final_arm.arm_id,
            arm_name=final_arm.paper_name,
            model_name=self.model_client.model_name,
            model_path=getattr(self.model_client, "model_path", self.model_client.model_name),
            dataset_name=self.dataset_name,
            prompt=final_step.prompt,
            task=task,
            feature_names=(list(feature_result.names) if feature_result is not None else []),
            feature_vector=(list(feature_result.values) if feature_result is not None else []),
            predictions=predictions,
            num_samples=1,
            pass_k=pass_k,
            num_correct=1 if passed else 0,
            reference_completion_tokens=reference_completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            accounted_completion_tokens=accounted_completion_tokens,
            total_tokens=total_tokens,
            generation_duration_sec=generation_duration_sec,
            generation_durations_by_phase_sec=generation_durations_by_phase_sec,
            generation_inference_durations_sec=generation_inference_durations_sec,
            completion_token_budget=completion_token_budget,
            output_length=output_length,
            pass_at_k=pass_at_k,
            reward=reward_value,
            reward_components=reward_components,
            selection_reason=selection_reason,
            metadata=metadata,
        )


def _attempt_to_prediction(step: AttemptStep, *, reward_config: RewardConfig) -> dict[str, Any]:
    evaluation = step.evaluation or {}
    prediction = {
        "phase": step.phase,
        "arm_id": step.arm_id,
        "artifact_type": step.artifact_type,
        "raw_output": step.output.raw_output,
        "cleaned_output": evaluation.get("cleaned_output", step.output.cleaned_output),
        "executable_code": evaluation.get("executable_code"),
        "prompt_tokens": step.output.prompt_tokens,
        "completion_tokens": step.output.completion_tokens,
        "accounted_completion_tokens": _accounted_completion_tokens(
            completion_tokens=step.output.completion_tokens,
            reward_config=reward_config,
        ),
        "total_tokens": _accounted_total_tokens(
            prompt_tokens=step.output.prompt_tokens,
            completion_tokens=step.output.completion_tokens,
            reward_config=reward_config,
        ),
        "generation_duration_sec": step.generation_duration_sec,
        "passed": (bool(evaluation.get("passed")) if step.evaluation is not None else None),
        "exit_code": evaluation.get("exit_code"),
        "stdout": evaluation.get("stdout"),
        "stderr": evaluation.get("stderr"),
        "duration_sec": evaluation.get("duration_sec"),
    }
    if step.details:
        prediction["details"] = dict(step.details)
    return prediction


def _decision_to_dict(decision: BanditDecision) -> dict[str, Any]:
    payload = {
        "selected_arm": decision.arm_id,
        "scores": dict(decision.scores),
        "expected_rewards": dict(decision.expected_rewards),
        "uncertainty_bonuses": dict(decision.uncertainty_bonuses),
        "pull_counts": dict(decision.pull_counts),
    }
    if decision.details:
        payload["details"] = dict(decision.details)
    return payload


def _decision_ranked_arms(decision: BanditDecision, *, available_arm_ids: list[str]) -> list[str]:
    details = dict(decision.details or {})
    ranked_arms = details.get("ranked_arms")
    if isinstance(ranked_arms, list) and ranked_arms:
        ordered = [arm_id for arm_id in ranked_arms if arm_id in available_arm_ids]
        for arm_id in available_arm_ids:
            if arm_id not in ordered:
                ordered.append(arm_id)
        return ordered
    if decision.scores:
        ranked_by_score = [
            arm_id
            for arm_id, _ in sorted(
                decision.scores.items(),
                key=lambda item: (-item[1], available_arm_ids.index(item[0]) if item[0] in available_arm_ids else len(available_arm_ids)),
            )
            if arm_id in available_arm_ids
        ]
        for arm_id in available_arm_ids:
            if arm_id not in ranked_by_score:
                ranked_by_score.append(arm_id)
        return ranked_by_score
    return list(available_arm_ids)


def _selector_feedback(record: GenerationRecord, *, metric: str) -> float:
    if metric == "pass_at_k":
        return float(record.pass_at_k or 0.0)
    if metric == "reward":
        return float(record.reward or 0.0)
    raise ValueError(f"Unsupported selector feedback metric: {metric}")


def _phase_generation_config(
    base_config: GenerationConfig,
    *,
    tasks: list[CodeGenTask],
    arm: PipelineArm,
    phase: str,
    max_new_tokens_cap: int | None = None,
) -> GenerationConfig:
    phase_max_new_tokens = arm.max_new_tokens_for_phase(phase, default=base_config.max_new_tokens)
    phase_temperature = base_config.temperature
    phase_top_p = base_config.top_p
    if max_new_tokens_cap is not None:
        phase_max_new_tokens = max(1, min(phase_max_new_tokens, int(max_new_tokens_cap)))
    phase_stop = arm.stop_sequences_for_phase(phase)
    if (
        phase_max_new_tokens == base_config.max_new_tokens
        and tuple(base_config.stop) == tuple(phase_stop)
        and phase_temperature == base_config.temperature
        and phase_top_p == base_config.top_p
    ):
        return base_config
    return replace(
        base_config,
        max_new_tokens=phase_max_new_tokens,
        temperature=phase_temperature,
        top_p=phase_top_p,
        stop=tuple(phase_stop),
    )


def _artifact_payload(step: AttemptStep, *, arm: PipelineArm) -> str:
    text = step.output.cleaned_output or ""
    for stop in arm.stop_sequences_for_phase(step.phase):
        if stop:
            text = text.replace(stop, "")
    return text.strip()


def _evaluation_error_log(
    evaluation: dict[str, Any] | None,
    *,
    failed_tests_heading: str = "[Failed Self Tests]",
) -> str:
    evaluation = evaluation or {}
    lines = [
        f"Exit code: {evaluation.get('exit_code')}",
    ]
    summary = _self_test_summary(evaluation)
    if summary:
        num_tests = summary.get("num_tests")
        passed_tests = summary.get("passed_tests")
        failed_tests = summary.get("failed_tests")
        lines.extend(
            [
                f"Passed tests: {passed_tests}/{num_tests}",
                f"Failed tests: {failed_tests}/{num_tests}",
            ]
        )
        failed_names = [test.get("name") for test in summary.get("tests", []) if not test.get("passed")]
        if failed_names:
            lines.extend([failed_tests_heading, "\n".join(str(name) for name in failed_names if name)])
    lines.extend(
        [
        "[Stdout]",
        str(evaluation.get("stdout") or ""),
        "[Stderr]",
        str(evaluation.get("stderr") or ""),
        ]
    )
    return "\n".join(lines).strip()


def _failed_tests_heading(task: CodeGenTask) -> str:
    return "[Failed Self Tests]"


def _a2_test_artifact_details(
    task: CodeGenTask,
    *,
    compiled_tests: str,
    tests: str,
) -> dict[str, Any]:
    compiled_tests = str(compiled_tests or "").strip()
    tests = str(tests or "").strip()
    details: dict[str, Any] = {}
    if tests:
        details["tests"] = tests
    if compiled_tests:
        details["self_tests"] = compiled_tests
    return details


def _compiled_tests_from_details(details: dict[str, Any] | None) -> str:
    payload = dict(details or {})
    for key in ("compiled_tests", "self_tests", "tests"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def _self_test_summary(evaluation: dict[str, Any] | None) -> dict[str, Any]:
    evaluation = evaluation or {}
    if "num_tests" not in evaluation:
        return {}
    return {
        "num_tests": evaluation.get("num_tests"),
        "passed_tests": evaluation.get("passed_tests"),
        "failed_tests": evaluation.get("failed_tests"),
        "tests": list(evaluation.get("tests") or []),
    }



def _is_explicit_static_order(static_order: str) -> bool:
    return "-" in str(static_order)


def _static_arm_order(arm_ids: tuple[str, ...], static_order: str) -> list[str]:
    if _is_explicit_static_order(static_order):
        ordered_arms = [part for part in str(static_order).split("-") if part]
        if sorted(ordered_arms) != sorted(arm_ids):
            raise ValueError(f"--static-order {static_order} must be a permutation of arm_ids={arm_ids}.")
        return ordered_arms
    if static_order not in arm_ids:
        raise ValueError(f"--static-order {static_order} is not in arm_ids={arm_ids}.")
    base = list(arm_ids)
    offset = base.index(static_order)
    return base[offset:] + base[:offset]


def _static_prior_budget_cap(
    *,
    arm_id: str,
    remaining_ranked_arms: list[str],
    remaining_budget: int,
    prior_weights: dict[str, float],
) -> tuple[int, float, float]:
    arm_weight = max(float(prior_weights.get(arm_id, 1.0)), 0.0)
    prior_denominator = sum(
        max(float(prior_weights.get(candidate, 1.0)), 0.0)
        for candidate in remaining_ranked_arms
    )
    if prior_denominator <= 0.0:
        return int(remaining_budget), arm_weight, prior_denominator
    raw_cap = (arm_weight / prior_denominator) * float(remaining_budget)
    allocated_cap = int(raw_cap)
    if remaining_budget > 0 and allocated_cap <= 0:
        allocated_cap = 1
    return min(allocated_cap, int(remaining_budget)), arm_weight, prior_denominator


def _remaining_completion_budget(
    *,
    total_completion_token_budget: int | None,
    spent_completion_tokens: int,
    reserved_for_future_tasks: int,
) -> int | None:
    if total_completion_token_budget is None:
        return None
    return total_completion_token_budget - spent_completion_tokens - reserved_for_future_tasks


def _task_ids_for_log(tasks: list[CodeGenTask]) -> str:
    return ",".join(task.task_id for task in tasks)


def _draft_failure_evaluation(message: str) -> dict[str, Any]:
    return {
        "cleaned_output": "",
        "executable_code": "",
        "passed": False,
        "exit_code": 1,
        "stdout": "",
        "stderr": message,
        "duration_sec": 0.0,
    }


def _mark_bara_steps_over_cap_failure(
    steps: list[AttemptStep],
    *,
    allocated_cap: int,
    actual_total_tokens: int,
) -> list[AttemptStep]:
    if not steps:
        return steps
    message = (
        "BARA attempt exceeded allocated token cap: "
        f"actual_total_tokens={int(actual_total_tokens)} > allocated_cap={int(allocated_cap)}."
    )
    final_step = steps[-1]
    evaluation = dict(final_step.evaluation or {})
    stderr = str(evaluation.get("stderr") or "").strip()
    evaluation.update(
        {
            "cleaned_output": evaluation.get("cleaned_output", final_step.output.cleaned_output),
            "executable_code": evaluation.get("executable_code"),
            "passed": False,
            "exit_code": 1,
            "stderr": f"{stderr}\n{message}".strip(),
            "duration_sec": float(evaluation.get("duration_sec") or 0.0),
        }
    )
    details = {
        **dict(final_step.details),
        "bara_over_allocated_token_cap": True,
        "bara_allocated_completion_token_cap": int(allocated_cap),
        "bara_actual_total_tokens": int(actual_total_tokens),
        "bara_charged_token_usage": int(allocated_cap),
    }
    return [
        *steps[:-1],
        replace(
            final_step,
            evaluation=evaluation,
            details=details,
        ),
    ]


def _should_skip_a2_self_test(task: CodeGenTask) -> bool:
    return task.task_id in _A2_SELF_TEST_BYPASS_TASK_IDS


def _normalize_structured_output(
    *,
    arm: PipelineArm,
    phase: str,
    output: ModelOutput,
) -> tuple[ModelOutput, dict[str, Any]]:
    if arm.arm_id != "A1" or phase != "final_code":
        return output, {}
    try:
        bundle = extract_a1_final_bundle(output.cleaned_output)
    except ValueError as exc:
        return (
            replace(
                output,
                cleaned_output="",
            ),
            {"bundle_parse_error": str(exc)},
        )
    return (
        replace(
            output,
            cleaned_output=bundle.code,
        ),
        {"plan": bundle.plan},
    )
