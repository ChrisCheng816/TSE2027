from __future__ import annotations

import base64
import io
import json
import math
import pickle
import tempfile
import unittest
import zlib
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from statistics import NormalDist
from unittest.mock import patch
import importlib

from codegen_agent import cli
import codegen_agent.dataset as dataset_module
from codegen_agent.arms import (
    build_prompt_bundle,
    get_arm_choices,
    get_base_arms,
    get_partition_arms,
    get_upgrade_arms,
    normalize_arm_ids,
)
from codegen_agent.bara import KNNArmBudgetAllocator, KNNRewardSumUCBSelector
from codegen_agent.config import DEFAULT_EXPERIMENT_CONFIG, PathConfig, RewardConfig, get_model_choices, resolve_model_path
from codegen_agent.dataset import (
    get_dataset_choices,
    load_codegen_tasks,
    resolve_dataset_spec,
    write_jsonl,
)
from codegen_agent.features import BigCodeBenchFeatureExtractor, visible_task_context_text
from codegen_agent.runner import ExperimentRunner, _compute_reward_components, _phase_generation_config
from codegen_agent.workflow.rewarding import (
    compute_repair_decision_reward,
    reward_formula_description,
    reward_weight_dict,
)
from codegen_agent.types import BanditDecision, CodeGenTask, FeatureExtractionResult, GenerationConfig, ModelOutput, PromptBundle








def _task(
    task_id: str,
    prompt: str,
    *,
    code_prompt: str | None = None,
    entry_point: str | None = None,
    dataset_name: str = "toy-dataset",
    prompt_source: str = "prompt",
    instruct_prompt: str | None = None,
    doc_struct: dict[str, object] | str | None = None,
) -> CodeGenTask:
    return CodeGenTask(
        dataset_name=dataset_name,
        task_id=task_id,
        prompt=prompt,
        prompt_source=prompt_source,
        instruct_prompt=instruct_prompt,
        code_prompt=code_prompt,
        entry_point=entry_point,
        doc_struct=doc_struct,
        canonical_solution="def task_func(value):\n    return value\n",
        test="def check(candidate):\n    assert candidate(1) == 1\n",
    )


def _reward_config_with_version(
    reward_version: str,
    *,
    pass_log_cost_weight: float = 0.08,
    fail_log_cost_weight: float = 0.02,
    log_cost_cap: float = 4.0,
) -> RewardConfig:
    return replace(
        RewardConfig(),
        reward_version=reward_version,
        v2_pass_log_cost_weight=pass_log_cost_weight,
        v2_fail_log_cost_weight=fail_log_cost_weight,
        v2_log_cost_cap=log_cost_cap,
    )


class _FixedSelector:
    def __init__(self, arm_id: str) -> None:
        self.arm_id = arm_id
        self.updates: list[tuple[str, list[float], float]] = []

    def select(self, context: list[float]) -> BanditDecision:
        return BanditDecision(
            arm_id=self.arm_id,
            scores={self.arm_id: 1.0},
            expected_rewards={self.arm_id: 1.0},
            uncertainty_bonuses={self.arm_id: 0.0},
            pull_counts={self.arm_id: len(self.updates)},
        )

    def update(self, arm_id: str, context: list[float], reward: float) -> None:
        self.updates.append((arm_id, list(context), reward))


class _ScriptedModelClient:
    def __init__(self, scripted_outputs: list[str]) -> None:
        self.model_name = "scripted-model"
        self.model_path = "scripted-model"
        self._outputs = list(scripted_outputs)
        self.calls: list[tuple[str, int]] = []
        self.batch_prompt_counts: list[int] = []
        self.prompt_history: list[PromptBundle] = []
        self.max_new_tokens_history: list[int] = []
        self.stop_history: list[tuple[str, ...]] = []

    def generate(self, prompt: PromptBundle, config: GenerationConfig) -> str:
        raise NotImplementedError

    def generate_batch(
        self,
        prompts: list[PromptBundle],
        config: GenerationConfig,
        *,
        n: int = 1,
        seed: int | None = None,
    ) -> list[list[ModelOutput]]:
        del seed
        self.batch_prompt_counts.append(len(prompts))
        outputs: list[list[ModelOutput]] = []
        for prompt in prompts:
            self.prompt_history.append(prompt)
            self.max_new_tokens_history.append(config.max_new_tokens)
            self.stop_history.append(tuple(config.stop))
            prompt_outputs: list[ModelOutput] = []
            for _ in range(max(n, 1)):
                text = self._outputs.pop(0)
                self.calls.append((prompt.arm_id, len(text)))
                prompt_outputs.append(
                    ModelOutput(
                        raw_output=text,
                        cleaned_output=text,
                        completion_tokens=max(len(text.split()), 1),
                        prompt_tokens=max(len(prompt.raw_prompt.split()), 1),
                    )
                )
            outputs.append(prompt_outputs)
        return outputs

    def count_tokens(self, text: str) -> int:
        return max(len((text or "").split()), 1)


class KNNRewardSumUCBSelectorTests(unittest.TestCase):
    def test_failed_observation_is_retained_in_neighbor_history(self) -> None:
        selector = KNNRewardSumUCBSelector(
            arm_ids=["A0", "A2", "A1"],
            feature_dim=2,
            k=5,
            beta=0.0,
            reward_max=1.0,
            random_seed=0,
        )

        first_decision = selector.select([1.0, 0.0])
        self.assertEqual(0, first_decision.details["neighbor_count"])
        self.assertTrue(first_decision.details["cold_start_randomized"])

        selector.observe(
            "A2",
            [1.0, 0.0],
            0.0,
            solved=False,
            metadata={"task_id": "task-failed"},
        )

        second_decision = selector.select([1.0, 0.0])

        self.assertEqual({"A0": 0, "A2": 1, "A1": 0}, second_decision.pull_counts)
        self.assertEqual(1, second_decision.details["neighbor_count"])
        self.assertFalse(second_decision.details["cold_start_randomized"])
        self.assertTrue(second_decision.details["forced_exploration"])
        self.assertEqual("A0", second_decision.details["forced_arm_id"])

    def test_min_pulls_forced_exploration_randomizes_under_pulled_arm(self) -> None:
        selector = KNNRewardSumUCBSelector(
            arm_ids=["A0", "A2", "A1"],
            feature_dim=2,
            k=5,
            beta=0.0,
            reward_max=1.0,
            min_pulls_per_arm=1,
            random_seed=0,
        )
        selector.observe("A2", [1.0, 0.0], 1.0, solved=True, metadata={"task_id": "task-1"})

        decision = selector.select([1.0, 0.0])

        self.assertTrue(decision.details["forced_exploration"])
        self.assertEqual(["A0", "A1"], decision.details["forced_candidate_arm_ids"])
        self.assertEqual("A1", decision.arm_id)
        self.assertEqual("A1", decision.details["forced_arm_id"])
        self.assertCountEqual(["A0", "A1"], decision.details["ranked_arms"][:2])
        self.assertEqual("A2", decision.details["ranked_arms"][2])

    def test_rank_and_allocate_min_pulls_counts_first_attempts_not_later_attempts(self) -> None:
        selector = KNNRewardSumUCBSelector(
            arm_ids=["A0", "A2", "A1"],
            feature_dim=2,
            k=5,
            beta=0.0,
            reward_max=1.0,
            min_pulls_per_arm=1,
            random_seed=0,
        )
        selector.observe(
            "A0",
            [1.0, 0.0],
            1.0,
            solved=True,
            metadata={"task_id": "task-1", "bara": True, "bara_rank": 2},
        )

        decision = selector.select([1.0, 0.0])

        self.assertTrue(decision.details["forced_exploration"])
        self.assertEqual(["A0", "A2", "A1"], decision.details["forced_candidate_arm_ids"])
        self.assertEqual({"A0": 0, "A2": 0, "A1": 0}, decision.details["first_arm_pull_counts"])
        self.assertEqual({"A0": 1, "A2": 0, "A1": 0}, decision.details["observation_counts"])

    def test_knn_k_counts_unique_tasks_not_arm_attempts(self) -> None:
        selector = KNNRewardSumUCBSelector(
            arm_ids=["A0", "A2", "A1"],
            feature_dim=2,
            k=2,
            beta=0.0,
            reward_max=1.0,
            min_pulls_per_arm=1,
            random_seed=0,
        )
        selector.observe("A0", [1.0, 0.0], 0.1, solved=False, metadata={"task_id": "task-close"})
        selector.observe("A2", [1.0, 0.0], 0.9, solved=True, metadata={"task_id": "task-close"})
        selector.observe("A1", [0.9, 0.1], 0.8, solved=True, metadata={"task_id": "task-mid"})
        selector.observe("A0", [0.0, 1.0], 1.0, solved=True, metadata={"task_id": "task-far"})

        decision = selector.select([1.0, 0.0])

        self.assertEqual(2, decision.details["neighbor_count"])
        self.assertEqual({"A0": 1, "A2": 1, "A1": 1}, decision.details["local_reward_counts"])
        self.assertEqual({"task-close", "task-mid"}, set(decision.details["nearest_neighbor_task_ids"]))
        self.assertEqual(["A2", "A1", "A0"], decision.details["ranked_arms"])

    def test_knn_ranking_randomizes_all_tied_scores(self) -> None:
        selector = KNNRewardSumUCBSelector(
            arm_ids=["A0", "A2", "A1"],
            feature_dim=2,
            k=3,
            beta=0.0,
            reward_max=1.0,
            min_pulls_per_arm=1,
            random_seed=0,
        )
        selector.observe("A0", [1.0, 0.0], 1.0, solved=True, metadata={"task_id": "task-a0"})
        selector.observe("A2", [1.0, 0.0], 1.0, solved=True, metadata={"task_id": "task-a2"})
        selector.observe("A1", [1.0, 0.0], 1.0, solved=True, metadata={"task_id": "task-a1"})

        decision = selector.select([1.0, 0.0])

        self.assertFalse(decision.details["forced_exploration"])
        self.assertEqual({"A0": 1.0, "A2": 1.0, "A1": 1.0}, decision.scores)
        self.assertCountEqual(["A0", "A2", "A1"], decision.details["ranked_arms"])

    def test_knn_ranking_randomizes_only_tied_score_group(self) -> None:
        selector = KNNRewardSumUCBSelector(
            arm_ids=["A0", "A2", "A1"],
            feature_dim=2,
            k=3,
            beta=0.0,
            reward_max=1.0,
            min_pulls_per_arm=1,
            random_seed=1,
        )
        selector.observe("A0", [1.0, 0.0], 1.0, solved=True, metadata={"task_id": "task-a0"})
        selector.observe("A2", [1.0, 0.0], 0.0, solved=False, metadata={"task_id": "task-a2"})
        selector.observe("A1", [1.0, 0.0], 1.0, solved=True, metadata={"task_id": "task-a1"})

        decision = selector.select([1.0, 0.0])

        self.assertFalse(decision.details["forced_exploration"])
        self.assertEqual({"A0": 1.0, "A2": 0.0, "A1": 1.0}, decision.scores)
        self.assertCountEqual(["A0", "A1"], decision.details["ranked_arms"][:2])
        self.assertEqual("A2", decision.details["ranked_arms"][2])

    def test_min_pulls_per_arm_forces_exploration_before_knn_scoring(self) -> None:
        selector = KNNRewardSumUCBSelector(
            arm_ids=["A0", "A2", "A1"],
            feature_dim=2,
            k=5,
            beta=0.0,
            reward_max=1.0,
            min_pulls_per_arm=2,
            random_seed=0,
        )

        first_decision = selector.select([1.0, 0.0])
        self.assertEqual("A0", first_decision.arm_id)
        self.assertTrue(first_decision.details["cold_start_randomized"])

        selector.observe("A0", [1.0, 0.0], 1.0, solved=True, metadata={"task_id": "task-0"})

        second_decision = selector.select([1.0, 0.0])
        self.assertEqual("A0", second_decision.arm_id)
        self.assertTrue(second_decision.details["forced_exploration"])
        self.assertEqual("A0", second_decision.details["forced_arm_id"])

        selector.observe("A0", [1.0, 0.0], 1.0, solved=True, metadata={"task_id": "task-1"})

        third_decision = selector.select([1.0, 0.0])
        self.assertEqual("A1", third_decision.arm_id)
        self.assertTrue(third_decision.details["forced_exploration"])
        self.assertEqual("A1", third_decision.details["forced_arm_id"])

    def test_knn_ucb_uses_local_neighbor_counts_by_default(self) -> None:
        selector = KNNRewardSumUCBSelector(
            arm_ids=["A0", "A2", "A1"],
            feature_dim=2,
            k=2,
            beta=0.25,
            reward_max=1.0,
            min_pulls_per_arm=1,
            random_seed=0,
        )
        selector.observe("A0", [1.0, 0.0], 0.5, solved=False, metadata={"task_id": "near-1"})
        selector.observe("A0", [0.99, 0.01], 0.5, solved=False, metadata={"task_id": "near-2"})
        selector.observe("A2", [0.0, 1.0], 1.0, solved=True, metadata={"task_id": "far-a2"})
        selector.observe("A1", [0.1, 0.9], 1.0, solved=True, metadata={"task_id": "far-a1"})

        decision = selector.select([1.0, 0.0])

        self.assertFalse(decision.details["forced_exploration"])
        self.assertEqual("local", decision.details["ucb_count_scope"])
        self.assertEqual({"A0": 2, "A2": 0, "A1": 0}, decision.details["local_reward_counts"])
        self.assertEqual({"A0": 2, "A2": 0, "A1": 0}, decision.details["ucb_denominator_counts"])

    def test_knn_ucb_can_reproduce_legacy_global_first_counts(self) -> None:
        selector = KNNRewardSumUCBSelector(
            arm_ids=["A0", "A2", "A1"],
            feature_dim=2,
            k=2,
            beta=0.25,
            reward_max=1.0,
            min_pulls_per_arm=1,
            ucb_count_scope="global_first",
            random_seed=0,
        )
        selector.observe("A0", [1.0, 0.0], 0.5, solved=False, metadata={"task_id": "near-1"})
        selector.observe("A0", [0.99, 0.01], 0.5, solved=False, metadata={"task_id": "near-2"})
        selector.observe("A2", [0.0, 1.0], 1.0, solved=True, metadata={"task_id": "far-a2"})
        selector.observe("A1", [0.1, 0.9], 1.0, solved=True, metadata={"task_id": "far-a1"})

        decision = selector.select([1.0, 0.0])

        self.assertEqual("global_first", decision.details["ucb_count_scope"])
        self.assertEqual({"A0": 2, "A2": 1, "A1": 1}, decision.details["ucb_denominator_counts"])


class KNNArmBudgetAllocatorTests(unittest.TestCase):


    def test_equal_split_allocator_splits_remaining_budget_across_remaining_arms(self) -> None:
        allocator = KNNArmBudgetAllocator(
            arm_ids=["A0", "A2", "A1"],
            feature_dim=2,
            k=5,
            allocation_strategy="equal_split",
        )

        allocation = allocator.allocate(
            arm_id="A0",
            context=[1.0, 0.0],
            ranked_arms=["A0", "A2", "A1"],
            remaining_budget=9000,
        )

        self.assertEqual("equal_split", allocation["allocation_mode"])
        self.assertEqual(["A0", "A2", "A1"], allocation["remaining_ranked_arms"])
        self.assertEqual(3, allocation["remaining_arm_count"])
        self.assertEqual(3000, allocation["allocated_completion_token_cap"])

    def test_static_prior_allocator_splits_remaining_budget_by_prior_weights(self) -> None:
        allocator = KNNArmBudgetAllocator(
            arm_ids=["A0", "A2", "A1"],
            feature_dim=2,
            k=5,
            allocation_strategy="static_prior",
        )

        allocation = allocator.allocate(
            arm_id="A2",
            context=[1.0, 0.0],
            ranked_arms=["A2", "A0", "A1"],
            remaining_budget=9000,
        )

        self.assertEqual("static_prior", allocation["allocation_mode"])
        self.assertEqual(["A2", "A0", "A1"], allocation["remaining_ranked_arms"])
        self.assertEqual(3, allocation["remaining_arm_count"])
        self.assertEqual(4500, allocation["allocated_completion_token_cap"])
        self.assertEqual(2.0, allocation["prior_weight"])
        self.assertEqual(4.0, allocation["prior_denominator_weight"])

    def test_empirical_quantile_ignores_censored_budget_observations(self) -> None:
        allocator = KNNArmBudgetAllocator(
            arm_ids=["A0", "A2", "A1"],
            feature_dim=2,
            k=5,
            allocation_strategy="empirical_quantile",
            budget_quantile=0.80,
            min_budget_neighbors=2,
            arm_prior_weights={"A0": 1.0, "A2": 1.0, "A1": 1.0},
        )
        observations = [
            ("u1", 1000, None),
            ("u2", 1200, None),
            ("c1", 5000, 5000),
            ("c2", 6000, 6000),
        ]
        for task_id, token_usage, token_cap in observations:
            allocator.observe(
                "A2",
                [1.0, 0.0],
                token_usage=token_usage,
                token_cap=token_cap,
                metadata={"task_id": task_id},
            )

        allocation = allocator.allocate(
            arm_id="A2",
            context=[1.0, 0.0],
            ranked_arms=["A2", "A0", "A1"],
            remaining_budget=9000,
        )

        self.assertEqual("local_empirical_quantile", allocation["allocation_mode"])
        self.assertEqual(4, allocation["observed_count"])
        self.assertEqual(2, allocation["censored_count"])
        self.assertEqual(0.99, allocation["budget_quantile"])
        self.assertEqual(6000, allocation["allocated_completion_token_cap"])

    def test_mle_allocator_fits_lognormal_quantile(self) -> None:
        allocator = KNNArmBudgetAllocator(
            arm_ids=["A0", "A2", "A1"],
            feature_dim=2,
            k=3,
            allocation_strategy="mle",
            risk_tolerance=0.10,
            arm_prior_weights={"A0": 10.0, "A2": 0.1, "A1": 10.0},
        )
        token_usages = [1000, 2000, 4000]
        for index, token_usage in enumerate(token_usages):
            allocator.observe(
                "A2",
                [1.0, 0.0],
                token_usage=token_usage,
                token_cap=None,
                metadata={"task_id": f"mle-{index}"},
            )

        allocation = allocator.allocate(
            arm_id="A2",
            context=[1.0, 0.0],
            ranked_arms=["A2", "A0", "A1"],
            remaining_budget=10000,
        )

        log_costs = [math.log(value) for value in token_usages]
        log_mean = sum(log_costs) / len(log_costs)
        log_sigma = math.sqrt(sum((value - log_mean) ** 2 for value in log_costs) / len(log_costs))
        expected_cap = int(math.floor(math.exp(log_mean + NormalDist().inv_cdf(0.90) * log_sigma)))

        self.assertEqual("lognormal_mle", allocation["allocation_mode"])
        self.assertEqual(3, allocation["budget_evidence_count"])
        self.assertEqual(3, allocation["uncensored_count"])
        self.assertEqual(0, allocation["censored_count"])
        self.assertAlmostEqual(0.90, allocation["lognormal_quantile"])
        self.assertEqual(expected_cap, allocation["allocated_completion_token_cap"])
        self.assertEqual("lognormal_mle", allocation["lognormal_fit_method"])

    def test_mle_allocator_uses_right_censored_lognormal_likelihood(self) -> None:
        allocator = KNNArmBudgetAllocator(
            arm_ids=["A0", "A2", "A1"],
            feature_dim=2,
            k=3,
            allocation_strategy="mle",
            risk_tolerance=0.10,
            arm_prior_weights={"A0": 10.0, "A2": 0.1, "A1": 10.0},
        )
        allocator.observe(
            "A2",
            [1.0, 0.0],
            token_usage=1000,
            token_cap=2000,
            metadata={"task_id": "uncensored-0"},
        )
        allocator.observe(
            "A2",
            [1.0, 0.0],
            token_usage=1100,
            token_cap=2000,
            metadata={"task_id": "uncensored-1"},
        )
        allocator.observe(
            "A2",
            [1.0, 0.0],
            token_usage=5000,
            token_cap=5000,
            metadata={"task_id": "censored-0"},
        )

        allocation = allocator.allocate(
            arm_id="A2",
            context=[1.0, 0.0],
            ranked_arms=["A2", "A0", "A1"],
            remaining_budget=10000,
        )

        uncensored_log_costs = [math.log(1000), math.log(1100)]
        uncensored_log_mean = sum(uncensored_log_costs) / len(uncensored_log_costs)
        uncensored_log_sigma = math.sqrt(
            sum((value - uncensored_log_mean) ** 2 for value in uncensored_log_costs)
            / len(uncensored_log_costs)
        )
        uncensored_only_cap = math.exp(
            uncensored_log_mean + NormalDist().inv_cdf(0.90) * uncensored_log_sigma
        )

        self.assertEqual("lognormal_mle", allocation["allocation_mode"])
        self.assertEqual("lognormal_mle", allocation["lognormal_fit_method"])
        self.assertEqual(2, allocation["uncensored_count"])
        self.assertEqual(1, allocation["censored_count"])
        self.assertGreater(allocation["lognormal_raw_completion_token_cap"], uncensored_only_cap)

    def test_mle_allocator_uses_prior_floor_for_lognormal_cap(self) -> None:
        allocator = KNNArmBudgetAllocator(
            arm_ids=["A0", "A2", "A1"],
            feature_dim=2,
            k=3,
            allocation_strategy="mle",
            risk_tolerance=0.10,
            arm_prior_weights={"A0": 1.0, "A2": 2.0, "A1": 1.0},
        )
        for index, token_usage in enumerate([1000, 1000, 1000]):
            allocator.observe(
                "A2",
                [1.0, 0.0],
                token_usage=token_usage,
                token_cap=None,
                metadata={"task_id": f"low-cost-{index}"},
            )

        allocation = allocator.allocate(
            arm_id="A2",
            context=[1.0, 0.0],
            ranked_arms=["A2", "A0", "A1"],
            remaining_budget=10000,
        )

        self.assertEqual("lognormal_mle", allocation["allocation_mode"])
        self.assertLess(allocation["lognormal_raw_completion_token_cap"], 5000)
        self.assertEqual(True, allocation["prior_floor_applied"])
        self.assertEqual(5000, allocation["allocated_completion_token_cap"])


class _ContextBytesIO(io.BytesIO):
    def __enter__(self) -> "_ContextBytesIO":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def _evaluate_by_output(task: CodeGenTask, cleaned_output: str, *, timeout_sec: int) -> dict[str, object]:
    del task, timeout_sec
    passed = "GOOD" in cleaned_output
    stderr = "" if passed else "AssertionError: failed test"
    return {
        "cleaned_output": cleaned_output,
        "executable_code": cleaned_output,
        "passed": passed,
        "exit_code": 0 if passed else 1,
        "stdout": "",
        "stderr": stderr,
        "duration_sec": 0.01,
    }


class BARATests(unittest.TestCase):
    def test_prompt_assets_exist_in_dedicated_directory(self) -> None:
        root = Path("prompts")
        expected = [
            root / "__init__.py",
            root / "shared/__init__.py",
            root / "shared/a0.py",
            root / "shared/a1.py",
            root / "shared/a2.py",
            root / "a0/__init__.py",
            root / "a0/final_code.py",
            root / "a1/__init__.py",
            root / "a1/final_code.py",
            root / "a2/__init__.py",
            root / "a2/draft_bundle.py",
            root / "a2/repair_code.py",
        ]

        for path in expected:
            self.assertTrue(path.exists(), f"missing prompt asset: {path}")

    def test_prompt_loader_renders_file_backed_templates(self) -> None:
        prompt_loader = importlib.import_module("codegen_agent.workflow.prompt_loader")
        module = importlib.import_module("codegen_agent.prompts.a2.repair_code")

        self.assertTrue(hasattr(module, "PROMPT"))
        self.assertIsInstance(module.PROMPT, str)

        rendered = prompt_loader.render_prompt_template(
            "a2/repair_code.py",
            task_prompt="Solve it.",
            error_log="Exit code: 1\nStderr:\nAssertionError: failed test",
        )

        self.assertIn("repair the code", rendered.lower())
        self.assertIn("Exit code: 1", rendered)
        self.assertIn("[Task]", rendered)
        self.assertIn("Solve it.", rendered)

    def test_default_selector_feedback_uses_reward(self) -> None:
        self.assertEqual("reward", DEFAULT_EXPERIMENT_CONFIG.bara.selector_feedback_metric)

    def test_default_reward_version_is_v2(self) -> None:
        self.assertEqual("v2", DEFAULT_EXPERIMENT_CONFIG.reward.reward_version)

    def test_default_config_disables_warm_start(self) -> None:
        self.assertFalse(DEFAULT_EXPERIMENT_CONFIG.warm_start.enabled)

    def test_default_batch_size_is_eight(self) -> None:
        self.assertEqual(8, DEFAULT_EXPERIMENT_CONFIG.batch_size)

    def test_model_registry_includes_new_qwen_variants(self) -> None:
        model_choices = get_model_choices()

        self.assertIn("qwen2.5-14b", model_choices)
        self.assertIn("qwen2.5-7b", model_choices)
        self.assertEqual("Qwen/Qwen2.5-Coder-14B-Instruct", resolve_model_path("qwen2.5-14b"))
        self.assertEqual("Qwen/Qwen2.5-Coder-7B-Instruct", resolve_model_path("qwen2.5-7b"))






    def test_reward_config_defaults_lock_v2_constants(self) -> None:
        reward_config = RewardConfig()

        self.assertEqual("v2", reward_config.reward_version)
        self.assertEqual(0.01, reward_config.v2_pass_log_cost_weight)
        self.assertEqual(0.005, reward_config.v2_fail_log_cost_weight)
        self.assertEqual(-0.8, reward_config.v2_fail_base_reward)
        self.assertEqual(6.8, reward_config.v2_log_cost_cap)

    def test_warm_start_feedback_uses_cost_aware_reward(self) -> None:
        config = replace(DEFAULT_EXPERIMENT_CONFIG, reward=_reward_config_with_version("v1"))
        row = {
            "pass_at_k": 1.0,
            "reward": 0.25,
            "completion_tokens": 50,
            "reference_completion_tokens": 20,
            "completion_token_budget": 30,
        }

        self.assertAlmostEqual(13.0 / 15.0, cli._warm_start_reward_from_row(row, config=config))

    def test_warm_start_reward_recomputes_against_active_reward_version(self) -> None:
        row = {
            "pass_at_k": 1.0,
            "reward": 0.25,
            "reference_completion_tokens": 20,
            "completion_tokens": 50,
            "completion_token_budget": 30,
        }
        v1_config = replace(DEFAULT_EXPERIMENT_CONFIG, reward=_reward_config_with_version("v1"))
        v2_config = replace(DEFAULT_EXPERIMENT_CONFIG, reward=_reward_config_with_version("v2"))

        v1_reward = cli._warm_start_reward_from_row(row, config=v1_config)
        v2_reward = cli._warm_start_reward_from_row(row, config=v2_config)

        self.assertAlmostEqual(
            _compute_reward_components(
                pass_at_k=1.0,
                reference_completion_tokens=20,
                completion_tokens=50,
                completion_token_budget=30,
                reward_config=v1_config.reward,
            )["total_reward"],
            v1_reward,
        )
        self.assertAlmostEqual(
            _compute_reward_components(
                pass_at_k=1.0,
                reference_completion_tokens=20,
                completion_tokens=50,
                completion_token_budget=30,
                reward_config=v2_config.reward,
            )["total_reward"],
            v2_reward,
        )
        self.assertNotEqual(v1_reward, v2_reward)


    def test_repair_decision_reward_uses_reward_version(self) -> None:
        v1_config = _reward_config_with_version("v1")
        v2_config = RewardConfig()
        repair_completion_tokens = 15
        repair_completion_token_budget = 100
        normalized_cost = min(
            repair_completion_tokens / repair_completion_token_budget,
            v1_config.max_normalized_completion_cost,
        )
        log_cost = min(math.log1p(repair_completion_tokens), v2_config.v2_log_cost_cap)

        self.assertAlmostEqual(
            v1_config.success_reward - (v1_config.pass_cost_weight * normalized_cost),
            compute_repair_decision_reward(
                repaired_passed=True,
                repair_completion_tokens=repair_completion_tokens,
                repair_completion_token_budget=repair_completion_token_budget,
                reward_config=v1_config,
            ),
        )
        self.assertAlmostEqual(
            -(v1_config.fail_cost_weight * normalized_cost),
            compute_repair_decision_reward(
                repaired_passed=False,
                repair_completion_tokens=repair_completion_tokens,
                repair_completion_token_budget=repair_completion_token_budget,
                reward_config=v1_config,
            ),
        )
        self.assertAlmostEqual(
            v2_config.success_reward - (v2_config.v2_pass_log_cost_weight * log_cost),
            compute_repair_decision_reward(
                repaired_passed=True,
                repair_completion_tokens=repair_completion_tokens,
                repair_completion_token_budget=repair_completion_token_budget,
                reward_config=v2_config,
            ),
        )
        self.assertAlmostEqual(
            v2_config.v2_fail_base_reward - (v2_config.v2_fail_log_cost_weight * log_cost),
            compute_repair_decision_reward(
                repaired_passed=False,
                repair_completion_tokens=repair_completion_tokens,
                repair_completion_token_budget=repair_completion_token_budget,
                reward_config=v2_config,
            ),
        )
        self.assertNotAlmostEqual(
            compute_repair_decision_reward(
                repaired_passed=True,
                repair_completion_tokens=repair_completion_tokens,
                repair_completion_token_budget=repair_completion_token_budget,
                reward_config=v1_config,
            ),
            compute_repair_decision_reward(
                repaired_passed=True,
                repair_completion_tokens=repair_completion_tokens,
                repair_completion_token_budget=repair_completion_token_budget,
                reward_config=v2_config,
            ),
        )

    def test_default_arm_set_uses_three_mainline_arms(self) -> None:
        self.assertEqual(("A0", "A1", "A2"), get_arm_choices())
        self.assertEqual(("A0", "A1", "A2"), tuple(arm.arm_id for arm in get_base_arms()))
        self.assertEqual((), tuple(arm.arm_id for arm in get_upgrade_arms()))
        self.assertEqual(("A0", "A1", "A2"), normalize_arm_ids(["a0", "plan", "test-guided"]))

    def test_prompt_bundles_encode_distinct_reasoning_biases(self) -> None:
        task = _task(
            "prompt-task",
            "Implement task_func so it preserves sorted order and handles empty input.",
            code_prompt="def task_func(values):\n    pass\n",
            entry_point="task_func",
        )

        cheap = build_prompt_bundle(task, get_partition_arms(["A0"])[0])
        repair = build_prompt_bundle(task, get_partition_arms(["A2"])[0])

        self.assertTrue(cheap.user.startswith("Write the final Python code"))
        self.assertTrue(repair.user.startswith("Write the Python code and a small self-test suite"))
        self.assertIn("[Task]", cheap.user)
        self.assertIn(task.prompt, cheap.user)
        self.assertIn(task.prompt, repair.user)
        self.assertEqual("", cheap.system)
        self.assertEqual("", repair.system)
        self.assertEqual(["user"], [message.role for message in cheap.messages])
        self.assertEqual(["user"], [message.role for message in repair.messages])
        self.assertNotIn("[SYSTEM]", cheap.raw_prompt)
        self.assertNotIn("[SYSTEM]", repair.raw_prompt)
        self.assertIn("write the final python code", cheap.user.lower())
        self.assertIn("write the python code and a small self-test suite", repair.user.lower())
        self.assertEqual("draft_bundle", repair.phase)
        self.assertNotIn("structured chain-of-thought", repair.user.lower())
        self.assertNotIn("let's think step by step.", repair.user.lower())

    def test_bigcodebench_prompt_bundle_includes_arm_specific_system_prompt_for_deepseek_only(self) -> None:
        task = _task(
            "bigcode-system",
            "Implement task_func correctly.",
            code_prompt="def task_func(values):\n    pass\n",
            entry_point="task_func",
            dataset_name="bigcodebench-v0.1.4",
            instruct_prompt="Use the provided signature and solve the task.",
        )

        with (
            patch("codegen_agent.workflow.prompt_builder.prompt_template_path", return_value=Path(__file__)),
            patch("codegen_agent.workflow.prompt_builder.render_prompt_template", side_effect=["A1 system prompt", "[Current Step]\nSolve it"]),
        ):
            prompt = build_prompt_bundle(
                task,
                get_partition_arms(["A1"])[0],
                model_name="deepseek-v2-lite",
                phase="final_code",
                artifacts={},
            )

        self.assertEqual("A1 system prompt", prompt.system)
        self.assertEqual(["system", "user"], [message.role for message in prompt.messages])
        self.assertIn("[SYSTEM]", prompt.raw_prompt)
        self.assertIn("A1 system prompt", prompt.raw_prompt)
        self.assertIn("[USER]", prompt.raw_prompt)

    def test_bigcodebench_prompt_bundle_skips_system_prompt_for_non_deepseek_models(self) -> None:
        task = _task(
            "bigcode-no-system",
            "Implement task_func correctly.",
            code_prompt="def task_func(values):\n    pass\n",
            entry_point="task_func",
            dataset_name="bigcodebench-v0.1.4",
            instruct_prompt="Use the provided signature and solve the task.",
        )

        prompt = build_prompt_bundle(
            task,
            get_partition_arms(["A1"])[0],
            model_name="qwen2.5-32b",
            phase="final_code",
            artifacts={},
        )

        self.assertEqual("", prompt.system)
        self.assertEqual(["user"], [message.role for message in prompt.messages])
        self.assertNotIn("[SYSTEM]", prompt.raw_prompt)

    def test_repair_prompt_treats_previous_code_and_failure_log_as_guidance(self) -> None:
        task = _task(
            "artifact-guidance",
            "Implement task_func with correct ordering and edge-case handling.",
            code_prompt="def task_func(values):\n    pass\n",
            entry_point="task_func",
        )

        repair = build_prompt_bundle(
            task,
            get_partition_arms(["A2"])[0],
            phase="repair_code",
            artifacts={
                "previous_code": "def task_func(values):\n    return []",
                "error_log": "Exit code: 1\nStdout:\n\nStderr:\nAssertionError: expected [1], got []",
            },
        )

        self.assertIn(task.prompt, repair.user)
        self.assertIn("repair the code", repair.user.lower())
        self.assertIn("[Error Log]", repair.user)
        self.assertIn("AssertionError", repair.user)
        self.assertIn("follow the task specification exactly", repair.user.lower())
        self.assertIn("output only python code", repair.user.lower())

    def test_sg_draft_prompt_requests_code_and_self_tests(self) -> None:
        task = _task(
            "draft-bundle",
            "Implement task_func with correct ordering and edge-case handling.",
            code_prompt="def task_func(values):\n    pass\n",
            entry_point="task_func",
        )

        draft = build_prompt_bundle(
            task,
            get_partition_arms(["A2"])[0],
            phase="draft_bundle",
            artifacts={},
        )

        self.assertIn("write the python code and a small self-test suite", draft.user.lower())
        self.assertIn("[CODE]", draft.user)
        self.assertIn("[TESTS]", draft.user)
        self.assertIn("five to seven", draft.user.lower())
        self.assertIn("test_1", draft.user)
        self.assertIn("test_3", draft.user)
        self.assertIn("test_5", draft.user)
        self.assertIn("do not call the tests yourself", draft.user.lower())
        self.assertIn(task.prompt.strip(), draft.user)

    def test_a2_prompt_requests_code_and_self_tests_for_test_guided_repair(self) -> None:
        task = _task(
            "test-guided",
            "Implement task_func with correct ordering and edge-case handling.",
            code_prompt="def task_func(values):\n    pass\n",
            entry_point="task_func",
        )

        prompt = build_prompt_bundle(
            task,
            get_partition_arms(["A2"])[0],
            phase="draft_bundle",
            artifacts={},
        )

        self.assertIn("write the python code and a small self-test suite", prompt.user.lower())
        self.assertIn("[CODE]", prompt.user)
        self.assertIn("[TESTS]", prompt.user)
        self.assertIn("do not call the tests yourself", prompt.user.lower())
        self.assertIn(task.prompt.strip(), prompt.user)

    def test_a1_prompt_requests_explicit_plan_and_code_with_paper_aligned_structure(self) -> None:
        task = _task(
            "self-planning",
            "Implement task_func with correct ordering and edge-case handling.",
            code_prompt="def task_func(values):\n    pass\n",
            entry_point="task_func",
        )

        prompt = build_prompt_bundle(
            task,
            get_partition_arms(["A1"])[0],
            phase="final_code",
            artifacts={},
        )

        self.assertIn("planning phase", prompt.user.lower())
        self.assertIn("implementation phase", prompt.user.lower())
        self.assertIn("[PLAN]", prompt.user)
        self.assertIn("[CODE]", prompt.user)
        self.assertIn("numbered list", prompt.user.lower())
        self.assertIn("single, easily implementable sub-task", prompt.user.lower())
        self.assertIn("[Example Task 1]", prompt.user)
        self.assertIn("[Example Plan 1]", prompt.user)
        self.assertNotIn("def encrypt(s):", prompt.user)
        self.assertIn(task.prompt.strip(), prompt.user)

    def test_bigcodebench_a1_uses_dataset_specific_prompt(self) -> None:
        task = _task(
            "bigcodebench-a1",
            "Build a pandas DataFrame and preserve the return type exactly.",
            code_prompt="import pandas as pd\ndef task_func(rows):\n    pass\n",
            entry_point="task_func",
            dataset_name="bigcodebench-v0.1.4",
            instruct_prompt="Build a pandas DataFrame and preserve the return type exactly.",
        )

        prompt = build_prompt_bundle(
            task,
            get_partition_arms(["A1"])[0],
            phase="final_code",
            artifacts={},
        )

        self.assertIn("exactly 4 numbered checklist items", prompt.user.lower())
        self.assertNotIn("[Task Shape Guidance]", prompt.user)
        self.assertIn("copy only explicit evidence", prompt.user.lower())
        self.assertIn("write `unspecified`", prompt.user)
        self.assertNotIn("[Example Task 1]", prompt.user)
        self.assertNotIn("[Example Plan 1]", prompt.user)

    def test_bigcodebench_a1_prompt_requires_short_plan_and_code_priority(self) -> None:
        prompt_text = importlib.import_module("codegen_agent.prompts.a1.final_code").PROMPT

        self.assertIn("[PLAN]", prompt_text)
        self.assertIn("[CODE]", prompt_text)
        self.assertIn("exactly 4 numbered checklist items", prompt_text.lower())
        self.assertIn("one sentence", prompt_text.lower())
        self.assertIn("code correctness has higher priority", prompt_text.lower())
        self.assertNotIn("[Task Shape Guidance]", prompt_text)
        self.assertNotIn("```", prompt_text)

    def test_bigcodebench_a1_prompt_requires_contract_checklist(self) -> None:
        prompt_text = importlib.import_module("codegen_agent.prompts.a1.final_code").PROMPT

        self.assertIn("copy only explicit evidence", prompt_text.lower())
        self.assertIn("do not paraphrase", prompt_text.lower())
        self.assertIn("do not infer", prompt_text.lower())
        self.assertIn("exact return type", prompt_text.lower())
        self.assertIn("exact object shape", prompt_text.lower())
        self.assertIn("write `none explicit`", prompt_text)
        self.assertIn("write `unspecified`", prompt_text)
        self.assertIn("required names/imports", prompt_text.lower())

    def test_bigcodebench_a1_prompt_requires_evidence_only_side_effects_and_failures(self) -> None:
        prompt_text = importlib.import_module("codegen_agent.prompts.a1.final_code").PROMPT

        self.assertIn("side effects", prompt_text.lower())
        self.assertIn("only explicit observable side effects", prompt_text.lower())
        self.assertIn("failure contract", prompt_text.lower())
        self.assertIn("only explicit exceptions or error behavior", prompt_text.lower())
        self.assertIn("do not choose a specific exception type", prompt_text.lower())
        self.assertIn("raise an error", prompt_text.lower())

    def test_bigcodebench_a1_prompt_requires_defined_names_in_final_code(self) -> None:
        prompt_text = importlib.import_module("codegen_agent.prompts.a1.final_code").PROMPT

        self.assertIn("ensure every referenced external name is defined or imported", prompt_text.lower())
        self.assertIn("np", prompt_text)
        self.assertIn("pd", prompt_text)
        self.assertIn("Path", prompt_text)
        self.assertIn("plt", prompt_text)

    def test_bigcodebench_a1_prompt_avoids_invented_schema_requirements(self) -> None:
        prompt_text = importlib.import_module("codegen_agent.prompts.a1.final_code").PROMPT

        self.assertIn("do not infer extra schema", prompt_text.lower())
        self.assertIn("tag", prompt_text.lower())
        self.assertIn("validation requirements", prompt_text.lower())

    def test_bigcodebench_a1_prompt_avoids_assuming_python_binary_name(self) -> None:
        prompt_text = importlib.import_module("codegen_agent.prompts.a1.final_code").PROMPT

        self.assertIn("current interpreter", prompt_text.lower())
        self.assertIn("python binary", prompt_text.lower())

    def test_bigcodebench_a1_prompt_preserves_mapping_multiplicity(self) -> None:
        prompt_text = importlib.import_module("codegen_agent.prompts.a1.final_code").PROMPT

        self.assertIn("multiplicity/count semantics", prompt_text.lower())
        self.assertIn("mappings", prompt_text.lower())

    def test_bigcodebench_a1_prompt_uses_fixed_four_item_plan_checklist(self) -> None:
        prompt_text = importlib.import_module("codegen_agent.prompts.a1.final_code").PROMPT

        self.assertIn("exactly 4 numbered checklist items", prompt_text.lower())
        self.assertIn("interface", prompt_text)
        self.assertIn("side effects", prompt_text.lower())
        self.assertIn("return contract", prompt_text.lower())
        self.assertIn("failure contract", prompt_text.lower())

    def test_bigcodebench_a1_prompt_preserves_task_and_starter_names(self) -> None:
        prompt_text = importlib.import_module("codegen_agent.prompts.a1.final_code").PROMPT

        self.assertIn("starter scaffold", prompt_text.lower())
        self.assertIn("constants", prompt_text.lower())
        self.assertIn("column names", prompt_text.lower())
        self.assertIn("do not rename", prompt_text.lower())

    def test_bigcodebench_a1_prompt_prefers_sys_executable_for_python_subprocesses(self) -> None:
        prompt_text = importlib.import_module("codegen_agent.prompts.a1.final_code").PROMPT

        self.assertIn("sys.executable", prompt_text)

    def test_bigcodebench_a1_prompt_requires_exact_input_type_inference(self) -> None:
        prompt_text = importlib.import_module("codegen_agent.prompts.a1.final_code").PROMPT

        self.assertIn("input value types", prompt_text.lower())
        self.assertIn("counts", prompt_text.lower())
        self.assertIn("integers", prompt_text.lower())
        self.assertIn("lists", prompt_text.lower())

    def test_bigcodebench_a1_prompt_reuses_starter_constants_for_labels(self) -> None:
        prompt_text = importlib.import_module("codegen_agent.prompts.a1.final_code").PROMPT

        self.assertIn("reuse", prompt_text.lower())
        self.assertIn("column labels", prompt_text.lower())
        self.assertIn("default numeric labels", prompt_text.lower())

    def test_bigcodebench_a1_prompt_does_not_render_task_shape_guidance(self) -> None:
        task = _task(
            "bigcodebench-pandas",
            "Return a pandas DataFrame with the required columns.",
            code_prompt="import pandas as pd\ndef task_func(rows):\n    pass\n",
            entry_point="task_func",
            dataset_name="bigcodebench-v0.1.4",
            instruct_prompt="Return a pandas DataFrame with the required columns.",
        )

        prompt = build_prompt_bundle(task, get_partition_arms(["A1"])[0], phase="final_code", artifacts={})

        self.assertNotIn("[Task Shape Guidance]", prompt.user)
        self.assertNotIn("Preserve DataFrame column names, row ordering, and return type exactly.", prompt.user)

    def test_bigcodebench_a1_prompt_is_not_specialized_for_plotting_tasks(self) -> None:
        task = _task(
            "bigcodebench-plot",
            "Plot a histogram and return the axes object.",
            code_prompt="import matplotlib.pyplot as plt\ndef task_func(values):\n    pass\n",
            entry_point="task_func",
            dataset_name="bigcodebench-v0.1.4",
            instruct_prompt="Plot a histogram and return the axes object.",
        )

        prompt = build_prompt_bundle(task, get_partition_arms(["A1"])[0], phase="final_code", artifacts={})

        self.assertNotIn("If the task requires a plot or chart, create it rather than only computing intermediate data.", prompt.user)

    def test_bigcodebench_a1_prompt_is_not_specialized_for_io_tasks(self) -> None:
        task = _task(
            "bigcodebench-io",
            "Read a CSV file path, parse the rows, and write JSON output.",
            code_prompt="import csv\nimport json\ndef task_func(file_path):\n    pass\n",
            entry_point="task_func",
            dataset_name="bigcodebench-v0.1.4",
            instruct_prompt="Read a CSV file path, parse the rows, and write JSON output.",
        )

        prompt = build_prompt_bundle(task, get_partition_arms(["A1"])[0], phase="final_code", artifacts={})

        self.assertNotIn("Preserve required file, path, parsing, and serialization side effects exactly.", prompt.user)

    def test_bigcodebench_a1_prompt_is_not_specialized_for_datetime_tasks(self) -> None:
        task = _task(
            "bigcodebench-datetime",
            "Convert UTC timestamps into a target timezone and return the datetime values.",
            code_prompt="from datetime import datetime\ndef task_func(time_strings):\n    pass\n",
            entry_point="task_func",
            dataset_name="bigcodebench-v0.1.4",
            instruct_prompt="Convert UTC timestamps into a target timezone and return the datetime values.",
        )

        prompt = build_prompt_bundle(task, get_partition_arms(["A1"])[0], phase="final_code", artifacts={})

        self.assertNotIn("Preserve timezone conversion, datetime parsing, and time-difference semantics exactly.", prompt.user)

    def test_bigcodebench_a1_prompt_is_not_specialized_for_network_tasks(self) -> None:
        task = _task(
            "bigcodebench-network",
            "Fetch a URL and send the parsed response over a socket.",
            code_prompt="import requests\nimport socket\ndef task_func(url):\n    pass\n",
            entry_point="task_func",
            dataset_name="bigcodebench-v0.1.4",
            instruct_prompt="Fetch a URL and send the parsed response over a socket.",
        )

        prompt = build_prompt_bundle(task, get_partition_arms(["A1"])[0], phase="final_code", artifacts={})

        self.assertNotIn("Preserve required request, socket, subprocess, or process side effects exactly.", prompt.user)

    def test_bigcodebench_a1_prompt_is_not_specialized_for_random_tasks(self) -> None:
        task = _task(
            "bigcodebench-random",
            "Randomly sample rows without replacement and keep the expected sample size.",
            code_prompt="import random\ndef task_func(rows):\n    pass\n",
            entry_point="task_func",
            dataset_name="bigcodebench-v0.1.4",
            instruct_prompt="Randomly sample rows without replacement and keep the expected sample size.",
        )

        prompt = build_prompt_bundle(task, get_partition_arms(["A1"])[0], phase="final_code", artifacts={})

        self.assertNotIn("Preserve sampling rules such as replacement, uniqueness, ordering, and randomness constraints exactly.", prompt.user)

    def test_extract_a1_final_bundle_splits_plan_and_code(self) -> None:
        a1_artifacts = importlib.import_module("codegen_agent.workflow.a1_artifacts")

        bundle = a1_artifacts.extract_a1_final_bundle(
            "[PLAN]\n"
            "1. Check the input.\n"
            "2. Return the transformed result.\n\n"
            "[CODE]\n"
            "def task_func(value):\n"
            "    return value\n"
        )

        self.assertEqual("1. Check the input.\n2. Return the transformed result.", bundle.plan)
        self.assertEqual("def task_func(value):\n    return value", bundle.code)

    def test_extract_a1_final_bundle_uses_fenced_code_before_trailing_explanation(self) -> None:
        a1_artifacts = importlib.import_module("codegen_agent.workflow.a1_artifacts")

        bundle = a1_artifacts.extract_a1_final_bundle(
            "[PLAN]\n"
            "1. Check the input.\n"
            "2. Return the transformed result.\n\n"
            "[CODE]\n"
            "```python\n"
            "def task_func(value):\n"
            "    return value\n"
            "```\n\n"
            "This code follows the plan by returning the value.\n"
        )

        self.assertEqual("def task_func(value):\n    return value", bundle.code)

    def test_extract_a1_final_bundle_rejects_plan_without_numbered_steps(self) -> None:
        a1_artifacts = importlib.import_module("codegen_agent.workflow.a1_artifacts")

        with self.assertRaises(ValueError):
            a1_artifacts.extract_a1_final_bundle(
                "[PLAN]\n"
                "Check the input.\n"
                "Return the transformed result.\n\n"
                "[CODE]\n"
                "def task_func(value):\n"
                "    return value\n"
            )

    def test_extract_sg_draft_bundle_splits_code_and_self_tests(self) -> None:
        a2_artifacts = importlib.import_module("codegen_agent.workflow.a2_artifacts")

        bundle = a2_artifacts.extract_a2_draft_bundle(
            "[CODE]\n"
            "def task_func(value):\n"
            "    return value\n\n"
            "[SELF_TESTS]\n"
            "def check(candidate):\n"
            "    assert candidate(1) == 1\n"
        )

        self.assertEqual("def task_func(value):\n    return value", bundle.code)
        self.assertEqual("def check(candidate):\n    assert candidate(1) == 1", bundle.self_tests)

    def test_extract_sg_draft_bundle_strips_trailing_code_fence_from_self_tests(self) -> None:
        a2_artifacts = importlib.import_module("codegen_agent.workflow.a2_artifacts")

        bundle = a2_artifacts.extract_a2_draft_bundle(
            "[CODE]\n"
            "def task_func(value):\n"
            "    return value\n\n"
            "[SELF_TESTS]\n"
            "def test_1(candidate):\n"
            "    assert candidate(1) == 1\n"
            "```\n"
        )

        self.assertEqual("def task_func(value):\n    return value", bundle.code)
        self.assertEqual("def test_1(candidate):\n    assert candidate(1) == 1", bundle.self_tests)

    def test_extract_sg_draft_bundle_accepts_python_tests_section_as_self_tests(self) -> None:
        a2_artifacts = importlib.import_module("codegen_agent.workflow.a2_artifacts")
        task = _task("bigcodebench-1", "Return the value.", dataset_name="bigcodebench-v0.1.4")

        bundle = a2_artifacts.extract_a2_draft_bundle(
            "[CODE]\n"
            "def task_func(value):\n"
            "    return value\n\n"
            "[TESTS]\n"
            "def test_1(candidate):\n"
            "    assert candidate(1) == 1\n",
            task=task,
        )

        self.assertEqual("def task_func(value):\n    return value", bundle.code)
        self.assertEqual("def test_1(candidate):\n    assert candidate(1) == 1", bundle.self_tests)
        self.assertEqual("def test_1(candidate):\n    assert candidate(1) == 1", bundle.tests)


    def test_extract_sg_draft_bundle_rejects_invalid_code_section_after_cleanup(self) -> None:
        a2_artifacts = importlib.import_module("codegen_agent.workflow.a2_artifacts")

        with self.assertRaises(ValueError):
            a2_artifacts.extract_a2_draft_bundle(
                "[CODE]\n"
                "def task_func(value)\n"
                "    return value\n\n"
                "[SELF_TESTS]\n"
                "def test_1(candidate):\n"
                "    assert candidate(1) == 1\n"
            )

    def test_repair_prompt_includes_self_tests_and_failure_log(self) -> None:
        task = _task(
            "repair-self-tests",
            "Implement task_func with correct ordering and edge-case handling.",
            code_prompt="def task_func(values):\n    pass\n",
            entry_point="task_func",
        )

        repair = build_prompt_bundle(
            task,
            get_partition_arms(["A2"])[0],
            phase="repair_code",
            artifacts={
                "previous_code": "def task_func(values):\n    return []",
                "self_tests": "def check(candidate):\n    assert candidate([1]) == [1]",
                "error_log": "Exit code: 1\nStdout:\n\nStderr:\nAssertionError: expected [1], got []",
            },
        )

        self.assertIn("[Self Tests]", repair.user)
        self.assertIn("assert candidate([1]) == [1]", repair.user)
        self.assertIn("AssertionError", repair.user)

    def test_bigcodebench_prompt_bundle_uses_instruct_prompt_verbatim_as_user_message(self) -> None:
        raw_instruct_prompt = (
            "Solve the task exactly as written.\n"
            "Do not change the public interface.\n"
            "Starter code:\n"
            "```python\n"
            "def task_func(x):\n"
            "    pass\n"
            "```"
        )
        task = _task(
            "bigcodebench-prompt",
            "fallback complete prompt that must not be used",
            dataset_name="bigcodebench-v0.1.4",
            prompt_source="instruct_prompt",
            instruct_prompt=raw_instruct_prompt,
            code_prompt="def task_func(x):\n    pass\n",
            entry_point="task_func",
        )

        specification = build_prompt_bundle(
            task,
            get_partition_arms(["A0"])[0],
            phase="final_code",
            artifacts={},
        )

        self.assertTrue(specification.user.startswith("[Current Step]"))
        self.assertTrue(specification.user.endswith(raw_instruct_prompt))
        self.assertNotIn("fallback complete prompt", specification.user)
        self.assertEqual(specification.system, build_prompt_bundle(task, get_partition_arms(["A0"])[0]).system)
        self.assertIn("Write the final Python code", specification.user)
        self.assertIn("[Task]", specification.user)

    def test_dataset_registry_includes_bigcodebench_hard(self) -> None:
        self.assertIn("bigcodebench-hard-v0.1.4", get_dataset_choices())
        dataset_spec = resolve_dataset_spec("bigcodebench-hard-v0.1.4")
        self.assertEqual("bigcode/bigcodebench-hard", dataset_spec.hf_dataset_path)
        self.assertEqual("v0.1.4", dataset_spec.hf_split)
        self.assertEqual(("instruct_prompt", "complete_prompt"), dataset_spec.prompt_fields)

    def test_bigcodebench_hard_prompt_bundle_uses_instruct_prompt_verbatim_as_user_message(self) -> None:
        raw_instruct_prompt = (
            "Solve the hard task exactly as written.\n"
            "Do not change the public interface.\n"
            "Starter code:\n"
            "```python\n"
            "def task_func(x):\n"
            "    pass\n"
            "```"
        )
        task = _task(
            "bigcodebench-hard-prompt",
            "fallback complete prompt that must not be used",
            dataset_name="bigcodebench-hard-v0.1.4",
            prompt_source="instruct_prompt",
            instruct_prompt=raw_instruct_prompt,
            code_prompt="def task_func(x):\n    pass\n",
            entry_point="task_func",
        )

        specification = build_prompt_bundle(
            task,
            get_partition_arms(["A0"])[0],
            phase="final_code",
            artifacts={},
        )

        self.assertTrue(specification.user.startswith("[Current Step]"))
        self.assertTrue(specification.user.endswith(raw_instruct_prompt))
        self.assertNotIn("fallback complete prompt", specification.user)

    def test_a2_draft_prompt_includes_doc_examples_without_forcing_example_derived_test(self) -> None:
        task = _task(
            "bigcodebench-doc-examples",
            "fallback complete prompt that must not be used",
            dataset_name="bigcodebench-v0.1.4",
            prompt_source="instruct_prompt",
            instruct_prompt="Write task_func exactly as specified.",
            code_prompt="def task_func(x):\n    pass\n",
            entry_point="task_func",
            doc_struct=json.dumps(
                {
                    "description": ["Return a float."],
                    "examples": [
                        ">>> result = task_func([1, 2, 3])",
                        ">>> isinstance(result, float)",
                        "True",
                    ],
                }
            ),
        )

        draft = build_prompt_bundle(
            task,
            get_partition_arms(["A2"])[0],
            phase="draft_bundle",
            artifacts={},
        )

        self.assertIn("[Doc Examples]", draft.user)
        self.assertIn(">>> result = task_func([1, 2, 3])", draft.user)
        self.assertIn(">>> isinstance(result, float)", draft.user)
        self.assertNotIn("at least 1 self-test", draft.user.lower())
        self.assertNotIn("documented example", draft.user.lower())
        self.assertIn("explicit in [task], [doc examples], or starter code", draft.user.lower())
        self.assertLess(draft.user.index("[Doc Examples]"), draft.user.rindex("[Task]"))

    def test_bigcodebench_hard_a2_draft_prompt_includes_doc_examples(self) -> None:
        task = _task(
            "bigcodebench-hard-doc-examples",
            "fallback complete prompt that must not be used",
            dataset_name="bigcodebench-hard-v0.1.4",
            prompt_source="instruct_prompt",
            instruct_prompt="Write task_func exactly as specified.",
            code_prompt="def task_func(x):\n    pass\n",
            entry_point="task_func",
            doc_struct=json.dumps(
                {
                    "description": ["Return a float."],
                    "examples": [
                        ">>> result = task_func([1, 2, 3])",
                        ">>> isinstance(result, float)",
                        "True",
                    ],
                }
            ),
        )

        draft = build_prompt_bundle(
            task,
            get_partition_arms(["A2"])[0],
            phase="draft_bundle",
            artifacts={},
        )

        self.assertIn("[Doc Examples]", draft.user)
        self.assertIn(">>> result = task_func([1, 2, 3])", draft.user)
        self.assertIn("documented example", draft.user.lower())

    def test_a2_draft_prompt_omits_doc_examples_section_when_examples_missing(self) -> None:
        task = _task(
            "bigcodebench-no-doc-examples",
            "fallback complete prompt that must not be used",
            dataset_name="bigcodebench-v0.1.4",
            prompt_source="instruct_prompt",
            instruct_prompt="Write task_func exactly as specified.",
            code_prompt="def task_func(x):\n    pass\n",
            entry_point="task_func",
            doc_struct=json.dumps({"description": ["Return a float."], "examples": []}),
        )

        draft = build_prompt_bundle(
            task,
            get_partition_arms(["A2"])[0],
            phase="draft_bundle",
            artifacts={},
        )

        self.assertNotIn("\n\n[Doc Examples]\n", draft.user)






























    def test_bigcodebench_a2_draft_prompt_still_uses_doc_examples_language(self) -> None:
        task = _task(
            "bigcodebench-still-doc-examples",
            "fallback complete prompt that must not be used",
            dataset_name="bigcodebench-v0.1.4",
            prompt_source="instruct_prompt",
            instruct_prompt="Write task_func exactly as specified.",
            code_prompt="def task_func(x):\n    pass\n",
            entry_point="task_func",
            doc_struct=json.dumps({"examples": ["example input", "example output"]}),
        )

        draft = build_prompt_bundle(
            task,
            get_partition_arms(["A2"])[0],
            phase="draft_bundle",
            artifacts={},
        )

        self.assertIn("[Doc Examples]", draft.user)
        self.assertIn("explicit in [task], [doc examples], or starter code", draft.user.lower())
        self.assertNotIn("documented example", draft.user.lower())

    def test_bigcodebench_a2_draft_prompt_uses_fewer_more_conservative_self_tests(self) -> None:
        task = _task(
            "bigcodebench-a2-selftest-conservative",
            "fallback complete prompt that must not be used",
            dataset_name="bigcodebench-v0.1.4",
            prompt_source="instruct_prompt",
            instruct_prompt="Write task_func exactly as specified.",
            code_prompt="def task_func(x):\n    pass\n",
            entry_point="task_func",
            doc_struct=json.dumps({"examples": ["example input", "example output"]}),
        )

        draft = build_prompt_bundle(
            task,
            get_partition_arms(["A2"])[0],
            phase="draft_bundle",
            artifacts={},
        )

        self.assertIn("Write five to seven self test cases", draft.user)
        self.assertNotIn("Write 5-7 focused self-tests", draft.user)
        self.assertNotIn("Cover 2 core specification behaviors, 2 boundary/error cases, and 1 tricky", draft.user)
        self.assertNotIn("at least 1 self-test must directly encode 1 documented example", draft.user)

    def test_bigcodebench_a2_draft_prompt_forbids_guessing_resources_and_illegal_self_tests(self) -> None:
        task = _task(
            "bigcodebench-a2-selftest-safety",
            "fallback complete prompt that must not be used",
            dataset_name="bigcodebench-v0.1.4",
            prompt_source="instruct_prompt",
            instruct_prompt="Write task_func exactly as specified.",
            code_prompt="def task_func(x):\n    pass\n",
            entry_point="task_func",
        )

        draft = build_prompt_bundle(
            task,
            get_partition_arms(["A2"])[0],
            phase="draft_bundle",
            artifacts={},
        )

        self.assertIn("Do not assume external files, directories, network access, processes, environment variables", draft.user)
        self.assertIn("If a behavior would require guessing hidden resources", draft.user)
        self.assertIn("[TESTS] must be valid Python", draft.user)
        self.assertIn("only top-level test functions", draft.user)

    def test_prompt_bundle_avoids_duplicate_starter_code_when_prompt_already_contains_scaffold(self) -> None:
        task = _task(
            "prompt-dedup",
            "Solve the task.\nYou should write self-contained code starting with:\n```\nimport math\n\ndef task_func(x):\n```\n",
            code_prompt="import math\n\ndef task_func(x):\n",
            entry_point="task_func",
        )

        cheap = build_prompt_bundle(task, get_partition_arms(["A0"])[0])

        self.assertEqual(1, cheap.user.count("import math"))
        self.assertEqual(0, cheap.user.count("Starter code:"))
        self.assertEqual("", cheap.system)
        self.assertNotIn("[SYSTEM]", cheap.raw_prompt)


    def test_repair_prompt_inlines_previous_code_into_single_user_message(self) -> None:
        task = _task(
            "multiturn-history",
            "Implement task_func using dynamic programming.",
            code_prompt="def task_func(value):\n    pass\n",
            entry_point="task_func",
        )

        prompt = build_prompt_bundle(
            task,
            get_partition_arms(["A2"])[0],
            phase="repair_code",
            artifacts={
                "previous_code": "def task_func(value):\n    return None",
                "error_log": "Exit code: 1\nStdout:\n\nStderr:\nAssertionError: expected 3, got None",
            },
        )

        self.assertEqual(
            ["user"],
            [message.role for message in prompt.messages],
        )
        self.assertIn("repair the code", prompt.messages[0].content.lower())
        self.assertIn("[Previous Code]", prompt.messages[0].content)
        self.assertIn("def task_func(value):\n    return None", prompt.messages[0].content)
        self.assertLess(
            prompt.messages[0].content.index("[Previous Code]"),
            prompt.messages[0].content.index("[Self Tests]"),
        )
        self.assertIn("AssertionError: expected 3, got None", prompt.messages[0].content)
        self.assertIn("[Task]", prompt.messages[0].content)

    def test_extract_returns_generalized_visible_context_features(self) -> None:
        extractor = BigCodeBenchFeatureExtractor(embedding_dim=8)
        task = _task(
            "feature-task",
            "Implement task_func. Examples:\nInput: [3, 2, 1]\nOutput: [1, 2, 3]\n"
            "Preserve sorted order, reject invalid values, and handle empty input. "
            "Read data from a file path, update the dataframe in place, "
            "and use dynamic programming with a heap-based frontier when helpful while keeping time complexity low.",
            code_prompt="import math\n\ndef task_func(values):\n    pass\n",
            entry_point="task_func",
        )

        result = extractor.extract(task)
        feature_map = dict(zip(result.names, result.values))

        self.assertEqual(len(result.names), len(result.values))
        self.assertEqual(17 + 8, len(result.names))
        self.assertIn("prompt_char_k_norm", feature_map)
        self.assertIn("example_count_norm", feature_map)
        self.assertIn("spec_term_count_norm", feature_map)
        self.assertIn("formatting_term_count_norm", feature_map)
        self.assertIn("io_term_count_norm", feature_map)
        self.assertIn("api_surface_term_count_norm", feature_map)
        self.assertIn("algorithm_term_count_norm", feature_map)
        self.assertIn("state_term_count_norm", feature_map)
        self.assertIn("data_structure_term_count_norm", feature_map)
        self.assertIn("complexity_term_count_norm", feature_map)
        self.assertNotIn("signature_arg_count_norm", feature_map)
        self.assertNotIn("has_entry_point", feature_map)
        self.assertNotIn("starter_code_present", feature_map)
        self.assertNotIn("mutation_side_effect_hint_flag", feature_map)
        self.assertIn("text_embedding_000", feature_map)
        self.assertIn("text_embedding_007", feature_map)
        self.assertGreater(feature_map["example_count_norm"], 0.0)
        self.assertGreater(feature_map["spec_term_count_norm"], 0.0)
        self.assertGreater(feature_map["formatting_term_count_norm"], 0.0)
        self.assertGreater(feature_map["io_term_count_norm"], 0.0)
        self.assertGreater(feature_map["api_surface_term_count_norm"], 0.0)
        self.assertGreater(feature_map["algorithm_term_count_norm"], 0.0)
        self.assertGreater(feature_map["state_term_count_norm"], 0.0)
        self.assertGreater(feature_map["data_structure_term_count_norm"], 0.0)
        self.assertGreater(feature_map["complexity_term_count_norm"], 0.0)

    def test_default_embedding_backend_is_qwen(self) -> None:
        self.assertEqual("qwen", DEFAULT_EXPERIMENT_CONFIG.embedding.backend)
        self.assertEqual(False, DEFAULT_EXPERIMENT_CONFIG.embedding.include_structured_features)
        self.assertEqual("3", DEFAULT_EXPERIMENT_CONFIG.embedding.qwen_cuda_visible_devices)
        self.assertEqual(0.1, DEFAULT_EXPERIMENT_CONFIG.embedding.qwen_gpu_memory_utilization)

    def test_extract_with_none_embedding_backend_returns_structured_features_only(self) -> None:
        task = _task(
            "structured-only-feature-task",
            "Implement task_func and return a list.",
            code_prompt="def task_func(values):\n    pass\n",
            entry_point="task_func",
        )

        extractor = BigCodeBenchFeatureExtractor(embedding_backend="none")
        result = extractor.extract(task)

        self.assertEqual(17, len(result.names))
        self.assertEqual(BigCodeBenchFeatureExtractor.STRUCTURED_FEATURE_NAMES, result.names)
        self.assertNotIn("text_embedding_000", result.names)

    def test_build_feature_extractor_disables_structured_features_by_default(self) -> None:
        class _StubEncoder:
            dim = 3

            def encode(self, text: str) -> list[float]:
                self.last_text = text
                return [0.25, -0.5, 0.75]

        task = _task(
            "default-qwen-feature-task",
            "Implement task_func and preserve ordering.",
            code_prompt="def task_func(values):\n    pass\n",
            entry_point="task_func",
        )

        with patch("codegen_agent.features.build_text_encoder", return_value=_StubEncoder()):
            extractor = cli.build_feature_extractor(DEFAULT_EXPERIMENT_CONFIG)
            result = extractor.extract(task)

        self.assertEqual(3, len(result.names))
        self.assertEqual(["text_embedding_000", "text_embedding_001", "text_embedding_002"], result.names)
        self.assertEqual([0.25, -0.5, 0.75], result.values)

    def test_extract_with_custom_text_encoder_appends_model_embedding_values(self) -> None:
        class _StubEncoder:
            dim = 3

            def encode(self, text: str) -> list[float]:
                self.last_text = text
                return [0.25, -0.5, 0.75]

        task = _task(
            "qwen-feature-task",
            "Implement task_func and preserve ordering.",
            code_prompt="def task_func(values):\n    pass\n",
            entry_point="task_func",
        )

        encoder = _StubEncoder()
        extractor = BigCodeBenchFeatureExtractor(text_encoder=encoder, embedding_backend="qwen")
        result = extractor.extract(task)

        self.assertEqual(17 + 3, len(result.names))
        self.assertEqual([0.25, -0.5, 0.75], result.values[-3:])
        self.assertEqual(
            ["text_embedding_000", "text_embedding_001", "text_embedding_002"],
            result.names[-3:],
        )
        self.assertIn("preserve ordering", encoder.last_text.lower())

    def test_extract_repair_context_includes_generic_failure_signals(self) -> None:
        extractor = BigCodeBenchFeatureExtractor(embedding_dim=8)
        task = _task(
            "repair-feature-task",
            "Implement task_func and preserve ordering.",
            code_prompt="def task_func(values):\n    pass\n",
            entry_point="task_func",
        )

        result = extractor.extract_repair_context(
            task,
            draft_output="def task_func(values):\n    return []\n",
            evaluation={
                "passed": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": "Traceback (most recent call last):\nAssertionError: expected [1], got []",
            },
            completion_tokens=17,
        )
        feature_map = dict(zip(result.names, result.values))

        self.assertIn("draft_completion_tokens_norm", feature_map)
        self.assertIn("failure_assertion_flag", feature_map)
        self.assertIn("failure_traceback_flag", feature_map)
        self.assertIn("failure_message_embedding_000", feature_map)
        self.assertGreater(feature_map["draft_completion_tokens_norm"], 0.0)
        self.assertEqual(1.0, feature_map["failure_assertion_flag"])
        self.assertEqual(1.0, feature_map["failure_traceback_flag"])

    def test_visible_task_context_text_uses_bigcodebench_instruct_prompt_only_for_embedding(self) -> None:
        task = _task(
            "bigcodebench-embedding-text",
            "fallback prompt should not be used",
            dataset_name="bigcodebench-v0.1.4",
            prompt_source="instruct_prompt",
            instruct_prompt="Use pandas to group rows by key.",
            code_prompt="def task_func(df):\n    pass\n",
            entry_point="task_func",
        )

        self.assertEqual("Use pandas to group rows by key.", visible_task_context_text(task))




































    def test_single_stage_bara_runner_updates_selector_once(self) -> None:
        task = _task("bara-task", "Return the input from task_func.", entry_point="task_func")
        selector = _FixedSelector("A2")
        model_client = _ScriptedModelClient(
            [
                "[CODE]\nBAD_SELF_TEST_CODE\n\n[SELF_TESTS]\ndef check(candidate):\n    assert candidate(1) == 1",
                "GOOD_REPAIRED_CODE",
            ]
        )
        runner = ExperimentRunner(
            model_client=model_client,
            selector=selector,
            generation_config=GenerationConfig(max_new_tokens=64),
            arms=get_partition_arms(["A0", "A2"]),
            evaluator=_evaluate_by_output,
        )

        def _self_test(task: CodeGenTask, cleaned_output: str, self_test_code: str, *, timeout_sec: int) -> dict[str, object]:
            del task, timeout_sec
            passed = cleaned_output == "GOOD_REPAIRED_CODE"
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": passed,
                "exit_code": 0 if passed else 1,
                "stdout": "",
                "stderr": "" if passed else f"AssertionError: self-test failed\n{self_test_code}",
                "duration_sec": 0.01,
            }

        with patch("codegen_agent.runner.evaluate_self_test_completion", side_effect=_self_test):
            records = runner.run_bara(
                [task],
                num_samples_per_task=1,
                pass_k=1,
                evaluation_timeout_sec=1,
            )

        self.assertEqual(1, len(records))
        self.assertEqual("A2", records[0].arm_id)
        self.assertEqual("A2", records[0].selection_reason["selector"]["selected_arm"])
        self.assertEqual(1, len(selector.updates))
        self.assertEqual("A2", selector.updates[0][0])
        self.assertGreater(selector.updates[0][2], 0.0)
        self.assertEqual(["draft_bundle", "repair_code"], [step["phase"] for step in records[0].predictions])
        self.assertEqual(2, len(model_client.prompt_history))
        self.assertEqual(model_client.prompt_history[0].system, model_client.prompt_history[1].system)
        self.assertIn("write the python code and a small self-test suite", model_client.prompt_history[0].user.lower())
        self.assertIn("repair the code", model_client.prompt_history[1].user.lower())
        self.assertIn("AssertionError: self-test failed", model_client.prompt_history[1].user)
        self.assertEqual([(), ()], model_client.stop_history)
        self.assertIn(task.prompt, model_client.prompt_history[1].user)
        self.assertIn(task.prompt, model_client.prompt_history[0].user)

    def test_run_all_arms_sg_uses_self_test_failure_log_before_final_official_eval(self) -> None:
        task = _task("sg-self-test-all-arms", "Return the input from task_func.", entry_point="task_func")
        model_client = _ScriptedModelClient(
            [
                "[CODE]\nBAD_SELF_TEST_CODE\n\n[SELF_TESTS]\ndef check(candidate):\n    assert candidate(1) == 1",
                "GOOD_REPAIRED_CODE",
            ]
        )
        official_evaluations: list[str] = []
        self_test_evaluations: list[str] = []

        def _official(task: CodeGenTask, cleaned_output: str, *, timeout_sec: int) -> dict[str, object]:
            del task, timeout_sec
            official_evaluations.append(cleaned_output)
            passed = cleaned_output == "GOOD_REPAIRED_CODE"
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": passed,
                "exit_code": 0 if passed else 1,
                "stdout": "",
                "stderr": "" if passed else "AssertionError: official failed",
                "duration_sec": 0.01,
            }

        def _self_test(task: CodeGenTask, cleaned_output: str, self_test_code: str, *, timeout_sec: int) -> dict[str, object]:
            del task, timeout_sec
            self_test_evaluations.append(cleaned_output)
            passed = cleaned_output == "GOOD_REPAIRED_CODE"
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": passed,
                "exit_code": 0 if passed else 1,
                "stdout": "",
                "stderr": "" if passed else f"AssertionError: self-test failed\n{self_test_code}",
                "duration_sec": 0.01,
                "num_tests": 5,
                "passed_tests": 5 if passed else 3,
                "failed_tests": 0 if passed else 2,
                "tests": [
                    {"name": "test_1", "passed": True},
                    {"name": "test_2", "passed": True},
                    {"name": "test_3", "passed": True},
                    {"name": "test_4", "passed": passed, "error_type": None if passed else "AssertionError"},
                    {"name": "test_5", "passed": passed, "error_type": None if passed else "AssertionError"},
                ],
            }

        runner = ExperimentRunner(
            model_client=model_client,
            selector=_FixedSelector("A0"),
            generation_config=GenerationConfig(max_new_tokens=64),
            arms=get_partition_arms(["A2"]),
            evaluator=_official,
        )

        with patch("codegen_agent.runner.evaluate_self_test_completion", side_effect=_self_test):
            records_by_arm = runner.run_all_arms(
                [task],
                num_samples_per_task=1,
                pass_k=1,
                evaluation_timeout_sec=1,
            )

        record = records_by_arm["A2"][0]
        self.assertEqual(["draft_bundle", "repair_code"], [step["phase"] for step in record.predictions])
        self.assertEqual(["BAD_SELF_TEST_CODE"], self_test_evaluations)
        self.assertEqual(["GOOD_REPAIRED_CODE"], official_evaluations)
        self.assertEqual(5, record.predictions[0]["details"]["self_test_summary"]["num_tests"])
        self.assertEqual(3, record.predictions[0]["details"]["self_test_summary"]["passed_tests"])
        self.assertEqual(2, record.predictions[0]["details"]["self_test_summary"]["failed_tests"])
        self.assertNotIn("self_test_summary", record.predictions[1].get("details", {}))
        self.assertNotIn("self_test_evaluation", record.predictions[1].get("details", {}))
        self.assertIn("write the python code and a small self-test suite", model_client.prompt_history[0].user.lower())
        self.assertIn("[self_tests]", model_client.prompt_history[0].user.lower())
        self.assertIn("[Self Tests]", model_client.prompt_history[1].user)
        self.assertIn("AssertionError: self-test failed", model_client.prompt_history[1].user)
        self.assertIn("Passed tests: 3/5", model_client.prompt_history[1].user)
        self.assertNotIn("official failed", model_client.prompt_history[1].user)

    def test_bara_sg_selection_runs_shared_self_test_workflow(self) -> None:
        task = _task("sg-self-test-bara", "Return the input from task_func.", entry_point="task_func")
        selector = _FixedSelector("A2")
        model_client = _ScriptedModelClient(
            [
                "[CODE]\nBAD_SELF_TEST_CODE\n\n[SELF_TESTS]\ndef check(candidate):\n    assert candidate(1) == 1",
                "GOOD_REPAIRED_CODE",
            ]
        )
        official_evaluations: list[str] = []
        self_test_evaluations: list[str] = []

        def _official(task: CodeGenTask, cleaned_output: str, *, timeout_sec: int) -> dict[str, object]:
            del task, timeout_sec
            official_evaluations.append(cleaned_output)
            passed = cleaned_output == "GOOD_REPAIRED_CODE"
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": passed,
                "exit_code": 0 if passed else 1,
                "stdout": "",
                "stderr": "" if passed else "AssertionError: official failed",
                "duration_sec": 0.01,
            }

        def _self_test(task: CodeGenTask, cleaned_output: str, self_test_code: str, *, timeout_sec: int) -> dict[str, object]:
            del task, timeout_sec
            self_test_evaluations.append(cleaned_output)
            passed = cleaned_output == "GOOD_REPAIRED_CODE"
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": passed,
                "exit_code": 0 if passed else 1,
                "stdout": "",
                "stderr": "" if passed else f"AssertionError: self-test failed\n{self_test_code}",
                "duration_sec": 0.01,
            }

        runner = ExperimentRunner(
            model_client=model_client,
            selector=selector,
            generation_config=GenerationConfig(max_new_tokens=64),
            arms=get_partition_arms(["A0", "A2"]),
            evaluator=_official,
        )

        with patch("codegen_agent.runner.evaluate_self_test_completion", side_effect=_self_test):
            records = runner.run_bara(
                [task],
                num_samples_per_task=1,
                pass_k=1,
                evaluation_timeout_sec=1,
            )

        self.assertEqual(1, len(records))
        self.assertEqual("A2", records[0].arm_id)
        self.assertEqual(["GOOD_REPAIRED_CODE"], official_evaluations)
        self.assertEqual(["BAD_SELF_TEST_CODE"], self_test_evaluations)
        self.assertEqual(2, len(model_client.prompt_history))
        self.assertIn("write the python code and a small self-test suite", model_client.prompt_history[0].user.lower())
        self.assertIn("AssertionError: self-test failed", model_client.prompt_history[1].user)
        self.assertIn("[Self Tests]", model_client.prompt_history[1].user)

    def test_run_all_arms_a2_batches_draft_then_repairs_failed_subset_only(self) -> None:
        tasks = [
            _task("sg-batch-pass", "Return the input from task_func.", entry_point="task_func"),
            _task("sg-batch-repair", "Return the input from task_func.", entry_point="task_func"),
        ]
        model_client = _ScriptedModelClient(
            [
                "[CODE]\nGOOD_DRAFT_CODE\n\n[SELF_TESTS]\ndef check(candidate):\n    assert candidate(1) == 1",
                "[CODE]\nBAD_DRAFT_CODE\n\n[SELF_TESTS]\ndef check(candidate):\n    assert candidate(1) == 1",
                "GOOD_REPAIRED_CODE",
            ]
        )
        official_evaluations: list[tuple[str, str]] = []
        self_test_evaluations: list[tuple[str, str]] = []

        def _official(task: CodeGenTask, cleaned_output: str, *, timeout_sec: int) -> dict[str, object]:
            del timeout_sec
            official_evaluations.append((task.task_id, cleaned_output))
            passed = cleaned_output in {"GOOD_DRAFT_CODE", "GOOD_REPAIRED_CODE"}
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": passed,
                "exit_code": 0 if passed else 1,
                "stdout": "",
                "stderr": "" if passed else "AssertionError: official failed",
                "duration_sec": 0.01,
            }

        def _self_test(task: CodeGenTask, cleaned_output: str, self_test_code: str, *, timeout_sec: int) -> dict[str, object]:
            del timeout_sec
            self_test_evaluations.append((task.task_id, cleaned_output))
            passed = cleaned_output == "GOOD_DRAFT_CODE"
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": passed,
                "exit_code": 0 if passed else 1,
                "stdout": "",
                "stderr": "" if passed else f"AssertionError: self-test failed\n{self_test_code}",
                "duration_sec": 0.01,
            }

        runner = ExperimentRunner(
            model_client=model_client,
            selector=_FixedSelector("A0"),
            generation_config=GenerationConfig(max_new_tokens=64),
            arms=get_partition_arms(["A2"]),
            evaluator=_official,
        )

        with patch("codegen_agent.runner.evaluate_self_test_completion", side_effect=_self_test):
            records_by_arm = runner.run_all_arms(
                tasks,
                num_samples_per_task=1,
                pass_k=1,
                evaluation_timeout_sec=1,
            )

        records = records_by_arm["A2"]
        self.assertEqual([2, 1], model_client.batch_prompt_counts)
        self.assertEqual(
            [["draft_bundle"], ["draft_bundle", "repair_code"]],
            [[step["phase"] for step in record.predictions] for record in records],
        )
        self.assertEqual(
            [("sg-batch-pass", "GOOD_DRAFT_CODE"), ("sg-batch-repair", "BAD_DRAFT_CODE")],
            self_test_evaluations,
        )
        self.assertEqual(
            [("sg-batch-pass", "GOOD_DRAFT_CODE"), ("sg-batch-repair", "GOOD_REPAIRED_CODE")],
            official_evaluations,
        )
        self.assertIn("AssertionError: self-test failed", model_client.prompt_history[2].user)
        self.assertIn(tasks[1].prompt, model_client.prompt_history[2].user)

    def test_run_all_arms_a2_batch_skips_self_test_for_bigcodebench_17(self) -> None:
        tasks = [
            _task("BigCodeBench/17", "Manage processes using task_func.", entry_point="task_func"),
            _task("sg-normal-pass", "Return the input from task_func.", entry_point="task_func"),
        ]
        model_client = _ScriptedModelClient(
            [
                "[CODE]\nBAD_OFFICIAL_CODE\n\n[SELF_TESTS]\ndef check(candidate):\n    assert candidate('notepad') is not None",
                "[CODE]\nGOOD_DRAFT_CODE\n\n[SELF_TESTS]\ndef check(candidate):\n    assert candidate(1) == 1",
                "GOOD_REPAIRED_CODE",
            ]
        )
        official_evaluations: list[tuple[str, str]] = []
        self_test_evaluations: list[tuple[str, str]] = []

        def _official(task: CodeGenTask, cleaned_output: str, *, timeout_sec: int) -> dict[str, object]:
            del timeout_sec
            official_evaluations.append((task.task_id, cleaned_output))
            passed = cleaned_output in {"GOOD_DRAFT_CODE", "GOOD_REPAIRED_CODE"}
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": passed,
                "exit_code": 0 if passed else 1,
                "stdout": "",
                "stderr": "" if passed else "AssertionError: official failed",
                "duration_sec": 0.01,
            }

        def _self_test(task: CodeGenTask, cleaned_output: str, self_test_code: str, *, timeout_sec: int) -> dict[str, object]:
            del timeout_sec
            self_test_evaluations.append((task.task_id, cleaned_output))
            passed = cleaned_output == "GOOD_DRAFT_CODE"
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": passed,
                "exit_code": 0 if passed else 1,
                "stdout": "",
                "stderr": "" if passed else f"AssertionError: self-test failed\n{self_test_code}",
                "duration_sec": 0.01,
            }

        runner = ExperimentRunner(
            model_client=model_client,
            selector=_FixedSelector("A0"),
            generation_config=GenerationConfig(max_new_tokens=64),
            arms=get_partition_arms(["A2"]),
            evaluator=_official,
        )

        with patch("codegen_agent.runner.evaluate_self_test_completion", side_effect=_self_test):
            records_by_arm = runner.run_all_arms(
                tasks,
                num_samples_per_task=1,
                pass_k=1,
                evaluation_timeout_sec=1,
            )

        records = records_by_arm["A2"]
        self.assertEqual([2, 1], model_client.batch_prompt_counts)
        self.assertEqual(
            [("sg-normal-pass", "GOOD_DRAFT_CODE")],
            self_test_evaluations,
        )
        self.assertEqual(
            [
                ("BigCodeBench/17", "BAD_OFFICIAL_CODE"),
                ("sg-normal-pass", "GOOD_DRAFT_CODE"),
                ("BigCodeBench/17", "GOOD_REPAIRED_CODE"),
            ],
            official_evaluations,
        )
        self.assertEqual(
            [["draft_bundle", "repair_code"], ["draft_bundle"]],
            [[step["phase"] for step in record.predictions] for record in records],
        )
        self.assertEqual(
            "task_id_self_test_bypass",
            records[0].predictions[0]["details"]["self_test_skipped_reason"],
        )
        self.assertIn("AssertionError: official failed", model_client.prompt_history[2].user)
        self.assertNotIn("AssertionError: self-test failed", model_client.prompt_history[2].user)

    def test_bara_skips_self_test_for_bigcodebench_17(self) -> None:
        task = _task("BigCodeBench/17", "Manage processes using task_func.", entry_point="task_func")
        selector = _FixedSelector("A2")
        model_client = _ScriptedModelClient(
            [
                "[CODE]\nBAD_OFFICIAL_CODE\n\n[SELF_TESTS]\ndef check(candidate):\n    assert candidate('notepad') is not None",
                "GOOD_REPAIRED_CODE",
            ]
        )
        official_evaluations: list[str] = []

        def _official(task: CodeGenTask, cleaned_output: str, *, timeout_sec: int) -> dict[str, object]:
            del task, timeout_sec
            official_evaluations.append(cleaned_output)
            passed = cleaned_output == "GOOD_REPAIRED_CODE"
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": passed,
                "exit_code": 0 if passed else 1,
                "stdout": "",
                "stderr": "" if passed else "AssertionError: official failed",
                "duration_sec": 0.01,
            }

        def _self_test(*_args, **_kwargs) -> dict[str, object]:
            raise AssertionError("self-test evaluator should be skipped for BigCodeBench/17")

        runner = ExperimentRunner(
            model_client=model_client,
            selector=selector,
            generation_config=GenerationConfig(max_new_tokens=64),
            arms=get_partition_arms(["A0", "A2"]),
            evaluator=_official,
        )

        with patch("codegen_agent.runner.evaluate_self_test_completion", side_effect=_self_test):
            records = runner.run_bara(
                [task],
                num_samples_per_task=1,
                pass_k=1,
                evaluation_timeout_sec=1,
            )

        self.assertEqual(["BAD_OFFICIAL_CODE", "GOOD_REPAIRED_CODE"], official_evaluations)
        self.assertEqual(["draft_bundle", "repair_code"], [step["phase"] for step in records[0].predictions])
        self.assertEqual("task_id_self_test_bypass", records[0].predictions[0]["details"]["self_test_skipped_reason"])
        self.assertIn("AssertionError: official failed", model_client.prompt_history[1].user)

    def test_run_all_arms_logs_a2_batch_phase_boundaries(self) -> None:
        tasks = [
            _task("sg-batch-pass", "Return the input from task_func.", entry_point="task_func"),
            _task("sg-batch-repair", "Return the input from task_func.", entry_point="task_func"),
        ]
        model_client = _ScriptedModelClient(
            [
                "[CODE]\nGOOD_DRAFT_CODE\n\n[SELF_TESTS]\ndef check(candidate):\n    assert candidate(1) == 1",
                "[CODE]\nBAD_DRAFT_CODE\n\n[SELF_TESTS]\ndef check(candidate):\n    assert candidate(1) == 1",
                "GOOD_REPAIRED_CODE",
            ]
        )

        def _official(task: CodeGenTask, cleaned_output: str, *, timeout_sec: int) -> dict[str, object]:
            del task, timeout_sec
            passed = cleaned_output in {"GOOD_DRAFT_CODE", "GOOD_REPAIRED_CODE"}
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": passed,
                "exit_code": 0 if passed else 1,
                "stdout": "",
                "stderr": "" if passed else "AssertionError: official failed",
                "duration_sec": 0.01,
            }

        def _self_test(task: CodeGenTask, cleaned_output: str, self_test_code: str, *, timeout_sec: int) -> dict[str, object]:
            del task, timeout_sec
            passed = cleaned_output == "GOOD_DRAFT_CODE"
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": passed,
                "exit_code": 0 if passed else 1,
                "stdout": "",
                "stderr": "" if passed else f"AssertionError: self-test failed\n{self_test_code}",
                "duration_sec": 0.01,
            }

        runner = ExperimentRunner(
            model_client=model_client,
            selector=_FixedSelector("A0"),
            generation_config=GenerationConfig(max_new_tokens=64),
            arms=get_partition_arms(["A2"]),
            evaluator=_official,
        )

        with patch("codegen_agent.runner.evaluate_self_test_completion", side_effect=_self_test), self.assertLogs(
            level="INFO"
        ) as logs:
            runner.run_all_arms(
                tasks,
                num_samples_per_task=1,
                pass_k=1,
                evaluation_timeout_sec=1,
            )

        joined = "\n".join(logs.output)
        self.assertIn("all_arms batch_start arm=A2", joined)
        self.assertIn("all_arms batch_end arm=A2", joined)
        self.assertIn("a2_batch draft_generate_start arm=A2", joined)
        self.assertIn("a2_batch draft_generate_end arm=A2", joined)
        self.assertIn("a2_batch draft_eval_start arm=A2", joined)
        self.assertIn("a2_batch draft_eval_end arm=A2", joined)
        self.assertIn("repair_candidates=1", joined)
        self.assertIn("a2_batch repair_generate_start arm=A2", joined)
        self.assertIn("a2_batch repair_generate_end arm=A2", joined)
        self.assertIn("a2_batch repair_eval_start arm=A2", joined)
        self.assertIn("a2_batch repair_eval_end arm=A2", joined)

    def test_run_all_arms_does_not_log_a2_draft_eval_task_details(self) -> None:
        tasks = [
            _task("sg-batch-pass", "Return the input from task_func.", entry_point="task_func"),
            _task("sg-batch-repair", "Return the input from task_func.", entry_point="task_func"),
        ]
        model_client = _ScriptedModelClient(
            [
                "[CODE]\nGOOD_DRAFT_CODE\n\n[SELF_TESTS]\ndef check(candidate):\n    assert candidate(1) == 1",
                "[CODE]\nBAD_DRAFT_CODE\n\n[SELF_TESTS]\ndef check(candidate):\n    assert candidate(1) == 1",
                "GOOD_REPAIRED_CODE",
            ]
        )

        def _official(task: CodeGenTask, cleaned_output: str, *, timeout_sec: int) -> dict[str, object]:
            del task, timeout_sec
            passed = cleaned_output in {"GOOD_DRAFT_CODE", "GOOD_REPAIRED_CODE"}
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": passed,
                "exit_code": 0 if passed else 1,
                "stdout": "",
                "stderr": "" if passed else "AssertionError: official failed",
                "duration_sec": 0.01,
            }

        def _self_test(task: CodeGenTask, cleaned_output: str, self_test_code: str, *, timeout_sec: int) -> dict[str, object]:
            del task, timeout_sec
            passed = cleaned_output == "GOOD_DRAFT_CODE"
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": passed,
                "exit_code": 0 if passed else 1,
                "stdout": "",
                "stderr": "" if passed else f"AssertionError: self-test failed\n{self_test_code}",
                "duration_sec": 0.01,
            }

        runner = ExperimentRunner(
            model_client=model_client,
            selector=_FixedSelector("A0"),
            generation_config=GenerationConfig(max_new_tokens=64),
            arms=get_partition_arms(["A2"]),
            evaluator=_official,
        )

        with patch("codegen_agent.runner.evaluate_self_test_completion", side_effect=_self_test), self.assertLogs(
            level="INFO"
        ) as logs:
            runner.run_all_arms(
                tasks,
                num_samples_per_task=1,
                pass_k=1,
                evaluation_timeout_sec=1,
            )

        joined = "\n".join(logs.output)
        self.assertNotIn("a2_batch draft_eval_task_start arm=A2", joined)
        self.assertNotIn("a2_batch draft_self_test_start arm=A2", joined)
        self.assertNotIn("a2_batch draft_self_test_end arm=A2", joined)
        self.assertNotIn("a2_batch draft_official_eval_start arm=A2", joined)
        self.assertNotIn("a2_batch draft_official_eval_end arm=A2", joined)
        self.assertNotIn("a2_batch draft_eval_task_end arm=A2", joined)

    def test_bara_respects_total_completion_budget_for_repair_stage(self) -> None:
        tasks = [
            _task(f"budget-task-{index}", "Return the input from task_func.", entry_point="task_func")
            for index in range(3)
        ]
        selector = _FixedSelector("A2")
        model_client = _ScriptedModelClient(["BAD", "GOOD", "BAD", "BAD", "BAD"])
        runner = ExperimentRunner(
            model_client=model_client,
            selector=selector,
            generation_config=GenerationConfig(max_new_tokens=64),
            bara_config=replace(DEFAULT_EXPERIMENT_CONFIG.bara, total_completion_token_budget=4),
            arms=get_partition_arms(["A0", "A2"]),
            evaluator=_evaluate_by_output,
        )

        records = runner.run_bara(
            tasks,
            num_samples_per_task=1,
            pass_k=1,
            evaluation_timeout_sec=1,
        )

        self.assertEqual(["A2", "A2", "A2"], [record.arm_id for record in records])
        self.assertLessEqual(sum(record.completion_tokens for record in records), 4)
        self.assertEqual(4, len(model_client.prompt_history))

    def test_default_bara_total_completion_budget_is_unbounded(self) -> None:
        self.assertIsNone(DEFAULT_EXPERIMENT_CONFIG.bara.total_completion_token_budget)





    def test_repair_workflow_stops_after_first_passing_attempt(self) -> None:
        task = _task("repair-task", "Implement task_func using dynamic programming.", entry_point="task_func")
        selector = _FixedSelector("A2")
        model_client = _ScriptedModelClient(
            ["[CODE]\nGOOD_FINAL_CODE\n\n[SELF_TESTS]\ndef check(candidate):\n    assert True"]
        )
        runner = ExperimentRunner(
            model_client=model_client,
            selector=selector,
            generation_config=GenerationConfig(max_new_tokens=64),
            arms=get_partition_arms(["A0", "A2"]),
            evaluator=_evaluate_by_output,
        )

        def _self_test(task: CodeGenTask, cleaned_output: str, self_test_code: str, *, timeout_sec: int) -> dict[str, object]:
            del task, self_test_code, timeout_sec
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": True,
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "duration_sec": 0.01,
            }

        with patch("codegen_agent.runner.evaluate_self_test_completion", side_effect=_self_test):
            records = runner.run_bara(
                [task],
                num_samples_per_task=1,
                pass_k=1,
                evaluation_timeout_sec=1,
            )

        self.assertEqual(1, len(records))
        self.assertEqual("A2", records[0].arm_id)
        self.assertEqual(["draft_bundle"], [step["phase"] for step in records[0].predictions])
        self.assertEqual(1, len(model_client.prompt_history))
        self.assertIn("write the python code and a small self-test suite", model_client.prompt_history[0].user.lower())
        self.assertEqual([1024], model_client.max_new_tokens_history)
        self.assertEqual([()], model_client.stop_history)
        self.assertEqual(1, len(selector.updates))

    def test_arm_token_budgets_scale_dg_lt_sg(self) -> None:
        arm_map = {arm.arm_id: arm for arm in get_partition_arms(["A0", "A2", "A2", "A1"])}

        self.assertEqual(2048, arm_map["A0"].max_new_tokens_for_phase("final_code"))
        self.assertEqual(2048, arm_map["A2"].max_new_tokens_for_phase("draft_bundle"))
        self.assertEqual(2048, arm_map["A2"].max_new_tokens_for_phase("repair_code"))
        self.assertEqual(1024, arm_map["A2"].max_new_tokens_for_phase("final_code"))
        self.assertEqual(2048, arm_map["A1"].max_new_tokens_for_phase("final_code"))
        with self.assertRaises(KeyError):
            arm_map["A2"].max_new_tokens_for_phase("final_code")
        self.assertLess(
            arm_map["A0"].total_token_budget(),
            arm_map["A2"].total_token_budget(),
        )
        self.assertEqual(
            arm_map["A0"].total_token_budget(),
            arm_map["A1"].total_token_budget(),
        )



    def test_a2_final_code_prompt_phase_is_unsupported(self) -> None:
        task = _task(
            "a2-final-code-unsupported",
            "Implement task_func(x).",
            code_prompt="def task_func(x):\n    pass\n",
            entry_point="task_func",
        )

        with self.assertRaises(KeyError):
            build_prompt_bundle(
                task,
                get_partition_arms(["A2"])[0],
                phase="final_code",
                artifacts={},
            )

    def test_a2_evaluates_raw_code_only_output(self) -> None:
        task = _task("a2-code-only", "Return a good value from task_func.", entry_point="task_func")
        model_client = _ScriptedModelClient(
            [
                "GOOD_CODE"
            ]
        )

        observed_cleaned_outputs: list[str] = []

        def _evaluate_code_only(task: CodeGenTask, cleaned_output: str, *, timeout_sec: int) -> dict[str, object]:
            del task, timeout_sec
            observed_cleaned_outputs.append(cleaned_output)
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": cleaned_output == "GOOD_CODE",
                "exit_code": 0 if cleaned_output == "GOOD_CODE" else 1,
                "stdout": "",
                "stderr": "" if cleaned_output == "GOOD_CODE" else "AssertionError: failed test",
                "duration_sec": 0.01,
            }

        runner = ExperimentRunner(
            model_client=model_client,
            selector=_FixedSelector("A2"),
            generation_config=GenerationConfig(max_new_tokens=64),
            arms=get_partition_arms(["A2"]),
            evaluator=_evaluate_code_only,
        )

        records_by_arm = runner.run_all_arms(
            [task],
            num_samples_per_task=1,
            pass_k=1,
            evaluation_timeout_sec=1,
        )

        self.assertEqual(["GOOD_CODE"], observed_cleaned_outputs)
        record = records_by_arm["A2"][0]
        self.assertEqual("A2", record.arm_id)
        self.assertEqual(1.0, record.pass_at_k)
        self.assertEqual("GOOD_CODE", record.predictions[0]["cleaned_output"])
        self.assertNotIn("details", record.predictions[0])

    def test_a1_evaluates_raw_code_only_output(self) -> None:
        task = _task("a1-code-only", "Return a good value from task_func.", entry_point="task_func")
        final_code = "def task_func():\n    return 'good'\n"
        model_client = _ScriptedModelClient(
            [
                "[PLAN]\n"
                "1. Return the required constant.\n"
                "2. Preserve the public interface.\n\n"
                "[CODE]\n"
                f"{final_code}"
            ]
        )

        observed_cleaned_outputs: list[str] = []

        def _evaluate_code_only(task: CodeGenTask, cleaned_output: str, *, timeout_sec: int) -> dict[str, object]:
            del task, timeout_sec
            observed_cleaned_outputs.append(cleaned_output)
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": cleaned_output == final_code.rstrip(),
                "exit_code": 0 if cleaned_output == final_code.rstrip() else 1,
                "stdout": "",
                "stderr": "" if cleaned_output == final_code.rstrip() else "AssertionError: failed test",
                "duration_sec": 0.01,
            }

        runner = ExperimentRunner(
            model_client=model_client,
            selector=_FixedSelector("A1"),
            generation_config=GenerationConfig(max_new_tokens=64),
            arms=get_partition_arms(["A1"]),
            evaluator=_evaluate_code_only,
        )

        records_by_arm = runner.run_all_arms(
            [task],
            num_samples_per_task=1,
            pass_k=1,
            evaluation_timeout_sec=1,
        )

        self.assertEqual([final_code.rstrip()], observed_cleaned_outputs)
        record = records_by_arm["A1"][0]
        self.assertEqual("A1", record.arm_id)
        self.assertEqual(1.0, record.pass_at_k)
        self.assertEqual(final_code.rstrip(), record.predictions[0]["cleaned_output"])
        self.assertEqual([2048], model_client.max_new_tokens_history)
        self.assertEqual(
            "1. Return the required constant.\n2. Preserve the public interface.",
            record.predictions[0]["details"]["plan"],
        )

    def test_bara_runner_executes_selected_a1_arm_instead_of_falling_back_to_a0(self) -> None:
        task = _task("bara-a1", "Return a good value from task_func.", entry_point="task_func")
        final_code = "def task_func():\n    return 'good'\n"
        model_client = _ScriptedModelClient(
            [
                "[PLAN]\n"
                "1. Interface: Copy the task_func interface.\n"
                "2. Side Effects: none explicit.\n"
                "3. Return Contract: Return the required constant.\n"
                "4. Failure Contract: unspecified.\n\n"
                "[CODE]\n"
                f"{final_code}"
            ]
        )

        observed_cleaned_outputs: list[str] = []

        def _evaluate_code_only(task: CodeGenTask, cleaned_output: str, *, timeout_sec: int) -> dict[str, object]:
            del task, timeout_sec
            observed_cleaned_outputs.append(cleaned_output)
            return {
                "cleaned_output": cleaned_output,
                "executable_code": cleaned_output,
                "passed": cleaned_output == final_code.rstrip(),
                "exit_code": 0 if cleaned_output == final_code.rstrip() else 1,
                "stdout": "",
                "stderr": "" if cleaned_output == final_code.rstrip() else "AssertionError: failed test",
                "duration_sec": 0.01,
            }

        runner = ExperimentRunner(
            model_client=model_client,
            selector=_FixedSelector("A1"),
            generation_config=GenerationConfig(max_new_tokens=64),
            arms=get_partition_arms(["A0", "A2", "A1"]),
            evaluator=_evaluate_code_only,
        )

        records = runner.run_bara(
            [task],
            num_samples_per_task=1,
            pass_k=1,
            evaluation_timeout_sec=1,
        )

        self.assertEqual(1, len(records))
        self.assertEqual("A1", records[0].arm_id)
        self.assertEqual("A1", records[0].selection_reason["selector"]["selected_arm"])
        self.assertEqual("A1", records[0].metadata["bara_selected_arm"])
        self.assertEqual([final_code.rstrip()], observed_cleaned_outputs)
        self.assertEqual(1, len(model_client.prompt_history))
        self.assertEqual([2048], model_client.max_new_tokens_history)
        self.assertIn("[PLAN]", model_client.prompt_history[0].user)
        self.assertIn("[CODE]", model_client.prompt_history[0].user)

    def test_clean_model_output_preserves_a1_structured_bundle_until_parser_stage(self) -> None:
        from codegen_agent.models import _clean_model_output

        task = _task("a1-cleaning", "Return a good value from task_func.", entry_point="task_func")
        prompt = build_prompt_bundle(
            task,
            get_partition_arms(["A1"])[0],
            phase="final_code",
        )
        raw_output = (
            "[PLAN]\n"
            "1. Return the required constant.\n"
            "2. Preserve the public interface.\n\n"
            "[CODE]\n"
            "def task_func():\n"
            "    return 'good'\n"
        )

        cleaned = _clean_model_output(raw_output, prompt)

        self.assertIn("[PLAN]", cleaned)
        self.assertIn("[CODE]", cleaned)
        self.assertIn("def task_func():", cleaned)

    def test_run_all_arms_rejects_unsupported_multi_sample_requests(self) -> None:
        task = _task("multi-sample", "Return the input from task_func.", entry_point="task_func")
        runner = ExperimentRunner(
            model_client=_ScriptedModelClient(["GOOD_CODE"]),
            selector=_FixedSelector("A0"),
            generation_config=GenerationConfig(max_new_tokens=64),
            arms=get_partition_arms(["A0", "A2"]),
            evaluator=_evaluate_by_output,
        )

        with self.assertRaises(ValueError):
            runner.run_all_arms(
                [task],
                num_samples_per_task=2,
                pass_k=1,
                evaluation_timeout_sec=1,
            )

    def test_run_bara_rejects_unsupported_pass_k(self) -> None:
        task = _task("pass-k", "Return the input from task_func.", entry_point="task_func")
        runner = ExperimentRunner(
            model_client=_ScriptedModelClient(["GOOD_CODE"]),
            selector=_FixedSelector("A0"),
            generation_config=GenerationConfig(max_new_tokens=64),
            arms=get_partition_arms(["A0", "A2"]),
            evaluator=_evaluate_by_output,
        )

        with self.assertRaises(ValueError):
            runner.run_bara(
                [task],
                num_samples_per_task=1,
                pass_k=2,
                evaluation_timeout_sec=1,
            )

    def test_run_all_arms_can_skip_feature_and_reward_computation(self) -> None:
        task = _task("skip-metrics", "Return the input from task_func.", entry_point="task_func")
        runner = ExperimentRunner(
            model_client=_ScriptedModelClient(["GOOD_CODE"]),
            selector=None,
            generation_config=GenerationConfig(max_new_tokens=64),
            feature_extractor=None,
            compute_features_and_reward=False,
            arms=get_partition_arms(["A0"]),
            evaluator=_evaluate_by_output,
        )

        records_by_arm = runner.run_all_arms(
            [task],
            num_samples_per_task=1,
            pass_k=1,
            evaluation_timeout_sec=1,
        )

        record = records_by_arm["A0"][0]
        self.assertEqual([], record.feature_names)
        self.assertEqual([], record.feature_vector)
        self.assertIsNone(record.reward)
        self.assertEqual({}, record.reward_components)
        self.assertEqual(1.0, record.pass_at_k)
        self.assertNotIn("feature_signals", record.metadata)
        self.assertNotIn("reward_formula", record.metadata)
        self.assertNotIn("reward_weights", record.metadata)

    def test_run_all_arms_writes_warm_start_rows_with_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                DEFAULT_EXPERIMENT_CONFIG,
                paths=PathConfig(experiment_results_dir=Path(tmpdir)),
                arm_ids=("A0", "A2"),
                all_arms_compute_reward_and_features=True,
                limit=1,
            )
            tasks = [_task("dg-1", "Return the input from task_func.", entry_point="task_func")]
            captured_rows: list[dict[str, object]] = []

            def _capture_rows(path: Path, rows: list[dict[str, object]]) -> None:
                del path
                captured_rows[:] = list(rows)

            class _FakeRunner:
                def run_all_arms(
                    self,
                    run_tasks,
                    *,
                    num_samples_per_task,
                    pass_k,
                    evaluation_timeout_sec,
                    on_arm_complete,
                ) -> None:
                    del run_tasks, num_samples_per_task, pass_k, evaluation_timeout_sec
                    for arm in get_partition_arms(["A0", "A2"]):
                        record = type(
                            "Record",
                            (),
                            {
                                "task_id": "dg-1",
                                "arm_id": arm.arm_id,
                                "reward": 0.5,
                                "pass_at_k": 1.0,
                                "num_samples": 1,
                                "num_correct": 1,
                                "reference_completion_tokens": 4,
                                "completion_tokens": 5,
                                "completion_token_budget": 6,
                                "output_length": 12,
                                "feature_names": ["bias", "text_embedding_000"],
                                "feature_vector": [1.0, 0.25],
                                "metadata": {"source": "test"},
                                "reward_components": {"total_reward": 0.5},
                                "model_name": "m",
                                "model_path": "m",
                                "dataset_name": "toy",
                            },
                        )()
                        on_arm_complete(arm, [record])

            with patch.object(cli, "_load_tasks", return_value=tasks), patch.object(
                cli, "_build_runner", return_value=_FakeRunner()
            ), patch.object(
                cli, "_write_records"
            ), patch.object(
                cli, "_write_summary"
            ), patch.object(
                cli, "write_jsonl", side_effect=_capture_rows
            ):
                cli._run_all_arms(config, static_completion=None, run_bara_after_all_arms=False)

        self.assertEqual(2, len(captured_rows))
        self.assertIn("context_feature_names", captured_rows[0])
        self.assertIn("context_vector", captured_rows[0])
        self.assertIn("output_length", captured_rows[0])
        self.assertEqual(["bias", "text_embedding_000"], captured_rows[0]["context_feature_names"])
        self.assertEqual([1.0, 0.25], captured_rows[0]["context_vector"])

    def test_run_all_arms_skips_reward_row_writes_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                DEFAULT_EXPERIMENT_CONFIG,
                paths=PathConfig(experiment_results_dir=Path(tmpdir)),
                arm_ids=("A0",),
                all_arms_compute_reward_and_features=False,
                limit=1,
            )
            tasks = [_task("dg-1", "Return the input from task_func.", entry_point="task_func")]

            class _FakeRunner:
                def run_all_arms(
                    self,
                    run_tasks,
                    *,
                    num_samples_per_task,
                    pass_k,
                    evaluation_timeout_sec,
                    on_arm_complete,
                ) -> None:
                    del run_tasks, num_samples_per_task, pass_k, evaluation_timeout_sec
                    record = type(
                        "Record",
                        (),
                        {
                            "task_id": "dg-1",
                            "arm_id": "A0",
                            "reward": None,
                            "pass_at_k": 1.0,
                            "num_samples": 1,
                            "num_correct": 1,
                            "reference_completion_tokens": 0,
                            "completion_tokens": 5,
                            "completion_token_budget": None,
                            "output_length": 12,
                            "feature_names": [],
                            "feature_vector": [],
                            "metadata": {"source": "test"},
                            "reward_components": {},
                            "model_name": "m",
                            "model_path": "m",
                            "dataset_name": "toy",
                        },
                    )()
                    on_arm_complete(get_partition_arms(["A0"])[0], [record])

            with patch.object(cli, "_load_tasks", return_value=tasks), patch.object(
                cli, "_build_runner", return_value=_FakeRunner()
            ), patch.object(
                cli, "_write_records"
            ), patch.object(
                cli, "_write_summary"
            ), patch.object(
                cli, "write_jsonl"
            ) as write_jsonl_mock:
                cli._run_all_arms(config, static_completion=None, run_bara_after_all_arms=False)

        write_jsonl_mock.assert_not_called()

    def test_run_all_arms_writes_repair_decision_rows_for_failed_dg_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                DEFAULT_EXPERIMENT_CONFIG,
                paths=PathConfig(experiment_results_dir=Path(tmpdir)),
                arm_ids=("A0", "A2"),
                all_arms_compute_reward_and_features=True,
                limit=1,
            )
            task = _task("repair-row-task", "Return the input from task_func.", entry_point="task_func")
            captured_rows: list[dict[str, object]] = []

            def _capture_rows(path: Path, rows: list[dict[str, object]]) -> None:
                del path
                captured_rows[:] = list(rows)

            dg_record = type(
                "Record",
                (),
                {
                    "task_id": task.task_id,
                    "task": task,
                    "arm_id": "A0",
                    "reward": -0.1,
                    "pass_at_k": 0.0,
                    "num_samples": 1,
                    "num_correct": 0,
                    "reference_completion_tokens": 4,
                    "completion_tokens": 5,
                    "completion_token_budget": 6,
                    "output_length": 12,
                    "feature_names": ["bias", "text_embedding_000"],
                    "feature_vector": [1.0, 0.25],
                    "metadata": {"source": "test"},
                    "reward_components": {"total_reward": -0.1},
                    "model_name": "m",
                    "model_path": "m",
                    "dataset_name": "toy",
                    "predictions": [
                        {
                            "phase": "final_code",
                            "raw_output": "BAD",
                            "cleaned_output": "BAD",
                            "completion_tokens": 5,
                            "passed": False,
                            "exit_code": 1,
                            "stdout": "",
                            "stderr": "AssertionError: failed test",
                        }
                    ],
                },
            )()
            sg_record = type(
                "Record",
                (),
                {
                    "task_id": task.task_id,
                    "task": task,
                    "arm_id": "A2",
                    "reward": 0.8,
                    "pass_at_k": 1.0,
                    "num_samples": 1,
                    "num_correct": 1,
                    "reference_completion_tokens": 4,
                    "completion_tokens": 9,
                    "completion_token_budget": 6,
                    "output_length": 12,
                    "feature_names": ["bias", "text_embedding_000"],
                    "feature_vector": [1.0, 0.25],
                    "metadata": {"source": "test"},
                    "reward_components": {"total_reward": 0.8},
                    "model_name": "m",
                    "model_path": "m",
                    "dataset_name": "toy",
                    "predictions": [
                        {
                            "phase": "final_code",
                            "raw_output": "BAD",
                            "cleaned_output": "BAD",
                            "completion_tokens": 5,
                            "passed": False,
                            "exit_code": 1,
                            "stdout": "",
                            "stderr": "AssertionError: failed test",
                        },
                        {
                            "phase": "repair_code",
                            "raw_output": "GOOD",
                            "cleaned_output": "GOOD",
                            "completion_tokens": 4,
                            "passed": True,
                            "exit_code": 0,
                            "stdout": "",
                            "stderr": "",
                        },
                    ],
                },
            )()

            class _FakeExtractor:
                def extract_repair_context(self, *_args, **_kwargs):
                    return type(
                        "DecisionFeatures",
                        (),
                        {
                            "names": ["bias", "failure_assertion_flag"],
                            "values": [1.0, 1.0],
                            "signals": {},
                        },
                    )()

            class _FakeRunner:
                feature_extractor = _FakeExtractor()

                def run_all_arms(
                    self,
                    run_tasks,
                    *,
                    num_samples_per_task,
                    pass_k,
                    evaluation_timeout_sec,
                    on_arm_complete,
                ) -> None:
                    del run_tasks, num_samples_per_task, pass_k, evaluation_timeout_sec
                    on_arm_complete(get_partition_arms(["A0"])[0], [dg_record])
                    on_arm_complete(get_partition_arms(["A2"])[0], [sg_record])

            with patch.object(cli, "_load_tasks", return_value=[task]), patch.object(
                cli, "_build_runner", return_value=_FakeRunner()
            ), patch.object(
                cli, "_write_records"
            ), patch.object(
                cli, "_write_summary"
            ), patch.object(
                cli, "write_jsonl", side_effect=_capture_rows
            ):
                cli._run_all_arms(config, static_completion=None, run_bara_after_all_arms=False)

        sg_row = next(row for row in captured_rows if row["arm_id"] == "A2")
        self.assertEqual("repair", sg_row["decision_selector_stage"])
        self.assertEqual(["bias", "failure_assertion_flag"], sg_row["decision_context_feature_names"])
        self.assertEqual([1.0, 1.0], sg_row["decision_context_vector"])
        self.assertIn("decision_reward", sg_row)

    def test_cli_bara_mode_uses_all_registered_arms_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                DEFAULT_EXPERIMENT_CONFIG,
                paths=PathConfig(experiment_results_dir=Path(tmpdir)),
                limit=1,
                run_mode="bara",
            )
            tasks = [_task("dg-1", "Return the input from task_func.", entry_point="task_func")]
            with patch.object(cli, "_load_tasks", return_value=tasks), patch.object(
                cli, "_run_bara_mainline", return_value=[]
            ) as run_bara, patch.object(cli, "_write_bara_outputs"):
                cli._run_bara(config, static_completion=None)

        bara_arms = tuple(arm.arm_id for arm in run_bara.call_args.kwargs["arms"])
        self.assertEqual(("A0", "A1", "A2"), bara_arms)

    def test_bara_paths_include_normalized_arm_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = PathConfig(experiment_results_dir=Path(tmpdir))

            output_path = paths.bara_output_path(
                "qwen2.5-32b",
                "bigcodebench-v0.1.4",
                ("A2", "A0"),
            )
            summary_path = paths.bara_summary_path(
                "qwen2.5-32b",
                "bigcodebench-v0.1.4",
                ("A2", "A0"),
            )

        self.assertEqual(
            Path(tmpdir)
            / "pass_only"
            / "qwen2.5-32b"
            / "qwen2.5-32b_bara_A0-A2.jsonl",
            output_path,
        )
        self.assertEqual(
            Path(tmpdir)
            / "pass_only"
            / "qwen2.5-32b"
            / "qwen2.5-32b_bara_A0-A2.summary.json",
            summary_path,
        )

    def test_bara_paths_can_include_budget_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = PathConfig(experiment_results_dir=Path(tmpdir))

            output_path = paths.bara_output_path(
                "qwen2.5-32b",
                "bigcodebench-v0.1.4",
                ("A2", "A0"),
                budget_label="budget-15000",
            )
            summary_path = paths.bara_summary_path(
                "qwen2.5-32b",
                "bigcodebench-v0.1.4",
                ("A2", "A0"),
                budget_label="budget-15000",
            )

        self.assertEqual(
            Path(tmpdir)
            / "pass_only"
            / "qwen2.5-32b"
            / "qwen2.5-32b_bara_A0-A2_budget-15000.jsonl",
            output_path,
        )
        self.assertEqual(
            Path(tmpdir)
            / "pass_only"
            / "qwen2.5-32b"
            / "qwen2.5-32b_bara_A0-A2_budget-15000.summary.json",
            summary_path,
        )


    def test_parser_rejects_batch_size_override(self) -> None:
        parser = cli.build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--run-mode",
                    "all_arms",
                    "--dataset-name",
                    "bigcodebench-v0.1.4",
                    "--batch-size",
                    "32",
                ]
            )





    def test_run_all_arms_builds_runner_without_warm_start_preload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                DEFAULT_EXPERIMENT_CONFIG,
                run_mode="all_arms",
                dataset_name="bigcodebench-v0.1.4",
                paths=PathConfig(experiment_results_dir=Path(tmpdir)),
                limit=1,
            )
            tasks = [_task("toy-1", "Return the input from task_func.", entry_point="task_func")]

            class _FakeRunner:
                def run_all_arms(
                    self,
                    run_tasks,
                    *,
                    num_samples_per_task,
                    pass_k,
                    evaluation_timeout_sec,
                    on_arm_complete,
                ) -> None:
                    del run_tasks, num_samples_per_task, pass_k, evaluation_timeout_sec
                    record = type(
                        "Record",
                        (),
                        {
                            "task_id": "toy-1",
                            "arm_id": "A0",
                            "reward": 0.0,
                            "pass_at_k": 0.0,
                            "num_samples": 1,
                            "num_correct": 0,
                            "reference_completion_tokens": 1,
                            "completion_tokens": 1,
                            "completion_token_budget": 1,
                            "output_length": 0,
                            "feature_names": ["bias"],
                            "feature_vector": [1.0],
                            "metadata": {},
                            "reward_components": {},
                            "model_name": "m",
                            "model_path": "m",
                            "dataset_name": "toy",
                        },
                    )()
                    on_arm_complete(get_partition_arms(["A0"])[0], [record])

            with patch.object(cli, "_load_tasks", return_value=tasks), patch.object(
                cli, "_build_runner", return_value=_FakeRunner()
            ) as build_runner, patch.object(cli, "_write_records"), patch.object(cli, "_write_summary"), patch.object(
                cli, "write_jsonl"
            ):
                cli._run_all_arms(config, static_completion=None, run_bara_after_all_arms=False)

        self.assertIn("apply_warm_start", build_runner.call_args.kwargs)
        self.assertFalse(build_runner.call_args.kwargs["apply_warm_start"])
        self.assertIn("enable_feature_extractor", build_runner.call_args.kwargs)
        self.assertFalse(build_runner.call_args.kwargs["enable_feature_extractor"])

    def test_build_runner_skips_feature_extractor_by_default_for_all_arms(self) -> None:
        config = replace(
            DEFAULT_EXPERIMENT_CONFIG,
            run_mode="all_arms",
            all_arms_compute_reward_and_features=False,
        )

        with patch.object(cli, "build_feature_extractor") as build_feature_extractor:
            runner = cli._build_runner(
                config,
                arms=get_partition_arms(["A0"]),
                static_completion="GOOD",
                apply_warm_start=False,
                enable_feature_extractor=config.all_arms_compute_reward_and_features,
            )

        build_feature_extractor.assert_not_called()
        self.assertIsNone(runner.selector)
        self.assertIsNone(runner.feature_extractor)
        self.assertFalse(runner.compute_features_and_reward)



    def test_build_runner_applies_warm_start_rows_before_bara(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                DEFAULT_EXPERIMENT_CONFIG,
                bandit=replace(
                    DEFAULT_EXPERIMENT_CONFIG.bandit,
                    selector=replace(
                        DEFAULT_EXPERIMENT_CONFIG.bandit.selector,
                        min_pulls_per_arm=1,
                    ),
                ),
                paths=PathConfig(experiment_results_dir=Path(tmpdir)),
                warm_start=replace(DEFAULT_EXPERIMENT_CONFIG.warm_start, enabled=True),
            )
            warm_path = config.paths.reward_output_path(
                config.model.model_name,
                config.warm_start.dataset_name,
            )
            write_jsonl(
                warm_path,
                [
                    {
                        "task_id": "warm-1",
                        "arm_id": "A0",
                        "pass_at_k": 0.0,
                        "completion_tokens": 40,
                        "reference_completion_tokens": 20,
                        "completion_token_budget": 30,
                        "context_vector": [1.0, 0.25],
                        "reward": -0.75,
                    },
                    {
                        "task_id": "warm-2",
                        "arm_id": "A2",
                        "pass_at_k": 1.0,
                        "completion_tokens": 1,
                        "reference_completion_tokens": 20,
                        "completion_token_budget": 30,
                        "context_vector": [1.0, 0.25],
                        "reward": 0.95,
                    },
                ],
            )

            class _FakeExtractor:
                FEATURE_NAMES = ["bias", "text_embedding_000"]
                def extract(self, task):
                    del task
                    return FeatureExtractionResult(names=list(self.FEATURE_NAMES), values=[1.0, 0.25])

            warm_tasks = [
                replace(_task("warm-1", "warm prompt", dataset_name=config.warm_start.dataset_name), task_kind="callable"),
                replace(_task("warm-2", "warm prompt", dataset_name=config.warm_start.dataset_name), task_kind="callable"),
            ]
            with patch.object(cli, "build_feature_extractor", return_value=_FakeExtractor()), patch.object(
                cli, "load_codegen_tasks", return_value=warm_tasks
            ):
                runner = cli._build_runner(
                    config,
                    arms=get_partition_arms(["A0", "A2"]),
                    static_completion="GOOD",
                )

        decision = runner.selector.select([1.0, 0.25])
        self.assertEqual("A2", decision.arm_id)

    def test_build_runner_ignores_repair_decision_warm_start_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                DEFAULT_EXPERIMENT_CONFIG,
                bandit=replace(DEFAULT_EXPERIMENT_CONFIG.bandit, selector=replace(DEFAULT_EXPERIMENT_CONFIG.bandit.selector, alpha=0.0)),
                paths=PathConfig(experiment_results_dir=Path(tmpdir)),
                warm_start=replace(DEFAULT_EXPERIMENT_CONFIG.warm_start, enabled=True),
            )
            warm_path = config.paths.reward_output_path(
                config.model.model_name,
                config.warm_start.dataset_name,
            )
            write_jsonl(
                warm_path,
                [
                    {
                        "task_id": "warm-repair-1",
                        "arm_id": "A2",
                        "decision_selector_stage": "repair",
                        "decision_context_feature_names": ["bias", "failure_assertion_flag"],
                        "decision_context_vector": [1.0, 1.0],
                        "decision_reward": 0.75,
                    }
                ],
            )

            class _FakeExtractor:
                FEATURE_NAMES = ["bias"]
                DECISION_FEATURE_NAMES = ["bias", "failure_assertion_flag"]

            with patch.object(cli, "build_feature_extractor", return_value=_FakeExtractor()):
                runner = cli._build_runner(
                    config,
                    arms=get_partition_arms(["A0", "A2"]),
                    static_completion="GOOD",
                )

        decision = runner.selector.select([1.0])
        self.assertEqual("A0", decision.arm_id)


    def test_build_runner_skips_warm_start_rows_with_mismatched_feature_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                DEFAULT_EXPERIMENT_CONFIG,
                bandit=replace(DEFAULT_EXPERIMENT_CONFIG.bandit, selector=replace(DEFAULT_EXPERIMENT_CONFIG.bandit.selector, alpha=0.0)),
                paths=PathConfig(experiment_results_dir=Path(tmpdir)),
                warm_start=replace(DEFAULT_EXPERIMENT_CONFIG.warm_start, enabled=True),
            )
            warm_path = config.paths.reward_output_path(
                config.model.model_name,
                config.warm_start.dataset_name,
            )
            write_jsonl(
                warm_path,
                [
                    {
                        "task_id": "warm-bad",
                        "arm_id": "A0",
                        "pass_at_k": 1.0,
                        "completion_tokens": 30,
                        "reference_completion_tokens": 20,
                        "completion_token_budget": 30,
                        "context_feature_names": ["bias", "different_feature"],
                        "context_vector": [1.0, 0.25],
                    }
                ],
            )

            class _FakeExtractor:
                FEATURE_NAMES = ["bias", "text_embedding_000"]

                def extract(self, task):
                    del task
                    return FeatureExtractionResult(names=list(self.FEATURE_NAMES), values=[1.0, 0.25])

            warm_tasks = [
                replace(_task("warm-bad", "warm prompt", dataset_name=config.warm_start.dataset_name), task_kind="callable"),
            ]
            with patch.object(cli, "build_feature_extractor", return_value=_FakeExtractor()), patch.object(
                cli, "load_codegen_tasks", return_value=warm_tasks
            ):
                runner = cli._build_runner(
                    config,
                    arms=get_partition_arms(["A0", "A2"]),
                    static_completion="GOOD",
                )

        decision = runner.selector.select([1.0, 0.25])
        self.assertEqual("A0", decision.arm_id)

    def test_build_runner_recomputes_warm_start_context_from_current_feature_extractor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                DEFAULT_EXPERIMENT_CONFIG,
                bandit=replace(
                    DEFAULT_EXPERIMENT_CONFIG.bandit,
                    selector=replace(
                        DEFAULT_EXPERIMENT_CONFIG.bandit.selector,
                        min_pulls_per_arm=1,
                    ),
                ),
                paths=PathConfig(experiment_results_dir=Path(tmpdir)),
                warm_start=replace(DEFAULT_EXPERIMENT_CONFIG.warm_start, enabled=True),
            )
            warm_path = config.paths.reward_output_path(
                config.model.model_name,
                config.warm_start.dataset_name,
            )
            write_jsonl(
                warm_path,
                [
                    {
                        "task_id": "warm-a0",
                        "arm_id": "A0",
                        "pass_at_k": 0.0,
                        "completion_tokens": 40,
                        "reference_completion_tokens": 20,
                        "completion_token_budget": 30,
                        "context_feature_names": ["bias", "stale_feature"],
                        "context_vector": [99.0, 99.0],
                        "reward": -0.75,
                    },
                    {
                        "task_id": "warm-a2",
                        "arm_id": "A2",
                        "pass_at_k": 1.0,
                        "completion_tokens": 1,
                        "reference_completion_tokens": 20,
                        "completion_token_budget": 30,
                        "context_feature_names": ["bias", "stale_feature"],
                        "context_vector": [-99.0, -99.0],
                        "reward": 0.95,
                    },
                ],
            )

            class _FakeExtractor:
                FEATURE_NAMES = ["bias", "fresh_feature"]

                def extract(self, task):
                    value = 0.0 if task.task_id == "warm-a0" else 1.0
                    return FeatureExtractionResult(names=list(self.FEATURE_NAMES), values=[1.0, value])

            warm_tasks = [
                replace(_task("warm-a0", "warm prompt", dataset_name=config.warm_start.dataset_name), task_kind="callable"),
                replace(_task("warm-a2", "warm prompt", dataset_name=config.warm_start.dataset_name), task_kind="callable"),
            ]

            with patch.object(cli, "build_feature_extractor", return_value=_FakeExtractor()), patch.object(
                cli, "load_codegen_tasks", return_value=warm_tasks
            ):
                runner = cli._build_runner(
                    config,
                    arms=get_partition_arms(["A0", "A2"]),
                    static_completion="GOOD",
                )

        decision = runner.selector.select([1.0, 1.0])
        self.assertEqual("A2", decision.arm_id)






    def test_reward_prefers_cheaper_passing_arm(self) -> None:
        cheaper_pass = _compute_reward_components(
            pass_at_k=1.0,
            reference_completion_tokens=120,
            completion_tokens=10,
            completion_token_budget=180,
            reward_config=_reward_config_with_version("v2"),
        )
        expensive_pass = _compute_reward_components(
            pass_at_k=1.0,
            reference_completion_tokens=120,
            completion_tokens=20,
            completion_token_budget=180,
            reward_config=_reward_config_with_version("v2"),
        )

        self.assertGreater(cheaper_pass["total_reward"], expensive_pass["total_reward"])
        self.assertEqual(1.0, cheaper_pass["pass_reward"])
        self.assertEqual(0.0, cheaper_pass["fail_penalty"])

    def test_reward_v1_formula_is_preserved(self) -> None:
        components = _compute_reward_components(
            pass_at_k=1.0,
            reference_completion_tokens=120,
            completion_tokens=240,
            completion_token_budget=180,
            reward_config=_reward_config_with_version("v1"),
        )

        self.assertEqual("v1", components["reward_version"])
        self.assertAlmostEqual(1.0 - (0.08 * min(240 / 180, 4.0)), components["total_reward"])

    def test_reward_v2_ignores_reference_token_inputs(self) -> None:
        left = _compute_reward_components(
            pass_at_k=1.0,
            reference_completion_tokens=50,
            completion_tokens=240,
            completion_token_budget=60,
            reward_config=_reward_config_with_version("v2"),
        )
        right = _compute_reward_components(
            pass_at_k=1.0,
            reference_completion_tokens=500,
            completion_tokens=240,
            completion_token_budget=600,
            reward_config=_reward_config_with_version("v2"),
        )

        self.assertEqual(left["total_reward"], right["total_reward"])

    def test_reward_v2_passing_reward_decreases_as_completion_tokens_grow(self) -> None:
        short_completion = _compute_reward_components(
            pass_at_k=1.0,
            reference_completion_tokens=120,
            completion_tokens=2,
            completion_token_budget=1,
            reward_config=_reward_config_with_version("v2"),
        )
        long_completion = _compute_reward_components(
            pass_at_k=1.0,
            reference_completion_tokens=120,
            completion_tokens=256,
            completion_token_budget=10_000,
            reward_config=_reward_config_with_version("v2"),
        )

        self.assertGreater(short_completion["total_reward"], long_completion["total_reward"])

    def test_reward_v2_prefers_passing_sample_over_failure_at_same_completion_tokens(self) -> None:
        pass_case = _compute_reward_components(
            pass_at_k=1.0,
            reference_completion_tokens=120,
            completion_tokens=2,
            completion_token_budget=1,
            reward_config=_reward_config_with_version("v2"),
        )
        fail_case = _compute_reward_components(
            pass_at_k=0.0,
            reference_completion_tokens=120,
            completion_tokens=2,
            completion_token_budget=10_000,
            reward_config=_reward_config_with_version("v2"),
        )

        self.assertGreater(pass_case["total_reward"], fail_case["total_reward"])

    def test_reward_v2_failure_includes_base_penalty(self) -> None:
        reward_config = RewardConfig()
        completion_tokens = 2
        log_cost = min(math.log1p(completion_tokens), reward_config.v2_log_cost_cap)

        fail_case = _compute_reward_components(
            pass_at_k=0.0,
            reference_completion_tokens=120,
            completion_tokens=completion_tokens,
            completion_token_budget=10_000,
            reward_config=reward_config,
        )

        self.assertAlmostEqual(
            reward_config.v2_fail_base_reward - (reward_config.v2_fail_log_cost_weight * log_cost),
            fail_case["total_reward"],
        )

    def test_reward_unknown_version_raises(self) -> None:
        reward_config = _reward_config_with_version("v3")

        with self.assertRaises(ValueError):
            _compute_reward_components(
                pass_at_k=1.0,
                reference_completion_tokens=120,
                completion_tokens=2,
                completion_token_budget=1,
                reward_config=reward_config,
            )

        with self.assertRaises(ValueError):
            cli._warm_start_reward_from_row(
                {
                    "pass_at_k": 1.0,
                    "completion_tokens": 50,
                    "reference_completion_tokens": 20,
                    "completion_token_budget": 30,
                },
                config=replace(DEFAULT_EXPERIMENT_CONFIG, reward=reward_config),
            )

        with self.assertRaises(ValueError):
            reward_formula_description(reward_config)

        with self.assertRaises(ValueError):
            reward_weight_dict(reward_config)

    def test_reward_v2_diagnostics_exclude_legacy_cost_weights(self) -> None:
        reward_config = _reward_config_with_version("v2")
        components = _compute_reward_components(
            pass_at_k=1.0,
            reference_completion_tokens=120,
            completion_tokens=2,
            completion_token_budget=1,
            reward_config=reward_config,
        )

        self.assertEqual("v2", components["reward_version"])
        self.assertIn("pass_log_cost_weight", components)
        self.assertIn("fail_log_cost_weight", components)
        self.assertIn("fail_base_reward", components)
        self.assertNotIn("pass_cost_weight", components)
        self.assertNotIn("fail_cost_weight", components)

        from codegen_agent.workflow.rewarding import reward_weight_dict

        weights = reward_weight_dict(reward_config)
        self.assertEqual("v2", weights["reward_version"])
        self.assertIn("pass_log_cost_weight", weights)
        self.assertIn("fail_log_cost_weight", weights)
        self.assertIn("fail_base_reward", weights)
        self.assertNotIn("pass_cost_weight", weights)
        self.assertNotIn("fail_cost_weight", weights)

    def test_reward_never_scores_failure_above_success(self) -> None:
        success = _compute_reward_components(
            pass_at_k=1.0,
            reference_completion_tokens=120,
            completion_tokens=240,
            completion_token_budget=180,
            reward_config=RewardConfig(),
        )
        failure = _compute_reward_components(
            pass_at_k=0.0,
            reference_completion_tokens=120,
            completion_tokens=60,
            completion_token_budget=180,
            reward_config=RewardConfig(),
        )

        self.assertGreater(success["total_reward"], failure["total_reward"])
        self.assertGreaterEqual(success["total_reward"], 0.0)
        self.assertLessEqual(failure["total_reward"], 0.0)

    def test_record_budget_anchor_uses_visible_task_not_hidden_solution(self) -> None:
        task = CodeGenTask(
            dataset_name="toy-dataset",
            task_id="visible-anchor",
            prompt="Implement task_func.",
            prompt_source="prompt",
            canonical_solution=" ".join(["hidden"] * 200),
            code_prompt="def task_func(x):\n    pass\n",
            entry_point="task_func",
            test="def check(candidate):\n    assert candidate(1) == 1\n",
        )
        model_client = _ScriptedModelClient(["GOOD_CODE"])
        runner = ExperimentRunner(
            model_client=model_client,
            selector=_FixedSelector("A0"),
            generation_config=GenerationConfig(max_new_tokens=64),
            arms=get_partition_arms(["A0", "A2"]),
            evaluator=_evaluate_by_output,
        )

        records = runner.run_bara(
            [task],
            num_samples_per_task=1,
            pass_k=1,
            evaluation_timeout_sec=1,
        )

        expected_visible_tokens = model_client.count_tokens(task.prompt + "\n" + task.code_prompt)
        self.assertEqual(expected_visible_tokens, records[0].reference_completion_tokens)
        self.assertEqual("v2", records[0].metadata["reward_version"])
        self.assertEqual("v2", records[0].metadata["reward_weights"]["reward_version"])
        self.assertEqual("v2", records[0].reward_components["reward_version"])
        self.assertEqual("v2", records[0].to_dict()["metadata"]["reward_version"])

    def test_online_rank_and_allocate_marks_over_cap_attempt_failed_and_charges_allocated_cap(self) -> None:
        task = _task(
            "bara-over-cap",
            "Return the input from task_func.",
            entry_point="task_func",
            dataset_name="bigcodebench-v0.1.4",
        )

        class _RankedSelector:
            def __init__(self) -> None:
                self.observations: list[tuple[str, bool]] = []

            def select(self, context: list[float]) -> BanditDecision:
                return BanditDecision(
                    arm_id="A0",
                    scores={"A0": 1.0, "A1": 0.5},
                    expected_rewards={"A0": 1.0, "A1": 0.5},
                    uncertainty_bonuses={"A0": 0.0, "A1": 0.0},
                    pull_counts={"A0": 0, "A1": 0},
                    details={"ranked_arms": ["A0", "A1"]},
                )

            def observe(self, arm_id: str, context: list[float], reward: float, *, solved: bool, metadata=None) -> None:
                del context, reward, metadata
                self.observations.append((arm_id, solved))

        class _FixedBudgetAllocator:
            def __init__(self) -> None:
                self.observations: list[tuple[str, int, int]] = []

            def allocate(self, *, arm_id, context, ranked_arms, remaining_budget):
                del context, ranked_arms, remaining_budget
                return {
                    "arm_id": arm_id,
                    "allocated_completion_token_cap": 50 if arm_id == "A0" else 500,
                    "allocation_mode": "test_fixed",
                }

            def observe(self, arm_id, context, *, token_usage, token_cap, metadata=None):
                del context, metadata
                self.observations.append((arm_id, int(token_usage), int(token_cap)))

        class _MismatchedPromptTokenModelClient(_ScriptedModelClient):
            def __init__(self) -> None:
                super().__init__(["GOOD_CODE", "[PLAN]\n1. Return the input.\n[CODE]\nGOOD_CODE"])
                self._prompt_tokens_by_call = [100, 1]

            def count_tokens(self, text: str) -> int:
                del text
                return 1

            def generate_batch(self, prompts, config, *, n=1, seed=None):
                del seed
                self.batch_prompt_counts.append(len(prompts))
                outputs = []
                for prompt in prompts:
                    self.prompt_history.append(prompt)
                    self.max_new_tokens_history.append(config.max_new_tokens)
                    self.stop_history.append(tuple(config.stop))
                    text = self._outputs.pop(0)
                    prompt_tokens = self._prompt_tokens_by_call.pop(0)
                    self.calls.append((prompt.arm_id, len(text)))
                    outputs.append(
                        [
                            ModelOutput(
                                raw_output=text,
                                cleaned_output=text,
                                completion_tokens=max(len(text.split()), 1),
                                prompt_tokens=prompt_tokens,
                            )
                        ]
                    )
                return outputs

        selector = _RankedSelector()
        budget_allocator = _FixedBudgetAllocator()
        model_client = _MismatchedPromptTokenModelClient()
        runner = ExperimentRunner(
            model_client=model_client,
            selector=selector,
            generation_config=GenerationConfig(max_new_tokens=4096),
            bara_config=replace(
                DEFAULT_EXPERIMENT_CONFIG.bara,
                rank_and_allocate=True,
                per_task_completion_token_budget=1000,
            ),
            budget_allocator=budget_allocator,
            arms=get_partition_arms(["A0", "A1"]),
            evaluator=_evaluate_by_output,
        )
        runner.feature_extractor = type(
            "FeatureExtractor",
            (),
            {"extract": lambda self, task: FeatureExtractionResult(["f1"], [1.0], {})},
        )()

        records = runner.run_bara(
            [task],
            num_samples_per_task=1,
            pass_k=1,
            evaluation_timeout_sec=1,
        )

        trace = records[0].selection_reason["bara_execution_trace"]
        self.assertEqual("A1", records[0].arm_id)
        self.assertEqual(["A0", "A1"], [item["arm_id"] for item in trace])
        self.assertEqual(True, trace[0]["over_allocated_token_cap"])
        self.assertEqual(False, trace[0]["passed"])
        self.assertEqual(50, trace[0]["charged_token_usage"])
        self.assertEqual([("A0", False), ("A1", True)], selector.observations)
        self.assertEqual(("A0", 50, 50), budget_allocator.observations[0])
        self.assertEqual(12, model_client.max_new_tokens_history[0])

    def test_record_metadata_tracks_v1_reward_version_and_formula(self) -> None:
        task = CodeGenTask(
            dataset_name="toy-dataset",
            task_id="visible-anchor-v1",
            prompt="Implement task_func.",
            prompt_source="prompt",
            canonical_solution=" ".join(["hidden"] * 200),
            code_prompt="def task_func(x):\n    pass\n",
            entry_point="task_func",
            test="def check(candidate):\n    assert candidate(1) == 1\n",
        )
        model_client = _ScriptedModelClient(["GOOD_CODE"])
        reward_config = _reward_config_with_version("v1")
        runner = ExperimentRunner(
            model_client=model_client,
            selector=_FixedSelector("A0"),
            generation_config=GenerationConfig(max_new_tokens=64),
            reward_config=reward_config,
            arms=get_partition_arms(["A0", "A2"]),
            evaluator=_evaluate_by_output,
        )

        records = runner.run_bara(
            [task],
            num_samples_per_task=1,
            pass_k=1,
            evaluation_timeout_sec=1,
        )

        self.assertEqual("v1", records[0].metadata["reward_version"])
        self.assertEqual(reward_formula_description(reward_config), records[0].metadata["reward_formula"])
        self.assertEqual("v1", records[0].metadata["reward_weights"]["reward_version"])
        self.assertEqual("v1", records[0].reward_components["reward_version"])
        self.assertEqual("v1", records[0].to_dict()["metadata"]["reward_version"])


if __name__ == "__main__":
    unittest.main()
