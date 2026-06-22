from __future__ import annotations

import math
import re

from ..config import RewardConfig
from ..features import visible_task_prompt
from ..types import CodeGenTask


_KNOWN_REWARD_VERSIONS = {"v1", "v2", "binary", "token_sqrt"}


def _validate_reward_version(reward_version: str) -> str:
    if reward_version not in _KNOWN_REWARD_VERSIONS:
        raise ValueError(f"Unknown reward_version: {reward_version}")
    return reward_version


def task_completion_token_budget(
    *,
    reference_completion_tokens: int,
    budget_multiplier: float,
    fixed_budget: int | None = None,
) -> int:
    if fixed_budget is not None:
        resolved_fixed_budget = int(fixed_budget)
        if resolved_fixed_budget <= 0:
            raise ValueError("fixed_budget must be positive when provided.")
        return resolved_fixed_budget
    if budget_multiplier <= 0.0:
        raise ValueError("budget_multiplier must be positive.")
    reference_completion_tokens = max(reference_completion_tokens, 1)
    return max(1, int(math.ceil(reference_completion_tokens * budget_multiplier)))


def accounted_prompt_tokens(
    *,
    prompt_tokens: int,
    reward_config: RewardConfig,
) -> int:
    prompt_tokens = max(int(prompt_tokens), 0)
    prompt_weight = float(getattr(reward_config, "prompt_token_cost_weight", 1.0))
    if prompt_weight < 0.0:
        raise ValueError("prompt_token_cost_weight must be non-negative.")
    return int(math.ceil(prompt_tokens * prompt_weight))


def accounted_completion_tokens(
    *,
    completion_tokens: int,
    reward_config: RewardConfig,
) -> int:
    completion_tokens = max(int(completion_tokens), 0)
    completion_weight = float(getattr(reward_config, "completion_token_cost_weight", 1.0))
    if completion_weight < 0.0:
        raise ValueError("completion_token_cost_weight must be non-negative.")
    return int(math.ceil(completion_tokens * completion_weight))


def accounted_total_tokens(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    reward_config: RewardConfig,
) -> int:
    prompt_cost = accounted_prompt_tokens(
        prompt_tokens=prompt_tokens,
        reward_config=reward_config,
    )
    completion_cost = accounted_completion_tokens(
        completion_tokens=completion_tokens,
        reward_config=reward_config,
    )
    if getattr(reward_config, "include_prompt_tokens_in_cost", True):
        return prompt_cost + completion_cost
    return completion_cost


def knn_total_tokens(
    *,
    prompt_tokens_excluding_system: int,
    completion_tokens: int,
    reward_config: RewardConfig,
) -> int:
    return accounted_total_tokens(
        prompt_tokens=max(int(prompt_tokens_excluding_system), 0),
        completion_tokens=max(int(completion_tokens), 0),
        reward_config=reward_config,
    )


def estimate_token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", str(text or ""), flags=re.UNICODE))


def remaining_raw_completion_token_cap(
    *,
    token_budget: int | None,
    prompt_tokens: int,
    reward_config: RewardConfig,
) -> int | None:
    if token_budget is None:
        return None
    remaining_budget = int(token_budget)
    if getattr(reward_config, "include_prompt_tokens_in_cost", True):
        remaining_budget -= accounted_prompt_tokens(
            prompt_tokens=prompt_tokens,
            reward_config=reward_config,
        )
    if remaining_budget <= 0:
        return 0
    completion_weight = float(getattr(reward_config, "completion_token_cost_weight", 1.0))
    if completion_weight <= 0.0:
        raise ValueError("completion_token_cost_weight must be positive.")
    return max(int(math.floor(remaining_budget / completion_weight)), 0)


def compute_reward_components(
    *,
    pass_at_k: float,
    reference_completion_tokens: int,
    completion_tokens: int,
    completion_token_budget: int,
    reward_config: RewardConfig,
    prompt_tokens: int = 0,
    cost_tokens: int | None = None,
) -> dict[str, float | str]:
    reward_version = _validate_reward_version(getattr(reward_config, "reward_version", "v1"))
    completion_tokens = max(completion_tokens, 0)
    prompt_tokens = max(prompt_tokens, 0)
    accounted_completion = accounted_completion_tokens(
        completion_tokens=completion_tokens,
        reward_config=reward_config,
    )
    effective_cost_tokens = max(
        int(cost_tokens if cost_tokens is not None else accounted_total_tokens(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reward_config=reward_config,
        )),
        0,
    )

    if reward_version == "binary":
        total_reward = reward_config.success_reward if pass_at_k > 0.0 else 0.0
        return {
            "reward_version": reward_version,
            "total_reward": total_reward,
            "pass_at_k": pass_at_k,
            "pass_reward": total_reward,
            "fail_penalty": 0.0,
            "reference_completion_tokens": float(reference_completion_tokens),
            "prompt_tokens": float(prompt_tokens),
            "completion_tokens": float(completion_tokens),
            "accounted_completion_tokens": float(accounted_completion),
            "cost_tokens": float(effective_cost_tokens),
            "completion_token_budget": float(completion_token_budget),
            "success_reward": reward_config.success_reward,
        }

    if reward_version == "v2":
        log_cost_cap = getattr(reward_config, "v2_log_cost_cap", reward_config.max_normalized_completion_cost)
        pass_log_cost_weight = getattr(reward_config, "v2_pass_log_cost_weight", 0.0)
        fail_log_cost_weight = getattr(reward_config, "v2_fail_log_cost_weight", 0.0)
        fail_base_reward = getattr(reward_config, "v2_fail_base_reward", 0.0)
        log_cost = min(math.log1p(effective_cost_tokens), log_cost_cap)
        if pass_at_k > 0.0:
            pass_reward = reward_config.success_reward * pass_at_k
            fail_penalty = 0.0
            total_reward = pass_reward - (pass_log_cost_weight * log_cost)
        else:
            pass_reward = 0.0
            fail_penalty = fail_log_cost_weight * log_cost
            total_reward = fail_base_reward - fail_penalty
        return {
            "reward_version": reward_version,
            "total_reward": total_reward,
            "pass_at_k": pass_at_k,
            "pass_reward": pass_reward,
            "fail_penalty": fail_penalty,
            "fail_base_reward": fail_base_reward,
            "reference_completion_tokens": float(reference_completion_tokens),
            "prompt_tokens": float(prompt_tokens),
            "completion_tokens": float(completion_tokens),
            "accounted_completion_tokens": float(accounted_completion),
            "cost_tokens": float(effective_cost_tokens),
            "completion_token_budget": float(completion_token_budget),
            "log_cost": log_cost,
            "log_cost_cap": log_cost_cap,
            "success_reward": reward_config.success_reward,
            "pass_log_cost_weight": pass_log_cost_weight,
            "fail_log_cost_weight": fail_log_cost_weight,
        }

    if reward_version == "token_sqrt":
        budget_tokens = max(int(completion_token_budget), 1)
        normalized_cost = min(max(float(effective_cost_tokens) / float(budget_tokens), 0.0), 1.0)
        sqrt_cost = math.sqrt(normalized_cost)
        squared_cost = normalized_cost * normalized_cost
        if pass_at_k > 0.0:
            pass_reward = reward_config.success_reward * pass_at_k
            fail_penalty = squared_cost
            total_reward = max(pass_reward - fail_penalty, 0.0)
        else:
            pass_reward = 0.0
            fail_penalty = 0.0
            total_reward = 0.0
        return {
            "reward_version": reward_version,
            "total_reward": total_reward,
            "pass_at_k": pass_at_k,
            "pass_reward": pass_reward,
            "fail_penalty": fail_penalty,
            "reference_completion_tokens": float(reference_completion_tokens),
            "prompt_tokens": float(prompt_tokens),
            "completion_tokens": float(completion_tokens),
            "accounted_completion_tokens": float(accounted_completion),
            "cost_tokens": float(effective_cost_tokens),
            "completion_token_budget": float(completion_token_budget),
            "normalized_cost": normalized_cost,
            "sqrt_cost": sqrt_cost,
            "squared_cost": squared_cost,
            "success_reward": reward_config.success_reward,
        }

    reference_completion_tokens = max(reference_completion_tokens, 1)
    completion_token_budget = max(completion_token_budget, 1)
    normalized_reference_cost = effective_cost_tokens / reference_completion_tokens
    normalized_budget_cost = effective_cost_tokens / completion_token_budget
    normalized_completion_cost = min(
        normalized_budget_cost,
        reward_config.max_normalized_completion_cost,
    )
    if pass_at_k > 0.0:
        pass_reward = reward_config.success_reward * pass_at_k
        fail_penalty = 0.0
        total_reward = pass_reward
    else:
        pass_reward = 0.0
        fail_penalty = 0.0
        total_reward = 0.0
    return {
        "reward_version": reward_version,
        "total_reward": total_reward,
        "pass_at_k": pass_at_k,
        "pass_reward": pass_reward,
        "fail_penalty": fail_penalty,
        "reference_completion_tokens": float(reference_completion_tokens),
        "prompt_tokens": float(prompt_tokens),
        "completion_tokens": float(completion_tokens),
        "accounted_completion_tokens": float(accounted_completion),
        "cost_tokens": float(effective_cost_tokens),
        "completion_token_budget": float(completion_token_budget),
        "normalized_reference_cost": normalized_reference_cost,
        "normalized_completion_cost": normalized_completion_cost,
        "success_reward": reward_config.success_reward,
    }


def compute_repair_decision_reward(
    *,
    repaired_passed: bool,
    repair_completion_tokens: int,
    repair_completion_token_budget: int,
    reward_config: RewardConfig,
    repair_prompt_tokens: int = 0,
    repair_cost_tokens: int | None = None,
) -> float:
    repair_completion_tokens = max(repair_completion_tokens, 0)
    repair_prompt_tokens = max(repair_prompt_tokens, 0)
    repair_completion_token_budget = max(repair_completion_token_budget, 1)
    reward_version = _validate_reward_version(getattr(reward_config, "reward_version", "v1"))
    effective_cost_tokens = max(
        int(repair_cost_tokens if repair_cost_tokens is not None else accounted_total_tokens(
            prompt_tokens=repair_prompt_tokens,
            completion_tokens=repair_completion_tokens,
            reward_config=reward_config,
        )),
        0,
    )

    if reward_version == "binary":
        return reward_config.success_reward if repaired_passed else 0.0

    if reward_version == "v2":
        log_cost = min(
            math.log1p(effective_cost_tokens),
            reward_config.v2_log_cost_cap,
        )
        if repaired_passed:
            return reward_config.success_reward - (
                reward_config.v2_pass_log_cost_weight * log_cost
            )
        return reward_config.v2_fail_base_reward - (
            reward_config.v2_fail_log_cost_weight * log_cost
        )

    normalized_repair_cost = min(
        effective_cost_tokens / repair_completion_token_budget,
        reward_config.max_normalized_completion_cost,
    )
    if repaired_passed:
        return reward_config.success_reward
    return 0.0


def reward_formula_description(reward_config: RewardConfig | None = None) -> str:
    reward_config = reward_config or RewardConfig()
    reward_version = _validate_reward_version(getattr(reward_config, "reward_version", "v1"))
    if reward_version == "binary":
        return "binary: pass gets success_reward; fail gets 0; token cost is ignored"
    if reward_version == "v2":
        return (
            "v2: success gets success_reward * pass_at_k minus capped log-cost for passing samples; "
            "failures get fixed base penalty plus capped log-cost penalty"
        )
    if reward_version == "token_sqrt":
        return (
            "token_sqrt: successes get success_reward * pass_at_k minus "
            "(cost_tokens / completion_token_budget)^2; failures get 0"
        )
    return (
        "v1: success gets success_reward * pass_at_k minus capped normalized completion cost; "
        "failures get capped normalized completion cost penalty only"
    )


def reward_weight_dict(reward_config: RewardConfig | None = None) -> dict[str, float | str]:
    reward_config = reward_config or RewardConfig()
    reward_version = _validate_reward_version(getattr(reward_config, "reward_version", "v1"))
    base = {
        "reward_version": reward_version,
        "completion_budget_multiplier": reward_config.completion_budget_multiplier,
        "max_normalized_completion_cost": reward_config.max_normalized_completion_cost,
        "include_prompt_tokens_in_cost": bool(getattr(reward_config, "include_prompt_tokens_in_cost", True)),
        "prompt_token_cost_weight": float(getattr(reward_config, "prompt_token_cost_weight", 1.0)),
        "completion_token_cost_weight": float(getattr(reward_config, "completion_token_cost_weight", 1.0)),
    }
    if reward_version == "binary":
        return {
            **base,
            "success_reward": reward_config.success_reward,
        }
    if reward_version == "v2":
        return {
            **base,
            "success_reward": reward_config.success_reward,
            "pass_log_cost_weight": reward_config.v2_pass_log_cost_weight,
            "fail_log_cost_weight": reward_config.v2_fail_log_cost_weight,
            "fail_base_reward": reward_config.v2_fail_base_reward,
            "log_cost_cap": reward_config.v2_log_cost_cap,
        }
    if reward_version == "token_sqrt":
        return {
            **base,
            "success_reward": reward_config.success_reward,
        }
    return {
        **base,
        "success_reward": reward_config.success_reward,
    }


def visible_task_budget_reference(task: CodeGenTask) -> str:
    parts = [visible_task_prompt(task)]
    if task.code_prompt:
        parts.append(task.code_prompt.strip())
    return "\n".join(part for part in parts if part)
