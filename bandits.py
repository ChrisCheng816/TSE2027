from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

from .types import BanditDecision


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return list(vector)
    return [value / norm for value in vector]


def _normal_log_survival(z_score: float) -> float:
    if not math.isfinite(z_score):
        return 0.0 if z_score < 0.0 else -math.inf
    if z_score < 8.0:
        survival = 1.0 - NormalDist().cdf(z_score)
        return math.log(max(survival, 1e-300))
    # Mills-ratio expansion keeps the right tail numerically stable.
    return (
        -0.5 * z_score * z_score
        - math.log(z_score)
        - 0.5 * math.log(2.0 * math.pi)
    )


def _lognormal_censored_negative_log_likelihood(
    *,
    log_uncensored: list[float],
    log_censored_lower_bounds: list[float],
    location: float,
    log_scale: float,
) -> float:
    if not math.isfinite(location) or not math.isfinite(log_scale):
        return math.inf
    if log_scale < -20.0 or log_scale > 20.0:
        return math.inf
    scale = math.exp(log_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        return math.inf

    nll = 0.0
    variance = scale * scale
    for value in log_uncensored:
        residual = value - location
        nll += log_scale + 0.5 * residual * residual / variance
    for lower_bound in log_censored_lower_bounds:
        z_score = (lower_bound - location) / scale
        log_survival = _normal_log_survival(z_score)
        if not math.isfinite(log_survival):
            return math.inf
        nll -= log_survival
    return nll


def _nelder_mead_2d(
    objective: Any,
    *,
    start: tuple[float, float],
    step: tuple[float, float],
    max_iterations: int = 120,
    tolerance: float = 1e-7,
) -> tuple[float, float, float, int, bool]:
    simplex = [
        (float(start[0]), float(start[1])),
        (float(start[0] + step[0]), float(start[1])),
        (float(start[0]), float(start[1] + step[1])),
    ]
    values = [float(objective(point)) for point in simplex]
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    for iteration in range(1, max_iterations + 1):
        ordered = sorted(zip(values, simplex), key=lambda item: item[0])
        values = [item[0] for item in ordered]
        simplex = [item[1] for item in ordered]
        if (
            max(abs(values[index] - values[0]) for index in range(1, 3)) < tolerance
            and max(
                math.hypot(simplex[index][0] - simplex[0][0], simplex[index][1] - simplex[0][1])
                for index in range(1, 3)
            )
            < math.sqrt(tolerance)
        ):
            return simplex[0][0], simplex[0][1], values[0], iteration, True

        centroid = (
            (simplex[0][0] + simplex[1][0]) / 2.0,
            (simplex[0][1] + simplex[1][1]) / 2.0,
        )
        worst = simplex[2]
        reflected = (
            centroid[0] + alpha * (centroid[0] - worst[0]),
            centroid[1] + alpha * (centroid[1] - worst[1]),
        )
        reflected_value = float(objective(reflected))

        if values[0] <= reflected_value < values[1]:
            simplex[2] = reflected
            values[2] = reflected_value
            continue

        if reflected_value < values[0]:
            expanded = (
                centroid[0] + gamma * (reflected[0] - centroid[0]),
                centroid[1] + gamma * (reflected[1] - centroid[1]),
            )
            expanded_value = float(objective(expanded))
            if expanded_value < reflected_value:
                simplex[2] = expanded
                values[2] = expanded_value
            else:
                simplex[2] = reflected
                values[2] = reflected_value
            continue

        if reflected_value < values[2]:
            contracted = (
                centroid[0] + rho * (reflected[0] - centroid[0]),
                centroid[1] + rho * (reflected[1] - centroid[1]),
            )
        else:
            contracted = (
                centroid[0] + rho * (worst[0] - centroid[0]),
                centroid[1] + rho * (worst[1] - centroid[1]),
            )
        contracted_value = float(objective(contracted))
        if contracted_value < values[2]:
            simplex[2] = contracted
            values[2] = contracted_value
            continue

        best = simplex[0]
        simplex = [
            best,
            (
                best[0] + sigma * (simplex[1][0] - best[0]),
                best[1] + sigma * (simplex[1][1] - best[1]),
            ),
            (
                best[0] + sigma * (simplex[2][0] - best[0]),
                best[1] + sigma * (simplex[2][1] - best[1]),
            ),
        ]
        values = [float(objective(point)) for point in simplex]

    ordered = sorted(zip(values, simplex), key=lambda item: item[0])
    best_value, best_point = ordered[0]
    return best_point[0], best_point[1], best_value, max_iterations, False


def _fit_censored_lognormal(
    *,
    log_uncensored: list[float],
    log_censored_lower_bounds: list[float],
) -> tuple[float, float, dict[str, Any]] | None:
    if not log_uncensored:
        return None
    all_log_values = [*log_uncensored, *log_censored_lower_bounds]
    initial_location = sum(all_log_values) / len(all_log_values)
    initial_variance = sum((value - initial_location) ** 2 for value in all_log_values) / len(all_log_values)
    initial_scale = math.sqrt(max(initial_variance, 1e-12))
    if not log_censored_lower_bounds:
        return (
            initial_location,
            initial_scale,
            {
                "lognormal_fit_method": "lognormal_mle",
                "lognormal_negative_log_likelihood": _lognormal_censored_negative_log_likelihood(
                    log_uncensored=log_uncensored,
                    log_censored_lower_bounds=[],
                    location=initial_location,
                    log_scale=math.log(initial_scale),
                ),
                "lognormal_optimizer_converged": True,
                "lognormal_optimizer_iterations": 0,
            },
        )

    start = (initial_location, math.log(initial_scale))
    step = (max(initial_scale * 0.25, 0.10), 0.25)

    def objective(point: tuple[float, float]) -> float:
        return _lognormal_censored_negative_log_likelihood(
            log_uncensored=log_uncensored,
            log_censored_lower_bounds=log_censored_lower_bounds,
            location=point[0],
            log_scale=point[1],
        )

    location, log_scale, objective_value, iterations, converged = _nelder_mead_2d(
        objective,
        start=start,
        step=step,
    )
    if not math.isfinite(objective_value):
        return None
    scale = math.exp(log_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    return (
        location,
        max(scale, 1e-6),
        {
            "lognormal_fit_method": "lognormal_mle",
            "lognormal_negative_log_likelihood": objective_value,
            "lognormal_optimizer_converged": converged,
            "lognormal_optimizer_iterations": iterations,
        },
    )


def normalize_selector_algorithm(value: str | None) -> str:
    normalized = str(value or "knn").strip().lower().replace("-", "").replace("_", "")
    if normalized in {"knn", "knnucb", "knnrewardsumucb", "v3"}:
        return "knn"
    raise ValueError(f"Unsupported selector algorithm: {value}")


def selector_apply_update(
    selector: Any,
    *,
    arm_id: str,
    context: list[float],
    reward: float,
    solved: bool,
    metadata: dict[str, Any] | None = None,
) -> None:
    observe = getattr(selector, "observe", None)
    if callable(observe):
        observe(
            arm_id,
            context,
            reward,
            solved=solved,
            metadata=metadata,
        )
        return
    selector.update(arm_id, context, reward)


def selector_apply_preload(
    selector: Any,
    *,
    arm_id: str,
    context: list[float],
    reward: float,
    solved: bool,
    metadata: dict[str, Any] | None = None,
) -> None:
    preload_observation = getattr(selector, "preload_observation", None)
    if callable(preload_observation):
        preload_observation(
            arm_id,
            context,
            reward,
            solved=solved,
            metadata=metadata,
        )
        return
    selector.preload(arm_id, context, reward)


@dataclass(frozen=True)
class _TaskObservation:
    context: list[float]
    arm_id: str
    reward: float
    solved: bool
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ArmBudgetObservation:
    context: list[float]
    arm_id: str
    token_usage: float
    uncensored: bool
    token_cap: float | None = None
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class KNNRewardSumUCBSelector:
    algorithm = "knn"

    def __init__(
        self,
        arm_ids: list[str] | tuple[str, ...],
        feature_dim: int,
        *,
        k: int = 20,
        beta: float = 0.5,
        reward_max: float = 1.0,
        distance_metric: str = "cosine",
        ucb_count_scope: str = "local",
        min_pulls_per_arm: int = 1,
        random_seed: int = 0,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if k <= 0:
            raise ValueError("k must be positive.")
        if reward_max <= 0.0:
            raise ValueError("reward_max must be positive.")
        if min_pulls_per_arm <= 0:
            raise ValueError("min_pulls_per_arm must be positive.")
        normalized_distance_metric = str(distance_metric or "cosine").strip().lower()
        if normalized_distance_metric not in {"cosine", "euclidean"}:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        normalized_ucb_count_scope = str(ucb_count_scope or "local").strip().lower()
        if normalized_ucb_count_scope not in {"local", "global_first"}:
            raise ValueError(f"Unsupported ucb_count_scope: {ucb_count_scope}")

        self.arm_ids = list(arm_ids)
        self.feature_dim = feature_dim
        self.k = k
        self.beta = beta
        self.reward_max = reward_max
        self.distance_metric = normalized_distance_metric
        self.ucb_count_scope = normalized_ucb_count_scope
        self.min_pulls_per_arm = min_pulls_per_arm
        self._arm_priority = {arm_id: index for index, arm_id in enumerate(self.arm_ids)}
        self._global_pulls = {arm_id: 0 for arm_id in self.arm_ids}
        self._first_arm_pulls = {arm_id: 0 for arm_id in self.arm_ids}
        self._task_history: list[_TaskObservation] = []
        self._random = random.Random(random_seed)

    def _rank_by_score_random_ties(self, scores: dict[str, float]) -> list[str]:
        ranked: list[str] = []
        ordered = sorted(
            scores.items(),
            key=lambda item: (-item[1], self._arm_priority[item[0]]),
        )
        group: list[str] = []
        group_score: float | None = None

        def flush_group() -> None:
            if not group:
                return
            if len(group) > 1:
                random.SystemRandom().shuffle(group)
            ranked.extend(group)
            group.clear()

        for arm_id, score in ordered:
            if group_score is None or math.isclose(score, group_score):
                group.append(arm_id)
                group_score = score if group_score is None else group_score
                continue
            flush_group()
            group.append(arm_id)
            group_score = score
        flush_group()
        return ranked

    def select(self, context: list[float]) -> BanditDecision:
        prepared_context = self._prepare_context(context)
        total_rounds = sum(self._global_pulls.values())

        if not self._task_history:
            ranked_arms = list(self.arm_ids)
            self._random.shuffle(ranked_arms)
            scores = {
                arm_id: float(len(ranked_arms) - index)
                for index, arm_id in enumerate(ranked_arms)
            }
            return BanditDecision(
                arm_id=ranked_arms[0],
                scores=scores,
                expected_rewards={arm_id: 0.0 for arm_id in self.arm_ids},
                uncertainty_bonuses={arm_id: 0.0 for arm_id in self.arm_ids},
                pull_counts=dict(self._first_arm_pulls),
                details={
                    "selector_algorithm": self.algorithm,
                    "ranked_arms": ranked_arms,
                    "neighbor_count": 0,
                    "requested_k": self.k,
                    "distance_metric": self.distance_metric,
                    "ucb_count_scope": self.ucb_count_scope,
                    "local_reward_sums": {arm_id: 0.0 for arm_id in self.arm_ids},
                    "local_reward_counts": {arm_id: 0 for arm_id in self.arm_ids},
                    "local_reward_means": {arm_id: 0.0 for arm_id in self.arm_ids},
                    "nearest_neighbors": [],
                    "cold_start_randomized": True,
                    "forced_exploration": False,
                    "first_arm_pull_counts": dict(self._first_arm_pulls),
                    "observation_counts": dict(self._global_pulls),
                },
            )

        forced_candidate_arm_ids = [
            arm_id for arm_id in self.arm_ids if self._first_arm_pulls[arm_id] < self.min_pulls_per_arm
        ]
        forced_arm_id: str | None = None
        forced_scores: dict[str, float] | None = None
        if forced_candidate_arm_ids:
            forced_arm_id = self._random.choice(forced_candidate_arm_ids)
            forced_scores = {
                candidate_arm_id: (
                    1_000_000.0
                    if candidate_arm_id == forced_arm_id
                    else 0.0
                )
                for candidate_arm_id in self.arm_ids
            }

        if forced_arm_id is not None and forced_scores is not None:
            ranked_arms = self._rank_by_score_random_ties(forced_scores)
            return BanditDecision(
                arm_id=forced_arm_id,
                scores=forced_scores,
                expected_rewards={arm_id: 0.0 for arm_id in self.arm_ids},
                uncertainty_bonuses={arm_id: 0.0 for arm_id in self.arm_ids},
                pull_counts=dict(self._first_arm_pulls),
                details={
                    "selector_algorithm": self.algorithm,
                    "ranked_arms": ranked_arms,
                    "neighbor_count": min(self.k, self._unique_task_history_count()),
                    "requested_k": self.k,
                    "distance_metric": self.distance_metric,
                    "ucb_count_scope": self.ucb_count_scope,
                    "local_reward_sums": {arm_id: 0.0 for arm_id in self.arm_ids},
                    "local_reward_counts": {arm_id: 0 for arm_id in self.arm_ids},
                    "local_reward_means": {arm_id: 0.0 for arm_id in self.arm_ids},
                    "nearest_neighbors": [],
                    "cold_start_randomized": False,
                    "forced_exploration": True,
                    "forced_arm_id": forced_arm_id,
                    "forced_candidate_arm_ids": forced_candidate_arm_ids,
                    "min_pulls_per_arm": self.min_pulls_per_arm,
                    "first_arm_pull_counts": dict(self._first_arm_pulls),
                    "observation_counts": dict(self._global_pulls),
                },
            )

        ranked_neighbor_tasks = self._ranked_neighbor_tasks(prepared_context)
        neighbor_count = min(self.k, len(ranked_neighbor_tasks))
        selected_neighbor_tasks = ranked_neighbor_tasks[:neighbor_count]
        ranked_neighbors = [
            (distance, observation)
            for distance, _, observations in selected_neighbor_tasks
            for observation in observations
        ]

        local_reward_sums = {arm_id: 0.0 for arm_id in self.arm_ids}
        local_reward_counts = {arm_id: 0 for arm_id in self.arm_ids}
        for _, observation in ranked_neighbors:
            local_reward_sums[observation.arm_id] += observation.reward
            local_reward_counts[observation.arm_id] += 1
        local_reward_means = {
            arm_id: (
                local_reward_sums[arm_id] / local_reward_counts[arm_id]
                if local_reward_counts[arm_id] > 0
                else 0.0
            )
            for arm_id in self.arm_ids
        }

        scores: dict[str, float] = {}
        uncertainty_bonuses: dict[str, float] = {}
        ucb_denominator_counts: dict[str, int] = {}
        for arm_id in self.arm_ids:
            denominator_count = (
                local_reward_counts[arm_id]
                if self.ucb_count_scope == "local"
                else self._first_arm_pulls[arm_id]
            )
            exploration_bonus = self.beta * self.reward_max * math.sqrt(
                math.log(total_rounds + 2.0) / (denominator_count + 1.0)
            )
            ucb_denominator_counts[arm_id] = denominator_count
            uncertainty_bonuses[arm_id] = exploration_bonus
            scores[arm_id] = local_reward_means[arm_id] + exploration_bonus

        ranked_arms = self._rank_by_score_random_ties(scores)
        return BanditDecision(
            arm_id=ranked_arms[0],
            scores=scores,
            expected_rewards=local_reward_means,
            uncertainty_bonuses=uncertainty_bonuses,
            pull_counts=dict(self._first_arm_pulls),
            details={
                "selector_algorithm": self.algorithm,
                "ranked_arms": ranked_arms,
                "neighbor_count": neighbor_count,
                "requested_k": self.k,
                "distance_metric": self.distance_metric,
                "ucb_count_scope": self.ucb_count_scope,
                "local_reward_sums": local_reward_sums,
                "local_reward_counts": local_reward_counts,
                "local_reward_means": local_reward_means,
                "ucb_denominator_counts": ucb_denominator_counts,
                "nearest_neighbor_task_ids": [
                    task_id for _, task_id, _ in selected_neighbor_tasks
                ],
                "nearest_neighbors": [
                    {
                        "task_id": observation.task_id,
                        "arm_id": observation.arm_id,
                        "reward": observation.reward,
                        "solved": observation.solved,
                        "distance": distance,
                    }
                    for distance, observation in ranked_neighbors
                ],
                "cold_start_randomized": False,
                "forced_exploration": False,
                "first_arm_pull_counts": dict(self._first_arm_pulls),
                "observation_counts": dict(self._global_pulls),
            },
        )

    def update(self, arm_id: str, context: list[float], reward: float) -> None:
        self.observe(arm_id, context, reward, solved=True, metadata=None)

    def preload(self, arm_id: str, context: list[float], reward: float) -> None:
        self.preload_observation(arm_id, context, reward, solved=True, metadata=None)

    def observe(
        self,
        arm_id: str,
        context: list[float],
        reward: float,
        *,
        solved: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        prepared_context = self._prepare_context(context)
        self._record_observation(
            arm_id,
            prepared_context,
            reward,
            solved=solved,
            metadata=metadata,
        )

    def preload_observation(
        self,
        arm_id: str,
        context: list[float],
        reward: float,
        *,
        solved: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        prepared_context = self._prepare_context(context)
        self._record_observation(
            arm_id,
            prepared_context,
            reward,
            solved=solved,
            metadata=metadata,
        )

    def _record_observation(
        self,
        arm_id: str,
        context: list[float],
        reward: float,
        *,
        solved: bool,
        metadata: dict[str, Any] | None,
    ) -> None:
        if arm_id not in self._global_pulls:
            raise KeyError(f"Unknown arm_id: {arm_id}")
        self._global_pulls[arm_id] += 1
        metadata = dict(metadata or {})
        if self._counts_as_first_arm_pull(metadata):
            self._first_arm_pulls[arm_id] += 1
        self._task_history.append(
            _TaskObservation(
                context=list(context),
                arm_id=arm_id,
                reward=float(reward),
                solved=bool(solved),
                task_id=(str(metadata.get("task_id")) if metadata.get("task_id") is not None else None),
                metadata=metadata,
            )
        )

    def _counts_as_first_arm_pull(self, metadata: dict[str, Any]) -> bool:
        if metadata.get("order_run") is True:
            if metadata.get("order_run_is_first_attempt") is not None:
                return bool(metadata.get("order_run_is_first_attempt"))
            rank = metadata.get("order_run_rank")
            if rank is not None:
                try:
                    return int(rank) == 1
                except (TypeError, ValueError):
                    return False
            return False
        return True

    def _prepare_context(self, context: list[float]) -> list[float]:
        if len(context) != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {len(context)}."
            )
        normalized = [float(value) for value in context]
        if self.distance_metric == "cosine":
            return _l2_normalize(normalized)
        return normalized

    def _unique_task_history_count(self) -> int:
        return len({self._task_group_key(index, observation) for index, observation in enumerate(self._task_history)})

    def _ranked_neighbor_tasks(
        self,
        context: list[float],
    ) -> list[tuple[float, str | None, list[_TaskObservation]]]:
        groups: dict[object, dict[str, Any]] = {}
        for index, observation in enumerate(self._task_history):
            key = self._task_group_key(index, observation)
            distance = self._distance(context, observation.context)
            group = groups.setdefault(
                key,
                {
                    "distance": distance,
                    "task_id": observation.task_id,
                    "observations": [],
                    "first_index": index,
                },
            )
            if distance < float(group["distance"]):
                group["distance"] = distance
            group["observations"].append(observation)

        return [
            (
                float(group["distance"]),
                group["task_id"],
                list(group["observations"]),
            )
            for group in sorted(
                groups.values(),
                key=lambda candidate: (
                    float(candidate["distance"]),
                    int(candidate["first_index"]),
                ),
            )
        ]

    def _task_group_key(self, index: int, observation: _TaskObservation) -> object:
        if observation.task_id is not None:
            return ("task", observation.task_id)
        return ("observation", index)

    def _distance(self, left: list[float], right: list[float]) -> float:
        if self.distance_metric == "euclidean":
            return math.sqrt(sum((a - b) * (a - b) for a, b in zip(left, right)))
        return 1.0 - _dot(left, right)


class KNNArmBudgetAllocator:
    def __init__(
        self,
        arm_ids: list[str] | tuple[str, ...],
        feature_dim: int,
        *,
        k: int = 20,
        distance_metric: str = "cosine",
        risk_tolerance: float = 0.01,
        arm_prior_weights: dict[str, float] | None = None,
        allocation_strategy: str = "mle",
        budget_quantile: float = 0.80,
        min_budget_neighbors: int = 5,
        censored_rate_threshold: float = 0.25,
        censored_cap_multiplier: float = 1.25,
        max_prior_multiplier: float = 2.0,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if k <= 0:
            raise ValueError("k must be positive.")
        normalized_distance_metric = str(distance_metric or "cosine").strip().lower()
        if normalized_distance_metric not in {"cosine", "euclidean"}:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        if not (0.0 < float(risk_tolerance) < 1.0):
            raise ValueError("risk_tolerance must be in (0, 1).")
        normalized_allocation_strategy = str(allocation_strategy or "mle").strip().lower()
        if normalized_allocation_strategy == "quantile":
            normalized_allocation_strategy = "empirical_quantile"
        if normalized_allocation_strategy == "budget":
            normalized_allocation_strategy = "static_prior"
        if normalized_allocation_strategy not in {
            "mle",
            "equal_split",
            "static_prior",
            "empirical_quantile",
            "uncensored_lognormal_mle",
        }:
            raise ValueError(f"Unsupported allocation_strategy: {allocation_strategy}")
        if not (0.0 < float(budget_quantile) <= 1.0):
            raise ValueError("budget_quantile must be in (0, 1].")
        if min_budget_neighbors <= 0:
            raise ValueError("min_budget_neighbors must be positive.")
        if not (0.0 <= float(censored_rate_threshold) <= 1.0):
            raise ValueError("censored_rate_threshold must be in [0, 1].")
        if float(censored_cap_multiplier) < 1.0:
            raise ValueError("censored_cap_multiplier must be at least 1.")
        if float(max_prior_multiplier) <= 0.0:
            raise ValueError("max_prior_multiplier must be positive.")

        self.arm_ids = list(arm_ids)
        self.feature_dim = feature_dim
        self.k = int(k)
        self.distance_metric = normalized_distance_metric
        self.risk_tolerance = float(risk_tolerance)
        self.allocation_strategy = normalized_allocation_strategy
        self.budget_quantile = float(budget_quantile)
        self.min_budget_neighbors = int(min_budget_neighbors)
        self.censored_rate_threshold = float(censored_rate_threshold)
        self.censored_cap_multiplier = float(censored_cap_multiplier)
        self.max_prior_multiplier = float(max_prior_multiplier)
        self._history_by_arm = {arm_id: [] for arm_id in self.arm_ids}
        default_prior_weights = {"A0": 1.0, "A1": 1.0, "A2": 2.0}
        resolved_prior_weights = dict(default_prior_weights)
        resolved_prior_weights.update({str(key): float(value) for key, value in (arm_prior_weights or {}).items()})
        self._prior_weights = resolved_prior_weights

    def observe(
        self,
        arm_id: str,
        context: list[float],
        *,
        token_usage: float,
        token_cap: float | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        prepared_context = self._prepare_context(context)
        if arm_id not in self._history_by_arm:
            raise KeyError(f"Unknown arm_id: {arm_id}")
        resolved_token_usage = max(float(token_usage), 0.0)
        resolved_token_cap = None if token_cap is None else max(float(token_cap), 0.0)
        uncensored = resolved_token_cap is None or resolved_token_usage < resolved_token_cap
        payload = dict(metadata or {})
        self._history_by_arm[arm_id].append(
            _ArmBudgetObservation(
                context=list(prepared_context),
                arm_id=arm_id,
                token_usage=resolved_token_usage,
                uncensored=uncensored,
                token_cap=resolved_token_cap,
                task_id=(str(payload.get("task_id")) if payload.get("task_id") is not None else None),
                metadata=payload,
            )
        )

    def preload_observation(
        self,
        arm_id: str,
        context: list[float],
        *,
        token_usage: float,
        token_cap: float | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.observe(
            arm_id,
            context,
            token_usage=token_usage,
            token_cap=token_cap,
            metadata=metadata,
        )

    def allocate(
        self,
        *,
        arm_id: str,
        context: list[float],
        ranked_arms: list[str] | tuple[str, ...],
        remaining_budget: int,
    ) -> dict[str, Any]:
        resolved_remaining_budget = max(int(remaining_budget), 0)
        if resolved_remaining_budget <= 0:
            return {
                "arm_id": arm_id,
                "allocated_completion_token_cap": 0,
                "allocation_mode": "empty_budget",
                "remaining_budget_before_arm": 0,
                "neighbor_count": 0,
                "requested_k": self.k,
                "distance_metric": self.distance_metric,
                "allocation_strategy": self.allocation_strategy,
            }
        if arm_id not in self._history_by_arm:
            raise KeyError(f"Unknown arm_id: {arm_id}")

        prepared_context = self._prepare_context(context)
        remaining_ranked_arms = [candidate for candidate in ranked_arms if candidate in self._history_by_arm]
        if arm_id not in remaining_ranked_arms:
            remaining_ranked_arms = [arm_id, *remaining_ranked_arms]

        if len(remaining_ranked_arms) == 1 or remaining_ranked_arms[-1] == arm_id:
            return {
                "arm_id": arm_id,
                "allocated_completion_token_cap": resolved_remaining_budget,
                "allocation_mode": "last_arm",
                "remaining_budget_before_arm": resolved_remaining_budget,
                "neighbor_count": min(self.k, len(self._history_by_arm[arm_id])),
                "requested_k": self.k,
                "distance_metric": self.distance_metric,
                "allocation_strategy": self.allocation_strategy,
                "prior_weight": self._prior_weight(arm_id),
            }

        if self.allocation_strategy == "equal_split":
            raw_cap = float(resolved_remaining_budget) / float(max(len(remaining_ranked_arms), 1))
            return {
                "arm_id": arm_id,
                "allocated_completion_token_cap": self._finalize_cap(raw_cap, resolved_remaining_budget),
                "allocation_mode": "equal_split",
                "remaining_budget_before_arm": resolved_remaining_budget,
                "remaining_ranked_arms": list(remaining_ranked_arms),
                "remaining_arm_count": len(remaining_ranked_arms),
                "neighbor_count": min(self.k, len(self._history_by_arm[arm_id])),
                "requested_k": self.k,
                "distance_metric": self.distance_metric,
                "allocation_strategy": self.allocation_strategy,
                "raw_completion_token_cap": raw_cap,
            }

        if self.allocation_strategy == "static_prior":
            prior_denominator_weight = sum(self._prior_weight(candidate) for candidate in remaining_ranked_arms)
            raw_cap = self._prior_cap(
                arm_id=arm_id,
                ranked_arms=remaining_ranked_arms,
                remaining_budget=resolved_remaining_budget,
            )
            return {
                "arm_id": arm_id,
                "allocated_completion_token_cap": self._finalize_cap(raw_cap, resolved_remaining_budget),
                "allocation_mode": "static_prior",
                "remaining_budget_before_arm": resolved_remaining_budget,
                "remaining_ranked_arms": list(remaining_ranked_arms),
                "remaining_arm_count": len(remaining_ranked_arms),
                "neighbor_count": min(self.k, len(self._history_by_arm[arm_id])),
                "requested_k": self.k,
                "distance_metric": self.distance_metric,
                "allocation_strategy": self.allocation_strategy,
                "prior_weight": self._prior_weight(arm_id),
                "prior_denominator_weight": prior_denominator_weight,
                "raw_completion_token_cap": raw_cap,
            }

        ranked_neighbors = self._ranked_neighbors(arm_id=arm_id, context=prepared_context)
        neighbor_count = min(self.k, len(ranked_neighbors))
        selected_neighbors = ranked_neighbors[:neighbor_count]
        prior_cap = self._prior_cap(
            arm_id=arm_id,
            ranked_arms=remaining_ranked_arms,
            remaining_budget=resolved_remaining_budget,
        )
        details: dict[str, Any] = {
            "arm_id": arm_id,
            "remaining_budget_before_arm": resolved_remaining_budget,
            "neighbor_count": neighbor_count,
            "requested_k": self.k,
            "distance_metric": self.distance_metric,
            "allocation_strategy": self.allocation_strategy,
            "prior_weight": self._prior_weight(arm_id),
            "prior_completion_token_cap": prior_cap,
            "nearest_neighbors": [
                {
                    "task_id": observation.task_id,
                    "token_usage": observation.token_usage,
                    "token_cap": observation.token_cap,
                    "uncensored": observation.uncensored,
                    "distance": distance,
                }
                for distance, observation in selected_neighbors
            ],
        }
        if self.allocation_strategy == "empirical_quantile":
            return self._allocate_empirical_quantile(
                details=details,
                arm_id=arm_id,
                selected_neighbors=selected_neighbors,
                prior_cap=prior_cap,
                remaining_budget=resolved_remaining_budget,
            )
        if self.allocation_strategy == "uncensored_lognormal_mle":
            return self._allocate_uncensored_lognormal_mle(
                details=details,
                arm_id=arm_id,
                selected_neighbors=selected_neighbors,
                prior_cap=prior_cap,
                remaining_budget=resolved_remaining_budget,
            )

        return self._allocate_lognormal_mle(
            details=details,
            selected_neighbors=selected_neighbors,
            prior_cap=prior_cap,
            remaining_budget=resolved_remaining_budget,
        )

    def _allocate_lognormal_mle(
        self,
        *,
        details: dict[str, Any],
        selected_neighbors: list[tuple[float, _ArmBudgetObservation]],
        prior_cap: float,
        remaining_budget: int,
    ) -> dict[str, Any]:
        if len(selected_neighbors) < self.k:
            details["allocation_mode"] = "prior_insufficient_neighbors"
            details["allocated_completion_token_cap"] = self._finalize_cap(prior_cap, remaining_budget)
            return details

        evidence: list[tuple[_ArmBudgetObservation, float]] = []
        for _, observation in selected_neighbors:
            lower_bound = self._token_lower_bound(observation)
            if lower_bound > 0.0:
                evidence.append((observation, lower_bound))
        if not evidence:
            details["allocation_mode"] = "prior_invalid_lognormal_mle"
            details["budget_evidence_count"] = len(selected_neighbors)
            details["allocated_completion_token_cap"] = self._finalize_cap(prior_cap, remaining_budget)
            return details

        lower_bounds = [lower_bound for _, lower_bound in evidence]
        log_uncensored = [
            math.log(lower_bound)
            for observation, lower_bound in evidence
            if observation.uncensored
        ]
        log_censored_lower_bounds = [
            math.log(lower_bound)
            for observation, lower_bound in evidence
            if not observation.uncensored
        ]
        fit = _fit_censored_lognormal(
            log_uncensored=log_uncensored,
            log_censored_lower_bounds=log_censored_lower_bounds,
        )
        if fit is None:
            details["allocation_mode"] = "prior_invalid_lognormal_mle"
            details["budget_evidence_count"] = len(selected_neighbors)
            details["uncensored_count"] = len(log_uncensored)
            details["censored_count"] = len(log_censored_lower_bounds)
            details["allocated_completion_token_cap"] = self._finalize_cap(prior_cap, remaining_budget)
            return details
        log_mean, log_sigma, fit_details = fit
        quantile = min(max(1.0 - self.risk_tolerance, 1e-6), 1.0 - 1e-6)
        z_score = NormalDist().inv_cdf(quantile)
        try:
            raw_cap = math.exp(log_mean + z_score * log_sigma)
        except OverflowError:
            raw_cap = float(remaining_budget)
        if not math.isfinite(raw_cap):
            raw_cap = float(remaining_budget)
        lognormal_raw_cap = raw_cap
        raw_cap = max(float(prior_cap), raw_cap)

        censored_count = len(log_censored_lower_bounds)
        details.update(
            {
                "allocation_mode": "lognormal_mle",
                "budget_evidence_count": len(lower_bounds),
                "uncensored_count": len(log_uncensored),
                "censored_count": censored_count,
                "lognormal_mu": log_mean,
                "lognormal_sigma": log_sigma,
                **fit_details,
                "lognormal_quantile": quantile,
                "lognormal_z_score": z_score,
                "budget_lower_bound_min": min(lower_bounds),
                "budget_lower_bound_max": max(lower_bounds),
                "lognormal_raw_completion_token_cap": lognormal_raw_cap,
                "prior_floor_applied": raw_cap > lognormal_raw_cap,
                "raw_completion_token_cap": raw_cap,
                "allocated_completion_token_cap": self._finalize_cap(raw_cap, remaining_budget),
            }
        )
        return details

    def _allocate_empirical_quantile(
        self,
        *,
        details: dict[str, Any],
        arm_id: str,
        selected_neighbors: list[tuple[float, _ArmBudgetObservation]],
        prior_cap: float,
        remaining_budget: int,
    ) -> dict[str, Any]:
        local_evidence = [observation for _, observation in selected_neighbors]
        evidence_source = "local"
        if len(local_evidence) < self.min_budget_neighbors:
            local_evidence = list(self._history_by_arm[arm_id])
            evidence_source = "global"

        if len(local_evidence) < self.min_budget_neighbors:
            details["allocation_mode"] = "prior_insufficient_empirical_quantile_neighbors"
            details["budget_evidence_source"] = evidence_source
            details["budget_evidence_count"] = len(local_evidence)
            details["min_budget_neighbors"] = self.min_budget_neighbors
            details["censored_count"] = sum(1 for observation in local_evidence if not observation.uncensored)
            details["allocated_completion_token_cap"] = self._finalize_cap(prior_cap, remaining_budget)
            return details

        observed_costs = [
            self._token_lower_bound(observation)
            for observation in local_evidence
            if self._token_lower_bound(observation) > 0.0
        ]
        if not observed_costs:
            details["allocation_mode"] = "prior_invalid_empirical_quantile"
            details["budget_evidence_source"] = evidence_source
            details["budget_evidence_count"] = len(local_evidence)
            details["allocated_completion_token_cap"] = self._finalize_cap(prior_cap, remaining_budget)
            return details

        quantile = min(max(1.0 - self.risk_tolerance, 1e-6), 1.0)
        raw_cap = self._nearest_rank_quantile(observed_costs, quantile)
        censored_count = sum(1 for observation in local_evidence if not observation.uncensored)
        details.update(
            {
                "allocation_mode": f"{evidence_source}_empirical_quantile",
                "budget_evidence_source": evidence_source,
                "budget_evidence_count": len(local_evidence),
                "min_budget_neighbors": self.min_budget_neighbors,
                "budget_quantile": quantile,
                "observed_count": len(observed_costs),
                "censored_count": censored_count,
                "capped_observations_treated_as_costs": censored_count,
                "budget_observed_min": min(observed_costs),
                "budget_observed_max": max(observed_costs),
                "raw_completion_token_cap": raw_cap,
                "allocated_completion_token_cap": self._finalize_cap(raw_cap, remaining_budget),
            }
        )
        return details

    def _allocate_uncensored_lognormal_mle(
        self,
        *,
        details: dict[str, Any],
        arm_id: str,
        selected_neighbors: list[tuple[float, _ArmBudgetObservation]],
        prior_cap: float,
        remaining_budget: int,
    ) -> dict[str, Any]:
        local_evidence = [observation for _, observation in selected_neighbors if observation.uncensored]
        evidence_source = "local"
        if len(local_evidence) < self.min_budget_neighbors:
            local_evidence = [observation for observation in self._history_by_arm[arm_id] if observation.uncensored]
            evidence_source = "global"

        if len(local_evidence) < self.min_budget_neighbors:
            details["allocation_mode"] = "prior_insufficient_uncensored_budget_neighbors"
            details["budget_evidence_source"] = evidence_source
            details["budget_evidence_count"] = len(local_evidence)
            details["min_budget_neighbors"] = self.min_budget_neighbors
            details["ignored_censored_count"] = sum(1 for _, observation in selected_neighbors if not observation.uncensored)
            details["allocated_completion_token_cap"] = self._finalize_cap(prior_cap, remaining_budget)
            return details

        uncensored_costs = [
            self._token_lower_bound(observation)
            for observation in local_evidence
            if self._token_lower_bound(observation) > 0.0
        ]
        if not uncensored_costs:
            details["allocation_mode"] = "prior_invalid_uncensored_lognormal_mle"
            details["budget_evidence_source"] = evidence_source
            details["budget_evidence_count"] = len(local_evidence)
            details["allocated_completion_token_cap"] = self._finalize_cap(prior_cap, remaining_budget)
            return details

        log_uncensored = [math.log(value) for value in uncensored_costs]
        fit = _fit_censored_lognormal(
            log_uncensored=log_uncensored,
            log_censored_lower_bounds=[],
        )
        if fit is None:
            details["allocation_mode"] = "prior_invalid_uncensored_lognormal_mle"
            details["budget_evidence_source"] = evidence_source
            details["budget_evidence_count"] = len(local_evidence)
            details["uncensored_count"] = len(log_uncensored)
            details["allocated_completion_token_cap"] = self._finalize_cap(prior_cap, remaining_budget)
            return details

        log_mean, log_sigma, fit_details = fit
        quantile = min(max(1.0 - self.risk_tolerance, 1e-6), 1.0 - 1e-6)
        z_score = NormalDist().inv_cdf(quantile)
        try:
            lognormal_raw_cap = math.exp(log_mean + z_score * log_sigma)
        except OverflowError:
            lognormal_raw_cap = float(remaining_budget)
        if not math.isfinite(lognormal_raw_cap):
            lognormal_raw_cap = float(remaining_budget)
        raw_cap = lognormal_raw_cap
        ignored_censored_count = sum(1 for observation in self._history_by_arm[arm_id] if not observation.uncensored)
        details.update(
            {
                "allocation_mode": f"{evidence_source}_uncensored_lognormal_mle",
                "budget_evidence_source": evidence_source,
                "budget_evidence_count": len(local_evidence),
                "min_budget_neighbors": self.min_budget_neighbors,
                "uncensored_count": len(uncensored_costs),
                "ignored_censored_count": ignored_censored_count,
                "lognormal_mu": log_mean,
                "lognormal_sigma": log_sigma,
                **fit_details,
                "lognormal_quantile": quantile,
                "lognormal_z_score": z_score,
                "lognormal_raw_completion_token_cap": lognormal_raw_cap,
                "prior_floor_applied": False,
                "budget_uncensored_min": min(uncensored_costs),
                "budget_uncensored_max": max(uncensored_costs),
                "raw_completion_token_cap": raw_cap,
                "allocated_completion_token_cap": self._finalize_cap(raw_cap, remaining_budget),
            }
        )
        return details

    def _ranked_neighbors(
        self,
        *,
        arm_id: str,
        context: list[float],
    ) -> list[tuple[float, _ArmBudgetObservation]]:
        return sorted(
            (
                (
                    self._distance(context, observation.context),
                    observation,
                )
                for observation in self._history_by_arm[arm_id]
            ),
            key=lambda item: item[0],
        )

    def _prior_cap(
        self,
        *,
        arm_id: str,
        ranked_arms: list[str],
        remaining_budget: int,
    ) -> float:
        remaining_prior = sum(self._prior_weight(candidate) for candidate in ranked_arms)
        if remaining_prior <= 0.0:
            return float(remaining_budget)
        return (self._prior_weight(arm_id) / remaining_prior) * float(remaining_budget)

    def _prior_weight(self, arm_id: str) -> float:
        return float(self._prior_weights.get(arm_id, 1.0))

    def _token_lower_bound(self, observation: _ArmBudgetObservation) -> float:
        if observation.uncensored:
            return max(float(observation.token_usage), 0.0)
        if observation.token_cap is not None:
            return max(float(observation.token_cap), 0.0)
        return max(float(observation.token_usage), 0.0)

    def _nearest_rank_quantile(self, values: list[float], quantile: float) -> float:
        if not values:
            raise ValueError("values must be non-empty.")
        sorted_values = sorted(float(value) for value in values)
        index = int(math.ceil(float(quantile) * len(sorted_values))) - 1
        index = min(max(index, 0), len(sorted_values) - 1)
        return sorted_values[index]

    def _prepare_context(self, context: list[float]) -> list[float]:
        if len(context) != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {len(context)}."
            )
        normalized = [float(value) for value in context]
        if self.distance_metric == "cosine":
            return _l2_normalize(normalized)
        return normalized

    def _distance(self, left: list[float], right: list[float]) -> float:
        if self.distance_metric == "euclidean":
            return math.sqrt(sum((a - b) * (a - b) for a, b in zip(left, right)))
        return 1.0 - _dot(left, right)

    def _finalize_cap(self, raw_cap: float, remaining_budget: int) -> int:
        bounded_cap = min(max(float(raw_cap), 0.0), float(remaining_budget))
        finalized = int(math.floor(bounded_cap))
        if remaining_budget > 0 and finalized <= 0:
            return 1
        return min(finalized, int(remaining_budget))


def build_partition_selector(
    *,
    arm_ids: list[str] | tuple[str, ...],
    feature_dim: int,
    selector_config: Any,
    random_seed: int | None = None,
) -> KNNRewardSumUCBSelector:
    algorithm = normalize_selector_algorithm(getattr(selector_config, "algorithm", "knn"))
    return KNNRewardSumUCBSelector(
        arm_ids=arm_ids,
        feature_dim=feature_dim,
        k=getattr(selector_config, "knn_k", 20),
        beta=getattr(selector_config, "knn_beta", 0.5),
        reward_max=getattr(selector_config, "knn_reward_max", 1.0),
        distance_metric=getattr(selector_config, "knn_distance_metric", "cosine"),
        ucb_count_scope=getattr(selector_config, "knn_ucb_count_scope", "local"),
        min_pulls_per_arm=getattr(selector_config, "min_pulls_per_arm", 1),
        random_seed=0 if random_seed is None else int(random_seed),
    )
