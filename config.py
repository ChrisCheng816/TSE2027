from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from .arms import DEFAULT_ARM_SET, normalize_arm_ids
from .types import GenerationConfig

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_EXPERIMENT_RESULTS_DIR = DEFAULT_RESULTS_ROOT / "main"
DEFAULT_ABLATION_RESULTS_DIR = DEFAULT_RESULTS_ROOT / "ablation"
DEFAULT_SENSITIVITY_RESULTS_DIR = DEFAULT_RESULTS_ROOT / "sensitivity"
DEFAULT_SEED_RESULTS_DIR = DEFAULT_RESULTS_ROOT / "seeds"
DEFAULT_REPEATED_SAMPLING_RESULTS_DIR = PROJECT_ROOT / "repeated_sampling"
DATASET_NAME = "bigcodebench-v0.1.4"
MODEL_SPECS = [
    {"name": "qwen2.5-32b", "path": "Qwen/Qwen2.5-Coder-32B-Instruct"},
    {"name": "qwen3-30b", "path": "Qwen/Qwen3-Coder-30B-A3B-Instruct"},
    {"name": "claude-opus-4.8", "path": "claude-opus-4-8"}
]
MODEL_SPEC_MAP = {spec["name"]: spec["path"] for spec in MODEL_SPECS}
DEFAULT_MODEL_NAME = MODEL_SPECS[0]["name"]


def format_adaptive_arm_ids(arm_ids: tuple[str, ...] | list[str]) -> str:
    normalized_arm_ids = normalize_arm_ids(tuple(arm_ids))
    if not normalized_arm_ids:
        return "no-arms"
    arm_order = {arm_id: index for index, arm_id in enumerate(DEFAULT_ARM_SET)}
    stable_arm_ids = tuple(sorted(normalized_arm_ids, key=lambda arm_id: arm_order.get(arm_id, len(arm_order))))
    return "-".join(stable_arm_ids)


def format_budget_group_label(per_task_completion_token_budget: int | None) -> str:
    if per_task_completion_token_budget is None:
        return "budget-auto"
    return f"budget-{int(per_task_completion_token_budget)}"


def results_dataset_bucket(dataset_name: str, *, token_optimized: bool = False) -> str | None:
    normalized_dataset_name = dataset_name.lower().replace("_", "-")
    if "bigcodebench" in normalized_dataset_name:
        return "token_aware" if token_optimized else "pass_only"
    return None


@dataclass(frozen=True)
class PathConfig:
    experiment_results_dir: Path = DEFAULT_EXPERIMENT_RESULTS_DIR
    repeated_sampling_results_dir: Path = DEFAULT_REPEATED_SAMPLING_RESULTS_DIR

    def dataset_results_dir(self, dataset_name: str, *, token_optimized: bool = False) -> Path:
        dataset_bucket = results_dataset_bucket(dataset_name, token_optimized=token_optimized)
        if dataset_bucket is None:
            return self.experiment_results_dir
        return self.experiment_results_dir / dataset_bucket

    def model_results_dir(
        self,
        model_name: str,
        dataset_name: str | None = None,
        *,
        token_optimized: bool = False,
    ) -> Path:
        if dataset_name is None:
            return self.experiment_results_dir / model_name
        return self.dataset_results_dir(dataset_name, token_optimized=token_optimized) / model_name


    def legacy_model_results_dir(self, model_name: str) -> Path:
        return self.experiment_results_dir / model_name

    def arm_output_path(self, model_name: str, arm_id: str, dataset_name: str) -> Path:
        return self.model_results_dir(model_name, dataset_name) / f"{model_name}_{arm_id}.jsonl"

    def arm_summary_path(self, model_name: str, arm_id: str, dataset_name: str) -> Path:
        return self.model_results_dir(model_name, dataset_name) / f"{model_name}_{arm_id}.summary.json"

    def repeated_sampling_model_results_dir(self, model_name: str, dataset_name: str) -> Path:
        dataset_bucket = results_dataset_bucket(dataset_name)
        if dataset_bucket is None:
            return self.repeated_sampling_results_dir / model_name
        return self.repeated_sampling_results_dir / dataset_bucket / model_name

    def repeated_sampling_output_path(
        self,
        model_name: str,
        arm_id: str,
        dataset_name: str,
        *,
        budget_label: str | None = None,
    ) -> Path:
        budget_suffix = f"_{budget_label}" if budget_label else ""
        return (
            self.repeated_sampling_model_results_dir(model_name, dataset_name)
            / f"{model_name}_{arm_id}{budget_suffix}.jsonl"
        )

    def repeated_sampling_summary_path(
        self,
        model_name: str,
        arm_id: str,
        dataset_name: str,
        *,
        budget_label: str | None = None,
    ) -> Path:
        return self.repeated_sampling_output_path(
            model_name,
            arm_id,
            dataset_name,
            budget_label=budget_label,
        ).with_suffix(".summary.json")

    def reward_output_path(self, model_name: str, dataset_name: str) -> Path:
        return self.model_results_dir(model_name, dataset_name) / f"{model_name}.reward_table.jsonl"

    def adaptive_output_path(
        self,
        model_name: str,
        dataset_name: str,
        arm_ids: tuple[str, ...] | list[str],
        *,
        budget_label: str | None = None,
    ) -> Path:
        adaptive_arm_ids = format_adaptive_arm_ids(arm_ids)
        budget_suffix = f"_{budget_label}" if budget_label else ""
        return (
            self.model_results_dir(model_name, dataset_name)
            / f"{model_name}_adaptive_{adaptive_arm_ids}{budget_suffix}.jsonl"
        )

    def adaptive_summary_path(
        self,
        model_name: str,
        dataset_name: str,
        arm_ids: tuple[str, ...] | list[str],
        *,
        budget_label: str | None = None,
    ) -> Path:
        adaptive_arm_ids = format_adaptive_arm_ids(arm_ids)
        budget_suffix = f"_{budget_label}" if budget_label else ""
        return (
            self.model_results_dir(model_name, dataset_name)
            / f"{model_name}_adaptive_{adaptive_arm_ids}{budget_suffix}.summary.json"
        )


@dataclass(frozen=True)
class BanditStageConfig:
    # ===== Mainline selector: local kNN-UCB =====
    # Used by --order_run and the order-suite variants that depend on adaptive
    # ranking. --random ignores the ranking after selection; --fix replaces it
    # with a deterministic order.
    algorithm: str = "knn"
    # min_pulls_per_arm is still global cold-start exploration; after cold start,
    # knn_ucb_count_scope="local" makes the UCB denominator use local neighbor counts.
    min_pulls_per_arm: int = 1
    knn_k: int = 30
    knn_beta: float = 0
    knn_reward_max: float = 1.0
    knn_distance_metric: str = "cosine"
    knn_ucb_count_scope: str = "local"

    # ===== Alternative / legacy selector knobs =====
    # Use knn_ucb_count_scope="global_first" to reproduce the old global-count
    # kNN-UCB variant.


@dataclass(frozen=True)
class BanditConfig:
    selector: BanditStageConfig = field(default_factory=BanditStageConfig)


@dataclass(frozen=True)

class AdaptiveConfig:
    # ===== Mainline order-suite controls =====
    # Applies to --order_run, --fix, and --random suite variants.
    selector_feedback_metric: str = "reward"
    # Dataset-specific budgets are used first for BigCodeBench.
    total_completion_token_budget: int | None = None
    bigcodebench_per_task_completion_token_budget: int | None = 15000
    per_task_completion_token_budget: int | None = None
    order_run: bool = False
    random_run: bool = False
    fix_run: bool = False
    fix_budget: bool = False
    free_run: bool = False
    order_run_random_seed: int = 0
    order_run_fix_start: str = "A0"

    # ===== Mainline adaptive budget allocator =====
    # Used by --order_run, --fix, and --random.
    # mle fits a right-censored log-normal cost model for adaptive caps.
    # Additional ablations are available via --budget-allocator:
    # empirical_quantile, uncensored_lognormal_mle, equal_split, and static_prior.
    order_run_budget_allocator: str = "mle"
    order_run_budget_quantile: float = 0.90
    order_run_budget_min_neighbors: int = 20

    # Fallback prior split used when empirical budget evidence is too sparse.
    # The MLE allocator also uses this split when log-normal MLE cannot be fit.
    # Mainline default gives the higher-cost A2 repair arm a larger fallback share.
    order_run_budget_prior_weights: dict[str, float] = field(
        default_factory=lambda: {"A0": 1.0, "A1": 1.0, "A2": 2.0}
    )

    # ===== Alternative / legacy adaptive knobs =====
    # selector_feedback_metric="pass_at_k" is available for ablations.
    # order_run_budget_epsilon is the MLE tail risk.
    order_run_budget_epsilon: float = 0.01


def adaptive_per_task_completion_token_budget(
    adaptive_config: AdaptiveConfig,
    dataset_name: str,
) -> int | None:
    dataset_bucket = results_dataset_bucket(dataset_name)
    if dataset_bucket in {"pass_only", "token_aware"}:
        return adaptive_config.bigcodebench_per_task_completion_token_budget
    return adaptive_config.per_task_completion_token_budget


@dataclass(frozen=True)
class RewardConfig:
    # ===== Mainline pass-only reward (PO) =====
    reward_version: str = "v2"
    success_reward: float = 1

    # ===== Shared mainline token accounting =====
    completion_budget_multiplier: float = 1.25
    max_normalized_completion_cost: float = 4
    include_prompt_tokens_in_cost: bool = True
    prompt_token_cost_weight: float = 1.0
    completion_token_cost_weight: float = 4

    # ===== Alternative / legacy reward knobs =====
    pass_cost_weight: float = 0.08
    fail_cost_weight: float = 0.02
    v2_pass_log_cost_weight: float = 0
    v2_fail_log_cost_weight: float = 0
    v2_fail_base_reward: float = 0
    v2_log_cost_cap: float = 6.8146


@dataclass(frozen=True)
class TokenOptimizedRewardConfig(RewardConfig):
    # ===== Mainline token-aware reward (TA) =====
    reward_version: str = "token_sqrt"
    success_reward: float = 1
    completion_budget_multiplier: float = 1.25
    max_normalized_completion_cost: float = 4
    include_prompt_tokens_in_cost: bool = True
    prompt_token_cost_weight: float = 1.0
    completion_token_cost_weight: float = 4

    # ===== Alternative / legacy reward knobs =====
    # Repeated here so this profile remains self-contained if copied or edited.
    pass_cost_weight: float = 0.08
    fail_cost_weight: float = 0.02
    v2_pass_log_cost_weight: float = 0
    v2_fail_log_cost_weight: float = 0
    v2_fail_base_reward: float = 0
    v2_log_cost_cap: float = 9.21044


@dataclass(frozen=True)
class WarmStartConfig:
    enabled: bool = False
    dataset_name: str = "bigcodebench-v0.1.4"
    max_rows: int | None = None
    strict: bool = False


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str = "qwen"
    include_structured_features: bool = False
    hash_dim: int = 32
    qwen_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    qwen_output_dim: int = 1024
    qwen_task_instruction: str = "Embed this programming task for adaptive code-generation arm routing."
    qwen_cuda_visible_devices: str = "2"
    qwen_gpu_memory_utilization: float = 0.1
    cache_dir: Path = PROJECT_ROOT / ".cache" / "qwen_embeddings"


@dataclass(frozen=True)
class ModelRuntimeConfig:
    model_name: str = DEFAULT_MODEL_NAME


@dataclass(frozen=True)
class VLLMConfig:
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 8192
    trust_remote_code: bool = True
    worker_multiproc_method: str = "spawn"
    cuda_visible_devices: str = "0,1,2,3"
    nccl_p2p_disable: str = "1"


@dataclass(frozen=True)
class ExperimentConfig:
    run_mode: str = "all_arms"
    dataset_name: str = DATASET_NAME
    arm_ids: tuple[str, ...] = DEFAULT_ARM_SET
    all_arms_compute_reward_and_features: bool = False
    repeated_sampling_arm_id: str | None = None
    limit: int | None = None
    shuffle_seed: int | None = None
    batch_size: int = 16
    generation_seed: int | None = 42
    num_samples_per_task: int = 1
    pass_k: int = 1
    evaluation_timeout_sec: int = 30
    progress_interval: int = 50
    paths: PathConfig = field(default_factory=PathConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    token_reward: RewardConfig = field(default_factory=TokenOptimizedRewardConfig)
    warm_start: WarmStartConfig = field(default_factory=WarmStartConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    model: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    generation: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(
            max_new_tokens=2048,
            temperature=0,
            stop=(),
        )
    )


DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()


def get_model_choices() -> tuple[str, ...]:
    return tuple(spec["name"] for spec in MODEL_SPECS)


def resolve_model_path(model_name: str) -> str:
    if model_name not in MODEL_SPEC_MAP:
        available = ", ".join(get_model_choices())
        raise KeyError(f"Unknown model name '{model_name}'. Available models: {available}")
    return MODEL_SPEC_MAP[model_name]
