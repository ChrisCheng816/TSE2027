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
DEFAULT_REPEATED_SAMPLING_RESULTS_DIR = DEFAULT_RESULTS_ROOT / "repeated_sampling"
DATASET_NAME = "bigcodebench-v0.1.4"
MODEL_SPECS = [
    {"name": "qwen2.5-32b", "path": "Qwen/Qwen2.5-Coder-32B-Instruct"},
    {"name": "qwen3-30b", "path": "Qwen/Qwen3-Coder-30B-A3B-Instruct"},
    {"name": "claude-opus-4.8", "path": "claude-opus-4-8"}
]
MODEL_SPEC_MAP = {spec["name"]: spec["path"] for spec in MODEL_SPECS}
DEFAULT_MODEL_NAME = MODEL_SPECS[0]["name"]
REPEATED_SAMPLING_WORKFLOW_SLUGS = {
    "A0": "direct_coding",
    "A1": "plan_structured_coding",
    "A2": "test_guided_repair",
}


def repeated_sampling_workflow_slug(arm_id: str) -> str:
    normalized_arm_id = normalize_arm_ids([arm_id])[0]
    try:
        return REPEATED_SAMPLING_WORKFLOW_SLUGS[normalized_arm_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported repeated-sampling arm: {arm_id}") from exc


def format_bara_arm_ids(arm_ids: tuple[str, ...] | list[str]) -> str:
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
    if "livecodebench" in normalized_dataset_name:
        return "livecodebench/token_aware" if token_optimized else "livecodebench/pass_only"
    if "bigcodebench" in normalized_dataset_name:
        return "bigcodebench/token_aware" if token_optimized else "bigcodebench/pass_only"
    return None


def single_dataset_results_bucket(dataset_name: str, *, token_optimized: bool = False) -> str | None:
    normalized_dataset_name = dataset_name.lower().replace("_", "-")
    if "livecodebench" in normalized_dataset_name:
        return "livecodebench/token_aware" if token_optimized else "livecodebench/pass_only"
    if "bigcodebench" in normalized_dataset_name:
        return "token_aware" if token_optimized else "pass_only"
    return None


def repeated_sampling_dataset_bucket(dataset_name: str) -> str:
    normalized_dataset_name = dataset_name.lower().replace("_", "-")
    if "livecodebench" in normalized_dataset_name:
        return "livecodebench"
    if "bigcodebench" in normalized_dataset_name:
        return "bigcodebench"
    return normalized_dataset_name


@dataclass(frozen=True)
class PathConfig:
    experiment_results_dir: Path = DEFAULT_EXPERIMENT_RESULTS_DIR
    repeated_sampling_results_dir: Path = DEFAULT_REPEATED_SAMPLING_RESULTS_DIR
    ablation_results_dir: Path = DEFAULT_ABLATION_RESULTS_DIR
    seed_results_dir: Path | None = None

    def dataset_results_dir(self, dataset_name: str, *, token_optimized: bool = False) -> Path:
        if self.experiment_results_dir.name in {"ablation", "sensitivity"}:
            dataset_bucket = single_dataset_results_bucket(dataset_name, token_optimized=token_optimized)
        else:
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


    def arm_output_path(self, model_name: str, arm_id: str, dataset_name: str) -> Path:
        return self.model_results_dir(model_name, dataset_name) / f"{model_name}_{arm_id}.jsonl"

    def arm_summary_path(self, model_name: str, arm_id: str, dataset_name: str) -> Path:
        return self.model_results_dir(model_name, dataset_name) / f"{model_name}_{arm_id}.summary.json"

    def repeated_sampling_model_results_dir(self, model_name: str, arm_id: str, dataset_name: str) -> Path:
        workflow_slug = repeated_sampling_workflow_slug(arm_id)
        dataset_bucket = repeated_sampling_dataset_bucket(dataset_name)
        return self.repeated_sampling_results_dir / dataset_bucket / model_name / workflow_slug

    def repeated_sampling_output_path(
        self,
        model_name: str,
        arm_id: str,
        dataset_name: str,
        *,
        budget_label: str | None = None,
    ) -> Path:
        budget_suffix = f"_{budget_label}" if budget_label else ""
        workflow_slug = repeated_sampling_workflow_slug(arm_id)
        return (
            self.repeated_sampling_model_results_dir(model_name, arm_id, dataset_name)
            / f"{model_name}_{workflow_slug}_{dataset_name}{budget_suffix}.jsonl"
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


@dataclass(frozen=True)
class BanditStageConfig:
    # ===== Mainline selector: local kNN-UCB =====
    # Used by BARA and the Uniform/Static ranking baselines.
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

class BARAConfig:
    # ===== Mainline BARA controls =====
    bigcodebench_per_task_completion_token_budget: int | None = 15000
    livecodebench_per_task_completion_token_budget: int | None = 15000
    rank_and_allocate: bool = False
    uniform_run: bool = False
    static_run: bool = False
    task_order_seed: int | None = None
    static_order: str = "A0-A1-A2"

    # ===== Mainline adaptive budget allocator =====
    # Used by BARA, Uniform, and Static.
    # mle fits a right-censored log-normal cost model for adaptive caps.
    # Additional ablations are available via --budget-allocator:
    # empirical_quantile, equal_split, and static_prior.
    budget_allocator: str = "mle"
    budget_quantile: float = 0.90
    budget_min_neighbors: int = 20

    # Fallback prior split used when empirical budget evidence is too sparse.
    # The MLE allocator also uses this split when log-normal MLE cannot be fit.
    # Mainline default gives the higher-cost A2 repair arm a larger fallback share.
    budget_prior_weights: dict[str, float] = field(
        default_factory=lambda: {"A0": 1.0, "A1": 1.0, "A2": 2.0}
    )

    # ===== Alternative / legacy adaptive knobs =====
    budget_epsilon: float = 0.01


def bara_per_task_completion_token_budget(
    bara_config: BARAConfig,
    dataset_name: str,
) -> int | None:
    normalized_dataset_name = str(dataset_name or "").lower().replace("_", "-")
    if "livecodebench" in normalized_dataset_name:
        return bara_config.livecodebench_per_task_completion_token_budget
    return bara_config.bigcodebench_per_task_completion_token_budget


@dataclass(frozen=True)
class RewardConfig:
    # ===== Mainline pass-only reward (PO) =====
    reward_version: str = "binary"
    success_reward: float = 1

    # ===== Shared mainline token accounting =====
    completion_budget_multiplier: float = 1.25
    max_normalized_completion_cost: float = 4
    include_prompt_tokens_in_cost: bool = True
    prompt_token_cost_weight: float = 1.0
    completion_token_cost_weight: float = 4



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
    token_aware: bool = False
    dataset_name: str = DATASET_NAME
    arm_ids: tuple[str, ...] = DEFAULT_ARM_SET
    repeated_sampling_arm_id: str | None = None
    limit: int | None = None
    task_order_seed: int | None = None
    batch_size: int = 16
    generation_seed: int | None = 42
    num_samples_per_task: int = 1
    pass_k: int = 1
    evaluation_timeout_sec: int = 30
    progress_interval: int = 50
    paths: PathConfig = field(default_factory=PathConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    bara: BARAConfig = field(default_factory=BARAConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    token_reward: RewardConfig = field(default_factory=TokenOptimizedRewardConfig)
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
