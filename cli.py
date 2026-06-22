from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from .arms import get_arm_choices, get_partition_arms, normalize_arm_ids
from .bara import (
    KNNArmBudgetAllocator,
    build_partition_selector,
    normalize_selector_algorithm,
)
from .config import (
    DEFAULT_ABLATION_RESULTS_DIR,
    DEFAULT_EXPERIMENT_CONFIG,
    ExperimentConfig,
    bara_per_task_completion_token_budget,
    format_bara_arm_ids,
    format_budget_group_label,
    get_model_choices,
)
from .dataset import get_dataset_choices, load_codegen_tasks, write_jsonl
from .features import build_feature_extractor
from .models import (
    AnthropicBatchModelClient,
    PlaceholderModelClient,
    StaticModelClient,
    VLLMModelClient,
    is_anthropic_model_name,
)
from .runner import (
    ExperimentRunner,
    build_experiment_summary,
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args()
    model_names = _requested_model_names(args)
    task_order_seeds = _requested_task_order_seeds(args)
    total_runs = len(model_names) * len(task_order_seeds)
    run_index = 0
    for model_name in model_names:
        for task_order_seed in task_order_seeds:
            run_index += 1
            config = _resolve_config(
                args,
                model_name=model_name,
                task_order_seed=task_order_seed,
            )
            if total_runs > 1:
                logging.info(
                    "[runs] start %s/%s model=%s task_order_seed=%s",
                    run_index,
                    total_runs,
                    config.model.model_name,
                    "canonical" if task_order_seed is None else task_order_seed,
                )
            _run_config(config, args=args)
    return 0


def _run_config(config: ExperimentConfig, *, args: argparse.Namespace) -> None:
    logging.info(
        "[start] run_mode=%s dataset=%s model=%s arms=%s limit=%s batch_size=%s samples=%s pass_k=%s progress_interval=%s task_order_seed=%s",
        config.run_mode,
        config.dataset_name,
        config.model.model_name,
        ",".join(config.arm_ids),
        config.limit,
        config.batch_size,
        config.num_samples_per_task,
        config.pass_k,
        config.progress_interval,
        "canonical" if config.task_order_seed is None else config.task_order_seed,
    )

    if config.run_mode == "bara":
        _run_bara(config, static_completion=args.static_completion)
    elif config.run_mode == "repeated_sampling":
        _run_repeated_sampling(config, static_completion=args.static_completion)
    else:
        _run_all_arms(
            config,
            static_completion=args.static_completion,
            run_bara_after_all_arms=args.then_bara,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BARA: Budgeted Adaptive Ranking and Allocation Framework."
    )
    parser.add_argument(
        "--run-mode",
        choices=("all_arms", "bara", "repeated_sampling"),
        default=None,
        help="Optional run-mode override. Defaults to the mode configured inside config.py.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        choices=get_dataset_choices(),
        help="Optional dataset override from the internal dataset registry.",
    )
    parser.add_argument(
        "--model-name",
        nargs="+",
        default=None,
        choices=get_model_choices(),
        help="Optional model alias override(s) from the internal model registry.",
    )
    parser.add_argument(
        "--arm-ids",
        nargs="+",
        default=None,
        help=(
            "Optional arm selection override for --run-mode all_arms. Supports multiple values like "
            "'--arm-ids A0 A1' or '--arm-ids a0 repair'. "
            "Comma-separated tokens are also accepted."
        ),
    )
    parser.add_argument(
        "--arm-id",
        default=None,
        help="Selected workflow for --run-mode repeated_sampling, e.g. A2.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional task limit override for smoke tests.")
    parser.add_argument(
        "--static-completion",
        default=None,
        help="Optional fixed completion string for dry runs without a real model.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override for this run, e.g. '2,3'.",
    )
    parser.add_argument(
        "--then-bara",
        action="store_true",
        help="When set with --run-mode all_arms, run BARA after per-workflow generation.",
    )
    parser.add_argument(
        "--t",
        type=float,
        default=None,
        help="Temperature override for --run-mode repeated_sampling only. Defaults to 1.",
    )
    parser.add_argument(
        "--uniform",
        dest="uniform_run",
        action="store_true",
        help="Use a uniformly shuffled workflow ranking instead of BARA ranking.",
    )
    parser.add_argument(
        "--static",
        dest="static_run",
        action="store_true",
        help="Use a static workflow schedule instead of BARA ranking.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Task-order seeds. Each seed is executed as an independent online run.",
    )
    parser.add_argument(
        "--static-order",
        default=None,
        help="Static workflow order, for example A0-A1-A2.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Per-task BARA token budget for the selected dataset."
    )
    parser.add_argument(
        "--budget-allocator",
        choices=(
            "quantile",
            "mle",
            "equal_split",
            "static_prior",
            "budget",
            "empirical_quantile",
        ),
        default=None,
        help="BARA budget allocator. quantile maps to empirical_quantile; budget maps to static_prior."
    )
    parser.add_argument("--budget-quantile", type=float, default=None)
    parser.add_argument("--budget-min-neighbors", type=int, default=None)
    parser.add_argument(
        "--token-aware",
        dest="token_aware",
        action="store_true",
        help="Use the token-aware reward (token_sqrt) and write results to token_aware/ instead of pass_only/.",
    )
    selector_group = parser.add_mutually_exclusive_group()
    selector_group.add_argument(
        "--knn",
        action="store_true",
        help="Use the k-NN reward-sum UCB selector (default).",
    )
    return parser



def _normalize_budget_allocator_arg(value: str) -> str:
    allocator = str(value or "").strip().lower()
    if allocator == "quantile":
        return "empirical_quantile"
    if allocator == "budget":
        return "static_prior"
    return allocator


def _requested_model_names(args: argparse.Namespace) -> list[str]:
    raw_model_names = getattr(args, "model_name", None)
    if raw_model_names is None:
        return [DEFAULT_EXPERIMENT_CONFIG.model.model_name]
    if isinstance(raw_model_names, str):
        return [raw_model_names]
    return [str(model_name) for model_name in raw_model_names]


def _requested_task_order_seeds(args: argparse.Namespace) -> list[int | None]:
    run_mode = args.run_mode if args.run_mode is not None else DEFAULT_EXPERIMENT_CONFIG.run_mode
    requested = getattr(args, "seeds", None)
    if requested is not None:
        if run_mode != "bara":
            raise ValueError("--seeds is only supported with --run-mode bara.")
        seeds = [int(seed) for seed in requested]
        if any(seed < 0 for seed in seeds):
            raise ValueError("--seeds values must be non-negative.")
        if len(set(seeds)) != len(seeds):
            raise ValueError("--seeds values must be unique.")
        return seeds
    if run_mode == "bara" and (
        getattr(args, "uniform_run", False) or getattr(args, "static_run", False)
    ):
        return [0]
    return [None]


def _resolve_config(
    args: argparse.Namespace,
    *,
    model_name: str | None = None,
    task_order_seed: int | None = None,
) -> ExperimentConfig:
    config = DEFAULT_EXPERIMENT_CONFIG
    resolved_run_mode = args.run_mode if args.run_mode is not None else config.run_mode
    model_config = config.model
    bandit_config = config.bandit
    selector_config = bandit_config.selector
    bara_config = config.bara
    vllm_config = config.vllm

    resolved_model_name = model_name
    if resolved_model_name is None:
        requested_model_names = _requested_model_names(args)
        resolved_model_name = requested_model_names[0] if requested_model_names else None
    if resolved_model_name is not None:
        model_config = replace(model_config, model_name=resolved_model_name)
    if getattr(args, "cuda_visible_devices", None) is not None:
        vllm_config = replace(vllm_config, cuda_visible_devices=args.cuda_visible_devices)
    if getattr(args, "knn", False):
        selector_config = replace(selector_config, algorithm="knn")
    bandit_config = replace(bandit_config, selector=selector_config)

    if (getattr(args, "uniform_run", False) or getattr(args, "static_run", False)) and resolved_run_mode != "bara":
        raise ValueError("--uniform and --static require --run-mode bara.")
    if getattr(args, "uniform_run", False) and getattr(args, "static_run", False):
        raise ValueError("--uniform cannot be combined with --static.")
    if getattr(args, "static_order", None) is not None and not getattr(args, "static_run", False):
        raise ValueError("--static-order only applies to --static.")

    bara_updates: dict[str, object] = {
        "rank_and_allocate": resolved_run_mode in {"bara", "repeated_sampling"},
        "task_order_seed": task_order_seed,
    }
    if getattr(args, "uniform_run", False):
        bara_updates["uniform_run"] = True
    if getattr(args, "static_run", False):
        bara_updates["static_run"] = True
    if getattr(args, "static_order", None) is not None:
        bara_updates["static_order"] = str(args.static_order)
    if getattr(args, "budget", None) is not None:
        if args.budget < 0:
            raise ValueError("--budget must be non-negative.")
        bara_updates["bigcodebench_per_task_completion_token_budget"] = int(args.budget)
    if getattr(args, "budget_allocator", None) is not None:
        bara_updates["budget_allocator"] = _normalize_budget_allocator_arg(args.budget_allocator)
    if getattr(args, "budget_quantile", None) is not None:
        bara_updates["budget_quantile"] = float(args.budget_quantile)
    if getattr(args, "budget_min_neighbors", None) is not None:
        bara_updates["budget_min_neighbors"] = int(args.budget_min_neighbors)
    bara_config = replace(bara_config, **bara_updates)

    repeated_sampling_arm_id = None
    if getattr(args, "arm_ids", None) is not None and resolved_run_mode != "all_arms":
        raise ValueError("--arm-ids is only supported with --run-mode all_arms. Use --arm-id for repeated_sampling.")
    if getattr(args, "arm_id", None) is not None and resolved_run_mode != "repeated_sampling":
        raise ValueError("--arm-id is only supported with --run-mode repeated_sampling.")

    arm_ids = config.arm_ids
    if getattr(args, "arm_ids", None) is not None:
        arm_ids = normalize_arm_ids(args.arm_ids)
    if resolved_run_mode == "repeated_sampling":
        if getattr(args, "arm_id", None) is None:
            raise ValueError("--run-mode repeated_sampling requires --arm-id.")
        if getattr(args, "budget", None) is None:
            raise ValueError("--run-mode repeated_sampling requires --budget.")
        repeated_sampling_arm_id = normalize_arm_ids([args.arm_id])[0]
        arm_ids = (repeated_sampling_arm_id,)
        bara_config = replace(
            bara_config,
            uniform_run=False,
            static_run=False,
        )

    generation_config = config.generation
    if resolved_run_mode == "repeated_sampling":
        temperature = 1 if getattr(args, "t", None) is None else args.t
        generation_config = replace(generation_config, temperature=temperature, top_p=0.95)
    elif getattr(args, "t", None) is not None:
        raise ValueError("--t is only supported with --run-mode repeated_sampling.")

    return replace(
        config,
        run_mode=resolved_run_mode,
        token_aware=bool(getattr(args, "token_aware", False)),
        dataset_name=args.dataset_name if args.dataset_name is not None else config.dataset_name,
        arm_ids=arm_ids,
        repeated_sampling_arm_id=repeated_sampling_arm_id,
        limit=args.limit if args.limit is not None else config.limit,
        task_order_seed=task_order_seed,
        model=model_config,
        bandit=bandit_config,
        bara=bara_config,
        vllm=vllm_config,
        generation=generation_config,
    )


def _run_repeated_sampling(config: ExperimentConfig, *, static_completion: str | None) -> None:
    if not config.repeated_sampling_arm_id:
        raise ValueError("--run-mode repeated_sampling requires --arm-id.")
    tasks = _load_tasks(config)
    arms = get_partition_arms((config.repeated_sampling_arm_id,))
    arm = arms[0]
    budget_label = format_budget_group_label(
        bara_per_task_completion_token_budget(config.bara, config.dataset_name)
    )
    logging.info(
        "[repeated_sampling] dataset=%s loaded %s tasks task_kinds=%s arm=%s budget=%s temperature=%s",
        config.dataset_name,
        len(tasks),
        _summarize_task_kinds(tasks),
        config.repeated_sampling_arm_id,
        budget_label,
        config.generation.temperature,
    )
    runner = _build_runner(config, arms=arms, static_completion=static_completion, enable_feature_extractor=False)
    records = runner.run_repeated_sampling(
        tasks,
        arm=arm,
        num_samples_per_task=config.num_samples_per_task,
        pass_k=config.pass_k,
        evaluation_timeout_sec=config.evaluation_timeout_sec,
    )
    output_path = config.paths.repeated_sampling_output_path(
        config.model.model_name,
        config.repeated_sampling_arm_id,
        config.dataset_name,
        budget_label=budget_label,
    )
    summary_path = config.paths.repeated_sampling_summary_path(
        config.model.model_name,
        config.repeated_sampling_arm_id,
        config.dataset_name,
        budget_label=budget_label,
    )
    _write_records(output_path=output_path, records=records)
    _write_summary(summary_path=summary_path, summary=build_experiment_summary(records, arms=tuple(arms)))
    logging.info("[repeated_sampling] wrote records=%s -> %s", len(records), output_path)


def _run_all_arms(
    config: ExperimentConfig,
    *,
    static_completion: str | None,
    run_bara_after_all_arms: bool = False,  # kept for --then-bara compatibility
) -> None:
    tasks = _load_tasks(config)
    arms = get_partition_arms(list(config.arm_ids))
    logging.info(
        "[all_arms] dataset=%s loaded %s tasks task_kinds=%s",
        config.dataset_name,
        len(tasks),
        _summarize_task_kinds(tasks),
    )
    runner = _build_runner(
        config,
        arms=arms,
        static_completion=static_completion,
        enable_feature_extractor=False,
    )

    def _persist_arm_records(arm, records) -> None:
        output_path = config.paths.arm_output_path(
            config.model.model_name,
            arm.arm_id,
            config.dataset_name,
        )
        summary_path = config.paths.arm_summary_path(
            config.model.model_name,
            arm.arm_id,
            config.dataset_name,
        )
        _write_records(output_path=output_path, records=records)
        _write_summary(summary_path=summary_path, summary=build_experiment_summary(records, arms=(arm,)))
        logging.info("[all_arms] wrote arm=%s records=%s -> %s", arm.arm_id, len(records), output_path)

    runner.run_all_arms(
        tasks,
        num_samples_per_task=config.num_samples_per_task,
        pass_k=config.pass_k,
        evaluation_timeout_sec=config.evaluation_timeout_sec,
        on_arm_complete=_persist_arm_records,
    )

    if run_bara_after_all_arms:
        logging.info("[all_arms] starting BARA after per-workflow generation")
        resolved_bara_config = replace(
            config,
            run_mode="bara",
            bara=replace(config.bara, rank_and_allocate=True),
        )
        bara_records = _run_bara_mainline(
            config=resolved_bara_config,
            tasks=tasks,
            arms=arms,
            static_completion=static_completion,
        )
        _write_bara_outputs(config=resolved_bara_config, arms=arms, records=bara_records)


def _run_bara(config: ExperimentConfig, *, static_completion: str | None) -> None:
    tasks = _load_tasks(config)
    arms = get_partition_arms(list(config.arm_ids))
    logging.info(
        "[BARA] dataset=%s loaded %s tasks task_kinds=%s",
        config.dataset_name,
        len(tasks),
        _summarize_task_kinds(tasks),
    )
    records = _run_bara_mainline(
        config=config,
        tasks=tasks,
        arms=arms,
        static_completion=static_completion,
    )
    _write_bara_outputs(config=config, arms=arms, records=records)



def _run_bara_mainline(*, config: ExperimentConfig, tasks, arms, static_completion: str | None):
    runner = _build_runner(config, arms=arms, static_completion=static_completion)
    return runner.run_bara(
        tasks,
        num_samples_per_task=config.num_samples_per_task,
        pass_k=config.pass_k,
        evaluation_timeout_sec=config.evaluation_timeout_sec,
    )


def _build_runner(
    config: ExperimentConfig,
    *,
    arms,
    static_completion: str | None,
    enable_feature_extractor: bool = True,
) -> ExperimentRunner:
    feature_extractor = build_feature_extractor(config) if enable_feature_extractor else None
    selector = None
    budget_allocator = None
    if feature_extractor is not None:
        selector = build_partition_selector(
            arm_ids=[arm.arm_id for arm in arms],
            feature_dim=len(feature_extractor.FEATURE_NAMES),
            selector_config=config.bandit.selector,
        )
        budget_allocator = KNNArmBudgetAllocator(
            arm_ids=[arm.arm_id for arm in arms],
            feature_dim=len(feature_extractor.FEATURE_NAMES),
            k=getattr(config.bandit.selector, "knn_k", 20),
            distance_metric=getattr(config.bandit.selector, "knn_distance_metric", "cosine"),
            risk_tolerance=getattr(config.bara, "budget_epsilon", 0.01),
            arm_prior_weights=getattr(config.bara, "budget_prior_weights", None),
            allocation_strategy=getattr(config.bara, "budget_allocator", "mle"),
            budget_quantile=getattr(config.bara, "budget_quantile", 0.80),
            min_budget_neighbors=getattr(config.bara, "budget_min_neighbors", 5),
        )
        logging.info(
            "[selector] algorithm=%s arms=%s feature_dim=%s",
            normalize_selector_algorithm(config.bandit.selector.algorithm),
            ",".join(arm.arm_id for arm in arms),
            len(feature_extractor.FEATURE_NAMES),
        )

    if static_completion is not None:
        model_client = StaticModelClient(
            completion=static_completion,
            model_name=config.model.model_name,
        )
    elif config.model.model_name != "MODEL_PLACEHOLDER":
        if is_anthropic_model_name(config.model.model_name):
            model_client = AnthropicBatchModelClient(model_name=config.model.model_name)
        else:
            model_client = VLLMModelClient(
                model_name=config.model.model_name,
                vllm_config=config.vllm,
            )
    else:
        model_client = PlaceholderModelClient(model_name=config.model.model_name)

    return ExperimentRunner(
        model_client=model_client,
        selector=selector,
        generation_config=config.generation,
        reward_config=config.token_reward if config.token_aware else config.reward,
        bara_config=config.bara,
        feature_extractor=feature_extractor,
        budget_allocator=budget_allocator,
        compute_features_and_reward=enable_feature_extractor,
        arms=tuple(arms),
        dataset_name=config.dataset_name,
        batch_size=config.batch_size,
        generation_seed=config.generation_seed,
        progress_interval=config.progress_interval,
        self_test_signature_compatible=False,
    )


def _load_tasks(config: ExperimentConfig):
    tasks = load_codegen_tasks(config.dataset_name, limit=config.limit)
    if config.task_order_seed is not None:
        random.Random(config.task_order_seed).shuffle(tasks)
    return tasks


def _summarize_task_kinds(tasks) -> str:
    counts = Counter((task.task_kind or "unspecified") for task in tasks)
    return ",".join(f"{task_kind}={counts[task_kind]}" for task_kind in sorted(counts))


def _write_records(*, output_path: Path, records) -> None:
    write_jsonl(output_path, [record.to_dict() for record in records])


def _write_summary(*, summary_path: Path, summary: dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_bara_outputs(*, config: ExperimentConfig, arms, records) -> None:
    budget_label = format_budget_group_label(
        bara_per_task_completion_token_budget(config.bara, config.dataset_name)
    )
    output_path = _bara_output_path(config=config, budget_label=budget_label)
    summary_path = output_path.with_suffix(".summary.json")
    _write_records(output_path=output_path, records=records)
    _write_summary(summary_path=summary_path, summary=build_experiment_summary(records, arms=tuple(arms)))
    logging.info("[BARA] wrote records=%s -> %s", len(records), output_path)


def _bara_output_path(*, config: ExperimentConfig, budget_label: str | None) -> Path:
    variant = _bara_variant(config)
    seed = config.task_order_seed
    allocator_prefix = _bara_allocator_prefix(config)
    use_seed_results = seed is not None and (variant == "bara" or seed != 0)
    if allocator_prefix:
        results_root = DEFAULT_ABLATION_RESULTS_DIR
    elif use_seed_results:
        results_root = config.paths.seed_results_dir
        if results_root is None:
            main_root = config.paths.experiment_results_dir
            results_root = main_root.parent / "seeds" if main_root.name == "main" else main_root / "seeds"
    else:
        results_root = config.paths.experiment_results_dir
    routed_paths = replace(config.paths, experiment_results_dir=results_root)

    budget_suffix = f"_{budget_label}" if budget_label else ""
    seed_suffix = ""
    if seed is not None and (variant == "bara" or seed != 0):
        seed_suffix = f"_seed{int(seed)}"

    if variant.startswith("static_order") or variant in {"bara", "uniform"}:
        filename = f"{config.model.model_name}_{allocator_prefix}{variant}{seed_suffix}{budget_suffix}.jsonl"
    else:
        arm_label = format_bara_arm_ids(config.arm_ids)
        filename = (
            f"{config.model.model_name}_{allocator_prefix}{variant}_{arm_label}"
            f"{seed_suffix}{budget_suffix}.jsonl"
        )
    return routed_paths.model_results_dir(config.model.model_name, config.dataset_name, token_optimized=config.token_aware) / filename


def _bara_variant(config: ExperimentConfig) -> str:
    bara = config.bara
    if getattr(bara, "static_run", False):
        static_order = str(getattr(bara, "static_order", "A0-A1-A2") or "A0-A1-A2")
        return f"static_order{static_order}"
    if getattr(bara, "uniform_run", False):
        return "uniform"
    return "bara"


def _bara_allocator_prefix(config: ExperimentConfig) -> str:
    allocator = str(getattr(config.bara, "budget_allocator", "") or "").strip().lower()
    default_allocator = str(DEFAULT_EXPERIMENT_CONFIG.bara.budget_allocator).strip().lower()
    if not allocator or allocator == default_allocator:
        return ""
    return f"allocator-{allocator}_"


if __name__ == "__main__":
    raise SystemExit(main())
