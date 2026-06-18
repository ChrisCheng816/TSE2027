#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PairStats:
    union_count: int
    intersection_count: int
    left_only_count: int
    right_only_count: int
    both_fail_count: int


@dataclass(frozen=True)
class AdaptiveStats:
    adaptive_file: Path
    num_tasks: int
    passed_count: int
    mean_completion_tokens: float | None
    arm_pull_counts: dict[str, int]


@dataclass(frozen=True)
class ModelStats:
    model_name: str
    reward_file: Path
    num_tasks: int
    arms: tuple[str, ...]
    single_counts: dict[str, int]
    all_union_count: int
    pair_stats: dict[tuple[str, str], PairStats]
    adaptive_stats: AdaptiveStats | None = None

    @property
    def best_single(self) -> tuple[str, int]:
        return max(self.single_counts.items(), key=lambda item: (item[1], item[0]))

    @property
    def best_pair(self) -> tuple[tuple[str, str], PairStats]:
        return max(
            self.pair_stats.items(),
            key=lambda item: (item[1].union_count, item[0]),
        )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_results_root = repo_root / "results" / "main"

    parser = argparse.ArgumentParser(
        description=(
            "Summarize reward_table experiment results with single-arm, pairwise, "
            "and multi-model comparisons."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_results_root,
        help=f"Experiment results root. Default: {default_results_root}",
    )
    parser.add_argument(
        "--dataset",
        default="bigcodebench-v0.1.4",
        help="Dataset name used to select the result bucket.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Model directory names to analyze. Default: auto-discover all models "
            "that contain a matching reward_table file."
        ),
    )
    parser.add_argument(
        "--sort-pairs-by",
        choices=("union_desc", "name"),
        default="union_desc",
        help="How to order pair rows in per-model summaries.",
    )
    return parser.parse_args()


def results_dataset_bucket(dataset: str) -> str | None:
    normalized_dataset = dataset.lower().replace("_", "-")
    if "bigcodebench" in normalized_dataset:
        return "pass_only"
    return None


def resolve_dataset_results_root(results_root: Path, dataset: str) -> Path:
    dataset_bucket = results_dataset_bucket(dataset)
    dataset_root = results_root / dataset_bucket if dataset_bucket is not None else results_root
    if dataset_bucket is not None and dataset_root.exists():
        return dataset_root
    return results_root


def discover_reward_files(results_root: Path, dataset: str) -> list[Path]:
    reward_files: list[Path] = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        matches = sorted(child.glob("*.reward_table.jsonl"))
        if matches:
            reward_files.append(matches[0])
    reward_files.sort(key=lambda path: (path.stat().st_mtime, path.name))
    return reward_files


def resolve_reward_files(
    results_root: Path,
    dataset: str,
    model_names: list[str] | None,
) -> list[Path]:
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")
    results_root = resolve_dataset_results_root(results_root, dataset)

    if model_names:
        reward_files: list[Path] = []
        for model_name in model_names:
            model_dir = results_root / model_name
            reward_file = model_dir / f"{model_name}.reward_table.jsonl"
            if not reward_file.exists():
                raise FileNotFoundError(f"Missing reward file: {reward_file}")
            reward_files.append(reward_file)
        return reward_files

    reward_files = discover_reward_files(results_root, dataset)
    if not reward_files:
        raise FileNotFoundError(
            f"No reward_table files found under {results_root} for dataset {dataset}"
        )
    return reward_files


def infer_model_name(reward_file: Path, dataset: str) -> str:
    suffix = ".reward_table.jsonl"
    if not reward_file.name.endswith(suffix):
        raise ValueError(f"Unexpected reward filename: {reward_file.name}")
    return reward_file.name[: -len(suffix)]


def _row_passed(row: dict[str, object]) -> bool:
    return (
        row.get("pass_at_k", 0) == 1.0
        or row.get("reward", 0) == 1.0
        or row.get("num_correct", 0) > 0
    )


def _load_adaptive_stats(results_dir: Path, model_name: str, dataset: str) -> AdaptiveStats | None:
    adaptive_files = sorted(results_dir.glob(f"{model_name}_adaptive_*.jsonl"))
    adaptive_file = adaptive_files[0] if adaptive_files else None
    if adaptive_file is None or not adaptive_file.exists():
        return None

    per_task: dict[str, dict[str, object]] = {}
    arm_pull_counts: dict[str, int] = {}
    completion_tokens: list[float] = []

    with adaptive_file.open() as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            row = json.loads(raw_line)
            task_id = str(row["task_id"])
            arm_id = str(row.get("arm_id") or "")
            arm_pull_counts[arm_id] = arm_pull_counts.get(arm_id, 0) + 1
            completion_value = row.get("completion_tokens")
            if completion_value is not None:
                completion_tokens.append(float(completion_value))
            per_task[task_id] = row

    return AdaptiveStats(
        adaptive_file=adaptive_file,
        num_tasks=len(per_task),
        passed_count=sum(1 for row in per_task.values() if _row_passed(row)),
        mean_completion_tokens=(
            sum(completion_tokens) / len(completion_tokens) if completion_tokens else None
        ),
        arm_pull_counts=arm_pull_counts,
    )


def load_model_stats(reward_file: Path, dataset: str) -> ModelStats:
    model_name = infer_model_name(reward_file, dataset)
    per_task: dict[str, dict[str, bool]] = {}
    arms: set[str] = set()

    with reward_file.open() as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            row = json.loads(raw_line)
            task_id = row["task_id"]
            arm_id = row["arm_id"]
            passed = _row_passed(row)
            per_task.setdefault(task_id, {})[arm_id] = passed
            arms.add(arm_id)

    ordered_arms = tuple(sorted(arms))
    single_counts = {
        arm: sum(task_results.get(arm, False) for task_results in per_task.values())
        for arm in ordered_arms
    }
    all_union_count = sum(
        any(task_results.get(arm, False) for arm in ordered_arms)
        for task_results in per_task.values()
    )

    pair_stats: dict[tuple[str, str], PairStats] = {}
    for left, right in combinations(ordered_arms, 2):
        union_count = 0
        intersection_count = 0
        left_only_count = 0
        right_only_count = 0
        both_fail_count = 0
        for task_results in per_task.values():
            left_passed = task_results.get(left, False)
            right_passed = task_results.get(right, False)
            if left_passed or right_passed:
                union_count += 1
            if left_passed and right_passed:
                intersection_count += 1
            elif left_passed:
                left_only_count += 1
            elif right_passed:
                right_only_count += 1
            else:
                both_fail_count += 1
        pair_stats[(left, right)] = PairStats(
            union_count=union_count,
            intersection_count=intersection_count,
            left_only_count=left_only_count,
            right_only_count=right_only_count,
            both_fail_count=both_fail_count,
        )

    return ModelStats(
        model_name=model_name,
        reward_file=reward_file,
        num_tasks=len(per_task),
        arms=ordered_arms,
        single_counts=single_counts,
        all_union_count=all_union_count,
        pair_stats=pair_stats,
        adaptive_stats=_load_adaptive_stats(reward_file.parent, model_name, dataset),
    )


def percent(count: int, total: int) -> str:
    return f"{(100.0 * count / total):.2f}%"


def signed_delta(value: int) -> str:
    return f"{value:+d}"


def render_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "  ".join("-" * width for width in widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in string_rows)
    return "\n".join(lines)


def sort_pair_items(
    pair_items: list[tuple[tuple[str, str], PairStats]],
    sort_mode: str,
) -> list[tuple[tuple[str, str], PairStats]]:
    if sort_mode == "name":
        return sorted(pair_items, key=lambda item: item[0])
    return sorted(
        pair_items,
        key=lambda item: (-item[1].union_count, item[0]),
    )


def print_model_summary(stats: ModelStats, sort_pairs_by: str) -> None:
    print(f"== {stats.model_name} ==")
    print(f"reward_file: {stats.reward_file}")
    print(f"num_tasks: {stats.num_tasks}")
    print()

    single_rows = [
        [arm, stats.single_counts[arm], percent(stats.single_counts[arm], stats.num_tasks)]
        for arm in stats.arms
    ]
    print("[Single Arms]")
    print(render_table(["arm", "passed", "pass_rate"], single_rows))
    print()

    pair_rows = []
    for (left, right), pair in sort_pair_items(
        list(stats.pair_stats.items()),
        sort_pairs_by,
    ):
        pair_rows.append(
            [
                f"{left}+{right}",
                pair.union_count,
                percent(pair.union_count, stats.num_tasks),
                pair.intersection_count,
                pair.left_only_count,
                pair.right_only_count,
                pair.both_fail_count,
            ]
        )
    print("[Pairwise]")
    print(
        render_table(
            [
                "pair",
                "passed",
                "pass_rate",
                "intersection",
                "left_only",
                "right_only",
                "both_fail",
            ],
            pair_rows,
        )
    )
    print()

    best_single_arm, best_single_count = stats.best_single
    best_pair_key, best_pair_stats = stats.best_pair
    overview_rows = [
        ["best_single", best_single_arm, best_single_count, percent(best_single_count, stats.num_tasks)],
        [
            "best_pair",
            "+".join(best_pair_key),
            best_pair_stats.union_count,
            percent(best_pair_stats.union_count, stats.num_tasks),
        ],
        [
            "all_arms_union",
            ",".join(stats.arms),
            stats.all_union_count,
            percent(stats.all_union_count, stats.num_tasks),
        ],
    ]
    print("[Overview]")
    print(render_table(["metric", "selection", "passed", "pass_rate"], overview_rows))
    print()

    if stats.adaptive_stats is not None:
        adaptive = stats.adaptive_stats
        adaptive_rows = [
            [arm_id, count, percent(count, adaptive.num_tasks)]
            for arm_id, count in sorted(adaptive.arm_pull_counts.items())
        ]
        print("[Adaptive]")
        print(f"adaptive_file: {adaptive.adaptive_file}")
        print(render_table(["arm", "pulls", "pull_rate"], adaptive_rows))
        print()

        adaptive_overview_rows = [
            ["adaptive", "selector", adaptive.passed_count, percent(adaptive.passed_count, adaptive.num_tasks)],
            [
                "adaptive_vs_best_single",
                f"selector-{best_single_arm}",
                signed_delta(adaptive.passed_count - best_single_count),
                percent(adaptive.passed_count, adaptive.num_tasks),
            ],
            [
                "adaptive_vs_best_pair",
                f"selector-{'+'.join(best_pair_key)}",
                signed_delta(adaptive.passed_count - best_pair_stats.union_count),
                percent(adaptive.passed_count, adaptive.num_tasks),
            ],
            [
                "adaptive_vs_all_arms_union",
                "selector-union",
                signed_delta(adaptive.passed_count - stats.all_union_count),
                percent(adaptive.passed_count, adaptive.num_tasks),
            ],
        ]
        print(render_table(["metric", "selection", "passed", "pass_rate"], adaptive_overview_rows))
        if adaptive.mean_completion_tokens is not None:
            print(f"mean_completion_tokens: {adaptive.mean_completion_tokens:.2f}")
        print()


def print_comparison(baseline: ModelStats, current: ModelStats) -> None:
    if baseline.num_tasks != current.num_tasks:
        raise ValueError(
            f"Task counts differ: {baseline.model_name}={baseline.num_tasks}, "
            f"{current.model_name}={current.num_tasks}"
        )
    if baseline.arms != current.arms:
        raise ValueError(
            f"Arm sets differ: {baseline.model_name}={baseline.arms}, "
            f"{current.model_name}={current.arms}"
        )

    print(f"== Comparison: {current.model_name} vs {baseline.model_name} ==")
    print()

    baseline_best_single_arm, baseline_best_single_count = baseline.best_single
    current_best_single_arm, current_best_single_count = current.best_single
    baseline_best_pair_key, baseline_best_pair_stats = baseline.best_pair
    current_best_pair_key, current_best_pair_stats = current.best_pair

    overview_rows = [
        [
            "best_single",
            f"{baseline_best_single_arm} ({baseline_best_single_count})",
            f"{current_best_single_arm} ({current_best_single_count})",
            signed_delta(current_best_single_count - baseline_best_single_count),
        ],
        [
            "best_pair",
            f"{'+'.join(baseline_best_pair_key)} ({baseline_best_pair_stats.union_count})",
            f"{'+'.join(current_best_pair_key)} ({current_best_pair_stats.union_count})",
            signed_delta(current_best_pair_stats.union_count - baseline_best_pair_stats.union_count),
        ],
        [
            "all_arms_union",
            str(baseline.all_union_count),
            str(current.all_union_count),
            signed_delta(current.all_union_count - baseline.all_union_count),
        ],
    ]
    print("[Overview Delta]")
    print(render_table(["metric", baseline.model_name, current.model_name, "delta"], overview_rows))
    print()

    single_rows = []
    for arm in baseline.arms:
        baseline_count = baseline.single_counts[arm]
        current_count = current.single_counts[arm]
        single_rows.append(
            [arm, baseline_count, current_count, signed_delta(current_count - baseline_count)]
        )
    print("[Single Arm Delta]")
    print(render_table(["arm", baseline.model_name, current.model_name, "delta"], single_rows))
    print()

    pair_rows = []
    for pair_key in sorted(baseline.pair_stats):
        baseline_count = baseline.pair_stats[pair_key].union_count
        current_count = current.pair_stats[pair_key].union_count
        pair_rows.append(
            [
                "+".join(pair_key),
                baseline_count,
                current_count,
                signed_delta(current_count - baseline_count),
            ]
        )
    print("[Pairwise Delta]")
    print(render_table(["pair", baseline.model_name, current.model_name, "delta"], pair_rows))
    print()

    if baseline.adaptive_stats is not None and current.adaptive_stats is not None:
        adaptive_rows = [
            [
                "adaptive",
                str(baseline.adaptive_stats.passed_count),
                str(current.adaptive_stats.passed_count),
                signed_delta(current.adaptive_stats.passed_count - baseline.adaptive_stats.passed_count),
            ]
        ]
        print("[Adaptive Delta]")
        print(render_table(["metric", baseline.model_name, current.model_name, "delta"], adaptive_rows))
        print()


def main() -> None:
    args = parse_args()
    reward_files = resolve_reward_files(args.results_root, args.dataset, args.models)
    stats_list = [load_model_stats(reward_file, args.dataset) for reward_file in reward_files]

    for stats in stats_list:
        print_model_summary(stats, args.sort_pairs_by)

    if len(stats_list) >= 2:
        baseline = stats_list[0]
        for current in stats_list[1:]:
            print_comparison(baseline, current)


if __name__ == "__main__":
    main()
