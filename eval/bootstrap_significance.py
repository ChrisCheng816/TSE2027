#!/usr/bin/env python3
"""Paired bootstrap significance tests for the BigCodeBench result tables.

The script uses one statistical procedure everywhere: paired bootstrap over
aligned task ids. Table-specific code only decides which per-task BARA and
baseline series correspond to the table cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_MODELS = ("qwen3-30b", "qwen2.5-32b", "claude-opus-4.8")
DEFAULT_BUDGETS = (10000, 15000, 20000)
DATASET = "bigcodebench-v0.1.4"
ARMS = ("A0", "A1", "A2")
ARM_LABEL = {"A0": "Direct Coding", "A1": "Plan-Structured Coding", "A2": "Test-Guided Repair"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RQ2_TABLE = {
    "qwen3-30b": {
        "Static": {"solve": [56.3, 60.5, 62.1], "firsthit": [48.1, 49.0, 49.4]},
        "Uniform": {"solve": [55.8, 60.5, 62.1], "firsthit": [46.3, 49.6, 50.1]},
        "BARA": {"solve": [59.0, 61.9, 62.1], "firsthit": [48.5, 53.6, 54.1]},
    },
    "qwen2.5-32b": {
        "Static": {"solve": [54.5, 60.2, 61.1], "firsthit": [46.7, 49.6, 50.4]},
        "Uniform": {"solve": [54.4, 60.1, 61.1], "firsthit": [46.1, 48.3, 49.2]},
        "BARA": {"solve": [58.0, 61.1, 61.4], "firsthit": [48.1, 52.2, 53.3]},
    },
    "claude-opus-4.8": {
        "Static": {"solve": [69.4, 74.4, 76.1], "firsthit": [62.5, 66.1, 65.6]},
        "Uniform": {"solve": [68.7, 75.1, 76.1], "firsthit": [62.4, 65.8, 66.1]},
        "BARA": {"solve": [72.4, 76.3, 76.9], "firsthit": [64.2, 66.4, 68.0]},
    },
}

RQ3_TABLE = {
    "qwen3-30b": {
        "Static prior split": {"solve": [56.1, 60.6, 61.8], "capped": [15.6, 5.7, 1.5]},
        "Equal split": {"solve": [56.5, 59.6, 61.2], "capped": [13.7, 6.9, 3.0]},
        "Empirical quantile": {"solve": [57.5, 60.8, 61.2], "capped": [17.8, 6.0, 3.7]},
        "BARA budget": {"solve": [59.0, 61.9, 62.1], "capped": [9.7, 1.2, 0.2]},
    },
    "qwen2.5-32b": {
        "Static prior split": {"solve": [56.9, 60.2, 60.9], "capped": [12.6, 3.9, 2.8]},
        "Equal split": {"solve": [56.5, 58.9, 60.4], "capped": [12.7, 6.3, 2.7]},
        "Empirical quantile": {"solve": [55.9, 61.0, 61.1], "capped": [17.7, 3.9, 3.5]},
        "BARA budget": {"solve": [58.0, 61.1, 61.4], "capped": [10.3, 1.6, 0.8]},
    },
    "claude-opus-4.8": {
        "Static prior split": {"solve": [67.9, 74.7, 75.4], "capped": [30.5, 6.4, 6.4]},
        "Equal split": {"solve": [71.0, 74.0, 75.4], "capped": [18.5, 7.5, 3.0]},
        "Empirical quantile": {"solve": [66.1, 73.0, 75.8], "capped": [33.0, 12.0, 5.8]},
        "BARA budget": {"solve": [72.4, 76.3, 76.9], "capped": [16.2, 3.6, 1.2]},
    },
}

RQ4_TABLE = {
    "qwen3-30b": {
        "PO": {"solve": [59.0, 61.9, 62.1], "firsthit": [48.4, 53.6, 54.1], "cost": [5.23, 6.95, 7.04]},
        "TA": {"solve": [58.9, 61.8, 62.0], "firsthit": [46.9, 49.4, 51.3], "cost": [4.85, 5.59, 5.99]},
    },
    "qwen2.5-32b": {
        "PO": {"solve": [58.0, 61.1, 61.4], "firsthit": [48.1, 52.2, 53.3], "cost": [4.95, 6.55, 7.14]},
        "TA": {"solve": [58.3, 61.3, 61.4], "firsthit": [48.0, 49.5, 50.7], "cost": [4.77, 5.40, 5.73]},
    },
    "claude-opus-4.8": {
        "PO": {"solve": [72.4, 76.3, 76.9], "firsthit": [64.2, 66.4, 68.0], "cost": [3.97, 5.37, 7.07]},
        "TA": {"solve": [72.5, 76.3, 77.0], "firsthit": [63.8, 66.1, 66.8], "cost": [3.93, 4.51, 4.64]},
    },
}

ALLOCATOR_FILE = {
    "Static prior split": "static_prior",
    "Equal split": "equal_split",
    "Empirical quantile": "empirical_quantile",
}


@dataclass(frozen=True)
class RunArrays:
    name: str
    path: Path | None
    task_ids: tuple[str, ...]
    solve: np.ndarray
    firsthit: np.ndarray
    capped: np.ndarray
    cost: np.ndarray
    capped_numer: int
    capped_denom: int


@dataclass(frozen=True)
class SeriesChoice:
    name: str
    files: tuple[str, ...]
    task_ids: tuple[str, ...]
    series: np.ndarray
    table_pt: float
    actual_pt: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-name", default=DATASET)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--budget", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def task_sort_key(task_id: str) -> tuple[int, str]:
    match = re.search(r"/(\d+)$", task_id)
    if match:
        return (int(match.group(1)), task_id)
    return (10**12, task_id)


def budget_index(budget: int) -> int:
    return list(DEFAULT_BUDGETS).index(int(budget))


def pct_point(series: np.ndarray) -> float:
    return 100.0 * float(np.mean(series))


def rounded_match(value: float, target: float) -> bool:
    return round(value, 1) == round(float(target), 1)


def load_run(path: Path, *, single_arm_budget: int | None = None) -> RunArrays:
    task_ids: list[str] = []
    solve: list[float] = []
    firsthit: list[float] = []
    capped: list[float] = []
    cost: list[float] = []
    capped_numer = 0
    capped_denom = 0
    seen: set[str] = set()
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id") or "")
            if not task_id:
                raise ValueError(f"{path}:{line_no} missing task_id")
            if task_id in seen:
                raise ValueError(f"{path}:{line_no} duplicate task_id={task_id}")
            seen.add(task_id)

            passed = float(row.get("pass_at_k") or row.get("passed") or row.get("reward") or 0.0) > 0.0
            if single_arm_budget is not None:
                accounted = float(row.get("prompt_tokens") or 0.0) + 4.0 * float(row.get("completion_tokens") or 0.0)
                solved = 1.0 if passed and accounted <= float(single_arm_budget) else 0.0
            else:
                solved = 1.0 if passed else 0.0

            summary = row.get("bara_summary") or row.get("order_run_summary") or {}
            trace = summary.get("trace") or []
            if summary.get("solved_rank") is not None:
                first = 1.0 if int(summary["solved_rank"]) == 1 else 0.0
            else:
                first = 1.0 if trace and bool(trace[0].get("passed")) else solved

            if trace:
                caps = [1.0 if item.get("over_allocated_token_cap") else 0.0 for item in trace]
                capped.append(float(np.mean(caps)))
                capped_numer += int(sum(caps))
                capped_denom += len(caps)
            else:
                capped.append(float("nan"))

            task_ids.append(task_id)
            solve.append(solved)
            firsthit.append(first)
            cost.append(float(row.get("total_tokens") or 0.0) / 1000.0)

    order = np.array([i for i, _ in sorted(enumerate(task_ids), key=lambda item: task_sort_key(item[1]))])
    return RunArrays(
        name=path.name,
        path=path,
        task_ids=tuple(task_ids[i] for i in order),
        solve=np.asarray(solve, dtype=np.float64)[order],
        firsthit=np.asarray(firsthit, dtype=np.float64)[order],
        capped=np.asarray(capped, dtype=np.float64)[order],
        cost=np.asarray(cost, dtype=np.float64)[order],
        capped_numer=capped_numer,
        capped_denom=capped_denom,
    )


def mean_run(name: str, runs: Iterable[RunArrays]) -> RunArrays:
    runs = tuple(runs)
    if not runs:
        raise ValueError("mean_run needs at least one run")
    task_ids = runs[0].task_ids
    for run in runs[1:]:
        if run.task_ids != task_ids:
            raise AssertionError(f"task ids do not align for {name}: {runs[0].name} vs {run.name}")
    return RunArrays(
        name=name,
        path=None,
        task_ids=task_ids,
        solve=np.column_stack([run.solve for run in runs]).mean(axis=1),
        firsthit=np.column_stack([run.firsthit for run in runs]).mean(axis=1),
        capped=np.column_stack([run.capped for run in runs]).mean(axis=1),
        cost=np.column_stack([run.cost for run in runs]).mean(axis=1),
        capped_numer=sum(run.capped_numer for run in runs),
        capped_denom=sum(run.capped_denom for run in runs),
    )


def glob_one(root: Path, pattern: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise AssertionError(f"expected exactly one file for {root / pattern}, found {len(paths)}")
    return paths[0]


def bara_runs(repo_root: Path, bucket: str, model: str, budget: int, dataset: str) -> list[RunArrays]:
    main_root = repo_root / "results" / "main" / bucket / model
    shuffle_root = repo_root / "results" / "seeds" / bucket / model
    paths = sorted(main_root.glob(f"{model}_bara_budget-{budget}.jsonl"))
    paths += sorted(shuffle_root.glob(f"{model}_bara_seed*_budget-{budget}.jsonl"))
    if len(paths) != 11:
        raise AssertionError(f"{model} {bucket} {budget}: expected 11 BARA runs, found {len(paths)}")
    return [load_run(path) for path in paths]


def static_runs(repo_root: Path, model: str, budget: int, dataset: str) -> list[RunArrays]:
    main_root = repo_root / "results" / "main" / "pass_only" / model
    shuffle_root = repo_root / "results" / "seeds" / "pass_only" / model
    paths = sorted(main_root.glob(f"{model}_static_order*_budget-{budget}.jsonl"))
    paths += sorted(shuffle_root.glob(f"{model}_static_order*_budget-{budget}.jsonl"))
    return [load_run(path) for path in paths]


def uniform_runs(repo_root: Path, model: str, budget: int, dataset: str) -> list[RunArrays]:
    main_root = repo_root / "results" / "main" / "pass_only" / model
    shuffle_root = repo_root / "results" / "seeds" / "pass_only" / model
    paths = sorted(main_root.glob(f"{model}_uniform_budget-{budget}.jsonl"))
    paths += sorted(shuffle_root.glob(f"{model}_uniform_seed*_budget-{budget}.jsonl"))
    return [load_run(path) for path in paths]


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def files_for_run(run: RunArrays) -> tuple[str, ...]:
    return tuple() if run.path is None else (display_path(run.path),)


def metric_value(run: RunArrays, metric: str) -> float:
    if metric == "solve":
        return pct_point(run.solve)
    if metric == "firsthit":
        return pct_point(run.firsthit)
    if metric == "capped":
        if run.capped_denom == 0:
            return float(np.nanmean(run.capped)) * 100.0
        return 100.0 * run.capped_numer / run.capped_denom
    if metric == "cost":
        return float(np.mean(run.cost))
    raise ValueError(metric)


def metric_series(run: RunArrays, metric: str) -> np.ndarray:
    if metric == "solve":
        return run.solve
    if metric == "firsthit":
        return run.firsthit
    if metric == "capped":
        return run.capped
    if metric == "cost":
        return run.cost
    raise ValueError(metric)


def select_run_matching_table(
    label: str,
    runs: list[RunArrays],
    *,
    solve_target: float,
    aux_target: float,
    aux_metric: str,
) -> RunArrays:
    def score(run: RunArrays) -> tuple[int, float, int, float, str]:
        solve_pt = metric_value(run, "solve")
        aux_pt = metric_value(run, aux_metric)
        return (
            0 if rounded_match(solve_pt, solve_target) else 1,
            abs(solve_pt - solve_target),
            0 if rounded_match(aux_pt, aux_target) else 1,
            abs(aux_pt - aux_target),
            run.name,
        )

    selected = min(runs, key=score)
    selected_score = score(selected)
    if selected_score[0] or selected_score[2]:
        print(
            f"[warn] {label}: closest run {selected.name} has "
            f"solve={metric_value(selected, 'solve'):.3f}, {aux_metric}={metric_value(selected, aux_metric):.3f}; "
            f"target solve={solve_target:.1f}, {aux_metric}={aux_target:.1f}"
        )
    return selected


def select_metric_series(
    label: str,
    runs: list[RunArrays],
    *,
    metric: str,
    target: float,
    prefer_mean: bool = False,
) -> RunArrays:
    def score(run: RunArrays) -> tuple[int, float, int, str]:
        pt = metric_value(run, metric)
        return (
            0 if rounded_match(pt, target) else 1,
            abs(pt - target),
            0 if prefer_mean and run.path is None else 1,
            run.name,
        )

    selected = min(runs, key=score)
    if score(selected)[0]:
        print(f"[warn] {label}: closest {metric} series {selected.name} has {metric}={metric_value(selected, metric):.3f}; target={target:.1f}")
    return selected


def choice_from_run(run: RunArrays, metric: str, table_pt: float) -> SeriesChoice:
    return SeriesChoice(
        name=run.name,
        files=files_for_run(run),
        task_ids=run.task_ids,
        series=metric_series(run, metric),
        table_pt=table_pt,
        actual_pt=metric_value(run, metric),
    )


def choice_from_runs(name: str, runs: Iterable[RunArrays], metric: str, table_pt: float) -> SeriesChoice:
    runs = tuple(runs)
    if not runs:
        raise ValueError(f"{name} needs at least one run")
    task_ids = runs[0].task_ids
    for run in runs[1:]:
        if run.task_ids != task_ids:
            raise AssertionError(f"task ids do not align for {name}: {runs[0].name} vs {run.name}")
    series = np.column_stack([metric_series(run, metric) for run in runs])
    actual_pt = float(np.nanmean(series)) if metric == "cost" else 100.0 * float(np.nanmean(series))
    files = tuple(display_path(run.path) for run in runs if run.path is not None)
    return SeriesChoice(
        name=name,
        files=files,
        task_ids=task_ids,
        series=series,
        table_pt=table_pt,
        actual_pt=actual_pt,
    )


def ensure_same_task_ids(label: str, *choices: SeriesChoice) -> None:
    task_ids = choices[0].task_ids
    for choice in choices[1:]:
        if choice.task_ids != task_ids:
            raise AssertionError(f"task ids do not align for {label}: {choices[0].name} vs {choice.name}")


def bootstrap_diff(bara: np.ndarray, baseline: np.ndarray, *, rng: np.random.Generator, samples: int, direction: str) -> tuple[float, float, float, float]:
    bara = np.asarray(bara, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if bara.ndim not in (1, 2) or baseline.ndim not in (1, 2):
        raise AssertionError(f"expected 1D or 2D series, got {bara.shape} vs {baseline.shape}")
    if bara.shape[0] != baseline.shape[0]:
        raise AssertionError(f"task dimension mismatch: {bara.shape} vs {baseline.shape}")
    if bara.ndim == 2 and baseline.ndim == 2 and bara.shape[1] != baseline.shape[1]:
        raise AssertionError(f"run dimension mismatch for paired run bootstrap: {bara.shape} vs {baseline.shape}")

    n = bara.shape[0]
    run_count = bara.shape[1] if bara.ndim == 2 else (baseline.shape[1] if baseline.ndim == 2 else 0)
    if run_count:
        diffs = np.empty(samples, dtype=np.float64)
        chunk_size = 256
        for start in range(0, samples, chunk_size):
            stop = min(start + chunk_size, samples)
            size = stop - start
            task_idx = rng.integers(0, n, size=(size, n), dtype=np.int32)
            run_idx = rng.integers(0, run_count, size=(size, run_count), dtype=np.int32)
            if bara.ndim == 2:
                bara_sample = bara[task_idx[:, :, None], run_idx[:, None, :]].mean(axis=(1, 2))
            else:
                bara_sample = bara[task_idx].mean(axis=1)
            if baseline.ndim == 2:
                baseline_sample = baseline[task_idx[:, :, None], run_idx[:, None, :]].mean(axis=(1, 2))
            else:
                baseline_sample = baseline[task_idx].mean(axis=1)
            if direction == "higher":
                diffs[start:stop] = bara_sample - baseline_sample
            elif direction == "lower":
                diffs[start:stop] = baseline_sample - bara_sample
            else:
                raise ValueError(direction)
    else:
        idx = rng.integers(0, n, size=(samples, n), dtype=np.int32)
        if direction == "higher":
            diffs = bara[idx].mean(axis=1) - baseline[idx].mean(axis=1)
        elif direction == "lower":
            diffs = baseline[idx].mean(axis=1) - bara[idx].mean(axis=1)
        else:
            raise ValueError(direction)

    if direction == "higher":
        point = float(np.mean(bara) - np.mean(baseline))
    elif direction == "lower":
        point = float(np.mean(baseline) - np.mean(bara))
    else:
        raise ValueError(direction)
    low, high = np.percentile(diffs, [2.5, 97.5])
    p_value = min(1.0, 2.0 * min(float(np.mean(diffs <= 0.0)), float(np.mean(diffs >= 0.0))))
    return point, float(low), float(high), p_value


def bootstrap_row(*, table: str, model: str, budget: int, metric: str, bara_label: str, bara: SeriesChoice, baseline_label: str, baseline: SeriesChoice, direction: str, rng: np.random.Generator, samples: int) -> dict[str, object]:
    ensure_same_task_ids(f"{table} {model} {budget} {metric}", bara, baseline)
    diff, low, high, p_value = bootstrap_diff(bara.series, baseline.series, rng=rng, samples=samples, direction=direction)
    significant = bool(low > 0.0 and diff > 0.0)
    if significant:
        result = "win"
    elif high < 0.0 and diff < 0.0:
        result = "loss"
    else:
        result = "ns"
    return {
        "table": table,
        "model": model,
        "budget": budget,
        "metric": metric,
        "direction": direction,
        "bara_label": bara_label,
        "bara_table_pt": bara.table_pt,
        "bara_actual_pt": bara.actual_pt,
        "bara_source": bara.name,
        "bara_files": ";".join(bara.files),
        "baseline_label": baseline_label,
        "baseline_table_pt": baseline.table_pt,
        "baseline_actual_pt": baseline.actual_pt,
        "baseline_source": baseline.name,
        "baseline_files": ";".join(baseline.files),
        "diff": diff,
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "significant": significant,
        "result": result,
        "n_tasks": len(bara.task_ids),
        "bara_runs": bara.series.shape[1] if bara.series.ndim == 2 else 1,
        "baseline_runs": baseline.series.shape[1] if baseline.series.ndim == 2 else 1,
    }


def single_arm_choice(repo_root: Path, model: str, arm: str, budget: int, dataset: str) -> SeriesChoice:
    path = repo_root / "results" / "main" / "pass_only" / model / f"{model}_{arm}.jsonl"
    run = load_run(path, single_arm_budget=budget)
    choice = choice_from_run(run, "solve", metric_value(run, "solve"))
    return SeriesChoice(ARM_LABEL[arm], choice.files, choice.task_ids, choice.series, choice.table_pt, choice.actual_pt)


def build_rq1_rows(args: argparse.Namespace, rng: np.random.Generator) -> list[dict[str, object]]:
    rows = []
    for model in args.models:
        for budget in args.budget:
            idx = budget_index(budget)
            bara = choice_from_runs(
                "stratified_11_bara_runs",
                bara_runs(args.repo_root, "pass_only", model, budget, args.dataset_name),
                "solve",
                RQ2_TABLE[model]["BARA"]["solve"][idx],
            )
            baselines = [single_arm_choice(args.repo_root, model, arm, budget, args.dataset_name) for arm in ARMS]
            baseline = max(baselines, key=lambda choice: choice.actual_pt)
            rows.append(bootstrap_row(table="rq1", model=model, budget=budget, metric="solve", bara_label="BARA", bara=bara, baseline_label=baseline.name, baseline=baseline, direction="higher", rng=rng, samples=args.bootstrap_samples))
    return rows


def build_rq2_rows(args: argparse.Namespace, rng: np.random.Generator) -> list[dict[str, object]]:
    rows = []
    for model in args.models:
        for budget in args.budget:
            idx = budget_index(budget)
            bara_run_set = bara_runs(args.repo_root, "pass_only", model, budget, args.dataset_name)
            static_run_set = static_runs(args.repo_root, model, budget, args.dataset_name)
            uniform_run_set = uniform_runs(args.repo_root, model, budget, args.dataset_name)
            bara = {
                "solve": choice_from_runs("stratified_11_bara_runs", bara_run_set, "solve", RQ2_TABLE[model]["BARA"]["solve"][idx]),
                "firsthit": choice_from_runs("stratified_11_bara_runs", bara_run_set, "firsthit", RQ2_TABLE[model]["BARA"]["firsthit"][idx]),
            }
            static_run = select_run_matching_table(f"RQ2 {model} {budget} static", static_run_set, solve_target=RQ2_TABLE[model]["Static"]["solve"][idx], aux_target=RQ2_TABLE[model]["Static"]["firsthit"][idx], aux_metric="firsthit")
            uniform_run = select_run_matching_table(f"RQ2 {model} {budget} uniform", uniform_run_set, solve_target=RQ2_TABLE[model]["Uniform"]["solve"][idx], aux_target=RQ2_TABLE[model]["Uniform"]["firsthit"][idx], aux_metric="firsthit")
            baselines = {
                "Static": {
                    "solve": choice_from_run(static_run, "solve", RQ2_TABLE[model]["Static"]["solve"][idx]),
                    "firsthit": choice_from_run(static_run, "firsthit", RQ2_TABLE[model]["Static"]["firsthit"][idx]),
                },
                "Uniform": {
                    "solve": choice_from_run(uniform_run, "solve", RQ2_TABLE[model]["Uniform"]["solve"][idx]),
                    "firsthit": choice_from_run(uniform_run, "firsthit", RQ2_TABLE[model]["Uniform"]["firsthit"][idx]),
                },
            }
            for metric in ("solve", "firsthit"):
                baseline_label = max(baselines, key=lambda name: baselines[name][metric].table_pt)
                rows.append(bootstrap_row(table="rq2", model=model, budget=budget, metric=metric, bara_label="BARA", bara=bara[metric], baseline_label=baseline_label, baseline=baselines[baseline_label][metric], direction="higher", rng=rng, samples=args.bootstrap_samples))
    return rows


def allocator_run(args: argparse.Namespace, model: str, budget: int, policy_label: str) -> RunArrays:
    policy = ALLOCATOR_FILE[policy_label]
    root = args.repo_root / "results" / "ablation" / "pass_only" / model
    path = glob_one(root, f"{model}_allocator-{policy}_bara_budget-{budget}.jsonl")
    return load_run(path)


def build_rq3_rows(args: argparse.Namespace, rng: np.random.Generator) -> list[dict[str, object]]:
    rows = []
    for model in args.models:
        for budget in args.budget:
            idx = budget_index(budget)
            bara_run_set = bara_runs(args.repo_root, "pass_only", model, budget, args.dataset_name)
            bara = {
                "solve": choice_from_runs("stratified_11_bara_runs", bara_run_set, "solve", RQ3_TABLE[model]["BARA budget"]["solve"][idx]),
                "capped": choice_from_runs("stratified_11_bara_runs", bara_run_set, "capped", RQ3_TABLE[model]["BARA budget"]["capped"][idx]),
            }
            baseline_choices = {}
            for label in ("Static prior split", "Equal split", "Empirical quantile"):
                run = allocator_run(args, model, budget, label)
                baseline_choices[label] = {
                    "solve": choice_from_run(run, "solve", RQ3_TABLE[model][label]["solve"][idx]),
                    "capped": choice_from_run(run, "capped", RQ3_TABLE[model][label]["capped"][idx]),
                }
            solve_label = max(baseline_choices, key=lambda label: baseline_choices[label]["solve"].table_pt)
            capped_label = min(baseline_choices, key=lambda label: baseline_choices[label]["capped"].table_pt)
            rows.append(bootstrap_row(table="rq3", model=model, budget=budget, metric="solve", bara_label="BARA budget", bara=bara["solve"], baseline_label=solve_label, baseline=baseline_choices[solve_label]["solve"], direction="higher", rng=rng, samples=args.bootstrap_samples))
            rows.append(bootstrap_row(table="rq3", model=model, budget=budget, metric="capped", bara_label="BARA budget", bara=bara["capped"], baseline_label=capped_label, baseline=baseline_choices[capped_label]["capped"], direction="lower", rng=rng, samples=args.bootstrap_samples))
    return rows


def build_rq4_rows(args: argparse.Namespace, rng: np.random.Generator) -> list[dict[str, object]]:
    rows = []
    for model in args.models:
        for budget in args.budget:
            idx = budget_index(budget)
            po_bara_runs = bara_runs(args.repo_root, "pass_only", model, budget, args.dataset_name)
            ta_bara_runs = bara_runs(args.repo_root, "token_aware", model, budget, args.dataset_name)
            for metric in ("solve", "firsthit", "cost"):
                po = choice_from_runs("stratified_11_po_bara_runs", po_bara_runs, metric, RQ4_TABLE[model]["PO"][metric][idx])
                ta = choice_from_runs("stratified_11_ta_bara_runs", ta_bara_runs, metric, RQ4_TABLE[model]["TA"][metric][idx])
                rows.append(bootstrap_row(table="rq4", model=model, budget=budget, metric=metric, bara_label="TA", bara=ta, baseline_label="PO", baseline=po, direction="lower" if metric == "cost" else "higher", rng=rng, samples=args.bootstrap_samples))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "All tests use paired bootstrap over aligned tasks; multi-run BARA methods additionally resample task-order runs.",
        "For direction=higher, diff=bara-baseline. For direction=lower, diff=baseline-bara.",
        "",
    ]
    for row in rows:
        unit = "K/task" if row["metric"] == "cost" else "%"
        scale = 1.0 if row["metric"] == "cost" else 100.0
        lines.append(
            f"{row['table']} {row['model']} {int(row['budget'])//1000}k {row['metric']}: "
            f"{row['bara_label']} table={float(row['bara_table_pt']):.3f}{unit} actual={float(row['bara_actual_pt']):.3f}{unit} source={row['bara_source']} | "
            f"{row['baseline_label']} table={float(row['baseline_table_pt']):.3f}{unit} actual={float(row['baseline_actual_pt']):.3f}{unit} source={row['baseline_source']} | "
            f"diff={float(row['diff'])*scale:.3f}{unit} CI=[{float(row['ci_low'])*scale:.3f},{float(row['ci_high'])*scale:.3f}] {row['result']}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (args.repo_root / "bootstrap")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    rq1 = build_rq1_rows(args, rng)
    rq2 = build_rq2_rows(args, rng)
    rq3 = build_rq3_rows(args, rng)
    rq4 = build_rq4_rows(args, rng)
    all_rows = rq1 + rq2 + rq3 + rq4

    write_csv(output_dir / "rq1_bootstrap.csv", rq1)
    write_csv(output_dir / "rq2_bootstrap.csv", rq2)
    write_csv(output_dir / "rq3_bootstrap.csv", rq3)
    write_csv(output_dir / "rq4_bootstrap.csv", rq4)
    write_csv(output_dir / "all_bootstrap.csv", all_rows)
    write_summary(output_dir / "summary.txt", all_rows)

    print((output_dir / "summary.txt").read_text())
    print(f"Wrote {output_dir / 'rq1_bootstrap.csv'}")
    print(f"Wrote {output_dir / 'rq2_bootstrap.csv'}")
    print(f"Wrote {output_dir / 'rq3_bootstrap.csv'}")
    print(f"Wrote {output_dir / 'rq4_bootstrap.csv'}")
    print(f"Wrote {output_dir / 'all_bootstrap.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
