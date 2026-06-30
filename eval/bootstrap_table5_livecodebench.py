#!/usr/bin/env python3
"""Paired bootstrap significance tests for the LiveCodeBench generalization table.

This entrypoint is intentionally separate from bootstrap_significance.py so the
existing RQ1-RQ4 bootstrap CLI and builders remain unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    from bootstrap_significance import (
        DEFAULT_BUDGETS,
        DEFAULT_REPEATED_SAMPLING_ROOT,
        PROJECT_ROOT,
        REPEATED_WORKFLOW_LABEL,
        bara_runs,
        bootstrap_row,
        choice_from_run,
        choice_from_runs,
        display_path,
        load_run,
        metric_value,
        single_arm_choices,
        write_csv,
    )
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package.
    from eval.bootstrap_significance import (
        DEFAULT_BUDGETS,
        DEFAULT_REPEATED_SAMPLING_ROOT,
        PROJECT_ROOT,
        REPEATED_WORKFLOW_LABEL,
        bara_runs,
        bootstrap_row,
        choice_from_run,
        choice_from_runs,
        display_path,
        load_run,
        metric_value,
        single_arm_choices,
        write_csv,
    )


DEFAULT_DATASET = "livecodebench-release_v6"
DEFAULT_MODELS = ("qwen3-30b", "qwen2.5-32b")
DEFAULT_OUTPUT_NAME = "table5_livecodebench_bootstrap.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--budget", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeated-sampling-root", type=Path, default=DEFAULT_REPEATED_SAMPLING_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    return parser.parse_args()


def _repeated_dataset_bucket(dataset: str) -> str:
    normalized_dataset = dataset.lower().replace("_", "-")
    if "livecodebench" in normalized_dataset:
        return "livecodebench"
    if "bigcodebench" in normalized_dataset:
        return "bigcodebench"
    return normalized_dataset


def available_repeated_sampling_choices(
    root: Path,
    model: str,
    budget: int,
    dataset: str,
) -> list:
    choices = []
    dataset_bucket = _repeated_dataset_bucket(dataset)
    model_root = root / dataset_bucket / model
    for workflow_dir in sorted(path for path in model_root.glob("*") if path.is_dir()):
        workflow = workflow_dir.name
        paths = sorted(workflow_dir.glob(f"{model}_{workflow}_{dataset}_budget-{budget}.jsonl"))
        if not paths:
            continue
        if len(paths) > 1:
            raise AssertionError(f"expected at most one repeated run for {workflow_dir}, found {len(paths)}")
        run = load_run(paths[0])
        label = REPEATED_WORKFLOW_LABEL.get(workflow, f"Repeated Sampling ({workflow})")
        choice = choice_from_run(run, "solve", metric_value(run, "solve"))
        choices.append(replace(choice, name=label, files=(display_path(paths[0]),)))
    if not choices:
        raise AssertionError(f"{model} budget-{budget}: found no repeated-sampling runs under {model_root}")
    return choices


def build_table5_rows(args: argparse.Namespace, rng: np.random.Generator) -> list[dict[str, object]]:
    rows = []
    for model in args.models:
        for budget in args.budget:
            bara_run_set = bara_runs(args.repo_root, "pass_only", model, budget, args.dataset_name)
            bara_choice = choice_from_runs(
                "stratified_11_bara_runs",
                bara_run_set,
                "solve",
                0.0,
            )
            bara = replace(bara_choice, table_pt=bara_choice.actual_pt)
            best_single = max(
                single_arm_choices(args.repo_root, model, budget, args.dataset_name),
                key=lambda choice: choice.actual_pt,
            )
            best_repeated = max(
                available_repeated_sampling_choices(
                    args.repeated_sampling_root,
                    model,
                    budget,
                    args.dataset_name,
                ),
                key=lambda choice: choice.actual_pt,
            )
            for comparison, baseline in (
                ("best_single_arm", best_single),
                ("best_repeated_sampling", best_repeated),
            ):
                row = bootstrap_row(
                    table="table5",
                    model=model,
                    budget=budget,
                    metric="solve",
                    bara_label="BARA",
                    bara=bara,
                    baseline_label=baseline.name,
                    baseline=baseline,
                    direction="higher",
                    rng=rng,
                    samples=args.bootstrap_samples,
                )
                row["comparison"] = comparison
                rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (args.repo_root / "bootstrap")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    rows = build_table5_rows(args, rng)
    path = output_dir / args.output_name
    write_csv(path, rows)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
