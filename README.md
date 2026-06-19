# Code Generation Adaptive Runner

This repository contains the experiment runner, prompts, and BigCodeBench result artifacts used for adaptive code-generation workflow selection. The active workflows are:

- `A0`: direct coding
- `A1`: plan-structured coding
- `A2`: test-guided repair

The main adaptive setting ranks these workflows online for each task and allocates a per-task completion-token budget across attempted workflows.

## Environment

Use Python 3.11. The repository includes `pyproject.toml` and `uv.lock`; `uv` is the preferred installer because it recreates the locked environment.

```bash
uv sync --frozen
source .venv/bin/activate
```

If `uv` is not available, create a virtual environment and install the project with pip:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Model-backed runs require the model runtime configured in `config.py`. Local Qwen runs use vLLM and CUDA; Anthropic runs require the relevant API credentials in the environment.

## Run Adaptive

Run commands from the repository root. This example runs the online adaptive order setting on BigCodeBench with the default pass-only reward design and a 15k per-task completion-token budget:

```bash
python -m codegen_agent \
  --run-mode adaptive \
  --order_run \
  --model-name qwen2.5-32b \
  --budget 15000
```

Useful options:

- `--model-name qwen2.5-32b`, `qwen3-30b`, or `claude-opus-4.8` selects the model alias from `config.py`.
- `--budget 10000`, `15000`, or `20000` sets the per-task completion-token budget used by adaptive order runs.
- `--cuda-visible-devices 0,1,2,3` overrides the CUDA devices for local vLLM runs.

## Result Directories

Results are stored under `results/` using relative paths. Each model has its own subdirectory under the relevant experiment bucket.

- `results/main/pass_only/`: main pass-only reward runs, including single-workflow outputs, fixed order, random order, and adaptive order outputs.
- `results/main/token_aware/`: main token-aware reward runs for comparing token-aware reward design against pass-only.
- `results/seeds/pass_only/`: additional pass-only seeded runs used to aggregate random, fixed, and order statistics.
- `results/seeds/token_aware/`: additional token-aware seeded runs.
- `results/sensitivity/pass_only/`: neighborhood-size sensitivity runs for pass-only reward.
- `results/sensitivity/token_aware/`: neighborhood-size sensitivity runs for token-aware reward.
- `results/ablation/pass_only/`: budget-allocation ablation artifacts. These are kept for analysis but are not required to run the main adaptive command above.

Each JSONL output has a matching `.summary.json` file with aggregate metrics.

