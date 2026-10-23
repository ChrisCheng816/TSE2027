# BARA

**BARA** is the **Budgeted Adaptive Ranking and Allocation Framework** for online code-generation workflow selection. For each task, BARA ranks the available workflows, allocates the remaining task budget, executes workflows sequentially, and updates its ranking and allocation state from observed outcomes.

The available workflows are Direct Coding, Plan-Structured Coding, and Test-Guided Repair.

## Environment

Use Python 3.11. Run all commands from the repository root.

The preferred installation uses the locked `uv` environment:

```bash
uv sync --frozen
source .venv/bin/activate
```

If `uv` is unavailable, create a local virtual environment and install the repository in editable mode:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

These commands create `.venv/` locally; it is intentionally not tracked by Git. Editable installation registers the repository source in that environment, so Python imports `codegen_agent` directly from the working tree.

Local Qwen runs use vLLM and CUDA. Anthropic runs require the corresponding API credentials.

## Run BARA

Run our framkework BARA with pass-only reward:

```bash
python -m codegen_agent \
  --run-mode bara \
  --model-name qwen2.5-32b \
  --budget 15000
```

Run the token-aware variant, which uses the token-cost reward:

```bash
python -m codegen_agent \
  --run-mode bara \
  --token-aware \
  --model-name qwen2.5-32b \
  --budget 15000
```

## Run Ranking Baselines

```bash
python -m codegen_agent \
  --run-mode bara \
  --uniform \
  --model-name qwen2.5-32b \
  --budget 15000
```

## Run Budget Allocation Baselines

```bash
python -m codegen_agent \
  --run-mode bara \
  --budget-allocator equal_split \
  --model-name qwen2.5-32b \
  --budget 15000
```

Useful options:

- `--budget-allocator equal_split`, `static_prior`, or `empirical_quantile` selects the allocation strategy.
- `--uniform` shuffles workflow order randomly; `--static --static-order A0-A1-A2` uses a fixed order.
- `--model-name qwen2.5-32b`, `qwen3-30b`, or `claude-opus-4.8` selects the model.
- `--budget 10000`, `15000`, or `20000` sets the per-task token budget.
- `--seeds 0 1 2` runs independent task-order seeds.
- `--cuda-visible-devices 0,1,2,3` selects CUDA devices for local vLLM runs.
- `--output-root /path/to/results` redirects `main/`, `seeds/`, `ablation/`, and `repeated_sampling/` outputs below a custom root.

## Results

Results use repository-relative paths. Each experiment bucket is grouped by dataset, feedback setting, and model.

- `results/main/<dataset>/pass_only/`: canonical pass-only BARA, Uniform seed 0, Static seed 0, and single-workflow results.
- `results/main/<dataset>/token_aware/`: corresponding main token-aware results.
- `results/repeated_sampling/<dataset>/`: repeated-sampling baselines grouped by model and workflow (`direct_coding`, `plan_structured_coding`, and `test_guided_repair`).
- `results/seeds/<dataset>/pass_only/`: pass-only BARA task-order seeds, including BARA seed 0, plus nonzero Uniform and Static seeds.
- `results/seeds/<dataset>/token_aware/`: corresponding token-aware seeded results.
- `results/sensitivity/pass_only/`: pass-only neighborhood-size sensitivity results.
- `results/sensitivity/token_aware/`: token-aware neighborhood-size sensitivity results.
- `results/ablation/pass_only/`: budget-allocation ablations.

Each JSONL output has a matching `.summary.json` file.

## Workflow Prompts

### Direct Coding

````text
Write the final Python code for the task.

Rules:
- Preserve the function signature, imports, and starter scaffold required by [Task].
- Preserve the input representation, return representation, and mutation behavior implied by [Task].
- Implement only behavior explicitly required by [Task].
- Output code only.

[Task]
{task_prompt}
````

### Plan-Structured Coding

````text
You are a python code generator, please solve the following task by first producing a concise self-plan and then implementing the final solution.
Here is the given programming task to solve:
`[TASK]`
`{task_prompt}`

Output exactly two sections in this order: `[PLAN]` and then `[CODE]`. Please add the labels `[PLAN]` and `[CODE]` before each section respectively.

`[PLAN]` requirements:
- Understand the natural language task description and arguments provided in `[TASK]`. Write a short implementation checklist before writing the code.
- Use these labels in order: Interface, Side Effects, Return Contract, Failure Contract.
- Keep the checklist concrete and implementation-guiding. Do not infer or fill in any behaviors not defined in `[TASK]`.
- Use the checklist to lock the required scaffold, function names, necessary imports, input/output format, and any specified error-handling behavior.
- Do not add validation, fallback logic, recovery behavior, or failure-handling rules unless they are explicitly required by the `[TASK]`.

`[CODE]` requirements:
- Please generate a correct python program that matches the specification and passes all tests based on the `[TASK]` and `[PLAN]`. You will use the following `starter code` to write the solution to the problem and enclose your code within delimiters.

```python
{starter_code}
```
````

### Test-Guided Repair

#### Draft

````text
Write the Python code and a small self-test suite for the task given below.

Rules:
- Output exactly two sections in this order: `[CODE]` and then `[TESTS]`. Please add the labels `[CODE]` and `[TESTS]` before each section respectively.
- In `[CODE]`, output only the Python solution code for the task.
- In `[TESTS]`, output only Python test code.
- Write five to seven self test cases as top-level functions to verify whether the generated function correctly solves the `[TASK]`, named consecutively as test_1, test_2, ..., test_7.
- Do not call the tests yourself. The harness will execute them one by one.
- Test only behavior that is explicit in `[TASK]`, `[Doc Examples]`, or starter code.
- `[TESTS]` must be valid Python containing only top-level test functions and any minimal imports they directly use.
- Preserve the function signature, imports, and starter scaffold required by `[TASK]`.
- Do not add explanations outside the two sections.

`[TASK]`
{task_prompt}
````

#### Repair

````text
`[Previous Code]`
{previous_code}

`[Tests]`
{self_tests}

`[Error Log]`
{error_log}

The initial code `[Previous Code]` did not pass the `[Tests]`. Repair the code in the `[Previous Code]` based on the `[Error Log]`.

Rules:
- Follow the task specification exactly.
- Use the `[Previous Code]`, `[Tests]`, and `[Error Log]` as the debugging context.
- Fix the observed error without solving a different problem.
- Preserve the required public interface and starter scaffold from `[Task]`.
- Output only Python code. Do not add explanations.

`[Task]`
{task_prompt}
````

## Supplementary Material for BARA

### A. Comparison with Related Paradigms

Table A1 contrasts BARA with LLM routing, cascade execution, and algorithm portfolios along the dimensions raised in the reviews.

| Dimension                       | LLM Routing           | Cascade Execution       | Algorithm Portfolio     | BARA                               |
| ------------------------------- | --------------------- | ----------------------- | ----------------------- | ---------------------------------- |
| Candidate type                  | Models                | Models                  | Solvers / prompts       | Inference workflows                |
| Selection scope                 | Single call           | Sequential, fixed order | One-shot, fixed mapping | Sequential, task-conditioned order |
| Task-level budget allocation    | No                    | No                      | No                      | Yes, per-attempt caps              |
| Workflow-specific cost modeling | No                    | No                      | No                      | Yes, censored log-normal           |
| Online adaptation               | Sometimes             | No                      | No                      | Yes, prequential evidence store    |
| Intermediate execution feedback | No                    | No                      | Rarely                  | Yes, accept/reject + token cost    |
| Optimization objective          | Cost-quality tradeoff | Cost reduction          | Solve rate              | Solve rate under fixed budget      |

**Table A1.** Comparison of BARA with related paradigms. BARA is the only method that jointly addresses task-conditioned ordering, per-attempt budget allocation, censored cost modeling, and online adaptation in a single framework.

### B. End-to-End Pseudocode

The following pseudocode describes the full BARA policy for one task episode. Equation references refer to the paper.

```text
Input:  task x, budget B, workflow pool G, evidence store D, neighborhood size k
Output: accepted solution or failure

1.  phi(x) <- Embed(x)                          // task representation
2.  N_k(x) <- kNN(phi(x), D, k)                 // retrieve k similar tasks
3.  for each g in G:
4.      I_g(x) <- {i in N_k(x) : g was executed on x_i}
5.      if |I_g(x)| > 0:
6.          mu_g(x) <- mean utility of g over I_g(x)  // Eq. (4)
7.      else:
8.          mu_g(x) <- 0
9.  sigma(x) <- sort G by descending mu_g(x)    // workflow ranking
        (ties broken uniformly at random;
         least-executed workflow placed first during coverage phase)
10. R <- B                                       // remaining budget
11. A <- {}                                      // attempted workflows
12. for j = 1, 2, ..., |G|:
13.     g_j <- sigma(x)[j]
14.     if R <= 0: break
15.     if j < |G|:
16.         fit log-normal (m_g, s_g) from censored costs in N_k(x) for g_j  // Eq. (5)
17.         b_j <- min(F^{-1}_{g_j}(1 - epsilon), R)                         // Eq. (6-7)
18.     else:
19.         b_j <- R                             // last workflow gets all remaining budget
20.     (c_j, outcome_j) <- Execute(g_j, x, b_j)   // run workflow under cap
21.     charged <- min(c_j, b_j)
22.     R <- R - charged
23.     A <- A union {g_j}
24.     u_j <- ComputeUtility(outcome_j, charged, B)  // Eq. (3), PO or TA
25.     D <- D union {(x, g_j, b_j, delta_j, charged, outcome_j)}  // update evidence
26.     if outcome_j == accepted:
27.         return solution
28. return failure
```

**Notes.**

- Line 9: during the coverage phase, the least-executed workflow is placed first until every workflow has at least n_0 = 20 first-position executions.
- Line 16: when fewer than a minimum number of usable observations are available for g_j, the allocator falls back to a prior weight split of 1:1:2 for Direct Coding, Plan-Structured Coding, and Test-Guided Repair respectively.
- Line 20: delta_j = 1 if execution completes before reaching b_j (exact cost observed), delta_j = 0 if truncated (cost right-censored at b_j).
