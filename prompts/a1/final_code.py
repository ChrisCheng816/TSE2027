PROMPT = """You are a python code generator, please solve the following task by first producing a concise self-plan and then implementing the final solution.
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
"""


# PROMPT = """Please solve the given programming task with self-planning before implementation.
# Here is the given programming task to solve:
# `[TASK]`
# {task_prompt}

# Output exactly two sections in this order: `[PLAN]` and then `[CODE]`. Please add the labels `[PLAN]` and `[CODE]` before each section respectively.

# `[PLAN]` requirements:
# - The plan must have exactly these three parts in order:
#   1. Subproblems
#   2. Solution Steps
#   3. Edge Cases
# - In Subproblems, decompose the task into a few concrete subproblems that are explicitly required by [Task].
# - In Solution Steps, give a short ordered list of implementation steps. Keep them high-level and implementation-guiding. Do not write code or pseudo-code.
# - In Edge Cases, list only edge cases that are directly supported by the task statement, signature, examples, or constraints.

# `[CODE]` requirements:
# - Please generate a correct python program that matches the specification and passes all tests based on the `[TASK]` and `[PLAN]`. Think step by step. You will use the following `starter code` to write the solution to the problem and enclose your code within delimiters.

# ```python
# {starter_code}
# ```
# """