PROMPT = """
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
"""