PROMPT = """
Write the final Python code for the task.

Rules:
- Preserve the function signature, imports, and starter scaffold required by [Task].
- Preserve the input representation, return representation, and mutation behavior implied by [Task].
- Implement only behavior explicitly required by [Task].
- Output code only.

[Task]
{task_prompt}
"""