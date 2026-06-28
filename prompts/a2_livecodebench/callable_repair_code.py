PROMPT = """
`[Previous Code]`
{Draft}

`[Tests]`
{tests}

`[Error Log]`
{error_log}

`[Task]`
{content}

The initial code `[Previous Code]` did not pass the `[TESTS]`. Please fix the bug in the `[Previous Code]` based on the `[Error Log]` and make sure that the fixed code matches the specification and passes all tests.
You are not allowed to modify the given function signature and imports.  Output only the fixed code in one code block and enclose your code within delimiters. Do not add any explanations, comments, or any words. You will use the following starter code to write the fixed code to the problem.

```python
{starter_code}
```"""
