PROMPT = """
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
"""