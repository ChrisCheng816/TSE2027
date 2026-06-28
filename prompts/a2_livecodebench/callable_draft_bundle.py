PROMPT = """Here is the given programming task to solve:
`[TASK]`
{content}

You need to output two sections: `[CODE]` and `[TESTS]` to complete the task. Here are the requirements for each section:

`[CODE]` requirements:
Please generate a correct python program that matches the specification and passes all tests. You will use the following starter code in `[TASK]` to write the solution to the problem and enclose your code within delimiters.

Here is the given `starter_coder` to start:
```python
{starter_code}
```

`[TESTS]` requirements:
- Provide five to seven distinct function-level test cases to verify whether the generated function correctly solves the task and follow the format of the given sample in `[TASK]`.
- Each test case must contain one concrete function call and the exact expected return value.
- Use the function name and interface specified in the `starter_code`.
- The expected output must be the exact return value of the function for that input.
- Follow exactly this format for every test case:
[Input]
<function call here>
[Output]
<expected return value here>
...

Write the Python code and a small self-test suite for the task given above. Output exactly two sections: `[CODE]` and `[TESTS]`. Please add the labels `[CODE]` and `[TESTS]` before each section respectively. Do not add any explanations, comments, or any words outside or inside section `[CODE]` and `[TESTS]`."""
