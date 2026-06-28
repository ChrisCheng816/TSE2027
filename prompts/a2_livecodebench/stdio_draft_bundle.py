PROMPT = """Here is the given programming task to solve:
`[TASK]`
{content}

You need to output two sections: `[CODE]` and `[TESTS]` to complete the task. Here are the requirements for each section:

`[CODE]` requirements:
Please generate a correct python program that matches the specification and passes all tests. Read the inputs from stdin, solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows:

``python 
# YOUR CODE HERE 
```

`[TESTS]` requirements:
- Provide five to seven distinct standard-input/output test cases to verify whether the generated program correctly solves the task and follow the format of the given sample in `[TASK]`.
- Each test case must contain one complete concrete input and the exact expected output.
- Construct each input strictly according to the problem specification and the input format described in the statement.
- The expected output must be the exact stdout for that input.
- Follow exactly this format for every test case:
[Input]
<full input here>
[Output]
<expected output here>
...

Write the Python code and a small self-test suite for the task given above. Output exactly two sections: `[CODE]` and `[TESTS]`. The output should include the labels `[CODE]` and `[TESTS]` before each section respectively. Do not add any explanations, comments, or any words outside or inside section `[CODE]` and `[TESTS]`.."""
