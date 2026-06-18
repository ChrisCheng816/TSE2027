PROMPT = """You are a python code generator, please output the sections in exactly this order: `[PLAN]` and then `[CODE]`. The output should include the labels `[PLAN]` and `[CODE]` before each section respectively.Do o not add any text outside [PLAN] and [CODE]. Do not wrap the code in Markdown code fences.
For example, the output format should be exactly look like:
`[PLAN]`
# The plan your generated for this task.
`[CODE]`
# The complete Python test cases you generated. Output code only.
"""