from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from codegen_agent.config import VLLMConfig
from codegen_agent.embeddings import QwenTextEncoder
from codegen_agent.models import _clean_model_output, _render_prompt
from codegen_agent.types import PromptMessage
from codegen_agent.models import StaticModelClient, VLLMModelClient
from codegen_agent.types import CodeGenTask, GenerationConfig, PromptBundle
from codegen_agent.workflow.a2_artifacts import extract_a2_draft_bundle
from codegen_agent.workflow.a1_artifacts import extract_a1_final_bundle


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        del tokenize, add_generation_prompt
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()


class _FakeCandidate:
    def __init__(self, text: str) -> None:
        self.text = text
        self.token_ids = [1, 2, 3]


class _FakeRequestOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [_FakeCandidate(text)]


class _LengthCheckingLLM:
    def __init__(self, max_prompt_tokens: int) -> None:
        self.max_prompt_tokens = max_prompt_tokens
        self.recorded_prompts: list[str] = []

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        del sampling_params, use_tqdm
        self.recorded_prompts = list(prompts)
        for prompt in prompts:
            if len(prompt.split()) > self.max_prompt_tokens:
                raise ValueError(
                    f"The decoder prompt (length {len(prompt.split())}) is longer than the maximum model length of {self.max_prompt_tokens}."
                )
        return [_FakeRequestOutput("def task_func(x):\n    return x\n") for _ in prompts]


class _FakeEmbeddingOutputPayload:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddingRequestOutput:
    def __init__(self, embedding: list[float]) -> None:
        self.outputs = _FakeEmbeddingOutputPayload(embedding)


class _RecordingEmbeddingLLM:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding
        self.calls: list[dict[str, object]] = []

    def embed(self, prompts, *, use_tqdm=False, pooling_params=None):
        self.calls.append(
            {
                "prompts": list(prompts),
                "use_tqdm": use_tqdm,
                "pooling_params": pooling_params,
            }
        )
        return [_FakeEmbeddingRequestOutput(self.embedding)]


class _WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(token_ids)


class _LengthCheckingEmbeddingLLM(_RecordingEmbeddingLLM):
    def __init__(self, embedding: list[float], *, max_prompt_tokens: int) -> None:
        super().__init__(embedding)
        self.max_prompt_tokens = max_prompt_tokens
        self.tokenizer = _WhitespaceTokenizer()

    def get_tokenizer(self):
        return self.tokenizer

    def embed(self, prompts, *, use_tqdm=False, pooling_params=None):
        self.calls.append(
            {
                "prompts": list(prompts),
                "use_tqdm": use_tqdm,
                "pooling_params": pooling_params,
            }
        )
        for prompt in prompts:
            if len(self.tokenizer.encode(prompt)) > self.max_prompt_tokens:
                raise ValueError(
                    f"The decoder prompt (length {len(self.tokenizer.encode(prompt))}) is longer than the maximum model length of {self.max_prompt_tokens}."
                )
        return [_FakeEmbeddingRequestOutput(self.embedding)]


class _FailOnceLengthCheckingEmbeddingLLM(_LengthCheckingEmbeddingLLM):
    def __init__(self, embedding: list[float], *, first_max_prompt_tokens: int, second_max_prompt_tokens: int) -> None:
        super().__init__(embedding, max_prompt_tokens=second_max_prompt_tokens)
        self.first_max_prompt_tokens = first_max_prompt_tokens

    def embed(self, prompts, *, use_tqdm=False, pooling_params=None):
        self.calls.append(
            {
                "prompts": list(prompts),
                "use_tqdm": use_tqdm,
                "pooling_params": pooling_params,
            }
        )
        if len(self.calls) == 1:
            for prompt in prompts:
                if len(self.tokenizer.encode(prompt)) > self.first_max_prompt_tokens:
                    raise ValueError(
                        f"The decoder prompt (length {len(self.tokenizer.encode(prompt))}) is longer than the maximum model length of {self.first_max_prompt_tokens}."
                    )
        for prompt in prompts:
            if len(self.tokenizer.encode(prompt)) > self.max_prompt_tokens:
                raise ValueError(
                    f"The decoder prompt (length {len(self.tokenizer.encode(prompt))}) is longer than the maximum model length of {self.max_prompt_tokens}."
                )
        return [_FakeEmbeddingRequestOutput(self.embedding)]


class VLLMModelClientTests(unittest.TestCase):
    def test_render_prompt_uses_full_message_list_including_system(self) -> None:
        prompt = PromptBundle(
            system="A1 system prompt",
            user="[Current Step]\nWrite code",
            arm_id="A1",
            raw_prompt="[SYSTEM]\n\nA1 system prompt\n\n[USER]\n\n[Current Step]\nWrite code",
            phase="final_code",
            messages=(
                PromptMessage(role="system", content="A1 system prompt"),
                PromptMessage(role="user", content="[Current Step]\nWrite code"),
            ),
        )

        rendered = _render_prompt(_FakeProcessor(), prompt)

        self.assertEqual("system: A1 system prompt\nuser: [Current Step]\nWrite code", rendered)

    def test_base_count_prompt_tokens_includes_system_message_content(self) -> None:
        client = StaticModelClient("def task_func(x):\n    return x\n")
        client.processor = None
        prompt = PromptBundle(
            system="A1 system prompt",
            user="[Current Step]\nWrite code",
            arm_id="A1",
            raw_prompt="[USER]\n\n[Current Step]\nWrite code",
            phase="final_code",
            messages=(
                PromptMessage(role="system", content="A1 system prompt"),
                PromptMessage(role="user", content="[Current Step]\nWrite code"),
            ),
        )

        counted = client.count_prompt_tokens(prompt)

        self.assertEqual(client.count_tokens("A1 system prompt\n\n[Current Step]\nWrite code"), counted)

    def test_qwen_text_encoder_truncates_overlong_embedding_prompts_to_max_model_len(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = QwenTextEncoder(
                model_name="Qwen/Qwen3-Embedding-0.6B",
                instruction="Embed this programming task.",
                output_dim=4,
                cache_dir=Path(tmpdir),
                vllm_config=VLLMConfig(max_model_len=6, cuda_visible_devices="3", gpu_memory_utilization=0.1),
            )
            fake_llm = _LengthCheckingEmbeddingLLM([1.0, 2.0, 3.0, 4.0], max_prompt_tokens=6)
            encoder._llm = fake_llm

            vector = encoder.encode("alpha beta gamma delta epsilon zeta eta theta")

        self.assertEqual(4, len(vector))
        self.assertEqual(1, len(fake_llm.calls))
        recorded_prompt = fake_llm.calls[0]["prompts"][0]
        self.assertLessEqual(len(recorded_prompt.split()), 6)

    def test_qwen_text_encoder_retries_with_shorter_prompt_after_embed_length_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = QwenTextEncoder(
                model_name="Qwen/Qwen3-Embedding-0.6B",
                instruction="Embed this programming task.",
                output_dim=4,
                cache_dir=Path(tmpdir),
                vllm_config=VLLMConfig(max_model_len=12, cuda_visible_devices="3", gpu_memory_utilization=0.1),
            )
            fake_llm = _FailOnceLengthCheckingEmbeddingLLM(
                [1.0, 2.0, 3.0, 4.0],
                first_max_prompt_tokens=5,
                second_max_prompt_tokens=5,
            )
            encoder._llm = fake_llm

            vector = encoder.encode("alpha beta gamma delta epsilon zeta eta theta")

        self.assertEqual(4, len(vector))
        self.assertEqual(2, len(fake_llm.calls))
        self.assertGreater(len(fake_llm.calls[0]["prompts"][0].split()), 5)
        self.assertLessEqual(len(fake_llm.calls[1]["prompts"][0].split()), 5)

    def test_qwen_text_encoder_reserves_token_headroom_before_embed_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = QwenTextEncoder(
                model_name="Qwen/Qwen3-Embedding-0.6B",
                instruction="Embed this programming task.",
                output_dim=4,
                cache_dir=Path(tmpdir),
                vllm_config=VLLMConfig(max_model_len=6, cuda_visible_devices="3", gpu_memory_utilization=0.1),
            )
            fake_llm = _LengthCheckingEmbeddingLLM([1.0, 2.0, 3.0, 4.0], max_prompt_tokens=5)
            encoder._llm = fake_llm

            vector = encoder.encode("alpha beta gamma delta epsilon zeta eta theta")

        self.assertEqual(4, len(vector))
        self.assertEqual(1, len(fake_llm.calls))
        self.assertLessEqual(len(fake_llm.calls[0]["prompts"][0].split()), 5)

    def test_qwen_text_encoder_passes_requested_output_dimensions_to_vllm_pooling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = QwenTextEncoder(
                model_name="Qwen/Qwen3-Embedding-0.6B",
                instruction="Embed this programming task.",
                output_dim=64,
                cache_dir=Path(tmpdir),
                vllm_config=VLLMConfig(cuda_visible_devices="3", gpu_memory_utilization=0.1),
            )
            fake_llm = _RecordingEmbeddingLLM([float(index) for index in range(64)])
            encoder._llm = fake_llm

            vector = encoder.encode("Sort the values and return a list.")

        self.assertEqual(64, len(vector))
        self.assertEqual(1, len(fake_llm.calls))
        pooling_params = fake_llm.calls[0]["pooling_params"]
        self.assertEqual(64, pooling_params.dimensions)

    def test_clean_model_output_repair_strips_appended_test_tail_from_realistic_repair_output(self) -> None:
        prompt = PromptBundle(
            system="",
            user="repair code",
            arm_id="A2",
            raw_prompt="repair code",
            phase="repair_code",
            messages=(PromptMessage(role="user", content="repair code"),),
        )

        raw_output = """import os

def task_func(path):
    return os.path.basename(path)

def helper():
    return "helper"

def test_1(candidate):
    payload = "unterminated
"""

        cleaned = _clean_model_output(raw_output, prompt)

        self.assertEqual(
            'import os\n\ndef task_func(path):\n    return os.path.basename(path)\n\ndef helper():\n    return "helper"',
            cleaned,
        )
        ast.parse(cleaned)

    def test_clean_model_output_repair_keeps_only_python_before_trailing_code_fence(self) -> None:
        prompt = PromptBundle(
            system="",
            user="repair code",
            arm_id="A2",
            raw_prompt="repair code",
            phase="repair_code",
            messages=(PromptMessage(role="user", content="repair code"),),
        )

        raw_output = """import random

def task_func():
    return random.randint(1, 3)

random.seed(42)
print(task_func())
```"""

        cleaned = _clean_model_output(raw_output, prompt)

        self.assertNotIn("```", cleaned)
        ast.parse(cleaned)

    def test_clean_model_output_repair_strips_runtime_test_setup_and_named_tests_from_valid_python_module(self) -> None:
        prompt = PromptBundle(
            system="",
            user="repair code",
            arm_id="A2",
            raw_prompt="repair code",
            phase="repair_code",
            messages=(PromptMessage(role="user", content="repair code"),),
        )

        raw_output = """import random

def task_func(password_length=10, salt="salty"):
    return password_length, salt

# Fixing the random seed for testing purposes
random.seed(0)

# Test cases
def test_1(candidate):
    assert candidate(12, "my_salt") == (12, "my_salt")

# Running tests
test_1(task_func)
"""

        cleaned = _clean_model_output(raw_output, prompt)

        self.assertIn('def task_func(password_length=10, salt="salty"):', cleaned)
        self.assertNotIn("random.seed(0)", cleaned)
        self.assertNotIn("def test_1(candidate):", cleaned)
        self.assertNotIn("test_1(task_func)", cleaned)
        ast.parse(cleaned)

    def test_clean_model_output_repair_strips_repair_narrative_before_duplicate_function(self) -> None:
        prompt = PromptBundle(
            system="",
            user="repair code",
            arm_id="A2",
            raw_prompt="repair code",
            phase="repair_code",
            messages=(PromptMessage(role="user", content="repair code"),),
        )

        raw_output = """import csv

def task_func(path):
    return {"path": path}

# Fix for test_1
def task_func(path):
    return {"fixed": path}
"""

        cleaned = _clean_model_output(raw_output, prompt)

        self.assertEqual(1, cleaned.count("def task_func(path):"))
        self.assertNotIn('return {"fixed": path}', cleaned)
        ast.parse(cleaned)

    def test_clean_model_output_repair_strips_modify_tests_narrative_tail(self) -> None:
        prompt = PromptBundle(
            system="",
            user="repair code",
            arm_id="A2",
            raw_prompt="repair code",
            phase="repair_code",
            messages=(PromptMessage(role="user", content="repair code"),),
        )

        raw_output = """import pandas as pd

def task_func(csv_file_path, column_name):
    return pd.read_csv(csv_file_path)

# To handle the test cases, we need to modify the test cases to use actual file paths instead of StringIO
# Here is the modified test_1 as an example, other tests should be similarly modified
def test_1(candidate):
    assert candidate("test.csv", "Date") is not None
"""

        cleaned = _clean_model_output(raw_output, prompt)

        self.assertIn("def task_func(csv_file_path, column_name):", cleaned)
        self.assertNotIn("def test_1(candidate):", cleaned)
        ast.parse(cleaned)

    def test_deepseek_bigcodebench_repair_strips_structured_test_tail(self) -> None:
        prompt = PromptBundle(
            system="",
            user="repair code",
            arm_id="A2",
            raw_prompt="repair code",
            phase="repair_code",
            messages=(PromptMessage(role="user", content="repair code"),),
            model_name="deepseek-v2-lite",
            dataset_name="bigcodebench-v0.1.4",
        )

        raw_output = """```python
def task_func():
    return "ok"
```
[TESTS]
import unittest
from task import task_func
"""

        cleaned = _clean_model_output(raw_output, prompt)

        self.assertEqual('def task_func():\n    return "ok"', cleaned)
        self.assertNotIn("[TESTS]", cleaned)
        ast.parse(cleaned)

    def test_clean_model_output_a1_final_preserves_structured_bundle_for_runner(self) -> None:
        prompt = PromptBundle(
            system="",
            user="final code",
            arm_id="A1",
            raw_prompt="final code",
            phase="final_code",
            messages=(PromptMessage(role="user", content="final code"),),
        )

        raw_output = """[PLAN]
Use the starter signature and return the doubled value.

[CODE]
```python
def task_func(x):
    return x * 2
```
This implementation doubles the input."""

        cleaned = _clean_model_output(raw_output, prompt)

        self.assertIn("[PLAN]", cleaned)
        self.assertIn("[CODE]", cleaned)
        self.assertIn("```python", cleaned)

    def test_a2_draft_parser_accepts_backtick_wrapped_markers_and_extracts_fenced_sections(self) -> None:
        raw_output = """`[CODE]`
```python
def task_func(x):
    return x * 2
```
This code doubles the input.

`[TESTS]`
```python
def test_1(candidate):
    assert candidate(2) == 4
```
These tests cover the basic case."""

        bundle = extract_a2_draft_bundle(raw_output)

        self.assertEqual("def task_func(x):\n    return x * 2", bundle.code)
        self.assertEqual("def test_1(candidate):\n    assert candidate(2) == 4", bundle.self_tests)
        self.assertNotIn("This code", bundle.code)
        self.assertNotIn("These tests", bundle.tests)
        ast.parse(bundle.code)
        ast.parse(bundle.self_tests)

    def test_a2_draft_parser_accepts_two_backtick_code_fences_from_prompt(self) -> None:
        raw_output = """[CODE]
``python
def solve() -> None:
    print(2)
```

[SELF_TESTS]
``python
def test_1(run_io):
    assert run_io("1\\n").strip() == "2"
```"""

        bundle = extract_a2_draft_bundle(raw_output)

        self.assertEqual("def solve() -> None:\n    print(2)", bundle.code)
        self.assertIn("def test_1(run_io):", bundle.self_tests)
        ast.parse(bundle.code)
        ast.parse(bundle.self_tests)








    def test_deepseek_bigcodebench_a2_draft_adds_missing_code_marker_before_tests(self) -> None:
        prompt = PromptBundle(
            system="",
            user="draft bundle",
            arm_id="A2",
            raw_prompt="draft bundle",
            phase="draft_bundle",
            messages=(PromptMessage(role="user", content="draft bundle"),),
            model_name="deepseek-v2-lite",
            dataset_name="bigcodebench-v0.1.4",
        )
        raw_output = """```python
def task_func(x):
    return x * 2
```
[TESTS]
```python
def test_1(candidate):
    assert candidate(2) == 4
```"""

        cleaned = _clean_model_output(raw_output, prompt)
        bundle = extract_a2_draft_bundle(cleaned)

        self.assertEqual("def task_func(x):\n    return x * 2", bundle.code)
        self.assertEqual("def test_1(candidate):\n    assert candidate(2) == 4", bundle.self_tests)

    def test_a1_bundle_parser_accepts_labeled_plan_and_extracts_fenced_code_only(self) -> None:
        raw_output = """[PLAN]
- **Interface**: `def task_func(x):`
- **Side Effects**: None
- **Return Contract**: `int`
- **Failure Contract**: None

[CODE]
```python
def task_func(x):
    return x * 2
```
This implementation doubles the input."""

        bundle = extract_a1_final_bundle(raw_output)

        self.assertIn("Interface", bundle.plan)
        self.assertEqual("def task_func(x):\n    return x * 2", bundle.code)
        self.assertNotIn("This implementation", bundle.code)
        ast.parse(bundle.code)

    def test_a1_bundle_parser_accepts_backtick_wrapped_section_markers(self) -> None:
        raw_output = """`[PLAN]`
1. Interface: `def task_func(x):`

`[CODE]`
```python
def task_func(x):
    return x * 2
```"""

        bundle = extract_a1_final_bundle(raw_output)

        self.assertIn("Interface", bundle.plan)
        self.assertEqual("def task_func(x):\n    return x * 2", bundle.code)
        ast.parse(bundle.code)

    def test_a1_bundle_parser_falls_back_to_fenced_python_when_sections_are_missing(self) -> None:
        raw_output = """```python
def task_func(x):
    # [PLAN]
    # double the input
    # [CODE]
    return x * 2
```"""

        bundle = extract_a1_final_bundle(raw_output)

        self.assertIn("did not include parseable", bundle.plan)
        self.assertIn("def task_func(x):", bundle.code)
        self.assertIn("return x * 2", bundle.code)
        ast.parse(bundle.code)







    def test_generate_batch_skips_prompts_that_exceed_max_model_len(self) -> None:
        client = object.__new__(VLLMModelClient)
        client.model_name = "fake-model"
        client.model_path = "fake-model"
        client.processor = _FakeProcessor()
        client.vllm_config = VLLMConfig(max_model_len=6)
        client.llm = _LengthCheckingLLM(max_prompt_tokens=6)

        short_prompt = PromptBundle(
            system="",
            user="short prompt",
            arm_id="A0",
            raw_prompt="one two three",
            messages=(PromptMessage(role="user", content="short prompt"),),
        )
        long_prompt = PromptBundle(
            system="",
            user="long prompt with many extra tokens",
            arm_id="A0",
            raw_prompt="one two three four five six",
            messages=(PromptMessage(role="user", content="long prompt with many extra tokens"),),
        )

        outputs = client.generate_batch(
            [short_prompt, long_prompt],
            GenerationConfig(max_new_tokens=32),
        )

        self.assertEqual(2, len(outputs))
        self.assertEqual(1, len(outputs[0]))
        self.assertEqual("def task_func(x):\n    return x", outputs[0][0].cleaned_output)
        self.assertGreater(outputs[0][0].prompt_tokens, 0)
        self.assertEqual([], outputs[1])
        self.assertEqual(["user: short prompt"], client.llm.recorded_prompts)

    def test_generate_batch_renders_user_only_message_for_chat_template(self) -> None:
        client = object.__new__(VLLMModelClient)
        client.model_name = "fake-model"
        client.model_path = "fake-model"
        client.processor = _FakeProcessor()
        client.vllm_config = VLLMConfig(max_model_len=64)
        client.llm = _LengthCheckingLLM(max_prompt_tokens=64)

        prompt = PromptBundle(
            system="",
            user="[Current Step]\nWrite final code",
            arm_id="A1",
            raw_prompt="fallback raw prompt",
            messages=(
                PromptMessage(role="user", content="step one"),
                PromptMessage(role="assistant", content="def task_func(x):\n    return None"),
                PromptMessage(role="user", content="[Current Step]\nWrite final code"),
            ),
        )

        outputs = client.generate_batch([prompt], GenerationConfig(max_new_tokens=32))

        self.assertEqual(1, len(outputs))
        self.assertEqual(1, len(outputs[0]))
        self.assertNotIn("assistant:", client.llm.recorded_prompts[0])
        self.assertNotIn("system:", client.llm.recorded_prompts[0])
        self.assertIn("user: [Current Step]\nWrite final code", client.llm.recorded_prompts[0])
        self.assertGreater(outputs[0][0].prompt_tokens, outputs[0][0].completion_tokens)


if __name__ == "__main__":
    unittest.main()
