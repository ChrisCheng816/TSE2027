from __future__ import annotations

import ast
import gc
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod

from .config import VLLMConfig, resolve_model_path
from .output_parsing import (
    extract_fenced_code_blocks as _extract_fenced_code_blocks,
    extract_first_fenced_code_block as _extract_first_fenced_code_block,
    remove_standalone_code_fence_lines as _shared_remove_standalone_code_fence_lines,
    split_structured_sections,
    strip_wrapping_code_fences as _shared_strip_wrapping_code_fences,
)
from .types import GenerationConfig, ModelOutput, PromptBundle


class ModelClient(ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model_path = model_name

    @abstractmethod
    def generate(self, prompt: PromptBundle, config: GenerationConfig) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_batch(
        self,
        prompts: list[PromptBundle],
        config: GenerationConfig,
        *,
        n: int = 1,
        seed: int | None = None,
    ) -> list[list[ModelOutput]]:
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def count_prompt_tokens(self, prompt: PromptBundle) -> int:
        return self.count_tokens(_prompt_text_for_token_count(prompt))

    def close(self) -> None:
        return None


def load_model(
    model_name: str,
    *,
    vllm_config: VLLMConfig,
):
    _apply_vllm_environment(vllm_config)
    AutoProcessor, LLM, _ = _import_vllm_dependencies()
    resolved_model_path = resolve_model_path(model_name)

    normalized_model_name = str(resolved_model_path or "").strip().lower()
    if "deepseek-vl-" in normalized_model_name and "vl2" not in normalized_model_name:
        raise RuntimeError(
            "Detected DeepSeek-VL v1 checkpoint. "
            "This project's vLLM backend does not support DeepSeek-VL v1 "
            "(model_type='multi_modality'). "
            "Use a vLLM-compatible model (e.g., Qwen/InternVL), or switch to "
            "a DeepSeek-VL2 checkpoint if available."
        )

    resolved_tp_size = _resolve_tensor_parallel_size(
        visible_devices=vllm_config.cuda_visible_devices,
    )
    processor = AutoProcessor.from_pretrained(
        resolved_model_path,
        trust_remote_code=vllm_config.trust_remote_code,
    )
    processor._uml_model_family = _detect_model_family(resolved_model_path)
    requested_max_model_len = vllm_config.max_model_len
    try:
        llm = LLM(
            model=resolved_model_path,
            tensor_parallel_size=resolved_tp_size,
            dtype=vllm_config.dtype,
            trust_remote_code=vllm_config.trust_remote_code,
            max_model_len=requested_max_model_len,
            gpu_memory_utilization=vllm_config.gpu_memory_utilization,
        )
    except Exception as exc:
        exc_text = str(exc or "")
        if "model type `multi_modality`" in exc_text:
            raise RuntimeError(
                "vLLM cannot load this checkpoint because it uses DeepSeek-VL v1 "
                "(model_type='multi_modality'), which is not supported by the current "
                "vLLM/Transformers runtime in this project. "
                "Use a vLLM-compatible model (for example Qwen/InternVL, or DeepSeek-VL2 "
                "variants if available) instead."
            ) from exc
        derived_max_model_len = _extract_derived_max_model_len(exc_text)
        if (
            "User-specified max_model_len" in exc_text
            and derived_max_model_len is not None
            and requested_max_model_len > derived_max_model_len
        ):
            logging.warning(
                "Model '%s' supports max_model_len=%s; retrying (requested=%s).",
                resolved_model_path,
                derived_max_model_len,
                requested_max_model_len,
            )
            try:
                llm = LLM(
                    model=resolved_model_path,
                    tensor_parallel_size=resolved_tp_size,
                    dtype=vllm_config.dtype,
                    trust_remote_code=vllm_config.trust_remote_code,
                    max_model_len=derived_max_model_len,
                    gpu_memory_utilization=vllm_config.gpu_memory_utilization,
                )
                return llm, processor, resolved_model_path
            except Exception as retry_exc:
                exc = retry_exc
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "<unset>")
        raise RuntimeError(
            "vLLM engine initialization failed. "
            f"tensor_parallel_size(resolved={resolved_tp_size}), "
            f"CUDA_VISIBLE_DEVICES={visible_devices}. "
            "The tensor-parallel size is derived automatically from the number of "
            "visible GPUs. Try adjusting CUDA_VISIBLE_DEVICES or reducing "
            "gpu_memory_utilization."
        ) from exc
    return llm, processor, resolved_model_path


class VLLMModelClient(ModelClient):
    def __init__(
        self,
        model_name: str,
        *,
        vllm_config: VLLMConfig,
    ) -> None:
        super().__init__(model_name=model_name)
        self.vllm_config = vllm_config
        self.llm, self.processor, self.model_path = load_model(
            model_name=model_name,
            vllm_config=vllm_config,
        )

    def generate(self, prompt: PromptBundle, config: GenerationConfig) -> str:
        batch_outputs = self.generate_batch([prompt], config, n=1, seed=None)
        if not batch_outputs or not batch_outputs[0]:
            return ""
        return batch_outputs[0][0].cleaned_output

    def generate_batch(
        self,
        prompts: list[PromptBundle],
        config: GenerationConfig,
        *,
        n: int = 1,
        seed: int | None = None,
    ) -> list[list[ModelOutput]]:
        _, _, SamplingParams = _import_vllm_dependencies()
        rendered_prompts = [_render_prompt(self.processor, prompt) for prompt in prompts]
        prompt_token_counts = [_count_text_tokens(self.processor, rendered_prompt) for rendered_prompt in rendered_prompts]
        predictions: list[list[ModelOutput]] = [[] for _ in prompts]
        valid_indices: list[int] = []
        valid_rendered_prompts: list[str] = []
        max_model_len = getattr(self.vllm_config, "max_model_len", None)
        for index, rendered_prompt in enumerate(rendered_prompts):
            prompt_token_count = prompt_token_counts[index]
            if max_model_len is not None and prompt_token_count > max_model_len:
                logging.warning(
                    "Skipping prompt because rendered length %s exceeds max_model_len=%s (arm=%s phase=%s).",
                    prompt_token_count,
                    max_model_len,
                    prompts[index].arm_id,
                    prompts[index].phase,
                )
                continue
            valid_indices.append(index)
            valid_rendered_prompts.append(rendered_prompt)
        sampling_kwargs = {
            "max_tokens": config.max_new_tokens,
            "n": n,
        }
        if config.temperature is not None:
            sampling_kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            sampling_kwargs["top_p"] = config.top_p
        if seed is not None:
            sampling_kwargs["seed"] = seed
        if config.stop:
            sampling_kwargs["stop"] = list(config.stop)
        sampling_params = SamplingParams(**sampling_kwargs)
        if not valid_rendered_prompts:
            return predictions
        outputs = self.llm.generate(valid_rendered_prompts, sampling_params=sampling_params, use_tqdm=False)
        for prompt_index, output in zip(valid_indices, outputs):
            prompt = prompts[prompt_index]
            if not getattr(output, "outputs", None):
                predictions[prompt_index] = []
                continue
            prompt_predictions: list[ModelOutput] = []
            for candidate in output.outputs:
                raw_output = candidate.text or ""
                prompt_predictions.append(
                    ModelOutput(
                        raw_output=raw_output,
                        cleaned_output=_clean_model_output(raw_output, prompt),
                        completion_tokens=_candidate_token_count(self.processor, candidate),
                        prompt_tokens=prompt_token_counts[prompt_index],
                    )
                )
            predictions[prompt_index] = prompt_predictions
        return predictions

    def count_tokens(self, text: str) -> int:
        return _count_text_tokens(self.processor, text)

    def count_prompt_tokens(self, prompt: PromptBundle) -> int:
        return _count_text_tokens(self.processor, _render_prompt(self.processor, prompt))

    def close(self) -> None:
        llm = getattr(self, "llm", None)
        if llm is None:
            return
        try:
            engine = getattr(llm, "llm_engine", None)
            engine_core = getattr(engine, "engine_core", None)
            shutdown = getattr(engine_core, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception:
            logging.warning("vLLM engine shutdown failed; continuing cleanup.", exc_info=True)
        finally:
            self.llm = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logging.debug("Unable to clear CUDA cache during vLLM shutdown.", exc_info=True)


class AnthropicBatchModelClient(ModelClient):
    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        poll_interval_sec: float | None = None,
        max_wait_sec: float | None = None,
    ) -> None:
        super().__init__(model_name=model_name)
        self.model_path = resolve_model_path(model_name)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic batch models.")
        self.base_url = (base_url or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        self.poll_interval_sec = (
            float(os.getenv("ANTHROPIC_BATCH_POLL_INTERVAL_SEC", "10"))
            if poll_interval_sec is None
            else float(poll_interval_sec)
        )
        self.max_wait_sec = (
            float(os.getenv("ANTHROPIC_BATCH_MAX_WAIT_SEC", "172800"))
            if max_wait_sec is None
            else float(max_wait_sec)
        )
        self.anthropic_version = os.getenv("ANTHROPIC_VERSION", "2023-06-01")

    def generate(self, prompt: PromptBundle, config: GenerationConfig) -> str:
        batch_outputs = self.generate_batch([prompt], config, n=1, seed=None)
        if not batch_outputs or not batch_outputs[0]:
            return ""
        return batch_outputs[0][0].cleaned_output

    def generate_batch(
        self,
        prompts: list[PromptBundle],
        config: GenerationConfig,
        *,
        n: int = 1,
        seed: int | None = None,
    ) -> list[list[ModelOutput]]:
        del seed
        if n != 1:
            raise ValueError("Anthropic Message Batches currently support this runner with n=1 only.")
        if not prompts:
            return []
        requests_payload = [
            {
                "custom_id": _anthropic_custom_id(index),
                "params": self._message_params(prompt, config),
            }
            for index, prompt in enumerate(prompts)
        ]
        logging.info("[anthropic_batch] creating batch model=%s requests=%s", self.model_path, len(requests_payload))
        batch = self._request_json(
            "POST",
            "/v1/messages/batches",
            json_payload={"requests": requests_payload},
        )
        batch_id = str(batch.get("id") or "")
        if not batch_id:
            raise RuntimeError(f"Anthropic batch creation returned no id: {batch}")
        batch = self._wait_for_batch(batch_id)
        results_by_id = self._download_results(batch)
        outputs: list[list[ModelOutput]] = []
        for index, prompt in enumerate(prompts):
            custom_id = _anthropic_custom_id(index)
            result_row = results_by_id.get(custom_id)
            outputs.append([self._model_output_from_result(result_row, prompt)] if result_row else [])
        return outputs

    def count_tokens(self, text: str) -> int:
        return _approximate_token_count(text)

    def count_prompt_tokens(self, prompt: PromptBundle) -> int:
        return _approximate_token_count(_prompt_text_for_token_count(prompt))

    def _message_params(self, prompt: PromptBundle, config: GenerationConfig) -> dict[str, object]:
        messages = _prompt_messages_for_render(prompt)
        system_text = ""
        api_messages: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            if not content:
                continue
            if role == "system":
                system_text = content
            elif role in {"user", "assistant"}:
                api_messages.append({"role": role, "content": content})
        if not api_messages:
            api_messages.append({"role": "user", "content": prompt.user or prompt.raw_prompt or ""})
        params: dict[str, object] = {
            "model": self.model_path,
            "max_tokens": max(1, int(config.max_new_tokens)),
            "messages": api_messages,
        }
        if system_text:
            params["system"] = system_text
        omit_sampling_params = _anthropic_omits_sampling_params(self.model_path)
        if config.temperature is not None and not omit_sampling_params:
            params["temperature"] = float(config.temperature)
        if config.top_p is not None and not omit_sampling_params and config.temperature is None:
            params["top_p"] = float(config.top_p)
        if config.stop:
            params["stop_sequences"] = list(config.stop)
        return params

    def _wait_for_batch(self, batch_id: str) -> dict[str, object]:
        start = time.monotonic()
        while True:
            batch = self._request_json("GET", f"/v1/messages/batches/{batch_id}")
            status = str(batch.get("processing_status") or "")
            counts = batch.get("request_counts") or {}
            logging.info("[anthropic_batch] status batch=%s status=%s counts=%s", batch_id, status, counts)
            if status == "ended":
                return batch
            if time.monotonic() - start > self.max_wait_sec:
                raise TimeoutError(f"Anthropic batch {batch_id} did not finish within {self.max_wait_sec} seconds.")
            self._sleep(self.poll_interval_sec)

    def _download_results(self, batch: dict[str, object]) -> dict[str, dict[str, object]]:
        batch_id = str(batch.get("id") or "")
        candidates = [f"/v1/messages/batches/{batch_id}/results"]
        results_url = str(batch.get("results_url") or "")
        if results_url:
            candidates.insert(0, results_url)
        last_error: Exception | None = None
        for path_or_url in candidates:
            try:
                text = self._request_text("GET", path_or_url)
                results: dict[str, dict[str, object]] = {}
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    custom_id = str(row.get("custom_id") or "")
                    if custom_id:
                        results[custom_id] = row
                return results
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Unable to download Anthropic batch results for {batch_id}") from last_error

    def _model_output_from_result(self, row: dict[str, object] | None, prompt: PromptBundle) -> ModelOutput:
        if not row:
            return ModelOutput(raw_output="", cleaned_output="", completion_tokens=0, prompt_tokens=0)
        result = row.get("result")
        if not isinstance(result, dict):
            return ModelOutput(raw_output="", cleaned_output="", completion_tokens=0, prompt_tokens=0)
        result_type = str(result.get("type") or "")
        if result_type != "succeeded":
            logging.warning("[anthropic_batch] request failed custom_id=%s result=%s", row.get("custom_id"), result)
            return ModelOutput(raw_output="", cleaned_output="", completion_tokens=0, prompt_tokens=0)
        message = result.get("message")
        if not isinstance(message, dict):
            return ModelOutput(raw_output="", cleaned_output="", completion_tokens=0, prompt_tokens=0)
        raw_output = _anthropic_message_text(message)
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
        prompt_tokens = int((usage or {}).get("input_tokens") or 0)
        completion_tokens = int((usage or {}).get("output_tokens") or 0)
        return ModelOutput(
            raw_output=raw_output,
            cleaned_output=_clean_model_output(raw_output, prompt),
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
        )

    def _request_json(self, method: str, path_or_url: str, *, json_payload: dict[str, object] | None = None) -> dict[str, object]:
        response = self._request(method, path_or_url, json_payload=json_payload)
        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"Anthropic API returned non-JSON response: {response.text[:500]}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Anthropic API returned unexpected JSON payload: {data!r}")
        return data

    def _request_text(self, method: str, path_or_url: str) -> str:
        return self._request(method, path_or_url).text

    def _request(self, method: str, path_or_url: str, *, json_payload: dict[str, object] | None = None):
        import requests

        url = path_or_url if path_or_url.startswith("http") else f"{self.base_url}{path_or_url}"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }
        response = requests.request(method, url, headers=headers, json=json_payload, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(f"Anthropic API {method} {url} failed: {response.status_code} {response.text[:1000]}")
        return response

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def is_anthropic_model_name(model_name: str) -> bool:
    try:
        resolved = resolve_model_path(model_name)
    except Exception:
        return False
    return str(resolved).startswith("claude-")


def _anthropic_omits_sampling_params(model_name: str) -> bool:
    normalized = str(model_name or "").strip().lower()
    return normalized.startswith("claude-opus-4-8") or normalized.startswith("claude-opus-4-7")


def _anthropic_custom_id(index: int) -> str:
    return f"req_{int(index):06d}"


def _anthropic_message_text(message: dict[str, object]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            chunks.append(str(block.get("text") or ""))
    return "".join(chunks)


class PlaceholderModelClient(ModelClient):
    def generate(self, prompt: PromptBundle, config: GenerationConfig) -> str:
        raise NotImplementedError(
            "Replace PlaceholderModelClient with a real model backend or set a valid model alias "
            "in config.py / --model-name so the CLI can instantiate VLLMModelClient."
        )

    def generate_batch(
        self,
        prompts: list[PromptBundle],
        config: GenerationConfig,
        *,
        n: int = 1,
        seed: int | None = None,
    ) -> list[list[ModelOutput]]:
        raise NotImplementedError(
            "Replace PlaceholderModelClient with a real model backend or set a valid model alias "
            "in config.py / --model-name so the CLI can instantiate VLLMModelClient."
        )

    def count_tokens(self, text: str) -> int:
        return _approximate_token_count(text)


class StaticModelClient(ModelClient):
    def __init__(self, completion: str, model_name: str = "STATIC_PLACEHOLDER") -> None:
        super().__init__(model_name=model_name)
        self.completion = completion
        self.processor = _load_processor_for_token_counting(model_name)

    def generate(self, prompt: PromptBundle, config: GenerationConfig) -> str:
        return self.completion

    def generate_batch(
        self,
        prompts: list[PromptBundle],
        config: GenerationConfig,
        *,
        n: int = 1,
        seed: int | None = None,
    ) -> list[list[ModelOutput]]:
        completion_tokens = _approximate_token_count(self.completion)
        return [
            [
                ModelOutput(
                    raw_output=self.completion,
                    cleaned_output=self.completion,
                    completion_tokens=completion_tokens,
                    prompt_tokens=self.count_tokens(_render_prompt(self.processor, prompt)),
                )
                for _ in range(max(n, 1))
            ]
            for prompt in prompts
        ]

    def count_tokens(self, text: str) -> int:
        if self.processor is not None:
            return _count_text_tokens(self.processor, text)
        return _approximate_token_count(text)

    def count_prompt_tokens(self, prompt: PromptBundle) -> int:
        rendered_prompt = _render_prompt(self.processor, prompt) if self.processor is not None else (prompt.raw_prompt or "")
        return self.count_tokens(rendered_prompt)


def _resolve_tensor_parallel_size(
    *,
    visible_devices: str | None = None,
) -> int:
    if visible_devices is None:
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    else:
        visible_devices = str(visible_devices).strip()
    if not visible_devices:
        raise ValueError(
            "cuda_visible_devices must expose at least one GPU because "
            "tensor_parallel_size is derived from it."
        )
    visible_gpu_count = len([device for device in visible_devices.split(",") if device.strip()])
    if visible_gpu_count <= 0:
        raise ValueError(
            "cuda_visible_devices must expose at least one GPU because "
            "tensor_parallel_size is derived from it."
        )
    return visible_gpu_count


def _apply_vllm_environment(vllm_config: VLLMConfig) -> None:
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = vllm_config.worker_multiproc_method
    os.environ["CUDA_VISIBLE_DEVICES"] = vllm_config.cuda_visible_devices
    os.environ["NCCL_P2P_DISABLE"] = vllm_config.nccl_p2p_disable


def _import_vllm_dependencies():
    try:
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
    except Exception as import_exc:  # pragma: no cover
        raise RuntimeError(
            "vLLM backend dependencies are not available. Install 'vllm' and 'transformers' "
            "in the current environment before using VLLMModelClient."
        ) from import_exc
    return AutoProcessor, LLM, SamplingParams


def _load_processor_for_token_counting(model_name: str):
    if not model_name or model_name == "STATIC_PLACEHOLDER" or model_name == "MODEL_PLACEHOLDER":
        return None
    try:
        from transformers import AutoProcessor
    except Exception:
        return None
    try:
        resolved_model_path = resolve_model_path(model_name)
    except Exception:
        return None
    try:
        return AutoProcessor.from_pretrained(
            resolved_model_path,
            trust_remote_code=True,
        )
    except Exception:
        logging.warning("Failed to load tokenizer/processor for static token counting: %s", model_name)
        return None


def _detect_model_family(model_name: str) -> str:
    normalized = str(model_name or "").strip().lower()
    if "qwen" in normalized:
        return "qwen"
    if "internvl" in normalized:
        return "internvl"
    if "deepseek" in normalized:
        return "deepseek"
    if "llava" in normalized:
        return "llava"
    if "minicpm" in normalized:
        return "minicpm"
    return "generic"


def _extract_derived_max_model_len(exc_text: str) -> int | None:
    patterns = (
        r"derived max_model_len(?:=| is )(\d+)",
        r"max_model_len(?:=| is )(\d+)",
        r"(\d+)\s*<\s*User-specified max_model_len",
        r"supports max_model_len=(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, exc_text)
        if match:
            return int(match.group(1))
    return None


def _render_prompt(processor, prompt: PromptBundle) -> str:
    tokenizer = getattr(processor, "tokenizer", None)
    chat_template_owner = tokenizer if tokenizer is not None else processor
    apply_chat_template = getattr(chat_template_owner, "apply_chat_template", None)
    messages = _prompt_messages_for_render(prompt)
    if callable(apply_chat_template):
        try:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            logging.warning("apply_chat_template rejected text-only messages; falling back to raw prompt.")
    return _prompt_text_for_token_count(prompt)


def _prompt_messages_for_render(prompt: PromptBundle) -> list[dict[str, str]]:
    if prompt.messages:
        system_messages = [message for message in prompt.messages if message.role == "system" and message.content]
        user_messages = [message for message in prompt.messages if message.role == "user" and message.content]
        messages: list[dict[str, str]] = []
        if system_messages:
            messages.append({"role": "system", "content": system_messages[-1].content})
        if user_messages:
            messages.append({"role": "user", "content": user_messages[-1].content})
        if messages:
            return messages
    messages = []
    if prompt.system:
        messages.append({"role": "system", "content": prompt.system})
    if prompt.user:
        messages.append({"role": "user", "content": prompt.user})
    return messages


def _prompt_text_for_token_count(prompt: PromptBundle) -> str:
    messages = _prompt_messages_for_render(prompt)
    if messages:
        return "\n\n".join(message["content"] for message in messages if message["content"])
    return prompt.raw_prompt or ""


def _clean_model_output(raw_text: str, prompt: PromptBundle) -> str:
    cleaned = raw_text or ""
    for prefix in (prompt.raw_prompt, prompt.user, prompt.system):
        if prefix and cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    cleaned = cleaned.strip()
    if prompt.phase == "draft_bundle":
        if _is_deepseek_bigcodebench_prompt(prompt):
            cleaned = _normalize_deepseek_bigcodebench_draft_bundle(cleaned)
        return _remove_standalone_code_fence_lines(_strip_wrapping_code_fences(cleaned))
    if prompt.arm_id == "A1" and prompt.phase == "final_code":
        return cleaned
    if _is_deepseek_bigcodebench_prompt(prompt):
        cleaned = _strip_deepseek_bigcodebench_structured_code_tail(cleaned)
    return _sanitize_python_completion(
        cleaned,
        preserve_runtime_entrypoint=_should_preserve_runtime_entrypoint(prompt),
    )


def _is_deepseek_bigcodebench_prompt(prompt: PromptBundle) -> bool:
    return (
        "deepseek" in str(getattr(prompt, "model_name", "") or "").lower()
        and "bigcodebench" in str(getattr(prompt, "dataset_name", "") or "").lower()
    )


def _normalize_deepseek_bigcodebench_draft_bundle(text: str) -> str:
    source = str(text or "").strip()
    if not source:
        return ""
    sections = split_structured_sections(
        source,
        aliases={
            "code": (),
            "tests": ("test", "testcase", "testcases"),
            "self_tests": ("selftest", "selftests", "selftestcase", "selftestcases"),
        },
    )
    if any(section.name == "code" for section in sections):
        return source

    tests_marker = re.search(
        r"(?im)^[ \t]*`?\[(?:TESTS?|SELF[_ -]?TESTS?|TEST[_ -]?CASES?|SELF[_ -]?TEST[_ -]?CASES?)\]`?[ \t]*$",
        source,
    )
    if tests_marker is None:
        return source

    leading_code = source[: tests_marker.start()].strip()
    code = _extract_fenced_code_block(leading_code)
    if code is None:
        code = _remove_standalone_code_fence_lines(_strip_wrapping_code_fences(leading_code))
    if not code:
        return source
    return f"[CODE]\n{code.strip()}\n\n{source[tests_marker.start():].strip()}"


def _strip_deepseek_bigcodebench_structured_code_tail(text: str) -> str:
    source = str(text or "").strip()
    if not source:
        return ""
    cutoff = re.search(
        r"(?im)^[ \t]*`?\[(?:CODE|TESTS?|SELF[_ -]?TESTS?|TEST[_ -]?CASES?|SELF[_ -]?TEST[_ -]?CASES?)\]`?[ \t]*$",
        source,
    )
    if cutoff is None:
        return source
    before = source[: cutoff.start()].strip()
    return before or source


def _sanitize_python_completion(text: str, *, preserve_runtime_entrypoint: bool = False) -> str:
    candidates = _candidate_python_completions(text)
    for candidate in candidates:
        stripped_test_tail = _strip_top_level_test_tail(
            candidate,
            preserve_runtime_entrypoint=preserve_runtime_entrypoint,
        )
        validated = _validated_python_module(stripped_test_tail)
        if validated is not None:
            return validated
    for candidate in candidates:
        validated = _validated_python_module(candidate)
        if validated is not None:
            return validated
    return candidates[0] if candidates else ""



def _candidate_python_completions(text: str) -> list[str]:
    candidates: list[str] = []
    base = _remove_standalone_code_fence_lines(_strip_wrapping_code_fences(text))

    marked_code = _extract_marked_code_section(base)
    if marked_code:
        candidates.append(marked_code)

    if base:
        candidates.append(base)

    fenced_code = _extract_fenced_code_block(text)
    if fenced_code:
        candidates.append(_remove_standalone_code_fence_lines(_strip_wrapping_code_fences(fenced_code)))

    return _unique_nonempty_candidates(candidates)


def _unique_nonempty_candidates(candidates: list[str]) -> list[str]:
    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(normalized)
    return unique_candidates


def _validated_python_module(text: str | None) -> str | None:
    normalized = str(text or "").strip()
    if not normalized:
        return None
    try:
        ast.parse(normalized)
    except SyntaxError:
        return None
    return normalized


def _validated_python_module_with_code_signal(text: str | None) -> str | None:
    validated = _validated_python_module(text)
    if validated is None or not _has_python_code_signal(validated):
        return None
    return validated


def _has_python_code_signal(text: str) -> bool:
    try:
        module = ast.parse(str(text or "").strip())
    except SyntaxError:
        return False
    signal_nodes = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.If,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Return,
    )
    for node in module.body:
        if isinstance(node, signal_nodes):
            return True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return True
    return False


def _strip_top_level_test_tail(text: str, *, preserve_runtime_entrypoint: bool = False) -> str:
    lines = text.splitlines()
    seen_named_blocks: set[str] = set()
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if line != stripped:
            continue
        if preserve_runtime_entrypoint and stripped.startswith("if __name__ =="):
            continue
        named_block = _top_level_named_block_name(stripped)
        if named_block is not None:
            if named_block in seen_named_blocks:
                return "\n".join(lines[:index]).strip()
            seen_named_blocks.add(named_block)
        if (
            stripped.startswith("def test_")
            or stripped.startswith("class Test")
            or stripped.startswith("if __name__ ==")
            or _is_test_artifact_runner(stripped)
        ):
            cutoff_index = _expand_test_artifact_cutoff(lines, index)
            return "\n".join(lines[:cutoff_index]).strip()
    return text.strip()


def _should_preserve_runtime_entrypoint(prompt: PromptBundle) -> bool:
    if str(getattr(prompt, "task_kind", "") or "").strip() == "stdin_stdout":
        return True
    user_text = (prompt.user or "").lower()
    return (
        "read the inputs from stdin" in user_text
        and "write the answer to stdout" in user_text
    )


def _top_level_named_block_name(line: str) -> str | None:
    match = re.match(r"(?:async\s+def|def|class)\s+([A-Za-z_]\w*)\b", line)
    if match is None:
        return None
    return match.group(1)


def _is_test_artifact_runner(line: str) -> bool:
    stripped = line.strip()
    return bool(
        re.match(r"test_\w+\s*\(", stripped)
        or stripped.startswith("unittest.main(")
    )


def _expand_test_artifact_cutoff(lines: list[str], start_index: int) -> int:
    probe_index = start_index - 1
    skipped_indexes: list[int] = []
    while probe_index >= 0:
        stripped = lines[probe_index].strip()
        if not stripped or stripped.startswith("#"):
            skipped_indexes.append(probe_index)
            probe_index -= 1
            continue
        break

    if probe_index >= 0 and _is_top_level_runtime_setup_statement(lines[probe_index]):
        cutoff_index = probe_index
        while cutoff_index - 1 >= 0:
            previous = lines[cutoff_index - 1].strip()
            if not previous or previous.startswith("#"):
                cutoff_index -= 1
                continue
            break
        return cutoff_index
    return start_index


def _is_top_level_runtime_setup_statement(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    if line.lstrip() != line:
        return False
    return not (
        stripped.startswith("import ")
        or stripped.startswith("from ")
        or stripped.startswith("def ")
        or stripped.startswith("async def ")
        or stripped.startswith("class ")
    )


def _extract_marked_code_section(text: str) -> str | None:
    sections = split_structured_sections(
        text,
        aliases={
            "code": (),
            "tests": ("test", "testcase", "testcases"),
            "self_tests": ("selftest", "selftests", "selftestcase", "selftestcases"),
            "plan": (),
            "contract": (),
        },
    )
    for section in sections:
        if section.name == "code":
            fenced_code = _extract_fenced_code_block(section.content)
            if fenced_code:
                return _remove_standalone_code_fence_lines(_strip_wrapping_code_fences(fenced_code))
            return _remove_standalone_code_fence_lines(_strip_wrapping_code_fences(section.content))
    return None


def _extract_fenced_code_block(text: str) -> str | None:
    return _extract_first_fenced_code_block(text)


def _strip_wrapping_code_fences(text: str) -> str:
    return _shared_strip_wrapping_code_fences(text)


def _remove_standalone_code_fence_lines(text: str) -> str:
    return _shared_remove_standalone_code_fence_lines(text)


def _candidate_token_count(processor, candidate) -> int:
    token_ids = getattr(candidate, "token_ids", None)
    if token_ids:
        return len(token_ids)
    return _count_text_tokens(processor, getattr(candidate, "text", "") or "")


def _count_text_tokens(processor, text: str) -> int:
    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer = tokenizer if tokenizer is not None else processor
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            return len(encode(text, add_special_tokens=False))
        except TypeError:
            try:
                return len(encode(text))
            except Exception:
                pass
        except Exception:
            pass
    return _approximate_token_count(text)


def _approximate_token_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
