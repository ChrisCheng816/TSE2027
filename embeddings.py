from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

from .config import EmbeddingConfig, VLLMConfig


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        return list(values)
    return [value / norm for value in values]


class NoTextEncoder:
    dim = 0

    def encode(self, text: str) -> list[float]:
        del text
        return []


class FrozenHashTextEncoder:
    def __init__(self, dim: int = 32) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive.")
        self.dim = dim

    def encode(self, text: str) -> list[float]:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", (text or "").lower())
        if not tokens:
            return [0.0] * self.dim

        values = [0.0] * self.dim
        for unit in self._iter_units(tokens):
            digest = hashlib.blake2b(unit.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            values[bucket] += sign
        return _l2_normalize(values)

    def _iter_units(self, tokens: list[str]) -> list[str]:
        units = list(tokens)
        units.extend(f"{left}::{right}" for left, right in zip(tokens, tokens[1:]))
        return units


class QwenTextEncoder:
    def __init__(
        self,
        *,
        model_name: str,
        instruction: str,
        output_dim: int,
        cache_dir: Path,
        vllm_config: VLLMConfig,
    ) -> None:
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        self.model_name = model_name
        self.instruction = instruction.strip()
        self.dim = output_dim
        self.cache_dir = cache_dir
        self.vllm_config = vllm_config
        self._llm: Any | None = None
        self._memory_cache: dict[str, list[float]] = {}

    def encode(self, text: str) -> list[float]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return [0.0] * self.dim
        cache_key = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        cached = self._memory_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        cache_path = self._cache_path(cache_key)
        if cache_path.exists():
            try:
                cached_values = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached_values, list) and len(cached_values) == self.dim:
                    vector = [float(value) for value in cached_values]
                    self._memory_cache[cache_key] = vector
                    return list(vector)
            except Exception:
                logging.warning("Failed to read embedding cache entry: %s", cache_path)

        llm = self._ensure_llm()
        prompt = self._truncate_prompt_to_max_model_len(
            llm,
            self._format_query(normalized_text),
        )
        from vllm import PoolingParams

        try:
            outputs = llm.embed(
                [prompt],
                use_tqdm=False,
                pooling_params=PoolingParams(dimensions=self.dim),
            )
        except ValueError as exc:
            retry_prompt = self._retry_prompt_after_length_error(llm, prompt, exc)
            if retry_prompt is None:
                raise
            outputs = llm.embed(
                [retry_prompt],
                use_tqdm=False,
                pooling_params=PoolingParams(dimensions=self.dim),
            )
        if not outputs:
            raise RuntimeError(f"Embedding model returned no outputs for model={self.model_name}.")
        raw_embedding = getattr(getattr(outputs[0], "outputs", None), "embedding", None)
        if raw_embedding is None:
            raise RuntimeError(f"Embedding model returned an unexpected payload for model={self.model_name}.")
        vector_values = [float(value) for value in raw_embedding]
        if len(vector_values) != self.dim:
            raise ValueError(
                f"Expected embedding_dim={self.dim} from model={self.model_name}, got {len(vector_values)}."
            )

        vector = _l2_normalize(vector_values)
        self._memory_cache[cache_key] = vector
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(vector), encoding="utf-8")
        return list(vector)

    def _format_query(self, query: str) -> str:
        if not self.instruction:
            return query
        return f"Instruct: {self.instruction}\nQuery:{query}"

    def _truncate_prompt_to_max_model_len(
        self,
        llm,
        prompt: str,
        *,
        max_prompt_tokens: int | None = None,
    ) -> str:
        raw_token_limit = max_prompt_tokens or int(getattr(self.vllm_config, "max_model_len", 0) or 0)
        token_limit = raw_token_limit if max_prompt_tokens is not None else max(1, raw_token_limit - 1)
        if token_limit <= 0:
            return prompt
        tokenizer = self._get_tokenizer(llm)
        if tokenizer is None:
            return prompt
        token_ids = self._encode_text(tokenizer, prompt)
        if token_ids is None or len(token_ids) <= token_limit:
            return prompt
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            return " ".join(str(prompt or "").split()[:token_limit])
        try:
            truncated_prompt = decode(token_ids[:token_limit], skip_special_tokens=True)
        except TypeError:
            truncated_prompt = decode(token_ids[:token_limit])
        logging.warning(
            "Truncating Qwen embedding prompt from %s tokens to %s tokens for model=%s.",
            len(token_ids),
            token_limit,
            self.model_name,
        )
        return str(truncated_prompt or "").strip()

    def _retry_prompt_after_length_error(self, llm, prompt: str, exc: Exception) -> str | None:
        max_prompt_tokens = self._extract_max_prompt_tokens(str(exc or ""))
        if max_prompt_tokens is None:
            return None
        retried_prompt = self._truncate_prompt_to_max_model_len(
            llm,
            prompt,
            max_prompt_tokens=max(1, max_prompt_tokens),
        )
        if retried_prompt == prompt:
            return None
        return retried_prompt

    def _get_tokenizer(self, llm):
        get_tokenizer = getattr(llm, "get_tokenizer", None)
        if callable(get_tokenizer):
            try:
                return get_tokenizer()
            except Exception:
                logging.warning("Failed to fetch tokenizer from embedding model=%s.", self.model_name)
        return None

    def _encode_text(self, tokenizer, text: str) -> list[Any] | None:
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            return None
        try:
            return list(encode(text, add_special_tokens=False))
        except TypeError:
            try:
                return list(encode(text))
            except Exception:
                return None
        except Exception:
            return None

    def _extract_max_prompt_tokens(self, exc_text: str) -> int | None:
        match = re.search(r"maximum model length of (\d+)", exc_text)
        if match:
            return int(match.group(1))
        return None

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _ensure_llm(self):
        if self._llm is None:
            self._apply_vllm_environment()
            from vllm import LLM

            self._llm = LLM(
                model=self.model_name,
                runner="pooling",
                tensor_parallel_size=self._resolve_tensor_parallel_size(),
                dtype=self.vllm_config.dtype,
                trust_remote_code=self.vllm_config.trust_remote_code,
                gpu_memory_utilization=self.vllm_config.gpu_memory_utilization,
                max_model_len=self.vllm_config.max_model_len,
                hf_overrides={
                    "is_matryoshka": True,
                    "matryoshka_dimensions": [self.dim],
                },
            )
        return self._llm

    def _apply_vllm_environment(self) -> None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = self.vllm_config.worker_multiproc_method
        os.environ["CUDA_VISIBLE_DEVICES"] = self.vllm_config.cuda_visible_devices
        os.environ["NCCL_P2P_DISABLE"] = self.vllm_config.nccl_p2p_disable

    def _resolve_tensor_parallel_size(self) -> int:
        visible_devices = str(self.vllm_config.cuda_visible_devices or "").strip()
        if not visible_devices:
            raise ValueError("cuda_visible_devices must expose at least one GPU for Qwen embedding.")
        visible_gpu_count = len([device for device in visible_devices.split(",") if device.strip()])
        if visible_gpu_count <= 0:
            raise ValueError("cuda_visible_devices must expose at least one GPU for Qwen embedding.")
        return visible_gpu_count


def build_text_encoder(
    *,
    embedding_config: EmbeddingConfig,
    vllm_config: VLLMConfig,
):
    backend = str(embedding_config.backend or "hash").strip().lower()
    if backend == "none":
        return NoTextEncoder()
    if backend == "hash":
        return FrozenHashTextEncoder(dim=embedding_config.hash_dim)
    if backend == "qwen":
        qwen_vllm_config = dataclass_replace(
            vllm_config,
            cuda_visible_devices=embedding_config.qwen_cuda_visible_devices,
            gpu_memory_utilization=embedding_config.qwen_gpu_memory_utilization,
        )
        return QwenTextEncoder(
            model_name=embedding_config.qwen_model_name,
            instruction=embedding_config.qwen_task_instruction,
            output_dim=embedding_config.qwen_output_dim,
            cache_dir=embedding_config.cache_dir,
            vllm_config=qwen_vllm_config,
        )
    raise ValueError(f"Unsupported embedding backend: {embedding_config.backend}")
