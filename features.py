from __future__ import annotations

import re

from .config import DEFAULT_EXPERIMENT_CONFIG, ExperimentConfig
from .dataset_families import is_bigcodebench_family_dataset
from .embeddings import FrozenHashTextEncoder, build_text_encoder
from .types import CodeGenTask, FeatureExtractionResult


def visible_task_prompt(task: CodeGenTask) -> str:
    instruct_prompt = (task.instruct_prompt or "").strip()
    if is_bigcodebench_family_dataset(task.dataset_name) and instruct_prompt:
        return instruct_prompt
    if instruct_prompt and task.prompt_source == "instruct_prompt":
        return instruct_prompt
    return (task.prompt or "").strip()


def visible_task_context_text(task: CodeGenTask) -> str:
    if is_bigcodebench_family_dataset(task.dataset_name):
        return visible_task_prompt(task)
    parts = [visible_task_prompt(task)]
    if task.code_prompt:
        parts.append(task.code_prompt.strip())
    if task.entry_point:
        parts.append(f"entry_point {task.entry_point}")
    return "\n".join(part for part in parts if part)


class BigCodeBenchFeatureExtractor:
    STRUCTURED_FEATURE_NAMES = [
        "bias",
        "prompt_char_k_norm",
        "code_prompt_char_k_norm",
        "example_count_norm",
        "starter_import_count_norm",
        "starter_def_count_norm",
        "spec_term_count_norm",
        "formatting_term_count_norm",
        "boundary_term_count_norm",
        "exception_term_count_norm",
        "io_term_count_norm",
        "api_surface_term_count_norm",
        "algorithm_term_count_norm",
        "state_term_count_norm",
        "data_structure_term_count_norm",
        "complexity_term_count_norm",
        "return_shape_hint_flag",
    ]
    REPAIR_FEATURE_NAMES = [
        "draft_completion_tokens_norm",
        "draft_output_chars_k_norm",
        "failure_exit_code_nonzero_flag",
        "failure_traceback_flag",
        "failure_assertion_flag",
        "failure_timeout_flag",
        "failure_syntax_flag",
        "failure_runtime_flag",
        "failure_import_flag",
        "failure_name_flag",
        "failure_type_flag",
        "failure_value_flag",
        "failure_key_flag",
        "failure_index_flag",
        "failure_attribute_flag",
        "failure_stdout_nonempty_flag",
        "failure_stderr_chars_k_norm",
    ]
    _EXAMPLE_TOKENS = ("example", "examples", "sample input", "sample output", "input:", "output:")
    _SPEC_TOKENS = (
        "must",
        "required",
        "exactly",
        "strictly",
        "preserve",
        "return",
        "output",
        "follow",
        "same",
        "public interface",
    )
    _FORMATTING_TOKENS = (
        "sorted",
        "order",
        "ordered",
        "ascending",
        "descending",
        "format",
        "column",
        "title",
        "label",
        "key",
    )
    _BOUNDARY_TOKENS = ("edge case", "empty", "zero", "none", "duplicate", "boundary", "corner", "missing", "null")
    _EXCEPTION_TOKENS = (
        "raise",
        "error",
        "exception",
        "invalid",
        "valueerror",
        "typeerror",
        "keyerror",
        "indexerror",
    )
    _IO_TOKENS = (
        "file",
        "path",
        "directory",
        "read",
        "write",
        "open",
        "load",
        "save",
        "json",
        "csv",
        "yaml",
        "xml",
        "stdin",
        "stdout",
    )
    _API_SURFACE_TOKENS = (
        "dataframe",
        "pandas",
        "numpy",
        "array",
        "plot",
        "matplotlib",
        "figure",
        "axis",
        "subprocess",
        "requests",
        "pathlib",
        "os.path",
        "datetime",
        "decimal",
        "regex",
    )
    _ALGORITHM_TOKENS = (
        "algorithm",
        "dynamic programming",
        "dp",
        "greedy",
        "recursive",
        "recursion",
        "backtracking",
        "binary search",
        "sliding window",
        "two pointers",
        "memoization",
        "bfs",
        "dfs",
        "topological",
        "simulate",
    )
    _STATE_TOKENS = (
        "state",
        "in place",
        "modify",
        "update",
        "mutate",
        "cache",
        "memo",
        "table",
        "branch",
        "condition",
        "loop",
        "iterate",
    )
    _DATA_STRUCTURE_TOKENS = (
        "heap",
        "queue",
        "stack",
        "hash map",
        "dictionary",
        "set",
        "tree",
        "graph",
        "matrix",
        "priority queue",
        "linked list",
        "trie",
        "deque",
    )
    _COMPLEXITY_TOKENS = (
        "time complexity",
        "space complexity",
        "complexity",
        "efficient",
        "linear time",
        "logarithmic",
        "quadratic",
        "o(",
        "optimize",
    )
    _RETURN_SHAPE_TOKENS = ("tuple", "dict", "dictionary", "list", "json", "dataframe", "array", "set", "iterator")
    _IMPORT_RE = re.compile(r"^\s*(?:from\s+\S+\s+import|import\s+\S+)", re.MULTILINE)
    _DEF_RE = re.compile(r"^\s*(?:async\s+def|def)\b", re.MULTILINE)

    def __init__(
        self,
        *,
        embedding_dim: int = 32,
        embedding_backend: str = "hash",
        include_structured_features: bool = True,
        text_encoder=None,
    ) -> None:
        self.embedding_backend = str(embedding_backend or "hash").strip().lower()
        self.include_structured_features = bool(include_structured_features)
        if text_encoder is not None:
            self.text_encoder = text_encoder
            self.embedding_dim = int(getattr(text_encoder, "dim", embedding_dim))
        elif self.embedding_backend == "none":
            self.text_encoder = None
            self.embedding_dim = 0
        else:
            self.embedding_dim = embedding_dim
            self.text_encoder = FrozenHashTextEncoder(dim=embedding_dim)
        structured_feature_names = list(self.STRUCTURED_FEATURE_NAMES) if self.include_structured_features else []
        self.FEATURE_NAMES = structured_feature_names + [f"text_embedding_{index:03d}" for index in range(self.embedding_dim)]
        self.DECISION_FEATURE_NAMES = (
            list(self.FEATURE_NAMES)
            + list(self.REPAIR_FEATURE_NAMES)
            + [f"failure_message_embedding_{index:03d}" for index in range(self.embedding_dim)]
        )
        self.UPGRADE_FEATURE_NAMES = list(self.FEATURE_NAMES)

    def extract(self, task: CodeGenTask) -> FeatureExtractionResult:
        prompt = visible_task_prompt(task)
        prompt_lower = prompt.lower()
        code_prompt = (task.code_prompt or "").strip()
        structured_values = [
            1.0,
            self._clip_ratio(len(prompt), 4000.0),
            self._clip_ratio(len(code_prompt), 800.0),
            self._clip_ratio(self._keyword_presence_count(prompt_lower, self._EXAMPLE_TOKENS), 5.0),
            self._clip_ratio(len(self._IMPORT_RE.findall(code_prompt)), 6.0),
            self._clip_ratio(len(self._DEF_RE.findall(code_prompt)), 4.0),
            self._clip_ratio(self._keyword_presence_count(prompt_lower, self._SPEC_TOKENS), 10.0),
            self._clip_ratio(self._keyword_presence_count(prompt_lower, self._FORMATTING_TOKENS), 8.0),
            self._clip_ratio(self._keyword_presence_count(prompt_lower, self._BOUNDARY_TOKENS), 8.0),
            self._clip_ratio(self._keyword_presence_count(prompt_lower, self._EXCEPTION_TOKENS), 8.0),
            self._clip_ratio(self._keyword_presence_count(prompt_lower, self._IO_TOKENS), 8.0),
            self._clip_ratio(self._keyword_presence_count(prompt_lower, self._API_SURFACE_TOKENS), 8.0),
            self._clip_ratio(self._keyword_presence_count(prompt_lower, self._ALGORITHM_TOKENS), 10.0),
            self._clip_ratio(self._keyword_presence_count(prompt_lower, self._STATE_TOKENS), 8.0),
            self._clip_ratio(self._keyword_presence_count(prompt_lower, self._DATA_STRUCTURE_TOKENS), 8.0),
            self._clip_ratio(self._keyword_presence_count(prompt_lower, self._COMPLEXITY_TOKENS), 8.0),
            self._flag(any(token in prompt_lower for token in self._RETURN_SHAPE_TOKENS)),
        ]
        embedding_values = [] if self.text_encoder is None else self.text_encoder.encode(self._embedding_text(task))
        values = (structured_values if self.include_structured_features else []) + embedding_values
        return FeatureExtractionResult(
            names=list(self.FEATURE_NAMES),
            values=values,
            signals={
                "prompt_chars": float(len(prompt)),
                "code_prompt_chars": float(len(code_prompt)),
                "spec_term_count": float(self._keyword_presence_count(prompt_lower, self._SPEC_TOKENS)),
                "io_term_count": float(self._keyword_presence_count(prompt_lower, self._IO_TOKENS)),
                "api_surface_term_count": float(self._keyword_presence_count(prompt_lower, self._API_SURFACE_TOKENS)),
                "algorithm_term_count": float(self._keyword_presence_count(prompt_lower, self._ALGORITHM_TOKENS)),
                "state_term_count": float(self._keyword_presence_count(prompt_lower, self._STATE_TOKENS)),
                "data_structure_term_count": float(self._keyword_presence_count(prompt_lower, self._DATA_STRUCTURE_TOKENS)),
                "complexity_term_count": float(self._keyword_presence_count(prompt_lower, self._COMPLEXITY_TOKENS)),
                "embedding_dim": float(self.embedding_dim),
            },
        )

    def extract_repair_context(
        self,
        task: CodeGenTask,
        *,
        draft_output: str,
        evaluation: dict[str, object] | None,
        completion_tokens: int,
    ) -> FeatureExtractionResult:
        base = self.extract(task)
        evaluation = evaluation or {}
        stdout = str(evaluation.get("stdout") or "")
        stderr = str(evaluation.get("stderr") or "")
        combined_error_text = "\n".join(part for part in (stdout, stderr) if part).lower()
        repair_values = [
            self._clip_ratio(completion_tokens, 1024.0),
            self._clip_ratio(len(draft_output or ""), 4000.0),
            self._flag(bool((evaluation.get("exit_code") or 0) not in (0, None))),
            self._flag("traceback" in combined_error_text),
            self._flag("assertionerror" in combined_error_text or "expected " in combined_error_text),
            self._flag("timeoutexpired" in combined_error_text or "timeout" in combined_error_text),
            self._flag("syntaxerror" in combined_error_text or "indentationerror" in combined_error_text),
            self._flag(bool(stderr.strip()) and "traceback" in combined_error_text),
            self._flag("importerror" in combined_error_text or "modulenotfounderror" in combined_error_text),
            self._flag("nameerror" in combined_error_text),
            self._flag("typeerror" in combined_error_text),
            self._flag("valueerror" in combined_error_text),
            self._flag("keyerror" in combined_error_text),
            self._flag("indexerror" in combined_error_text),
            self._flag("attributeerror" in combined_error_text),
            self._flag(bool(stdout.strip())),
            self._clip_ratio(len(stderr), 4000.0),
        ]
        message_embedding = [] if self.text_encoder is None else self.text_encoder.encode(combined_error_text)
        values = list(base.values) + repair_values + message_embedding
        return FeatureExtractionResult(
            names=list(self.DECISION_FEATURE_NAMES),
            values=values,
            signals={
                **base.signals,
                "draft_completion_tokens": float(completion_tokens),
                "draft_output_chars": float(len(draft_output or "")),
                "stderr_chars": float(len(stderr)),
            },
        )

    def _embedding_text(self, task: CodeGenTask) -> str:
        return visible_task_context_text(task)

    def _keyword_presence_count(self, text: str, tokens: tuple[str, ...]) -> int:
        return sum(1 for token in tokens if token in text)

    def _clip_ratio(self, numerator: float, denominator: float) -> float:
        if denominator <= 0:
            raise ValueError("denominator must be positive.")
        return max(0.0, min(float(numerator) / denominator, 1.0))

    def _flag(self, value: bool) -> float:
        return 1.0 if value else 0.0


def build_feature_extractor(
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
) -> BigCodeBenchFeatureExtractor:
    encoder = build_text_encoder(
        embedding_config=config.embedding,
        vllm_config=config.vllm,
    )
    return BigCodeBenchFeatureExtractor(
        embedding_dim=int(getattr(encoder, "dim", 0)),
        embedding_backend=config.embedding.backend,
        include_structured_features=config.embedding.include_structured_features,
        text_encoder=encoder,
    )
