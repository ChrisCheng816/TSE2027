from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CodeGenTask:
    dataset_name: str
    task_id: str
    prompt: str
    prompt_source: str
    task_kind: str | None = None
    complete_prompt: str | None = None
    instruct_prompt: str | None = None
    canonical_solution: str | None = None
    code_prompt: str | None = None
    test: str | None = None
    entry_point: str | None = None
    test_spec: dict[str, Any] | None = None
    doc_struct: dict[str, Any] | str | None = None
    libs: list[str] | str | None = None
    raw_record: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "task_id": self.task_id,
            "prompt": self.prompt,
            "prompt_source": self.prompt_source,
            "task_kind": self.task_kind,
            "complete_prompt": self.complete_prompt,
            "instruct_prompt": self.instruct_prompt,
            "canonical_solution": self.canonical_solution,
            "code_prompt": self.code_prompt,
            "test": self.test,
            "entry_point": self.entry_point,
            "test_spec": self.test_spec,
            "doc_struct": self.doc_struct,
            "libs": self.libs,
            "raw_record": self.raw_record,
            "metadata": self.metadata,
        }
BigCodeBenchTask = CodeGenTask


@dataclass(frozen=True)
class PromptMessage:
    role: str
    content: str
    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
        }


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str
    arm_id: str
    raw_prompt: str
    phase: str = "final_code"
    messages: tuple[PromptMessage, ...] = ()
    model_name: str = ""
    dataset_name: str = ""
    task_kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "user": self.user,
            "arm_id": self.arm_id,
            "raw_prompt": self.raw_prompt,
            "phase": self.phase,
            "messages": [message.to_dict() for message in self.messages],
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "task_kind": self.task_kind,
        }


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float | None = None
    top_p: float | None = None
    max_new_tokens: int = 1024
    stop: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelOutput:
    raw_output: str
    cleaned_output: str
    completion_tokens: int
    prompt_tokens: int = 0


@dataclass(frozen=True)
class FeatureExtractionResult:
    names: list[str]
    values: list[float]
    signals: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BanditDecision:
    arm_id: str
    scores: dict[str, float]
    expected_rewards: dict[str, float]
    uncertainty_bonuses: dict[str, float]
    pull_counts: dict[str, int]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRecord:
    task_id: str
    arm_id: str
    arm_name: str
    model_name: str
    model_path: str
    dataset_name: str
    prompt: PromptBundle
    task: BigCodeBenchTask
    feature_names: list[str]
    feature_vector: list[float]
    predictions: list[dict[str, Any]]
    num_samples: int = 1
    pass_k: int = 1
    num_correct: int = 0
    reference_completion_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    accounted_completion_tokens: int = 0
    total_tokens: int = 0
    generation_duration_sec: float = 0.0
    generation_durations_by_phase_sec: dict[str, float] = field(default_factory=dict)
    generation_inference_durations_sec: list[float] = field(default_factory=list)
    completion_token_budget: int | None = None
    output_length: int = 0
    pass_at_k: float | None = None
    reward: float | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    selection_reason: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "arm_id": self.arm_id,
            "arm_name": self.arm_name,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "dataset_name": self.dataset_name,
            "prompt": self.prompt.to_dict(),
            "feature_names": self.feature_names,
            "feature_vector": self.feature_vector,
            "predictions": self.predictions,
            "num_samples": self.num_samples,
            "pass_k": self.pass_k,
            "num_correct": self.num_correct,
            "reference_completion_tokens": self.reference_completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "accounted_completion_tokens": self.accounted_completion_tokens,
            "total_tokens": self.total_tokens,
            "generation_duration_sec": self.generation_duration_sec,
            "generation_durations_by_phase_sec": self.generation_durations_by_phase_sec,
            "generation_inference_durations_sec": self.generation_inference_durations_sec,
            "completion_token_budget": self.completion_token_budget,
            "output_length": self.output_length,
            "pass_at_k": self.pass_at_k,
            "reward": self.reward,
            "reward_components": self.reward_components,
            "selection_reason": self.selection_reason,
            "metadata": self.metadata,
        }
