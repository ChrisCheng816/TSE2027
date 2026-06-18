from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PipelineArm:
    arm_id: str
    paper_name: str
    stage: str
    hypothesis: str
    max_major_calls: int
    phase_token_budgets: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    phase_stop_sequences: tuple[tuple[str, tuple[str, ...]], ...] = field(default_factory=tuple)
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def max_new_tokens_for_phase(self, phase: str, *, default: int | None = None) -> int:
        budget_map = dict(self.phase_token_budgets)
        if phase in budget_map:
            return budget_map[phase]
        if default is not None:
            return default
        raise KeyError(f"Unsupported phase={phase} for arm_id={self.arm_id}")

    def total_token_budget(self) -> int:
        return sum(limit for _, limit in self.phase_token_budgets)

    def stop_sequences_for_phase(self, phase: str) -> tuple[str, ...]:
        stop_map = dict(self.phase_stop_sequences)
        return stop_map.get(phase, ())


DEFAULT_PIPELINE_ARMS: tuple[PipelineArm, ...] = (
    PipelineArm(
        arm_id="A0",
        paper_name="A0",
        stage="single",
        hypothesis="One-layer direct generation baseline for tasks with low specification complexity and low solution complexity.",
        max_major_calls=1,
        phase_token_budgets=(("final_code", 4096),),
        aliases=("a0", "arm0", "arm-0", "direct", "baseline"),
    ),
    PipelineArm(
        arm_id="A1",
        paper_name="A1",
        stage="single",
        hypothesis="Single-shot self-planning / SCoT-style workflow that first locks a concise implementation plan and program structure, then writes the final code while evaluating code only.",
        max_major_calls=1,
        phase_token_budgets=(("final_code", 4096),),
        aliases=(
            "a1",
            "arm1",
            "arm-1",
            "self-planning",
            "self_planning",
            "scot",
            "plan",
            "plan-structured",
            "plan_structured",
        ),
    ),
    PipelineArm(
        arm_id="A2",
        paper_name="A2",
        stage="single",
        hypothesis="Two-phase generate-and-repair workflow that writes code directly, evaluates it with self-tests, and performs at most one repair round with failure logs when needed.",
        max_major_calls=2,
        phase_token_budgets=(("draft_bundle", 4096), ("repair_code", 4096)),
        aliases=(
            "a2",
            "arm2",
            "arm-2",
            "repair",
            "repair-once",
            "test-guided",
            "test_guided",
            "two-stage",
        ),
    ),
)

DEFAULT_ARM_SET = tuple(arm.arm_id for arm in DEFAULT_PIPELINE_ARMS)


def get_arm_choices(arm_library: tuple[PipelineArm, ...] = DEFAULT_PIPELINE_ARMS) -> tuple[str, ...]:
    return tuple(arm.arm_id for arm in arm_library)


def normalize_arm_ids(
    arm_ids: list[str] | tuple[str, ...] | None = None,
    *,
    arm_library: tuple[PipelineArm, ...] = DEFAULT_PIPELINE_ARMS,
) -> tuple[str, ...]:
    if arm_ids is None:
        return tuple(arm.arm_id for arm in arm_library)

    canonical_by_id = {arm.arm_id.upper(): arm.arm_id for arm in arm_library}
    canonical_by_alias = {
        alias.strip().upper(): arm.arm_id
        for arm in arm_library
        for alias in arm.aliases
        if alias.strip()
    }
    resolved: list[str] = []

    for raw_value in arm_ids:
        for token in str(raw_value).split(","):
            candidate = token.strip()
            if not candidate:
                continue
            normalized = candidate.upper()
            if normalized in canonical_by_id:
                arm_id = canonical_by_id[normalized]
            elif normalized in canonical_by_alias:
                arm_id = canonical_by_alias[normalized]
            else:
                available_ids = ", ".join(get_arm_choices(arm_library))
                raise ValueError(f"Unknown arm selector '{candidate}'. Use arm ids like {available_ids}.")
            if arm_id not in resolved:
                resolved.append(arm_id)
    return tuple(resolved)


def get_partition_arms(
    arm_ids: list[str] | tuple[str, ...] | None = None,
    *,
    arm_library: tuple[PipelineArm, ...] = DEFAULT_PIPELINE_ARMS,
) -> tuple[PipelineArm, ...]:
    if arm_ids is None:
        return arm_library
    arm_ids = normalize_arm_ids(arm_ids, arm_library=arm_library)
    arm_map = {arm.arm_id: arm for arm in arm_library}
    return tuple(arm_map[arm_id] for arm_id in arm_ids)


def get_base_arms(
    arm_library: tuple[PipelineArm, ...] = DEFAULT_PIPELINE_ARMS,
) -> tuple[PipelineArm, ...]:
    return tuple(arm for arm in arm_library if arm.stage in {"base", "single"})


def get_upgrade_arms(
    arm_library: tuple[PipelineArm, ...] = DEFAULT_PIPELINE_ARMS,
) -> tuple[PipelineArm, ...]:
    return tuple(arm for arm in arm_library if arm.stage == "upgrade")
