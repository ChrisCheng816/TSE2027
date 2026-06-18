from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path


PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "prompts"
PROMPT_MODULE_PREFIX = "codegen_agent.prompts"


class _DefaultPromptValueDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return ""


def prompt_template_path(relative_path: str) -> Path:
    return PROMPTS_ROOT / relative_path


def prompt_template_module(relative_path: str) -> str:
    module_path = Path(relative_path)
    if module_path.suffix != ".py":
        raise ValueError(f"Prompt template must be a Python module: {relative_path}")
    return ".".join((PROMPT_MODULE_PREFIX, *module_path.with_suffix("").parts))


@lru_cache(maxsize=None)
def load_prompt_template(relative_path: str) -> str:
    module = importlib.import_module(prompt_template_module(relative_path))
    template = getattr(module, "PROMPT")
    if not isinstance(template, str):
        raise TypeError(f"PROMPT must be a string in {relative_path}")
    return template.strip()


def render_prompt_template(relative_path: str, **values: str) -> str:
    resolved_values = {key: str(value) for key, value in values.items()}
    template = load_prompt_template(relative_path)
    return template.format_map(_DefaultPromptValueDict(resolved_values)).strip()
