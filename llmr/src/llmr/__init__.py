"""llmr — LLM-powered GenerativeBackend for KRROOD.

The package root exposes the high-level public API lazily so subpackages such as
``llmr.hypotheses`` can be imported without pulling in optional backend/runtime
dependencies during module import.
"""

from __future__ import annotations

from importlib import import_module
from typing_extensions import Any, Dict, Tuple

__all__ = [
    "LLMBackend",
    "plan_from_instruction",
    "sequential_plan_from_instruction",
    "plan_from_match",
    "instance_from_match",
    "LLMActionClassificationFailed",
    "LLMActionRegistryEmpty",
    "LLMProviderNotSupported",
    "LLMSlotFillingFailed",
    "LLMUnresolvedRequiredFields",
]

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "LLMBackend": ("llmr.backend", "LLMBackend"),
    "plan_from_instruction": ("llmr.factory", "plan_from_instruction"),
    "sequential_plan_from_instruction": ("llmr.factory", "sequential_plan_from_instruction"),
    "plan_from_match": ("llmr.factory", "plan_from_match"),
    "instance_from_match": ("llmr.factory", "instance_from_match"),
    "LLMActionClassificationFailed": (
        "llmr.exceptions",
        "LLMActionClassificationFailed",
    ),
    "LLMActionRegistryEmpty": ("llmr.exceptions", "LLMActionRegistryEmpty"),
    "LLMProviderNotSupported": ("llmr.exceptions", "LLMProviderNotSupported"),
    "LLMSlotFillingFailed": ("llmr.exceptions", "LLMSlotFillingFailed"),
    "LLMUnresolvedRequiredFields": (
        "llmr.exceptions",
        "LLMUnresolvedRequiredFields",
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve public root exports lazily to avoid heavy import side effects."""

    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
