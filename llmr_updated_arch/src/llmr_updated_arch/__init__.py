"""Public API for the component-oriented llmr architecture."""

from __future__ import annotations

from importlib import import_module
from typing_extensions import Any, Dict, Tuple

__all__ = [
    "LLMBackend",
    "ActionResolutionPipeline",
    "ResolutionContext",
    "ActionResolutionResult",
    "SemanticGenerator",
    "GroundingResolver",
    "ProjectionBuilder",
    "instance_from_instruction",
    "plan_from_instruction",
    "sequential_plan_from_instruction",
    "instance_from_match",
    "plan_from_match",
]

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "LLMBackend": ("llmr_updated_arch.core.backend", "LLMBackend"),
    "ActionResolutionPipeline": (
        "llmr_updated_arch.core.pipeline",
        "ActionResolutionPipeline",
    ),
    "ResolutionContext": ("llmr_updated_arch.core.context", "ResolutionContext"),
    "ActionResolutionResult": (
        "llmr_updated_arch.core.result",
        "ActionResolutionResult",
    ),
    "SemanticGenerator": ("llmr_updated_arch.core.contracts", "SemanticGenerator"),
    "GroundingResolver": ("llmr_updated_arch.core.contracts", "GroundingResolver"),
    "ProjectionBuilder": ("llmr_updated_arch.core.contracts", "ProjectionBuilder"),
    "instance_from_instruction": (
        "llmr_updated_arch.entrypoints.instruction",
        "instance_from_instruction",
    ),
    "plan_from_instruction": (
        "llmr_updated_arch.entrypoints.instruction",
        "plan_from_instruction",
    ),
    "sequential_plan_from_instruction": (
        "llmr_updated_arch.entrypoints.instruction",
        "sequential_plan_from_instruction",
    ),
    "instance_from_match": ("llmr_updated_arch.entrypoints.match", "instance_from_match"),
    "plan_from_match": ("llmr_updated_arch.entrypoints.pycram", "plan_from_match"),
}


def __getattr__(name: str) -> Any:
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
