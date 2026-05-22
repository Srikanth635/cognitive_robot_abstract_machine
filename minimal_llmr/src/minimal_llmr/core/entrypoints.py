"""User-facing entry points for minimal_llmr.

  instance_from_instruction()  — NL instruction → resolved action instance
  instance_from_match()        — underspecified Match → resolved action instance
  underspecified_match_for()   — convenience: build a fully-free Match for an action class
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TYPE_CHECKING  # Optional kept for world_context_provider

from minimal_llmr.backend import LLMBackend
from minimal_llmr.bridge.template import underspecified_match
from minimal_llmr.core.errors import ActionClassificationFailed
from minimal_llmr.inference.classifier import build_action_registry, infer_action_class

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def instance_from_instruction(
    instruction: str,
    llm: "BaseChatModel",
    action_base_class: type,
    strict_required: bool = True,
) -> Any:
    """Classify *instruction*, build an underspecified Match, and resolve it.

    :param instruction:        Natural-language robot instruction.
    :param llm:                LangChain-compatible chat model.
    :param action_base_class:  Base class of the action hierarchy (e.g. ActionDesignator).
                               All concrete dataclass subclasses are discovered automatically
                               via KRROOD's recursive_subclasses().
    :param strict_required:    Raise on unresolved required parameters (default True).
    :raises ActionClassificationFailed: If the instruction cannot be mapped to an action class.
    """
    action_registry = build_action_registry(action_base_class)
    classification = infer_action_class(instruction, llm=llm, action_registry=action_registry)
    action_cls = action_registry.get(classification.action_type) if classification else None
    if action_cls is None:
        raise ActionClassificationFailed(instruction=instruction)

    match = underspecified_match(action_cls)
    backend = LLMBackend(llm=llm, instruction=instruction, strict_required=strict_required)
    return next(iter(backend.evaluate(match)))


def instance_from_match(
    match: Any,
    llm: "BaseChatModel",
    instruction: Optional[str] = None,
    world_context_provider: Optional[Callable[[], str]] = None,
    strict_required: bool = False,
) -> Any:
    """Resolve an underspecified KRROOD Match and return the concrete action instance.

    :param match:                   KRROOD Match expression with free parameters.
    :param llm:                     LangChain-compatible chat model.
    :param instruction:             Optional NL instruction to ground parameter descriptions.
    :param world_context_provider:  Optional callable returning a world context string.
    :param strict_required:         Raise on unresolved required parameters (default False).
    """
    backend = LLMBackend(
        llm=llm,
        instruction=instruction,
        world_context_provider=world_context_provider,
        strict_required=strict_required,
    )
    return next(iter(backend.evaluate(match)))


def underspecified_match_for(action_cls: type) -> Any:
    """Convenience wrapper: build a fully-free Match for *action_cls*."""
    return underspecified_match(action_cls)
