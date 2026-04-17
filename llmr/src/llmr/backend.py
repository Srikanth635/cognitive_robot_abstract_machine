"""LLMBackend — GenerativeBackend implementation that uses an LLM to fill underspecified Match slots.

World context is derived from SymbolGraph.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing_extensions import Any, Callable, Dict, Iterable, Optional, Type

logger = logging.getLogger(__name__)

from krrood.entity_query_language.backends import GenerativeBackend
from krrood.entity_query_language.query.match import Match
from krrood.symbol_graph.symbol_graph import Symbol

from krrood.entity_query_language.utils import T

from llmr.exceptions import LLMSlotFillingFailed, LLMUnresolvedRequiredFields
from llmr.match_inspection import (
    match_bindings,
    unresolved_required_fields,
)
from llmr.slot_resolution import resolve_binding_value

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


# ── Typed sentinel ─────────────────────────────────────────────────────────────

class _Unresolved:
    """Singleton sentinel returned when a slot cannot be resolved.

    Using a dedicated class (rather than ``object()``) lets type checkers
    distinguish unresolved returns from legitimate ``Any`` values, and gives
    a descriptive ``repr`` in log output.
    """

    _instance: "_Unresolved | None" = None

    def __new__(cls) -> "_Unresolved":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<UNRESOLVED>"


_UNRESOLVED = _Unresolved()


# ── LLMBackend ─────────────────────────────────────────────────────────────────

@dataclass
class LLMBackend(GenerativeBackend):
    """A GenerativeBackend that uses an LLM to fill underspecified Match slots."""

    llm: "BaseChatModel"
    """LangChain BaseChatModel — the reasoning engine for slot filling and action classification."""

    groundable_type: Type[Symbol] = field(default=Symbol)
    """
    Symbol subclass scoping entity grounding and world serialisation.
    Defaults to ``Symbol`` (all instances); pass ``Body`` for physical-body-only scope.
    """

    instruction: Optional[str] = field(kw_only=True, default=None)
    """
    NL instruction included in the slot-filler prompt for semantic grounding
    (e.g. ``"the milk from the table"``).  Omit when the action type and fixed
    slots already carry the intent.
    """

    world_context_provider: Optional[Callable[[], str]] = field(kw_only=True, default=None)
    """
    Callable returning a world-context string.  Replaces the default SymbolGraph
    serialisation when provided.  Useful for injecting a custom or pre-cached
    world description.
    """

    strict_required: bool = field(kw_only=True, default=False)
    """
    When ``True``, raise :class:`~llmr.exceptions.LLMUnresolvedRequiredFields`
    if required action fields remain unresolved instead of constructing a partially
    resolved action.
    """

    # ── Core interface ─────────────────────────────────────────────────────────

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        """Resolve all free slots in *expression* and yield a fully-constructed action instance."""

        # ── 1. Parse free / fixed slots from the Match variable graph ──────────
        bindings = match_bindings(expression, unresolved=_UNRESOLVED)
        free_bindings = [binding for binding in bindings if binding.is_free]
        fixed_slots: Dict[str, Any] = {
            binding.prompt_name: binding.value
            for binding in bindings
            if not binding.is_free
        }

        if not free_bindings:
            expression._update_kwargs_from_literal_values()
            yield expression.construct_instance()
            return

        # ── 2. World context ───────────────────────────────────────────────────
        world_context = self._get_world_context()

        # ── 3. Run the slot filler (LLM call with dynamic prompt) ─────────────
        # krrood already resolved each field's type via get_field_type_endpoint()
        # and stored it in attr_match.assigned_variable._type_ — we use those
        # types directly below instead of re-running full action-class introspection.
        llm_free_slot_names = [
            binding.prompt_name
            for binding in free_bindings
        ]
        output = None
        if llm_free_slot_names:
            from llmr.reasoning.slot_filler import run_slot_filler
            output = run_slot_filler(
                instruction=self.instruction,
                action_cls=expression.type,
                free_slot_names=llm_free_slot_names,
                fixed_slots=fixed_slots,
                world_context=world_context,
                llm=self.llm,
            )
            if output is None:
                raise LLMSlotFillingFailed(action_name=expression.type.__name__)

        # ── 4. Resolve each free slot ──────────────────────────────────────────
        from llmr.pycram_bridge.introspector import PycramIntrospector
        from llmr.world.grounder import EntityGrounder

        _intro = PycramIntrospector()
        grounder = EntityGrounder(self.groundable_type)
        slot_by_name = {slot.field_name: slot for slot in output.slots} if output else {}
        # Tracks successfully resolved top-level values so nested entity
        # auto-grounding can use them (e.g. arm → pick matching Manipulator).
        resolved_params: Dict[str, Any] = {}

        for binding in free_bindings:
            resolved = resolve_binding_value(
                binding=binding,
                introspector=_intro,
                slot_by_name=slot_by_name,
                grounder=grounder,
                resolved_params=resolved_params,
                unresolved=_UNRESOLVED,
            )

            if resolved is _UNRESOLVED:
                logger.debug(
                    "LLMBackend: field '%s' unresolved — leaving as default.",
                    binding.attribute_name,
                )
                continue

            resolved_params[binding.attribute_name] = resolved
            try:
                binding.variable._value_ = resolved
            except Exception as exc:
                logger.warning(
                    "LLMBackend: cannot set field '%s': %s",
                    binding.attribute_name,
                    exc,
                )

        if self.strict_required:
            unresolved_required = unresolved_required_fields(expression, _intro)
            if unresolved_required:
                raise LLMUnresolvedRequiredFields(
                    action_name=getattr(expression.type, "__name__", str(expression.type)),
                    unresolved_fields=unresolved_required,
                )

        expression._update_kwargs_from_literal_values()
        yield expression.construct_instance()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_world_context(self) -> str:
        if self.world_context_provider is not None:
            try:
                return self.world_context_provider()
            except Exception as exc:
                logger.warning(
                    "LLMBackend: world_context_provider raised %s — falling back to SymbolGraph.",
                    exc,
                )
        from llmr.world.serializer import serialize_world_from_symbol_graph
        return serialize_world_from_symbol_graph(self.groundable_type)
