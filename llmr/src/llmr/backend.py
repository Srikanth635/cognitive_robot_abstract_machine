"""LLM-backed `GenerativeBackend` for resolving underspecified `Match` objects.

World context is derived from `SymbolGraph`, and all KRROOD access is funneled
through `llmr.bridge`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing_extensions import Any, Callable, Dict, Iterable, List, Optional, Type

from krrood.entity_query_language.backends import GenerativeBackend
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.utils import T
from krrood.symbol_graph.symbol_graph import Symbol

from llmr.bridge.match_reader import (
    construct_action,
    snapshot_match,
    missing_required_fields,
    bind_slot_value,
)
from llmr.bridge.introspect import ActionFieldIntrospector
from llmr.exceptions import LLMSlotFillingFailed, LLMUnresolvedRequiredFields
from llmr.resolution.resolver import resolve_slot
from llmr.schemas import ActionClassificationResult, ActionAnnotationBundle

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from llmr.hypotheses.projection import ProjectionOrchestrator as HypothesisGraphManager
    from llmr.reasoning import Reasoner

logger = logging.getLogger(__name__)


# ── Typed sentinel ─────────────────────────────────────────────────────────────


class _UnresolvedSentinel:
    """Typed sentinel returned when a slot cannot be resolved."""

    def __repr__(self) -> str:
        return "<UNRESOLVED>"


_UNRESOLVED = _UnresolvedSentinel()


# ── LLMBackend ─────────────────────────────────────────────────────────────────


@dataclass
class LLMBackend(GenerativeBackend):
    """A GenerativeBackend that uses an LLM to fill underspecified Match slots."""

    llm: "BaseChatModel"
    """LangChain BaseChatModel — the reasoning engine for slot filling and action classification."""

    symbol_type: Type[Symbol] = field(default=Symbol)
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

    world_context_provider: Optional[Callable[[], str]] = field(
        kw_only=True, default=None
    )
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

    reasoners: List["Reasoner"] = field(default_factory=list, kw_only=True)
    """
    Optional extra :class:`~llmr.reasoning.Reasoner` implementations invoked
    after slot filling completes.  Each reasoner annotates
    :attr:`semantics` in place (e.g. populating ``motion_phases``, ``frames``,
    ``extra["my_reasoner"]``).  Failures are logged and suppressed so grounding
    and execution continue unaffected.
    """

    classification: Optional[ActionClassificationResult] = field(default=None, kw_only=True)
    """
    Optional upstream :class:`~llmr.schemas.ActionClassificationResult`.  When set,
    it is copied into :attr:`semantics` at the start of :meth:`_evaluate` so
    downstream reasoners and consumers see the classification context without
    the caller having to mutate ``backend.semantics`` directly.
    """

    hypothesis_graph_manager: Optional["HypothesisGraphManager"] = field(
        default=None, kw_only=True, repr=False
    )
    """
    Optional :class:`~llmr.hypotheses.projection.ProjectionOrchestrator` used
    to project reasoner sidecars into the epistemic hypothesis graph after
    action construction. Projection failures are logged and suppressed.
    """

    semantics: Optional[ActionAnnotationBundle] = field(default=None, init=False, repr=False)
    """
    Open-schema sidecar populated during :meth:`_evaluate`.  Accumulates every
    LLM-inferred annotation around the action (classification, slot filling,
    future motion phases, FrameNet roles, etc.) without affecting grounding or
    PyCRAM execution.  Callers that want this metadata read it off the backend
    after evaluation — e.g. ``backend.semantics.slot_filling``.  Reset each call.
    """

    # ── Core interface ─────────────────────────────────────────────────────────

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        """Resolve all free slots in *expression* and yield a concrete action instance."""

        # ── 1. Snapshot the Match expression into plain data ──────────────────
        introspector = ActionFieldIntrospector()
        match_data = snapshot_match(expression, introspector, unresolved=_UNRESOLVED)
        self.semantics = ActionAnnotationBundle(
            action_type=match_data.action_name,
            instruction=self.instruction,
            classification=self.classification,
        )

        if not match_data.free_slots:
            yield construct_action(match_data)
            return

        # ── 2. World context ───────────────────────────────────────────────────
        world_context = self._build_world_context()

        # ── 3. Slot filler (LLM call with dynamic prompt) ─────────────────────
        from llmr.reasoning.slot_filler import fill_slots

        slot_filling_output = fill_slots(
            instruction=self.instruction,
            action_cls=match_data.action_type,
            free_slot_names=match_data.free_slot_names,
            fixed_slots=match_data.fixed_bindings,
            world_context=world_context,
            llm=self.llm,
        )
        if slot_filling_output is None:
            raise LLMSlotFillingFailed(action_name=match_data.action_name)
        self.semantics.slot_filling = slot_filling_output

        # ── 4. Resolve each free slot and write it back into the Match ─────────
        from llmr.resolution.grounder import EntityGrounding

        grounder = EntityGrounding(self.symbol_type)
        slot_values_by_name = {slot.field_name: slot for slot in slot_filling_output.slots}
        # Successfully resolved top-level values are threaded into nested entity
        # auto-grounding (e.g. arm → matching Manipulator).
        grounded_bindings: Dict[str, Any] = {}

        for slot in match_data.free_slots:
            resolved = resolve_slot(
                slot=slot,
                slot_by_name=slot_values_by_name,
                grounder=grounder,
                resolved_params=grounded_bindings,
                unresolved=_UNRESOLVED,
            )

            if resolved is _UNRESOLVED:
                logger.debug(
                    "LLMBackend: field '%s' unresolved — leaving as default.",
                    slot.attribute_name,
                )
                continue

            grounded_bindings[slot.attribute_name] = resolved
            bind_slot_value(slot, resolved)

        # ── 5. Extra reasoners (optional, failure-isolated) ───────────────────
        for reasoner in self.reasoners:
            try:
                reasoner.annotate(self.semantics, match_data, world_context)
            except Exception as exc:
                logger.warning(
                    "LLMBackend: reasoner %r raised %s — annotation skipped.",
                    reasoner,
                    exc,
                )

        if self.strict_required:
            unresolved = missing_required_fields(match_data, introspector)
            if unresolved:
                raise LLMUnresolvedRequiredFields(
                    action_name=match_data.action_name,
                    unresolved_fields=unresolved,
                )

        action = construct_action(match_data)
        self._project_hypotheses(
            action=action,
            match_data=match_data,
            world_context=world_context,
        )
        yield action

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_world_context(self) -> str:
        if self.world_context_provider is not None:
            try:
                return self.world_context_provider()
            except Exception as exc:
                logger.warning(
                    "LLMBackend: world_context_provider raised %s — falling back to SymbolGraph.",
                    exc,
                )
        from llmr.bridge.world_reader import render_world_context

        return render_world_context(self.symbol_type)

    def _project_hypotheses(
        self,
        action: T,
        match_data: Any,
        world_context: str,
    ) -> None:
        """Project resolved action semantics into the hypothesis graph."""

        if self.hypothesis_graph_manager is None or self.semantics is None:
            return

        from llmr.hypotheses.projection import ProjectionInput

        resolved_slots = {
            slot.prompt_name: slot.value
            for slot in match_data.slots
            if not slot.is_free
        }

        try:
            self.hypothesis_graph_manager.project(
                ProjectionInput(
                    instruction=self.instruction,
                    action=action,
                    action_type=match_data.action_name,
                    semantics=self.semantics,
                    match_data=match_data,
                    resolved_slots=resolved_slots,
                    world_context=world_context,
                    symbol_type=self.symbol_type,
                    llm_model_name=self._infer_model_name(),
                )
            )
        except Exception as exc:
            logger.warning(
                "LLMBackend: hypothesis graph projection raised %s — projection skipped.",
                exc,
            )

    def _infer_model_name(self) -> Optional[str]:
        """Return a best-effort model identifier for projection metadata."""

        for attr_name in ("model_name", "model", "model_id"):
            value = getattr(self.llm, attr_name, None)
            if isinstance(value, str) and value:
                return value
        return type(self.llm).__name__ or None
