"""sg_model-backed `GenerativeBackend` for resolving underspecified `Match` objects.

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

from llmr.bridge.introspect import ActionFieldIntrospector
from llmr.bridge.match_reader import (
    bind_slot_value,
    construct_action,
    missing_required_fields,
    snapshot_match,
)
from llmr.exceptions import LLMSlotFillingFailed, LLMUnresolvedRequiredFields
from llmr.resolution.resolver import resolve_slot
from llmr.schemas import ActionAnnotationBundle, ActionClassificationResult
from llmr.hypotheses import BuildInput, BuildOrchestrator, HypothesisGraph

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from llmr.reasoning import Reasoner
    from llmr.hypotheses import BuildResult

logger = logging.getLogger(__name__)


class _UnresolvedSentinel:
    """Typed sentinel returned when a slot cannot be resolved."""

    def __repr__(self) -> str:
        return "<UNRESOLVED>"


_UNRESOLVED = _UnresolvedSentinel()


@dataclass
class LLMBackend(GenerativeBackend):
    """A GenerativeBackend that uses an LLM and projects sidecars into sg_model."""

    llm: "BaseChatModel"
    """LangChain BaseChatModel used for slot filling and action classification."""

    symbol_type: Type[Symbol] = field(default=Symbol)
    """
    Symbol subclass scoping entity grounding and world serialisation.
    Defaults to ``Symbol`` (all instances); pass ``Body`` for physical-body-only scope.
    """

    instruction: Optional[str] = field(kw_only=True, default=None)
    """
    NL instruction included in the slot-filler prompt for semantic grounding
    (e.g. ``"the milk from the table"``). Omit when the action type and fixed
    slots already carry the intent.
    """

    world_context_provider: Optional[Callable[[], str]] = field(
        kw_only=True, default=None
    )
    """
    Callable returning a world-context string. Replaces the default SymbolGraph
    serialisation when provided.
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
    after slot filling completes. Each reasoner annotates :attr:`semantics` in place.
    """

    classification: Optional[ActionClassificationResult] = field(
        default=None, kw_only=True
    )
    """
    Optional upstream :class:`~llmr.schemas.ActionClassificationResult`. When set,
    it is copied into :attr:`semantics` at the start of :meth:`_evaluate`.
    """

    sg_model_orchestrator: Optional[BuildOrchestrator] = field(
        default=None, kw_only=True, repr=False
    )
    """
    Optional shared :class:`~llmr.hypotheses.BuildOrchestrator` used to project
    reasoner sidecars into the object-domain hypothesis graph after action
    construction. When omitted, one is created lazily on first use.
    """

    semantics: Optional[ActionAnnotationBundle] = field(default=None, init=False, repr=False)
    """
    Open-schema sidecar populated during :meth:`_evaluate`. Accumulates every
    LLM-inferred annotation around the action without affecting grounding or
    execution. Reset each call.
    """

    last_build_result: Optional["BuildResult"] = field(default=None, init=False, repr=False)
    """Most recent sg_model build result, if any."""

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        """Resolve all free slots in *expression* and yield a concrete action instance."""

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

        world_context = self._build_world_context()

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

        from llmr.resolution.grounder import EntityGrounding

        grounder = EntityGrounding(self.symbol_type)
        slot_values_by_name = {slot.field_name: slot for slot in slot_filling_output.slots}
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
                    "LLMBackend: field '%s' unresolved - leaving as default.",
                    slot.attribute_name,
                )
                continue

            grounded_bindings[slot.attribute_name] = resolved
            bind_slot_value(slot, resolved)

        for reasoner in self.reasoners:
            try:
                reasoner.annotate(self.semantics, match_data, world_context)
            except Exception as exc:
                logger.warning(
                    "LLMBackend: reasoner %r raised %s - annotation skipped.",
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
        self._build_sg_model(
            action=action,
            match_data=match_data,
            world_context=world_context,
        )
        yield action

    @property
    def hypothesis_graph(self) -> Optional[HypothesisGraph]:
        """Return the current sg_model graph, if an orchestrator exists."""

        if self.sg_model_orchestrator is None:
            return None
        return self.sg_model_orchestrator.graph

    def _build_world_context(self) -> str:
        if self.world_context_provider is not None:
            try:
                return self.world_context_provider()
            except Exception as exc:
                logger.warning(
                    "LLMBackend: world_context_provider raised %s - falling back to SymbolGraph.",
                    exc,
                )
        from llmr.bridge.world_reader import render_world_context

        return render_world_context(self.symbol_type)

    def _build_sg_model(
        self,
        action: T,
        match_data: Any,
        world_context: str,
    ) -> None:
        """Project resolved action semantics into the sg_model repository."""

        if self.semantics is None:
            return

        resolved_slots = {
            slot.prompt_name: slot.value
            for slot in match_data.slots
            if not slot.is_free
        }
        orchestrator = self._get_or_create_orchestrator()

        try:
            result = orchestrator.build(
                BuildInput(
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
            self.last_build_result = result
            for warning in result.warnings:
                logger.warning("LLMBackend: sg_model build warning - %s", warning)
        except Exception as exc:
            logger.warning(
                "LLMBackend: sg_model projection raised %s - build skipped.",
                exc,
            )

    def _get_or_create_orchestrator(self) -> BuildOrchestrator:
        if self.sg_model_orchestrator is None:
            self.sg_model_orchestrator = BuildOrchestrator.with_default_builders()
        return self.sg_model_orchestrator

    def _infer_model_name(self) -> Optional[str]:
        """Return a best-effort model identifier for build metadata."""

        for attr_name in ("model_name", "model", "model_id"):
            value = getattr(self.llm, attr_name, None)
            if isinstance(value, str) and value:
                return value
        return type(self.llm).__name__ or None
