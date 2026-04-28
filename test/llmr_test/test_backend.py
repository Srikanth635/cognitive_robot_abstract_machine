"""Tests for :mod:`llmr.backend` — LLMBackend._evaluate pipeline with scripted LLM."""

from __future__ import annotations

from typing_extensions import Any, Dict

import pytest

from llmr.backend import LLMBackend, _UNRESOLVED, _UnresolvedSentinel
from llmr.bridge.match_reader import underspecified_match
from llmr.exceptions import LLMSlotFillingFailed, LLMUnresolvedRequiredFields
from llmr.schemas import (
    SlotFillingOutput,
    ActionAnnotationBundle,
    EntityDescription,
    SlotValue,
)

from ._fixtures.actions import (
    GraspType,
    MockNavigateAction,
    MockPickUpAction,
    MockRequiredNestedAction,
)
from ._fixtures.symbols import WorldBody
from ._fixtures.worlds import symbol_world  # noqa: F401
from .scripted_llm import ScriptedLLM


class TestUnresolvedSentinel:
    """The module-level ``_UNRESOLVED`` sentinel."""

    def test_repr(self) -> None:
        assert repr(_UNRESOLVED) == "<UNRESOLVED>"

    def test_is_unique(self) -> None:
        """Two ``_UnresolvedSentinel()`` instances are not equal — the module exposes only one."""
        assert _UNRESOLVED is not _UnresolvedSentinel()


class TestEvaluateFastPath:
    """Fully-resolved Match expressions bypass the LLM call."""

    def test_no_free_slots_skips_llm(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """When every required field is already bound, ``_evaluate`` yields directly."""
        milk = symbol_world["milk_on_table"]
        match = underspecified_match(MockNavigateAction)
        slot = next(iter(match.matches_with_variables))
        slot.assigned_variable._value_ = milk

        # An LLM that would crash if called — verifies the fast path is taken.
        crashing_llm = ScriptedLLM(responses=[])
        backend = LLMBackend(llm=crashing_llm)
        result = next(iter(backend.evaluate(match)))
        assert isinstance(result, MockNavigateAction)
        assert result.target_location is milk


class TestEvaluateHappyPath:
    """LLM-driven slot filling with a scripted entity_description."""

    def test_resolves_single_entity_slot(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        match = underspecified_match(MockPickUpAction)
        response = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(
                    field_name="object_designator",
                    entity_description=EntityDescription(name="milk_on_table"),
                )
            ],
        )
        backend = LLMBackend(
            llm=ScriptedLLM(responses=[response]),
            symbol_type=WorldBody,
        )
        result = next(iter(backend.evaluate(match)))
        assert isinstance(result, MockPickUpAction)
        assert result.object_designator is symbol_world["milk_on_table"]


class TestEvaluateErrorPaths:
    """Error behaviours: LLM returning nothing and strict_required unresolved."""

    def test_llm_failure_raises_slot_filling_failed(self) -> None:
        """When the LLM returns ``None``, the backend raises :class:`LLMSlotFillingFailed`."""
        match = underspecified_match(MockPickUpAction)

        class NullLLM(ScriptedLLM):
            def with_structured_output(self, schema: Any, **kwargs: Any):
                from langchain_core.runnables import RunnableLambda

                def _broken(messages: Any, **kw: Any) -> Any:
                    raise RuntimeError("LLM is down")

                return RunnableLambda(_broken)

        backend = LLMBackend(llm=NullLLM(responses=[]), symbol_type=WorldBody)
        with pytest.raises(LLMSlotFillingFailed):
            next(iter(backend.evaluate(match)))

    def test_strict_required_raises_when_unresolved(self) -> None:
        """Empty slot output with ``strict_required=True`` raises :class:`LLMUnresolvedRequiredFields`."""
        match = underspecified_match(MockPickUpAction)
        response = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(
                    field_name="object_designator",
                    entity_description=EntityDescription(name="does_not_exist"),
                )
            ],
        )
        backend = LLMBackend(
            llm=ScriptedLLM(responses=[response]),
            symbol_type=WorldBody,
            strict_required=True,
        )
        with pytest.raises(LLMUnresolvedRequiredFields) as exc_info:
            next(iter(backend.evaluate(match)))
        assert "object_designator" in exc_info.value.unresolved_fields


class TestWorldContextProvider:
    """Custom ``world_context_provider`` is used and falls back on exception."""

    def test_provider_injects_custom_text(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        captured: Dict[str, str] = {}

        class RecordingScripted(ScriptedLLM):
            def with_structured_output(self, schema: Any, **kwargs: Any):
                runnable = super().with_structured_output(schema, **kwargs)
                original_invoke = runnable.invoke

                def _invoke(messages: Any, **kw: Any) -> Any:
                    captured["user"] = next(
                        msg["content"] for msg in messages if msg["role"] == "user"
                    )
                    return original_invoke(messages, **kw)

                runnable.invoke = _invoke  # type: ignore[method-assign]
                return runnable

        match = underspecified_match(MockPickUpAction)
        response = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(
                    field_name="object_designator",
                    entity_description=EntityDescription(name="milk_on_table"),
                )
            ],
        )
        backend = LLMBackend(
            llm=RecordingScripted(responses=[response]),
            symbol_type=WorldBody,
            world_context_provider=lambda: "## Custom Context\nSPECIAL_WORLD",
        )
        next(iter(backend.evaluate(match)))
        assert "SPECIAL_WORLD" in captured["user"]

    def test_provider_exception_falls_back_to_symbol_graph(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """If the custom provider raises, the backend still serialises the SymbolGraph."""

        def _boom() -> str:
            raise RuntimeError("provider down")

        backend = LLMBackend(
            llm=ScriptedLLM(responses=[]),
            symbol_type=WorldBody,
            world_context_provider=_boom,
        )
        ctx = backend._build_world_context()
        assert "World State Summary" in ctx


class TestSemanticsSidecar:
    """`backend.semantics` accumulates LLM-inferred annotations during `_evaluate`."""

    def test_semantics_none_before_evaluate(self) -> None:
        """Semantics is ``None`` until ``_evaluate`` runs."""
        backend = LLMBackend(llm=ScriptedLLM(responses=[]))
        assert backend.semantics is None

    def test_semantics_populated_on_fast_path(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """Even when no slot filling happens, semantics is initialised with action_type."""
        milk = symbol_world["milk_on_table"]
        match = underspecified_match(MockNavigateAction)
        slot = next(iter(match.matches_with_variables))
        slot.assigned_variable._value_ = milk

        backend = LLMBackend(llm=ScriptedLLM(responses=[]))
        next(iter(backend.evaluate(match)))
        assert isinstance(backend.semantics, ActionAnnotationBundle)
        assert backend.semantics.action_type == "MockNavigateAction"
        assert backend.semantics.slot_filling is None

    def test_semantics_captures_slot_filling_output(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """After a successful slot-filling call, semantics holds the raw LLM output."""
        match = underspecified_match(MockPickUpAction)
        response = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(
                    field_name="object_designator",
                    entity_description=EntityDescription(name="milk_on_table"),
                )
            ],
        )
        backend = LLMBackend(
            llm=ScriptedLLM(responses=[response]),
            symbol_type=WorldBody,
        )
        next(iter(backend.evaluate(match)))
        assert backend.semantics is not None
        assert backend.semantics.action_type == "MockPickUpAction"
        assert backend.semantics.slot_filling is response


class TestReasonerPlugin:
    """Extra :class:`Reasoner` implementations run after slot filling, failures isolated."""

    def _happy_path_backend(
        self, reasoners: list
    ) -> "tuple[LLMBackend, SlotFillingOutput]":
        response = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(
                    field_name="object_designator",
                    entity_description=EntityDescription(name="milk_on_table"),
                )
            ],
        )
        backend = LLMBackend(
            llm=ScriptedLLM(responses=[response]),
            symbol_type=WorldBody,
            reasoners=reasoners,
        )
        return backend, response

    def test_reasoner_annotates_semantics(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """A Reasoner's ``annotate`` is called and can write to ``semantics.extra``."""
        from llmr.reasoning import Reasoner

        calls: Dict[str, Any] = {}

        class MarkerReasoner(Reasoner):
            def annotate(self, semantics, match_data, world_context) -> None:
                calls["action_type"] = semantics.action_type
                calls["world_context_present"] = bool(world_context)
                semantics.extra["marker"] = "visited"

        match = underspecified_match(MockPickUpAction)
        backend, _ = self._happy_path_backend([MarkerReasoner()])
        next(iter(backend.evaluate(match)))

        assert calls["action_type"] == "MockPickUpAction"
        assert calls["world_context_present"] is True
        assert backend.semantics.extra["marker"] == "visited"

    def test_reasoner_exception_is_swallowed(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """A failing reasoner does not break slot filling or block execution."""

        from llmr.reasoning import Reasoner

        class FailingReasoner(Reasoner):
            def annotate(self, semantics, match_data, world_context) -> None:
                raise RuntimeError("reasoner exploded")

        class QuietReasoner(Reasoner):
            def annotate(self, semantics, match_data, world_context) -> None:
                semantics.extra["quiet"] = True

        match = underspecified_match(MockPickUpAction)
        backend, _ = self._happy_path_backend([FailingReasoner(), QuietReasoner()])
        result = next(iter(backend.evaluate(match)))
        assert isinstance(result, MockPickUpAction)
        # The quiet reasoner still ran after the failing one.
        assert backend.semantics.extra.get("quiet") is True


class TestHypothesisGraphIntegration:
    """Optional hypothesis-graph projection runs after action construction."""

    def test_graph_manager_receives_resolved_projection_context(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        match = underspecified_match(MockPickUpAction)
        response = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(
                    field_name="object_designator",
                    entity_description=EntityDescription(name="milk_on_table"),
                )
            ],
        )

        calls: Dict[str, Any] = {}

        class RecordingGraphManager:
            def project(self, context: Any) -> None:
                calls["context"] = context

        backend = LLMBackend(
            llm=ScriptedLLM(responses=[response]),
            symbol_type=WorldBody,
            instruction="pick up the milk",
            hypothesis_graph_manager=RecordingGraphManager(),
        )
        result = next(iter(backend.evaluate(match)))

        context = calls["context"]
        assert context.action is result
        assert context.action_type == "MockPickUpAction"
        assert context.semantics is backend.semantics
        assert context.match_data.action_name == "MockPickUpAction"
        assert context.resolved_slots["object_designator"] is symbol_world["milk_on_table"]
        assert context.instruction == "pick up the milk"

    def test_graph_manager_failure_is_swallowed(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        match = underspecified_match(MockPickUpAction)
        response = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(
                    field_name="object_designator",
                    entity_description=EntityDescription(name="milk_on_table"),
                )
            ],
        )

        class FailingGraphManager:
            def project(self, context: Any) -> None:
                raise RuntimeError("projection exploded")

        backend = LLMBackend(
            llm=ScriptedLLM(responses=[response]),
            symbol_type=WorldBody,
            hypothesis_graph_manager=FailingGraphManager(),
        )
        result = next(iter(backend.evaluate(match)))
        assert isinstance(result, MockPickUpAction)
