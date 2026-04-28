"""Tests for :mod:`llmr.factory` — user-facing NL-driven entry points.

``execute_single`` is patched per-test because it wraps PyCRAM's real factory,
which the pure-llmr tests must not invoke.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing_extensions import Any, Dict, List

import pytest

from llmr.exceptions import LLMActionClassificationFailed
from llmr.factory import plan_from_instruction, sequential_plan_from_instruction, plan_from_match, instance_from_match
from llmr.schemas import (
    ActionClassificationResult,
    SlotFillingOutput,
    EntityDescription,
    SlotValue,
)

from ._fixtures.actions import MockNavigateAction, MockPickUpAction
from ._fixtures.symbols import WorldBody
from ._fixtures.worlds import symbol_world  # noqa: F401
from .scripted_llm import ScriptedLLM


@pytest.fixture
def fake_execute_single(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    """Replace ``execute_single`` in factory with a no-op recorder.

    Returns the list of ``(match, context)`` tuples that the factory forwarded.
    """
    calls: List[Any] = []

    def _fake(match: Any, context: Any) -> Any:
        calls.append((match, context))
        return SimpleNamespace(perform=lambda: None, match=match, context=context)

    monkeypatch.setattr("llmr.factory.execute_single", _fake)
    return calls


class TestNlPlan:
    """:func:`plan_from_instruction` — classify → build match → backend → plan node."""

    def test_classifies_and_returns_plan_node(
        self,
        fake_execute_single: List[Any],
        symbol_world: Dict[str, Any],  # noqa: F811
    ) -> None:
        llm = ScriptedLLM(
            responses=[
                ActionClassificationResult(action_type="MockPickUpAction"),
                SlotFillingOutput(
                    action_type="MockPickUpAction",
                    slots=[
                        SlotValue(
                            field_name="object_designator",
                            entity_description=EntityDescription(
                                name="milk_on_table"
                            ),
                        )
                    ],
                ),
            ]
        )
        context = SimpleNamespace(query_backend=None)
        plan = plan_from_instruction(
            instruction="pick up the milk",
            context=context,
            llm=llm,
            symbol_type=WorldBody,
            action_registry={"MockPickUpAction": MockPickUpAction},
        )
        # Factory forwarded (match, context) exactly once.
        assert len(fake_execute_single) == 1
        forwarded_match, forwarded_context = fake_execute_single[0]
        assert forwarded_context is context
        assert forwarded_match.type is MockPickUpAction
        # Backend was attached to the context, with the classification routed
        # in as a kwarg.  ``_evaluate`` — stubbed out in this test — is what
        # would copy it onto ``backend.semantics.classification`` at run time.
        assert context.query_backend is not None
        assert context.query_backend.classification is not None
        assert context.query_backend.classification.action_type == "MockPickUpAction"
        # PlanNode stub is returned.
        assert hasattr(plan, "perform")

    def test_threads_reasoners_and_hypothesis_graph_manager_to_backend(
        self,
        fake_execute_single: List[Any],
        symbol_world: Dict[str, Any],  # noqa: F811
    ) -> None:
        llm = ScriptedLLM(
            responses=[
                ActionClassificationResult(action_type="MockPickUpAction"),
                SlotFillingOutput(
                    action_type="MockPickUpAction",
                    slots=[
                        SlotValue(
                            field_name="object_designator",
                            entity_description=EntityDescription(
                                name="milk_on_table"
                            ),
                        )
                    ],
                ),
            ]
        )
        reasoners = [object()]
        graph_manager = object()
        context = SimpleNamespace(query_backend=None)

        plan_from_instruction(
            instruction="pick up the milk",
            context=context,
            llm=llm,
            symbol_type=WorldBody,
            action_registry={"MockPickUpAction": MockPickUpAction},
            reasoners=reasoners,
            hypothesis_graph_manager=graph_manager,
        )

        assert context.query_backend.reasoners is reasoners
        assert context.query_backend.hypothesis_graph_manager is graph_manager

    def test_raises_when_classifier_returns_none(
        self, fake_execute_single: List[Any]
    ) -> None:
        """An unknown action_type from the classifier yields ``LLMActionClassificationFailed``."""
        llm = ScriptedLLM(
            responses=[ActionClassificationResult(action_type="NotARegisteredAction")]
        )
        context = SimpleNamespace(query_backend=None)
        with pytest.raises(LLMActionClassificationFailed):
            plan_from_instruction(
                instruction="do something weird",
                context=context,
                llm=llm,
                action_registry={"MockPickUpAction": MockPickUpAction},
            )
        assert fake_execute_single == []


class TestNlSequential:
    """:func:`sequential_plan_from_instruction` — decompose then plan each step."""

    def test_returns_one_plan_per_decomposed_step(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_execute_single: List[Any],
        symbol_world: Dict[str, Any],  # noqa: F811
    ) -> None:
        # Stub the decomposer so the test does not invoke the LLM twice.
        from llmr.reasoning import decomposer as decomposer_mod

        def _stub_decompose(self: Any, instruction: str) -> Any:
            return decomposer_mod.DecomposedPlan(
                steps=["navigate to the table", "pick up the milk"],
                dependencies={},
            )

        monkeypatch.setattr(decomposer_mod.TaskDecomposer, "decompose", _stub_decompose)

        llm = ScriptedLLM(
            responses=[
                # Step 1: classification + slot filler for MockNavigateAction
                ActionClassificationResult(action_type="MockNavigateAction"),
                SlotFillingOutput(
                    action_type="MockNavigateAction",
                    slots=[
                        SlotValue(
                            field_name="target_location",
                            entity_description=EntityDescription(name="table"),
                        )
                    ],
                ),
                # Step 2: classification + slot filler for MockPickUpAction
                ActionClassificationResult(action_type="MockPickUpAction"),
                SlotFillingOutput(
                    action_type="MockPickUpAction",
                    slots=[
                        SlotValue(
                            field_name="object_designator",
                            entity_description=EntityDescription(
                                name="milk_on_table"
                            ),
                        )
                    ],
                ),
            ]
        )
        registry = {
            "MockNavigateAction": MockNavigateAction,
            "MockPickUpAction": MockPickUpAction,
        }
        context = SimpleNamespace(query_backend=None)
        plans = sequential_plan_from_instruction(
            instruction="go to the table and pick up the milk",
            context=context,
            llm=llm,
            symbol_type=WorldBody,
            action_registry=registry,
        )
        assert len(plans) == 2
        assert len(fake_execute_single) == 2

    def test_threads_reasoners_and_hypothesis_graph_manager_to_each_step(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_execute_single: List[Any],
        symbol_world: Dict[str, Any],  # noqa: F811
    ) -> None:
        from llmr.reasoning import decomposer as decomposer_mod

        def _stub_decompose(self: Any, instruction: str) -> Any:
            return decomposer_mod.DecomposedPlan(
                steps=["navigate to the table", "pick up the milk"],
                dependencies={},
            )

        monkeypatch.setattr(decomposer_mod.TaskDecomposer, "decompose", _stub_decompose)

        llm = ScriptedLLM(
            responses=[
                ActionClassificationResult(action_type="MockNavigateAction"),
                SlotFillingOutput(
                    action_type="MockNavigateAction",
                    slots=[
                        SlotValue(
                            field_name="target_location",
                            entity_description=EntityDescription(name="table"),
                        )
                    ],
                ),
                ActionClassificationResult(action_type="MockPickUpAction"),
                SlotFillingOutput(
                    action_type="MockPickUpAction",
                    slots=[
                        SlotValue(
                            field_name="object_designator",
                            entity_description=EntityDescription(
                                name="milk_on_table"
                            ),
                        )
                    ],
                ),
            ]
        )
        reasoners = [object()]
        graph_manager = object()
        registry = {
            "MockNavigateAction": MockNavigateAction,
            "MockPickUpAction": MockPickUpAction,
        }
        context = SimpleNamespace(query_backend=None)

        plans = sequential_plan_from_instruction(
            instruction="go to the table and pick up the milk",
            context=context,
            llm=llm,
            symbol_type=WorldBody,
            action_registry=registry,
            reasoners=reasoners,
            hypothesis_graph_manager=graph_manager,
        )

        assert len(plans) == 2
        assert all(call_context.query_backend.reasoners is reasoners for _, call_context in fake_execute_single)
        assert all(
            call_context.query_backend.hypothesis_graph_manager is graph_manager
            for _, call_context in fake_execute_single
        )


class TestResolveMatch:
    """:func:`plan_from_match` — backend wiring for an already-built Match."""

    def test_attaches_backend_and_delegates_to_execute_single(
        self,
        fake_execute_single: List[Any],
        symbol_world: Dict[str, Any],  # noqa: F811
    ) -> None:
        from llmr.bridge.match_reader import underspecified_match

        match = underspecified_match(MockPickUpAction)
        llm = ScriptedLLM(
            responses=[
                SlotFillingOutput(
                    action_type="MockPickUpAction",
                    slots=[
                        SlotValue(
                            field_name="object_designator",
                            entity_description=EntityDescription(
                                name="milk_on_table"
                            ),
                        )
                    ],
                )
            ]
        )
        context = SimpleNamespace(query_backend=None)
        plan = plan_from_match(
            match,
            context=context,
            llm=llm,
            symbol_type=WorldBody,
            instruction="pick up the milk",
        )
        assert hasattr(plan, "perform")
        assert context.query_backend is not None
        assert fake_execute_single == [(match, context)]

    def test_threads_reasoners_and_hypothesis_graph_manager(
        self,
        fake_execute_single: List[Any],
        symbol_world: Dict[str, Any],  # noqa: F811
    ) -> None:
        from llmr.bridge.match_reader import underspecified_match

        match = underspecified_match(MockPickUpAction)
        llm = ScriptedLLM(
            responses=[
                SlotFillingOutput(
                    action_type="MockPickUpAction",
                    slots=[
                        SlotValue(
                            field_name="object_designator",
                            entity_description=EntityDescription(
                                name="milk_on_table"
                            ),
                        )
                    ],
                )
            ]
        )
        reasoners = [object()]
        graph_manager = object()
        context = SimpleNamespace(query_backend=None)

        plan_from_match(
            match,
            context=context,
            llm=llm,
            symbol_type=WorldBody,
            instruction="pick up the milk",
            reasoners=reasoners,
            hypothesis_graph_manager=graph_manager,
        )

        assert context.query_backend.reasoners is reasoners
        assert context.query_backend.hypothesis_graph_manager is graph_manager


class TestResolveParams:
    """:func:`instance_from_match` — non-executing variant that returns the action instance."""

    def test_returns_constructed_action_instance(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        from llmr.bridge.match_reader import underspecified_match

        match = underspecified_match(MockPickUpAction)
        llm = ScriptedLLM(
            responses=[
                SlotFillingOutput(
                    action_type="MockPickUpAction",
                    slots=[
                        SlotValue(
                            field_name="object_designator",
                            entity_description=EntityDescription(
                                name="milk_on_table"
                            ),
                        )
                    ],
                )
            ]
        )
        result = instance_from_match(
            match,
            llm=llm,
            symbol_type=WorldBody,
            instruction="pick up the milk",
        )
        assert isinstance(result, MockPickUpAction)
        assert result.object_designator is symbol_world["milk_on_table"]

    def test_threads_reasoners_and_hypothesis_graph_manager(
        self,
        monkeypatch: pytest.MonkeyPatch,
        symbol_world: Dict[str, Any],  # noqa: F811
    ) -> None:
        from llmr.bridge.match_reader import underspecified_match

        match = underspecified_match(MockPickUpAction)
        llm = ScriptedLLM(
            responses=[
                SlotFillingOutput(
                    action_type="MockPickUpAction",
                    slots=[
                        SlotValue(
                            field_name="object_designator",
                            entity_description=EntityDescription(
                                name="milk_on_table"
                            ),
                        )
                    ],
                )
            ]
        )

        reasoners = [object()]
        graph_manager = object()
        captured: Dict[str, Any] = {}

        class _BackendStub:
            def evaluate(self, match: Any):
                captured["match"] = match
                return iter([MockPickUpAction(object_designator=symbol_world["milk_on_table"])])

        def _fake_backend(**kwargs: Any) -> Any:
            captured["kwargs"] = kwargs
            return _BackendStub()

        monkeypatch.setattr("llmr.factory._make_llm_backend", _fake_backend)

        result = instance_from_match(
            match,
            llm=llm,
            symbol_type=WorldBody,
            instruction="pick up the milk",
            reasoners=reasoners,
            hypothesis_graph_manager=graph_manager,
        )

        assert isinstance(result, MockPickUpAction)
        assert captured["match"] is match
        assert captured["kwargs"]["reasoners"] is reasoners
        assert captured["kwargs"]["hypothesis_graph_manager"] is graph_manager
