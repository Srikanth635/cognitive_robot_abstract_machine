"""Tests for LLMBackend — GenerativeBackend implementation using LLM.

Uses ScriptedLLM with pre-built responses. Real SymbolGraph cleared via autouse fixture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Optional
import pytest

from .scripted_llm import ScriptedLLM
from .test_actions import (
    MockPickUpAction,
    MockGraspDescription,
    GraspType,
)

from llmr.backend import LLMBackend
from llmr.exceptions import LLMSlotFillingFailed, LLMUnresolvedRequiredFields
from llmr.schemas.entities import EntityDescriptionSchema
from llmr.schemas.slots import ActionReasoningOutput, SlotValue
from krrood.symbol_graph.symbol_graph import Symbol
from krrood.entity_query_language.factories import variable_from
from krrood.entity_query_language.query.match import Match


class Manipulator(Symbol):
    def __init__(self, name: str = "manipulator"):
        self.name = name


class RaisingLLM(ScriptedLLM):
    def with_structured_output(self, schema, **kwargs):
        raise AssertionError("LLM should not be called for fully specified matches")


class MockBody(Symbol):
    def __init__(self, name: str):
        self.name = name


@dataclass
class MockRequiredNestedAction:
    object_designator: Symbol
    grasp_description: MockGraspDescription


@dataclass
class MockGraspWithManipulator:
    grasp_type: GraspType
    manipulator: Manipulator


@dataclass
class MockRequiredManipulatorAction:
    grasp_description: MockGraspWithManipulator


@dataclass
class MockNestedWithTimeoutAction:
    grasp_description: MockGraspDescription
    timeout: Optional[float] = None


class TestLLMBackendFields:
    """LLMBackend initialization and field validation."""

    def test_llm_is_required(self) -> None:
        """LLMBackend requires an llm parameter."""
        with pytest.raises(TypeError):
            LLMBackend()  # type: ignore

    def test_groundable_type_defaults_to_symbol(self) -> None:
        """groundable_type defaults to Symbol."""
        llm = ScriptedLLM(responses=[])
        backend_inst = LLMBackend(llm=llm)
        assert backend_inst.groundable_type is Symbol

    def test_instruction_defaults_to_none(self) -> None:
        """instruction parameter defaults to None."""
        llm = ScriptedLLM(responses=[])
        backend_inst = LLMBackend(llm=llm)
        assert backend_inst.instruction is None

    def test_strict_required_defaults_to_false(self) -> None:
        """strict_required defaults to False."""
        llm = ScriptedLLM(responses=[])
        backend_inst = LLMBackend(llm=llm)
        assert backend_inst.strict_required is False

    def test_accepts_all_parameters(self) -> None:
        """LLMBackend accepts all documented parameters."""
        llm = ScriptedLLM(responses=[])
        backend_inst = LLMBackend(
            llm=llm,
            groundable_type=Symbol,
            instruction="test",
            strict_required=True,
        )
        assert backend_inst.llm is llm
        assert backend_inst.groundable_type is Symbol
        assert backend_inst.instruction == "test"
        assert backend_inst.strict_required is True


class TestLLMBackendEvaluate:
    """LLMBackend.evaluate() — GenerativeBackend.evaluate implementation."""

    def test_fully_specified_match_constructs_without_llm(self) -> None:
        """A fully specified Match is constructed directly without an LLM call."""
        milk = MockBody("milk")
        backend_inst = LLMBackend(llm=RaisingLLM(responses=[]))

        results = list(
            backend_inst.evaluate(Match(MockPickUpAction)(object_designator=milk))
        )

        assert results == [MockPickUpAction(object_designator=milk)]

    def test_fully_specified_symbolic_variable_constructs_concrete_value(self) -> None:
        """A fixed singleton KRROOD variable is evaluated before construction."""
        milk = MockBody("milk")
        backend_inst = LLMBackend(llm=RaisingLLM(responses=[]))

        results = list(
            backend_inst.evaluate(
                Match(MockPickUpAction)(object_designator=variable_from([milk]))
            )
        )

        assert results == [MockPickUpAction(object_designator=milk)]
        assert results[0].object_designator is milk

    def test_fully_specified_nested_match_constructs_without_llm(self) -> None:
        """A fully specified nested Match is recursively constructed by KRROOD."""
        milk = MockBody("milk")
        backend_inst = LLMBackend(llm=RaisingLLM(responses=[]))

        results = list(
            backend_inst.evaluate(
                Match(MockRequiredNestedAction)(
                    object_designator=milk,
                    grasp_description=Match(MockGraspDescription)(
                        grasp_type=GraspType.TOP,
                    ),
                )
            )
        )

        assert results == [
            MockRequiredNestedAction(
                object_designator=milk,
                grasp_description=MockGraspDescription(grasp_type=GraspType.TOP),
            )
        ]

    def test_evaluate_preserves_fixed_slots(self) -> None:
        """evaluate() respects fixed slot values."""
        output = ActionReasoningOutput(
            action_type="MockPickUpAction",
            slots=[SlotValue(field_name="timeout", value="30.0")],
        )
        llm = ScriptedLLM(responses=[output])
        backend_inst = LLMBackend(llm=llm)
        milk = MockBody("milk")

        results = list(
            backend_inst.evaluate(
                Match(MockPickUpAction)(object_designator=milk, timeout=...)
            )
        )

        assert results == [
            MockPickUpAction(object_designator=milk, timeout=30.0)
        ]

    def test_top_level_complex_ellipsis_requires_nested_match(self) -> None:
        """Complex dataclass fields must be represented as nested KRROOD Match objects."""
        output = ActionReasoningOutput(
            action_type="MockRequiredNestedAction",
            slots=[
                SlotValue(
                    field_name="grasp_description.grasp_type",
                    value="FRONT",
                )
            ],
        )
        llm = ScriptedLLM(responses=[output])
        milk = MockBody("milk")

        with pytest.raises(LLMUnresolvedRequiredFields) as exc_info:
            list(
                LLMBackend(llm=llm, strict_required=True).evaluate(
                    Match(MockRequiredNestedAction)(
                        object_designator=milk,
                        grasp_description=...,
                    )
                )
            )

        assert exc_info.value.unresolved_fields == ["grasp_description"]

    def test_fixed_nested_slots_are_passed_to_llm_as_dotted_paths(self, monkeypatch) -> None:
        """Fixed nested leaves are exposed to the LLM with canonical dotted names."""
        captured = {}

        def fake_run_slot_filler(**kwargs):
            captured.update(kwargs)
            return ActionReasoningOutput(
                action_type="MockNestedWithTimeoutAction",
                slots=[SlotValue(field_name="timeout", value="5.0")],
            )

        monkeypatch.setattr(
            "llmr.reasoning.slot_filler.run_slot_filler",
            fake_run_slot_filler,
        )

        results = list(
            LLMBackend(llm=ScriptedLLM(responses=[])).evaluate(
                Match(MockNestedWithTimeoutAction)(
                    grasp_description=Match(MockGraspDescription)(
                        grasp_type=GraspType.FRONT,
                    ),
                    timeout=...,
                )
            )
        )

        assert captured["free_slot_names"] == ["timeout"]
        assert captured["fixed_slots"] == {
            "grasp_description.grasp_type": GraspType.FRONT,
        }
        assert results == [
            MockNestedWithTimeoutAction(
                grasp_description=MockGraspDescription(grasp_type=GraspType.FRONT),
                timeout=5.0,
            )
        ]

    def test_strict_required_accepts_resolved_nested_match(self) -> None:
        """A required top-level field can be satisfied by a resolved nested Match."""
        output = ActionReasoningOutput(
            action_type="MockRequiredNestedAction",
            slots=[
                SlotValue(
                    field_name="grasp_description.grasp_type",
                    value="FRONT",
                )
            ],
        )
        milk = MockBody("milk")

        results = list(
            LLMBackend(llm=ScriptedLLM(responses=[output]), strict_required=True).evaluate(
                Match(MockRequiredNestedAction)(
                    object_designator=milk,
                    grasp_description=Match(MockGraspDescription)(grasp_type=...),
                )
            )
        )

        assert results == [
            MockRequiredNestedAction(
                object_designator=milk,
                grasp_description=MockGraspDescription(grasp_type=GraspType.FRONT),
            )
        ]

    def test_nested_manipulator_leaf_auto_grounds_before_entity_grounding(
        self,
        monkeypatch,
    ) -> None:
        """Nested Manipulator leaves use SymbolGraph auto-grounding, not Body grounding."""
        fallback_manipulator = Manipulator("left_manipulator")

        def fake_ground_expected_entity(
            raw_type,
            description,
            resolved_params,
            symbol_graph=None,
        ):
            return fallback_manipulator

        monkeypatch.setattr(
            "llmr.world.grounder.ground_expected_entity",
            fake_ground_expected_entity,
        )

        output = ActionReasoningOutput(
            action_type="MockRequiredManipulatorAction",
            slots=[
                SlotValue(
                    field_name="grasp_description.grasp_type",
                    value="TOP",
                ),
                SlotValue(
                    field_name="grasp_description.manipulator",
                    entity_description=EntityDescriptionSchema(name="gripper"),
                ),
            ],
        )

        results = list(
            LLMBackend(
                llm=ScriptedLLM(responses=[output]),
                groundable_type=MockBody,
                strict_required=True,
            ).evaluate(
                Match(MockRequiredManipulatorAction)(
                    grasp_description=Match(MockGraspWithManipulator)(
                        grasp_type=...,
                        manipulator=...,
                    ),
                )
            )
        )

        assert results == [
            MockRequiredManipulatorAction(
                grasp_description=MockGraspWithManipulator(
                    grasp_type=GraspType.TOP,
                    manipulator=fallback_manipulator,
                )
            )
        ]
        assert results[0].grasp_description.manipulator is fallback_manipulator

    def test_strict_required_raises_for_unresolved_required_slot(self) -> None:
        """strict_required=True reports required slots that remain Ellipsis."""
        llm = ScriptedLLM(
            responses=[ActionReasoningOutput(action_type="MockPickUpAction", slots=[])]
        )

        with pytest.raises(LLMUnresolvedRequiredFields) as exc_info:
            list(
                LLMBackend(llm=llm, strict_required=True).evaluate(
                    Match(MockPickUpAction)(object_designator=...)
                )
            )

        assert exc_info.value.unresolved_fields == ["object_designator"]

    def test_strict_required_raises_for_unresolved_required_nested_match(self) -> None:
        """strict_required=True catches required nested matches with unresolved leaves."""
        milk = MockBody("milk")
        llm = ScriptedLLM(
            responses=[ActionReasoningOutput(action_type="MockRequiredNestedAction", slots=[])]
        )

        with pytest.raises(LLMUnresolvedRequiredFields) as exc_info:
            list(
                LLMBackend(llm=llm, strict_required=True).evaluate(
                    Match(MockRequiredNestedAction)(
                        object_designator=milk,
                        grasp_description=Match(MockGraspDescription)(grasp_type=...),
                    )
                )
            )

        assert exc_info.value.unresolved_fields == ["grasp_description"]

    def test_coerce_primitive_float_via_evaluate(self) -> None:
        """String float value from LLM is coerced to float, not kept as string."""
        output = ActionReasoningOutput(
            action_type="MockPickUpAction",
            slots=[SlotValue(field_name="timeout", value="42.5")],
        )
        milk = MockBody("milk")
        results = list(
            LLMBackend(llm=ScriptedLLM(responses=[output])).evaluate(
                Match(MockPickUpAction)(object_designator=milk, timeout=...)
            )
        )
        assert results == [MockPickUpAction(object_designator=milk, timeout=42.5)]
        assert isinstance(results[0].timeout, float)

    def test_slot_filler_none_raises_slot_filling_failed(self, monkeypatch) -> None:
        """A missing LLM slot-filler output is surfaced as a typed exception."""
        monkeypatch.setattr(
            "llmr.reasoning.slot_filler.run_slot_filler",
            lambda **kwargs: None,
        )

        with pytest.raises(LLMSlotFillingFailed):
            list(
                LLMBackend(llm=ScriptedLLM(responses=[])).evaluate(
                    Match(MockPickUpAction)(object_designator=...)
                )
            )


class TestGetWorldContext:
    """LLMBackend._get_world_context() — world context generation."""

    def test_get_world_context_returns_string(self) -> None:
        """_get_world_context returns a non-empty world state description."""
        llm = ScriptedLLM(responses=[])
        backend_inst = LLMBackend(llm=llm)
        context = backend_inst._get_world_context()
        assert "## World State Summary" in context
        assert "## Available Semantic Types" in context

    def test_get_world_context_uses_provider_when_set(self) -> None:
        """_get_world_context uses world_context_provider if set."""
        def custom_provider():
            return "custom world context"

        llm = ScriptedLLM(responses=[])
        backend_inst = LLMBackend(
            llm=llm,
            world_context_provider=custom_provider,
        )
        context = backend_inst._get_world_context()
        assert context == "custom world context"


class TestPrivateEvaluate:
    """LLMBackend._evaluate() — private Match resolution."""

    def test_private_evaluate_handles_match(self) -> None:
        """_evaluate() processes Match expressions."""
        milk = MockBody("milk")
        llm = RaisingLLM(responses=[])
        backend_inst = LLMBackend(llm=llm)

        results = list(
            backend_inst._evaluate(Match(MockPickUpAction)(object_designator=milk))
        )

        assert results == [MockPickUpAction(object_designator=milk)]

