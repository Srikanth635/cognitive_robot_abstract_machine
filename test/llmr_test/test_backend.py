"""Tests for LLMBackend — GenerativeBackend implementation using LLM.

Uses ScriptedLLM with pre-built responses. Real SymbolGraph cleared via autouse fixture.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import pytest

from .scripted_llm import ScriptedLLM
from .test_actions import (
    MockPickUpAction,
    MockGraspDescription,
    GraspType,
)

from llmr import backend
from llmr.backend import LLMBackend, coerce_primitive
from llmr.exceptions import LLMSlotFillingFailed, LLMUnresolvedRequiredFields
from llmr.pycram_bridge.introspector import FieldKind
from llmr.schemas.entities import EntityDescriptionSchema
from llmr.schemas.slots import ActionReasoningOutput, SlotValue
from krrood.symbol_graph.symbol_graph import Symbol
from krrood.entity_query_language.query.match import Match


@dataclass
class FakeComplex:
    manipulator: object


class Manipulator:
    ...


class PrefixedNameLike:
    def __init__(self, name):
        self.name = name


class RaisingGrounder:
    def ground(self, _description):
        raise AssertionError("Manipulator fallback should not call the entity grounder")


class RaisingLLM(ScriptedLLM):
    def with_structured_output(self, schema, **kwargs):
        raise AssertionError("LLM should not be called for fully specified matches")


class MockBody(Symbol):
    def __init__(self, name: str):
        self.name = name


# ── Existing tests (kept for compatibility) ──────────────────────────────────


def test_reconstruct_complex_falls_back_for_unresolved_required_entity(monkeypatch):
    fallback_manipulator = object()
    fspec = SimpleNamespace(
        raw_type=FakeComplex,
        sub_fields=[
            SimpleNamespace(
                name="manipulator",
                kind=FieldKind.ENTITY,
                raw_type=Manipulator,
                is_optional=False,
            )
        ],
    )
    slot_by_name = {
        "grasp_description.manipulator": SimpleNamespace(
            entity_description=EntityDescriptionSchema(
                name="robot",
                semantic_type="Robot",
            ),
            value=None,
        )
    }

    monkeypatch.setattr(
        backend,
        "_auto_ground_sub_entity",
        lambda raw_type, resolved_params: fallback_manipulator,
    )

    result = backend._reconstruct_complex(
        field_name="grasp_description",
        fspec=fspec,
        slot_by_name=slot_by_name,
        grounder=RaisingGrounder(),
        resolved_params={},
    )

    assert result.manipulator is fallback_manipulator


def test_auto_ground_sub_entity_handles_prefixed_arm_names(monkeypatch):
    alpha = SimpleNamespace(name=PrefixedNameLike("alpha_manipulator"))
    beta = SimpleNamespace(name=PrefixedNameLike("beta_manipulator"))
    resolved_params = {"arm": SimpleNamespace(name=PrefixedNameLike("alpha"))}

    class FakeSymbolGraph:
        def get_instances_of_type(self, _raw_type):
            return [beta, alpha]

    monkeypatch.setattr(backend, "SymbolGraph", None, raising=False)
    monkeypatch.setattr(
        "krrood.symbol_graph.symbol_graph.SymbolGraph",
        lambda: FakeSymbolGraph(),
    )

    result = backend._auto_ground_sub_entity(Manipulator, resolved_params)

    assert result is alpha


# ── New tests for expanded coverage ────────────────────────────────────────


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

    def test_evaluate_reconstructs_complex_enum_slot(self) -> None:
        """Dotted complex sub-field output reconstructs the complex dataclass."""
        output = ActionReasoningOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(
                    field_name="grasp_description.grasp_type",
                    value="FRONT",
                )
            ],
        )
        llm = ScriptedLLM(responses=[output])
        milk = MockBody("milk")

        results = list(
            LLMBackend(llm=llm).evaluate(
                Match(MockPickUpAction)(
                    object_designator=milk,
                    grasp_description=...,
                )
            )
        )

        assert results == [
            MockPickUpAction(
                object_designator=milk,
                grasp_description=MockGraspDescription(grasp_type=GraspType.FRONT),
            )
        ]

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


class TestCoercePrimitive:
    """coerce_primitive() — string-to-type coercion for LLM slot values."""

    def test_float_string_coerced_to_float(self) -> None:
        """String float is cast to float."""
        assert coerce_primitive("42.5", float) == 42.5
        assert isinstance(coerce_primitive("42.5", float), float)

    def test_int_string_coerced_to_int(self) -> None:
        """String int is cast to int."""
        assert coerce_primitive("7", int) == 7
        assert isinstance(coerce_primitive("7", int), int)

    @pytest.mark.parametrize("value,expected", [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
    ])
    def test_bool_string_coerced_to_bool(self, value: str, expected: bool) -> None:
        """Bool string variants are correctly coerced to bool."""
        assert coerce_primitive(value, bool) is expected

    def test_str_passthrough(self) -> None:
        """String values for str fields are returned unchanged."""
        assert coerce_primitive("hello", str) == "hello"

    def test_invalid_float_returns_original_string(self) -> None:
        """Non-numeric string for float field returns the string unchanged."""
        assert coerce_primitive("not_a_float", float) == "not_a_float"

    def test_invalid_int_returns_original_string(self) -> None:
        """Non-numeric string for int field returns the string unchanged."""
        assert coerce_primitive("not_an_int", int) == "not_an_int"


class TestGetWorldContext:
    """LLMBackend._get_world_context() — world context generation."""

    def test_get_world_context_returns_string(self) -> None:
        """_get_world_context returns a non-empty world state description."""
        llm = ScriptedLLM(responses=[])
        backend_inst = LLMBackend(llm=llm)
        context = backend_inst._get_world_context()
        assert "## World State Summary" in context
        assert "## Semantic annotations" in context

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
