"""Tests for generic LLM slot-output value coercion."""
from __future__ import annotations

from typing_extensions import Optional

from llmr.match_inspection import MatchBinding
from llmr.pycram_bridge.introspector import FieldKind
from llmr.schemas.slots import SlotValue
from llmr.slot_resolution import (
    coerce_enum,
    coerce_primitive,
    resolve_binding_value,
)

from .test_actions import GraspType


class FakeIntrospector:
    def __init__(self, kind):
        self.kind = kind

    def classify_type(self, _field_type):
        return self.kind


def make_binding(field_type, prompt_name: str = "slot") -> MatchBinding:
    return MatchBinding(
        attribute_name=prompt_name,
        prompt_name=prompt_name,
        variable=object(),
        field_type=field_type,
        value=...,
    )


def test_coerce_enum_exact_match() -> None:
    assert coerce_enum("FRONT", GraspType) is GraspType.FRONT


def test_coerce_enum_case_insensitive_match() -> None:
    assert coerce_enum("front", GraspType) is GraspType.FRONT


def test_coerce_enum_unknown_value_falls_back_to_first_member(capfd) -> None:
    result = coerce_enum("UNKNOWN_GRASP", GraspType)
    captured = capfd.readouterr()

    assert result is GraspType.FRONT
    assert "UNKNOWN_GRASP" in captured.err


def test_coerce_primitive_casts_numeric_and_bool_values() -> None:
    assert coerce_primitive("42.5", float) == 42.5
    assert coerce_primitive("7", int) == 7
    assert coerce_primitive("yes", bool) is True
    assert coerce_primitive("no", bool) is False


def test_coerce_primitive_unwraps_optional_single_type() -> None:
    assert coerce_primitive("3.5", Optional[float]) == 3.5


def test_coerce_primitive_keeps_invalid_numeric_strings() -> None:
    assert coerce_primitive("not_a_float", float) == "not_a_float"
    assert coerce_primitive("not_an_int", int) == "not_an_int"


def test_resolve_binding_value_coerces_enum_slot() -> None:
    unresolved = object()

    result = resolve_binding_value(
        binding=make_binding(GraspType),
        introspector=FakeIntrospector(FieldKind.ENUM),
        slot_by_name={"slot": SlotValue(field_name="slot", value="TOP")},
        grounder=None,
        resolved_params={},
        unresolved=unresolved,
    )

    assert result is GraspType.TOP


def test_resolve_binding_value_coerces_primitive_slot() -> None:
    unresolved = object()

    result = resolve_binding_value(
        binding=make_binding(float),
        introspector=FakeIntrospector(FieldKind.PRIMITIVE),
        slot_by_name={"slot": SlotValue(field_name="slot", value="4.25")},
        grounder=None,
        resolved_params={},
        unresolved=unresolved,
    )

    assert result == 4.25


def test_resolve_binding_value_returns_unresolved_for_missing_slot() -> None:
    unresolved = object()

    result = resolve_binding_value(
        binding=make_binding(GraspType),
        introspector=FakeIntrospector(FieldKind.ENUM),
        slot_by_name={},
        grounder=None,
        resolved_params={},
        unresolved=unresolved,
    )

    assert result is unresolved
