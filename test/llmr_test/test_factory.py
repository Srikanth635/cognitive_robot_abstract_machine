"""Tests for factory functions — nl_plan, nl_sequential, resolve_match, resolve_params.

Uses ScriptedLLM with pre-built responses. Real SymbolGraph cleared via autouse.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from .scripted_llm import ScriptedLLM
from .test_actions import GraspType, MockGraspDescription, MockPickUpAction

from llmr.factory import resolve_params
from llmr.match_construction import required_match
from llmr.exceptions import LLMUnresolvedRequiredFields
from llmr.schemas.slots import ActionReasoningOutput, SlotValue
from krrood.entity_query_language.query.match import Match
from krrood.symbol_graph.symbol_graph import Symbol


class MockBody(Symbol):
    def __init__(self, name: str):
        self.name = name


@dataclass
class MockRequiredComplexAction:
    grasp_description: MockGraspDescription


class TestResolveParams:
    """resolve_params() — standalone parameter resolution (no context/execution)."""

    def test_returns_concrete_action_instance(self) -> None:
        """resolve_params returns a concrete action instance."""
        output = ActionReasoningOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(field_name="timeout", value="12.5")
            ],
        )
        llm = ScriptedLLM(responses=[output])
        milk = MockBody("milk")

        match = Match(MockPickUpAction)(object_designator=milk, timeout=...)

        result = resolve_params(match, llm=llm, strict_required=False)

        assert result == MockPickUpAction(object_designator=milk, timeout=12.5)

    def test_does_not_require_context(self) -> None:
        """resolve_params does not require a PyCRAM Context."""
        output = ActionReasoningOutput(action_type="MockPickUpAction", slots=[])
        llm = ScriptedLLM(responses=[output])
        milk = MockBody("milk")

        match = Match(MockPickUpAction)(object_designator=milk)

        result = resolve_params(match, llm=llm)

        assert result == MockPickUpAction(object_designator=milk)

    def test_accepts_custom_instructions(self) -> None:
        """resolve_params accepts instruction parameter for context."""
        output = ActionReasoningOutput(action_type="MockPickUpAction", slots=[])
        llm = ScriptedLLM(responses=[output])
        milk = MockBody("milk")

        match = Match(MockPickUpAction)(object_designator=milk)

        result = resolve_params(
            match,
            llm=llm,
            instruction="pick up the milk",
        )

        assert result == MockPickUpAction(object_designator=milk)

    def test_accepts_groundable_type(self) -> None:
        """resolve_params accepts groundable_type parameter."""
        output = ActionReasoningOutput(action_type="MockPickUpAction", slots=[])
        llm = ScriptedLLM(responses=[output])
        milk = MockBody("milk")

        match = Match(MockPickUpAction)(object_designator=milk)

        result = resolve_params(
            match,
            llm=llm,
            groundable_type=Symbol,
        )

        assert result == MockPickUpAction(object_designator=milk)

    def test_strict_required_mode(self) -> None:
        """resolve_params respects strict_required parameter."""
        output = ActionReasoningOutput(action_type="MockPickUpAction", slots=[])
        llm = ScriptedLLM(responses=[output])

        match = Match(MockPickUpAction)(object_designator=...)

        with pytest.raises(LLMUnresolvedRequiredFields) as exc_info:
            resolve_params(
                match,
                llm=llm,
                strict_required=True,
            )

        assert exc_info.value.unresolved_fields == ["object_designator"]

    def test_resolves_factory_generated_required_complex_match(self) -> None:
        """Factory-generated nested complex Matches resolve through KRROOD leaves."""
        output = ActionReasoningOutput(
            action_type="MockRequiredComplexAction",
            slots=[
                SlotValue(
                    field_name="grasp_description.grasp_type",
                    value="TOP",
                )
            ],
        )
        llm = ScriptedLLM(responses=[output])

        match = required_match(MockRequiredComplexAction)

        result = resolve_params(match, llm=llm, strict_required=True)

        assert result == MockRequiredComplexAction(
            grasp_description=MockGraspDescription(grasp_type=GraspType.TOP)
        )
