"""Tests for slot-filler — LLM-driven parameter resolution.

Uses ScriptedLLM with pre-built responses. No network, no API keys.
"""

from __future__ import annotations

import pytest
from ..scripted_llm import RecordingLLM, ScriptedLLM
from .._fixtures.actions import (
    GraspType,
    MockNavigateAction,
    MockPickUpAction,
)
from llmr.exceptions import LLMActionRegistryEmpty
from llmr.reasoning.slot_filler import (
    infer_action_class,
    fill_slots,
)
from llmr.schemas import (
    ActionClassificationResult,
    SlotFillingOutput,
    EntityDescription,
    SlotValue,
)


def _last_user_prompt(llm: RecordingLLM) -> str:
    messages = llm.messages[-1]
    return next(msg["content"] for msg in messages if msg["role"] == "user")


class TestClassifyAction:
    """infer_action_class() — NL instruction → action class."""

    def test_returns_classification_for_registered_name(self) -> None:
        """Returns the raw ActionClassificationResult; callers do their own registry lookup."""
        registry = {"MockPickUpAction": MockPickUpAction}
        llm = ScriptedLLM(
            responses=[ActionClassificationResult(action_type="MockPickUpAction")]
        )
        classification = infer_action_class("pick up the milk", llm, action_registry=registry)
        assert classification is not None
        assert classification.action_type == "MockPickUpAction"
        assert registry[classification.action_type] is MockPickUpAction

    def test_returns_classification_even_for_unknown_name(self) -> None:
        """The raw classification is returned untouched when action_type is not in the registry."""
        registry = {"MockPickUpAction": MockPickUpAction}
        llm = ScriptedLLM(responses=[ActionClassificationResult(action_type="UnknownAction")])
        classification = infer_action_class("do something", llm, action_registry=registry)
        assert classification is not None
        assert classification.action_type == "UnknownAction"
        assert classification.action_type not in registry

    def test_raises_registry_empty_when_no_actions(self) -> None:
        """Raises LLMActionRegistryEmpty when registry is empty."""
        llm = ScriptedLLM(responses=[ActionClassificationResult(action_type="X")])
        with pytest.raises(LLMActionRegistryEmpty):
            infer_action_class("pick up milk", llm, action_registry={})

    def test_preserves_confidence_and_reasoning(self) -> None:
        """infer_action_class returns confidence and reasoning unchanged."""
        llm = ScriptedLLM(
            responses=[
                ActionClassificationResult(
                    action_type="MockNavigateAction",
                    confidence=0.95,
                    reasoning="user said navigate",
                )
            ]
        )
        classification = infer_action_class(
            "navigate to the kitchen",
            llm,
            action_registry={"MockNavigateAction": MockNavigateAction},
        )
        assert classification is not None
        assert classification.action_type == "MockNavigateAction"
        assert classification.confidence == 0.95
        assert classification.reasoning == "user said navigate"


class TestRunSlotFiller:
    """fill_slots() — action class + free slots → filled parameters."""

    def test_returns_action_reasoning_output(self) -> None:
        """fill_slots returns SlotFillingOutput."""
        expected = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(
                    field_name="object_designator",
                    entity_description=EntityDescription(name="milk"),
                    reasoning="instruction mentions milk",
                )
            ],
        )
        llm = ScriptedLLM(responses=[expected])
        result = fill_slots(
            instruction="pick up the milk",
            action_cls=MockPickUpAction,
            free_slot_names=["object_designator"],
            fixed_slots={},
            world_context="milk is on the table",
            llm=llm,
        )
        assert result is not None
        assert result.action_type == "MockPickUpAction"
        assert len(result.slots) == 1

    def test_returns_none_when_llm_raises(self) -> None:
        """fill_slots returns None when LLM call fails."""
        # Create an LLM that raises an exception on invoke
        from langchain_core.runnables import RunnableLambda

        class FailingLLM:
            def with_structured_output(self, schema):
                def _failing_invoke(messages):
                    raise RuntimeError("LLM error")

                return RunnableLambda(_failing_invoke)

        llm = FailingLLM()
        result = fill_slots(
            instruction="pick up",
            action_cls=MockPickUpAction,
            free_slot_names=["object_designator"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )
        assert result is None

    def test_handles_multiple_free_slots(self) -> None:
        """fill_slots processes multiple free slots."""
        output = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(field_name="object_designator", value="milk"),
                SlotValue(field_name="timeout", value="30.0"),
            ],
        )
        llm = ScriptedLLM(responses=[output])
        result = fill_slots(
            instruction="pick up milk",
            action_cls=MockPickUpAction,
            free_slot_names=["object_designator", "timeout"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )
        assert result is not None
        assert len(result.slots) == 2

    def test_prompt_lists_required_output_field_names(self) -> None:
        """Prompt explicitly names every SlotValue the LLM must return."""
        output = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(field_name="object_designator", value="milk"),
                SlotValue(field_name="timeout", value="30.0"),
            ],
        )
        llm = RecordingLLM(responses=[output])

        fill_slots(
            instruction="pick up milk",
            action_cls=MockPickUpAction,
            free_slot_names=["object_designator", "timeout"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )

        prompt = _last_user_prompt(llm)
        assert "Required free slot field_names:" in prompt
        assert "  - object_designator" in prompt
        assert "  - timeout" in prompt
        assert "Return exactly 2 SlotValue entries" in prompt
        assert "Do not omit enum or primitive slots" in prompt

    def test_retries_and_merges_when_required_slots_are_omitted(self) -> None:
        """A partial LLM response gets one correction call and is merged."""
        first = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[SlotValue(field_name="object_designator", value="milk")],
        )
        repair = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[SlotValue(field_name="timeout", value="30.0")],
        )
        llm = RecordingLLM(responses=[first, repair])

        result = fill_slots(
            instruction="pick up milk",
            action_cls=MockPickUpAction,
            free_slot_names=["object_designator", "timeout"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )

        assert result is not None
        assert [slot.field_name for slot in result.slots] == [
            "object_designator",
            "timeout",
        ]
        assert len(llm.messages) == 2
        repair_prompt = _last_user_prompt(llm)
        assert (
            "Correction: the previous structured response omitted required free slots."
            in repair_prompt
        )
        assert "  - timeout" in repair_prompt

    def test_retries_for_missing_nested_dotted_slot(self) -> None:
        """Missing nested enum SlotValues are detected by dotted field name."""
        first = SlotFillingOutput(action_type="MockPickUpAction", slots=[])
        repair = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[SlotValue(field_name="grasp_description.grasp_type", value="FRONT")],
        )
        llm = RecordingLLM(responses=[first, repair])

        result = fill_slots(
            instruction="grasp from front",
            action_cls=MockPickUpAction,
            free_slot_names=["MockPickUpAction.grasp_description.grasp_type"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )

        assert result is not None
        assert result.slots[0].field_name == "grasp_description.grasp_type"
        assert len(llm.messages) == 2

    def test_passes_world_context_to_prompt(self) -> None:
        """fill_slots includes world_context in the LLM prompt."""
        output = SlotFillingOutput(action_type="MockPickUpAction", slots=[])
        llm = RecordingLLM(responses=[output])
        world_context = "milk is on the table, table is in kitchen"

        fill_slots(
            instruction="pick up the milk",
            action_cls=MockPickUpAction,
            free_slot_names=[],
            fixed_slots={},
            world_context=world_context,
            llm=llm,
        )

        assert world_context in _last_user_prompt(llm)

    def test_top_level_complex_slot_is_not_expanded(self) -> None:
        """Top-level complex dataclass slots are left to nested KRROOD Match leaves."""
        output = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[],
        )
        llm = RecordingLLM(responses=[output])
        result = fill_slots(
            instruction="pick up",
            action_cls=MockPickUpAction,
            free_slot_names=["grasp_description"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )
        assert result is not None

        prompt = _last_user_prompt(llm)
        assert "grasp_description.grasp_type" not in prompt
        assert "Complex slots" not in prompt

    def test_optional_instruction(self) -> None:
        """fill_slots works with None instruction."""
        output = SlotFillingOutput(action_type="MockPickUpAction", slots=[])
        llm = ScriptedLLM(responses=[output])
        result = fill_slots(
            instruction=None,
            action_cls=MockPickUpAction,
            free_slot_names=[],
            fixed_slots={},
            world_context="",
            llm=llm,
        )
        assert result is not None

    def test_strips_field_name_prefixes(self) -> None:
        """fill_slots handles 'ClassName.field' format."""
        output = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[SlotValue(field_name="object_designator", value="x")],
        )
        llm = ScriptedLLM(responses=[output])
        # Free slot names may have 'MockPickUpAction.' prefix
        result = fill_slots(
            instruction="pick up",
            action_cls=MockPickUpAction,
            free_slot_names=["MockPickUpAction.object_designator"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )
        assert result is not None

    def test_uses_per_field_docstrings(self) -> None:
        """Prompt includes docstrings extracted from action class."""
        output = SlotFillingOutput(action_type="MockPickUpAction", slots=[])
        llm = RecordingLLM(responses=[output])
        fill_slots(
            instruction="pick up",
            action_cls=MockPickUpAction,
            free_slot_names=["object_designator"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )

        prompt = _last_user_prompt(llm)
        assert "Action type: MockPickUpAction" in prompt
        assert "Minimal stand-in for PyCRAM PickUpAction." in prompt
        assert "object_designator" in prompt
        assert "The object to pick up." in prompt

    def test_enum_slot_includes_valid_members(self) -> None:
        """Prompt lists enum member names for ENUM slots."""
        output = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[SlotValue(field_name="grasp_description.grasp_type", value="FRONT")],
        )
        llm = RecordingLLM(responses=[output])
        fill_slots(
            instruction="grasp from front",
            action_cls=MockPickUpAction,
            free_slot_names=["grasp_description.grasp_type"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )

        prompt = _last_user_prompt(llm)
        assert "grasp_description.grasp_type" in prompt
        assert "allowed values: FRONT | TOP | SIDE" in prompt

    def test_nested_enum_slot_includes_valid_members(self) -> None:
        """Prompt lists enum values when the free slot is already nested."""
        output = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[SlotValue(field_name="grasp_description.grasp_type", value="FRONT")],
        )
        llm = RecordingLLM(responses=[output])
        fill_slots(
            instruction="grasp from front",
            action_cls=MockPickUpAction,
            free_slot_names=["MockPickUpAction.grasp_description.grasp_type"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )

        prompt = _last_user_prompt(llm)
        assert "grasp_description.grasp_type" in prompt
        assert "allowed values: FRONT | TOP | SIDE" in prompt
        assert "Additional free slots" not in prompt

    def test_fixed_slots_are_included_in_prompt(self) -> None:
        """Prompt includes fixed slots so the LLM can preserve them."""
        output = SlotFillingOutput(action_type="MockPickUpAction", slots=[])
        llm = RecordingLLM(responses=[output])

        fill_slots(
            instruction="pick up the milk",
            action_cls=MockPickUpAction,
            free_slot_names=["timeout"],
            fixed_slots={"object_designator": "milk"},
            world_context="world",
            llm=llm,
        )

        prompt = _last_user_prompt(llm)
        assert "Already-fixed slots" in prompt
        assert "object_designator = 'milk'" in prompt


class TestPrefixedDottedSlotPrompt:
    """Prompt rendering for KRROOD-prefixed dotted slot names."""

    def test_prefixed_dotted_enum_slot_renders_allowed_values_not_fallback(
        self,
    ) -> None:
        """A fully-prefixed dotted enum slot renders allowed values, not the fallback section."""
        output = SlotFillingOutput(
            action_type="MockPickUpAction",
            slots=[SlotValue(field_name="grasp_description.grasp_type", value="SIDE")],
        )
        llm = RecordingLLM(responses=[output])
        fill_slots(
            instruction="grasp from side",
            action_cls=MockPickUpAction,
            free_slot_names=["MockPickUpAction.grasp_description.grasp_type"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )

        prompt = _last_user_prompt(llm)
        assert "grasp_description.grasp_type" in prompt
        assert "allowed values:" in prompt
        assert "FRONT" in prompt and "TOP" in prompt and "SIDE" in prompt
        assert "Additional free slots" not in prompt
