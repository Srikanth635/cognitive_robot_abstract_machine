"""Tests for :mod:`llmr.schemas` — Pydantic LLM I/O models.

Covers :class:`EntityDescription`, :class:`SlotValue`,
:class:`SlotFillingOutput`, :class:`ActionClassificationResult`, and the
:class:`ActionAnnotationBundle` sidecar.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmr.schemas import (
    ActionClassificationResult,
    SlotFillingOutput,
    ActionAnnotationBundle,
    FrameNetCoreElements,
    EntityDescription,
    MotionPhase,
    FlanaganMotionPlan,
    FrameNetAnnotation,
    FrameNetPeripheralElements,
    SlotValue,
)


class TestEntityDescription:
    """EntityDescription Pydantic model."""

    def test_all_fields_accepted(self) -> None:
        """All fields (name, semantic_type, spatial_context, attributes) accepted."""
        schema = EntityDescription(
            name="milk bottle",
            semantic_type="FoodItem",
            spatial_context="on the kitchen counter",
            attributes={"color": "white", "size": "medium"},
        )
        assert schema.name == "milk bottle"
        assert schema.semantic_type == "FoodItem"
        assert schema.spatial_context == "on the kitchen counter"
        assert schema.attributes == {"color": "white", "size": "medium"}

    def test_only_name_required(self) -> None:
        """Only name is required; others default to None."""
        schema = EntityDescription(name="table")
        assert schema.name == "table"
        assert schema.semantic_type is None
        assert schema.spatial_context is None
        assert schema.attributes is None

    def test_missing_name_raises_validation_error(self) -> None:
        """Missing name field raises ValidationError."""
        with pytest.raises(ValidationError):
            EntityDescription(semantic_type="Surface")

    def test_attributes_defaults_to_none(self) -> None:
        """Attributes field defaults to None when not provided."""
        schema = EntityDescription(name="cup")
        assert schema.attributes is None

    def test_round_trip_json(self) -> None:
        """Schema can be serialized to JSON and reconstructed."""
        original = EntityDescription(
            name="red cup", semantic_type="Container", attributes={"color": "red"}
        )
        json_str = original.model_dump_json()
        reconstructed = EntityDescription.model_validate_json(json_str)
        assert reconstructed.name == original.name
        assert reconstructed.semantic_type == original.semantic_type
        assert reconstructed.attributes == original.attributes

    def test_extra_fields_ignored(self) -> None:
        """Extra fields are ignored (not rejected) by Pydantic."""
        schema = EntityDescription(name="object", unknown_field="ignored")
        assert schema.name == "object"
        assert not hasattr(schema, "unknown_field")


class TestSlotValue:
    """SlotValue Pydantic model — single resolved slot."""

    def test_entity_slot_with_description(self) -> None:
        """Entity slot with entity_description populated."""
        slot = SlotValue(
            field_name="object_designator",
            entity_description=EntityDescription(name="milk"),
            reasoning="instruction mentions milk",
        )
        assert slot.field_name == "object_designator"
        assert slot.entity_description.name == "milk"
        assert slot.value is None
        assert slot.reasoning == "instruction mentions milk"

    def test_primitive_slot_with_value(self) -> None:
        """Primitive slot with value string."""
        slot = SlotValue(
            field_name="timeout", value="30.0", reasoning="reasonable timeout"
        )
        assert slot.field_name == "timeout"
        assert slot.value == "30.0"
        assert slot.entity_description is None

    def test_field_name_required(self) -> None:
        """field_name is required."""
        with pytest.raises(ValidationError):
            SlotValue(value="some_value")

    def test_value_and_entity_description_both_optional(self) -> None:
        """Both value and entity_description can be None."""
        slot = SlotValue(field_name="some_field")
        assert slot.value is None
        assert slot.entity_description is None

    def test_reasoning_defaults_to_empty_string(self) -> None:
        """reasoning field defaults to empty string."""
        slot = SlotValue(field_name="field", value="val")
        assert slot.reasoning == ""

    def test_dotted_field_name_for_complex_subfield(self) -> None:
        """Dotted field names represent complex sub-fields."""
        slot = SlotValue(
            field_name="grasp_description.grasp_type",
            value="FRONT",
            reasoning="user said front-facing grasp",
        )
        assert slot.field_name == "grasp_description.grasp_type"
        assert slot.value == "FRONT"


class TestSlotFillingOutput:
    """SlotFillingOutput Pydantic model."""

    def test_valid_output_with_slots(self) -> None:
        """Valid output with action_type and slots list."""
        output = SlotFillingOutput(
            action_type="PickUpAction",
            slots=[
                SlotValue(
                    field_name="object_designator",
                    entity_description=EntityDescription(name="milk"),
                ),
                SlotValue(field_name="timeout", value="30.0"),
            ],
        )
        assert output.action_type == "PickUpAction"
        assert len(output.slots) == 2
        assert output.overall_reasoning == ""

    def test_action_type_required(self) -> None:
        """action_type is required."""
        with pytest.raises(ValidationError):
            SlotFillingOutput(slots=[])

    def test_overall_reasoning_defaults_to_empty(self) -> None:
        """overall_reasoning defaults to empty string."""
        output = SlotFillingOutput(action_type="NavigateAction", slots=[])
        assert output.overall_reasoning == ""

    def test_slots_can_be_empty(self) -> None:
        """slots list can be empty (though unusual)."""
        output = SlotFillingOutput(action_type="SomeAction", slots=[])
        assert output.slots == []

    def test_complex_nested_slots(self) -> None:
        """Multiple slots including complex sub-fields."""
        output = SlotFillingOutput(
            action_type="PickUpAction",
            slots=[
                SlotValue(
                    field_name="object_designator",
                    entity_description=EntityDescription(name="milk"),
                ),
                SlotValue(field_name="grasp_description.grasp_type", value="TOP"),
                SlotValue(
                    field_name="grasp_description.manipulator",
                    entity_description=EntityDescription(name="left_gripper"),
                ),
            ],
            overall_reasoning="found milk and chose top grasp with left gripper",
        )
        assert len(output.slots) == 3
        assert (
            output.overall_reasoning
            == "found milk and chose top grasp with left gripper"
        )


class TestActionClassificationResult:
    """ActionClassificationResult Pydantic model."""

    def test_confidence_defaults_to_one(self) -> None:
        """confidence field defaults to 1.0."""
        clf = ActionClassificationResult(action_type="PickUpAction")
        assert clf.confidence == 1.0

    def test_confidence_clamped_between_zero_and_one(self) -> None:
        """confidence must be in [0.0, 1.0]."""
        clf = ActionClassificationResult(action_type="X", confidence=0.75)
        assert clf.confidence == 0.75

    def test_confidence_zero_accepted(self) -> None:
        """confidence=0.0 is valid."""
        clf = ActionClassificationResult(action_type="X", confidence=0.0)
        assert clf.confidence == 0.0

    def test_confidence_one_accepted(self) -> None:
        """confidence=1.0 is valid."""
        clf = ActionClassificationResult(action_type="X", confidence=1.0)
        assert clf.confidence == 1.0

    def test_confidence_out_of_range_rejected(self) -> None:
        """confidence > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            ActionClassificationResult(action_type="X", confidence=1.5)

    def test_reasoning_defaults_to_empty(self) -> None:
        """reasoning field defaults to empty string."""
        clf = ActionClassificationResult(action_type="NavigateAction")
        assert clf.reasoning == ""

    def test_action_type_required(self) -> None:
        """action_type is required."""
        with pytest.raises(ValidationError):
            ActionClassificationResult(confidence=0.9)

    def test_full_classification_output(self) -> None:
        """Complete classification with all fields."""
        clf = ActionClassificationResult(
            action_type="PickUpAction",
            confidence=0.95,
            reasoning="user said pick up the milk",
        )
        assert clf.action_type == "PickUpAction"
        assert clf.confidence == 0.95
        assert clf.reasoning == "user said pick up the milk"


class TestActionSemantics:
    """ActionAnnotationBundle Pydantic sidecar — open-schema reasoner aggregate."""

    def test_minimal_construction_all_optional(self) -> None:
        """Only action_type is required; every reasoner slot defaults to None/empty."""
        sem = ActionAnnotationBundle(action_type="PickUpAction")
        assert sem.action_type == "PickUpAction"
        assert sem.classification is None
        assert sem.slot_filling is None
        assert sem.motion_phases is None
        assert sem.frames is None
        assert sem.preconditions is None
        assert sem.postconditions is None
        assert sem.affordances is None
        assert sem.extra == {}

    def test_missing_action_type_raises(self) -> None:
        """action_type is required."""
        with pytest.raises(ValidationError):
            ActionAnnotationBundle()

    def test_carries_core_reasoner_outputs(self) -> None:
        """classification and slot_filling nest their respective Pydantic models."""
        sem = ActionAnnotationBundle(
            action_type="PickUpAction",
            classification=ActionClassificationResult(
                action_type="PickUpAction", confidence=0.9
            ),
            slot_filling=SlotFillingOutput(
                action_type="PickUpAction",
                slots=[SlotValue(field_name="timeout", value="5.0")],
            ),
        )
        assert sem.classification.confidence == 0.9
        assert sem.slot_filling.slots[0].field_name == "timeout"

    def test_motion_phases_accepts_flanagan_representation(self) -> None:
        """motion_phases accepts a FlanaganMotionPlan from FlanaganReasoner."""
        sem = ActionAnnotationBundle(
            action_type="PickUpAction",
            motion_phases=FlanaganMotionPlan(
                instruction="pick up the milk",
                phases=[
                    MotionPhase(phase="Approach", target_object="milk"),
                    MotionPhase(phase="Grasp", target_object="milk",
                                  preconditions={"gripper_open": True}),
                    MotionPhase(phase="Lift", target_object="milk",
                                  goal_state={"milk_lifted": True}),
                ],
            ),
        )
        assert len(sem.motion_phases.phases) == 3
        assert sem.motion_phases.phases[0].phase == "Approach"
        assert sem.motion_phases.phases[1].preconditions == {"gripper_open": True}
        assert sem.motion_phases.phases[2].goal_state == {"milk_lifted": True}

    def test_frames_accepts_framenet_representation(self) -> None:
        """frames accepts a FrameNetAnnotation from FrameNetReasoner."""
        sem = ActionAnnotationBundle(
            action_type="PickUpAction",
            frames=FrameNetAnnotation(
                framenet="picking_up_object",
                frame="Getting",
                **{"lexical-unit": "pick_up.v"},
                core=FrameNetCoreElements(agent="robot", theme="milk", source="table"),
                peripheral=FrameNetPeripheralElements(direction="upward"),
            ),
        )
        assert sem.frames.frame == "Getting"
        assert sem.frames.core.theme == "milk"
        assert sem.frames.core.source == "table"
        assert sem.frames.peripheral.direction == "upward"

    def test_extra_is_open_bag(self) -> None:
        """extra accepts arbitrary future reasoner payloads."""
        sem = ActionAnnotationBundle(action_type="X")
        sem.extra["experimental_reasoner"] = {"foo": 1, "bar": ["a", "b"]}
        assert sem.extra["experimental_reasoner"]["foo"] == 1

    def test_round_trip_json(self) -> None:
        """ActionAnnotationBundle serializes and reconstructs losslessly."""
        original = ActionAnnotationBundle(
            action_type="PickUpAction",
            classification=ActionClassificationResult(action_type="PickUpAction"),
            preconditions=["object is graspable"],
            extra={"run_id": "abc-123"},
        )
        reconstructed = ActionAnnotationBundle.model_validate_json(original.model_dump_json())
        assert reconstructed.action_type == "PickUpAction"
        assert reconstructed.classification.action_type == "PickUpAction"
        assert reconstructed.preconditions == ["object is graspable"]
        assert reconstructed.extra == {"run_id": "abc-123"}
