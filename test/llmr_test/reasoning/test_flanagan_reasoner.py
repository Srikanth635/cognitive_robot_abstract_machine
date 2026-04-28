"""Tests for :mod:`llmr.reasoning.flanagan_reasoner`.

Covers the 2-call pipeline redesign introduced in Phases 1-4:
  - _canonicalize_phase_name alias table and fallbacks            (Phase 1)
  - _PhaseStep free-str phase field                               (Phase 1)
  - _PhaseAnnotation / _AnnotatedPlan unified annotation schemas  (Phase 2)
  - render_resolved_slots match-data → prompt text conversion     (Phase 3)
  - FlanaganReasoner.annotate() 2-call pipeline                   (Phase 4)

No API keys or network access — all LLM calls use a lightweight mock.
"""

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from llmr.bridge.introspect import FieldKind
from llmr.bridge.match_reader import MatchSnapshot, MatchField, render_resolved_slots as render_resolved_slots
from llmr.reasoning.flanagan_reasoner import (
    _AnnotatedPlan,
    _ConditionEntry,
    _ForceDynamics,
    _ForceProfile,
    _ObjectAwarePhasePlanner,
    _PhaseAnnotation,
    _PhaseStep,
    _SensorSignal,
    _canonicalize_phase_name as _normalize_phase_name,
    FlanaganReasoner,
)
from llmr.schemas import ActionAnnotationBundle, MotionPhase


# ── Test helpers ──────────────────────────────────────────────────────────────


def _make_sequential_llm(responses: List[Any]):
    """Return (mock_llm, recorded) where mock_llm returns responses in order.

    Each call to ``with_structured_output(...).invoke(...)`` consumes the next
    entry in *responses* and appends the raw prompt input to *recorded*.
    Uses a shared counter across calls so successive ``with_structured_output``
    invocations each advance the sequence correctly.
    """
    counter = [0]
    recorded: List[Any] = []

    def _wso(schema: Any, **kwargs: Any) -> RunnableLambda:
        def _invoke(prompt_value: Any, **kw: Any) -> Any:
            recorded.append(prompt_value)
            obj = responses[counter[0] % len(responses)]
            counter[0] += 1
            return obj

        return RunnableLambda(_invoke)

    llm = MagicMock()
    llm.with_structured_output.side_effect = _wso
    return llm, recorded


def _prompt_text(recorded: List[Any], call_index: int) -> str:
    """Extract the rendered prompt string from *recorded* at *call_index*."""
    val = recorded[call_index]
    if hasattr(val, "to_messages"):
        return val.to_messages()[0].content
    return str(val)


def _make_slot(
    prompt_name: str,
    value: Any,
    kind: FieldKind,
    is_free: bool = False,
) -> MatchField:
    return MatchField(
        attribute_name=prompt_name,
        prompt_name=prompt_name,
        field_type=type(value),
        field_kind=kind,
        value=value,
        is_free=is_free,
        _variable=None,  # type: ignore[arg-type]
    )


def _make_match_data(name: str, slots: List[MatchField]) -> MatchSnapshot:
    return MatchSnapshot(
        action_type=type(None),
        action_name=name,
        slots=slots,
        _expression=None,  # type: ignore[arg-type]
    )


def _minimal_semantics(instruction: str = "pick up the milk") -> ActionAnnotationBundle:
    return ActionAnnotationBundle(action_type="PickUpAction", instruction=instruction)


# ── TestNormalizePhase ────────────────────────────────────────────────────────


class TestNormalizePhase:
    """_normalize_phase_name alias lookup and fallbacks."""

    def test_known_synonym_maps_to_canonical(self) -> None:
        assert _normalize_phase_name("grab") == "Grasp"
        assert _normalize_phase_name("carry") == "Transport"
        assert _normalize_phase_name("set_down") == "Place"
        assert _normalize_phase_name("pull_out") == "Withdraw"
        assert _normalize_phase_name("pick_up") == "Grasp"
        assert _normalize_phase_name("flip") == "Reorient"

    def test_already_canonical_returns_unchanged(self) -> None:
        for name in ("Grasp", "Lift", "Transport", "Place", "Approach", "Release"):
            assert _normalize_phase_name(name) == name

    def test_case_insensitive_lookup(self) -> None:
        assert _normalize_phase_name("GRASP") == "Grasp"
        assert _normalize_phase_name("transport") == "Transport"
        assert _normalize_phase_name("LIFT") == "Lift"

    def test_phrase_with_spaces_normalised(self) -> None:
        assert _normalize_phase_name("pick up") == "Grasp"
        assert _normalize_phase_name("move to") == "Approach"

    def test_novel_phrase_title_cased(self) -> None:
        assert _normalize_phase_name("scoop") == "Scoop"
        assert _normalize_phase_name("spread_sauce") == "Spread_Sauce"

    def test_empty_string_returns_empty(self) -> None:
        assert _normalize_phase_name("") == ""


# ── TestPhaseSchemas ──────────────────────────────────────────────────────────


class TestPhaseSchemas:
    """_PhaseStep and _ObjectAwarePhasePlanner schema validation."""

    def test_phase_step_accepts_arbitrary_string(self) -> None:
        """phase: str — any value accepted, not restricted to canonical names."""
        step = _PhaseStep(phase="pick_up_and_carry", target_object="cup")
        assert step.phase == "pick_up_and_carry"

    def test_phase_step_accepts_canonical_name(self) -> None:
        step = _PhaseStep(phase="Grasp", target_object="milk", description="grip the milk")
        assert step.phase == "Grasp"
        assert step.description == "grip the milk"

    def test_phase_step_description_optional(self) -> None:
        step = _PhaseStep(phase="Lift", target_object="box")
        assert step.description is None

    def test_phase_step_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            _PhaseStep(phase="Grasp", target_object="cup", unknown="extra")

    def test_object_aware_planner_roundtrip(self) -> None:
        original = _ObjectAwarePhasePlanner(phases=[
            _PhaseStep(phase="Approach", target_object="cup"),
            _PhaseStep(phase="scoop", target_object="sauce", description="novel phase"),
        ])
        restored = _ObjectAwarePhasePlanner.model_validate_json(original.model_dump_json())
        assert len(restored.phases) == 2
        assert restored.phases[1].phase == "scoop"


# ── TestAnnotationSchemas ─────────────────────────────────────────────────────


class TestAnnotationSchemas:
    """_PhaseAnnotation and _AnnotatedPlan schema construction and serialisation."""

    def _full_annotation(self) -> _PhaseAnnotation:
        return _PhaseAnnotation(
            phase_key="Grasp_milk",
            preconditions=[_ConditionEntry(key="gripper_open", value=True)],
            goal_state=[_ConditionEntry(key="milk_grasped", value=True)],
            force_dynamics=_ForceDynamics(
                contact=True,
                motion_type="pinch",
                force_exerted="5-8N",
                force_profile=_ForceProfile(
                    type="compression", expected_range_N=[5.0, 8.0]
                ),
            ),
            sensory_feedback=[_SensorSignal(signal_name="force_sensor_N", value=6.0)],
            possible_failures=["milk slips"],
            recovery_strategies=["reposition gripper"],
            max_duration_sec=2.5,
            urgency="medium",
        )

    def test_full_construction(self) -> None:
        ann = self._full_annotation()
        assert ann.phase_key == "Grasp_milk"
        assert ann.preconditions[0].key == "gripper_open"
        assert ann.force_dynamics.contact is True
        assert ann.force_dynamics.force_profile.expected_range_N == [5.0, 8.0]
        assert ann.sensory_feedback[0].value == 6.0
        assert ann.urgency == "medium"

    def test_default_fields_for_non_contact_phase(self) -> None:
        """Approach phase has no contact — force_dynamics is None, lists empty."""
        ann = _PhaseAnnotation(phase_key="Approach_cup")
        assert ann.force_dynamics is None
        assert ann.preconditions == []
        assert ann.goal_state == []
        assert ann.sensory_feedback == []
        assert ann.possible_failures == []
        assert ann.recovery_strategies == []
        assert ann.max_duration_sec == 5.0
        assert ann.urgency == "medium"

    def test_urgency_literal_validated(self) -> None:
        with pytest.raises(ValidationError):
            _PhaseAnnotation(phase_key="X", urgency="critical")  # type: ignore

    def test_condition_entry_value_types(self) -> None:
        """_ConditionEntry accepts bool, str, int, float, None."""
        assert _ConditionEntry(key="a", value=True).value is True
        assert _ConditionEntry(key="b", value="open").value == "open"
        assert _ConditionEntry(key="c", value=3).value == 3
        assert _ConditionEntry(key="d", value=1.5).value == 1.5
        assert _ConditionEntry(key="e", value=None).value is None

    def test_annotated_plan_roundtrip_json(self) -> None:
        plan = _AnnotatedPlan(phases=[self._full_annotation()])
        restored = _AnnotatedPlan.model_validate_json(plan.model_dump_json())
        assert restored.phases[0].phase_key == "Grasp_milk"
        assert restored.phases[0].force_dynamics.contact is True

    def test_annotated_plan_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            _PhaseAnnotation(phase_key="X", unknown_field="bad")


# ── TestFormatSlots ───────────────────────────────────────────────────────────


class _FakeEnum(Enum):
    LEFT = "left"
    RIGHT = "right"


class _FakeBody:
    """Duck-typed body with name but no global_pose."""
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeBodyWithPose(_FakeBody):
    """Duck-typed body with name and global_pose."""
    def __init__(self, name: str, x: float, y: float, z: float) -> None:
        super().__init__(name)
        self.global_pose = SimpleNamespace(
            to_position=lambda: SimpleNamespace(x=x, y=y, z=z)
        )


class TestFormatSlots:
    """render_resolved_slots(match_data) → human-readable prompt text."""

    def test_no_slots_returns_placeholder(self) -> None:
        md = _make_match_data("PickUpAction", slots=[])
        assert render_resolved_slots(md) == "(no resolved parameters)"

    def test_all_free_slots_returns_placeholder(self) -> None:
        md = _make_match_data("PickUpAction", slots=[
            _make_slot("arm", ..., FieldKind.PRIMITIVE, is_free=True),
        ])
        assert render_resolved_slots(md) == "(no resolved parameters)"

    def test_primitive_slot_shows_value(self) -> None:
        md = _make_match_data("PickUpAction", slots=[
            _make_slot("timeout", "30.0", FieldKind.PRIMITIVE),
        ])
        text = render_resolved_slots(md)
        assert "PickUpAction" in text
        assert "timeout" in text
        assert "30.0" in text

    def test_enum_slot_shows_name(self) -> None:
        md = _make_match_data("PickUpAction", slots=[
            _make_slot("arm", _FakeEnum.LEFT, FieldKind.PRIMITIVE),
        ])
        text = render_resolved_slots(md)
        assert "arm" in text
        assert "LEFT" in text

    def test_entity_slot_without_pose_shows_type(self) -> None:
        body = _FakeBody("milk_bottle")
        md = _make_match_data("PickUpAction", slots=[
            _make_slot("object_designator", body, FieldKind.ENTITY),
        ])
        text = render_resolved_slots(md)
        assert "object_designator" in text
        assert "milk_bottle" in text
        assert "_FakeBody" in text

    def test_entity_slot_with_pose_shows_position(self) -> None:
        body = _FakeBodyWithPose("milk_bottle", 1.2, 0.5, 0.8)
        md = _make_match_data("PickUpAction", slots=[
            _make_slot("object_designator", body, FieldKind.ENTITY),
        ])
        text = render_resolved_slots(md)
        assert "1.20m" in text
        assert "0.50m" in text
        assert "0.80m" in text

    def test_type_ref_slot_shows_class_name(self) -> None:
        md = _make_match_data("PickUpAction", slots=[
            _make_slot("symbol_type", _FakeBody, FieldKind.TYPE_REF),
        ])
        text = render_resolved_slots(md)
        assert "_FakeBody" in text
        assert "(type)" in text

    def test_complex_slot_skipped(self) -> None:
        """COMPLEX top-level slots are omitted; their sub-fields appear separately."""
        md = _make_match_data("PickUpAction", slots=[
            _make_slot("grasp_description", object(), FieldKind.COMPLEX),
            _make_slot("grasp_description.grasp_type", "TOP", FieldKind.PRIMITIVE),
        ])
        text = render_resolved_slots(md)
        assert "grasp_description.grasp_type" in text
        # The top-level COMPLEX slot name alone should not appear as its own entry
        lines = [l for l in text.splitlines() if l.strip().startswith("- grasp_description:")]
        assert lines == []

    def test_free_slots_omitted(self) -> None:
        """Free (unresolved) slots are not included in the output."""
        md = _make_match_data("PickUpAction", slots=[
            _make_slot("arm", _FakeEnum.LEFT, FieldKind.PRIMITIVE, is_free=False),
            _make_slot("timeout", ..., FieldKind.PRIMITIVE, is_free=True),
        ])
        text = render_resolved_slots(md)
        assert "arm" in text
        assert "timeout" not in text

    def test_action_name_appears_in_header(self) -> None:
        md = _make_match_data("NavigateAction", slots=[
            _make_slot("speed", "0.5", FieldKind.PRIMITIVE),
        ])
        text = render_resolved_slots(md)
        assert "NavigateAction" in text


# ── TestFlanaganReasonerAnnotate ──────────────────────────────────────────────


def _default_plan_response() -> _ObjectAwarePhasePlanner:
    return _ObjectAwarePhasePlanner(phases=[
        _PhaseStep(phase="Grasp", target_object="milk", description="grip the milk"),
        _PhaseStep(phase="Lift", target_object="milk", description="lift off table"),
    ])


def _default_annotation_response() -> _AnnotatedPlan:
    return _AnnotatedPlan(phases=[
        _PhaseAnnotation(
            phase_key="Grasp_milk",
            preconditions=[_ConditionEntry(key="gripper_open", value=True)],
            goal_state=[_ConditionEntry(key="milk_grasped", value=True)],
            force_dynamics=_ForceDynamics(
                contact=True, motion_type="pinch", force_exerted="5N"
            ),
            sensory_feedback=[_SensorSignal(signal_name="contact_detected", value=True)],
            possible_failures=["milk slips"],
            recovery_strategies=["reposition gripper"],
            max_duration_sec=2.5,
            urgency="medium",
        ),
        _PhaseAnnotation(
            phase_key="Lift_milk",
            preconditions=[_ConditionEntry(key="milk_grasped", value=True)],
            goal_state=[_ConditionEntry(key="milk_lifted", value=True)],
            force_dynamics=_ForceDynamics(
                contact=True, motion_type="vertical_lift", force_exerted="5N"
            ),
            sensory_feedback=[_SensorSignal(signal_name="arm_height_m", value=0.3)],
            possible_failures=["grip slippage"],
            recovery_strategies=["increase grip force"],
            max_duration_sec=2.0,
            urgency="low",
        ),
    ])


class TestFlanaganReasonerAnnotate:
    """FlanaganReasoner.annotate() integration tests with a mock LLM."""

    def test_skips_when_no_instruction(self) -> None:
        """annotate() is a no-op when semantics.instruction is empty."""
        llm, recorded = _make_sequential_llm([_default_plan_response()])
        reasoner = FlanaganReasoner(llm=llm)
        semantics = ActionAnnotationBundle(action_type="PickUpAction", instruction=None)
        md = _make_match_data("PickUpAction", [])

        reasoner.annotate(semantics, md, world_context="")

        assert semantics.motion_phases is None
        assert len(recorded) == 0

    def test_makes_exactly_two_llm_calls(self) -> None:
        """annotate() invokes with_structured_output twice: Call 1 and Call 2."""
        llm, recorded = _make_sequential_llm([
            _default_plan_response(),
            _default_annotation_response(),
        ])
        reasoner = FlanaganReasoner(llm=llm)
        md = _make_match_data("PickUpAction", [])

        reasoner.annotate(_minimal_semantics(), md, world_context="test scene")

        assert len(recorded) == 2

    def test_sets_motion_phases_on_semantics(self) -> None:
        """annotate() populates semantics.motion_phases with a FlanaganMotionPlan."""
        llm, _ = _make_sequential_llm([
            _default_plan_response(),
            _default_annotation_response(),
        ])
        reasoner = FlanaganReasoner(llm=llm)
        semantics = _minimal_semantics()
        md = _make_match_data("PickUpAction", [])

        reasoner.annotate(semantics, md, world_context="")

        assert semantics.motion_phases is not None
        assert semantics.motion_phases.instruction == "pick up the milk"
        assert len(semantics.motion_phases.phases) == 2

    def test_world_context_appears_in_annotation_prompt(self) -> None:
        """world_context is passed to Call 2 (annotation), not just Call 1 (decompose)."""
        world_ctx = "kitchen scene: milk_bottle at (1.2m, 0.5m, 0.8m)"
        llm, recorded = _make_sequential_llm([
            _default_plan_response(),
            _default_annotation_response(),
        ])
        reasoner = FlanaganReasoner(llm=llm)
        md = _make_match_data("PickUpAction", [])

        reasoner.annotate(_minimal_semantics(), md, world_context=world_ctx)

        annotation_prompt = _prompt_text(recorded, call_index=1)
        assert world_ctx in annotation_prompt

    def test_resolved_slots_appear_in_annotation_prompt(self) -> None:
        """Resolved match slots are formatted and injected into the annotation prompt."""
        llm, recorded = _make_sequential_llm([
            _default_plan_response(),
            _default_annotation_response(),
        ])
        reasoner = FlanaganReasoner(llm=llm)
        md = _make_match_data("PickUpAction", slots=[
            _make_slot("arm", _FakeEnum.LEFT, FieldKind.PRIMITIVE),
        ])

        reasoner.annotate(_minimal_semantics(), md, world_context="")

        annotation_prompt = _prompt_text(recorded, call_index=1)
        assert "LEFT" in annotation_prompt

    def test_no_world_context_placeholder_injected(self) -> None:
        """When world_context is empty, a placeholder is injected so the prompt stays valid."""
        llm, recorded = _make_sequential_llm([
            _default_plan_response(),
            _default_annotation_response(),
        ])
        reasoner = FlanaganReasoner(llm=llm)
        md = _make_match_data("PickUpAction", [])

        reasoner.annotate(_minimal_semantics(), md, world_context="")

        annotation_prompt = _prompt_text(recorded, call_index=1)
        assert "(no world context available)" in annotation_prompt

    def test_phases_composed_correctly_from_annotation(self) -> None:
        """MotionPhase fields are populated from _PhaseAnnotation lookup by phase_key."""
        llm, _ = _make_sequential_llm([
            _default_plan_response(),
            _default_annotation_response(),
        ])
        reasoner = FlanaganReasoner(llm=llm)
        semantics = _minimal_semantics()
        md = _make_match_data("PickUpAction", [])

        reasoner.annotate(semantics, md, world_context="")

        grasp_phase = semantics.motion_phases.phases[0]
        assert grasp_phase.phase == "Grasp"
        assert grasp_phase.target_object == "milk"
        assert grasp_phase.preconditions == {"gripper_open": True}
        assert grasp_phase.goal_state == {"milk_grasped": True}
        assert grasp_phase.force_dynamics["contact"] is True
        assert grasp_phase.force_dynamics["motion_type"] == "pinch"
        assert grasp_phase.sensory_feedback == {"contact_detected": True}
        assert grasp_phase.failure_and_recovery["possible_failures"] == ["milk slips"]
        assert grasp_phase.temporal_constraints["max_duration_sec"] == 2.5
        assert grasp_phase.temporal_constraints["urgency"] == "medium"

    def test_symbol_field_generated_from_phase_and_object(self) -> None:
        """symbol is auto-generated as '->[ robot <verb>s <object>]'."""
        llm, _ = _make_sequential_llm([
            _default_plan_response(),
            _default_annotation_response(),
        ])
        reasoner = FlanaganReasoner(llm=llm)
        semantics = _minimal_semantics()

        reasoner.annotate(semantics, _make_match_data("PickUpAction", []), world_context="")

        assert semantics.motion_phases.phases[0].symbol == "->[ robot grasps milk]"
        assert semantics.motion_phases.phases[1].symbol == "->[ robot lifts milk]"

    def test_annotation_failure_produces_empty_dicts(self) -> None:
        """When Call 2 fails, phases are present but have empty annotation dicts."""
        plan = _default_plan_response()

        counter = [0]
        recorded: List[Any] = []

        def _wso(schema: Any, **kwargs: Any) -> RunnableLambda:
            call_number = counter[0]
            counter[0] += 1

            def _invoke(prompt_value: Any, **kw: Any) -> Any:
                recorded.append(prompt_value)
                if call_number == 0:
                    return plan
                raise RuntimeError("simulated annotation failure")

            return RunnableLambda(_invoke)

        llm = MagicMock()
        llm.with_structured_output.side_effect = _wso

        reasoner = FlanaganReasoner(llm=llm)
        semantics = _minimal_semantics()

        reasoner.annotate(semantics, _make_match_data("PickUpAction", []), world_context="")

        assert semantics.motion_phases is not None
        assert len(semantics.motion_phases.phases) == 2
        for phase in semantics.motion_phases.phases:
            assert phase.preconditions == {}
            assert phase.goal_state == {}
            assert phase.force_dynamics == {}
            assert phase.sensory_feedback == {}
            assert phase.failure_and_recovery == {}
            assert phase.temporal_constraints == {}

    def test_phase_key_normalised_in_composition(self) -> None:
        """Annotation is looked up by phase_key = '{normalized_phase}_{object}'."""
        plan = _ObjectAwarePhasePlanner(phases=[
            _PhaseStep(phase="grab", target_object="cup"),  # synonym → Grasp
        ])
        annotation = _AnnotatedPlan(phases=[
            _PhaseAnnotation(
                phase_key="Grasp_cup",  # uses canonical, normalised name
                goal_state=[_ConditionEntry(key="cup_grasped", value=True)],
            )
        ])
        llm, _ = _make_sequential_llm([plan, annotation])
        reasoner = FlanaganReasoner(llm=llm)
        semantics = _minimal_semantics()

        reasoner.annotate(semantics, _make_match_data("PickUpAction", []), world_context="")

        phase = semantics.motion_phases.phases[0]
        assert phase.phase == "Grasp"
        assert phase.goal_state == {"cup_grasped": True}

    def test_previous_actions_passed_to_call1(self) -> None:
        """previous_actions constructor arg is forwarded to the decompose prompt."""
        llm, recorded = _make_sequential_llm([
            _default_plan_response(),
            _default_annotation_response(),
        ])
        reasoner = FlanaganReasoner(
            llm=llm, previous_actions=["pick up the cup from the table"]
        )

        reasoner.annotate(
            _minimal_semantics("place it in the sink"),
            _make_match_data("PlaceAction", []),
            world_context="",
        )

        decompose_prompt = _prompt_text(recorded, call_index=0)
        assert "pick up the cup from the table" in decompose_prompt
