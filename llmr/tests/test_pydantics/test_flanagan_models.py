"""Tests for Flanagan motion-phase Pydantic models."""

import pytest
from pydantic import ValidationError

from llmr.workflows.pydantics.flanagan_models import (
    FlanaganState,
    ForceDynamics,
    ForceDynamicsMap,
    ForceProfile,
    ObjectAwarePhasePlanner,
    PhaseReasoningOutput,
    PhaseStep,
)


class TestPhaseStep:
    def test_valid_phase(self) -> None:
        step = PhaseStep(phase="Grasp", target_object="cup")
        assert step.phase == "Grasp"
        assert step.target_object == "cup"

    def test_invalid_phase_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PhaseStep(phase="InvalidPhase", target_object="cup")

    def test_description_optional(self) -> None:
        step = PhaseStep(phase="Lift", target_object="box")
        assert step.description is None

    def test_description_provided(self) -> None:
        step = PhaseStep(phase="Approach", target_object="table", description="Move to table")
        assert step.description == "Move to table"

    def test_all_valid_phases(self) -> None:
        valid_phases = [
            "Approach", "Grasp", "Lift", "Transport", "Place", "Align",
            "Cut", "Slice", "Tilt", "Pour", "Insert", "Withdraw",
            "Release", "Reorient", "Stabilize", "Inspect",
        ]
        for phase in valid_phases:
            step = PhaseStep(phase=phase, target_object="obj")
            assert step.phase == phase

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            PhaseStep(phase="Grasp", target_object="cup", unknown="x")


class TestObjectAwarePhasePlanner:
    def test_phases_list(self) -> None:
        planner = ObjectAwarePhasePlanner(
            phases=[
                PhaseStep(phase="Approach", target_object="drawer"),
                PhaseStep(phase="Grasp", target_object="handle"),
            ]
        )
        assert len(planner.phases) == 2
        assert planner.phases[0].phase == "Approach"


class TestForceDynamics:
    def test_construction(self) -> None:
        fd = ForceDynamics(
            contact=True,
            motion_type="gripper_closure",
            force_exerted="firm_grip",
            force_profile=ForceProfile(type="compression", expected_range_N=[5.0, 10.0]),
        )
        assert fd.contact is True
        assert fd.force_profile.expected_range_N == [5.0, 10.0]

    def test_no_contact(self) -> None:
        fd = ForceDynamics(
            contact=False,
            motion_type="linear_translation",
            force_exerted="minimal",
            force_profile=ForceProfile(type="motion_control", expected_range_N=[0, 2]),
        )
        assert fd.contact is False

    def test_force_profile_optional_fields(self) -> None:
        fp = ForceProfile(type="torque")
        assert fp.expected_range_N is None
        assert fp.expected_range_Nm is None


class TestPhaseReasoningOutput:
    def test_valid_construction(self) -> None:
        output = PhaseReasoningOutput(
            reasoning="Pick up requires approach then grasp.",
            sub_goals=["move to object", "grasp object", "lift object"],
            phases=["Approach", "Grasp", "Lift"],
        )
        assert output.reasoning.startswith("Pick up")
        assert len(output.phases) == 3

    def test_invalid_phase_in_reasoning_output(self) -> None:
        with pytest.raises(ValidationError):
            PhaseReasoningOutput(
                reasoning="test",
                sub_goals=[],
                phases=["FakePhase"],
            )
