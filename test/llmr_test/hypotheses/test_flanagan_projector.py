"""Tests for the concrete Flanagan hypothesis projector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Any

from llmr.hypotheses import (
    FlanaganGraphView,
    FlanaganProjector,
    HasMotionPhaseEdge,
    HypothesisGraph,
    MotionPhaseHypothesisNode,
    MotionPlanHypothesisNode,
    ProjectionInput,
)


@dataclass
class _FakeAction:
    name: str = "pickup"


@dataclass
class _FakeMatchData:
    action_type: type
    action_name: str
    slots: list[object]
    _expression: object = object()


@dataclass
class _FakeMotionPhase:
    phase: str
    target_object: str
    description: str | None = None
    symbol: str = ""
    preconditions: dict[str, Any] = field(default_factory=dict)
    goal_state: dict[str, Any] = field(default_factory=dict)
    force_dynamics: dict[str, Any] = field(default_factory=dict)
    sensory_feedback: dict[str, Any] = field(default_factory=dict)
    failure_and_recovery: dict[str, Any] = field(default_factory=dict)
    temporal_constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeMotionPlan:
    instruction: str
    phases: list[_FakeMotionPhase]


@dataclass
class _FakeSemantics:
    action_type: str
    instruction: str | None = None
    motion_phases: object | None = None


def _context(with_motion_plan: bool = True) -> ProjectionInput:
    semantics = _FakeSemantics(
        action_type="PickUpAction",
        instruction="pick up the milk from the table",
        motion_phases=_FakeMotionPlan(
            instruction="pick up the milk from the table",
            phases=[
                _FakeMotionPhase(
                    phase="Approach",
                    target_object="milk",
                    description="move toward the milk",
                    symbol="->[ robot approaches milk]",
                    temporal_constraints={"max_duration_sec": 2.0, "urgency": "low"},
                ),
                _FakeMotionPhase(
                    phase="Grasp",
                    target_object="milk",
                    description="grip the milk",
                    symbol="->[ robot grasps milk]",
                    preconditions={"gripper_open": True},
                    goal_state={"milk_grasped": True},
                    force_dynamics={"contact": True, "motion_type": "pinch"},
                    sensory_feedback={"LEFT_force_sensor_N": 6.0},
                    failure_and_recovery={
                        "possible_failures": ["slip"],
                        "recovery_strategies": ["retry grasp"],
                    },
                    temporal_constraints={"max_duration_sec": 2.5, "urgency": "high"},
                ),
            ],
        )
        if with_motion_plan
        else None,
    )
    return ProjectionInput(
        instruction="pick up the milk from the table",
        action=_FakeAction(),
        action_type="PickUpAction",
        semantics=semantics,
        match_data=_FakeMatchData(
            action_type=_FakeAction,
            action_name="PickUpAction",
            slots=[],
        ),
        resolved_slots={},
        world_context="world",
        symbol_type=object,
        llm_model_name="gpt-test",
    )


class TestFlanaganProjector:
    def test_supports_only_when_motion_plan_exists(self) -> None:
        projector = FlanaganProjector()
        assert projector.supports(_context(with_motion_plan=True)) is True
        assert projector.supports(_context(with_motion_plan=False)) is False

    def test_project_builds_plan_and_phase_claims(self) -> None:
        projector = FlanaganProjector()
        projection = projector.project(_context())

        plan_nodes = [
            node for node in projection.nodes if isinstance(node, MotionPlanHypothesisNode)
        ]
        phase_nodes = [
            node for node in projection.nodes if isinstance(node, MotionPhaseHypothesisNode)
        ]

        assert len(plan_nodes) == 1
        assert len(phase_nodes) == 2
        assert [phase.phase_name for phase in phase_nodes] == ["Approach", "Grasp"]

    def test_project_preserves_query_friendly_phase_fields(self) -> None:
        projector = FlanaganProjector()
        graph = HypothesisGraph()
        graph.add_projection(projector.project(_context()))
        view = FlanaganGraphView(graph)

        phase_by_name = {phase.phase_name: phase for phase in view.phases()}
        grasp = phase_by_name["Grasp"]

        assert grasp.contact is True
        assert grasp.motion_type == "pinch"
        assert grasp.max_duration_sec == 2.5
        assert grasp.urgency == "high"
        assert grasp.possible_failures == ("slip",)
        assert grasp.recovery_strategies == ("retry grasp",)

    def test_full_projection_inserts_cleanly_into_graph(self) -> None:
        projector = FlanaganProjector()
        graph = HypothesisGraph()
        graph.add_projection(projector.project(_context()))
        view = FlanaganGraphView(graph)

        assert len(graph.instructions) == 1
        assert len(graph.actions) == 1
        assert len(graph.reasoner_runs) == 1
        assert len(view.plans()) == 1
        assert len(view.phases()) == 2
        assert len(graph.edge_domain(HasMotionPhaseEdge)) == 2
