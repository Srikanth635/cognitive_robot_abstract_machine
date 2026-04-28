"""Tests for the concrete FrameNet hypothesis projector."""

from __future__ import annotations

from dataclasses import dataclass

from llmr.hypotheses import (
    ClaimStatus,
    FrameNetGraphView,
    FrameNetProjector,
    FrameRoleHypothesisNode,
    GroundedByEdge,
    GroundingState,
    HasRoleEdge,
    HypothesisGraph,
    SlotBindingEvidenceNode,
    SupportedByEdge,
    SymbolGroundingEvidenceNode,
)
from llmr.hypotheses.projection import ProjectionInput


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
class _FakeCoreElements:
    theme: str | None = None
    source: str | None = None
    goal: str | None = None
    agent: str | None = None
    patient: str | None = None
    instrument: str | None = None
    result: str | None = None


@dataclass
class _FakePeripheralElements:
    direction: str | None = None
    location: str | None = None
    manner: str | None = None
    time: str | None = None
    purpose: str | None = None
    quantity: str | None = None
    portion: str | None = None
    speed: str | None = None
    path: str | None = None


@dataclass
class _FakeFrameNetRepresentation:
    framenet: str
    frame: str
    lexical_unit: str
    core: _FakeCoreElements
    peripheral: _FakePeripheralElements


@dataclass
class _FakeSemantics:
    action_type: str
    instruction: str | None = None
    frames: object | None = None


@dataclass
class _FakeSymbol:
    name: str


def _context(with_frames: bool = True) -> ProjectionInput:
    semantics = _FakeSemantics(
        action_type="PickUpAction",
        instruction="pick up the milk from the table",
        frames=_FakeFrameNetRepresentation(
            framenet="picking_up_object",
            frame="Getting",
            lexical_unit="pick_up.v",
            core=_FakeCoreElements(
                theme="milk",
                source="table",
                goal="robot grasp",
            ),
            peripheral=_FakePeripheralElements(direction="upward"),
        )
        if with_frames
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
        resolved_slots={
            "object_designator": _FakeSymbol("milk"),
            "support_surface": _FakeSymbol("table"),
        },
        world_context="world",
        symbol_type=object,
        llm_model_name="gpt-test",
    )


class TestFrameNetProjector:
    def test_supports_only_when_frames_exist(self) -> None:
        projector = FrameNetProjector()
        assert projector.supports(_context(with_frames=True)) is True
        assert projector.supports(_context(with_frames=False)) is False

    def test_project_builds_expected_frame_and_role_claims(self) -> None:
        projector = FrameNetProjector()
        projection = projector.project(_context())

        frame_nodes = [
            node for node in projection.nodes if node.__class__.__name__ == "FrameHypothesisNode"
        ]
        role_nodes = [
            node for node in projection.nodes if isinstance(node, FrameRoleHypothesisNode)
        ]

        assert len(frame_nodes) == 1
        assert len(role_nodes) == 4
        assert sorted(role.role_name for role in role_nodes) == [
            "direction",
            "goal",
            "source",
            "theme",
        ]

    def test_project_emits_support_and_grounding_evidence_for_resolved_entities(self) -> None:
        projector = FrameNetProjector()
        projection = projector.project(_context())

        slot_evidence = [
            node for node in projection.nodes if isinstance(node, SlotBindingEvidenceNode)
        ]
        grounding_evidence = [
            node
            for node in projection.nodes
            if isinstance(node, SymbolGroundingEvidenceNode)
        ]
        support_edges = [
            edge for edge in projection.edges if isinstance(edge, SupportedByEdge)
        ]
        grounding_edges = [
            edge for edge in projection.edges if isinstance(edge, GroundedByEdge)
        ]

        assert len(slot_evidence) == 2
        assert len(grounding_evidence) == 2
        assert len(support_edges) == 2
        assert len(grounding_edges) == 2

    def test_role_status_distinguishes_grounded_and_hypothesis_only_roles(self) -> None:
        projector = FrameNetProjector()
        graph = HypothesisGraph()
        projection = projector.project(_context())
        graph.add_projection(projection)
        view = FrameNetGraphView(graph)

        role_by_name = {role.role_name: role for role in view.roles()}

        assert role_by_name["theme"].meta.status == ClaimStatus.SUPPORTED
        assert (
            role_by_name["theme"].meta.grounding == GroundingState.SYMBOL_GROUNDED
        )
        assert role_by_name["source"].meta.status == ClaimStatus.SUPPORTED
        assert (
            role_by_name["source"].meta.grounding == GroundingState.SYMBOL_GROUNDED
        )
        assert role_by_name["goal"].meta.status == ClaimStatus.HYPOTHESIS
        assert role_by_name["goal"].meta.grounding == GroundingState.TEXT_ONLY
        assert role_by_name["direction"].meta.status == ClaimStatus.HYPOTHESIS
        assert role_by_name["direction"].meta.grounding == GroundingState.TEXT_ONLY

    def test_full_projection_inserts_cleanly_into_graph(self) -> None:
        projector = FrameNetProjector()
        graph = HypothesisGraph()
        projection = projector.project(_context())
        graph.add_projection(projection)
        view = FrameNetGraphView(graph)

        assert len(graph.instructions) == 1
        assert len(graph.actions) == 1
        assert len(graph.reasoner_runs) == 1
        assert len(view.frames()) == 1
        assert len(view.roles()) == 4
        assert len(graph.edge_domain(HasRoleEdge)) == 4
