"""Tests for hypothesis graph views."""

from __future__ import annotations

from llmr.hypotheses import (
    AboutActionEdge,
    ActionNode,
    ClaimStatus,
    EvokesMotionPlanEdge,
    EvokesFrameEdge,
    FlanaganGraphView,
    FrameHypothesisNode,
    FrameNetGraphView,
    FrameRoleHypothesisNode,
    GroundedByEdge,
    GroundingState,
    HasMotionPhaseEdge,
    HasRoleEdge,
    HypothesisGraph,
    HypothesisGraphView,
    HypothesisMeta,
    InstructionNode,
    MotionPhaseHypothesisNode,
    MotionPlanHypothesisNode,
    ProducedClaimEdge,
    ReasonerRunNode,
    SlotBindingEvidenceNode,
    SupportedByEdge,
    SymbolGroundingEvidenceNode,
)


class _FakeAction:
    pass


class _FakeSymbol:
    def __init__(self, name: str) -> None:
        self.name = name


def _meta(
    *,
    source_reasoner: str = "framenet_reasoner",
    status: ClaimStatus = ClaimStatus.HYPOTHESIS,
    grounding: GroundingState = GroundingState.TEXT_ONLY,
    run_id: str | None = None,
) -> HypothesisMeta:
    return HypothesisMeta(
        source_reasoner=source_reasoner,
        status=status,
        grounding=grounding,
        run_id=run_id,
    )


def _sample_graph() -> tuple[HypothesisGraph, dict[str, object]]:
    graph = HypothesisGraph()
    action = _FakeAction()
    milk = _FakeSymbol("milk")

    instruction = InstructionNode(
        id="i1",
        meta=_meta(),
        text="pick up the milk from the table",
        normalized_text="pick up the milk from the table",
    )
    action_node = ActionNode(
        id="a1",
        meta=_meta(),
        action_ref=action,
        action_type="PickUpAction",
    )
    run = ReasonerRunNode(
        id="run1",
        meta=_meta(run_id="run1"),
        reasoner_name="framenet_reasoner",
        run_id="run1",
        model_name="gpt-test",
        prompt_version="framenet_v1",
        action_type="PickUpAction",
        instruction_text="pick up the milk from the table",
    )
    frame = FrameHypothesisNode(
        id="f1",
        meta=_meta(run_id="run1"),
        frame="Getting",
        lexical_unit="pick_up.v",
        framenet_label="picking_up_object",
        action_type="PickUpAction",
        instruction_text="pick up the milk from the table",
    )
    theme = FrameRoleHypothesisNode(
        id="r1",
        meta=_meta(
            status=ClaimStatus.SUPPORTED,
            grounding=GroundingState.SYMBOL_GROUNDED,
            run_id="run1",
        ),
        role_family="core",
        role_name="theme",
        filler_text="milk",
        filler_kind="entity",
    )
    goal = FrameRoleHypothesisNode(
        id="r2",
        meta=_meta(run_id="run1"),
        role_family="core",
        role_name="goal",
        filler_text="robot grasp",
        filler_kind="abstract",
    )
    slot_ev = SlotBindingEvidenceNode(
        id="ev1",
        meta=_meta(
            status=ClaimStatus.SUPPORTED,
            grounding=GroundingState.SLOT_ALIGNED,
            run_id="run1",
        ),
        slot_name="object_designator",
        value_ref=milk,
        value_repr="milk",
    )
    grounding_ev = SymbolGroundingEvidenceNode(
        id="ev2",
        meta=_meta(
            status=ClaimStatus.SUPPORTED,
            grounding=GroundingState.SYMBOL_GROUNDED,
            run_id="run1",
        ),
        query_text="milk",
        symbol_ref=milk,
        symbol_type="_FakeSymbol",
        grounding_method="resolved_slot_symbol_match",
    )

    for node in [instruction, action_node, run, frame, theme, goal, slot_ev, grounding_ev]:
        graph.add_node(node)

    for edge in [
        EvokesFrameEdge(id="e1", meta=_meta(run_id="run1"), src_id="i1", dst_id="f1"),
        AboutActionEdge(id="e2", meta=_meta(run_id="run1"), src_id="f1", dst_id="a1"),
        ProducedClaimEdge(id="e3", meta=_meta(run_id="run1"), src_id="run1", dst_id="f1"),
        ProducedClaimEdge(id="e4", meta=_meta(run_id="run1"), src_id="run1", dst_id="r1"),
        ProducedClaimEdge(id="e5", meta=_meta(run_id="run1"), src_id="run1", dst_id="r2"),
        HasRoleEdge(id="e6", meta=_meta(run_id="run1"), src_id="f1", dst_id="r1"),
        HasRoleEdge(id="e7", meta=_meta(run_id="run1"), src_id="f1", dst_id="r2"),
        SupportedByEdge(id="e8", meta=_meta(run_id="run1"), src_id="r1", dst_id="ev1"),
        GroundedByEdge(id="e9", meta=_meta(run_id="run1"), src_id="r1", dst_id="ev2"),
    ]:
        graph.add_edge(edge)

    return graph, {"action": action}


def _sample_flanagan_graph() -> tuple[HypothesisGraph, dict[str, object]]:
    graph = HypothesisGraph()
    action = _FakeAction()

    instruction = InstructionNode(
        id="i1",
        meta=_meta(source_reasoner="flanagan_reasoner"),
        text="pick up the milk from the table",
        normalized_text="pick up the milk from the table",
    )
    action_node = ActionNode(
        id="a1",
        meta=_meta(source_reasoner="flanagan_reasoner"),
        action_ref=action,
        action_type="PickUpAction",
    )
    run = ReasonerRunNode(
        id="run1",
        meta=_meta(source_reasoner="flanagan_reasoner", run_id="run1"),
        reasoner_name="flanagan_reasoner",
        run_id="run1",
        model_name="gpt-test",
        prompt_version="flanagan_v1",
        action_type="PickUpAction",
        instruction_text="pick up the milk from the table",
    )
    plan = MotionPlanHypothesisNode(
        id="p1",
        meta=_meta(source_reasoner="flanagan_reasoner", run_id="run1"),
        action_type="PickUpAction",
        instruction_text="pick up the milk from the table",
        phase_count=2,
    )
    approach = MotionPhaseHypothesisNode(
        id="mp1",
        meta=_meta(source_reasoner="flanagan_reasoner", run_id="run1"),
        phase_index=0,
        phase_name="Approach",
        target_object="milk",
        description="move toward the milk",
        symbol="->[ robot approaches milk]",
        temporal_constraints={"max_duration_sec": 2.0, "urgency": "low"},
        max_duration_sec=2.0,
        urgency="low",
    )
    grasp = MotionPhaseHypothesisNode(
        id="mp2",
        meta=_meta(source_reasoner="flanagan_reasoner", run_id="run1"),
        phase_index=1,
        phase_name="Grasp",
        target_object="milk",
        description="grip the milk",
        symbol="->[ robot grasps milk]",
        force_dynamics={"contact": True, "motion_type": "pinch"},
        failure_and_recovery={
            "possible_failures": ["slip"],
            "recovery_strategies": ["retry grasp"],
        },
        temporal_constraints={"max_duration_sec": 2.5, "urgency": "high"},
        contact=True,
        motion_type="pinch",
        max_duration_sec=2.5,
        urgency="high",
        possible_failures=("slip",),
        recovery_strategies=("retry grasp",),
    )

    for node in [instruction, action_node, run, plan, approach, grasp]:
        graph.add_node(node)

    for edge in [
        EvokesMotionPlanEdge(
            id="e1",
            meta=_meta(source_reasoner="flanagan_reasoner", run_id="run1"),
            src_id="i1",
            dst_id="p1",
        ),
        AboutActionEdge(
            id="e2",
            meta=_meta(source_reasoner="flanagan_reasoner", run_id="run1"),
            src_id="p1",
            dst_id="a1",
        ),
        ProducedClaimEdge(
            id="e3",
            meta=_meta(source_reasoner="flanagan_reasoner", run_id="run1"),
            src_id="run1",
            dst_id="p1",
        ),
        ProducedClaimEdge(
            id="e4",
            meta=_meta(source_reasoner="flanagan_reasoner", run_id="run1"),
            src_id="run1",
            dst_id="mp1",
        ),
        ProducedClaimEdge(
            id="e5",
            meta=_meta(source_reasoner="flanagan_reasoner", run_id="run1"),
            src_id="run1",
            dst_id="mp2",
        ),
        HasMotionPhaseEdge(
            id="e6",
            meta=_meta(source_reasoner="flanagan_reasoner", run_id="run1"),
            src_id="p1",
            dst_id="mp1",
        ),
        HasMotionPhaseEdge(
            id="e7",
            meta=_meta(source_reasoner="flanagan_reasoner", run_id="run1"),
            src_id="p1",
            dst_id="mp2",
        ),
    ]:
        graph.add_edge(edge)

    return graph, {"action": action}


class TestHypothesisGraphView:
    def test_generic_view_exposes_typed_domains(self) -> None:
        graph, _ = _sample_graph()
        view = HypothesisGraphView(graph)

        assert len(view.nodes(FrameHypothesisNode)) == 1
        assert len(view.edges(HasRoleEdge)) == 2

    def test_generic_view_returns_same_view_type_for_run_subgraph(self) -> None:
        graph, _ = _sample_graph()
        view = HypothesisGraphView(graph)

        subgraph = view.subgraph_for_run("run1")
        assert isinstance(subgraph, HypothesisGraphView)
        assert subgraph.graph.node_count == graph.subgraph_for_run("run1").node_count


class TestFrameNetGraphView:
    def test_claim_accessors(self) -> None:
        graph, refs = _sample_graph()
        view = FrameNetGraphView(graph)

        assert len(view.claims()) == 3
        assert len(view.root_claims()) == 1
        assert len(view.claims_for_run("run1")) == 3
        assert len(view.claims_for_action(refs["action"])) == 3

    def test_frames_and_roles(self) -> None:
        graph, _ = _sample_graph()
        view = FrameNetGraphView(graph)

        assert [frame.id for frame in view.frames()] == ["f1"]
        assert [role.id for role in view.roles()] == ["r1", "r2"]
        assert [frame.id for frame in view.frames_by_frame("Getting")] == ["f1"]
        assert [role.id for role in view.roles_by_role_name("theme")] == ["r1"]

    def test_roles_for_frame(self) -> None:
        graph, _ = _sample_graph()
        view = FrameNetGraphView(graph)
        frame = view.frames()[0]

        assert [role.id for role in view.roles_for_frame(frame)] == ["r1", "r2"]

    def test_role_status_views(self) -> None:
        graph, _ = _sample_graph()
        view = FrameNetGraphView(graph)

        assert [role.id for role in view.grounded_roles()] == ["r1"]
        assert [role.id for role in view.supported_roles()] == ["r1"]
        assert [role.id for role in view.hypothesis_only_roles()] == ["r2"]

    def test_action_subgraph_returns_same_view_type(self) -> None:
        graph, refs = _sample_graph()
        view = FrameNetGraphView(graph)

        subgraph = view.action_subgraph(refs["action"])
        assert isinstance(subgraph, FrameNetGraphView)
        assert [frame.id for frame in subgraph.frames()] == ["f1"]


class TestFlanaganGraphView:
    def test_claim_accessors(self) -> None:
        graph, refs = _sample_flanagan_graph()
        view = FlanaganGraphView(graph)

        assert len(view.claims()) == 3
        assert len(view.root_claims()) == 1
        assert len(view.claims_for_run("run1")) == 3
        assert len(view.claims_for_action(refs["action"])) == 3

    def test_plans_and_phases(self) -> None:
        graph, _ = _sample_flanagan_graph()
        view = FlanaganGraphView(graph)

        assert [plan.id for plan in view.plans()] == ["p1"]
        assert [phase.id for phase in view.phases()] == ["mp1", "mp2"]
        assert [phase.id for phase in view.phases_by_name("Grasp")] == ["mp2"]
        assert [phase.id for phase in view.phases_by_target_object("milk")] == [
            "mp1",
            "mp2",
        ]

    def test_phases_for_plan_and_filters(self) -> None:
        graph, _ = _sample_flanagan_graph()
        view = FlanaganGraphView(graph)
        plan = view.plans()[0]

        assert [phase.id for phase in view.phases_for_plan(plan)] == ["mp1", "mp2"]
        assert [phase.id for phase in view.contact_phases()] == ["mp2"]
        assert [phase.id for phase in view.phases_with_failures()] == ["mp2"]
        assert [phase.id for phase in view.high_urgency_phases()] == ["mp2"]

    def test_action_subgraph_returns_same_view_type(self) -> None:
        graph, refs = _sample_flanagan_graph()
        view = FlanaganGraphView(graph)

        subgraph = view.action_subgraph(refs["action"])
        assert isinstance(subgraph, FlanaganGraphView)
        assert [plan.id for plan in subgraph.plans()] == ["p1"]
