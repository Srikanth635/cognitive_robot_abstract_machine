"""Tests for llmr.hypotheses.projection."""

from __future__ import annotations

from dataclasses import dataclass

from llmr.hypotheses import (
    ClaimStatus,
    FrameHypothesisNode,
    GroundingState,
    HypothesisGraph,
    ProjectionOrchestrator,
    HypothesisMeta,
    HypothesisProjection,
    ProjectionInput,
    HypothesisProjector,
    ProjectorRegistry,
    InstructionNode,
)


@dataclass
class _FakeAction:
    name: str = "action"


@dataclass
class _FakeMatchData:
    action_type: type
    action_name: str
    slots: list[object]
    _expression: object = object()


@dataclass
class _FakeSemantics:
    action_type: str
    instruction: str | None = None
    frames: object | None = None


def _meta(
    *,
    status: ClaimStatus = ClaimStatus.HYPOTHESIS,
    grounding: GroundingState = GroundingState.TEXT_ONLY,
) -> HypothesisMeta:
    return HypothesisMeta(
        source_reasoner="dummy",
        status=status,
        grounding=grounding,
    )


def _context() -> ProjectionInput:
    action = _FakeAction()
    return ProjectionInput(
        instruction="pick up the milk",
        action=action,
        action_type="PickUpAction",
        semantics=_FakeSemantics(action_type="PickUpAction", instruction="pick up the milk"),
        match_data=_FakeMatchData(action_type=type(action), action_name="PickUpAction", slots=[]),
        resolved_slots={},
        world_context="world",
        symbol_type=object,
        llm_model_name="gpt-test",
    )


class _ProjectorA(HypothesisProjector):
    REASONER_NAME = "a"

    def supports(self, context: ProjectionInput) -> bool:
        return True

    def project(self, context: ProjectionInput) -> HypothesisProjection:
        instruction = InstructionNode(
            id="i1",
            meta=_meta(),
            text=context.instruction or "",
            normalized_text=(context.instruction or "").lower(),
        )
        frame = FrameHypothesisNode(
            id="f1",
            meta=_meta(),
            frame="Getting",
            lexical_unit="pick_up.v",
            framenet_label="picking_up_object",
            action_type=context.action_type,
            instruction_text=context.instruction,
        )
        return HypothesisProjection(nodes=[instruction, frame], edges=[])


class _ProjectorB(HypothesisProjector):
    REASONER_NAME = "b"

    def supports(self, context: ProjectionInput) -> bool:
        return context.instruction == "pick up the milk"

    def project(self, context: ProjectionInput) -> HypothesisProjection:
        frame = FrameHypothesisNode(
            id="f2",
            meta=_meta(status=ClaimStatus.SUPPORTED),
            frame="Getting",
            lexical_unit="pick_up.v",
            framenet_label="pickup",
            action_type=context.action_type,
            instruction_text=context.instruction,
        )
        return HypothesisProjection(nodes=[frame], edges=[])


class _FailingProjector(HypothesisProjector):
    REASONER_NAME = "fail"

    def supports(self, context: ProjectionInput) -> bool:
        return True

    def project(self, context: ProjectionInput) -> HypothesisProjection:
        raise RuntimeError("boom")


class TestProjectionContext:
    def test_context_instantiates(self) -> None:
        context = _context()
        assert context.action_type == "PickUpAction"
        assert context.llm_model_name == "gpt-test"


class TestProjectionBundle:
    def test_projection_is_passive_bundle(self) -> None:
        projection = HypothesisProjection(nodes=[], edges=[], warnings=["x"])
        assert projection.warnings == ["x"]


class TestProjectorRegistry:
    def test_registry_preserves_order(self) -> None:
        registry = ProjectorRegistry()
        a = _ProjectorA()
        b = _ProjectorB()
        registry.register(a)
        registry.register(b)
        assert registry.projectors == [a, b]

    def test_matching_returns_only_supported_projectors(self) -> None:
        registry = ProjectorRegistry()
        a = _ProjectorA()
        b = _ProjectorB()
        registry.register(a)
        registry.register(b)
        assert registry.matching(_context()) == [a, b]

        other = ProjectionInput(
            instruction="other",
            action=_FakeAction(),
            action_type="PickUpAction",
            semantics=_FakeSemantics(action_type="PickUpAction", instruction="other"),
            match_data=_FakeMatchData(action_type=_FakeAction, action_name="PickUpAction", slots=[]),
            resolved_slots={},
            world_context="world",
            symbol_type=object,
        )
        assert registry.matching(other) == [a]


class TestGraphManager:
    def test_manager_inserts_single_projection(self) -> None:
        graph = HypothesisGraph()
        registry = ProjectorRegistry([_ProjectorA()])
        manager = ProjectionOrchestrator(graph=graph, registry=registry)
        manager.project(_context())
        assert len(graph.instructions) == 1
        assert len(graph.domain(FrameHypothesisNode)) == 1

    def test_manager_inserts_multiple_projections(self) -> None:
        graph = HypothesisGraph()
        registry = ProjectorRegistry([_ProjectorA(), _ProjectorB()])
        manager = ProjectionOrchestrator(graph=graph, registry=registry)
        manager.project(_context())
        assert len(graph.instructions) == 1
        assert len(graph.domain(FrameHypothesisNode)) == 2

    def test_manager_continues_after_one_projector_fails(self) -> None:
        graph = HypothesisGraph()
        registry = ProjectorRegistry([_FailingProjector(), _ProjectorA()])
        manager = ProjectionOrchestrator(graph=graph, registry=registry)
        manager.project(_context())
        assert len(graph.instructions) == 1
        assert len(graph.domain(FrameHypothesisNode)) == 1

    def test_manager_does_nothing_when_nothing_matches(self) -> None:
        class _NeverProjector(HypothesisProjector):
            REASONER_NAME = "never"

            def supports(self, context: ProjectionInput) -> bool:
                return False

            def project(self, context: ProjectionInput) -> HypothesisProjection:
                raise AssertionError("should never be called")

        graph = HypothesisGraph()
        registry = ProjectorRegistry([_NeverProjector()])
        manager = ProjectionOrchestrator(graph=graph, registry=registry)
        manager.project(_context())
        assert graph.instructions == []
        assert graph.domain(FrameHypothesisNode) == []
