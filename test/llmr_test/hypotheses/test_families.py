"""Tests for hypothesis family descriptors."""

from __future__ import annotations

from llmr.hypotheses import (
    FlanaganFamily,
    FlanaganGraphView,
    FlanaganProjector,
    FrameNetFamily,
    FrameNetGraphView,
    FrameNetProjector,
    HypothesisFamily,
    HypothesisGraph,
    ProjectionOrchestrator,
    ProjectorRegistry,
    get_all_families,
    hypothesis_family,
)
from llmr.hypotheses.projectors.framenet.constants import FRAMENET_REASONER_NAME
from llmr.hypotheses.projectors.flanagan.constants import FLANAGAN_REASONER_NAME


class TestFamilyBase:
    def test_framenet_family_is_hypothesis_family(self) -> None:
        assert issubclass(FrameNetFamily, HypothesisFamily)

    def test_flanagan_family_is_hypothesis_family(self) -> None:
        assert issubclass(FlanaganFamily, HypothesisFamily)

    def test_make_projector(self) -> None:
        projector = FrameNetFamily.make_projector()
        assert isinstance(projector, FrameNetProjector)

    def test_make_flanagan_projector(self) -> None:
        projector = FlanaganFamily.make_projector()
        assert isinstance(projector, FlanaganProjector)

    def test_make_view(self) -> None:
        view = FrameNetFamily.make_view(HypothesisGraph())
        assert isinstance(view, FrameNetGraphView)

    def test_make_flanagan_view(self) -> None:
        view = FlanaganFamily.make_view(HypothesisGraph())
        assert isinstance(view, FlanaganGraphView)

    def test_make_registry(self) -> None:
        registry = FrameNetFamily.make_registry()
        assert isinstance(registry, ProjectorRegistry)
        assert len(registry.projectors) == 1
        assert isinstance(registry.projectors[0], FrameNetProjector)

    def test_make_flanagan_registry(self) -> None:
        registry = FlanaganFamily.make_registry()
        assert isinstance(registry, ProjectorRegistry)
        assert len(registry.projectors) == 1
        assert isinstance(registry.projectors[0], FlanaganProjector)

    def test_make_manager(self) -> None:
        manager = FrameNetFamily.make_manager(HypothesisGraph())
        assert isinstance(manager, ProjectionOrchestrator)
        assert len(manager.registry.projectors) == 1
        assert isinstance(manager.registry.projectors[0], FrameNetProjector)

    def test_make_flanagan_manager(self) -> None:
        manager = FlanaganFamily.make_manager(HypothesisGraph())
        assert isinstance(manager, ProjectionOrchestrator)
        assert len(manager.registry.projectors) == 1
        assert isinstance(manager.registry.projectors[0], FlanaganProjector)


class TestHypothesisFamilyDecorator:
    def test_framenet_family_reasoner_name_set_by_decorator(self) -> None:
        assert FrameNetFamily.REASONER_NAME == FRAMENET_REASONER_NAME

    def test_flanagan_family_reasoner_name_set_by_decorator(self) -> None:
        assert FlanaganFamily.REASONER_NAME == FLANAGAN_REASONER_NAME

    def test_get_all_families_includes_both_registered_families(self) -> None:
        families = get_all_families()
        assert FrameNetFamily in families
        assert FlanaganFamily in families

    def test_get_all_families_returns_list(self) -> None:
        assert isinstance(get_all_families(), list)

    def test_hypothesis_family_decorator_registers_custom_family(self) -> None:
        @hypothesis_family(reasoner="test_only_reasoner")
        class _TestFamily(HypothesisFamily):
            PROJECTOR_TYPE = None  # type: ignore[assignment]
            VIEW_TYPE = None  # type: ignore[assignment]

        try:
            assert _TestFamily in get_all_families()
            assert _TestFamily.REASONER_NAME == "test_only_reasoner"
        finally:
            from llmr.hypotheses.families.base import _FAMILY_REGISTRY
            _FAMILY_REGISTRY.pop("test_only_reasoner", None)

    def test_hypothesis_family_decorator_sets_reasoner_name_attribute(self) -> None:
        @hypothesis_family(reasoner="another_test_reasoner")
        class _AnotherFamily(HypothesisFamily):
            PROJECTOR_TYPE = None  # type: ignore[assignment]
            VIEW_TYPE = None  # type: ignore[assignment]

        try:
            assert _AnotherFamily.REASONER_NAME == "another_test_reasoner"
        finally:
            from llmr.hypotheses.families.base import _FAMILY_REGISTRY
            _FAMILY_REGISTRY.pop("another_test_reasoner", None)


class TestMakeOrchestrator:
    def test_make_orchestrator_returns_projection_orchestrator(self) -> None:
        graph = HypothesisGraph()
        orchestrator = graph.make_orchestrator()
        assert isinstance(orchestrator, ProjectionOrchestrator)

    def test_make_orchestrator_includes_all_registered_families(self) -> None:
        graph = HypothesisGraph()
        orchestrator = graph.make_orchestrator()
        projector_types = {type(p) for p in orchestrator.registry.projectors}
        assert FrameNetProjector in projector_types
        assert FlanaganProjector in projector_types

    def test_make_orchestrator_uses_same_graph(self) -> None:
        graph = HypothesisGraph()
        orchestrator = graph.make_orchestrator()
        assert orchestrator.graph is graph
