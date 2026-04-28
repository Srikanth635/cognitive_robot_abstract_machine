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
)


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
