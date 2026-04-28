"""Family-level abstractions tying together projector and view types."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

from typing_extensions import Generic, TypeVar

from llmr.hypotheses.graph import HypothesisGraph
from llmr.hypotheses.projection import (
    HypothesisProjector,
    ProjectionOrchestrator,
    ProjectorRegistry,
)
from llmr.hypotheses.views import ReasonerGraphView

TProjector = TypeVar("TProjector", bound=HypothesisProjector)
TView = TypeVar("TView", bound=ReasonerGraphView)


@dataclass(frozen=True)
class HypothesisFamily(ABC, Generic[TProjector, TView]):
    """Bind a reasoner family to its projector and graph view types.

    Concrete families provide one stable place to construct:
    - the reasoner-specific projector
    - a matching query/view facade over an existing `HypothesisGraph`
    - a `ProjectorRegistry` pre-populated with that family's projector
    """

    REASONER_NAME: ClassVar[str]
    PROJECTOR_TYPE: ClassVar[type[HypothesisProjector]]
    VIEW_TYPE: ClassVar[type[ReasonerGraphView]]

    @classmethod
    def make_projector(cls) -> TProjector:
        """Return a fresh projector for this family."""

        return cls.PROJECTOR_TYPE()  # type: ignore[return-value,call-arg]

    @classmethod
    def make_view(cls, graph: HypothesisGraph) -> TView:
        """Return the family's typed view over *graph*."""

        return cls.VIEW_TYPE(graph)  # type: ignore[return-value,call-arg]

    @classmethod
    def make_registry(cls) -> ProjectorRegistry:
        """Return a projector registry containing just this family's projector."""

        return ProjectorRegistry([cls.make_projector()])

    @classmethod
    def make_manager(cls, graph: HypothesisGraph) -> ProjectionOrchestrator:
        """Return a projection orchestrator for this family over *graph*."""

        return ProjectionOrchestrator(graph=graph, registry=cls.make_registry())
