"""Generic projection framework from reasoner sidecars into HypothesisGraph."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing_extensions import Any, ClassVar, Optional

from llmr.hypotheses.elements import HypothesisEdge, HypothesisNode
from llmr.hypotheses.graph import HypothesisGraph

if TYPE_CHECKING:
    from llmr.bridge.match_reader import MatchSnapshot as MatchData
    from llmr.schemas import ActionAnnotationBundle as ActionSemantics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectionInput:
    """Normalized input consumed by all hypothesis projectors."""

    instruction: Optional[str]
    action: Any
    action_type: str
    semantics: ActionSemantics
    match_data: MatchData
    resolved_slots: dict[str, Any]
    world_context: str
    symbol_type: type
    llm_model_name: Optional[str] = None


@dataclass(frozen=True)
class HypothesisProjection:
    """Self-contained set of nodes and edges produced by a projector."""

    nodes: list[HypothesisNode]
    edges: list[HypothesisEdge]
    warnings: list[str] = field(default_factory=list)


class HypothesisProjector(ABC):
    """Abstract projector from sidecar semantics to graph projection."""

    REASONER_NAME: ClassVar[str]

    @abstractmethod
    def supports(self, context: ProjectionInput) -> bool:
        """Return whether this projector can handle *context*."""

    @abstractmethod
    def project(self, context: ProjectionInput) -> HypothesisProjection:
        """Build a graph projection from *context*."""


@dataclass
class ProjectorRegistry:
    """Ordered registry of available hypothesis projectors."""

    projectors: list[HypothesisProjector] = field(default_factory=list)

    def register(self, projector: HypothesisProjector) -> None:
        """Register *projector* preserving insertion order."""

        self.projectors.append(projector)

    def matching(
        self, context: ProjectionInput
    ) -> list[HypothesisProjector]:
        """Return projectors whose ``supports`` method accepts *context*."""

        return [projector for projector in self.projectors if projector.supports(context)]


@dataclass
class ProjectionOrchestrator:
    """Run matching projectors and insert their results into a graph."""

    graph: HypothesisGraph
    registry: ProjectorRegistry

    def project(self, context: ProjectionInput) -> None:
        """Project *context* into the graph using all matching projectors.

        Projector failures are isolated so one failing projector does not block
        the rest of the graph population pipeline.
        """

        for projector in self.registry.matching(context):
            try:
                projection = projector.project(context)
            except Exception as exc:
                logger.warning(
                    "ProjectionOrchestrator: projector %r raised %s — projection skipped.",
                    projector,
                    exc,
                )
                continue
            self.graph.add_projection(projection)


