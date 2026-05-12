"""Builder registry and orchestration for sg_model object construction."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from llmr.hypotheses.build import BuildInput, BuildResult
from llmr.hypotheses.graph import HypothesisGraph

logger = logging.getLogger(__name__)


class HypothesisBuilder(ABC):
    """Abstract builder from reasoner sidecar data into sg_model objects."""

    @abstractmethod
    def supports(self, context: BuildInput) -> bool:
        """Return whether this builder can handle *context*."""

    @abstractmethod
    def build(self, context: BuildInput, graph: HypothesisGraph) -> BuildResult:
        """Build sg_model objects from *context* into *graph*."""


@dataclass
class BuilderRegistry:
    """Ordered registry of available sg_model builders."""

    builders: list[HypothesisBuilder] = field(default_factory=list)

    def register(self, builder: HypothesisBuilder) -> None:
        """Register *builder* preserving insertion order."""

        self.builders.append(builder)

    def matching(self, context: BuildInput) -> list[HypothesisBuilder]:
        """Return builders whose ``supports`` method accepts *context*."""

        return [builder for builder in self.builders if builder.supports(context)]


@dataclass
class BuildOrchestrator:
    """Run matching builders and accumulate their results into one repository."""

    graph: HypothesisGraph = field(default_factory=HypothesisGraph)
    registry: BuilderRegistry = field(default_factory=BuilderRegistry)

    def build(self, context: BuildInput) -> BuildResult:
        """Build all matching hypothesis families for *context*."""

        roots = []
        warnings = []
        for builder in self.registry.matching(context):
            try:
                result = builder.build(context, self.graph)
            except Exception as exc:
                logger.warning(
                    "BuildOrchestrator: builder %r raised %s - build skipped.",
                    builder,
                    exc,
                )
                warnings.append(
                    f"builder {type(builder).__name__} failed: {type(exc).__name__}: {exc}"
                )
                continue
            roots.extend(result.roots)
            warnings.extend(result.warnings)
        return BuildResult(roots=roots, warnings=warnings)

    @classmethod
    def with_default_builders(cls, *, graph: HypothesisGraph | None = None) -> "BuildOrchestrator":
        """Return an orchestrator pre-populated with the built-in builder set."""

        from llmr.hypotheses.builders.flanagan import FlanaganBuilder
        from llmr.hypotheses.builders.framenet import FrameNetBuilder

        registry = BuilderRegistry([FrameNetBuilder(), FlanaganBuilder()])
        return cls(graph=HypothesisGraph() if graph is None else graph, registry=registry)
