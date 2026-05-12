"""Component contracts for the action-resolution architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from typing_extensions import Any

from llmr_updated_arch.core.context import ResolutionContext
from llmr_updated_arch.core.result import ActionResolutionResult
from llmr_updated_arch.integrations.krrood.match_reader import MatchField
from llmr_updated_arch.schemas import SemanticBundle


@dataclass(frozen=True)
class SemanticArtifact:
    """One named semantic output produced before grounding."""

    generator_name: str
    artifact_type: str
    value: Any


@dataclass(frozen=True)
class GroundingResult:
    """Concrete value produced by a grounding resolver."""

    slot_name: str
    value: Any
    resolved: bool = True


class SemanticGenerator(Protocol):
    """Generate structured semantic artifacts from a resolution context."""

    name: str

    def generate(self, context: ResolutionContext) -> SemanticArtifact | None:
        """Return a semantic artifact, or ``None`` when generation is skipped."""


class GroundingResolver(Protocol):
    """Resolve generated semantics into concrete values for one match slot."""

    def supports(self, slot: MatchField, bundle: SemanticBundle) -> bool:
        """Return whether this resolver can handle *slot*."""

    def resolve(
        self,
        slot: MatchField,
        context: ResolutionContext,
        resolved_params: dict[str, Any],
        unresolved: Any,
    ) -> GroundingResult:
        """Resolve *slot* from *context*."""


class ProjectionBuilder(Protocol):
    """Project a resolved action result into a hypothesis graph."""

    def supports(self, result: ActionResolutionResult) -> bool:
        """Return whether this builder can project *result*."""

    def build(self, result: ActionResolutionResult, graph: Any) -> Any:
        """Build projection objects into *graph*."""
