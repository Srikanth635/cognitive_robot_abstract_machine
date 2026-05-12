"""Abstract base entities for the sg_model hypothesis object model."""

from __future__ import annotations

from abc import ABC
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING

from llmr_updated_arch.hypotheses.meta import HypothesisMeta, _shorten_identifier

if TYPE_CHECKING:
    from llmr_updated_arch.hypotheses.graph import HypothesisGraph


@dataclass(eq=False)
class Hypothesis(ABC):
    """Base identity contract shared by all sg_model entities."""

    id: str
    meta: HypothesisMeta
    _register_to: InitVar["HypothesisGraph | None"] = field(
        default=None, kw_only=True
    )

    def __post_init__(self, _register_to: "HypothesisGraph | None") -> None:
        if _register_to is not None:
            self.register_with(_register_to)

    @property
    def short_id(self) -> str:
        """Compact display form for this entity id."""

        return _shorten_identifier(self.id)

    @property
    def display_id(self) -> str:
        """Alias for the compact id used in UI-facing contexts."""

        return self.short_id

    def register_with(self, graph: "HypothesisGraph") -> "Hypothesis":
        """Register this entity with a repository after full initialization."""

        graph.add(self)
        return self


@dataclass(eq=False)
class AnchorHypothesis(Hypothesis, ABC):
    """Abstract entity anchoring claims to an instruction, action, or run."""


@dataclass(eq=False)
class ClaimHypothesis(Hypothesis, ABC):
    """Abstract entity representing an epistemic claim."""


@dataclass(eq=False)
class ProjectedClaimHypothesis(ClaimHypothesis, ABC):
    """Claim entity produced by a pluggable llmr reasoner."""


@dataclass(eq=False)
class EvidenceHypothesis(Hypothesis, ABC):
    """Abstract entity providing structured support or grounding."""
