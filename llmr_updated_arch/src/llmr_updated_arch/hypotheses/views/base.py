"""View abstractions layered on top of the sg_model repository."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

from typing_extensions import Optional, Tuple, TypeVar

from llmr_updated_arch.hypotheses.entities.base import ClaimHypothesis, Hypothesis
from llmr_updated_arch.hypotheses.graph import HypothesisGraph

THypothesis = TypeVar("THypothesis", bound=Hypothesis)
TClaim = TypeVar("TClaim", bound=ClaimHypothesis)


@dataclass(frozen=True)
class HypothesisGraphView:
    """Thin typed facade over a generic sg_model repository."""

    graph: HypothesisGraph

    def nodes(self, entity_type: type[THypothesis]) -> list[THypothesis]:
        """Return repository entities of *entity_type* preserving insertion order."""

        return self.graph.domain(entity_type)

    def get(self, entity_id: str) -> Optional[Hypothesis]:
        """Return the entity with *entity_id*, if present."""

        return self.graph.get(entity_id)

    def nodes_for_run(self, run_id: str) -> list[Hypothesis]:
        """Return entities tagged with *run_id*."""

        return self.graph.nodes_for_run(run_id)

    def nodes_from_reasoner(self, reasoner_name: str) -> list[Hypothesis]:
        """Return entities attributed to *reasoner_name*."""

        return self.graph.nodes_from_reasoner(reasoner_name)


@dataclass(frozen=True)
class ReasonerGraphView(HypothesisGraphView, ABC):
    """Base view contract shared by reasoner-family query facades."""

    REASONER_NAME: ClassVar[str]
    ROOT_CLAIM_TYPES: ClassVar[Tuple[type[ClaimHypothesis], ...]] = ()
    CLAIM_TYPES: ClassVar[Tuple[type[ClaimHypothesis], ...]] = ()

    def claims(self) -> list[ClaimHypothesis]:
        """Return claim entities belonging to this reasoner family."""

        return [
            claim
            for claim_type in self.CLAIM_TYPES
            for claim in self.graph.domain(claim_type)
            if claim.meta.source_reasoner == self.REASONER_NAME
        ]

    def root_claims(self) -> list[ClaimHypothesis]:
        """Return top-level claim entities belonging to this reasoner family."""

        return [
            claim
            for claim_type in self.ROOT_CLAIM_TYPES
            for claim in self.graph.domain(claim_type)
            if claim.meta.source_reasoner == self.REASONER_NAME
        ]

    def claims_for_run(self, run_id: str) -> list[ClaimHypothesis]:
        """Return this family's claims tagged with *run_id*."""

        return [
            claim
            for claim in self.claims()
            if claim.meta.run_id == run_id
        ]

    def claims_for_action(self, action_ref: object) -> list[ClaimHypothesis]:
        """Return this family's claims in the action-local reasoning closure."""

        root_claims = [
            claim
            for claim in self.root_claims()
            if getattr(getattr(claim, "action", None), "action_ref", None) is action_ref
        ]
        if not root_claims:
            return []

        ordered: list[ClaimHypothesis] = []
        seen_ids: set[str] = set()
        run_ids: list[str] = []
        for claim in root_claims:
            if claim.id not in seen_ids:
                ordered.append(claim)
                seen_ids.add(claim.id)
            if claim.meta.run_id is not None and claim.meta.run_id not in run_ids:
                run_ids.append(claim.meta.run_id)

        for run_id in run_ids:
            for claim in self.claims_for_run(run_id):
                if claim.id in seen_ids:
                    continue
                ordered.append(claim)
                seen_ids.add(claim.id)
        return ordered
