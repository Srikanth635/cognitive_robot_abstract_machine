"""Flanagan-specific graph query facade."""

from __future__ import annotations

from dataclasses import dataclass

from llmr.hypotheses.projectors.flanagan.constants import FLANAGAN_REASONER_NAME
from llmr.hypotheses.projectors.flanagan.edges import HasPhaseEdge
from llmr.hypotheses.projectors.flanagan.nodes import (
    PhaseClaimNode,
    PlanClaimNode,
)
from llmr.hypotheses.views.base import ReasonerGraphView


@dataclass(frozen=True)
class FlanaganGraphView(ReasonerGraphView):
    """Typed query facade for Flanagan motion-plan hypothesis nodes."""

    REASONER_NAME = FLANAGAN_REASONER_NAME
    ROOT_CLAIM_TYPES = (PlanClaimNode,)
    CLAIM_TYPES = (PlanClaimNode, PhaseClaimNode)

    def plans(self) -> list[PlanClaimNode]:
        """Return Flanagan motion-plan claims preserving graph insertion order."""

        return [
            node
            for node in self.graph.domain(PlanClaimNode)
            if node.meta.source_reasoner == self.REASONER_NAME
        ]

    def phases(self) -> list[PhaseClaimNode]:
        """Return Flanagan motion-phase claims preserving graph insertion order."""

        return [
            node
            for node in self.graph.domain(PhaseClaimNode)
            if node.meta.source_reasoner == self.REASONER_NAME
        ]

    def phases_by_name(self, phase_name: str) -> list[PhaseClaimNode]:
        """Return motion-phase claims whose canonical name equals *phase_name*."""

        return [phase for phase in self.phases() if phase.phase_name == phase_name]

    def phases_by_target_object(
        self, target_object: str
    ) -> list[PhaseClaimNode]:
        """Return motion-phase claims whose target object equals *target_object*."""

        return [
            phase for phase in self.phases() if phase.target_object == target_object
        ]

    def phases_for_plan(
        self, plan_node: PlanClaimNode
    ) -> list[PhaseClaimNode]:
        """Return phase claims attached to *plan_node* through HasPhaseEdge."""

        return [
            phase
            for phase in self.graph.get_targets(
                plan_node.id, HasPhaseEdge, PhaseClaimNode
            )
            if phase.meta.source_reasoner == self.REASONER_NAME
        ]

    def contact_phases(self) -> list[PhaseClaimNode]:
        """Return motion phases that expect contact."""

        return [phase for phase in self.phases() if phase.contact]

    def phases_with_failures(self) -> list[PhaseClaimNode]:
        """Return motion phases that include explicit failure modes."""

        return [phase for phase in self.phases() if phase.possible_failures]

    def high_urgency_phases(self) -> list[PhaseClaimNode]:
        """Return motion phases marked as high urgency."""

        return [phase for phase in self.phases() if phase.urgency == "high"]
