"""Derived rustworkx graph adapter for sg_model repositories."""

from __future__ import annotations

from dataclasses import dataclass

from rustworkx import PyDiGraph

from llmr_updated_arch.hypotheses.entities.base import Hypothesis
from llmr_updated_arch.hypotheses.entities.common import Action, Instruction, ReasonerRun
from llmr_updated_arch.hypotheses.entities.flanagan import (
    FailureModeClaim,
    ForceDynamicEvent,
    GoalConditionClaim,
    PhaseClaim,
    PlanClaim,
    PreconditionClaim,
    RecoveryStrategyClaim,
)
from llmr_updated_arch.hypotheses.entities.framenet import FrameClaim, RoleClaim
from llmr_updated_arch.hypotheses.graph import HypothesisGraph


@dataclass(frozen=True)
class DerivedRelation:
    """Edge payload for presentation-oriented derived graphs."""

    relation_name: str
    source_id: str
    target_id: str
    color: str = "black"
    style: str = "solid"

    @property
    def label(self) -> str:
        return self.relation_name.replace("_", " ")

    def __str__(self) -> str:
        return self.label


def to_pydigraph(
    repository: HypothesisGraph,
) -> PyDiGraph[Hypothesis, DerivedRelation]:
    """Project the repository into a derived ``PyDiGraph`` for visualization."""

    graph: PyDiGraph[Hypothesis, DerivedRelation] = PyDiGraph()
    node_indices: dict[str, int] = {}
    for entity in repository:
        node_indices[entity.id] = graph.add_node(entity)

    seen_edges: set[tuple[str, str, str]] = set()

    def add_relation(
        source: Hypothesis,
        target: Hypothesis | None,
        relation_name: str,
        *,
        color: str = "black",
        style: str = "solid",
    ) -> None:
        if target is None or not repository.has(target.id):
            return
        edge_key = (source.id, target.id, relation_name)
        if edge_key in seen_edges:
            return
        seen_edges.add(edge_key)
        graph.add_edge(
            node_indices[source.id],
            node_indices[target.id],
            DerivedRelation(
                relation_name=relation_name,
                source_id=source.id,
                target_id=target.id,
                color=color,
                style=style,
            ),
        )

    for entity in repository:
        if isinstance(entity, Instruction):
            for frame in entity.frames:
                add_relation(entity, frame, "evokes_frame")
            for plan in entity.plans:
                add_relation(entity, plan, "evokes_plan")
            continue

        if isinstance(entity, ReasonerRun):
            for claim in entity.claims:
                if isinstance(claim, (PlanClaim, FrameClaim)):
                    add_relation(entity, claim, "produced_claim")
            continue

        if isinstance(entity, FrameClaim):
            add_relation(entity, entity.action, "about_action", color="slategray")
            for role in entity.roles:
                add_relation(entity, role, "has_role")
            continue

        if isinstance(entity, RoleClaim):
            for support in entity.supported_by:
                add_relation(
                    entity,
                    support,
                    "supported_by",
                    color="goldenrod",
                    style="dashed",
                )
            for grounding in entity.grounded_by:
                add_relation(
                    entity,
                    grounding,
                    "grounded_by",
                    color="green",
                    style="dashed",
                )
            continue

        if isinstance(entity, PlanClaim):
            add_relation(entity, entity.action, "about_action", color="slategray")
            for phase in entity.phases:
                add_relation(entity, phase, "has_phase")
            continue

        if isinstance(entity, PhaseClaim):
            add_relation(
                entity,
                entity.entry_event,
                "entry_event",
                color="darkorchid4",
                style="dashed",
            )
            add_relation(
                entity,
                entity.exit_event,
                "exit_event",
                color="darkorchid4",
                style="dashed",
            )
            for precondition in entity.preconditions:
                add_relation(
                    entity,
                    precondition,
                    "has_precondition",
                    color="darkorange3",
                )
            for goal_condition in entity.goal_conditions:
                add_relation(
                    entity,
                    goal_condition,
                    "has_goal_condition",
                    color="deepskyblue4",
                )
            for failure_mode in entity.failure_modes:
                add_relation(
                    entity,
                    failure_mode,
                    "has_failure_mode",
                    color="firebrick3",
                )
            for recovery_strategy in entity.recovery_strategies:
                add_relation(
                    entity,
                    recovery_strategy,
                    "has_recovery_strategy",
                    color="seagreen4",
                )
            continue

        if isinstance(
            entity,
            (
                ForceDynamicEvent,
                PreconditionClaim,
                GoalConditionClaim,
                FailureModeClaim,
                RecoveryStrategyClaim,
            ),
        ):
            continue

        if isinstance(entity, Action):
            continue

    return graph
