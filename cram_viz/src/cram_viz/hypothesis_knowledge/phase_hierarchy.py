"""
The phase-first presentation of a projected phase structure.

A projection links the structure to every fragment it defines as equally weighted
siblings (phases, roles, events, transitions, control points, ...), which scatters
everything belonging to one phase across unrelated lists. This overlay rebuilds the
*presentation graph only* -- never the stored hypothesis, so every ``pouring_*`` EQL
variable stays exactly as queryable as it was -- so phases become the graph's one
primary branch, and each phase nests everything that involves it: its own fields, and
the fragments that only point *at* it (transitions, force couplings, gaze fixations,
control points at its boundary events).

Apply this to a projection built with ``HypothesisGraphPresentation.COMPLETE``, a flat
starting point with no grouping, so this module owns the entire grouping decision.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import StrEnum

import rustworkx as rx
from llm_reasoner.knowledge.hypotheses.graph import (
    HypothesisGraph,
    HypothesisGraphNode,
    HypothesisGraphNodeKind,
    HypothesisGraphRelation,
    SyntheticHypothesisRelation,
)
from llm_reasoner.knowledge.hypotheses.view import HypothesisView
from llm_reasoner.knowledge.ontologies.predictive_manipulation import core
from llm_reasoner.knowledge.ontologies.terms import DescriptionEntity, PropertyTerm
from typing_extensions import Callable, Optional

ROOT_LABEL = "Phase Structure"
"""
The label the root shows, whatever the structure's own label.
"""

GROUP_TYPE_NAME = "PhaseHierarchyGroup"
"""
The type name of the group nodes this overlay adds.
"""

SUBGOAL_REDUNDANT_PREDICATES = frozenset({"attainment_signalled_by"})
"""
The fields of a subgoal that restate what its phase already shows: the event signalling
the subgoal is the phase's terminating event.
"""


class PhaseGroup(StrEnum):
    """
    The groups this overlay draws, by the label they show.
    """

    PHASES = "Phases"
    """
    The structure's phases, under the root.
    """

    ROLE_TAKERS = "Role Takers"
    """
    The roles a phase involves.
    """

    CONDITIONS = "Conditions"
    """
    The condition requirements of a phase.
    """

    PARAMETERS = "Parameters"
    """
    The parameters active in a phase.
    """

    INITIATED_BY = "Initiated By"
    """
    The events starting a phase.
    """

    TERMINATED_BY = "Terminated By"
    """
    The events ending a phase.
    """

    TRANSITIONS = "Transitions"
    """
    The transitions leading out of or into a phase.
    """

    FORCE_COUPLINGS = "Force Couplings"
    """
    The grip-load couplings active in a phase.
    """

    GAZE_FIXATIONS = "Gaze Fixations"
    """
    The gaze fixations serving a phase.
    """

    CONTROL_POINTS = "Control Points"
    """
    The control points at a phase's boundary events.
    """

    UNASSIGNED = "Other Structure Knowledge"
    """
    The fragments no phase reaches, under the root.
    """


class PhaseHierarchyRelation(StrEnum):
    """
    Links drawn only by the phase-first presentation, from a phase to fragments that
    point at it.
    """

    HAS_TRANSITION = "has_transition"
    HAS_FORCE_COUPLING = "has_force_coupling"
    HAS_GAZE_FIXATION = "has_gaze_fixation"
    HAS_CONTROL_POINT = "has_control_point"
    UNASSIGNED = "unassigned"


@dataclass(frozen=True)
class PhaseHierarchyOverlay:
    """
    Rebuilds a projection's presentation graph around its phases.
    """

    def apply(
        self, projection: HypothesisGraph, view: HypothesisView
    ) -> HypothesisGraph:
        """
        Relabel the root, regroup it and every phase, and return the same projection.
        """
        projection.graph[projection.root_index] = dataclasses.replace(
            projection.root, label=ROOT_LABEL
        )
        indices = {
            id(projection.graph[index].source): index
            for index in projection.graph.node_indices()
            if projection.graph[index].source is not None
        }
        structure = view.root
        self.clear_out_edges(projection, projection.root_index)
        self.add_group(
            projection,
            projection.root_index,
            PhaseGroup.PHASES,
            "defines_phase",
            structure.defines_phase,
            indices,
        )
        for phase in structure.defines_phase:
            self.rebuild_phase(projection, view, phase, indices)
        self.remove_back_references(projection, view, indices)
        self.remove_redundant_subgoal_links(projection, structure, indices)
        self.group_unreached(projection, indices)
        return projection

    def rebuild_phase(
        self,
        projection: HypothesisGraph,
        view: HypothesisView,
        phase: core.ActionPhase,
        indices: dict[int, int],
    ) -> None:
        """
        Regroup one phase's own fields and attach the fragments pointing at it.
        """
        phase_index = indices[id(phase)]
        self.clear_out_edges(projection, phase_index)
        for field_name, target in (
            ("aims_for", phase.aims_for),
            ("controlled_by", phase.controlled_by),
        ):
            if target is not None and id(target) in indices:
                self.connect(
                    projection,
                    phase_index,
                    field_name,
                    indices[id(target)],
                    self.property_of(phase, field_name),
                )
        boundaries = [*phase.initiated_by, *phase.terminated_by]
        groups: list[tuple[PhaseGroup, str, list[DescriptionEntity]]] = [
            (PhaseGroup.ROLE_TAKERS, "involves_role", phase.involves_role),
            (
                PhaseGroup.CONDITIONS,
                "has_condition_requirement",
                phase.has_condition_requirement,
            ),
            (PhaseGroup.PARAMETERS, "has_active_parameter", phase.has_active_parameter),
            (PhaseGroup.INITIATED_BY, "initiated_by", phase.initiated_by),
            (PhaseGroup.TERMINATED_BY, "terminated_by", phase.terminated_by),
            (
                PhaseGroup.TRANSITIONS,
                PhaseHierarchyRelation.HAS_TRANSITION,
                [
                    transition
                    for transition in view.root.defines_transition
                    if transition.from_phase is phase or transition.to_phase is phase
                ],
            ),
            (
                PhaseGroup.FORCE_COUPLINGS,
                PhaseHierarchyRelation.HAS_FORCE_COUPLING,
                view.force_couplings_in(phase),
            ),
            (
                PhaseGroup.GAZE_FIXATIONS,
                PhaseHierarchyRelation.HAS_GAZE_FIXATION,
                view.gaze_fixations_for(phase),
            ),
            (
                PhaseGroup.CONTROL_POINTS,
                PhaseHierarchyRelation.HAS_CONTROL_POINT,
                [
                    point
                    for point in view.root.defines_control_point
                    if any(point.at_event_type is event for event in boundaries)
                ],
            ),
        ]
        for group, predicate, members in groups:
            self.add_group(projection, phase_index, group, predicate, members, indices)

    def remove_back_references(
        self, projection: HypothesisGraph, view: HypothesisView, indices: dict[int, int]
    ) -> None:
        """
        Remove the links from transitions, force couplings and gaze fixations back to
        the phases they are now grouped under, which would otherwise show a phase as a
        child of a fragment nested beneath it.
        """
        phase_indices = {indices[id(phase)] for phase in view.root.defines_phase}
        grouped = [
            *view.root.defines_transition,
            *[
                coupling
                for phase in view.root.defines_phase
                for coupling in view.force_couplings_in(phase)
            ],
            *view.root.defines_gaze_fixation,
        ]
        for fragment in grouped:
            self.remove_out_edges(
                projection,
                indices[id(fragment)],
                lambda target, relation: target in phase_indices,
            )

    def remove_redundant_subgoal_links(
        self,
        projection: HypothesisGraph,
        structure: core.ActionPhaseStructure,
        indices: dict[int, int],
    ) -> None:
        """
        Remove a subgoal's links that restate what its phase already shows.
        """
        for phase in structure.defines_phase:
            if phase.aims_for is None:
                continue
            self.remove_out_edges(
                projection,
                indices[id(phase.aims_for)],
                lambda target, relation: relation.predicate
                in SUBGOAL_REDUNDANT_PREDICATES,
            )

    def group_unreached(
        self, projection: HypothesisGraph, indices: dict[int, int]
    ) -> None:
        """
        Group under the root every fragment no phase reaches, so nothing becomes
        unreachable in the viewer.
        """
        reachable = rx.descendants(projection.graph, projection.root_index)
        unreached = [
            projection.graph[index].source
            for index in projection.graph.node_indices()
            if index != projection.root_index
            and index not in reachable
            and projection.graph[index].source is not None
        ]
        self.add_group(
            projection,
            projection.root_index,
            PhaseGroup.UNASSIGNED,
            PhaseHierarchyRelation.UNASSIGNED,
            unreached,
            indices,
        )

    @staticmethod
    def remove_out_edges(
        projection: HypothesisGraph,
        node_index: int,
        selects: Callable[[int, HypothesisGraphRelation], bool],
    ) -> None:
        """
        Remove the links leaving a node that the selector picks by target index and
        relation.
        """
        for edge_index, (_, target, relation) in list(
            projection.graph.incident_edge_index_map(node_index).items()
        ):
            if selects(target, relation):
                projection.graph.remove_edge_from_index(edge_index)

    def clear_out_edges(self, projection: HypothesisGraph, node_index: int) -> None:
        """
        Remove every link leaving a node.
        """
        self.remove_out_edges(projection, node_index, lambda target, relation: True)

    def add_group(
        self,
        projection: HypothesisGraph,
        parent_index: int,
        group: PhaseGroup,
        predicate: str,
        members: list[DescriptionEntity],
        indices: dict[int, int],
    ) -> None:
        """
        Add one labelled group under a node, or nothing when it has no members.

        The group's links keep the ontology property the predicate names on the node's
        fragment, if any.
        """
        member_indices = list(
            dict.fromkeys(
                indices[id(member)] for member in members if id(member) in indices
            )
        )
        if not member_indices:
            return
        property_iri = self.property_of(
            projection.graph[parent_index].source, predicate
        )
        group_index = projection.graph.add_node(
            HypothesisGraphNode(
                identifier=(
                    f"phase-hierarchy:{projection.graph[parent_index].identifier}:"
                    f"{group.name.lower()}"
                ),
                label=f"{group.value} ({len(member_indices)})",
                type_name=GROUP_TYPE_NAME,
                kind=HypothesisGraphNodeKind.GROUP,
            )
        )
        self.connect(
            projection, parent_index, SyntheticHypothesisRelation.GROUPS, group_index
        )
        for member_index in member_indices:
            self.connect(projection, group_index, predicate, member_index, property_iri)

    @staticmethod
    def property_of(owner: Optional[DescriptionEntity], predicate: str) -> str:
        """
        :return: The IRI of the ontology property a field of the owner states, or an
            empty string when the predicate is no such field.
        """
        term = None if owner is None else PropertyTerm.of_field(type(owner), predicate)
        if term is None:
            return ""
        return term.iri

    @staticmethod
    def connect(
        projection: HypothesisGraph,
        source_index: int,
        predicate: str,
        target_index: int,
        property_iri: str = "",
    ) -> None:
        """
        Add one link between two nodes, stated by the given ontology property or,
        without one, drawn by this presentation.
        """
        projection.graph.add_edge(
            source_index,
            target_index,
            HypothesisGraphRelation(
                source=projection.graph[source_index],
                predicate=predicate,
                target=projection.graph[target_index],
                property_iri=property_iri,
            ),
        )
