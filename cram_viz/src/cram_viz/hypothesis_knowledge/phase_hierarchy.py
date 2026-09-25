"""
The phase-first presentation of a projected Flanagan phase structure.

By default a projection groups every one of the phase structure's list-valued
fields as equally-weighted siblings of the root (phases, roles, event types,
parameters, transitions, control points, force couplings, continuous predictions),
which scatters everything belonging to one phase across unrelated top-level lists.
This overlay rebuilds the *presentation graph only* -- never the underlying
``PouringPhaseStructure``/``HypothesisRecord``, so every ``pouring_*`` EQL variable
stays exactly as queryable as it was -- so phases become the graph's one primary
branch, and each phase nests everything that involves it: role takers, its
subgoal, conditions, monitored signals, governing models, boundary events, and
(since these otherwise only point *at* a phase in reverse) its parameters, force
couplings, continuous predictions, transitions and control points.

Apply this to a projection built with
``presentation=HypothesisGraphPresentation.COMPLETE`` -- a fully flat starting point
with no grouping anywhere -- so this module owns the entire grouping decision rather
than un-doing the projector's own ``GROUPED`` root handling.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import rustworkx as rx
from llm_reasoner.knowledge.hypothesis.graph import (
    HypothesisGraph,
    HypothesisGraphNode,
    HypothesisGraphNodeKind,
    HypothesisGraphRelation,
    SyntheticHypothesisRelation,
)
from llm_reasoner.knowledge.microworlds.flanagan.core import ActionPhase, RoleSlot

from cram_viz.hypothesis_knowledge.action_frame import (
    ROLE_FILLER_TYPE_NAME,
    UNRESOLVED_ROLE_FILLER_TYPE_NAME,
    MinskyActionFrameTemplate,
    role_type_name,
)

#: predicates a subgoal's own fields project, which restate content the phase
#: itself already shows elsewhere (see ``_strip_redundant_subgoal_edges``)
SUBGOAL_REDUNDANT_PREDICATES = frozenset(
    {"condition_on_role", "attainment_signalled_by"}
)

#: display label the root gets, independent of whatever the underlying structure's
#: own ``label``/class name is -- generic across any Flanagan phase structure, not
#: specific to pouring
ROOT_LABEL = "FlanaganModel"

#: (group label, ``ActionPhase`` field name) for the fields the projector already
#: connects directly to a phase; only re-grouped here, never re-derived
REAL_PHASE_GROUPS: tuple[tuple[str, str], ...] = (
    ("Role Takers", "phase_involves"),
    ("Conditions", "has_condition_requirement"),
    ("Monitored Signals", "monitors_signal"),
    ("Governing Models", "governed_by"),
    ("Initiated By", "initiated_by"),
    ("Terminated By", "terminated_by"),
)


class PhaseHierarchyRelation(StrEnum):
    """
    Relationships introduced only by the phase-hierarchy presentation.
    """

    HAS_PARAMETER = "has_parameter"
    HAS_FORCE_COUPLING = "has_force_coupling"
    HAS_CONTINUOUS_PREDICTION = "has_continuous_prediction"
    HAS_TRANSITION = "has_transition"
    HAS_CONTROL_POINT = "has_control_point"
    HAS_NAME = "has_name"
    UNASSIGNED = "unassigned"


@dataclass(frozen=True)
class PhaseHierarchyOverlay:
    """
    Rebuild a projection's presentation graph around its phases.
    """

    def apply(self, projection: HypothesisGraph) -> HypothesisGraph:
        """
        Relabel the root, regroup it and every phase, and return the same projection.
        """
        self._relabel_root(projection)
        fragment_index = self._index_by_fragment_id(projection)
        structure = projection.root.source

        # Relabelling role slots must happen before any edge pointing at one is
        # (re)built below: a ``HypothesisGraphRelation`` captures its ``target``
        # node object at edge-creation time (rustworkx edge payloads are not
        # re-resolved from the node array on read), so an edge built before the
        # relabel would keep showing the pre-relabel text forever.
        self._relabel_role_slots(projection)

        self._clear_out_edges(projection, projection.root_index)
        self._add_group(
            projection,
            projection.root_index,
            "Phases",
            "defines_phase",
            structure.defines_phase,
            fragment_index,
        )
        for phase in structure.defines_phase:
            self._rebuild_phase(projection, structure, phase, fragment_index)
        self._strip_phase_backreferences(projection, structure, fragment_index)
        self._strip_redundant_subgoal_edges(projection, structure, fragment_index)
        self._ground_role_slots(projection)
        self._add_orphan_safety_net(projection, fragment_index)
        return projection

    @staticmethod
    def _relabel_root(projection: HypothesisGraph) -> None:
        projection.graph[projection.root_index] = dataclasses.replace(
            projection.root, label=ROOT_LABEL
        )

    @staticmethod
    def _index_by_fragment_id(projection: HypothesisGraph) -> dict[int, int]:
        """
        Fragment object identity -> its node index, for every real fragment.
        """
        index_by_fragment_id: dict[int, int] = {}
        for index in projection.graph.node_indices():
            source = projection.graph[index].source
            if source is not None:
                index_by_fragment_id[id(source)] = index
        return index_by_fragment_id

    def _rebuild_phase(
        self,
        projection: HypothesisGraph,
        structure: Any,
        phase: ActionPhase,
        fragment_index: dict[int, int],
    ) -> None:
        """
        Regroup one phase's real fields and attach its synthetic categories.
        """
        phase_index = fragment_index[id(phase)]
        self._clear_out_edges(projection, phase_index)

        if phase.aims_for is not None and id(phase.aims_for) in fragment_index:
            self._connect(
                projection, phase_index, "aims_for", fragment_index[id(phase.aims_for)]
            )
        for label, field_name in REAL_PHASE_GROUPS:
            self._add_group(
                projection,
                phase_index,
                label,
                field_name,
                getattr(phase, field_name),
                fragment_index,
            )

        self._add_group(
            projection,
            phase_index,
            "Parameters",
            PhaseHierarchyRelation.HAS_PARAMETER.value,
            [
                parameter
                for parameter in getattr(structure, "defines_parameter", [])
                if phase in parameter.parameter_in_phase
            ],
            fragment_index,
        )
        self._add_group(
            projection,
            phase_index,
            "Force Couplings",
            PhaseHierarchyRelation.HAS_FORCE_COUPLING.value,
            [
                coupling
                for coupling in getattr(structure, "has_force_coupling", [])
                if phase in coupling.coupling_active_in
            ],
            fragment_index,
        )
        self._add_group(
            projection,
            phase_index,
            "Continuous Predictions",
            PhaseHierarchyRelation.HAS_CONTINUOUS_PREDICTION.value,
            [
                prediction
                for prediction in getattr(structure, "has_continuous_prediction", [])
                if phase in prediction.predicted_for_phase
            ],
            fragment_index,
        )
        self._add_group(
            projection,
            phase_index,
            "Transitions",
            PhaseHierarchyRelation.HAS_TRANSITION.value,
            [
                transition
                for transition in getattr(structure, "has_transition", [])
                if transition.from_phase is phase or transition.to_phase is phase
            ],
            fragment_index,
        )
        boundary_events = set(phase.initiated_by) | set(phase.terminated_by)
        self._add_group(
            projection,
            phase_index,
            "Control Points",
            PhaseHierarchyRelation.HAS_CONTROL_POINT.value,
            [
                control_point
                for control_point in getattr(structure, "has_control_point", [])
                if control_point.at_event_type in boundary_events
            ],
            fragment_index,
        )

    @staticmethod
    def _strip_phase_backreferences(
        projection: HypothesisGraph, structure: Any, fragment_index: dict[int, int]
    ) -> None:
        """
        Remove a transition's/coupling's/prediction's own edges back to the phases they
        were just grouped under.

        ``PhaseTransition.from_phase``/``to_phase``, ``GripLoadCoupling
        .coupling_active_in`` and ``ContinuousSignalPrediction.predicted_for_phase`` are
        exactly the reverse-pointer fields ``_rebuild_phase`` reads to attach each
        fragment under its owning phase(s); left in place, that same field also still
        points forward from the fragment back at the phase, so expanding e.g. a
        transition reveals the very phase it is nested under as if it were a child.
        Every other real edge (``gated_by``, ``predicted_load``, ...) is left untouched.
        """
        phase_indices = {
            fragment_index[id(phase)]
            for phase in structure.defines_phase
            if id(phase) in fragment_index
        }
        backreferencing_fragments = (
            list(getattr(structure, "has_transition", []))
            + list(getattr(structure, "has_force_coupling", []))
            + list(getattr(structure, "has_continuous_prediction", []))
        )
        for fragment in backreferencing_fragments:
            fragment_node_index = fragment_index.get(id(fragment))
            if fragment_node_index is None:
                continue
            for source_index, target_index, _ in list(
                projection.graph.out_edges(fragment_node_index)
            ):
                if target_index in phase_indices:
                    projection.graph.remove_edge(source_index, target_index)

    @staticmethod
    def _strip_redundant_subgoal_edges(
        projection: HypothesisGraph, structure: Any, fragment_index: dict[int, int]
    ) -> None:
        """
        Remove a subgoal's own ``condition_on_role`` and ``attainment_signalled_by``
        edges, since both restate content the phase itself already shows.

        A subgoal is, by definition (see :class:`~llm_reasoner.knowledge.microworlds
        .flanagan.core.Subgoal`), the postcondition whose attainment is signalled by the
        *phase's own terminating event* -- so ``attainment_signalled_by`` never names
        anything beyond what "Terminated By" already does. Its ``condition_on_role`` is
        likewise typically the same participants already listed under "Role Takers".
        ``violation_signalled_by`` is the one field with content found nowhere else at
        the phase level, and is left untouched.
        """
        for phase in structure.defines_phase:
            subgoal = phase.aims_for
            if subgoal is None:
                continue
            subgoal_index = fragment_index.get(id(subgoal))
            if subgoal_index is None:
                continue
            for source_index, target_index, relation in list(
                projection.graph.out_edges(subgoal_index)
            ):
                if relation.predicate in SUBGOAL_REDUNDANT_PREDICATES:
                    projection.graph.remove_edge(source_index, target_index)

    @staticmethod
    def _relabel_role_slots(projection: HypothesisGraph) -> None:
        """
        Show a role slot's bound type (e.g. ``'Source Container'``) instead of the
        illustrative filler every ``RoleSlot`` instance's own free-text ``label``
        otherwise shows -- ``binds_as``, not the free-text ``label``, is what actually
        distinguishes one role from another.

        Prefers ``binds_as.label`` (already written with spacing, e.g. ``'Source
        Container'``) over :func:`role_type_name`'s bare IRI fragment (e.g.
        ``'SourceContainer'``), which stays reserved for exact-match lookups such as
        :meth:`.action_frame.MinskyActionFrameTemplate.entity_for_role`.
        """
        for index in projection.graph.node_indices():
            node = projection.graph[index]
            if not isinstance(node.source, RoleSlot):
                continue
            binds_as = node.source.binds_as
            if binds_as is None:
                continue
            label = binds_as.label or role_type_name(node.source)
            if label is None:
                continue
            projection.graph[index] = dataclasses.replace(node, label=label)

    @staticmethod
    def _ground_role_slots(projection: HypothesisGraph) -> None:
        """
        Add a leaf under every role slot holding its concrete filler for this
        instruction, or ``"?"`` when it isn't named.

        A ``RoleSlot`` never carries a candidate filler of its own -- it is unbound by
        design (see :mod:`.action_frame`'s module docstring) -- so there is nothing here
        to check the instruction against directly. Instead this reuses ``FrameModel``'s
        own hand-curated candidates via
        :meth:`MinskyActionFrameTemplate.entity_for_role`: whichever roles that template
        covers inherit its grounding decision exactly, so the same role reads the same
        way whether viewed from ``FrameModel`` or ``FlanaganModel``. Roles the template
        doesn't cover (e.g. ``GraspSite``) have no candidate at all and are always left
        unresolved.

        The leaf's own ``source`` is left ``None`` (not the role) so it never contends
        with the role's own node for the identity -> node-index map used to highlight
        EQL results.
        """
        template = MinskyActionFrameTemplate.cereal_pouring()
        instruction_text = projection.instruction.label.lower()
        for index in list(projection.graph.node_indices()):
            node = projection.graph[index]
            role = node.source
            if not isinstance(role, RoleSlot):
                continue
            entity = template.entity_for_role(role)
            grounded = (
                entity is not None and entity.concrete_name.lower() in instruction_text
            )
            name_node = HypothesisGraphNode(
                identifier=f"phase-hierarchy:{node.identifier}:name",
                label=entity.concrete_name if grounded else "?",
                type_name=(
                    ROLE_FILLER_TYPE_NAME
                    if grounded
                    else UNRESOLVED_ROLE_FILLER_TYPE_NAME
                ),
                kind=HypothesisGraphNodeKind.FRAGMENT,
            )
            name_index = projection.graph.add_node(name_node)
            PhaseHierarchyOverlay._connect(
                projection, index, PhaseHierarchyRelation.HAS_NAME.value, name_index
            )

    @staticmethod
    def _add_orphan_safety_net(
        projection: HypothesisGraph, fragment_index: dict[int, int]
    ) -> None:
        """
        Catch every real fragment no phase ended up pointing at, directly or
        transitively, so nothing becomes permanently unreachable in the UI.
        """
        reachable = rx.descendants(projection.graph, projection.root_index)
        orphans = [
            projection.graph[index].source
            for index in projection.graph.node_indices()
            if index != projection.root_index
            and index not in reachable
            and projection.graph[index].source is not None
        ]
        PhaseHierarchyOverlay._add_group(
            projection,
            projection.root_index,
            "Other Structure Knowledge",
            PhaseHierarchyRelation.UNASSIGNED.value,
            orphans,
            fragment_index,
        )

    @staticmethod
    def _clear_out_edges(projection: HypothesisGraph, node_index: int) -> None:
        for source_index, target_index, _ in list(
            projection.graph.out_edges(node_index)
        ):
            projection.graph.remove_edge(source_index, target_index)

    @staticmethod
    def _add_group(
        projection: HypothesisGraph,
        parent_index: int,
        category: str,
        predicate: str,
        members: list[Any],
        fragment_index: dict[int, int],
    ) -> None:
        """
        Add one labelled category group under a node, or nothing if it's empty.
        """
        member_indices = [
            fragment_index[id(member)]
            for member in members
            if id(member) in fragment_index
        ]
        if not member_indices:
            return

        group_node = HypothesisGraphNode(
            identifier=(
                f"phase-hierarchy:{projection.graph[parent_index].identifier}:"
                f"{category.lower().replace(' ', '_')}"
            ),
            label=f"{category} ({len(member_indices)})",
            type_name="PhaseHierarchyGroup",
            kind=HypothesisGraphNodeKind.GROUP,
        )
        group_index = projection.graph.add_node(group_node)
        PhaseHierarchyOverlay._connect(
            projection,
            parent_index,
            SyntheticHypothesisRelation.GROUPS.value,
            group_index,
        )
        for member_index in member_indices:
            PhaseHierarchyOverlay._connect(
                projection, group_index, predicate, member_index
            )

    @staticmethod
    def _connect(
        projection: HypothesisGraph,
        source_index: int,
        predicate: str,
        target_index: int,
    ) -> None:
        projection.graph.add_edge(
            source_index,
            target_index,
            HypothesisGraphRelation(
                source=projection.graph[source_index],
                predicate=predicate,
                target=projection.graph[target_index],
            ),
        )
