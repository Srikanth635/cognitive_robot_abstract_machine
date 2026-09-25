"""
Tests for the phase-first presentation of a projected Flanagan phase structure.

The projected graph is purely a presentation-layer artefact: nothing here touches the
underlying ``PouringPhaseStructure``/``HypothesisRecord`` or any EQL variable built from
it (see ``test_kb.py``'s ``TestHypothesisKnowledgeQueries``, unaffected by this module).
"""

from __future__ import annotations

import pytest
import rustworkx as rx
from cram_viz.hypothesis_knowledge import build_hypothesis_record, build_projection
from cram_viz.hypothesis_knowledge.action_frame import (
    ROLE_FILLER_TYPE_NAME,
    UNRESOLVED_ROLE_FILLER_TYPE_NAME,
    MinskyActionFrameTemplate,
    role_type_name,
)
from cram_viz.hypothesis_knowledge.phase_hierarchy import (
    ROOT_LABEL,
    PhaseHierarchyOverlay,
)
from cram_viz.hypothesis_knowledge.pouring_structure import (
    build_pouring_phase_structure,
)
from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ActionPhaseStructure,
    RoleSlot,
    Subgoal,
)
from llm_reasoner.knowledge.microworlds.flanagan.pouring import PouringPhaseStructure
from llm_reasoner.knowledge.microworlds.flanagan.prehension import (
    ContactEstablishment,
    GripForceSetpoint,
    ReachPhase,
    SlipEvent,
)
from llm_reasoner.knowledge.ontological.decorators import ConceptReference


@pytest.fixture
def structure():
    return build_pouring_phase_structure()


@pytest.fixture
def bare_projection(structure):
    """
    A projection of the sample structure, with no overlay applied yet.
    """
    return build_projection(build_hypothesis_record(structure), apply_overlays=False)


def node_for(projection, fragment):
    """
    The graph node whose source *is* the given fragment.
    """
    return next(node for node in projection.nodes if node.source is fragment)


def children_of(projection, node_identifier):
    return [
        relation.target
        for relation in projection.relations
        if relation.source.identifier == node_identifier
    ]


def group_member_sources(projection, parent_identifier, label_prefix):
    """
    The set of ``.source`` objects under the one child group of ``parent_identifier``
    whose label starts with ``label_prefix``, or ``None`` if no such group exists.
    """
    group = next(
        (
            child
            for child in children_of(projection, parent_identifier)
            if child.label.startswith(label_prefix)
        ),
        None,
    )
    if group is None:
        return None
    return {child.source for child in children_of(projection, group.identifier)}


# %% root
class TestRoot:
    def test_root_label_is_flanagan_model(self, bare_projection):
        PhaseHierarchyOverlay().apply(bare_projection)

        assert bare_projection.root.label == ROOT_LABEL

    def test_root_groups_every_phase_together(self, bare_projection, structure):
        # the sample's parameters are never linked to a phase via
        # parameter_in_phase (see TestOrphanSafetyNet), so the root legitimately also
        # carries a catch-all group alongside "Phases" -- this only asserts the
        # phases branch itself, not that it is the root's only child
        PhaseHierarchyOverlay().apply(bare_projection)

        phases_group = next(
            child
            for child in children_of(bare_projection, bare_projection.root.identifier)
            if child.label.startswith("Phases")
        )
        assert phases_group.label == "Phases (%d)" % len(structure.defines_phase)
        assert group_member_sources(
            bare_projection, bare_projection.root.identifier, "Phases"
        ) == set(structure.defines_phase)


# %% real per-phase fields, mirrored exactly
class TestRealPhaseFields:
    #: (group label prefix, ActionPhase field name)
    GROUPS = (
        ("Role Takers", "phase_involves"),
        ("Conditions", "has_condition_requirement"),
        ("Monitored Signals", "monitors_signal"),
        ("Governing Models", "governed_by"),
        ("Initiated By", "initiated_by"),
        ("Terminated By", "terminated_by"),
    )

    def test_every_phase_group_mirrors_its_real_field_exactly(
        self, bare_projection, structure
    ):
        PhaseHierarchyOverlay().apply(bare_projection)

        for phase in structure.defines_phase:
            phase_node = node_for(bare_projection, phase)
            for label_prefix, field_name in self.GROUPS:
                expected = set(getattr(phase, field_name)) or None
                actual = group_member_sources(
                    bare_projection, phase_node.identifier, label_prefix
                )
                assert actual == expected, "%s / %s: expected %r, got %r" % (
                    phase.label,
                    label_prefix,
                    expected,
                    actual,
                )

    def test_subgoal_stays_a_direct_ungrouped_edge(self, bare_projection, structure):
        # aims_for is functional (single-valued), unlike the list fields above, so it
        # is not wrapped in a group -- matching how the projector itself only groups
        # list-valued fields
        PhaseHierarchyOverlay().apply(bare_projection)

        phase = structure.defines_phase[0]
        phase_node = node_for(bare_projection, phase)
        direct_children = {
            child.source
            for child in children_of(bare_projection, phase_node.identifier)
        }
        assert phase.aims_for in direct_children


# %% synthetic reverse-pointer categories
class TestSyntheticPhaseFields:
    def test_force_couplings_attach_exactly_where_active(
        self, bare_projection, structure
    ):
        PhaseHierarchyOverlay().apply(bare_projection)

        for phase in structure.defines_phase:
            phase_node = node_for(bare_projection, phase)
            expected = {
                coupling
                for coupling in structure.has_force_coupling
                if phase in coupling.coupling_active_in
            } or None
            assert (
                group_member_sources(
                    bare_projection, phase_node.identifier, "Force Couplings"
                )
                == expected
            )

    def test_continuous_predictions_attach_exactly_where_predicted(
        self, bare_projection, structure
    ):
        PhaseHierarchyOverlay().apply(bare_projection)

        for phase in structure.defines_phase:
            phase_node = node_for(bare_projection, phase)
            expected = {
                prediction
                for prediction in structure.has_continuous_prediction
                if phase in prediction.predicted_for_phase
            } or None
            assert (
                group_member_sources(
                    bare_projection, phase_node.identifier, "Continuous Predictions"
                )
                == expected
            )

    def test_control_points_attach_via_shared_boundary_event(
        self, bare_projection, structure
    ):
        PhaseHierarchyOverlay().apply(bare_projection)

        for phase in structure.defines_phase:
            phase_node = node_for(bare_projection, phase)
            boundary_events = set(phase.initiated_by) | set(phase.terminated_by)
            expected = {
                control_point
                for control_point in structure.has_control_point
                if control_point.at_event_type in boundary_events
            } or None
            assert (
                group_member_sources(
                    bare_projection, phase_node.identifier, "Control Points"
                )
                == expected
            )

    def test_transitions_attach_to_both_their_endpoints(
        self, bare_projection, structure
    ):
        PhaseHierarchyOverlay().apply(bare_projection)

        for phase in structure.defines_phase:
            phase_node = node_for(bare_projection, phase)
            expected = {
                transition
                for transition in structure.has_transition
                if transition.from_phase is phase or transition.to_phase is phase
            } or None
            assert (
                group_member_sources(
                    bare_projection, phase_node.identifier, "Transitions"
                )
                == expected
            )


# %% clicking a shared node must not reveal its own owning phase back as a "child"
class TestPhaseBackreferencesAreStripped:
    def test_a_transition_no_longer_points_back_at_its_own_phases(
        self, bare_projection, structure
    ):
        PhaseHierarchyOverlay().apply(bare_projection)

        for transition in structure.has_transition:
            transition_node = node_for(bare_projection, transition)
            children = {
                child.source
                for child in children_of(bare_projection, transition_node.identifier)
            }
            assert transition.from_phase not in children
            assert transition.to_phase not in children

    def test_a_force_coupling_no_longer_points_back_at_its_active_phases(
        self, bare_projection, structure
    ):
        PhaseHierarchyOverlay().apply(bare_projection)

        for coupling in structure.has_force_coupling:
            coupling_node = node_for(bare_projection, coupling)
            children = {
                child.source
                for child in children_of(bare_projection, coupling_node.identifier)
            }
            assert children.isdisjoint(set(coupling.coupling_active_in))

    def test_a_continuous_prediction_no_longer_points_back_at_its_phases(
        self, bare_projection, structure
    ):
        PhaseHierarchyOverlay().apply(bare_projection)

        for prediction in structure.has_continuous_prediction:
            prediction_node = node_for(bare_projection, prediction)
            children = {
                child.source
                for child in children_of(bare_projection, prediction_node.identifier)
            }
            assert children.isdisjoint(set(prediction.predicted_for_phase))

    def test_a_transitions_other_real_fields_survive_the_stripping(
        self, bare_projection, structure
    ):
        # only the phase back-reference is removed -- gated_by (a real, non-phase
        # field) must still be there
        PhaseHierarchyOverlay().apply(bare_projection)

        transition = structure.has_transition[0]
        transition_node = node_for(bare_projection, transition)
        children = {
            child.source
            for child in children_of(bare_projection, transition_node.identifier)
        }
        assert set(transition.gated_by) <= children


# %% parameters (not exercised by the checked-in sample -- a minimal fixture instead)
class TestParameterAttachment:
    def test_a_parameter_attaches_to_every_phase_naming_it(self):
        phase = ReachPhase(
            label="Reach",
            aims_for=None,
            terminated_by=[],
            phase_involves=[],
        )
        other_phase = ReachPhase(
            label="Other", aims_for=None, terminated_by=[], phase_involves=[]
        )
        parameter = GripForceSetpoint(label="grip force", parameter_in_phase=[phase])
        structure = PouringPhaseStructure(
            label="",
            decomposes_task=ConceptReference(iri="urn:test:task", label="test"),
            defines_phase=[phase, other_phase],
            defines_parameter=[parameter],
        )
        record = build_hypothesis_record(structure, instruction="test")
        projection = build_projection(record, instruction="test", apply_overlays=False)

        PhaseHierarchyOverlay().apply(projection)

        phase_node = node_for(projection, phase)
        other_phase_node = node_for(projection, other_phase)
        assert group_member_sources(
            projection, phase_node.identifier, "Parameters"
        ) == {parameter}
        assert (
            group_member_sources(projection, other_phase_node.identifier, "Parameters")
            is None
        )


# %% orphan safety net
class TestOrphanSafetyNet:
    def test_a_role_no_phase_involves_lands_in_the_catch_all_group(self):
        involved_role = RoleSlot(
            label="involved",
            binds_as=ConceptReference(iri="urn:test:RoleA", label="RoleA"),
        )
        unreferenced_role = RoleSlot(
            label="unreferenced",
            binds_as=ConceptReference(iri="urn:test:RoleB", label="RoleB"),
        )
        phase = ReachPhase(
            label="Reach",
            aims_for=None,
            terminated_by=[],
            phase_involves=[involved_role],
        )
        structure = ActionPhaseStructure(
            label="",
            decomposes_task=ConceptReference(iri="urn:test:task", label="test"),
            defines_phase=[phase],
            defines_role=[involved_role, unreferenced_role],
        )
        record = build_hypothesis_record(structure, instruction="test")
        projection = build_projection(record, instruction="test", apply_overlays=False)

        PhaseHierarchyOverlay().apply(projection)

        assert group_member_sources(
            projection, projection.root.identifier, "Other Structure Knowledge"
        ) == {unreferenced_role}

    def test_no_catch_all_group_when_everything_is_reachable(self):
        role = RoleSlot(
            label="involved",
            binds_as=ConceptReference(iri="urn:test:RoleA", label="RoleA"),
        )
        phase = ReachPhase(
            label="Reach", aims_for=None, terminated_by=[], phase_involves=[role]
        )
        structure = ActionPhaseStructure(
            label="",
            decomposes_task=ConceptReference(iri="urn:test:task", label="test"),
            defines_phase=[phase],
            defines_role=[role],
        )
        record = build_hypothesis_record(structure, instruction="test")
        projection = build_projection(record, instruction="test", apply_overlays=False)

        PhaseHierarchyOverlay().apply(projection)

        assert (
            group_member_sources(
                projection, projection.root.identifier, "Other Structure Knowledge"
            )
            is None
        )

    def test_every_real_fragment_stays_reachable_from_root_for_the_sample(
        self, bare_projection
    ):
        # the invariant the safety net exists to guarantee, checked with the same
        # reachability primitive the overlay itself uses -- not by hand-deriving
        # which of the sample's ~90 fragments end up in the catch-all group
        PhaseHierarchyOverlay().apply(bare_projection)

        reachable = rx.descendants(bare_projection.graph, bare_projection.root_index)
        for node in bare_projection.nodes:
            if node.source is None or node is bare_projection.root:
                continue
            index = next(
                i
                for i in bare_projection.graph.node_indices()
                if bare_projection.graph[i] is node
            )
            assert index in reachable, "%s is unreachable from the root" % node.label


# %% role slots show their bound type, not the uniform class name
class TestRoleSlotLabels:
    def test_every_role_slot_shows_its_bound_types_spaced_label(
        self, bare_projection, structure
    ):
        PhaseHierarchyOverlay().apply(bare_projection)

        for role in structure.defines_role:
            node = node_for(bare_projection, role)
            assert node.label == role.binds_as.label

    def test_a_specific_role_slot_shows_the_readable_type_name(
        self, bare_projection, structure
    ):
        PhaseHierarchyOverlay().apply(bare_projection)

        source_container = next(
            role
            for role in structure.defines_role
            if role_type_name(role) == "SourceContainer"
        )
        node = node_for(bare_projection, source_container)

        assert node.label == "Source Container"

    def test_a_role_takers_group_shows_the_relabelled_text_too(
        self, bare_projection, structure
    ):
        # regression test: a HypothesisGraphRelation snapshots its target node at
        # edge-creation time, so a "Role Takers" edge built before relabelling
        # would keep showing the pre-relabel text forever -- see apply()'s
        # ordering comment
        PhaseHierarchyOverlay().apply(bare_projection)

        phase = next(
            phase
            for phase in structure.defines_phase
            if any(
                role.binds_as.label == "Source Container"
                for role in phase.phase_involves
            )
        )
        phase_node = node_for(bare_projection, phase)
        role_takers_group = next(
            child
            for child in children_of(bare_projection, phase_node.identifier)
            if child.label.startswith("Role Takers")
        )
        role_taker_labels = {
            grandchild.label
            for grandchild in children_of(bare_projection, role_takers_group.identifier)
        }
        assert "Source Container" in role_taker_labels


# %% role slots ground to the instruction via FrameModel's own grounded candidates
class TestRoleSlotGrounding:
    def test_every_role_slot_matches_frame_models_own_grounding_decision(
        self, bare_projection, structure
    ):
        PhaseHierarchyOverlay().apply(bare_projection)

        template = MinskyActionFrameTemplate.cereal_pouring()
        instruction = bare_projection.instruction.label.lower()
        for role in structure.defines_role:
            entity = template.entity_for_role(role)
            grounded = (
                entity is not None and entity.concrete_name.lower() in instruction
            )
            expected_label = entity.concrete_name if grounded else "?"
            expected_type = (
                ROLE_FILLER_TYPE_NAME if grounded else UNRESOLVED_ROLE_FILLER_TYPE_NAME
            )

            role_node = node_for(bare_projection, role)
            name_leaf = next(
                child for child in children_of(bare_projection, role_node.identifier)
            )
            assert name_leaf.label == expected_label
            assert name_leaf.type_name == expected_type

    def test_a_role_covered_by_frame_model_and_named_in_the_instruction_is_grounded(
        self, bare_projection, structure
    ):
        PhaseHierarchyOverlay().apply(bare_projection)

        source_container = next(
            role
            for role in structure.defines_role
            if role_type_name(role) == "SourceContainer"
        )
        role_node = node_for(bare_projection, source_container)
        (name_leaf,) = children_of(bare_projection, role_node.identifier)

        assert name_leaf.label == "Cereal Box"
        assert name_leaf.type_name == ROLE_FILLER_TYPE_NAME

    def test_a_role_covered_by_frame_model_but_not_named_stays_unresolved(self):
        # the default instruction never names an effector, matching FrameModel's
        # own Agent role staying unresolved
        structure = build_pouring_phase_structure()
        record = build_hypothesis_record(structure)
        projection = build_projection(record, apply_overlays=False)
        PhaseHierarchyOverlay().apply(projection)

        effector = next(
            role
            for role in structure.defines_role
            if role_type_name(role) == "GraspingEffector"
        )
        role_node = node_for(projection, effector)
        (name_leaf,) = children_of(projection, role_node.identifier)

        assert name_leaf.label == "?"
        assert name_leaf.type_name == UNRESOLVED_ROLE_FILLER_TYPE_NAME

    def test_a_role_with_no_frame_model_counterpart_is_always_unresolved(
        self, bare_projection, structure
    ):
        PhaseHierarchyOverlay().apply(bare_projection)

        grasp_site = next(
            role
            for role in structure.defines_role
            if role_type_name(role) == "GraspSite"
        )
        role_node = node_for(bare_projection, grasp_site)
        (name_leaf,) = children_of(bare_projection, role_node.identifier)

        assert name_leaf.label == "?"
        assert name_leaf.type_name == UNRESOLVED_ROLE_FILLER_TYPE_NAME

    def test_grounding_follows_the_instruction_actually_used_to_build_the_projection(
        self,
    ):
        structure = build_pouring_phase_structure()
        instruction = "Pour cereal from the cereal box"  # omits the bowl
        record = build_hypothesis_record(structure, instruction=instruction)
        projection = build_projection(
            record, instruction=instruction, apply_overlays=False
        )
        PhaseHierarchyOverlay().apply(projection)

        target_receptacle = next(
            role
            for role in structure.defines_role
            if role_type_name(role) == "TargetReceptacle"
        )
        role_node = node_for(projection, target_receptacle)
        (name_leaf,) = children_of(projection, role_node.identifier)

        assert name_leaf.label == "?"
        assert name_leaf.type_name == UNRESOLVED_ROLE_FILLER_TYPE_NAME

    def test_the_role_slot_stays_the_only_node_carrying_its_own_object_identity(
        self, bare_projection, structure
    ):
        # the name leaf must not also carry source=role, or the identity ->
        # rendered-node map used for EQL highlighting (built by picking the last
        # node seen for each object id) would ambiguously resolve a RoleSlot hit
        # to the tiny "?" leaf instead of the role node itself
        PhaseHierarchyOverlay().apply(bare_projection)

        for role in structure.defines_role:
            nodes_with_this_identity = [
                node for node in bare_projection.nodes if node.source is role
            ]
            assert len(nodes_with_this_identity) == 1


# %% a subgoal doesn't repeat its own phase's role takers / terminating event
class TestSubgoalRedundancy:
    def test_subgoal_drops_its_condition_on_role_and_attainment_signalled_by_edges(
        self, bare_projection, structure
    ):
        PhaseHierarchyOverlay().apply(bare_projection)

        for phase in structure.defines_phase:
            subgoal_node = node_for(bare_projection, phase.aims_for)
            predicates = {
                relation.predicate
                for relation in bare_projection.relations
                if relation.source.identifier == subgoal_node.identifier
            }
            assert "condition_on_role" not in predicates
            assert "attainment_signalled_by" not in predicates

    def test_subgoal_becomes_a_true_leaf_when_it_has_no_violation_signal(
        self, bare_projection, structure
    ):
        # the sample never sets violation_signalled_by, so once the two redundant
        # edges above are gone there is nothing left to show
        PhaseHierarchyOverlay().apply(bare_projection)

        phase = structure.defines_phase[0]
        assert phase.aims_for.violation_signalled_by is None
        subgoal_node = node_for(bare_projection, phase.aims_for)

        assert children_of(bare_projection, subgoal_node.identifier) == []

    def test_subgoal_keeps_its_violation_signal_when_the_structure_sets_one(self):
        violation_event = SlipEvent(label="slip", is_failure_event=True)
        attainment_event = ContactEstablishment(label="contact established")
        subgoal = Subgoal(
            label="Reach subgoal",
            condition_on_role=[],
            violation_signalled_by=violation_event,
            attainment_signalled_by=attainment_event,
        )
        phase = ReachPhase(
            label="Reach",
            aims_for=subgoal,
            terminated_by=[attainment_event],
            phase_involves=[],
        )
        structure = ActionPhaseStructure(
            label="",
            decomposes_task=ConceptReference(iri="urn:test:task", label="test"),
            defines_phase=[phase],
        )
        record = build_hypothesis_record(structure, instruction="test")
        projection = build_projection(record, instruction="test", apply_overlays=False)

        PhaseHierarchyOverlay().apply(projection)

        subgoal_node = node_for(projection, subgoal)
        children_sources = {
            child.source for child in children_of(projection, subgoal_node.identifier)
        }
        assert children_sources == {violation_event}
