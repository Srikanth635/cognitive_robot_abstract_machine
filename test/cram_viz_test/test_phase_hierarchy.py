"""
Tests for the phase-first presentation of a projected phase structure, on the checked-in
sample hypothesis.
"""

import pytest
import rustworkx as rx

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from llm_reasoner.hypotheses.graph import (  # noqa: E402
    HypothesisGraphNodeKind,
    HypothesisGraphPresentation,
    HypothesisGraphProjector,
    SyntheticHypothesisRelation,
)

from cram_viz import hypothesis_knowledge  # noqa: E402
from cram_viz.hypothesis_knowledge.phase_hierarchy import (  # noqa: E402
    ROOT_LABEL,
    PhaseGroup,
    PhaseHierarchyOverlay,
)

# %% helpers


@pytest.fixture(scope="module")
def view():
    return hypothesis_knowledge.sample_view()


@pytest.fixture(scope="module")
def projection(view):
    return PhaseHierarchyOverlay().apply(
        HypothesisGraphProjector().project(view, HypothesisGraphPresentation.COMPLETE),
        view,
    )


def index_of(projection, fragment):
    (index,) = [
        index
        for index in projection.graph.node_indices()
        if projection.graph[index].source is fragment
    ]
    return index


def children(projection, index):
    return [
        (relation.predicate, target)
        for _, target, relation in projection.graph.out_edges(index)
    ]


def group_members(projection, parent_index, group):
    """
    Return the fragments under the parent's group, or ``None`` without one.
    """
    groups = [
        target
        for predicate, target in children(projection, parent_index)
        if predicate == SyntheticHypothesisRelation.GROUPS
        and projection.graph[target].label.startswith(group.value + " (")
    ]
    if not groups:
        return None
    (group_index,) = groups
    return [
        projection.graph[target].source
        for _, target in children(projection, group_index)
    ]


def same_fragments(found, expected):
    return found is not None and sorted(map(id, found)) == sorted(map(id, expected))


# %% root


class TestRoot:
    def test_root_shows_the_hierarchy_label(self, projection):
        assert projection.root.label == ROOT_LABEL

    def test_root_groups_every_phase(self, projection, view):
        assert same_fragments(
            group_members(projection, projection.root_index, PhaseGroup.PHASES),
            view.root.defines_phase,
        )


# %% the phase's own fields


class TestPhaseFields:
    @pytest.mark.parametrize(
        "group, field_values",
        [
            (PhaseGroup.ROLE_TAKERS, lambda phase: phase.involves_role),
            (PhaseGroup.CONDITIONS, lambda phase: phase.has_condition_requirement),
            (PhaseGroup.PARAMETERS, lambda phase: phase.has_active_parameter),
            (PhaseGroup.INITIATED_BY, lambda phase: phase.initiated_by),
            (PhaseGroup.TERMINATED_BY, lambda phase: phase.terminated_by),
        ],
    )
    def test_each_group_holds_exactly_its_field(
        self, projection, view, group, field_values
    ):
        for phase in view.root.defines_phase:
            members = group_members(projection, index_of(projection, phase), group)
            held = list(field_values(phase))
            if not held:
                assert members is None
                continue
            assert same_fragments(members, held)

    def test_subgoal_and_controller_stay_direct_edges(self, projection, view):
        for phase in view.root.defines_phase:
            direct = {
                predicate: projection.graph[target].source
                for predicate, target in children(
                    projection, index_of(projection, phase)
                )
            }
            if phase.aims_for is not None:
                assert direct["aims_for"] is phase.aims_for
            if phase.controlled_by is not None:
                assert direct["controlled_by"] is phase.controlled_by


# %% fragments pointing at a phase


class TestFragmentsPointingAtAPhase:
    def test_transitions_attach_to_both_their_phases(self, projection, view):
        for phase in view.root.defines_phase:
            expected = [
                transition
                for transition in view.root.defines_transition
                if transition.from_phase is phase or transition.to_phase is phase
            ]
            members = group_members(
                projection, index_of(projection, phase), PhaseGroup.TRANSITIONS
            )
            assert (
                members is None if not expected else same_fragments(members, expected)
            )

    def test_force_couplings_attach_where_active(self, projection, view):
        for phase in view.root.defines_phase:
            expected = view.force_couplings_in(phase)
            members = group_members(
                projection, index_of(projection, phase), PhaseGroup.FORCE_COUPLINGS
            )
            assert (
                members is None if not expected else same_fragments(members, expected)
            )

    def test_gaze_fixations_attach_where_they_serve(self, projection, view):
        for phase in view.root.defines_phase:
            expected = view.gaze_fixations_for(phase)
            members = group_members(
                projection, index_of(projection, phase), PhaseGroup.GAZE_FIXATIONS
            )
            assert (
                members is None if not expected else same_fragments(members, expected)
            )

    def test_control_points_attach_at_the_phases_boundary_events(
        self, projection, view
    ):
        attached = 0
        for phase in view.root.defines_phase:
            boundaries = [*phase.initiated_by, *phase.terminated_by]
            expected = [
                point
                for point in view.root.defines_control_point
                if any(point.at_event_type is event for event in boundaries)
            ]
            attached += len(expected)
            members = group_members(
                projection, index_of(projection, phase), PhaseGroup.CONTROL_POINTS
            )
            assert (
                members is None if not expected else same_fragments(members, expected)
            )
        assert attached > 0


class TestBackReferencesAreRemoved:
    @pytest.mark.parametrize(
        "fragments",
        [
            lambda root: root.defines_transition,
            lambda root: root.defines_force_coupling,
            lambda root: root.defines_gaze_fixation,
        ],
    )
    def test_grouped_fragment_no_longer_points_back_at_a_phase(
        self, projection, view, fragments
    ):
        phases = {index_of(projection, phase) for phase in view.root.defines_phase}
        for fragment in fragments(view.root):
            targets = {
                target
                for _, target in children(projection, index_of(projection, fragment))
            }
            assert not targets & phases

    def test_transition_keeps_its_gating_events(self, projection, view):
        gated = [
            transition
            for transition in view.root.defines_transition
            if transition.gated_by
        ]
        assert gated
        for transition in gated:
            assert {
                id(projection.graph[target].source)
                for predicate, target in children(
                    projection, index_of(projection, transition)
                )
                if predicate == "gated_by"
            } == {id(event) for event in transition.gated_by}

    def test_subgoal_no_longer_repeats_its_signalling_event(self, projection, view):
        for phase in view.root.defines_phase:
            if phase.aims_for is None:
                continue
            predicates = {
                predicate
                for predicate, _ in children(
                    projection, index_of(projection, phase.aims_for)
                )
            }
            assert "attainment_signalled_by" not in predicates


# %% reachability


class TestReachability:
    def test_every_fragment_is_reachable_from_the_root(self, projection):
        reachable = rx.descendants(projection.graph, projection.root_index)
        unreachable = [
            projection.graph[index].identifier
            for index in projection.graph.node_indices()
            if projection.graph[index].kind is HypothesisGraphNodeKind.FRAGMENT
            and index not in reachable
        ]
        assert unreachable == []
