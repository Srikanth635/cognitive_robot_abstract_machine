"""
Tests for the hand-authored, EQL-queryable cereal-pouring PouringPhaseStructure and its
live graph projection.
"""

import pytest

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from cram_viz.hypothesis_knowledge.pouring_structure import (  # noqa: E402
    build_pouring_phase_structure,
)
from llm_reasoner.knowledge.hypothesis.graph import (  # noqa: E402
    HypothesisGraphNodeKind,
)
from llm_reasoner.knowledge.hypothesis.record import HypothesisRecord  # noqa: E402
from llm_reasoner.knowledge.hypothesis.scope import ScopeViolation  # noqa: E402
from llm_reasoner.knowledge.microworlds.flanagan.constraints import SUITE  # noqa: E402
from llm_reasoner.knowledge.microworlds.flanagan.core import (  # noqa: E402
    ActionPhase,
    ComparisonOutcome,
    CorrectionDirection,
    ContinuousSignalPrediction,
    EstimateUpdateDirection,
    MechanicalEventType,
    PhaseTransition,
    PredictionErrorKind,
    RecoveryMode,
    RegulationMode,
    RoleSlot,
    SensorimotorControlPoint,
)
from llm_reasoner.knowledge.microworlds.flanagan.pouring import (  # noqa: E402
    PouringPhaseStructure,
)
from llm_reasoner.knowledge.microworlds.flanagan.prehension import (  # noqa: E402
    ContactEstablishment,
    ExpectedFrictionCoefficient,
    ExpectedMass,
    GripForceSetpoint,
    GripLoadCoupling,
    Liftoff,
    LoadForceRateSetpoint,
    LoadForceSetpoint,
    SafeContactForce,
    SlipEvent,
    TransportPhase,
)
from llm_reasoner.knowledge.query import HypothesisView  # noqa: E402

from cram_viz import hypothesis_knowledge  # noqa: E402

# %% acceptance


class TestReferenceStructureAcceptance:
    """
    The checked-in reference knowledge satisfies its domain contract.
    """

    def test_complete_structure_has_no_acceptance_violations(self) -> None:
        structure = build_pouring_phase_structure()
        record = hypothesis_knowledge.build_hypothesis_record(structure)

        violations = SUITE.evaluate(record.scope.fragments)

        assert violations == []


class TestReferenceRegulationAndTermination:
    """
    Regulation and phase boundaries remain independent model dimensions.
    """

    def test_hybrid_transport_phase_still_ends_at_a_discrete_event(self) -> None:
        structure = build_pouring_phase_structure()
        transport = next(
            phase
            for phase in structure.defines_phase
            if isinstance(phase, TransportPhase)
        )

        assert transport.regulation_mode is RegulationMode.HYBRID
        assert [event.label for event in transport.terminated_by] == [
            "pour pose reached"
        ]


# %% content


class TestPouringPhaseStructureContent:
    def test_returns_a_real_pouring_phase_structure(self):
        structure = build_pouring_phase_structure()
        assert isinstance(structure, PouringPhaseStructure)

    def test_has_real_phases(self):
        structure = build_pouring_phase_structure()
        assert structure.defines_phase
        assert all(isinstance(phase, ActionPhase) for phase in structure.defines_phase)

    def test_has_real_roles(self):
        structure = build_pouring_phase_structure()
        assert structure.defines_role
        assert all(isinstance(role, RoleSlot) for role in structure.defines_role)

    def test_has_real_event_types(self):
        structure = build_pouring_phase_structure()
        assert structure.defines_event_type
        assert all(
            isinstance(event, MechanicalEventType)
            for event in structure.defines_event_type
        )

    def test_has_real_transitions_chaining_the_phases(self):
        structure = build_pouring_phase_structure()
        assert structure.has_transition
        assert all(
            isinstance(transition, PhaseTransition)
            for transition in structure.has_transition
        )
        for transition in structure.has_transition:
            assert transition.from_phase in structure.defines_phase
            assert transition.to_phase in structure.defines_phase

    def test_has_real_control_points(self):
        structure = build_pouring_phase_structure()
        assert structure.has_control_point
        assert all(
            isinstance(point, SensorimotorControlPoint)
            for point in structure.has_control_point
        )

    def test_has_a_real_force_coupling(self):
        structure = build_pouring_phase_structure()
        assert structure.has_force_coupling
        assert all(
            isinstance(coupling, GripLoadCoupling)
            for coupling in structure.has_force_coupling
        )

    def test_has_real_continuous_predictions(self):
        structure = build_pouring_phase_structure()
        assert structure.has_continuous_prediction
        assert all(
            isinstance(prediction, ContinuousSignalPrediction)
            for prediction in structure.has_continuous_prediction
        )

    def test_every_transition_is_gated_by_a_real_event(self):
        structure = build_pouring_phase_structure()
        for transition in structure.has_transition:
            assert transition.gated_by
            for event in transition.gated_by:
                assert event in structure.defines_event_type

    def test_every_phase_aims_for_a_subgoal_signalled_by_a_real_event(self):
        structure = build_pouring_phase_structure()
        for phase in structure.defines_phase:
            assert phase.aims_for is not None
            assert phase.aims_for.attainment_signalled_by is not None

    def test_no_phase_is_both_initiated_and_terminated_by_the_same_event(self):
        # a phase bounded by the same event at both ends has zero duration, which
        # contradicts every phase's own description (e.g. Hold: "maintenance of a
        # stable grasp" is not instantaneous)
        structure = build_pouring_phase_structure()
        for phase in structure.defines_phase:
            assert set(phase.initiated_by).isdisjoint(phase.terminated_by), phase.label

    def test_release_stages_show_their_unambiguous_names(self):
        structure = build_pouring_phase_structure()
        labels_by_type = {
            type(phase).__name__: phase.label for phase in structure.defines_phase
        }

        assert labels_by_type["UnloadPhase"] == "Unload"
        assert labels_by_type["ContactReleasePhase"] == "ContactRelease"
        assert labels_by_type["WithdrawPhase"] == "Withdraw"

    def test_every_parameter_is_linked_to_at_least_one_phase(self):
        structure = build_pouring_phase_structure()
        for parameter in structure.defines_parameter:
            assert parameter.parameter_in_phase, parameter.label
            for phase in parameter.parameter_in_phase:
                assert phase in structure.defines_phase

    def test_every_phase_model_is_declared_by_the_structure(self) -> None:
        structure = build_pouring_phase_structure()
        governed_models = {
            model for phase in structure.defines_phase for model in phase.governed_by
        }

        assert governed_models <= set(structure.defines_internal_model)

    def test_the_spill_event_is_reachable_as_a_conditions_violation(self):
        structure = build_pouring_phase_structure()
        spill_event = next(
            event
            for event in structure.defines_event_type
            if type(event).__name__ == "SpillEvent"
        )
        conditions = [
            requirement.on_condition
            for phase in structure.defines_phase
            for requirement in phase.has_condition_requirement
        ]
        assert any(
            condition.violation_signalled_by is spill_event for condition in conditions
        )


# %% predictive-control scenarios


class TestReferencePredictiveControl:
    """
    The reference structure covers normal and paper-grounded error responses.
    """

    def test_declares_every_predictive_control_fragment_at_the_structure_root(
        self,
    ) -> None:
        structure = build_pouring_phase_structure()

        assert structure.defines_prediction
        assert structure.defines_observation
        assert structure.has_comparison
        assert structure.defines_error
        assert structure.defines_correction
        assert structure.has_model_update
        assert structure.defines_initial_state_estimate

    def test_successful_comparison_permits_a_transition_and_supplies_its_state(
        self,
    ) -> None:
        structure = build_pouring_phase_structure()
        gated_transitions = [
            transition
            for transition in structure.has_transition
            if transition.permitted_by
        ]

        assert gated_transitions
        for transition in gated_transitions:
            for comparison in transition.permitted_by:
                assert comparison.outcome is ComparisonOutcome.MATCH
                assert transition in comparison.permits_transition
                assert comparison.confirms_subgoal is transition.from_phase.aims_for
                assert (
                    comparison.supplies_state_estimate.for_phase is transition.to_phase
                )

    def test_absent_liftoff_increases_load_force_and_expected_mass(self) -> None:
        structure = build_pouring_phase_structure()
        error = next(
            candidate
            for candidate in structure.defines_error
            if candidate.kind is PredictionErrorKind.ABSENT
        )
        comparison = next(
            candidate
            for candidate in structure.has_comparison
            if error in candidate.has_error
            and isinstance(candidate.compares_prediction.predicts_event, Liftoff)
        )
        correction = next(
            candidate
            for candidate in comparison.invokes_correction
            if error in candidate.triggered_by_error
        )
        update = next(
            candidate
            for candidate in comparison.produces_model_update
            if error in candidate.triggered_by_error
        )

        assert comparison.compares_observation is None
        assert correction.direction is CorrectionDirection.INCREASE
        assert correction.recovery_mode is RecoveryMode.REPEAT_PHASE
        assert all(
            isinstance(setpoint, LoadForceSetpoint)
            for setpoint in correction.targets_setpoint
        )
        assert update.direction is EstimateUpdateDirection.INCREASE
        assert all(
            isinstance(estimate, ExpectedMass) for estimate in update.updates_estimate
        )
        assert all(
            estimate in update.updates_representation.holds_estimate
            for estimate in update.updates_estimate
        )

    def test_early_liftoff_reduces_future_load_force_rate(self) -> None:
        structure = build_pouring_phase_structure()
        error = next(
            candidate
            for candidate in structure.defines_error
            if candidate.kind is PredictionErrorKind.EARLY
        )
        comparison = next(
            candidate
            for candidate in structure.has_comparison
            if error in candidate.has_error
        )
        correction = comparison.invokes_correction[0]

        assert isinstance(comparison.compares_observation.observes_event, Liftoff)
        assert correction.direction is CorrectionDirection.DECREASE
        assert correction.recovery_mode is RecoveryMode.CONTINUE_PHASE
        assert all(
            isinstance(setpoint, LoadForceRateSetpoint)
            for setpoint in correction.targets_setpoint
        )

    def test_unexpected_slip_increases_grip_force_and_revises_friction(
        self,
    ) -> None:
        structure = build_pouring_phase_structure()
        comparison = next(
            candidate
            for candidate in structure.has_comparison
            if candidate.has_error
            and candidate.has_error[0].kind is PredictionErrorKind.UNEXPECTED
            and candidate.compares_observation is not None
            and isinstance(candidate.compares_observation.observes_event, SlipEvent)
        )
        correction = comparison.invokes_correction[0]
        update = comparison.produces_model_update[0]

        assert correction.direction is CorrectionDirection.INCREASE
        assert all(
            isinstance(setpoint, GripForceSetpoint)
            for setpoint in correction.targets_setpoint
        )
        assert update.direction is EstimateUpdateDirection.DECREASE
        assert all(
            isinstance(estimate, ExpectedFrictionCoefficient)
            for estimate in update.updates_estimate
        )

    def test_unexpected_contact_stops_motion_to_restore_safe_contact(self) -> None:
        structure = build_pouring_phase_structure()
        comparison = next(
            candidate
            for candidate in structure.has_comparison
            if candidate.has_error
            and candidate.has_error[0].kind is PredictionErrorKind.UNEXPECTED
            and candidate.compares_observation is not None
            and isinstance(
                candidate.compares_observation.observes_event,
                ContactEstablishment,
            )
        )
        correction = comparison.invokes_correction[0]

        assert correction.direction is CorrectionDirection.STOP
        assert correction.recovery_mode is RecoveryMode.REPLAN
        assert all(
            isinstance(effect, SafeContactForce)
            for effect in correction.intended_effect
        )

    def test_knowledge_view_returns_typed_corrections_for_a_control_event(
        self,
    ) -> None:
        structure = build_pouring_phase_structure()
        record = hypothesis_knowledge.build_hypothesis_record(structure)
        view = HypothesisView(record)
        liftoff = next(
            event
            for event in structure.defines_event_type
            if isinstance(event, Liftoff)
        )
        expected = [
            correction
            for control_point in structure.has_control_point
            if control_point.at_event_type is liftoff
            for comparison in control_point.has_comparison
            if comparison.outcome is ComparisonOutcome.MISMATCH
            for correction in comparison.invokes_correction
        ]

        answers = view.corrective_for(liftoff)

        assert [answer.value for answer in answers] == expected

    def test_knowledge_view_identifies_the_structure_declaring_a_correction(
        self,
    ) -> None:
        structure = build_pouring_phase_structure()
        record = hypothesis_knowledge.build_hypothesis_record(structure)
        view = HypothesisView(record)

        owner = view.declares(structure.defines_correction[0])

        assert owner is not None
        assert owner.value is structure


class TestPouringPhaseStructureProjection:
    def test_wrapping_in_a_hypothesis_record_does_not_raise_scope_violation(self):
        structure = build_pouring_phase_structure()
        try:
            hypothesis_knowledge.build_hypothesis_record(structure)
        except ScopeViolation as error:
            pytest.fail(
                f"hand-authored structure violates the knowledge scope: {error}"
            )

    def test_record_root_resolves_back_to_the_structure(self):
        structure = build_pouring_phase_structure()
        record = hypothesis_knowledge.build_hypothesis_record(structure)
        assert isinstance(record, HypothesisRecord)
        assert record.root is structure

    def test_projection_root_node_wraps_the_structure(self):
        structure = build_pouring_phase_structure()
        record = hypothesis_knowledge.build_hypothesis_record(structure)
        projection = hypothesis_knowledge.build_projection(record)
        assert projection.root.source is structure
        assert projection.root.kind is HypothesisGraphNodeKind.ROOT

    def test_projection_contains_every_real_phase(self):
        structure = build_pouring_phase_structure()
        record = hypothesis_knowledge.build_hypothesis_record(structure)
        projection = hypothesis_knowledge.build_projection(record)
        projected_sources = {node.source for node in projection.nodes}
        assert set(structure.defines_phase) <= projected_sources


@pytest.mark.usefixtures("sample_hypothesis_knowledge")
class TestActiveKnowledge:
    def test_the_active_structure_is_a_real_pouring_phase_structure(self):
        assert isinstance(
            hypothesis_knowledge.active().structure, PouringPhaseStructure
        )

    def test_the_projection_root_is_the_active_structure(self):
        assert (
            hypothesis_knowledge.active().projection.root.source
            is hypothesis_knowledge.active().structure
        )

    def test_node_identifier_lookup_resolves_a_real_phase(self):
        phase = hypothesis_knowledge.active().structure.defines_phase[0]
        identifier = hypothesis_knowledge.node_identifier_for(phase)
        assert identifier is not None
        node_ids = {
            node.identifier for node in hypothesis_knowledge.active().projection.nodes
        }
        assert identifier in node_ids

    def test_node_identifier_lookup_returns_none_for_an_unrelated_object(self):
        assert hypothesis_knowledge.node_identifier_for(object()) is None


@pytest.mark.usefixtures("sample_hypothesis_knowledge")
class TestDisplayOnlyRepresentations:
    def test_projection_carries_all_three_representations(self):
        type_names = {
            node.type_name for node in hypothesis_knowledge.active().projection.nodes
        }
        assert "PouringPhaseStructure" in type_names
        assert "MinskyActionFrame" in type_names
        assert "ImageSchemaProfile" in type_names

    def test_display_only_nodes_stay_out_of_every_eql_domain(self):
        """
        The action-frame and image-schema branches are shown but never queried
        (decision: ActionPhases only), so none of their nodes may be a fragment the
        structure itself declares.
        """
        knowledge = hypothesis_knowledge.active()
        structure = knowledge.structure
        queryable = {
            id(fragment)
            for fragment in (
                *structure.defines_phase,
                *structure.defines_role,
                *structure.defines_event_type,
                *structure.has_transition,
                *structure.has_control_point,
                *structure.has_force_coupling,
                *structure.has_continuous_prediction,
            )
        }
        display_only_types = {"MinskyActionFrame", "EntityFrame", "ImageSchemaProfile"}
        display_only_nodes = [
            node
            for node in knowledge.projection.nodes
            if node.type_name in display_only_types
        ]
        assert display_only_nodes
        assert all(id(node.source) not in queryable for node in display_only_nodes)


class TestNothingActiveUntilGenerated:
    def test_no_knowledge_is_active_before_an_instruction_is_generated(self):
        assert hypothesis_knowledge.active() is None

    def test_node_identifier_lookup_is_answerable_with_nothing_active(self):
        assert hypothesis_knowledge.node_identifier_for(object()) is None

    def test_activating_the_sample_makes_it_the_active_knowledge(
        self, sample_hypothesis_knowledge
    ):
        assert hypothesis_knowledge.active() is sample_hypothesis_knowledge
        assert not sample_hypothesis_knowledge.generated
