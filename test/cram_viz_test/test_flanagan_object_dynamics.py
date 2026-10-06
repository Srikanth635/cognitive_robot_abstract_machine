from __future__ import annotations

from cram_viz.hypothesis_knowledge import build_hypothesis_record
from cram_viz.hypothesis_knowledge.pouring_structure import (
    build_pouring_phase_structure,
)
from llm_reasoner.knowledge.hypothesis.provenance import Provenance
from llm_reasoner.knowledge.hypothesis.record import HypothesisRecord
from llm_reasoner.knowledge.microworlds.flanagan.core import ActionPhase, Estimate
from llm_reasoner.knowledge.microworlds.flanagan.gaze import (
    GazeControlProfile,
    GazeEpisode,
    GazeTarget,
)
from llm_reasoner.knowledge.microworlds.flanagan.object_dynamics import (
    CoordinateFrame,
    DynamicsModel,
    LimbDynamicsModel,
    MemoryPersistence,
    ObjectDynamicsModel,
    PerturbationKind,
    Representation,
    RepresentationAccuracy,
    RepresentationPurpose,
    TransferOutcome,
)
from llm_reasoner.knowledge.microworlds.registry import POURING_PROFILE
from llm_reasoner.knowledge.query import HypothesisView

# %% canonical control pathway


class TestCanonicalInternalModelComposition:
    """
    The canonical structure exposes the complete predictive pathway.
    """

    def test_composition_reaches_mechanical_and_sensory_predictions(self) -> None:
        structure = build_pouring_phase_structure()

        assert len(structure.has_internal_model_composition) == 1
        composition = structure.has_internal_model_composition[0]
        assert composition.motor_command.generated_by is composition.inverse_model
        assert composition.motor_command.pursues_state is composition.desired_state
        assert composition.efference_copy.copies_command is composition.motor_command
        assert (
            composition.efference_copy in composition.dynamics_model.uses_efference_copy
        )
        assert composition.predicted_mechanical_consequence
        assert composition.predicted_sensory_consequence
        assert all(
            prediction.generated_by is composition.forward_model
            for prediction in composition.predicted_sensory_consequence
        )

    def test_every_declared_estimate_has_a_representation(self) -> None:
        structure = build_pouring_phase_structure()
        represented_estimates = {
            estimate
            for model in structure.defines_internal_model
            if isinstance(model, Representation)
            for estimate in model.holds_estimate
        }
        represented_estimates.update(
            estimate
            for model in structure.defines_internal_model
            if isinstance(model, Representation)
            for estimate in model.holds_estimate
        )
        declared_estimates = {
            parameter
            for parameter in structure.defines_parameter
            if isinstance(parameter, Estimate)
        }

        assert represented_estimates == declared_estimates


# %% object and limb contexts


class TestCanonicalDynamicsDissociation:
    """
    Object and limb perturbations remain independently represented.
    """

    def test_equivalent_loads_target_object_and_limb_models(self) -> None:
        structure = build_pouring_phase_structure()
        loads = [
            perturbation
            for perturbation in structure.defines_dynamics_perturbation
            if perturbation.kind is PerturbationKind.LOAD
        ]

        assert len(loads) == 2
        assert {type(load.affects_model) for load in loads} == {
            ObjectDynamicsModel,
            LimbDynamicsModel,
        }
        assert loads[0].equivalent_to == [loads[1]]
        assert loads[1].equivalent_to == [loads[0]]

    def test_object_and_hand_rotations_have_different_frames(self) -> None:
        structure = build_pouring_phase_structure()
        rotations = [
            perturbation
            for perturbation in structure.defines_dynamics_perturbation
            if perturbation.kind is PerturbationKind.ROTATION
        ]

        assert len(rotations) == 2
        assert {rotation.coordinate_frame for rotation in rotations} == {
            CoordinateFrame.OBJECT_CENTERED,
            CoordinateFrame.HAND_CENTERED,
        }
        transfer = next(
            transfer
            for transfer in structure.has_dynamics_transfer
            if transfer.from_perturbation in rotations
        )
        assert transfer.outcome is TransferOutcome.DIFFERS

    def test_object_memory_is_derecruited_after_release(self) -> None:
        structure = build_pouring_phase_structure()

        assert len(structure.has_sensorimotor_memory) == 1
        memory = structure.has_sensorimotor_memory[0]
        assert memory.persistence is MemoryPersistence.WHILE_OBJECT_COUPLED
        assert all(
            isinstance(model, ObjectDynamicsModel)
            for model in memory.stores_representation
        )


# %% size-weight dissociation


class TestCanonicalRepresentationDissociation:
    """
    Accurate motor scaling coexists with biased perceptual judgment.
    """

    def test_sensorimotor_and_perceptual_weight_estimates_are_separate(self) -> None:
        structure = build_pouring_phase_structure()

        assert len(structure.has_representation_dissociation) == 1
        dissociation = structure.has_representation_dissociation[0]
        assert (
            dissociation.sensorimotor_representation.purpose
            is RepresentationPurpose.SENSORIMOTOR_CONTROL
        )
        assert (
            dissociation.perceptual_representation.purpose
            is RepresentationPurpose.PERCEPTUAL_JUDGMENT
        )
        assert (
            dissociation.sensorimotor_estimate is not dissociation.perceptual_estimate
        )
        assert dissociation.sensorimotor_accuracy is RepresentationAccuracy.ACCURATE
        assert dissociation.perceptual_accuracy is RepresentationAccuracy.BIASED


# %% typed queries


class TestDynamicsQueries:
    """
    Typed queries retrieve dynamics and purpose without label parsing.
    """

    def test_object_and_limb_models_are_queried_independently(self) -> None:
        structure = build_pouring_phase_structure()
        view = HypothesisView(build_hypothesis_record(structure))

        assert [type(answer.value) for answer in view.object_dynamics()] == [
            ObjectDynamicsModel
        ]
        assert [type(answer.value) for answer in view.limb_dynamics()] == [
            LimbDynamicsModel
        ]

    def test_representation_query_filters_by_estimate_and_purpose(self) -> None:
        structure = build_pouring_phase_structure()
        dissociation = structure.has_representation_dissociation[0]
        view = HypothesisView(build_hypothesis_record(structure))

        answers = view.representation_for(
            dissociation.perceptual_estimate,
            RepresentationPurpose.PERCEPTUAL_JUDGMENT,
        )

        assert [answer.value for answer in answers] == [
            dissociation.perceptual_representation
        ]


class TestPredictiveControlQueries:
    """
    Typed queries expose the complete prediction-correction control path.
    """

    def test_control_point_returns_expected_events_and_comparisons(self) -> None:
        structure = build_pouring_phase_structure()
        control_point = structure.has_control_point[0]
        view = HypothesisView(build_hypothesis_record(structure))

        assert [answer.value for answer in view.expected_events_at(control_point)] == [
            prediction.predicts_event
            for prediction in control_point.has_prediction
            if prediction.predicts_event is not None
        ]
        assert [answer.value for answer in view.comparison_at(control_point)] == list(
            control_point.has_comparison
        )

    def test_prediction_error_returns_its_corrections_and_model_updates(
        self,
    ) -> None:
        structure = build_pouring_phase_structure()
        comparison = next(
            candidate for candidate in structure.has_comparison if candidate.has_error
        )
        event = comparison.compares_prediction.predicts_event
        view = HypothesisView(build_hypothesis_record(structure))
        expected_errors = [
            error
            for candidate in structure.has_comparison
            if candidate.compares_prediction is not None
            and candidate.compares_prediction.predicts_event is event
            for error in candidate.has_error
        ]

        errors = view.prediction_errors_for(event)

        assert [answer.value for answer in errors] == expected_errors
        error = errors[0]
        assert [answer.value for answer in view.corrections_for(error)] == [
            correction
            for correction in structure.defines_correction
            if error.value in correction.triggered_by_error
        ]
        assert [answer.value for answer in view.model_updates_for(error)] == [
            update
            for update in structure.has_model_update
            if error.value in update.triggered_by_error
        ]

    def test_successful_comparison_returns_the_next_phase(self) -> None:
        structure = build_pouring_phase_structure()
        comparison = next(
            candidate
            for candidate in structure.has_comparison
            if candidate.permits_transition
        )
        view = HypothesisView(build_hypothesis_record(structure))

        assert [answer.value for answer in view.next_phase_after(comparison)] == [
            transition.to_phase for transition in comparison.permits_transition
        ]

    def test_phase_returns_dynamics_force_and_continuous_control(self) -> None:
        structure = build_pouring_phase_structure()
        view = HypothesisView(build_hypothesis_record(structure))
        dynamics_phase = next(
            phase
            for phase in structure.defines_phase
            if any(isinstance(model, DynamicsModel) for model in phase.governed_by)
        )
        force_phase = structure.has_force_coupling[0].coupling_active_in[0]
        continuous_phase = structure.has_continuous_prediction[0].predicted_for_phase[0]

        assert [answer.value for answer in view.dynamics_model_for(dynamics_phase)] == [
            model
            for model in dynamics_phase.governed_by
            if isinstance(model, DynamicsModel)
        ]
        assert [answer.value for answer in view.force_coupling_for(force_phase)] == [
            coupling
            for coupling in structure.has_force_coupling
            if force_phase in coupling.coupling_active_in
        ]
        assert [
            answer.value for answer in view.continuous_predictions_for(continuous_phase)
        ] == [
            prediction
            for prediction in structure.has_continuous_prediction
            if continuous_phase in prediction.predicted_for_phase
        ]


class TestGazeQueries:
    """
    Typed gaze queries resolve targets through their phase relation.
    """

    def test_phase_returns_its_gaze_target(self) -> None:
        phase = ActionPhase(local_id="reach-phase")
        target = GazeTarget(local_id="contact-target", target_for_phase=[phase])
        profile = GazeControlProfile(
            local_id="gaze-profile",
            episodes=[GazeEpisode(local_id="gaze-episode", target=target)],
        )
        record = HypothesisRecord.from_root(
            profile,
            action_type=POURING_PROFILE.task,
            provenance=Provenance(
                generator="test",
                prompt_version="test",
            ),
        )
        view = HypothesisView(record)

        answers = view.gaze_target_for(phase)

        assert [answer.value for answer in answers] == [target]
