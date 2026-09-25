from __future__ import annotations

from cram_viz.hypothesis_knowledge import build_hypothesis_record
from cram_viz.hypothesis_knowledge.pouring_structure import (
    build_pouring_phase_structure,
)
from llm_reasoner.knowledge.microworlds.flanagan.core import Estimate
from llm_reasoner.knowledge.microworlds.flanagan.object_dynamics import (
    CoordinateFrame,
    LimbDynamicsModel,
    MemoryPersistence,
    ObjectDynamicsModel,
    PerturbationKind,
    Representation,
    RepresentationAccuracy,
    RepresentationPurpose,
    TransferOutcome,
)
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
