"""
Acceptance constraints for represented object and limb dynamics.
"""

from __future__ import annotations

from llm_reasoner.knowledge.microworlds.flanagan.core import Estimate
from llm_reasoner.knowledge.microworlds.flanagan.object_dynamics import (
    DynamicsTransfer,
    InternalModelComposition,
    LimbDynamicsModel,
    ObjectDynamicsModel,
    ObjectDynamicsPhaseStructure,
    PerceptualRepresentation,
    Representation,
    RepresentationAccuracy,
    RepresentationDissociation,
    RepresentationPurpose,
    RepresentationUpdate,
)
from llm_reasoner.knowledge.ontological.constraints import Closure, Forbidden

# %% constraint predicates


def composition_is_incoherent(composition: InternalModelComposition) -> bool:
    """
    Return whether a command-to-prediction pathway breaks identity links.
    """
    if (
        composition.desired_state is None
        or composition.inverse_model is None
        or composition.motor_command is None
        or composition.efference_copy is None
        or composition.dynamics_model is None
        or composition.forward_model is None
    ):
        return False
    if composition.motor_command.generated_by is not composition.inverse_model:
        return True
    if composition.motor_command.pursues_state is not composition.desired_state:
        return True
    if composition.efference_copy.copies_command is not composition.motor_command:
        return True
    if (
        isinstance(composition.dynamics_model, ObjectDynamicsModel)
        and composition.efference_copy
        not in composition.dynamics_model.uses_efference_copy
    ):
        return True
    return any(
        prediction.generated_by is not composition.forward_model
        for prediction in composition.predicted_sensory_consequence
    )


def structure_has_unowned_estimate(
    structure: ObjectDynamicsPhaseStructure,
) -> bool:
    """
    Return whether a declared estimate belongs to no representation.
    """
    representations = [
        model
        for model in structure.defines_internal_model
        if isinstance(model, Representation)
    ]
    return any(
        isinstance(parameter, Estimate)
        and not any(
            parameter in representation.holds_estimate
            for representation in representations
        )
        for parameter in structure.defines_parameter
    )


def structure_reuses_estimate_across_purposes(
    structure: ObjectDynamicsPhaseStructure,
) -> bool:
    """
    Return whether one estimate silently serves incompatible purposes.
    """
    representations = [
        model
        for model in structure.defines_internal_model
        if isinstance(model, Representation)
    ]
    for sensorimotor in representations:
        if sensorimotor.purpose is not RepresentationPurpose.SENSORIMOTOR_CONTROL:
            continue
        for perceptual in representations:
            if perceptual.purpose is not RepresentationPurpose.PERCEPTUAL_JUDGMENT:
                continue
            if any(
                estimate in perceptual.holds_estimate
                for estimate in sensorimotor.holds_estimate
            ):
                return True
    return False


def dissociation_is_incoherent(
    dissociation: RepresentationDissociation,
) -> bool:
    """
    Return whether a claimed representation dissociation is ill-formed.
    """
    if (
        dissociation.sensorimotor_representation is None
        or dissociation.perceptual_representation is None
        or dissociation.sensorimotor_estimate is None
        or dissociation.perceptual_estimate is None
    ):
        return False
    return bool(
        dissociation.sensorimotor_representation.purpose
        is not RepresentationPurpose.SENSORIMOTOR_CONTROL
        or dissociation.perceptual_representation.purpose
        is not RepresentationPurpose.PERCEPTUAL_JUDGMENT
        or dissociation.sensorimotor_estimate
        not in dissociation.sensorimotor_representation.holds_estimate
        or dissociation.perceptual_estimate
        not in dissociation.perceptual_representation.holds_estimate
        or dissociation.sensorimotor_estimate is dissociation.perceptual_estimate
        or dissociation.sensorimotor_accuracy is not RepresentationAccuracy.ACCURATE
        or dissociation.perceptual_accuracy is not RepresentationAccuracy.BIASED
    )


def representation_update_is_incoherent(update: RepresentationUpdate) -> bool:
    """
    Return whether an update changes content outside its memory context.
    """
    if update.updates_representation is None or update.persists_in is None:
        return False
    if update.updates_representation not in update.persists_in.stores_representation:
        return True
    return any(
        estimate not in update.updates_representation.holds_estimate
        for estimate in update.updates_estimate
    )


def transfer_is_incoherent(transfer: DynamicsTransfer) -> bool:
    """
    Return whether a transfer compares unrelated perturbation contexts.
    """
    if transfer.from_perturbation is None or transfer.to_perturbation is None:
        return False
    return bool(
        transfer.from_perturbation is transfer.to_perturbation
        or transfer.from_perturbation.kind is not transfer.to_perturbation.kind
        or transfer.to_perturbation not in transfer.from_perturbation.equivalent_to
        or transfer.from_perturbation not in transfer.to_perturbation.equivalent_to
    )


def representation_has_incompatible_purpose(
    representation: Representation,
) -> bool:
    """
    Return whether a representation's type conflicts with its purpose.
    """
    if isinstance(representation, PerceptualRepresentation):
        return representation.purpose is not RepresentationPurpose.PERCEPTUAL_JUDGMENT
    if isinstance(representation, (ObjectDynamicsModel, LimbDynamicsModel)):
        return representation.purpose is not RepresentationPurpose.SENSORIMOTOR_CONTROL
    return False


def representation_lacks_estimated_role(representation: Representation) -> bool:
    """
    Return whether represented estimates refer to an uncovered role.
    """
    return any(
        role not in representation.represents_role
        for estimate in representation.holds_estimate
        for role in estimate.parameter_on_role
    )


# %% acceptance constraints


OBJECT_DYNAMICS_CONSTRAINTS = [
    Forbidden(
        id="C51",
        applies_to=InternalModelComposition,
        when=composition_is_incoherent,
        message="internal-model composition breaks the command-to-prediction chain",
        rationale="The desired state, inverse model, command, efference copy, "
        "represented dynamics, forward model, and predictions must form one "
        "identity-preserving control pathway.",
    ),
    Forbidden(
        id="C52",
        applies_to=ObjectDynamicsPhaseStructure,
        when=structure_has_unowned_estimate,
        message="declared estimate belongs to no representation",
        rationale="Every estimate needs an explicit coordinate frame and functional "
        "purpose supplied by its owning representation.",
    ),
    Forbidden(
        id="C53",
        applies_to=ObjectDynamicsPhaseStructure,
        when=structure_reuses_estimate_across_purposes,
        message="one estimate is reused for sensorimotor and perceptual purposes",
        rationale="A perceptual judgment must not silently become the estimate used "
        "for motor scaling.",
    ),
    Forbidden(
        id="C54",
        applies_to=RepresentationDissociation,
        when=dissociation_is_incoherent,
        message="representation dissociation has incompatible purposes or estimates",
        rationale="A dissociation requires distinct estimates held by correctly "
        "purposed representations.",
    ),
    Forbidden(
        id="C55",
        applies_to=RepresentationUpdate,
        when=representation_update_is_incoherent,
        message="representation update changes content outside its declared memory",
        rationale="An update must change estimates held by a representation retained "
        "in the named sensorimotor memory.",
    ),
    Forbidden(
        id="C56",
        applies_to=DynamicsTransfer,
        when=transfer_is_incoherent,
        message="dynamics transfer compares non-equivalent perturbation contexts",
        rationale="A transfer result must compare distinct, reciprocal contexts of "
        "the same perturbation kind.",
    ),
    Forbidden(
        id="C57",
        applies_to=Representation,
        when=representation_has_incompatible_purpose,
        message="representation type is incompatible with its declared purpose",
        rationale="Dynamics models support sensorimotor control, while perceptual "
        "representations support perceptual judgment.",
    ),
    Forbidden(
        id="C58",
        applies_to=Representation,
        when=representation_lacks_estimated_role,
        message="representation omits a role named by one of its estimates",
        rationale="Represented content must identify every participant whose "
        "properties or dynamics its estimates describe.",
    ),
    Closure(
        id="C13l",
        applies_to=ObjectDynamicsPhaseStructure,
        container="defines_internal_model",
        reached_via=(
            "defines_phase.governed_by",
            "defines_prediction.generated_by",
            "has_model_update.updates_representation",
            "has_internal_model_composition.inverse_model",
            "has_internal_model_composition.dynamics_model",
            "has_internal_model_composition.forward_model",
            "has_sensorimotor_memory.stores_representation",
            "has_representation_update.updates_representation",
            "defines_dynamics_perturbation.affects_model",
            "has_representation_dissociation.sensorimotor_representation",
            "has_representation_dissociation.perceptual_representation",
        ),
        rationale="Every internal model or representation referenced by a structure "
        "must be explicitly declared.",
    ),
    Closure(
        id="C13m",
        applies_to=ObjectDynamicsPhaseStructure,
        container="defines_desired_state",
        reached_via=(
            "defines_motor_command.pursues_state",
            "has_internal_model_composition.desired_state",
        ),
        rationale="Every desired state used by a command must be declared.",
    ),
    Closure(
        id="C13n",
        applies_to=ObjectDynamicsPhaseStructure,
        container="defines_motor_command",
        reached_via=(
            "defines_efference_copy.copies_command",
            "has_internal_model_composition.motor_command",
        ),
        rationale="Every copied or composed motor command must be declared.",
    ),
    Closure(
        id="C13o",
        applies_to=ObjectDynamicsPhaseStructure,
        container="defines_efference_copy",
        reached_via=(
            "has_internal_model_composition.efference_copy",
            "defines_internal_model.uses_efference_copy",
        ),
        rationale="Every efference copy used by represented dynamics must be declared.",
    ),
    Closure(
        id="C13p",
        applies_to=ObjectDynamicsPhaseStructure,
        container="defines_dynamics_perturbation",
        reached_via=(
            "defines_dynamics_perturbation.equivalent_to",
            "has_dynamics_transfer.from_perturbation",
            "has_dynamics_transfer.to_perturbation",
        ),
        rationale="Every compared dynamics perturbation must be declared.",
    ),
    Closure(
        id="C13q",
        applies_to=ObjectDynamicsPhaseStructure,
        container="has_sensorimotor_memory",
        reached_via=("has_representation_update.persists_in",),
        rationale="Every memory receiving an update must be declared.",
    ),
    Closure(
        id="C13r",
        applies_to=ObjectDynamicsPhaseStructure,
        container="defines_parameter",
        reached_via=(
            "defines_internal_model.holds_estimate",
            "defines_motor_command.commands_setpoint",
            "has_representation_update.updates_estimate",
            "has_representation_dissociation.sensorimotor_estimate",
            "has_representation_dissociation.perceptual_estimate",
        ),
        rationale="Every estimate or setpoint used by internal-model control must be "
        "declared by the structure.",
    ),
    Closure(
        id="C13s",
        applies_to=ObjectDynamicsPhaseStructure,
        container="defines_role",
        reached_via=("defines_internal_model.represents_role",),
        rationale="Every role represented by a dynamics model must be declared.",
    ),
    Closure(
        id="C13t",
        applies_to=ObjectDynamicsPhaseStructure,
        container="defines_event_type",
        reached_via=(
            "has_internal_model_composition.predicted_mechanical_consequence",
        ),
        rationale="Every mechanically predicted consequence must be declared.",
    ),
    Closure(
        id="C13u",
        applies_to=ObjectDynamicsPhaseStructure,
        container="defines_prediction",
        reached_via=("has_internal_model_composition.predicted_sensory_consequence",),
        rationale="Every sensory consequence produced by a composition must be "
        "declared.",
    ),
]
