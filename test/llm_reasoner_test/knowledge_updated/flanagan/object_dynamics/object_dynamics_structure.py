"""
A phase structure with represented dynamics, built on the core's grasp structure.

It adds what the object-dynamics module defines: a command that an inverse model
selects, its efference copy, the object and limb dynamics that interpret the copy, a
memory that retains a revised load estimate, matched load perturbations with the
transfer between them, and a perceptual judgment that diverges from the sensorimotor
estimate. Every kind of fragment the module defines appears at least once.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    InverseModel,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    ObjectPropertyEstimate,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    EstimateUpdateDirection,
    MeasurementUnit,
    ParameterValueForm,
    PhysicalDimension,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.commands import (
    DesiredState,
    EfferenceCopy,
    MotorCommand,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.composition import (
    InternalModelComposition,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.dissociation import (
    RepresentationDissociation,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.memory import (
    RetainedModelUpdate,
    SensorimotorMemory,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.perturbation import (
    DynamicsPerturbation,
    DynamicsTransfer,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.representations import (
    LimbDynamicsModel,
    LoadForceEstimate,
    ObjectDynamicsModel,
    PerceptualRepresentation,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.structure import (
    ObjectDynamicsPhaseStructure,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.value_partitions import (
    CoordinateFrame,
    MemoryPersistence,
    PerturbationKind,
    RepresentationPurpose,
    TransferOutcome,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity

from ..core_structure import (
    CoreStructure,
    build_valid_structure,
    fragments_of,
    magnitude,
)

# %% a concrete stand-in for an extension point the core leaves abstract


@dataclass(eq=False)
class QuantitativePropertyEstimate(ObjectPropertyEstimate):
    """
    An object property estimate that holds a magnitude, standing in for the concrete
    ones that later modules declare.
    """


# %% the structure


@dataclass
class ObjectDynamicsParts:
    """
    The structure and every named part of it, so a test can break one thing.
    """

    core: CoreStructure
    structure: ObjectDynamicsPhaseStructure
    load_force: LoadForceEstimate
    perceived_heaviness: QuantitativePropertyEstimate
    inverse_model: InverseModel
    desired_state: DesiredState
    command: MotorCommand
    copy: EfferenceCopy
    object_model: ObjectDynamicsModel
    limb_model: LimbDynamicsModel
    perceptual: PerceptualRepresentation
    composition: InternalModelComposition
    memory: SensorimotorMemory
    update: RetainedModelUpdate
    object_load: DynamicsPerturbation
    limb_load: DynamicsPerturbation
    transfer: DynamicsTransfer
    dissociation: RepresentationDissociation

    def fragments(self) -> list[KnowledgeEntity]:
        """
        Return the structure and every fragment reachable from it, nested ones included.
        """
        return fragments_of(self.structure)


def build_valid_object_dynamics() -> ObjectDynamicsParts:
    """
    Build a fresh structure that satisfies every acceptance rule of the core and of the
    object-dynamics module.
    """
    core = build_valid_structure()
    role = core.manipulandum

    load_force = LoadForceEstimate(
        label="expected load force",
        value_form=ParameterValueForm.QUANTITATIVE,
        dimension=PhysicalDimension.FORCE,
        parameter_range=magnitude(1.0, 4.0, MeasurementUnit.NEWTON),
        parameter_on_role=[role],
        scales=[core.grip],
    )
    perceived_heaviness = QuantitativePropertyEstimate(
        label="perceived heaviness",
        value_form=ParameterValueForm.QUANTITATIVE,
        dimension=PhysicalDimension.MASS,
        parameter_range=magnitude(0.2, 0.4, MeasurementUnit.KILOGRAM),
        parameter_on_role=[role],
    )

    inverse_model = InverseModel(label="grip inverse model")
    desired_state = DesiredState(label="held firmly", desired_condition=[core.touching])
    command = MotorCommand(
        label="squeeze command",
        pursues_state=desired_state,
        generated_by=inverse_model,
        commands_setpoint=[core.grip],
    )
    copy = EfferenceCopy(label="squeeze copy", copies_command=command)

    object_model = ObjectDynamicsModel(
        label="object dynamics",
        holds_estimate=[load_force],
        coordinate_frame=CoordinateFrame.OBJECT_CENTERED,
        purpose=RepresentationPurpose.SENSORIMOTOR_CONTROL,
        represents_role=[role],
        uses_efference_copy=[copy],
    )
    limb_model = LimbDynamicsModel(
        label="limb dynamics",
        holds_estimate=[load_force],
        coordinate_frame=CoordinateFrame.ARM_CENTERED,
        purpose=RepresentationPurpose.SENSORIMOTOR_CONTROL,
        represents_role=[role],
        uses_efference_copy=[copy],
    )
    perceptual = PerceptualRepresentation(
        label="weight judgment",
        holds_estimate=[perceived_heaviness],
        coordinate_frame=CoordinateFrame.OBJECT_CENTERED,
        purpose=RepresentationPurpose.PERCEPTUAL_JUDGMENT,
        represents_role=[role],
    )

    composition = InternalModelComposition(
        label="grip load pathway",
        efference_copy=copy,
        dynamics_model=object_model,
        forward_model=core.forward_model,
        predicted_mechanical_consequence=[core.contact],
        predicted_sensory_consequence=[core.prediction],
    )

    memory = SensorimotorMemory(
        label="object memory",
        stores_representation=[object_model],
        memory_persistence=MemoryPersistence.WHILE_OBJECT_COUPLED,
    )
    update = RetainedModelUpdate(
        label="heavier than expected",
        triggered_by_error=[core.absent],
        updates_estimate=[load_force],
        updates_representation=object_model,
        update_direction=EstimateUpdateDirection.INCREASE,
        persists_in=memory,
    )
    core.mismatch.produces_model_update = [update]

    object_load = DynamicsPerturbation(
        label="load on the object",
        perturbation_kind=PerturbationKind.LOAD,
        affects_model=object_model,
        coordinate_frame=CoordinateFrame.OBJECT_CENTERED,
    )
    limb_load = DynamicsPerturbation(
        label="load on the arm",
        perturbation_kind=PerturbationKind.LOAD,
        affects_model=limb_model,
        coordinate_frame=CoordinateFrame.ARM_CENTERED,
    )
    object_load.matched_with = [limb_load]
    limb_load.matched_with = [object_load]
    transfer = DynamicsTransfer(
        label="object load to arm load",
        from_perturbation=object_load,
        to_perturbation=limb_load,
        transfer_outcome=TransferOutcome.DOES_NOT_GENERALIZE,
    )

    dissociation = RepresentationDissociation(
        label="size-weight illusion",
        sensorimotor_representation=object_model,
        perceptual_representation=perceptual,
        sensorimotor_estimate=load_force,
        perceptual_estimate=perceived_heaviness,
    )

    inherited = {
        item.name: core.structure.__dict__[item.name]
        for item in fields(core.structure)
        if item.init
    }
    inherited["defines_parameter"] = [
        *core.structure.defines_parameter,
        load_force,
        perceived_heaviness,
    ]
    inherited["defines_internal_model"] = [
        *core.structure.defines_internal_model,
        inverse_model,
        object_model,
        limb_model,
        perceptual,
    ]
    inherited["defines_model_update"] = [update]
    structure = ObjectDynamicsPhaseStructure(
        **{
            name: list(value) if isinstance(value, list) else value
            for name, value in inherited.items()
        },
        defines_desired_state=[desired_state],
        defines_motor_command=[command],
        defines_efference_copy=[copy],
        defines_internal_model_composition=[composition],
        defines_sensorimotor_memory=[memory],
        defines_dynamics_perturbation=[object_load, limb_load],
        defines_dynamics_transfer=[transfer],
        defines_representation_dissociation=[dissociation],
    )
    core.structure = structure
    return ObjectDynamicsParts(
        core=core,
        structure=structure,
        load_force=load_force,
        perceived_heaviness=perceived_heaviness,
        inverse_model=inverse_model,
        desired_state=desired_state,
        command=command,
        copy=copy,
        object_model=object_model,
        limb_model=limb_model,
        perceptual=perceptual,
        composition=composition,
        memory=memory,
        update=update,
        object_load=object_load,
        limb_load=limb_load,
        transfer=transfer,
        dissociation=dissociation,
    )
