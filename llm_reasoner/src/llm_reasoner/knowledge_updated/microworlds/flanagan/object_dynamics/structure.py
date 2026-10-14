"""
A phase structure that also declares the dynamics its controller represents.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    InternalModel,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.structure import (
    ActionPhaseStructure,
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
    SensorimotorMemory,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.perturbation import (
    DynamicsPerturbation,
    DynamicsTransfer,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import (
    CORE,
    OBJECT_DYNAMICS,
)
from llm_reasoner.knowledge_updated.ontological.properties import relation
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus


@OBJECT_DYNAMICS.concept(status=EvidenceStatus.ENGINEERING_EXTENSION)
@dataclass(eq=False)
class ObjectDynamicsPhaseStructure(ActionPhaseStructure):
    """
    A phase structure with explicit internal models and dynamics evidence.

    Every estimate it declares must be held by a representation it declares. Retained
    revisions of those estimates are declared with the model updates of the core.
    """

    defines_internal_model: list[InternalModel] = relation(
        minimum_count=1,
        owns=True,
        iri=CORE.namespace.property_term("defines_internal_model"),
    )
    """
    The internal models and represented content declared by the structure.
    """

    defines_desired_state: list[DesiredState] = relation(owns=True)
    """
    The desired states supplied to inverse models.
    """

    defines_motor_command: list[MotorCommand] = relation(owns=True)
    """
    The outgoing motor commands declared by the structure.
    """

    defines_efference_copy: list[EfferenceCopy] = relation(owns=True)
    """
    The command copies available to dynamics models.
    """

    defines_internal_model_composition: list[InternalModelComposition] = relation(
        owns=True
    )
    """
    The command-to-prediction pathways declared by the structure.
    """

    defines_sensorimotor_memory: list[SensorimotorMemory] = relation(owns=True)
    """
    The memories retaining sensorimotor representations.
    """

    defines_dynamics_perturbation: list[DynamicsPerturbation] = relation(owns=True)
    """
    The load and rotation contexts used to assess transfer.
    """

    defines_dynamics_transfer: list[DynamicsTransfer] = relation(owns=True)
    """
    The transfer results declared by the structure.
    """

    defines_representation_dissociation: list[RepresentationDissociation] = relation(
        owns=True
    )
    """
    The sensorimotor-perceptual dissociations declared by the structure.
    """
