"""
The closed sets of alternatives the represented dynamics choose between.

They are orthogonal dimensions of a representation, a memory or a perturbation, so they
are enumerations and not sibling classes.
"""

from __future__ import annotations

from enum import StrEnum

from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import OBJECT_DYNAMICS

# %% representations


@OBJECT_DYNAMICS.value_partition()
class CoordinateFrame(StrEnum):
    """
    The reference frame in which represented dynamics are encoded.
    """

    OBJECT_CENTERED = "object_centered"
    """
    Coordinates fixed to the manipulated object.
    """

    ARM_CENTERED = "arm_centered"
    """
    Intrinsic or extrinsic coordinates associated with the arm.
    """

    HAND_CENTERED = "hand_centered"
    """
    Coordinates fixed to the hand or end effector.
    """

    GAZE_CENTERED = "gaze_centered"
    """
    Coordinates organized relative to current gaze.
    """


@OBJECT_DYNAMICS.value_partition()
class RepresentationPurpose(StrEnum):
    """
    The function for which represented content is recruited.
    """

    SENSORIMOTOR_CONTROL = "sensorimotor_control"
    """
    Content used to predict and scale motor commands.
    """

    PERCEPTUAL_JUDGMENT = "perceptual_judgment"
    """
    Content used for conscious property judgments.
    """


# %% memory


@OBJECT_DYNAMICS.value_partition()
class MemoryPersistence(StrEnum):
    """
    How long a sensorimotor memory remains recruited.
    """

    WHILE_OBJECT_COUPLED = "while_object_coupled"
    """
    The memory is recruited only while its object remains coupled.
    """

    AFTER_RELEASE = "after_release"
    """
    The memory remains available after the object is released.
    """


# %% transfer


@OBJECT_DYNAMICS.value_partition()
class PerturbationKind(StrEnum):
    """
    The manipulation applied when dynamics transfer is assessed.
    """

    LOAD = "load"
    """
    An external force or load is applied.
    """

    ROTATION = "rotation"
    """
    The object, hand, or arm orientation is changed.
    """


@OBJECT_DYNAMICS.value_partition()
class TransferOutcome(StrEnum):
    """
    Whether adaptation acquired in one context carries over to another.
    """

    GENERALIZES = "generalizes"
    """
    Adaptation acquired in the first context also holds in the second.
    """

    DOES_NOT_GENERALIZE = "does_not_generalize"
    """
    Adaptation acquired in the first context does not hold in the second.
    """
