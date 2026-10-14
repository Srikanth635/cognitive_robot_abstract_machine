"""
The closed sets of alternatives a substance transfer chooses between.
"""

from __future__ import annotations

from enum import StrEnum

from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import POURING


@POURING.value_partition()
class MaterialRegime(StrEnum):
    """
    The physical flow behavior governing a transferred substance.
    """

    LIQUID = "liquid"
    """
    A low-shear material that flows and may slosh.
    """

    GRANULAR = "granular"
    """
    Discrete particles whose bulk flow is not characterized by viscosity.
    """

    COHESIVE = "cohesive"
    """
    A highly viscous or cohesive material with delayed flow.
    """


@POURING.value_partition()
class TransferTerminationReason(StrEnum):
    """
    The condition intentionally or exceptionally ending transfer.
    """

    TARGET_QUANTITY = "target_quantity"
    """
    The requested transferred quantity has been reached.
    """

    SOURCE_EMPTY = "source_empty"
    """
    No transferable material remains in the source.
    """

    EXPLICIT_INSTRUCTION = "explicit_instruction"
    """
    An external instruction requests transfer cessation.
    """

    FAILURE = "failure"
    """
    A spill or other failure forces transfer cessation.
    """
