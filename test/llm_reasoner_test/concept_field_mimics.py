"""
A fragment whose concept fields carry what a concept field of the knowledge modules
carries: the linked class its values are instances of, and a restriction requiring a
value of a linked class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import ClassVar, List, Optional

from llm_reasoner.knowledge.ontologies import (
    CRAM_AICOR_NAMESPACE,
    HYPOTHESIS_CONCEPT_NAMESPACE,
)
from llm_reasoner.knowledge.ontologies.terms import (
    ClassTerm,
    ConceptReference,
    DescriptionEntity,
    NamedClass,
    PropertyRestriction,
    PropertyTerm,
    Quantifier,
)

# %% terms


class CategorisedActionTerm(StrEnum):
    """
    The IRIs of the mimic's class and of its properties.
    """

    CLASS = (
        "http://www.ease-crc.org/ont/llm_reasoner/test/concept-fields#CategorisedAction"
    )
    """
    The class of the mimic.
    """

    PERFORMED_CATEGORY = (
        "http://www.ease-crc.org/ont/llm_reasoner/test/concept-fields#performedCategory"
    )
    """
    The one category of action performed, which must be a kind of transferring.
    """

    REALIZED_CATEGORY = (
        "http://www.ease-crc.org/ont/llm_reasoner/test/concept-fields#realizedCategory"
    )
    """
    The categories of action realized, one of which must be a primitive action.
    """


class CategoryClass(StrEnum):
    """
    The classes the mimic's concept fields name.
    """

    MANIPULATION_ACTION_CATEGORY = CRAM_AICOR_NAMESPACE + "ManipulationActionCategory"
    """
    The class every value of both properties is an instance of.
    """

    TRANSFERRING = CRAM_AICOR_NAMESPACE + "Transferring"
    """
    The class a performed category must specialise.
    """

    PRIMITIVE_ACTION = CRAM_AICOR_NAMESPACE + "PrimitiveAction"
    """
    The class one realized category must specialise.
    """

    POURING = CRAM_AICOR_NAMESPACE + "Pouring"
    """
    A kind of transferring.
    """

    GRASPING = CRAM_AICOR_NAMESPACE + "Grasping"
    """
    A primitive action, and no kind of transferring.
    """

    ACCUMULATED_STATE = CRAM_AICOR_NAMESPACE + "AccumulatedState"
    """
    A state, and no category of action.
    """

    DECANTING = HYPOTHESIS_CONCEPT_NAMESPACE + "Decanting"
    """
    A category of the hypothesis itself, which the linked ontologies lack.
    """

    @property
    def concept(self) -> ConceptReference:
        """
        The concept naming the class.
        """
        return ConceptReference(iri=self.value)


# %% the fragment


@dataclass(eq=False)
class CategorisedAction(DescriptionEntity):
    """
    An action described by the categories of action it performs and realizes.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri=CategorisedActionTerm.CLASS,
        restrictions=(
            PropertyRestriction(
                property_iri=CategorisedActionTerm.PERFORMED_CATEGORY,
                quantifier=Quantifier.SOME,
                filler=NamedClass(iri=CategoryClass.TRANSFERRING),
            ),
            PropertyRestriction(
                property_iri=CategorisedActionTerm.REALIZED_CATEGORY,
                quantifier=Quantifier.SOME,
                filler=NamedClass(iri=CategoryClass.PRIMITIVE_ACTION),
            ),
        ),
    )
    """
    The class the mimic stands for.
    """

    performed_category: Optional[ConceptReference] = field(
        default=None,
        metadata=PropertyTerm(
            iri=CategorisedActionTerm.PERFORMED_CATEGORY,
            functional=True,
            linked_range=CategoryClass.MANIPULATION_ACTION_CATEGORY,
        ).as_dict(),
    )
    """
    The one category of action performed.
    """

    realized_category: List[ConceptReference] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri=CategorisedActionTerm.REALIZED_CATEGORY,
            functional=False,
            linked_range=CategoryClass.MANIPULATION_ACTION_CATEGORY,
        ).as_dict(),
    )
    """
    The categories of action realized.
    """
