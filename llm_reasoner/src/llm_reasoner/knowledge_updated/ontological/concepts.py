"""
What the ontology records about one concept, and the rules a single concept must obey.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from enum import StrEnum
from typing import Optional

from krrood.patterns.role import Role

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge_updated.ontological.axioms import ClassAxioms
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.references import Alignment
from llm_reasoner.knowledge_updated.ontological.vocabulary import (
    ConceptLevel,
    EvidenceStatus,
    Rigidity,
)

# %% invalid concepts


class ConceptFault(StrEnum):
    """
    Why a concept's description breaks a rule.
    """

    NOT_A_DATACLASS = "not_a_dataclass"
    """
    The concept is not a dataclass, so it has no fields to describe.
    """

    ANTI_RIGID_AT_DESCRIPTION_LEVEL = "anti_rigid_at_description_level"
    """
    A concept that classifies kinds is declared anti-rigid, which only makes sense for
    things that exist in an episode.
    """

    DESCRIPTION_WITHOUT_IDENTITY = "description_without_identity"
    """
    A description-level concept is not a knowledge entity, so it has no stable
    identifier or label.
    """

    ROLE_WITHOUT_TAKER = "role_without_taker"
    """
    An anti-rigid concept names nothing that can hold the role.
    """

    TAKER_WITHOUT_ROLE = "taker_without_role"
    """
    A rigid concept names a role taker.
    """

    ANTI_RIGID_NOT_A_ROLE = "anti_rigid_not_a_role"
    """
    An anti-rigid concept is not realised as a role.
    """

    RIGID_IS_A_ROLE = "rigid_is_a_role"
    """
    A rigid concept is realised as a role.
    """

    CLAIMS_WITHOUT_STATUS = "claims_without_status"
    """
    The concept cites source claims but states no evidence status.
    """


@dataclass
class InvalidConcept(LLMReasonerError):
    """
    A concept is described in a way that breaks a rule.
    """

    concept: type
    """
    The class being described.
    """

    reason: ConceptFault
    """
    The rule it breaks.
    """

    def __str__(self) -> str:
        return f"invalid concept {self.concept.__qualname__}: {self.reason.value}"


# %% concept metadata


@dataclass(frozen=True)
class ConceptMeta:
    """
    Everything the ontology records about one concept.
    """

    cls: type
    """
    The class that realises the concept.
    """

    iri: str
    """
    The concept's identifier.
    """

    module: str
    """
    Name of the ontology module that declares the concept.
    """

    rigidity: Rigidity = Rigidity.RIGID
    """
    Whether membership is essential to what has it.
    """

    level: ConceptLevel = ConceptLevel.DESCRIPTION
    """
    Whether the concept classifies kinds or actual things.
    """

    alignments: tuple[Alignment, ...] = ()
    """
    Terms of other ontologies the concept is aligned to.
    """

    abstract: bool = False
    """
    Whether the concept only groups its subconcepts and is never instantiated itself.
    """

    axioms: ClassAxioms = field(default_factory=ClassAxioms)
    """
    What the concept states about its instances.
    """

    role_taker: Optional[type] = None
    """
    The class whose instances can hold the role, for an anti-rigid concept.
    """

    status: Optional[EvidenceStatus] = None
    """
    How well a source supports the concept, when the concept comes from one.
    """

    claims: tuple[str, ...] = ()
    """
    Identifiers of the source claims the concept represents.
    """

    def __post_init__(self) -> None:
        """
        Check the rules that concern this concept alone.

        :raises InvalidConcept: on the first rule broken.
        :raises InvalidAxiom: when an axiom cannot hold of the class.
        """
        self._check_kind()
        self._check_rigidity()
        if self.claims and self.status is None:
            raise InvalidConcept(self.cls, ConceptFault.CLAIMS_WITHOUT_STATUS)
        self.axioms.check_against(self.cls)

    @property
    def local_name(self) -> str:
        """
        The name of the class that realises the concept.
        """
        return self.cls.__name__

    def _check_kind(self) -> None:
        """
        Check that the class can be described at the level the concept declares.
        """
        if not is_dataclass(self.cls):
            raise InvalidConcept(self.cls, ConceptFault.NOT_A_DATACLASS)
        if (
            self.rigidity is Rigidity.ANTI_RIGID
            and self.level is ConceptLevel.DESCRIPTION
        ):
            raise InvalidConcept(self.cls, ConceptFault.ANTI_RIGID_AT_DESCRIPTION_LEVEL)
        if self.level is ConceptLevel.DESCRIPTION and not issubclass(
            self.cls, KnowledgeEntity
        ):
            raise InvalidConcept(self.cls, ConceptFault.DESCRIPTION_WITHOUT_IDENTITY)

    def _check_rigidity(self) -> None:
        """
        Check that rigidity, role taker and realisation agree.
        """
        is_role = issubclass(self.cls, Role)
        if self.rigidity is Rigidity.ANTI_RIGID:
            if self.role_taker is None:
                raise InvalidConcept(self.cls, ConceptFault.ROLE_WITHOUT_TAKER)
            if not is_role:
                raise InvalidConcept(self.cls, ConceptFault.ANTI_RIGID_NOT_A_ROLE)
            return
        if self.role_taker is not None:
            raise InvalidConcept(self.cls, ConceptFault.TAKER_WITHOUT_ROLE)
        if is_role:
            raise InvalidConcept(self.cls, ConceptFault.RIGID_IS_A_ROLE)
