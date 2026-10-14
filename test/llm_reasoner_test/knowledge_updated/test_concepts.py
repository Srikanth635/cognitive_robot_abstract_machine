from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import pytest
from krrood.patterns.role import Role
from krrood.symbol_graph.symbol_graph import Symbol
from llm_reasoner.knowledge_updated.ontological.axioms import (
    AxiomFault,
    ClassAxioms,
    InvalidAxiom,
)
from llm_reasoner.knowledge_updated.ontological.concepts import (
    ConceptFault,
    ConceptMeta,
    InvalidConcept,
)
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import attribute
from llm_reasoner.knowledge_updated.ontological.references import Alignment
from llm_reasoner.knowledge_updated.ontological.vocabulary import (
    AlignmentRelation,
    ConceptLevel,
    EvidenceStatus,
    Rigidity,
)

# %% fixtures

IRI = "http://example.org/ontology/core#Thing"


class Speed(StrEnum):
    """
    How fast a thing goes.
    """

    SLOW = "slow"
    """
    Below walking pace.
    """


@dataclass(eq=False)
class Thing(KnowledgeEntity):
    """
    A described thing.
    """

    speed: Optional[Speed] = attribute(functional=True)
    """
    How fast it goes.
    """


@dataclass(eq=False)
class Holder(Role[Symbol]):
    """
    A role that some symbol holds.
    """


@dataclass
class PlainDataclass:
    """
    A dataclass that is not an entity.
    """


class NotADataclass:
    """
    A class that is not a dataclass.
    """


def meta_of(cls: type, **overrides) -> ConceptMeta:
    """
    Describe a class with the neutral settings, changed by the overrides.
    """
    return ConceptMeta(cls=cls, iri=IRI, module="core", **overrides)


def role_meta_of(cls: type, **overrides) -> ConceptMeta:
    """
    Describe a class as a role held by a symbol, changed by the overrides.
    """
    settings = {
        "rigidity": Rigidity.ANTI_RIGID,
        "level": ConceptLevel.OCCURRENCE,
        "role_taker": Symbol,
    }
    settings.update(overrides)
    return meta_of(cls, **settings)


def fault_of(build) -> ConceptFault:
    """
    Build a concept description that is expected to be rejected, and return why.
    """
    with pytest.raises(InvalidConcept) as rejection:
        build()
    return rejection.value.reason


# %% accepted concepts


class TestAcceptedConcepts:
    """
    A description that violates no rule is kept as given.
    """

    def test_neutral_description_takes_the_documented_defaults(self) -> None:
        meta = meta_of(Thing)

        assert meta.rigidity is Rigidity.RIGID
        assert meta.level is ConceptLevel.DESCRIPTION
        assert meta.alignments == ()
        assert meta.abstract is False
        assert meta.axioms == ClassAxioms()
        assert meta.role_taker is None
        assert meta.status is None
        assert meta.claims == ()

    def test_concept_is_named_after_its_class(self) -> None:
        assert meta_of(Thing).local_name == Thing.__name__

    def test_alignments_status_and_claims_are_kept(self) -> None:
        alignment = Alignment(
            iri="http://example.org/other#Object",
            relation=AlignmentRelation.SUBCLASS_OF,
        )

        meta = meta_of(
            Thing,
            alignments=(alignment,),
            status=EvidenceStatus.EMPIRICAL_FINDING,
            claims=("FJ-003",),
        )

        assert meta.alignments == (alignment,)
        assert meta.status is EvidenceStatus.EMPIRICAL_FINDING
        assert meta.claims == ("FJ-003",)

    def test_role_held_by_a_symbol_is_accepted(self) -> None:
        meta = role_meta_of(Holder)

        assert meta.role_taker is Symbol

    def test_occurrence_level_concept_needs_no_entity_identity(self) -> None:
        meta_of(PlainDataclass, level=ConceptLevel.OCCURRENCE)

    def test_axioms_that_hold_of_the_class_are_accepted(self) -> None:
        meta_of(Thing, axioms=ClassAxioms.of(fixed_values={"speed": Speed.SLOW}))


# %% rejected concepts


class TestRejectedConcepts:
    """
    A description that breaks a rule is refused where it is written.
    """

    def test_class_must_be_a_dataclass(self) -> None:
        assert fault_of(lambda: meta_of(NotADataclass)) is ConceptFault.NOT_A_DATACLASS

    def test_description_level_concept_must_be_an_entity(self) -> None:
        assert (
            fault_of(lambda: meta_of(PlainDataclass))
            is ConceptFault.DESCRIPTION_WITHOUT_IDENTITY
        )

    def test_anti_rigid_concept_must_be_occurrence_level(self) -> None:
        assert (
            fault_of(lambda: role_meta_of(Holder, level=ConceptLevel.DESCRIPTION))
            is ConceptFault.ANTI_RIGID_AT_DESCRIPTION_LEVEL
        )

    def test_anti_rigid_concept_must_name_its_taker(self) -> None:
        assert (
            fault_of(lambda: role_meta_of(Holder, role_taker=None))
            is ConceptFault.ROLE_WITHOUT_TAKER
        )

    def test_rigid_concept_cannot_name_a_taker(self) -> None:
        assert (
            fault_of(lambda: meta_of(Thing, role_taker=Symbol))
            is ConceptFault.TAKER_WITHOUT_ROLE
        )

    def test_anti_rigid_concept_must_be_realised_as_a_role(self) -> None:
        assert (
            fault_of(lambda: role_meta_of(PlainDataclass))
            is ConceptFault.ANTI_RIGID_NOT_A_ROLE
        )

    def test_rigid_concept_cannot_be_realised_as_a_role(self) -> None:
        assert (
            fault_of(lambda: meta_of(Holder, level=ConceptLevel.OCCURRENCE))
            is ConceptFault.RIGID_IS_A_ROLE
        )

    def test_claims_need_a_status(self) -> None:
        assert (
            fault_of(lambda: meta_of(Thing, claims=("FJ-003",)))
            is ConceptFault.CLAIMS_WITHOUT_STATUS
        )

    def test_rejection_names_the_class(self) -> None:
        with pytest.raises(InvalidConcept) as rejection:
            meta_of(NotADataclass)

        assert rejection.value.concept is NotADataclass

    def test_axiom_that_cannot_hold_of_the_class_is_refused(self) -> None:
        with pytest.raises(InvalidAxiom) as rejection:
            meta_of(Thing, axioms=ClassAxioms.of(fixed_values={"missing": 1}))

        assert rejection.value.reason is AxiomFault.UNKNOWN_PROPERTY


# %% documentation


class TestConceptVocabularyIsDocumented:
    """
    Every field and member of the concept vocabulary carries documentation.
    """

    def test_every_field_is_documented(self) -> None:
        assert DocumentationReader().find_undocumented_fields(ConceptMeta) == ()

    def test_every_fault_is_documented(self) -> None:
        assert DocumentationReader().find_undocumented_members(ConceptFault) == ()
