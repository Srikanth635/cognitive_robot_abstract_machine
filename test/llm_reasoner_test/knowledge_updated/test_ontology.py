from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import pytest
from llm_reasoner.knowledge_updated.ontological.axioms import ClassAxioms
from llm_reasoner.knowledge_updated.ontological.concepts import (
    ConceptFault,
    InvalidConcept,
)
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.ontology import (
    DeclarationFault,
    DuplicateModule,
    InvalidDeclaration,
    Ontology,
    OntologyModule,
    PropertyDeclaration,
    ValuePartitionMeta,
)
from llm_reasoner.knowledge_updated.ontological.properties import (
    PropertyMeta,
    attribute,
    relation,
)
from llm_reasoner.knowledge_updated.ontological.references import Alignment
from llm_reasoner.knowledge_updated.ontological.vocabulary import (
    AlignmentRelation,
    EvidenceStatus,
)

# %% fixtures

CORE_NAMESPACE = "http://example.org/ontology/core#"
PREHENSION_NAMESPACE = "http://example.org/ontology/prehension#"
POURING_NAMESPACE = "http://example.org/ontology/pouring#"
EXPLICIT_IRI = "urn:test:reachDistance"


class Gait(StrEnum):
    """
    How a mover walks.
    """

    WALK = "walk"
    """
    One foot always on the ground.
    """


@dataclass(eq=False)
class Phase(KnowledgeEntity):
    """
    A stretch of an action.
    """

    duration: Optional[float] = attribute(functional=True)
    """
    How long the phase lasts.
    """

    successors: list[Phase] = relation()
    """
    The phases that may follow.
    """


@dataclass(eq=False)
class ReachPhase(Phase):
    """
    A phase that moves towards something.
    """

    reach_distance: Optional[float] = attribute(functional=True, iri=EXPLICIT_IRI)
    """
    How far the phase reaches.
    """


@dataclass(eq=False)
class PourPhase(ReachPhase):
    """
    A reaching phase that tilts a container.
    """


@dataclass(eq=False)
class Other(KnowledgeEntity):
    """
    An entity unrelated to the phases.
    """


class NotADataclass:
    """
    A class that cannot be described.
    """


def build_ontology() -> tuple[Ontology, OntologyModule, OntologyModule]:
    """
    Return an ontology with a core module and a prehension module that depends on it.
    """
    ontology = Ontology(name="test")
    core = ontology.module("core", version="1.0.0", namespace=CORE_NAMESPACE)
    prehension = ontology.module(
        "prehension",
        version="2.0.0",
        namespace=PREHENSION_NAMESPACE,
        depends_on=(core,),
    )
    return ontology, core, prehension


def fault_of(declaration) -> DeclarationFault:
    """
    Run a declaration that is expected to be refused, and return why.
    """
    with pytest.raises(InvalidDeclaration) as refusal:
        declaration()
    return refusal.value.reason


# %% modules


class TestModules:
    """
    A microworld's ontology is divided into modules with their own namespaces.
    """

    def test_module_records_its_name_version_namespace_and_dependencies(self) -> None:
        _, core, prehension = build_ontology()

        assert (prehension.name, prehension.version) == ("prehension", "2.0.0")
        assert prehension.namespace.iri == PREHENSION_NAMESPACE
        assert prehension.depends_on == (core,)
        assert core.depends_on == ()

    def test_modules_are_listed_in_declaration_order(self) -> None:
        ontology, core, prehension = build_ontology()

        assert ontology.modules() == (core, prehension)

    def test_versions_are_reported_by_module_name(self) -> None:
        ontology, _, _ = build_ontology()

        assert ontology.versions() == {"core": "1.0.0", "prehension": "2.0.0"}

    def test_module_name_can_be_declared_only_once(self) -> None:
        ontology, _, _ = build_ontology()

        with pytest.raises(DuplicateModule) as refusal:
            ontology.module("core", version="9", namespace=POURING_NAMESPACE)

        assert refusal.value.name == "core"

    def test_dependencies_are_listed_dependency_first(self) -> None:
        ontology, core, prehension = build_ontology()
        pouring = ontology.module(
            "pouring",
            version="1.0.0",
            namespace=POURING_NAMESPACE,
            depends_on=(prehension,),
        )

        assert pouring.transitive_dependencies() == (core, prehension)
        assert core.transitive_dependencies() == ()


# %% declaring concepts


class TestConceptDeclaration:
    """
    A module's decorator records a class as a concept and leaves the class untouched.
    """

    def test_decorator_returns_the_class_it_was_given(self) -> None:
        _, core, _ = build_ontology()

        assert core.concept()(Phase) is Phase

    def test_decorator_syntax_declares_a_class_defined_alongside(self) -> None:
        _, core, _ = build_ontology()

        @core.concept()
        @dataclass(eq=False)
        class Local(KnowledgeEntity):
            """
            A concept declared with decorator syntax.
            """

        assert core.ontology.find_concept(Local) is not None

    def test_concept_iri_is_derived_from_the_namespace_and_class_name(self) -> None:
        ontology, core, _ = build_ontology()

        core.concept()(Phase)

        assert ontology.find_concept(Phase).iri == CORE_NAMESPACE + "Phase"

    def test_explicit_iri_overrides_the_derived_one(self) -> None:
        ontology, core, _ = build_ontology()

        core.concept(iri="urn:test:phase")(Phase)

        assert ontology.find_concept(Phase).iri == "urn:test:phase"

    def test_concept_records_the_module_that_declared_it(self) -> None:
        ontology, core, prehension = build_ontology()

        core.concept()(Phase)
        prehension.concept()(ReachPhase)

        assert ontology.find_concept(Phase).module == "core"
        assert ontology.find_concept(ReachPhase).module == "prehension"

    def test_declaration_arguments_reach_the_concept_description(self) -> None:
        ontology, core, _ = build_ontology()
        alignment = Alignment(
            iri="http://example.org/other#Task",
            relation=AlignmentRelation.SUBCLASS_OF,
        )

        core.concept(
            alignments=[alignment],
            abstract=True,
            disjoint_with=["Other"],
            disjoint_siblings=True,
            fixed_values={"duration": 1.0},
            status=EvidenceStatus.AUTHOR_HYPOTHESIS,
            claims=["FJ-003"],
        )(Phase)

        described = ontology.find_concept(Phase)
        assert described.alignments == (alignment,)
        assert described.abstract is True
        assert described.axioms == ClassAxioms.of(
            disjoint_with=["Other"],
            disjoint_siblings=True,
            fixed_values={"duration": 1.0},
        )
        assert described.status is EvidenceStatus.AUTHOR_HYPOTHESIS
        assert described.claims == ("FJ-003",)

    def test_allowed_values_reach_the_concept_description(self) -> None:
        ontology, core, _ = build_ontology()

        core.concept(allowed_values={"duration": (1.0, 2.0)})(Phase)

        assert ontology.find_concept(Phase).axioms.allowed_values_of("duration") == (
            frozenset({1.0, 2.0})
        )

    def test_overlap_names_reach_the_concept_description(self) -> None:
        ontology, core, _ = build_ontology()

        core.concept(may_overlap_with=["Other"])(Phase)

        assert ontology.find_concept(Phase).axioms.may_overlap_with == ("Other",)

    def test_rejected_concept_is_not_recorded(self) -> None:
        ontology, core, _ = build_ontology()

        with pytest.raises(InvalidConcept) as refusal:
            core.concept()(NotADataclass)

        assert refusal.value.reason is ConceptFault.NOT_A_DATACLASS
        assert ontology.find_concept(NotADataclass) is None
        assert ontology.concepts() == ()


# %% looking concepts up


class TestConceptLookup:
    """
    Concepts are found by class or IRI and listed in declaration order.
    """

    def test_concept_is_found_by_class_and_by_iri(self) -> None:
        ontology, core, _ = build_ontology()
        core.concept()(Phase)

        by_class = ontology.find_concept(Phase)

        assert by_class is not None
        assert ontology.find_concept_by_iri(by_class.iri) is by_class

    def test_unknown_class_and_iri_are_not_found(self) -> None:
        ontology, _, _ = build_ontology()

        assert ontology.find_concept(Phase) is None
        assert ontology.find_concept_by_iri(CORE_NAMESPACE + "Phase") is None

    def test_concepts_are_listed_in_declaration_order(self) -> None:
        ontology, core, prehension = build_ontology()
        core.concept()(Phase)
        prehension.concept()(ReachPhase)
        core.concept()(Other)

        assert [meta.cls for meta in ontology.concepts()] == [Phase, ReachPhase, Other]

    def test_listing_can_be_limited_to_one_module(self) -> None:
        ontology, core, prehension = build_ontology()
        core.concept()(Phase)
        prehension.concept()(ReachPhase)
        core.concept()(Other)

        assert [meta.cls for meta in ontology.concepts(module="core")] == [
            Phase,
            Other,
        ]


# %% refused declarations


class TestRefusedDeclarations:
    """
    Declarations that would make the ontology ambiguous are refused.
    """

    def test_iri_can_identify_only_one_concept(self) -> None:
        _, core, prehension = build_ontology()
        core.concept(iri="urn:test:shared")(Phase)

        assert (
            fault_of(lambda: prehension.concept(iri="urn:test:shared")(Other))
            is DeclarationFault.DUPLICATE_IRI
        )

    def test_class_can_be_declared_only_once(self) -> None:
        _, core, prehension = build_ontology()
        core.concept()(Phase)

        assert (
            fault_of(lambda: prehension.concept()(Phase))
            is DeclarationFault.CLASS_ALREADY_DECLARED
        )

    def test_refused_declaration_leaves_the_ontology_unchanged(self) -> None:
        ontology, core, prehension = build_ontology()
        core.concept()(Phase)

        fault_of(lambda: prehension.concept()(Phase))

        assert [meta.cls for meta in ontology.concepts()] == [Phase]
        assert ontology.find_concept(Phase).module == "core"

    def test_same_class_may_be_declared_in_independent_ontologies(self) -> None:
        first, first_core, _ = build_ontology()
        second, second_core, _ = build_ontology()

        first_core.concept()(Phase)
        second_core.concept()(Phase)

        assert first.find_concept(Phase) is not second.find_concept(Phase)


# %% module dependencies


class TestModuleDependencies:
    """
    A concept may specialise only concepts of its own module or of modules it depends on.
    """

    def test_specialising_a_concept_of_a_dependency_is_accepted(self) -> None:
        ontology, core, prehension = build_ontology()
        core.concept()(Phase)

        prehension.concept()(ReachPhase)

        assert ontology.find_concept(ReachPhase) is not None

    def test_specialising_a_concept_of_the_same_module_is_accepted(self) -> None:
        ontology, core, _ = build_ontology()
        core.concept()(Phase)

        core.concept()(ReachPhase)

        assert ontology.find_concept(ReachPhase) is not None

    def test_specialising_through_a_chain_of_dependencies_is_accepted(self) -> None:
        ontology, core, prehension = build_ontology()
        pouring = ontology.module(
            "pouring",
            version="1.0.0",
            namespace=POURING_NAMESPACE,
            depends_on=(prehension,),
        )
        core.concept()(Phase)

        pouring.concept()(ReachPhase)

        assert ontology.find_concept(ReachPhase).module == "pouring"

    def test_specialising_a_concept_of_an_unrelated_module_is_refused(self) -> None:
        ontology, core, _ = build_ontology()
        pouring = ontology.module(
            "pouring", version="1.0.0", namespace=POURING_NAMESPACE
        )
        core.concept()(Phase)

        assert (
            fault_of(lambda: pouring.concept()(ReachPhase))
            is DeclarationFault.BASE_FROM_UNDECLARED_DEPENDENCY
        )

    def test_indirect_ancestors_are_checked_too(self) -> None:
        ontology, core, _ = build_ontology()
        pouring = ontology.module(
            "pouring", version="1.0.0", namespace=POURING_NAMESPACE
        )
        core.concept()(Phase)

        assert (
            fault_of(lambda: pouring.concept()(PourPhase))
            is DeclarationFault.BASE_FROM_UNDECLARED_DEPENDENCY
        )


# %% value partitions


class TestValuePartitions:
    """
    An enumeration is declared as a closed set of alternatives.
    """

    def test_enumeration_is_recorded_with_a_derived_iri(self) -> None:
        ontology, core, _ = build_ontology()

        assert core.value_partition()(Gait) is Gait

        partition = ontology.find_value_partition(Gait)
        assert partition == ValuePartitionMeta(
            enumeration=Gait, iri=CORE_NAMESPACE + "Gait", module="core"
        )

    def test_partitions_can_be_listed_by_module(self) -> None:
        ontology, core, prehension = build_ontology()
        core.value_partition()(Gait)

        assert [p.enumeration for p in ontology.value_partitions()] == [Gait]
        assert ontology.value_partitions(module="prehension") == ()

    def test_only_an_enumeration_can_be_a_partition(self) -> None:
        _, core, _ = build_ontology()

        assert (
            fault_of(lambda: core.value_partition()(Phase))
            is DeclarationFault.NOT_AN_ENUMERATION
        )

    def test_partition_and_concept_cannot_share_an_iri(self) -> None:
        _, core, _ = build_ontology()
        core.concept(iri="urn:test:shared")(Phase)

        assert (
            fault_of(lambda: core.value_partition(iri="urn:test:shared")(Gait))
            is DeclarationFault.DUPLICATE_IRI
        )


# %% property declarations


class TestPropertyDeclarations:
    """
    Each concept declares the properties written in its own body.
    """

    def declarations(self, ontology: Ontology) -> dict[str, PropertyDeclaration]:
        return {
            declaration.field_name: declaration
            for declaration in ontology.property_declarations()
        }

    def test_property_iri_is_derived_from_the_field_name_in_the_module_namespace(
        self,
    ) -> None:
        ontology, core, _ = build_ontology()
        core.concept()(Phase)

        declared = self.declarations(ontology)

        assert declared["duration"].iri == CORE_NAMESPACE + "duration"
        assert declared["successors"].iri == CORE_NAMESPACE + "successors"

    def test_explicit_property_iri_overrides_the_derived_one(self) -> None:
        ontology, core, prehension = build_ontology()
        core.concept()(Phase)
        prehension.concept()(ReachPhase)

        assert self.declarations(ontology)["reach_distance"].iri == EXPLICIT_IRI

    def test_declaration_records_its_owner_module_and_characteristics(self) -> None:
        ontology, core, prehension = build_ontology()
        core.concept()(Phase)
        prehension.concept()(ReachPhase)

        declaration = self.declarations(ontology)["reach_distance"]

        assert declaration.owner is ReachPhase
        assert declaration.module == "prehension"
        assert (
            declaration.meta == PropertyMeta.declared_on(ReachPhase)["reach_distance"]
        )

    def test_inherited_properties_belong_to_the_class_that_wrote_them(self) -> None:
        ontology, core, prehension = build_ontology()
        core.concept()(Phase)
        prehension.concept()(ReachPhase)

        owners = {
            declaration.field_name: declaration.owner
            for declaration in ontology.property_declarations()
        }

        assert owners == {
            "duration": Phase,
            "successors": Phase,
            "reach_distance": ReachPhase,
        }

    def test_listing_can_be_limited_to_one_module(self) -> None:
        ontology, core, prehension = build_ontology()
        core.concept()(Phase)
        prehension.concept()(ReachPhase)

        names = [
            declaration.field_name
            for declaration in ontology.property_declarations(module="prehension")
        ]

        assert names == ["reach_distance"]

    def test_concept_without_described_properties_declares_none(self) -> None:
        ontology, core, _ = build_ontology()
        core.concept()(Other)

        assert ontology.property_declarations() == ()


# %% documentation


class TestOntologyVocabularyIsDocumented:
    """
    Every field and member of the ontology vocabulary carries documentation.
    """

    @pytest.mark.parametrize(
        "dataclass_type", [ValuePartitionMeta, PropertyDeclaration]
    )
    def test_every_field_is_documented(self, dataclass_type: type) -> None:
        assert DocumentationReader().find_undocumented_fields(dataclass_type) == ()

    def test_every_fault_is_documented(self) -> None:
        assert DocumentationReader().find_undocumented_members(DeclarationFault) == ()

    def test_module_class_is_documented(self) -> None:
        assert DocumentationReader().read_class(OntologyModule) is not None
