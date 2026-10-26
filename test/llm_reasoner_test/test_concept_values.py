"""
Tests of concept values as instances of their classes: a fragment's concept is written
as an anonymous instance of a CRAM-AICOR, DUL or hypothesis class, read back from one,
checked closed-world by the told class hierarchy and entailed from the classes a
fragment's restrictions require.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from operator import attrgetter

import pytest
from rdflib import OWL, RDF, RDFS, BNode, Graph, URIRef
from rdflib.term import Node

from llm_reasoner.knowledge.acceptance.restrictions import ClosedWorldMembership
from llm_reasoner.knowledge.acceptance.derived import (
    RangesHold,
    RangeViolation,
    RestrictionsHold,
)
from llm_reasoner.knowledge.admission.consistency import (
    ConsistencyCheck,
    ContradictionLocaliser,
)
from llm_reasoner.knowledge.admission.report import OntologyMerge
from llm_reasoner.knowledge.generation.schema import GenerationSchema
from llm_reasoner.knowledge.generation.wire import FragmentDecoder
from llm_reasoner.knowledge.ontologies import DUL_NAMESPACE, PhysicalMeasurementFile
from llm_reasoner.knowledge.ontologies.linked import (
    LinkedClassHierarchy,
    LinkedOntologyTerms,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
from llm_reasoner.knowledge.ontologies.physical_measurements import flow_onset
from llm_reasoner.knowledge.ontologies.predictive_manipulation import core
from llm_reasoner.knowledge.ontologies.terms import (
    ClassExpression,
    ComplementOf,
    ConceptReference,
    DescriptionEntity,
    IntersectionOf,
    NamedClass,
    PropertyRestriction,
    PropertyTerm,
    Quantifier,
    UnionOf,
)
from llm_reasoner.knowledge.ontologies.vocabulary import (
    Vocabulary,
    declared_value_types,
)
from llm_reasoner.knowledge.representation.exceptions import (
    AmbiguousConceptType,
    ConceptWrittenAsIndividual,
)
from llm_reasoner.knowledge.representation.rdf import GraphReader, GraphWriter

from .concept_field_mimics import CategorisedAction, CategoryClass
from .test_dul_alignment import ROOT_CAUSES, SHIPPED_HYPOTHESES
from .test_generation import GenerationCase

# %% helpers


class FragmentName(StrEnum):
    """
    The IRIs the tests write fragments under.
    """

    ACTION = "http://example.org/hypothesis/concepts#action"
    """
    The one fragment of a test.
    """


def written(fragment: DescriptionEntity) -> Graph:
    """
    :return: One fragment written under :attr:`FragmentName.ACTION`.
    """
    return GraphWriter(namespace="").write_fragments(
        [fragment], {id(fragment): URIRef(FragmentName.ACTION)}
    )


def value_node(graph: Graph, property_iri: str) -> Node:
    """
    :return: The one node the fragment has as the property's value.
    """
    (node,) = graph.objects(URIRef(FragmentName.ACTION), URIRef(property_iri))
    return node


def decomposing(task: CategoryClass) -> core.ActionPhaseStructure:
    """
    :return: A structure decomposing a task.
    """
    return core.ActionPhaseStructure(decomposes_task=task.concept)


def decomposed_task_iri() -> str:
    """
    :return: The IRI of the property naming the task a structure decomposes.
    """
    return PropertyTerm.of_field(core.ActionPhaseStructure, "decomposes_task").iri


# %% the told class hierarchy


class TestLinkedClassHierarchy:
    def test_a_class_specialises_its_told_superclasses(self):
        assert LinkedClassHierarchy.pinned().is_subclass(
            CategoryClass.POURING, CategoryClass.TRANSFERRING
        )

    def test_a_class_does_not_specialise_an_unrelated_class(self):
        assert not LinkedClassHierarchy.pinned().is_subclass(
            CategoryClass.GRASPING, CategoryClass.TRANSFERRING
        )

    def test_the_alignment_carries_a_copy_of_dul_to_dul(self):
        assert LinkedClassHierarchy.pinned().is_subclass(
            CategoryClass.POURING, DUL_NAMESPACE + "InformationObject"
        )

    def test_a_class_of_ours_specialises_the_linked_classes_it_links_to(self):
        hierarchy = LinkedClassHierarchy.pinned()
        assert all(
            hierarchy.is_subclass(core.ActionPhaseStructure.term.iri, linked)
            for linked in core.ActionPhaseStructure.term.linked_terms
        )


# %% writing and reading


class TestWritingConcepts:
    def test_a_linked_concept_is_an_anonymous_instance_of_its_class(self):
        graph = written(decomposing(CategoryClass.POURING))
        node = value_node(graph, decomposed_task_iri())
        assert isinstance(node, BNode)
        assert set(graph.objects(node, RDF.type)) == {URIRef(CategoryClass.POURING)}

    def test_a_linked_concept_is_not_declared_a_class_again(self):
        graph = written(decomposing(CategoryClass.POURING))
        assert set(graph.predicate_objects(URIRef(CategoryClass.POURING))) == set()

    def test_a_concept_of_the_hypothesis_is_a_class_under_its_propertys_range(self):
        graph = written(
            CategorisedAction(performed_category=CategoryClass.DECANTING.concept)
        )
        assert set(graph.predicate_objects(URIRef(CategoryClass.DECANTING))) == {
            (RDF.type, OWL.Class),
            (RDFS.subClassOf, URIRef(CategoryClass.MANIPULATION_ACTION_CATEGORY)),
        }

    def test_a_literal_takes_its_propertys_range_datatype(self):
        setup = flow_onset.PhysicsSimulationSetup(run_count=3)
        term = PropertyTerm.of_field(flow_onset.PhysicsSimulationSetup, "run_count")
        ontology = Graph().parse(
            PhysicalMeasurementFile.FLOW_ONSET.path, format="turtle"
        )
        literal = value_node(written(setup), term.iri)
        assert literal.datatype == ontology.value(URIRef(term.iri), RDFS.range)
        assert literal.toPython() == setup.run_count


class TestReadingConcepts:
    def test_a_written_concept_is_read_back(self):
        graph = written(decomposing(CategoryClass.POURING))
        (structure,) = GraphReader().read(graph).fragments
        assert structure.decomposes_task == CategoryClass.POURING.concept

    def test_a_concept_written_as_an_individual_is_refused(self):
        graph = written(decomposing(CategoryClass.POURING))
        node = value_node(graph, decomposed_task_iri())
        graph.remove((URIRef(FragmentName.ACTION), None, node))
        graph.add(
            (
                URIRef(FragmentName.ACTION),
                URIRef(decomposed_task_iri()),
                URIRef(CategoryClass.POURING),
            )
        )
        with pytest.raises(ConceptWrittenAsIndividual):
            GraphReader().read(graph)

    def test_a_concept_of_several_classes_is_refused(self):
        graph = written(decomposing(CategoryClass.POURING))
        node = value_node(graph, decomposed_task_iri())
        graph.add((node, RDF.type, URIRef(CategoryClass.GRASPING)))
        with pytest.raises(AmbiguousConceptType):
            GraphReader().read(graph)


# %% the closed-world checks


class TestConceptRestrictions:
    def test_a_concept_of_a_subclass_of_the_required_class_meets_it(self):
        action = CategorisedAction(
            performed_category=CategoryClass.POURING.concept,
            realized_category=[CategoryClass.GRASPING.concept],
        )
        assert RestrictionsHold().violations([action]) == []

    def test_a_concept_outside_the_required_class_violates_it(self):
        action = CategorisedAction(
            performed_category=CategoryClass.GRASPING.concept,
            realized_category=[CategoryClass.GRASPING.concept],
        )
        (violation,) = RestrictionsHold().violations([action])
        assert violation.expression == CategorisedAction.term.restrictions[0]

    def test_a_concept_of_the_hypothesis_does_not_meet_a_linked_class(self):
        action = CategorisedAction(
            performed_category=CategoryClass.DECANTING.concept,
            realized_category=[CategoryClass.GRASPING.concept],
        )
        (violation,) = RestrictionsHold().violations([action])
        assert violation.expression == CategorisedAction.term.restrictions[0]


class TestAnyConcept:
    def test_a_concept_of_the_hypothesis_is_some_thing(self):
        condition = action_core.StateCondition(
            condition_has_state=[CategoryClass.DECANTING.concept]
        )
        (requirement,) = [
            restriction
            for restriction in action_core.StateCondition.term.restrictions
            if restriction.property_iri
            == PropertyTerm.of_field(
                action_core.StateCondition, "condition_has_state"
            ).iri
        ]
        assert ClosedWorldMembership().members(requirement, [condition]) == [condition]

    def test_a_state_left_out_is_not_entailed_by_some_thing(self):
        condition = action_core.StateCondition()
        Vocabulary().fill_entailed_values(condition)
        assert condition.condition_has_state == []


class TestConceptRanges:
    def test_a_concept_outside_its_propertys_range_is_a_range_violation(self):
        action = CategorisedAction(
            performed_category=CategoryClass.ACCUMULATED_STATE.concept
        )
        (violation,) = RangesHold().violations([action])
        assert isinstance(violation, RangeViolation)
        assert violation.field_name == "performed_category"

    def test_a_concept_of_a_subclass_of_the_range_is_in_range(self):
        action = CategorisedAction(
            performed_category=CategoryClass.POURING.concept,
            realized_category=[CategoryClass.GRASPING.concept],
        )
        assert RangesHold().violations([action]) == []

    def test_a_concept_of_the_hypothesis_is_in_its_propertys_range(self):
        action = CategorisedAction(performed_category=CategoryClass.DECANTING.concept)
        assert RangesHold().violations([action]) == []


# %% entailment


class TestEntailedConcepts:
    def test_an_unset_single_concept_takes_the_required_class(self):
        action = CategorisedAction()
        Vocabulary().fill_entailed_values(action)
        assert action.performed_category == CategoryClass.TRANSFERRING.concept

    def test_a_stated_single_concept_is_kept(self):
        action = CategorisedAction(performed_category=CategoryClass.POURING.concept)
        Vocabulary().fill_entailed_values(action)
        assert action.performed_category == CategoryClass.POURING.concept

    def test_concepts_none_of_the_required_class_gain_it(self):
        action = CategorisedAction(realized_category=[CategoryClass.POURING.concept])
        Vocabulary().fill_entailed_values(action)
        assert action.realized_category == [
            CategoryClass.POURING.concept,
            CategoryClass.PRIMITIVE_ACTION.concept,
        ]

    def test_concepts_one_of_the_required_class_are_kept(self):
        action = CategorisedAction(realized_category=[CategoryClass.GRASPING.concept])
        Vocabulary().fill_entailed_values(action)
        assert action.realized_category == [CategoryClass.GRASPING.concept]


# %% the concepts a model writes


class TestDecodedConcepts:
    def test_a_linked_term_that_is_no_class_is_a_problem(self):
        decoder = FragmentDecoder(GenerationSchema.for_root(core.ActionPhaseStructure))
        (problem,) = decoder.decode(
            GenerationCase.LINKED_PROPERTY_AS_CONCEPT.text
        ).problems
        assert "aicor:abstainsOn" in problem

    def test_linked_terms_are_checked_without_being_given(self):
        decoder = FragmentDecoder(GenerationSchema.for_root(core.ActionPhaseStructure))
        problems = decoder.decode(
            GenerationCase.CONCEPTS_LINKED_AND_LOCAL.text
        ).problems
        assert [problem for problem in problems if "EffectorAtGraspSite" in problem]


# %% locating contradictions


@dataclass
class ContradictedByOneClass(ConsistencyCheck):
    """
    Judges a graph inconsistent exactly when something in it is an instance of one
    class.
    """

    contradicting_class: str
    """
    The class no instance of which is consistent.
    """

    def is_consistent(self, graph: Graph) -> bool:
        return (None, RDF.type, URIRef(self.contradicting_class)) not in graph


class TestLocalisingConcepts:
    def test_a_contradicting_concept_is_located_at_the_fragment_citing_it(self):
        graph = written(decomposing(CategoryClass.POURING))
        conflict = ContradictionLocaliser(
            ContradictedByOneClass(contradicting_class=CategoryClass.POURING)
        ).conflict(graph)
        assert conflict == [URIRef(FragmentName.ACTION)]


# %% the concept properties of the modules


def concept_fields() -> dict[str, PropertyTerm]:
    """
    :return: The term of every concept field of the generated vocabulary, by the IRI
        of its property.
    """
    return {
        term.iri: term
        for generated in Vocabulary().classes
        for declared in fields(generated)
        if declared_value_types(generated)[declared.name] is ConceptReference
        for term in [PropertyTerm.of_field(generated, declared.name)]
    }


class OpenConceptProperty(StrEnum):
    """
    The concept properties whose values may be of any class, so they state no range.
    """

    CATEGORY = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCategory"
    """
    The category a categorical value names.
    """

    JUSTIFICATION = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasJustification"
    """
    What justifies an exemption.
    """

    TRIGGER = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#triggeredBy"
    """
    What ends a phase from outside.
    """


class TestConceptPropertiesOfTheModules:
    def test_every_concept_property_but_the_open_ones_ranges_over_a_linked_class(self):
        assert {
            iri
            for iri, term in concept_fields().items()
            if not LinkedOntologyTerms.pinned().declares(term.linked_range)
        } == set(OpenConceptProperty)

    def test_no_class_requires_a_concept_as_an_individual(self):
        concept_properties = set(concept_fields())
        assert [
            (generated.__name__, restriction.property_iri)
            for generated in Vocabulary().classes
            for restriction in generated.term.restrictions
            if isinstance(restriction, PropertyRestriction)
            and restriction.quantifier == Quantifier.VALUE
            and restriction.property_iri in concept_properties
        ] == []

    def test_a_structure_decomposing_a_state_is_out_of_range(self):
        structure = decomposing(CategoryClass.ACCUMULATED_STATE)
        assert [
            violation.field_name for violation in RangesHold().violations([structure])
        ] == ["decomposes_task"]

    def test_a_task_the_hypothesis_coins_is_a_kind_of_its_propertys_range(self):
        graph = written(decomposing(CategoryClass.DECANTING))
        term = PropertyTerm.of_field(core.ActionPhaseStructure, "decomposes_task")
        assert term.linked_range
        assert (
            URIRef(CategoryClass.DECANTING),
            RDFS.subClassOf,
            URIRef(term.linked_range),
        ) in graph


# %% reasoning


def linked_classes_named(expression: ClassExpression) -> set[str]:
    """
    :return: The classes of the linked ontologies a class expression names, as a class
        or as the value of a value restriction.
    """
    if isinstance(expression, NamedClass):
        return (
            {expression.iri} if LinkedOntologyTerms.concerns(expression.iri) else set()
        )
    if isinstance(expression, (IntersectionOf, UnionOf)):
        return {
            iri
            for operand in expression.operands
            for iri in linked_classes_named(operand)
        }
    if isinstance(expression, ComplementOf):
        return linked_classes_named(expression.operand)
    if not isinstance(expression, PropertyRestriction):
        return set()
    named = (
        set() if expression.filler is None else linked_classes_named(expression.filler)
    )
    if isinstance(expression.value, str) and LinkedOntologyTerms.concerns(
        expression.value
    ):
        named.add(expression.value)
    return named


def concepts_in_use() -> set[str]:
    """
    :return: The linked classes the knowledge's classes name in their restrictions and
        definitions, and the linked concepts the shipped hypotheses cite.
    """
    named = {
        iri
        for generated in Vocabulary().classes
        for expression in (
            *generated.term.restrictions,
            *(
                ()
                if generated.term.definition is None
                else (generated.term.definition,)
            ),
        )
        for iri in linked_classes_named(expression)
    }
    cited = {
        value.iri
        for path in SHIPPED_HYPOTHESES
        for fragment in GraphReader()
        .read(Graph().parse(path, format="turtle"))
        .fragments
        for declared in fields(fragment)
        for held in [attrgetter(declared.name)(fragment)]
        for value in (held if isinstance(held, list) else [held])
        if isinstance(value, ConceptReference)
        and LinkedOntologyTerms.concerns(value.iri)
    }
    return named | cited


@pytest.mark.slow
class TestHierarchyAgainstReasoner:
    def test_the_told_hierarchy_holds_every_superclass_hermit_infers(self, hermit):
        concepts = sorted(concepts_in_use())
        inferred = hermit.superclasses(
            OntologyMerge.knowledge(left_out=(ROOT_CAUSES,)).graph(), concepts
        )
        hierarchy = LinkedClassHierarchy.pinned()
        untold = {
            concept: sorted(
                superclass
                for superclass in inferred[concept]
                if superclass != str(OWL.Thing)
                and not hierarchy.is_subclass(concept, superclass)
            )
            for concept in concepts
        }
        assert {
            concept: missing for concept, missing in untold.items() if missing
        } == {}
