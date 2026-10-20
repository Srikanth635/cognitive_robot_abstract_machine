"""
Tests of what the ontology says of kinds of object: a container has an opening, and
which of the code's classes are containers.

Tests that need CRAM-AICOR or DUL read them from the files named by
:class:`LinkedOntologyVariable` and are skipped when those are unset, and the test that
reasons needs ``owlready2`` and a Java runtime and is slow.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
from rdflib import OWL, RDF, RDFS, XSD, BNode, Graph, Literal, URIRef

from llm_reasoner.knowledge.ontologies import (
    CRAM_AICOR_NAMESPACE,
    CRAM_AICOR_ONTOLOGY_IRI,
    DUL_NAMESPACE,
)
from llm_reasoner.knowledge.ontologies.frame_fields import CodeOntology
from llm_reasoner.knowledge.ontologies.object_kinds import (
    OBJECT_KINDS_FILE,
    OBJECT_KINDS_NAMESPACE,
    ObjectKinds,
)

from .test_predictive_manipulation_ontology import (
    cram_aicor_path,
    declared_terms,
    dul_path,
    ontology_iri,
    owlready2,
    referenced_terms,
)

CONTAINER = URIRef(CRAM_AICOR_NAMESPACE + "Container")
"""
CRAM-AICOR's container.
"""

HAS_OPENING = URIRef(OBJECT_KINDS_NAMESPACE + "hasOpening")
"""
The property saying that a container has an opening.
"""

# %% fixtures


@pytest.fixture(scope="module")
def graph() -> Graph:
    """
    The object kinds file.
    """
    return Graph().parse(OBJECT_KINDS_FILE, format="turtle")


@pytest.fixture(scope="module")
def code() -> CodeOntology:
    """
    The ontology of the code's classes.
    """
    return CodeOntology.shipped()


def container_classes(graph: Graph) -> set[URIRef]:
    """
    :return: The classes of the code the file puts under CRAM-AICOR's container.
    """
    return {
        subject
        for subject in graph.subjects(RDFS.subClassOf, CONTAINER)
        if isinstance(subject, URIRef)
    }


# %% the file


class TestFile:
    def test_it_declares_one_ontology_importing_cram_aicor_and_the_code_classes(
        self, graph, code
    ):
        iri = ontology_iri(graph)
        assert str(iri) == OBJECT_KINDS_NAMESPACE.rstrip("#")
        assert set(graph.objects(iri, OWL.imports)) == {
            URIRef(CRAM_AICOR_ONTOLOGY_IRI),
            URIRef(code.namespace.rstrip("#")),
        }

    def test_it_declares_only_the_property_that_a_container_has_an_opening(self, graph):
        assert declared_terms(graph) - {ontology_iri(graph)} == {HAS_OPENING}
        assert (HAS_OPENING, RDF.type, OWL.DatatypeProperty) in graph
        assert (HAS_OPENING, RDFS.range, XSD.boolean) in graph
        assert (HAS_OPENING, RDFS.domain, CONTAINER) in graph

    def test_the_property_has_a_label_and_a_definition(self, graph):
        assert graph.value(HAS_OPENING, RDFS.label) is not None
        assert graph.value(HAS_OPENING, RDFS.comment) is not None

    def test_the_container_is_said_to_have_an_opening(self, graph):
        restrictions = [
            node
            for node in graph.objects(CONTAINER, RDFS.subClassOf)
            if isinstance(node, BNode)
        ]
        assert any(
            (node, OWL.onProperty, HAS_OPENING) in graph
            and (node, OWL.hasValue, Literal(True)) in graph
            for node in restrictions
        )

    def test_it_puts_classes_of_the_code_under_the_container(self, graph, code):
        classes = container_classes(graph)
        assert classes
        assert all(str(cls).startswith(code.namespace) for cls in classes)

    def test_every_class_it_names_exists_in_the_code_ontology(self, graph, code):
        assert all(
            (cls, RDF.type, OWL.Class) in code.graph for cls in container_classes(graph)
        )

    def test_it_uses_only_its_own_cram_aicor_and_code_terms(self, graph, code):
        imported = set(graph.objects(None, OWL.imports))
        outside = {
            term
            for term in referenced_terms(graph) - imported - {ontology_iri(graph)}
            if not str(term).startswith(
                (OBJECT_KINDS_NAMESPACE, CRAM_AICOR_NAMESPACE, code.namespace)
            )
        }
        assert outside == set()


# %% reading it


def code_classes(code: CodeOntology) -> set[URIRef]:
    """
    :return: Every class of the code ontology.
    """
    return {
        cls
        for cls in code.graph.subjects(RDF.type, OWL.Class)
        if isinstance(cls, URIRef)
    }


def name_of(cls: URIRef, code: CodeOntology) -> str:
    """
    :return: The module and class name of a class of the code ontology.
    """
    return str(cls)[len(code.namespace) :]


def under_the_container(graph: Graph, code: CodeOntology) -> set[URIRef]:
    """
    :return: The classes the file puts under the container and every class of the code
        that specialises one of them.
    """
    return {
        cls
        for cls in code_classes(code)
        if set(code.class_and_bases(cls)) & container_classes(graph)
    }


class TestObjectKinds:
    def test_a_class_the_file_puts_under_the_container_has_an_opening(
        self, graph, code
    ):
        for cls in container_classes(graph):
            assert ObjectKinds.shipped().has_opening(name_of(cls, code)), cls

    def test_a_class_that_specialises_one_of_them_has_an_opening_too(self, graph, code):
        specialisations = under_the_container(graph, code) - container_classes(graph)
        assert specialisations
        for cls in specialisations:
            assert ObjectKinds.shipped().has_opening(name_of(cls, code)), cls

    def test_a_class_under_none_of_them_has_none(self, graph, code):
        others = code_classes(code) - under_the_container(graph, code)
        assert others
        for cls in others:
            assert not ObjectKinds.shipped().has_opening(name_of(cls, code)), cls

    def test_a_class_the_code_ontology_lacks_has_none(self):
        assert not ObjectKinds.shipped().has_opening("no.such.Class")


# %% reasoning


@pytest.mark.slow
class TestReasoning:
    def test_it_is_consistent_with_cram_aicor_and_dul(
        self, owlready2: ModuleType, cram_aicor_path: Path, dul_path: Path, tmp_path
    ):
        merged = Graph().parse(cram_aicor_path, format="turtle")
        merged.parse(dul_path, format="xml")
        merged.parse(OBJECT_KINDS_FILE, format="turtle")
        merged += CodeOntology.shipped().graph
        merged.remove((None, OWL.imports, None))
        merged_path = tmp_path / "merged.nt"
        merged.serialize(merged_path, format="nt", encoding="utf-8")
        world = owlready2.World()
        world.get_ontology(merged_path.as_uri()).load()
        owlready2.sync_reasoner_hermit(world, infer_property_values=False, debug=0)
        assert [
            cls
            for cls in world.classes()
            if cls.iri.startswith(OBJECT_KINDS_NAMESPACE)
            and owlready2.Nothing in cls.equivalent_to
        ] == []
