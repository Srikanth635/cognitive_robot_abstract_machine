"""
Tests of the failures perspective, the third module attached to the action core: the
structure of its module file, the vocabulary generated from it, its links to the core,
CRAM-AICOR and DUL, where its knowledge is rooted, and what an OWL reasoner concludes
from it.

Tests that need CRAM-AICOR or DUL read them from the files named by
:class:`LinkedOntologyVariable` and are skipped when those are unset, and the test that
reasons needs ``owlready2`` and a Java runtime and is slow.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
from rdflib import OWL, RDF, RDFS, Graph, Namespace, URIRef

from llm_reasoner.knowledge.ontologies import (
    ACTION_PERSPECTIVES_NAMESPACE_ROOT,
    CRAM_AICOR_NAMESPACE,
    CRAM_AICOR_ONTOLOGY_IRI,
    DUL_NAMESPACE,
    DUL_ONTOLOGY_IRI,
    KNOWLEDGE_ONTOLOGY_MODULES,
    ActionPerspectiveFile,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import (
    action_core,
    failures,
)
from llm_reasoner.knowledge.ontologies.perspective_roots import PERSPECTIVE_ROOTS
from llm_reasoner.knowledge.ontology_translation.writing import VocabularyWriter

from .test_predictive_manipulation_ontology import (
    cram_aicor_graph,
    cram_aicor_path,
    declared_terms,
    dul_graph,
    dul_path,
    ontology_iri,
    owlready2,
    referenced_terms,
)

FAILURES = Namespace(ActionPerspectiveFile.FAILURES.namespace)
"""
The terms of the failures perspective.
"""

CORE_IRI = URIRef(ActionPerspectiveFile.ACTION_CORE.namespace.rstrip("#"))
"""
The IRI of the action core's module.
"""

# %% fixtures


@pytest.fixture(scope="module")
def graph() -> Graph:
    """
    The failures module.
    """
    return Graph().parse(ActionPerspectiveFile.FAILURES.path, format="turtle")


@pytest.fixture(scope="module")
def core_graph() -> Graph:
    """
    The action core module.
    """
    return Graph().parse(ActionPerspectiveFile.ACTION_CORE.path, format="turtle")


# %% the module file


class TestModuleFile:
    def test_the_failures_come_after_the_core_the_frame_and_the_image_schemas(self):
        assert KNOWLEDGE_ONTOLOGY_MODULES[:4] == (
            ActionPerspectiveFile.ACTION_CORE,
            ActionPerspectiveFile.ACTION_FRAME,
            ActionPerspectiveFile.IMAGE_SCHEMAS,
            ActionPerspectiveFile.FAILURES,
        )

    def test_it_declares_one_ontology_importing_the_core_cram_aicor_and_dul(
        self, graph
    ):
        iri = ontology_iri(graph)
        assert str(iri) == ActionPerspectiveFile.FAILURES.namespace.rstrip("#")
        assert set(graph.objects(iri, OWL.imports)) == {
            CORE_IRI,
            URIRef(CRAM_AICOR_ONTOLOGY_IRI),
            URIRef(DUL_ONTOLOGY_IRI),
        }

    def test_every_term_is_declared_in_its_namespace(self, graph):
        assert all(
            str(term).startswith(ActionPerspectiveFile.FAILURES.namespace)
            for term in declared_terms(graph)
            if term != ontology_iri(graph)
        )

    def test_only_its_own_the_cores_cram_aicor_and_dul_terms_are_used(self, graph):
        cited = set(graph.objects(None, RDFS.seeAlso))
        imported = set(graph.objects(None, OWL.imports))
        outside = {
            term
            for term in referenced_terms(graph) - cited - imported
            if not str(term).startswith(
                (
                    ACTION_PERSPECTIVES_NAMESPACE_ROOT,
                    CRAM_AICOR_NAMESPACE,
                    DUL_NAMESPACE,
                )
            )
        }
        assert outside == set()

    def test_every_term_has_a_label_and_a_definition(self, graph):
        for term in declared_terms(graph):
            assert graph.value(term, RDFS.label) is not None, term
            assert graph.value(term, RDFS.comment) is not None, term

    def test_it_states_nothing_about_the_cores_frame_itself(self, graph):
        assert (
            None,
            RDFS.domain,
            URIRef(ActionPerspectiveFile.ACTION_CORE.namespace + "ActionFrame"),
        ) not in graph


# %% the generated vocabulary


class TestVocabulary:
    def test_the_committed_module_matches_the_regenerated_source(self):
        module_file = ActionPerspectiveFile.FAILURES
        assert module_file.vocabulary_path.read_text() == (
            VocabularyWriter.for_knowledge_modules().source(module_file)
        )

    def test_the_cores_vocabulary_holds_nothing_of_the_failures(self):
        assert ActionPerspectiveFile.FAILURES.namespace not in (
            VocabularyWriter.for_knowledge_modules().source(
                ActionPerspectiveFile.ACTION_CORE
            )
        )

    def test_the_core_s_frame_gains_no_field_for_a_failure(self):
        assert not hasattr(action_core.ActionFrame, "has_failure")

    def test_a_failure_names_no_frame_of_its_own(self):
        assert not hasattr(failures.Failure, "failure_of_frame")

    def test_the_parameter_change_directions_are_an_enumeration_of_the_ontologys_members(
        self, graph
    ):
        members = set(graph.subjects(RDF.type, FAILURES.ParameterChangeDirection))
        assert {member.value for member in failures.ParameterChangeDirection} == {
            str(member) for member in members
        }


# %% where the knowledge is rooted


class TestRoot:
    def test_a_stereotype_roots_the_knowledge_and_names_the_core_frame_it_is_of(self):
        root = PERSPECTIVE_ROOTS[ActionPerspectiveFile.FAILURES]
        assert root.root is failures.FailureStereotype
        assert root.frame_field == "failure_stereotype_of"


# %% links to the core, CRAM-AICOR and DUL


class TestExternalLinks:
    def test_every_core_term_used_is_declared_by_the_core(self, graph, core_graph):
        used = {
            term
            for term in referenced_terms(graph)
            if str(term).startswith(ActionPerspectiveFile.ACTION_CORE.namespace)
        }
        assert used - declared_terms(core_graph) == set()

    def test_every_cram_aicor_term_used_is_declared_by_cram_aicor(
        self, graph, cram_aicor_graph
    ):
        used = {
            term
            for term in referenced_terms(graph)
            if str(term).startswith(CRAM_AICOR_NAMESPACE)
        }
        assert used - declared_terms(cram_aicor_graph) == set()

    def test_every_dul_term_used_is_declared_by_dul(self, graph, dul_graph):
        used = {
            term
            for term in referenced_terms(graph)
            if str(term).startswith(DUL_NAMESPACE)
        }
        assert used - declared_terms(dul_graph) == set()


# %% reasoning


@pytest.mark.slow
class TestReasoning:
    def test_the_failures_are_consistent_with_the_core_cram_aicor_and_dul_and_every_class_satisfiable(
        self, owlready2: ModuleType, cram_aicor_path: Path, dul_path: Path, tmp_path
    ):
        merged = Graph().parse(cram_aicor_path, format="turtle")
        merged.parse(dul_path, format="xml")
        merged.parse(ActionPerspectiveFile.ACTION_CORE.path, format="turtle")
        merged.parse(ActionPerspectiveFile.FAILURES.path, format="turtle")
        merged.remove((None, OWL.imports, None))
        merged_path = tmp_path / "merged.nt"
        merged.serialize(merged_path, format="nt", encoding="utf-8")
        world = owlready2.World()
        world.get_ontology(merged_path.as_uri()).load()
        owlready2.sync_reasoner_hermit(world, infer_property_values=False, debug=0)
        failure_classes = [
            cls
            for cls in world.classes()
            if cls.iri.startswith(ActionPerspectiveFile.FAILURES.namespace)
        ]
        assert failure_classes
        assert [
            cls for cls in failure_classes if owlready2.Nothing in cls.equivalent_to
        ] == []
