"""
Tests of the action-frame perspective, the first module attached to the action core: the
structure of its module file, the vocabulary generated from it, its links to the core,
CRAM-AICOR and DUL, and what an OWL reasoner concludes from it.

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
from llm_reasoner.knowledge.ontologies.action_perspectives import action_frame
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

FRAME = Namespace(ActionPerspectiveFile.ACTION_FRAME.namespace)
"""
The terms of the action-frame perspective.
"""

CORE_IRI = URIRef(ActionPerspectiveFile.ACTION_CORE.namespace.rstrip("#"))
"""
The IRI of the action core's module.
"""

PROPOSALS_DIRECTORY = ActionPerspectiveFile.ACTION_FRAME.directory / "proposals"
"""
Where proposed vocabularies wait for review.
"""

# %% fixtures


@pytest.fixture(scope="module")
def graph() -> Graph:
    """
    The action-frame module.
    """
    return Graph().parse(ActionPerspectiveFile.ACTION_FRAME.path, format="turtle")


@pytest.fixture(scope="module")
def core_graph() -> Graph:
    """
    The action core module.
    """
    return Graph().parse(ActionPerspectiveFile.ACTION_CORE.path, format="turtle")


# %% the module file


class TestModuleFile:
    def test_the_frame_comes_right_after_the_core(self):
        assert KNOWLEDGE_ONTOLOGY_MODULES[:2] == (
            ActionPerspectiveFile.ACTION_CORE,
            ActionPerspectiveFile.ACTION_FRAME,
        )

    def test_it_declares_one_ontology_importing_the_core_cram_aicor_and_dul(
        self, graph
    ):
        iri = ontology_iri(graph)
        assert str(iri) == ActionPerspectiveFile.ACTION_FRAME.namespace.rstrip("#")
        assert set(graph.objects(iri, OWL.imports)) == {
            CORE_IRI,
            URIRef(CRAM_AICOR_ONTOLOGY_IRI),
            URIRef(DUL_ONTOLOGY_IRI),
        }

    def test_every_term_is_declared_in_its_namespace(self, graph):
        assert all(
            str(term).startswith(ActionPerspectiveFile.ACTION_FRAME.namespace)
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

    def test_the_module_is_kept_once_and_not_as_a_proposal_too(self):
        assert not (PROPOSALS_DIRECTORY / ActionPerspectiveFile.ACTION_FRAME).exists()


# %% the generated vocabulary


class TestVocabulary:
    def test_the_committed_module_matches_the_regenerated_source(self):
        module_file = ActionPerspectiveFile.ACTION_FRAME
        assert module_file.vocabulary_path.read_text() == (
            VocabularyWriter.for_knowledge_modules().source(module_file)
        )

    def test_the_cores_vocabulary_holds_nothing_of_the_frame(self):
        assert ActionPerspectiveFile.ACTION_FRAME.namespace not in (
            VocabularyWriter.for_knowledge_modules().source(
                ActionPerspectiveFile.ACTION_CORE
            )
        )

    def test_every_kind_of_condition_is_a_slot_condition_in_python_too(self):
        for condition in (
            action_frame.FeatureCondition,
            action_frame.AffordanceCondition,
            action_frame.StateCondition,
            action_frame.PostureCondition,
        ):
            assert issubclass(condition, action_frame.SlotCondition), condition

    def test_the_postures_are_an_enumeration_of_the_ontologys_members(self, graph):
        members = set(graph.subjects(RDF.type, FRAME.Posture))
        assert {member.name for member in action_frame.Posture} == {
            str(member).split("#")[-1].upper() for member in members
        }


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
    def test_the_frame_is_consistent_with_the_core_cram_aicor_and_dul_and_every_class_satisfiable(
        self, owlready2: ModuleType, cram_aicor_path: Path, dul_path: Path, tmp_path
    ):
        merged = Graph().parse(cram_aicor_path, format="turtle")
        merged.parse(dul_path, format="xml")
        merged.parse(ActionPerspectiveFile.ACTION_CORE.path, format="turtle")
        merged.parse(ActionPerspectiveFile.ACTION_FRAME.path, format="turtle")
        merged.remove((None, OWL.imports, None))
        merged_path = tmp_path / "merged.nt"
        merged.serialize(merged_path, format="nt", encoding="utf-8")
        world = owlready2.World()
        world.get_ontology(merged_path.as_uri()).load()
        owlready2.sync_reasoner_hermit(world, infer_property_values=False, debug=0)
        frame_classes = [
            cls
            for cls in world.classes()
            if cls.iri.startswith(ActionPerspectiveFile.ACTION_FRAME.namespace)
        ]
        assert frame_classes
        assert [
            cls for cls in frame_classes if owlready2.Nothing in cls.equivalent_to
        ] == []
