"""
Tests of the image-schemas perspective, the second module attached to the action core:
the structure of its module file, the vocabulary generated from it, its links to the
core, CRAM-AICOR and DUL, where its knowledge is rooted, and what an OWL reasoner
concludes from it.

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
    image_schemas,
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

SCHEMAS = Namespace(ActionPerspectiveFile.IMAGE_SCHEMAS.namespace)
"""
The terms of the image-schemas perspective.
"""

CORE_IRI = URIRef(ActionPerspectiveFile.ACTION_CORE.namespace.rstrip("#"))
"""
The IRI of the action core's module.
"""

# %% fixtures


@pytest.fixture(scope="module")
def graph() -> Graph:
    """
    The image-schemas module.
    """
    return Graph().parse(ActionPerspectiveFile.IMAGE_SCHEMAS.path, format="turtle")


@pytest.fixture(scope="module")
def core_graph() -> Graph:
    """
    The action core module.
    """
    return Graph().parse(ActionPerspectiveFile.ACTION_CORE.path, format="turtle")


# %% the module file


class TestModuleFile:
    def test_the_schemas_come_after_the_core_and_the_frame(self):
        assert KNOWLEDGE_ONTOLOGY_MODULES[:3] == (
            ActionPerspectiveFile.ACTION_CORE,
            ActionPerspectiveFile.ACTION_FRAME,
            ActionPerspectiveFile.IMAGE_SCHEMAS,
        )

    def test_it_declares_one_ontology_importing_the_core_cram_aicor_and_dul(
        self, graph
    ):
        iri = ontology_iri(graph)
        assert str(iri) == ActionPerspectiveFile.IMAGE_SCHEMAS.namespace.rstrip("#")
        assert set(graph.objects(iri, OWL.imports)) == {
            CORE_IRI,
            URIRef(CRAM_AICOR_ONTOLOGY_IRI),
            URIRef(DUL_ONTOLOGY_IRI),
        }

    def test_every_term_is_declared_in_its_namespace(self, graph):
        assert all(
            str(term).startswith(ActionPerspectiveFile.IMAGE_SCHEMAS.namespace)
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
        module_file = ActionPerspectiveFile.IMAGE_SCHEMAS
        assert module_file.vocabulary_path.read_text() == (
            VocabularyWriter.for_knowledge_modules().source(module_file)
        )

    def test_the_cores_vocabulary_holds_nothing_of_the_schemas(self):
        assert ActionPerspectiveFile.IMAGE_SCHEMAS.namespace not in (
            VocabularyWriter.for_knowledge_modules().source(
                ActionPerspectiveFile.ACTION_CORE
            )
        )

    def test_the_core_s_frame_gains_no_field_for_a_schema(self):
        assert not hasattr(action_core.ActionFrame, "has_schema")

    def test_every_kind_of_schema_is_an_image_schema_in_python_too(self):
        for schema in (
            image_schemas.SourcePathGoalSchema,
            image_schemas.ContainmentSchema,
            image_schemas.VerticalitySchema,
            image_schemas.SupportSchema,
            image_schemas.ContactSchema,
        ):
            assert issubclass(schema, image_schemas.ImageSchema), schema

    def test_the_relation_kinds_are_an_enumeration_of_the_ontologys_members(
        self, graph
    ):
        members = set(graph.subjects(RDF.type, SCHEMAS.RelationKind))
        assert {member.value for member in image_schemas.RelationKind} == {
            str(member) for member in members
        }


# %% where the knowledge is rooted


class TestRoot:
    def test_a_stereotype_roots_the_knowledge_and_names_the_core_frame_it_is_of(self):
        root = PERSPECTIVE_ROOTS[ActionPerspectiveFile.IMAGE_SCHEMAS]
        assert root.root is image_schemas.SchemaStereotype
        assert root.frame_field == "schema_stereotype_of"


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
    def test_the_schemas_are_consistent_with_the_core_cram_aicor_and_dul_and_every_class_satisfiable(
        self, owlready2: ModuleType, cram_aicor_path: Path, dul_path: Path, tmp_path
    ):
        merged = Graph().parse(cram_aicor_path, format="turtle")
        merged.parse(dul_path, format="xml")
        merged.parse(ActionPerspectiveFile.ACTION_CORE.path, format="turtle")
        merged.parse(ActionPerspectiveFile.IMAGE_SCHEMAS.path, format="turtle")
        merged.remove((None, OWL.imports, None))
        merged_path = tmp_path / "merged.nt"
        merged.serialize(merged_path, format="nt", encoding="utf-8")
        world = owlready2.World()
        world.get_ontology(merged_path.as_uri()).load()
        owlready2.sync_reasoner_hermit(world, infer_property_values=False, debug=0)
        schema_classes = [
            cls
            for cls in world.classes()
            if cls.iri.startswith(ActionPerspectiveFile.IMAGE_SCHEMAS.namespace)
        ]
        assert schema_classes
        assert [
            cls for cls in schema_classes if owlready2.Nothing in cls.equivalent_to
        ] == []
