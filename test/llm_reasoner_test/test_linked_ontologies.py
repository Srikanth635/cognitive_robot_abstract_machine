"""
Tests of reading the terms the linked ontologies declare, which concept references in a
hypothesis must name.
"""

from __future__ import annotations

from pathlib import Path

from llm_reasoner.knowledge.ontologies import (
    CRAM_AICOR_NAMESPACE,
    DUL_NAMESPACE,
    LinkedOntologyVariable,
)
from llm_reasoner.knowledge.ontologies.linked import LinkedOntologyTerms, OntologyFiles

CASES = Path(__file__).parent / "linked_ontology_cases"
"""
Excerpts of the linked ontologies.
"""

EXCERPTS = OntologyFiles(
    paths=[CASES / "cram_aicor_excerpt.ttl", CASES / "dul_excerpt.owl"]
)
"""
The excerpts, one in Turtle and one in RDF/XML.
"""


class TestLinkedOntologyTerms:
    """
    The terms are those the linked ontology files declare.
    """

    def test_terms_declared_in_turtle_and_in_rdf_xml_are_read(self):
        terms = LinkedOntologyTerms.from_files(EXCERPTS)
        assert terms.declares(CRAM_AICOR_NAMESPACE + "Grasping")
        assert terms.declares(DUL_NAMESPACE + "Task")

    def test_term_no_file_declares_is_not_a_term(self):
        terms = LinkedOntologyTerms.from_files(EXCERPTS)
        assert not terms.declares(CRAM_AICOR_NAMESPACE + "EffectorAtGraspSite")

    def test_closest_terms_are_suggested_for_a_misspelt_one(self):
        terms = LinkedOntologyTerms.from_files(EXCERPTS)
        assert terms.closest(CRAM_AICOR_NAMESPACE + "Graspin")[0] == (
            CRAM_AICOR_NAMESPACE + "Grasping"
        )

    def test_files_are_named_by_the_environment(self, monkeypatch):
        for variable, path in zip(LinkedOntologyVariable, EXCERPTS.paths):
            monkeypatch.setenv(variable, str(path))
        assert OntologyFiles.linked_from_environment() == EXCERPTS

    def test_no_files_without_the_environment(self, monkeypatch):
        for variable in LinkedOntologyVariable:
            monkeypatch.delenv(variable, raising=False)
        assert OntologyFiles.linked_from_environment() is None
