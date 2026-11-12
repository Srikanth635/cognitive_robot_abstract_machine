"""
Tests of reading the terms the linked ontologies declare, which concept references in a
hypothesis must name.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_reasoner.knowledge.vocabulary.ontology_modules import CRAM_AICOR_NAMESPACE, DUL_NAMESPACE
from llm_reasoner.knowledge.vocabulary.linked import (
    LinkedOntologyFile,
    LinkedOntologyTerms,
    OntologyFiles,
)

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


# %% the pinned linked ontologies


class TestPinnedLinkedOntologies:
    """
    The linked ontologies are pinned copies shipped with the package, each recorded with
    the digest of its bytes.
    """

    def test_each_pinned_file_has_the_digest_its_source_records(self):
        sources = json.loads(
            (LinkedOntologyFile.CRAM_AICOR.parent / "sources.json").read_text()
        )
        for linked_file in LinkedOntologyFile:
            assert hashlib.sha256(linked_file.read_bytes()).hexdigest() == (
                sources[linked_file.name]["sha256"]
            )

    def test_the_pinned_files_are_cram_aicor_and_dul(self):
        assert OntologyFiles.pinned().paths == [
            LinkedOntologyFile.CRAM_AICOR,
            LinkedOntologyFile.DUL,
        ]

    def test_the_pinned_terms_hold_cram_aicors_and_duls(self):
        terms = LinkedOntologyTerms.pinned()
        assert terms.declares(CRAM_AICOR_NAMESPACE + "Pouring")
        assert terms.declares(DUL_NAMESPACE + "Task")
