from __future__ import annotations

from pathlib import Path

import pytest
import rdflib
import llm_reasoner.knowledge_updated.microworlds.flanagan.core  # noqa: F401  declares the core
from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    SOMA,
    related_to,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import FLANAGAN
from llm_reasoner.knowledge_updated.ontological.vocabulary import AlignmentRelation

# %% the alignment helpers


class TestAlignmentHelpers:
    """
    Alignments to the foundational ontologies are written by term name.
    """

    def test_subclass_alignment_names_the_term_in_its_ontology(self) -> None:
        alignment = subclass_of(SOMA, "AgentRole")

        assert alignment.iri == SOMA.term("AgentRole")
        assert alignment.relation is AlignmentRelation.SUBCLASS_OF

    def test_related_alignment_is_marked_as_only_related(self) -> None:
        alignment = related_to(DUL, "Situation")

        assert alignment.iri == DUL.term("Situation")
        assert alignment.relation is AlignmentRelation.RELATED_TO

    def test_the_two_foundational_ontologies_have_different_namespaces(self) -> None:
        assert DUL != SOMA


# %% alignments against the published ontologies

ONTOLOGY_DIRECTORY = Path.home() / "ontologies"
"""
Where the foundational ontologies are kept on a development machine.
"""

FOUNDATIONAL_FILES = ("DUL.owl", "SOMA.owl")
"""
The files that define the terms the microworld aligns to.
"""

FILES_ARE_PRESENT = all(
    (ONTOLOGY_DIRECTORY / name).is_file() for name in FOUNDATIONAL_FILES
)


@pytest.fixture(scope="module")
def defined_classes() -> set[str]:
    """
    Return the IRIs of the classes the foundational ontologies define.
    """
    graph = rdflib.Graph()
    for name in FOUNDATIONAL_FILES:
        graph.parse(ONTOLOGY_DIRECTORY / name, format="xml")
    return {
        str(subject)
        for subject in graph.subjects(rdflib.RDF.type, rdflib.OWL.Class)
        if isinstance(subject, rdflib.URIRef)
    }


@pytest.mark.skipif(
    not FILES_ARE_PRESENT,
    reason="the DUL and SOMA ontologies are not present on this machine",
)
class TestAlignmentsExistInTheFoundationalOntologies:
    """
    Every term the microworld aligns to is defined by the ontology it names.
    """

    def declared_alignments(self) -> set[str]:
        return {
            alignment.iri
            for concept in FLANAGAN.concepts()
            for alignment in concept.alignments
        }

    def test_microworld_declares_alignments_to_check(self) -> None:
        assert self.declared_alignments()

    def test_every_declared_alignment_names_a_defined_class(
        self, defined_classes: set[str]
    ) -> None:
        assert self.declared_alignments() - defined_classes == set()

    def test_a_term_no_ontology_defines_would_be_reported(
        self, defined_classes: set[str]
    ) -> None:
        invented = {DUL.term("NotATermOfDul")}

        assert invented - defined_classes == invented
