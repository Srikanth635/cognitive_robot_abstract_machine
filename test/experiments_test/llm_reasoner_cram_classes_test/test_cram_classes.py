"""
Tests that the ontology of CRAM's classes kept in llm_reasoner is the one its selection
gives for the code as it is now.
"""

from __future__ import annotations

from pathlib import Path

import llm_reasoner.knowledge.ontologies
from rdflib import Graph
from rdflib.compare import graph_diff, isomorphic, to_isomorphic

from llm_reasoner.world.class_ontology.reading import ClassOntologyReader
from llm_reasoner.world.class_ontology.selection import ClassSelection
from llm_reasoner.world.class_ontology.writing import ClassOntologyWriter

CRAM_CLASSES = Path(llm_reasoner.knowledge.ontologies.__file__).parent / "cram_classes"
"""
Where the selection of CRAM's classes and their ontology are kept.
"""

SELECTION = CRAM_CLASSES / "cram-classes.json"
"""
Which of CRAM's classes the ontology describes.
"""

ONTOLOGY = CRAM_CLASSES / "cram-classes.ttl"
"""
The ontology, as committed.
"""

SHOWN_DIFFERENCES = 10
"""
How many differing statements a failure shows.
"""


def test_the_committed_ontology_is_written_from_the_code_as_it_is_now():
    """
    When this fails the classes have changed: write the ontology again with ``python -m
    llm_reasoner.world.class_ontology`` and the two paths above.
    """
    selection = ClassSelection.from_file(SELECTION)
    written = ClassOntologyWriter(
        ontology=ClassOntologyReader(selection=selection).read(),
        namespace=selection.namespace,
    ).graph()
    committed = Graph().parse(ONTOLOGY, format="turtle")
    if not isomorphic(written, committed):
        _, only_written, only_committed = graph_diff(
            to_isomorphic(written), to_isomorphic(committed)
        )
        raise AssertionError(
            "the committed ontology differs from the code: "
            f"{len(only_written)} statements only in the code, e.g. "
            f"{sorted(only_written)[:SHOWN_DIFFERENCES]}; "
            f"{len(only_committed)} only in the file, e.g. "
            f"{sorted(only_committed)[:SHOWN_DIFFERENCES]}"
        )
