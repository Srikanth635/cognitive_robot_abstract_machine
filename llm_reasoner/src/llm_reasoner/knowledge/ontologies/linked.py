"""
The ontologies the microworld links to, CRAM-AICOR and DOLCE+DnS Ultralite, which are
not part of the package: reading their files, and the terms they declare.
"""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from pathlib import Path

from rdflib import RDF, Graph
from typing_extensions import Optional, Sequence

from llm_reasoner.knowledge.ontologies import (
    CRAM_AICOR_NAMESPACE,
    DUL_NAMESPACE,
    LinkedOntologyVariable,
)

LINKED_NAMESPACES = (CRAM_AICOR_NAMESPACE, DUL_NAMESPACE)
"""
The namespaces of the terms the linked ontologies declare.
"""

RDF_XML_SUFFIXES = (".owl", ".rdf", ".xml")
"""
The file suffixes of ontologies written in RDF/XML rather than Turtle.
"""


@dataclass(frozen=True)
class OntologyFiles:
    """
    Ontology files, in Turtle or RDF/XML.
    """

    paths: Sequence[Path]
    """
    The files.
    """

    @classmethod
    def linked_from_environment(cls) -> Optional[OntologyFiles]:
        """
        :return: The files of the linked ontologies the environment names, or ``None``
            when it does not name all of them.
        """
        locations = [os.environ.get(variable) for variable in LinkedOntologyVariable]
        if not all(locations):
            return None
        return cls(paths=[Path(location) for location in locations])

    def graph(self) -> Graph:
        """
        :return: Every file read into one graph.
        """
        graph = Graph()
        for path in self.paths:
            graph.parse(
                path, format="xml" if path.suffix in RDF_XML_SUFFIXES else "turtle"
            )
        return graph


@dataclass(frozen=True)
class LinkedOntologyTerms:
    """
    The terms the linked ontologies declare, which a concept reference naming one of
    them must be among.
    """

    terms: frozenset[str]
    """
    The IRIs of every class, individual and property the files declare.
    """

    @classmethod
    def from_files(cls, files: OntologyFiles) -> LinkedOntologyTerms:
        """
        :return: The terms the files declare.
        """
        return cls(
            terms=frozenset(
                str(subject) for subject in files.graph().subjects(RDF.type, None)
            )
        )

    @classmethod
    def from_environment(cls) -> Optional[LinkedOntologyTerms]:
        """
        :return: The terms of the linked ontologies the environment names, or ``None``
            when it does not name all of them.
        """
        files = OntologyFiles.linked_from_environment()
        return cls.from_files(files) if files is not None else None

    @staticmethod
    def concerns(iri: str) -> bool:
        """
        :return: Whether the term is in the namespace of a linked ontology, so a linked
            ontology has to declare it.
        """
        return iri.startswith(LINKED_NAMESPACES)

    def declares(self, iri: str) -> bool:
        """
        :return: Whether a linked ontology declares the term.
        """
        return iri in self.terms

    def closest(self, iri: str, count: int = 3) -> list[str]:
        """
        :return: The declared terms of the same namespace whose names read most like
            the given term's.
        """
        namespace, separator, name = iri.rpartition("#")
        candidates = {
            term.rpartition("#")[2]: term
            for term in self.terms
            if term.startswith(namespace + separator)
        }
        return [
            candidates[match]
            for match in difflib.get_close_matches(name, list(candidates), n=count)
        ]
