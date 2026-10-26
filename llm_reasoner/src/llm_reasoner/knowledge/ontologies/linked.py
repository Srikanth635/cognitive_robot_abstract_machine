"""
The ontologies the microworld links to, CRAM-AICOR and DOLCE+DnS Ultralite, as pinned
copies shipped with the package: reading their files, the classes they declare, and the
hierarchy their classes and ours form.
"""

from __future__ import annotations

import difflib
from collections import deque
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from krrood.patterns.path_enumeration import PathEnumeration
from rdflib import OWL, RDF, RDFS, Graph, URIRef
from rdflib.collection import Collection
from rdflib.term import Node
from typing_extensions import Iterator, Mapping, Sequence

from llm_reasoner.knowledge.ontologies import (
    CRAM_AICOR_NAMESPACE,
    DUL_NAMESPACE,
    KNOWLEDGE_ONTOLOGY_MODULES,
)

LINKED_NAMESPACES = (CRAM_AICOR_NAMESPACE, DUL_NAMESPACE)
"""
The namespaces of the terms the linked ontologies declare.
"""

RDF_XML_SUFFIXES = (".owl", ".rdf", ".xml")
"""
The file suffixes of ontologies written in RDF/XML rather than Turtle.
"""


# %% the pinned files


class LinkedOntologyFile(PathEnumeration):
    """
    The pinned copy of each linked ontology; ``sources.json`` beside them records where
    each comes from and the digest of its bytes.
    """

    CRAM_AICOR = Path(__file__).parent / "linked_ontologies" / "cram-aicor.ttl"
    """
    CRAM-AICOR, in Turtle.
    """

    DUL = Path(__file__).parent / "linked_ontologies" / "DUL.owl"
    """
    DOLCE+DnS Ultralite, in RDF/XML.
    """


class AlignmentFile(PathEnumeration):
    """
    The modules of ours that align the linked ontologies with each other.
    """

    DUL_ALIGNMENT = Path(__file__).parent / "alignment" / "dul-alignment.ttl"
    """
    CRAM-AICOR's copies of DUL's classes equated with DUL.
    """


# %% reading the files


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
    def pinned(cls) -> OntologyFiles:
        """
        :return: The pinned copies of the linked ontologies, CRAM-AICOR first.
        """
        return cls(paths=list(LinkedOntologyFile))

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
    The classes the linked ontologies declare, which a concept naming one of them must
    be among.
    """

    terms: frozenset[str]
    """
    The IRIs of every class the files declare.
    """

    @classmethod
    def from_files(cls, files: OntologyFiles) -> LinkedOntologyTerms:
        """
        :return: The classes the files declare.
        """
        return cls(
            terms=frozenset(
                str(subject)
                for subject in files.graph().subjects(RDF.type, OWL.Class)
                if isinstance(subject, URIRef)
            )
        )

    @classmethod
    @cache
    def pinned(cls) -> LinkedOntologyTerms:
        """
        :return: The classes the pinned linked ontologies declare, read once.
        """
        return cls.from_files(OntologyFiles.pinned())

    @staticmethod
    def concerns(iri: str) -> bool:
        """
        :return: Whether the term is in the namespace of a linked ontology, so a linked
            ontology has to declare it.
        """
        return iri.startswith(LINKED_NAMESPACES)

    def declares(self, iri: str) -> bool:
        """
        :return: Whether a linked ontology declares the term as a class.
        """
        return iri in self.terms

    def closest(self, iri: str, count: int = 3) -> list[str]:
        """
        :return: The declared classes of the same namespace whose names read most like
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


# %% the class hierarchy


@dataclass(frozen=True)
class LinkedClassHierarchy:
    """
    The named superclasses every class is told to have, directly or through others, by
    ``rdfs:subClassOf``, ``owl:equivalentClass`` and the named operands of the
    intersections they state.
    """

    superclasses: Mapping[str, frozenset[str]]
    """
    The IRIs of every class's told superclasses, the class itself included, by the
    class's IRI.
    """

    @classmethod
    def from_graph(cls, graph: Graph) -> LinkedClassHierarchy:
        """
        :return: The hierarchy the graph's classes form.
        """
        parents: dict[Node, set[URIRef]] = {}
        for subject, value in graph.subject_objects(RDFS.subClassOf):
            parents.setdefault(subject, set()).update(cls.named_in(graph, value))
        for first, second in graph.subject_objects(OWL.equivalentClass):
            parents.setdefault(first, set()).update(cls.named_in(graph, second))
            parents.setdefault(second, set()).update(cls.named_in(graph, first))
        classes = {
            subject
            for subject in graph.subjects(RDF.type, OWL.Class)
            if isinstance(subject, URIRef)
        } | {subject for subject in parents if isinstance(subject, URIRef)}
        return cls(
            superclasses={
                str(named): frozenset(
                    str(superclass) for superclass in cls.closure(named, parents)
                )
                for named in classes
            }
        )

    @classmethod
    @cache
    def pinned(cls) -> LinkedClassHierarchy:
        """
        :return: The hierarchy of the pinned linked ontologies, aligned with each other,
            together with the knowledge's own modules, read once.
        """
        return cls.from_graph(
            OntologyFiles(
                paths=[
                    *LinkedOntologyFile,
                    *AlignmentFile,
                    *(module_file.path for module_file in KNOWLEDGE_ONTOLOGY_MODULES),
                ]
            ).graph()
        )

    @staticmethod
    def named_in(graph: Graph, node: Node) -> Iterator[URIRef]:
        """
        :return: The named classes a class expression is or intersects.
        """
        if isinstance(node, URIRef):
            yield node
            return
        operands = graph.value(node, OWL.intersectionOf)
        if operands is None:
            return
        for operand in Collection(graph, operands):
            yield from LinkedClassHierarchy.named_in(graph, operand)

    @staticmethod
    def closure(named: URIRef, parents: Mapping[Node, set[URIRef]]) -> set[URIRef]:
        """
        :return: The class and every class reached from it through its parents.
        """
        reached, pending = {named}, deque([named])
        while pending:
            for parent in parents.get(pending.popleft(), set()):
                if parent not in reached:
                    reached.add(parent)
                    pending.append(parent)
        return reached

    def declares(self, iri: str) -> bool:
        """
        :return: Whether the class is in the hierarchy.
        """
        return iri in self.superclasses

    def is_subclass(self, specific: str, general: str) -> bool:
        """
        :return: Whether the specific class is told to specialise the general one, or is
            it; every class specialises ``owl:Thing``.
        """
        return general == str(OWL.Thing) or general in self.superclasses.get(
            specific, frozenset((specific,))
        )
