"""
The terms a proposal may build on: those of the action core and those of the ontologies
it links to, CRAM-AICOR and DUL, each named with the prefix of its ontology.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path

from rdflib import OWL, RDF, RDFS, Graph, URIRef
from typing_extensions import Any, Dict, List, Optional, Sequence, Tuple

from llm_reasoner.knowledge.ontologies import (
    CRAM_AICOR_NAMESPACE,
    DUL_NAMESPACE,
    ActionPerspectiveFile,
)
from llm_reasoner.knowledge.ontologies.linked import OntologyFiles

PREFIX_SEPARATOR = ":"
"""
What separates a prefix from a name, as in ``aicor:Container``.
"""


class Prefix(StrEnum):
    """
    The prefixes a proposal names the terms of other ontologies with.
    """

    CORE = "core"
    """
    The action core.
    """

    AICOR = "aicor"
    """
    CRAM-AICOR.
    """

    DUL = "dul"
    """
    DOLCE+DnS Ultralite.
    """

    @property
    def namespace(self) -> str:
        """
        The namespace the prefix stands for.
        """
        return {
            Prefix.CORE: ActionPerspectiveFile.ACTION_CORE.namespace,
            Prefix.AICOR: CRAM_AICOR_NAMESPACE,
            Prefix.DUL: DUL_NAMESPACE,
        }[self]

    @classmethod
    def prefixed(cls, term: URIRef) -> Optional[str]:
        """
        :return: A term named with the prefix of its ontology; ``None`` for a term of
            none of them.
        """
        for prefix in cls:
            if str(term).startswith(prefix.namespace):
                return prefix + PREFIX_SEPARATOR + str(term)[len(prefix.namespace) :]
        return None

    @classmethod
    def iri_of(cls, name: str) -> Optional[URIRef]:
        """
        :return: The IRI a prefixed name stands for; ``None`` for a prefix that is none
            of these.
        """
        prefix, local = name.split(PREFIX_SEPARATOR, 1)
        if prefix not in {member.value for member in cls}:
            return None
        return URIRef(cls(prefix).namespace + local)


def is_prefixed(name: str) -> bool:
    """
    :return: Whether a name refers to a term of another ontology rather than one the
        proposal declares.
    """
    return PREFIX_SEPARATOR in name


class TermKind(StrEnum):
    """
    The kinds of term a proposal refers to.
    """

    CLASS = "class"
    PROPERTY = "property"
    INDIVIDUAL = "individual"


KIND_OF_TYPE = {
    OWL.Class: TermKind.CLASS,
    OWL.ObjectProperty: TermKind.PROPERTY,
    OWL.DatatypeProperty: TermKind.PROPERTY,
    OWL.NamedIndividual: TermKind.INDIVIDUAL,
}
"""
The kind of term each declaration declares.
"""


@dataclass(frozen=True)
class CatalogueEntry:
    """
    A term a proposal may build on.
    """

    name: str
    """
    The term, named with the prefix of its ontology.
    """

    kind: TermKind
    """
    What kind of term it is.
    """

    meaning: str
    """
    Its definition; empty when its ontology gives none.
    """

    parents: Tuple[str, ...]
    """
    The terms it specialises, among those of the catalogue.
    """

    domain: Optional[str]
    """
    For a property, the class its subjects belong to, when its ontology says.
    """

    value_class: Optional[str]
    """
    For a property, the class its values belong to, its range, when its ontology says.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The term as the model reads it, leaving out what its ontology does not
            say.
        """
        written: Dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.meaning:
            written["meaning"] = self.meaning
        if self.parents:
            written["parents"] = list(self.parents)
        if self.domain is not None:
            written["domain"] = self.domain
        if self.value_class is not None:
            written["range"] = self.value_class
        return written


def entries_of(graph: Graph, prefixes: Sequence[Prefix]) -> List[CatalogueEntry]:
    """
    :return: Every term a graph declares in the namespaces of the given prefixes, in the
        order of their names.
    """
    entries: Dict[str, CatalogueEntry] = {}
    for declaration_type, kind in KIND_OF_TYPE.items():
        for term in graph.subjects(RDF.type, declaration_type):
            name = Prefix.prefixed(term) if isinstance(term, URIRef) else None
            if name is None or Prefix(name.split(PREFIX_SEPARATOR)[0]) not in prefixes:
                continue
            if name in entries:
                continue
            parent_property = (
                RDFS.subPropertyOf if kind is TermKind.PROPERTY else RDFS.subClassOf
            )
            entries[name] = CatalogueEntry(
                name=name,
                kind=kind,
                meaning=str(graph.value(term, RDFS.comment) or ""),
                parents=tuple(
                    sorted(
                        parent
                        for parent in (
                            Prefix.prefixed(found)
                            for found in graph.objects(term, parent_property)
                            if isinstance(found, URIRef)
                        )
                        if parent is not None
                    )
                ),
                domain=(
                    Prefix.prefixed(graph.value(term, RDFS.domain))
                    if isinstance(graph.value(term, RDFS.domain), URIRef)
                    else None
                ),
                value_class=(
                    Prefix.prefixed(graph.value(term, RDFS.range))
                    if isinstance(graph.value(term, RDFS.range), URIRef)
                    else None
                ),
            )
    return [entries[name] for name in sorted(entries)]


@dataclass(frozen=True)
class TermCatalogue:
    """
    The terms of the action core and of the linked ontologies a proposal may build on.
    """

    core_graph: Graph
    """
    The action core module.
    """

    core: Tuple[CatalogueEntry, ...]
    """
    The action core's terms.
    """

    linked: Tuple[CatalogueEntry, ...]
    """
    The terms of CRAM-AICOR and DUL.
    """

    @classmethod
    def from_files(cls, core: Path, linked: Sequence[Path]) -> TermCatalogue:
        """
        :param core: The action core module file.
        :param linked: The files of CRAM-AICOR and DUL, in Turtle or RDF/XML.
        :return: The catalogue of their terms.
        """
        core_graph = Graph().parse(core, format="turtle")
        return cls(
            core_graph=core_graph,
            core=tuple(entries_of(core_graph, [Prefix.CORE])),
            linked=tuple(
                entries_of(
                    OntologyFiles(paths=linked).graph(), [Prefix.AICOR, Prefix.DUL]
                )
            ),
        )

    def core_terms(self) -> List[CatalogueEntry]:
        """
        :return: The action core's terms.
        """
        return list(self.core)

    def linked_terms(self) -> List[CatalogueEntry]:
        """
        :return: The terms of CRAM-AICOR and DUL.
        """
        return list(self.linked)

    def kind_of(self, name: str) -> Optional[TermKind]:
        """
        :return: What kind of term a prefixed name names; ``None`` when no ontology of
            the catalogue declares it.
        """
        return self.kinds.get(name)

    @cached_property
    def kinds(self) -> Dict[str, TermKind]:
        """
        The kind of every term, by its prefixed name.
        """
        return {entry.name: entry.kind for entry in (*self.core, *self.linked)}
