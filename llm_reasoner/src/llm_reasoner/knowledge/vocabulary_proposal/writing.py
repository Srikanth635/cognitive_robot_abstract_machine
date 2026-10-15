"""
Writing a proposed vocabulary as an ontology module of the action perspectives, in the
form the hand-written modules take, so the vocabulary generator reads it as it reads
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from itertools import groupby

from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, URIRef
from rdflib.collection import Collection
from rdflib.compare import to_canonical_graph
from typing_extensions import Callable, List

from llm_reasoner.knowledge.ontologies import (
    CRAM_AICOR_ONTOLOGY_IRI,
    DUL_ONTOLOGY_IRI,
    ActionPerspectiveFile,
)
from llm_reasoner.knowledge.vocabulary_proposal.catalogue import Prefix, is_prefixed
from llm_reasoner.knowledge.vocabulary_proposal.perspective import (
    Perspective,
    label_of,
)
from llm_reasoner.knowledge.vocabulary_proposal.proposal import (
    ProposedClass,
    ProposedProperty,
    ProposedRestriction,
    ProposedValuePartition,
    RestrictionKind,
    VocabularyProposal,
)

PROPOSED_VERSION = "0.1.0"
"""
The version a proposed module starts at.
"""

LANGUAGE = "en"
"""
The language labels and definitions are written in.
"""

MODULE_LABEL = "Action perspectives: {perspective}"
"""
The label of a perspective's module.
"""

RESTRICTION_PREDICATES = {
    RestrictionKind.SOME: OWL.someValuesFrom,
    RestrictionKind.ONLY: OWL.allValuesFrom,
    RestrictionKind.VALUE: OWL.hasValue,
}
"""
How each kind of restriction names its filler.
"""

TURTLE = "turtle"
"""
The format a module is written in.
"""

PREFIX_DIRECTIVE = "@prefix"
"""
How a line declaring a prefix starts in Turtle.
"""

SECTION_HEADER = "# %% {section}"
"""
The comment opening a section of a module, as in the hand-written modules.
"""


class ModuleSection(StrEnum):
    """
    The sections of a written module, in the order they are written.
    """

    ONTOLOGY = "ontology"
    CLASSES = "classes"
    VALUE_PARTITIONS = "value partitions"
    PROPERTIES = "properties"


@dataclass(frozen=True)
class TermBlock:
    """
    The statements about one term of a module, written together.
    """

    section: ModuleSection
    """
    The section the block is written in.
    """

    graph: Graph
    """
    The statements, with the blank nodes only they refer to.
    """


@dataclass(frozen=True)
class ProposalWriter:
    """
    Writes a proposed vocabulary as the module of its perspective.
    """

    proposal: VocabularyProposal
    """
    What is written.
    """

    perspective: Perspective
    """
    The perspective the vocabulary is of.
    """

    def iri(self, name: str) -> URIRef:
        """
        :return: The IRI of a term: a prefixed name in its ontology, a bare name in the
            perspective's.
        """
        if is_prefixed(name):
            return Prefix.iri_of(name)
        return URIRef(self.perspective.namespace + name)

    def empty_graph(self) -> Graph:
        """
        :return: A graph knowing the prefixes of the module's terms.
        """
        graph = Graph()
        graph.bind(self.perspective.value.split("-")[-1], self.perspective.namespace)
        for prefix in Prefix:
            graph.bind(prefix.value, prefix.namespace)
        return graph

    def block(
        self, section: ModuleSection, write: Callable[[Graph], None]
    ) -> TermBlock:
        """
        :return: The block of what ``write`` writes.
        """
        graph = self.empty_graph()
        write(graph)
        return TermBlock(section=section, graph=graph)

    def blocks(self) -> List[TermBlock]:
        """
        :return: A block for the ontology and one for each term, in the order the
            proposal declares them.
        """
        return [
            self.block(ModuleSection.ONTOLOGY, self.write_header),
            *(
                self.block(
                    ModuleSection.CLASSES, partial(self.write_class, proposed=proposed)
                )
                for proposed in self.proposal.classes
            ),
            *(
                self.block(
                    ModuleSection.VALUE_PARTITIONS,
                    partial(self.write_value_partition, partition=partition),
                )
                for partition in self.proposal.value_partitions
            ),
            *(
                self.block(
                    ModuleSection.PROPERTIES,
                    partial(self.write_property, proposed=proposed),
                )
                for proposed in self.proposal.properties
            ),
        ]

    def graph(self) -> Graph:
        """
        :return: The module as a graph.
        """
        graph = self.empty_graph()
        for block in self.blocks():
            graph += block.graph
        return graph

    def turtle(self) -> str:
        """
        :return: The module as Turtle, one block per term in the proposal's order, so an
            edit to one term changes only its own lines.
        """
        blocks = self.blocks()
        prefixes = [
            line
            for line in self.graph().serialize(format=TURTLE).splitlines()
            if line.startswith(PREFIX_DIRECTIVE)
        ]
        sections = [
            SECTION_HEADER.format(section=section.value)
            + "\n\n"
            + "\n\n".join(self.statements(block) for block in grouped)
            for section, grouped in groupby(blocks, key=lambda block: block.section)
        ]
        return "\n".join(prefixes) + "\n\n" + "\n\n".join(sections) + "\n"

    def statements(self, block: TermBlock) -> str:
        """
        :return: A block's statements as Turtle, its blank nodes named by their content
            so that their order does not depend on the rest of the module.
        """
        canonical = self.empty_graph()
        canonical += to_canonical_graph(block.graph)
        return "\n".join(
            line
            for line in canonical.serialize(format=TURTLE).splitlines()
            if not line.startswith(PREFIX_DIRECTIVE)
        ).strip()

    def write_header(self, graph: Graph) -> None:
        """
        Write the module's ontology, which imports the action core, CRAM-AICOR and DUL.
        """
        ontology = URIRef(self.perspective.ontology_iri)
        graph.add((ontology, RDF.type, OWL.Ontology))
        graph.add(
            (
                ontology,
                OWL.versionIRI,
                URIRef(f"{self.perspective.ontology_iri}/{PROPOSED_VERSION}"),
            )
        )
        graph.add((ontology, OWL.versionInfo, Literal(PROPOSED_VERSION)))
        for imported in (
            ActionPerspectiveFile.ACTION_CORE.namespace.rstrip("#"),
            CRAM_AICOR_ONTOLOGY_IRI,
            DUL_ONTOLOGY_IRI,
        ):
            graph.add((ontology, OWL.imports, URIRef(imported)))
        graph.add(
            (
                ontology,
                RDFS.label,
                Literal(
                    MODULE_LABEL.format(perspective=self.perspective.label),
                    lang=LANGUAGE,
                ),
            )
        )
        graph.add(
            (ontology, RDFS.comment, Literal(self.proposal.summary, lang=LANGUAGE))
        )

    def write_named(self, graph: Graph, term: URIRef, name: str, meaning: str) -> None:
        """
        Write a term's label, made from its name, and its definition.
        """
        graph.add((term, RDFS.label, Literal(label_of(name), lang=LANGUAGE)))
        graph.add((term, RDFS.comment, Literal(meaning, lang=LANGUAGE)))

    def write_class(self, graph: Graph, proposed: ProposedClass) -> None:
        """
        Write a class, the classes it specialises, its restrictions and its sources.
        """
        term = self.iri(proposed.name)
        graph.add((term, RDF.type, OWL.Class))
        self.write_named(graph, term, proposed.name, proposed.meaning)
        for parent in proposed.parents:
            graph.add((term, RDFS.subClassOf, self.iri(parent)))
        for restriction in proposed.restrictions:
            graph.add((term, RDFS.subClassOf, self.restriction(graph, restriction)))
        for source in proposed.see_also:
            graph.add((term, RDFS.seeAlso, URIRef(source)))

    def restriction(self, graph: Graph, restriction: ProposedRestriction) -> BNode:
        """
        :return: A restriction, written into the graph.
        """
        node = BNode()
        graph.add((node, RDF.type, OWL.Restriction))
        graph.add((node, OWL.onProperty, self.iri(restriction.on_property)))
        graph.add(
            (
                node,
                RESTRICTION_PREDICATES[restriction.kind],
                self.iri(restriction.filler),
            )
        )
        return node

    def write_value_partition(
        self, graph: Graph, partition: ProposedValuePartition
    ) -> None:
        """
        Write a value partition as the class of exactly its members, each an individual
        of it and distinct from the others.
        """
        term = self.iri(partition.name)
        members = [self.iri(member.name) for member in partition.members]
        graph.add((term, RDF.type, OWL.Class))
        self.write_named(graph, term, partition.name, partition.meaning)
        for parent in partition.parents:
            graph.add((term, RDFS.subClassOf, self.iri(parent)))
        enumeration = BNode()
        graph.add((enumeration, RDF.type, OWL.Class))
        graph.add((enumeration, OWL.oneOf, Collection(graph, BNode(), members).uri))
        graph.add((term, OWL.equivalentClass, enumeration))
        for member, individual in zip(partition.members, members):
            graph.add((individual, RDF.type, OWL.NamedIndividual))
            graph.add((individual, RDF.type, term))
            self.write_named(graph, individual, member.name, member.meaning)
        distinct = BNode()
        graph.add((distinct, RDF.type, OWL.AllDifferent))
        graph.add(
            (distinct, OWL.distinctMembers, Collection(graph, BNode(), members).uri)
        )

    def write_property(self, graph: Graph, proposed: ProposedProperty) -> None:
        """
        Write an object property, its domain, its range if it has one, and the
        properties it specialises.
        """
        term = self.iri(proposed.name)
        graph.add((term, RDF.type, OWL.ObjectProperty))
        if proposed.functional:
            graph.add((term, RDF.type, OWL.FunctionalProperty))
        self.write_named(graph, term, proposed.name, proposed.meaning)
        graph.add((term, RDFS.domain, self.iri(proposed.domain)))
        if proposed.value_class is not None:
            graph.add((term, RDFS.range, self.iri(proposed.value_class)))
        for parent in proposed.parents:
            graph.add((term, RDFS.subPropertyOf, self.iri(parent)))
        for source in proposed.see_also:
            graph.add((term, RDFS.seeAlso, URIRef(source)))
