"""
Writing a proposed vocabulary as an ontology module of the action perspectives, in the
form the hand-written modules take, so the vocabulary generator reads it as it reads
them.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, URIRef
from rdflib.collection import Collection

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

    def graph(self) -> Graph:
        """
        :return: The module as a graph.
        """
        graph = Graph()
        graph.bind(self.perspective.value.split("-")[-1], self.perspective.namespace)
        for prefix in Prefix:
            graph.bind(prefix.value, prefix.namespace)
        self.write_header(graph)
        for proposed in self.proposal.classes:
            self.write_class(graph, proposed)
        for partition in self.proposal.value_partitions:
            self.write_value_partition(graph, partition)
        for proposed in self.proposal.properties:
            self.write_property(graph, proposed)
        return graph

    def turtle(self) -> str:
        """
        :return: The module as Turtle.
        """
        return self.graph().serialize(format="turtle")

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
        Write an object property, its domain, range and the properties it specialises.
        """
        term = self.iri(proposed.name)
        graph.add((term, RDF.type, OWL.ObjectProperty))
        if proposed.functional:
            graph.add((term, RDF.type, OWL.FunctionalProperty))
        self.write_named(graph, term, proposed.name, proposed.meaning)
        graph.add((term, RDFS.domain, self.iri(proposed.domain)))
        graph.add((term, RDFS.range, self.iri(proposed.value_class)))
        for parent in proposed.parents:
            graph.add((term, RDFS.subPropertyOf, self.iri(parent)))
        for source in proposed.see_also:
            graph.add((term, RDFS.seeAlso, URIRef(source)))
