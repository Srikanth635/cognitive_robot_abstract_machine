"""
Writing hypotheses as RDF in the ontology's own terms, and reading RDF back into
vocabulary instances.

Every class, property and enumeration member of the vocabulary carries the IRI of the
ontology term it stands for, so a hypothesis written as RDF can be checked by a reasoner
together with the ontology, stored as Turtle, and read back.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import StrEnum
from operator import attrgetter

from rdflib import OWL, RDF, RDFS, XSD, Graph, Literal, URIRef
from rdflib.term import Node
from typing_extensions import Any, Optional, Sequence

from llm_reasoner.acceptance.vocabulary import Vocabulary
from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.generation.schema import is_inferred
from llm_reasoner.generation.wire import DecodedHypothesis
from llm_reasoner.knowledge.ontologies import PREDICTIVE_MANIPULATION_NAMESPACE_ROOT
from llm_reasoner.knowledge.ontologies.terms import (
    ConceptReference,
    DescriptionEntity,
    PropertyTerm,
)

HYPOTHESIS_NAMESPACE_ROOT = "http://www.ease-crc.org/ont/llm_reasoner/hypothesis/"
"""
The IRI prefix of the namespaces stored hypotheses write their fragments in.
"""

ANNOTATION_PROPERTIES = (RDFS.comment,)
"""
Properties that annotate an individual without describing it, and are not read.
"""

# %% exceptions


@dataclass
class AmbiguousIndividualType(LLMReasonerError):
    """
    An individual belongs to several vocabulary classes, none of which specialises all
    the others, so no single generated class can stand for it.
    """

    individual: str
    """
    The IRI of the individual.
    """

    classes: list[str]
    """
    The names of the competing classes.
    """

    def __post_init__(self):
        super().__init__(
            f"{self.individual} belongs to {', '.join(self.classes)}, of which none "
            "specialises the others"
        )


@dataclass
class SeveralValuesOfFunctionalProperty(LLMReasonerError):
    """
    An individual states several values of a property that takes at most one.
    """

    individual: str
    """
    The IRI of the individual.
    """

    property_iri: str
    """
    The IRI of the property.
    """

    def __post_init__(self):
        super().__init__(
            f"{self.individual} states several values of {self.property_iri}, which "
            "takes at most one"
        )


# %% writing


@dataclass
class GraphWriter:
    """
    Writes a hypothesis as RDF.
    """

    namespace: str
    """
    The namespace the fragments are named in.
    """

    def write(self, hypothesis: DecodedHypothesis) -> Graph:
        """
        :return: Every fragment as an individual of its ontology class, with its
            property values, leaving out properties the ontology infers from
            sub-properties.
        """
        names = {
            id(fragment): URIRef(
                self.namespace
                + (hypothesis.identifier_of(fragment) or f"fragment_{position}")
            )
            for position, fragment in enumerate(hypothesis.fragments)
        }
        return self.write_fragments(hypothesis.fragments, names)

    def write_fragments(
        self, fragments: Sequence[DescriptionEntity], names: dict[int, URIRef]
    ) -> Graph:
        """
        :param fragments: The fragments to write.
        :param names: The IRI of every fragment, keyed by its identity; a value naming a
            fragment without one is left out.
        :return: Every fragment as an individual of its ontology class, with its
            property values, leaving out properties the ontology infers from
            sub-properties.
        """
        graph = Graph()
        for fragment in fragments:
            subject = names[id(fragment)]
            graph.add((subject, RDF.type, OWL.NamedIndividual))
            graph.add((subject, RDF.type, URIRef(type(fragment).term.iri)))
            if fragment.label:
                graph.add((subject, RDFS.label, Literal(fragment.label)))
            for declared in fields(fragment):
                term = PropertyTerm.of_field(type(fragment), declared.name)
                if term is None or is_inferred(type(fragment), declared.name):
                    continue
                value = attrgetter(declared.name)(fragment)
                for held in value if isinstance(value, list) else [value]:
                    node = self.node(held, names)
                    if node is not None:
                        graph.add((subject, URIRef(term.iri), node))
        return graph

    @staticmethod
    def node(value: Any, names: dict[int, URIRef]) -> Optional[Node]:
        """
        :return: The RDF node standing for a value, or ``None`` for an absent value or a
            fragment outside the hypothesis.
        """
        if value is None:
            return None
        if isinstance(value, DescriptionEntity):
            return names.get(id(value))
        if isinstance(value, StrEnum):
            return URIRef(value.value)
        if isinstance(value, ConceptReference):
            return URIRef(value.iri)
        if isinstance(value, bool):
            return Literal(value)
        if isinstance(value, (int, float)):
            return Literal(float(value), datatype=XSD.double)
        return Literal(value)


# %% reading


@dataclass(frozen=True)
class Statement:
    """
    One statement about an individual that was not read into a field.
    """

    subject: Node
    """
    The individual the statement is about.
    """

    predicate: Node
    """
    The property the statement uses.
    """

    object: Node
    """
    The value the statement states.
    """


@dataclass
class ReadHypothesis:
    """
    The vocabulary instances an RDF graph describes.
    """

    fragments: list[DescriptionEntity]
    """
    Every individual typed with a vocabulary class, as an instance of it.
    """

    identifiers: dict[int, str] = field(default_factory=dict)
    """
    The IRI of each fragment, keyed by the fragment's identity.
    """

    unread_statements: list[Statement] = field(default_factory=list)
    """
    The statements about the fragments that no field of theirs stands for.
    """

    def root_of(self, root_type: type[DescriptionEntity]) -> DescriptionEntity:
        """
        :return: The one fragment of the given type, which a stored hypothesis was
            generated for.
        """
        (root,) = [
            fragment for fragment in self.fragments if isinstance(fragment, root_type)
        ]
        return root


@dataclass
class GraphReader:
    """
    Reads RDF typed with ontology classes into vocabulary instances.
    """

    vocabulary: Vocabulary = field(default_factory=Vocabulary)
    """
    The generated vocabulary the IRIs are looked up in.
    """

    def read(self, graph: Graph) -> ReadHypothesis:
        """
        :return: Every individual typed with a vocabulary class as an instance of its
            most specific class, linked by its property values.
        """
        instances: dict[Node, DescriptionEntity] = {}
        for subject in sorted(set(graph.subjects(RDF.type, None)), key=str):
            generated = self.class_of(graph, subject)
            if generated is not None:
                instances[subject] = generated()
        unread: list[Statement] = []
        for subject, instance in instances.items():
            unread.extend(self.fill(graph, subject, instance, instances))
            self.vocabulary.fill_entailed_values(instance)
        return ReadHypothesis(
            fragments=list(instances.values()),
            identifiers={
                id(instance): str(subject) for subject, instance in instances.items()
            },
            unread_statements=unread,
        )

    def class_of(
        self, graph: Graph, subject: Node
    ) -> Optional[type[DescriptionEntity]]:
        """
        :return: The most specific generated class among an individual's types, or
            ``None`` when none of its types is a vocabulary class. Defined classes only
            classify, so they are passed over when another class is stated.
        """
        stated = [
            definition
            for type_iri in graph.objects(subject, RDF.type)
            if str(type_iri).startswith(PREDICTIVE_MANIPULATION_NAMESPACE_ROOT)
            for definition in [self.vocabulary.definition_named(str(type_iri))]
            if isinstance(definition, type)
            and issubclass(definition, DescriptionEntity)
        ]
        primitive = [
            definition for definition in stated if definition.term.definition is None
        ]
        candidates = primitive or stated
        most_specific = [
            candidate
            for candidate in candidates
            if all(issubclass(candidate, other) for other in candidates)
        ]
        if not candidates:
            return None
        if not most_specific:
            raise AmbiguousIndividualType(
                individual=str(subject),
                classes=sorted(candidate.__name__ for candidate in candidates),
            )
        return most_specific[0]

    def fill(
        self,
        graph: Graph,
        subject: Node,
        instance: DescriptionEntity,
        instances: dict[Node, DescriptionEntity],
    ) -> list[Statement]:
        """
        Set an instance's fields from the statements about its individual.

        :return: The statements no field of the instance stands for.
        """
        fields_by_property = {
            term.iri: (declared.name, term.functional)
            for declared in fields(instance)
            for term in [PropertyTerm.of_field(type(instance), declared.name)]
            if term is not None
        }
        values: dict[str, list[Any]] = {}
        unread = []
        for predicate, value in graph.predicate_objects(subject):
            if predicate == RDF.type or predicate in ANNOTATION_PROPERTIES:
                continue
            if predicate == RDFS.label:
                instance.label = str(value)
                continue
            if str(predicate) not in fields_by_property:
                unread.append(
                    Statement(subject=subject, predicate=predicate, object=value)
                )
                continue
            values.setdefault(str(predicate), []).append(self.value(value, instances))
        for property_iri, read_values in values.items():
            name, functional = fields_by_property[property_iri]
            if functional and len(read_values) > 1:
                raise SeveralValuesOfFunctionalProperty(
                    individual=str(subject), property_iri=property_iri
                )
            setattr(instance, name, read_values[0] if functional else read_values)
        return unread

    def value(self, node: Node, instances: dict[Node, DescriptionEntity]) -> Any:
        """
        :return: The Python value an RDF node stands for: a fragment, an enumeration
            member, a concept reference or a literal.
        """
        if node in instances:
            return instances[node]
        if isinstance(node, Literal):
            return node.toPython()
        if str(node) in self.vocabulary.members:
            return self.vocabulary.members[str(node)]
        return ConceptReference(iri=str(node))
