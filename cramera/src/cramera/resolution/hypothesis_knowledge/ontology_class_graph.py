"""
The ontology behind hypotheses, drawn at the level of classes: the classes the hypotheses
instantiate, the classes they specialise up to CRAM-AICOR and DUL, the concepts they cite,
and how they link one class to another.
"""

from __future__ import annotations

import inspect
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
from enum import StrEnum
from functools import cache
from operator import attrgetter

from llm_reasoner.knowledge.access import (
    ActionPerspectiveFile,
    ClosedWorldMembership,
    ConceptReference,
    described,
    DescriptionEntity,
    HypothesisView,
    is_inferred,
    OBJECT_KINDS_FILE,
    OntologyFiles,
    PREDICTIVE_MANIPULATION_NAMESPACE_ROOT,
    PredictiveManipulationOntologyFile,
    PropertyTerm,
    Vocabulary,
)
from rdflib import OWL, RDF, RDFS, Graph, URIRef
from rdflib.collection import Collection
from typing_extensions import Any, Optional

from cramera.resolution.hypothesis_knowledge.ontology_backing import OntologyBacking, TermSource

EVIDENCE_STATUS = URIRef(PREDICTIVE_MANIPULATION_NAMESPACE_ROOT + "core#evidenceStatus")
"""
The annotation stating the evidence status of a class or a source claim.
"""

INSTANCES_LISTED = 12
"""
How many instances of a class its details name before summarising the rest.
"""

# %% graph model


class OntologyNodeCategory(StrEnum):
    """
    What a class of the graph is to the hypothesis.
    """

    USED = "used"
    """
    A microworld class the hypothesis instantiates.
    """

    MICROWORLD = "microworld"
    """
    A microworld class without instances of its own: one a used class specialises, or a
    defined class whose definition instances meet.
    """

    LINKED = "linked"
    """
    A CRAM-AICOR or DUL class or concept the microworld specialises or the hypothesis
    cites.
    """

    INTRODUCED = "introduced"
    """
    A concept the hypothesis introduced because no linked ontology declares it.
    """


class OntologyEdgeKind(StrEnum):
    """
    How two classes of the graph are linked.
    """

    IS_A = "is_a"
    """
    The source class specialises the target class.
    """

    COMPONENT = "component"
    """
    The hypothesis's structure declares instances of the target class as its
    components.
    """

    RELATION = "relation"
    """
    The hypothesis links instances of the source class to instances of the target class
    through an ontology property.
    """

    CITES = "cites"
    """
    Instances of the source class cite the target concept as a value.
    """

    DISJOINT = "disjoint"
    """
    The ontology states that nothing belongs to both classes.
    """

    CLASSIFIED_AS = "classified_as"
    """
    Instances of the source class meet the definition of the target defined class, so
    its requirements apply to them.
    """


@dataclass(frozen=True)
class SourceClaim:
    """
    A claim from the literature a class represents.
    """

    iri: str
    """
    The IRI of the claim.
    """

    text: str
    """
    What the claim states.
    """

    evidence_status: str
    """
    How strongly the source supports the claim, as a prefixed term.
    """


@dataclass(frozen=True)
class OntologyClassNode:
    """
    One class of the graph.
    """

    iri: str
    """
    The IRI of the class or concept.
    """

    label: str
    """
    The name drawn beside the node.
    """

    category: OntologyNodeCategory
    """
    What the class is to the hypothesis.
    """

    source: TermSource
    """
    The ontology the class belongs to.
    """

    instance_count: int = 0
    """
    How many fragments of the hypothesis instantiate the class itself.
    """

    instances: tuple[str, ...] = ()
    """
    The names of the first instances, in the order the hypothesis lists them.
    """

    description: str = ""
    """
    What the ontology says the class is.
    """

    evidence_status: str = ""
    """
    How strongly the literature supports the class, as a prefixed term.
    """

    claims: tuple[SourceClaim, ...] = ()
    """
    The claims from the literature the class represents.
    """

    requirements: tuple[str, ...] = ()
    """
    What the class requires of its instances, in words.
    """

    definition: str = ""
    """
    The condition that makes something an instance of a defined class, in words.
    """


@dataclass(frozen=True)
class OntologyClassEdge:
    """
    One link between two classes of the graph.
    """

    source: str
    """
    The IRI of the class the link starts at.
    """

    target: str
    """
    The IRI of the class the link ends at.
    """

    kind: OntologyEdgeKind
    """
    How the classes are linked.
    """

    property_iri: str = ""
    """
    The IRI of the ontology property stating the link, for a link the hypothesis
    states.
    """

    count: int = 1
    """
    How many links of the hypothesis the edge stands for, or how many instances a
    definition classifies.
    """


# %% what the ontology files say


@dataclass(frozen=True)
class OntologyText:
    """
    The statements of the ontology files the graph reads descriptions, claims and
    disjointness from.
    """

    graph: Graph
    """
    The modules of the knowledge and the kinds of object, with CRAM-AICOR and DUL when
    the environment names them.
    """

    @classmethod
    @cache
    def shipped(cls) -> OntologyText:
        """
        :return: The statements of the modules of the knowledge, of the kinds of object
            and of the linked ontologies the environment names.
        """
        paths = [
            module.path
            for family in (PredictiveManipulationOntologyFile, ActionPerspectiveFile)
            for module in family
        ]
        paths.append(OBJECT_KINDS_FILE)
        linked = OntologyFiles.linked_from_environment()
        if linked is not None:
            paths.extend(linked.paths)
        return cls(graph=OntologyFiles(paths=paths).graph())

    def description(self, iri: str) -> str:
        """
        :return: What the ontology says a term is, or an empty string.
        """
        comment = self.graph.value(URIRef(iri), RDFS.comment)
        return "" if comment is None else str(comment)

    def requirements_of(self, iri: str) -> tuple[str, ...]:
        """
        :return: What the ontology requires of the instances of a class through a value
            a property must have, in words; empty for a class with no such restriction.
        """
        restrictions = (
            restriction
            for restriction in self.graph.objects(URIRef(iri), RDFS.subClassOf)
            if (restriction, RDF.type, OWL.Restriction) in self.graph
        )
        return tuple(
            f"{self.graph.value(restricted, RDFS.label) or restricted}: {value}"
            for restriction in restrictions
            for restricted in [self.graph.value(restriction, OWL.onProperty)]
            for value in self.graph.objects(restriction, OWL.hasValue)
        )

    def claim(self, iri: str, backing: OntologyBacking) -> SourceClaim:
        """
        :return: A claim with its text and evidence status.
        """
        status = self.graph.value(URIRef(iri), EVIDENCE_STATUS)
        return SourceClaim(
            iri=iri,
            text=self.description(iri),
            evidence_status="" if status is None else backing.prefixed(str(status)),
        )

    def disjoint_pairs(self) -> set[frozenset[str]]:
        """
        :return: Every pair of classes the ontology declares disjoint.
        """
        pairs = {
            frozenset({str(first), str(second)})
            for first, second in self.graph.subject_objects(OWL.disjointWith)
        }
        for group in self.graph.subjects(RDF.type, OWL.AllDisjointClasses):
            members = [
                str(member)
                for member in Collection(
                    self.graph, self.graph.value(group, OWL.members)
                )
            ]
            pairs |= {
                frozenset({first, second})
                for index, first in enumerate(members)
                for second in members[index + 1 :]
            }
        return pairs


# %% building


@dataclass
class OntologyClassGraph:
    """
    The classes behind one hypothesis and the links between them.
    """

    nodes: list[OntologyClassNode]
    """
    Every class and concept.
    """

    edges: list[OntologyClassEdge]
    """
    Every link between them.
    """

    @classmethod
    def of(cls, *views: HypothesisView) -> OntologyClassGraph:
        """
        :return: The graph of the classes the hypotheses instantiate, the classes they
            specialise, the concepts they cite, and their links between them.
        """
        return OntologyClassGraphBuilder(views=views).build()


@dataclass
class OntologyClassGraphBuilder:
    """
    Builds the class graph of hypotheses drawn together.
    """

    views: tuple[HypothesisView, ...]
    """
    The hypotheses drawn.
    """

    text: OntologyText = field(default_factory=OntologyText.shipped)
    """
    The ontology statements descriptions, claims and disjointness are read from.
    """

    backing: OntologyBacking = field(default_factory=OntologyBacking)
    """
    Writes terms with their prefixes and tells which ontology a term comes from.
    """

    vocabulary: Vocabulary = field(default_factory=Vocabulary)
    """
    The generated vocabulary classes are read from.
    """

    @property
    def fragments(self) -> list[DescriptionEntity]:
        """
        Every fragment of every hypothesis drawn.
        """
        return [fragment for view in self.views for fragment in view.fragments]

    def fragment_name(self, fragment: DescriptionEntity) -> str:
        """
        :return: The label of a fragment, or its name in the hypothesis holding it.
        """
        if fragment.label:
            return fragment.label
        (view,) = [view for view in self.views if id(fragment) in view.identifiers]
        return view.citation_of(fragment).fragment_name

    def build(self) -> OntologyClassGraph:
        """
        :return: The class graph of the hypotheses.
        """
        instances: dict[type[DescriptionEntity], list[DescriptionEntity]] = defaultdict(
            list
        )
        for fragment in self.fragments:
            instances[type(fragment)].append(fragment)
        classes = self.classes_specialised(instances)
        nodes: dict[str, OntologyClassNode] = {
            vocabulary_class.term.iri: self.class_node(
                vocabulary_class, instances.get(vocabulary_class, [])
            )
            for vocabulary_class in classes
        }
        edges = self.is_a_edges(classes) + self.hypothesis_edges()
        for edge in edges:
            if edge.target not in nodes:
                nodes[edge.target] = self.term_node(edge.target)
        for edge in self.classification_edges(instances):
            if edge.target not in nodes:
                nodes[edge.target] = self.class_node(
                    self.vocabulary.definition_named(edge.target), []
                )
            edges.append(edge)
        edges += self.disjoint_edges(set(nodes))
        return OntologyClassGraph(nodes=list(nodes.values()), edges=edges)

    @staticmethod
    def classes_specialised(
        instances: dict[type[DescriptionEntity], list[DescriptionEntity]],
    ) -> list[type[DescriptionEntity]]:
        """
        :return: The instantiated classes and every microworld class they specialise.
        """
        classes: dict[str, type[DescriptionEntity]] = {}
        for instantiated in instances:
            for vocabulary_class in instantiated.__mro__:
                if issubclass(vocabulary_class, DescriptionEntity) and (
                    "term" in vars(vocabulary_class)
                ):
                    classes.setdefault(vocabulary_class.term.iri, vocabulary_class)
        return list(classes.values())

    def class_node(
        self,
        vocabulary_class: type[DescriptionEntity],
        instances: list[DescriptionEntity],
    ) -> OntologyClassNode:
        """
        :return: The node of a microworld class.
        """
        term = vocabulary_class.term
        return OntologyClassNode(
            iri=term.iri,
            label=vocabulary_class.__name__,
            category=(
                OntologyNodeCategory.USED
                if instances
                else OntologyNodeCategory.MICROWORLD
            ),
            source=self.backing.source_of(term.iri),
            instance_count=len(instances),
            instances=tuple(
                self.fragment_name(instance)
                for instance in instances[:INSTANCES_LISTED]
            ),
            description=self.text.description(term.iri)
            or (inspect.getdoc(vocabulary_class) or ""),
            evidence_status=(
                self.backing.prefixed(term.evidence_status)
                if term.evidence_status
                else ""
            ),
            claims=tuple(self.text.claim(claim, self.backing) for claim in term.claims),
            requirements=tuple(
                described(restriction, False) for restriction in term.restrictions
            ),
            definition=(
                "" if term.definition is None else described(term.definition, False)
            ),
        )

    def term_node(self, iri: str) -> OntologyClassNode:
        """
        :return: The node of a CRAM-AICOR or DUL class, or of a cited concept.
        """
        source = self.backing.source_of(iri)
        return OntologyClassNode(
            iri=iri,
            label=self.backing.prefixed(iri),
            category=(
                OntologyNodeCategory.INTRODUCED
                if source is TermSource.HYPOTHESIS
                else OntologyNodeCategory.LINKED
            ),
            source=source,
            description=self.text.description(iri),
            requirements=self.text.requirements_of(iri),
        )

    @staticmethod
    def is_a_edges(classes: list[type[DescriptionEntity]]) -> list[OntologyClassEdge]:
        """
        :return: Every class linked to its direct microworld superclasses and to the
            CRAM-AICOR and DUL classes it specialises.
        """
        return [
            OntologyClassEdge(
                source=vocabulary_class.term.iri,
                target=superior,
                kind=OntologyEdgeKind.IS_A,
            )
            for vocabulary_class in classes
            for superior in dict.fromkeys(
                [
                    *(
                        base.term.iri
                        for base in vocabulary_class.__bases__
                        if issubclass(base, DescriptionEntity) and "term" in vars(base)
                    ),
                    *vocabulary_class.term.linked_terms,
                ]
            )
        ]

    def hypothesis_edges(self) -> list[OntologyClassEdge]:
        """
        :return: The links the hypothesis states, from the class of one fragment to the
            class of the fragment or the concept it links to, counted.
        """
        counts: Counter[tuple[str, OntologyEdgeKind, str, str]] = Counter()
        for fragment in self.fragments:
            components = self.vocabulary.component_fields(type(fragment))
            for declared in fields(fragment):
                property_term = PropertyTerm.of_field(type(fragment), declared.name)
                if property_term is None or is_inferred(type(fragment), declared.name):
                    continue
                held: Any = attrgetter(declared.name)(fragment)
                for value in held if isinstance(held, list) else [held]:
                    kind, target = self.linked(value, declared.name in components)
                    if kind is not None:
                        counts[
                            (type(fragment).term.iri, kind, property_term.iri, target)
                        ] += 1
        return [
            OntologyClassEdge(
                source=source,
                target=target,
                kind=kind,
                property_iri=property_iri,
                count=count,
            )
            for (source, kind, property_iri, target), count in counts.items()
        ]

    @staticmethod
    def linked(
        value: Any, declares_component: bool
    ) -> tuple[Optional[OntologyEdgeKind], str]:
        """
        :return: How a value links its fragment's class to another class, and the IRI
            of that class, or no kind for a value that is no fragment or concept.
        """
        if isinstance(value, DescriptionEntity):
            return (
                (
                    OntologyEdgeKind.COMPONENT
                    if declares_component
                    else OntologyEdgeKind.RELATION
                ),
                type(value).term.iri,
            )
        if isinstance(value, ConceptReference):
            return OntologyEdgeKind.CITES, value.iri
        return None, ""

    def classification_edges(
        self, instances: dict[type[DescriptionEntity], list[DescriptionEntity]]
    ) -> list[OntologyClassEdge]:
        """
        :return: A link from every instantiated class to each defined class whose
            definition some of its instances meet, counting them, unless the class
            already specialises the defined class.
        """
        membership = ClosedWorldMembership(vocabulary=self.vocabulary)
        return [
            OntologyClassEdge(
                source=instantiated.term.iri,
                target=defined.term.iri,
                kind=OntologyEdgeKind.CLASSIFIED_AS,
                count=len(members),
            )
            for defined in self.vocabulary.defined_classes
            for instantiated, fragments in instances.items()
            if not issubclass(instantiated, defined)
            for members in [membership.members(defined.term.definition, fragments)]
            if members
        ]

    def disjoint_edges(self, iris: set[str]) -> list[OntologyClassEdge]:
        """
        :return: A link between every two classes of the graph the ontology declares
            disjoint.
        """
        return [
            OntologyClassEdge(
                source=first, target=second, kind=OntologyEdgeKind.DISJOINT
            )
            for pair in self.text.disjoint_pairs()
            if pair <= iris
            for first, second in [sorted(pair)]
        ]
