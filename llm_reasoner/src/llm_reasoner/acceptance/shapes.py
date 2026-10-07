"""
The structural rules the SHACL shapes beside the ontology state: relationships between
several fragments, such as the transitions of a structure connecting all its phases,
which an OWL restriction on one class cannot express.

A hypothesis is written as RDF together with the ontology modules and validated against
the shapes closed-world. Each result names the shape it comes from, the fragment at
fault and the fragment it concerns.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, fields
from functools import cache, cached_property
from operator import attrgetter

import pyshacl
from rdflib import OWL, RDF, RDFS, SH, Graph, URIRef
from rdflib.term import Node
from typing_extensions import Any, Iterable, Optional, Sequence

from llm_reasoner.acceptance.derived import DerivedRule
from llm_reasoner.acceptance.rules import AcceptanceRule, Severity, Violation
from llm_reasoner.acceptance.vocabulary import Vocabulary
from llm_reasoner.admission.rdf import GraphWriter
from llm_reasoner.ontologies import (
    PREDICTIVE_MANIPULATION_SHAPES_NAMESPACE_ROOT,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.ontologies.linked import OntologyFiles
from llm_reasoner.ontologies.terms import DescriptionEntity, PropertyTerm

CHECKED_NAMESPACE = PREDICTIVE_MANIPULATION_SHAPES_NAMESPACE_ROOT + "checked#"
"""
The namespace the fragments being checked are named in while they are validated.
"""

HIERARCHY_PROPERTIES = (RDFS.subClassOf, RDFS.subPropertyOf)
"""
The ontology statements saying which classes and properties specialise which, so that a
shape about a class covers its subclasses.
"""

SEVERITY_OF_RESULT = {
    SH.Violation: Severity.ERROR,
    SH.Warning: Severity.WARNING,
    SH.Info: Severity.WARNING,
}
"""
How a SHACL result severity bears on admitting the hypothesis.
"""

# %% shapes


@dataclass(frozen=True)
class StructureShape:
    """
    One structural rule: a shape about the fragments of one class.
    """

    iri: URIRef
    """
    The IRI of the shape.
    """

    target: type[DescriptionEntity]
    """
    The vocabulary class whose fragments the shape constrains.
    """

    description: str
    """
    What the rule requires, in plain words.
    """

    @property
    def name(self) -> str:
        """
        The shape's local name, which names the rule.
        """
        return self.iri.split("#")[-1]


@dataclass
class StructureShapes:
    """
    The SHACL shapes of every module that states structural rules, with the ontology
    modules they are checked against.
    """

    graph: Graph
    """
    Every shape.
    """

    ontology: Graph
    """
    What the shapes read of the ontology modules: which named classes and properties
    specialise which, and what is stated about the individuals the vocabulary
    enumerates, such as the dimension each unit measures.
    """

    vocabulary: Vocabulary = field(default_factory=Vocabulary)
    """
    The generated vocabulary the shapes' target classes are looked up in.
    """

    @classmethod
    @cache
    def shipped(cls) -> StructureShapes:
        """
        :return: The shapes shipped beside the ontology modules.
        """
        graph = Graph()
        for module_file in PredictiveManipulationOntologyFile:
            if module_file.shapes_path.exists():
                graph.parse(module_file.shapes_path, format="turtle")
        modules = OntologyFiles(
            paths=[
                module_file.path for module_file in PredictiveManipulationOntologyFile
            ]
        ).graph()
        ontology = Graph()
        for statement in HIERARCHY_PROPERTIES:
            for subject, _, superior in modules.triples((None, statement, None)):
                if isinstance(superior, URIRef):
                    ontology.add((subject, statement, superior))
        for individual in modules.subjects(RDF.type, OWL.NamedIndividual):
            for _, predicate, value in modules.triples((individual, None, None)):
                if isinstance(value, URIRef):
                    ontology.add((individual, predicate, value))
        return cls(graph=graph, ontology=ontology)

    @cached_property
    def shapes(self) -> list[StructureShape]:
        """
        Every shape, in the order of its IRI.
        """
        return [
            StructureShape(
                iri=shape,
                target=self.vocabulary.definition_named(
                    str(self.graph.value(shape, SH.targetClass))
                ),
                description=str(self.graph.value(shape, RDFS.comment)),
            )
            for shape in sorted(self.graph.subjects(RDF.type, SH.NodeShape))
        ]

    def named(self, iri: URIRef) -> StructureShape:
        """
        :return: The shape of the given IRI.
        """
        (shape,) = [shape for shape in self.shapes if shape.iri == iri]
        return shape

    def applicable_to(
        self, classes: Iterable[type[DescriptionEntity]]
    ) -> list[StructureShape]:
        """
        :return: The shapes constraining fragments of any of the given classes.
        """
        classes = list(classes)
        return [
            shape
            for shape in self.shapes
            if any(issubclass(cls, shape.target) for cls in classes)
        ]


# %% checking


@dataclass(frozen=True, kw_only=True)
class ShapeViolation(Violation):
    """
    A fragment failing the structural rule of one shape.
    """

    shape: URIRef
    """
    The shape stating the rule.
    """


@dataclass
class ShapesHold(AcceptanceRule):
    """
    Checks the fragments against the structural rules the shapes state.
    """

    shapes: StructureShapes = field(default_factory=StructureShapes.shipped)
    """
    The shapes the fragments are validated against.
    """

    checked_shapes: Optional[frozenset[URIRef]] = None
    """
    The shapes whose violations are reported, or ``None`` for every shape.
    """

    @property
    def name(self) -> DerivedRule:
        return DerivedRule.SHAPES_HOLD

    @staticmethod
    def reachable(fragments: Sequence[DescriptionEntity]) -> list[DescriptionEntity]:
        """
        :return: The given fragments and every fragment their ontology properties
            reach, each once, the given ones first.
        """
        seen: dict[int, DescriptionEntity] = {}
        queue = deque(fragments)
        while queue:
            fragment = queue.popleft()
            if id(fragment) in seen:
                continue
            seen[id(fragment)] = fragment
            for declared in fields(fragment):
                if PropertyTerm.of_field(type(fragment), declared.name) is None:
                    continue
                value: Any = attrgetter(declared.name)(fragment)
                queue.extend(
                    held
                    for held in (value if isinstance(value, list) else [value])
                    if isinstance(held, DescriptionEntity)
                )
        return list(seen.values())

    def reports(self, violation: ShapeViolation, checked: set[int]) -> bool:
        """
        :return: Whether a violation concerns one of the fragments checked and comes
            from a shape whose violations are reported.
        """
        return id(violation.fragment) in checked and (
            self.checked_shapes is None or violation.shape in self.checked_shapes
        )

    def violations(self, fragments: Sequence[DescriptionEntity]) -> list[Violation]:
        written = self.reachable(fragments)
        names = {
            id(fragment): URIRef(f"{CHECKED_NAMESPACE}fragment_{position}")
            for position, fragment in enumerate(written)
        }
        fragment_named = {names[id(fragment)]: fragment for fragment in written}
        checked = {id(fragment) for fragment in fragments}
        _, results, _ = pyshacl.validate(
            GraphWriter(namespace=CHECKED_NAMESPACE).write_fragments(written, names),
            shacl_graph=self.shapes.graph,
            ont_graph=self.shapes.ontology,
            advanced=True,
            allow_warnings=True,
        )
        violations = [
            self.violation(results, result, fragment_named)
            for result in results.subjects(RDF.type, SH.ValidationResult)
        ]
        position = {identity: index for index, identity in enumerate(names)}
        return sorted(
            (violation for violation in violations if self.reports(violation, checked)),
            key=lambda violation: (
                violation.rule,
                position[id(violation.fragment)],
                [position[id(involved)] for involved in violation.involved],
            ),
        )

    @staticmethod
    def violation(
        results: Graph, result: Node, fragment_named: dict[Node, DescriptionEntity]
    ) -> ShapeViolation:
        """
        :return: The violation one validation result reports.
        """
        shape = URIRef(results.value(result, SH.sourceShape))
        focus = results.value(result, SH.focusNode)
        value: Optional[Node] = results.value(result, SH.value)
        return ShapeViolation(
            rule=shape.split("#")[-1],
            fragment=fragment_named[focus],
            reason=str(results.value(result, SH.resultMessage)),
            severity=SEVERITY_OF_RESULT[results.value(result, SH.resultSeverity)],
            involved=(
                (fragment_named[value],)
                if value in fragment_named and value != focus
                else ()
            ),
            shape=shape,
        )
