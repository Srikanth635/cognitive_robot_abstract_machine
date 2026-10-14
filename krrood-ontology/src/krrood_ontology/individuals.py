"""Validate an asserted A-Box and populate existing generated model objects."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

from rdflib import OWL, RDF, RDFS, XSD, BNode, Graph, Literal, URIRef

from krrood_ontology.config import OntologyInput
from krrood_ontology.identity import IdentityConstraints
from krrood_ontology.ontology import OntologyDocument
from krrood_ontology.runtime import (
    FieldMapping,
    GeneratedModel,
    IndividualLoadingError,
    LoadedIndividuals,
)


@dataclass(frozen=True)
class IndividualPlan:
    """Validated class selection and assertions for one named individual."""

    iri: str
    class_iri: str
    asserted_types: tuple[str, ...]


@dataclass(frozen=True)
class AssertionPlan:
    """A validated field value or reference ready for second-pass assignment."""

    subject: str
    mapping: FieldMapping
    value: Any


@dataclass
class IndividualLoader:
    """Load named individuals with chain-compatible explicit class types."""

    model: GeneratedModel

    def load(
        self, source: OntologyInput | Path | str, *, identity_policy: str = "reject"
    ) -> LoadedIndividuals:
        """Preflight all assertions, create objects, then populate their fields."""
        source = (
            source if isinstance(source, OntologyInput) else OntologyInput(Path(source))
        )
        document = OntologyDocument.load(source)
        content = source.path.read_bytes()
        if hashlib.sha256(content).hexdigest() != document.sha256:
            raise ValueError("Source changed during individual loading")
        return self._materialize(
            document.graph,
            identity_policy,
            document.sha256,
            content,
            document.format.value,
        )

    def load_graph(
        self, graph: Graph, *, identity_policy: str = "reject"
    ) -> LoadedIndividuals:
        """Materialize a defensive graph copy without reparsing its RDF terms.

        Uses the current native graph. Call inside an owned isolation boundary
        when constructor or descriptor failures must discard partial objects.
        """
        captured = Graph()
        for prefix, namespace in graph.namespaces():
            captured.bind(prefix, namespace)
        captured += graph
        content = captured.serialize(format="nt", encoding="utf-8")
        return self._materialize(
            captured,
            identity_policy,
            hashlib.sha256(content).hexdigest(),
            content,
            "nt",
        )

    def _materialize(
        self, graph, identity_policy, source_sha256, content, source_format
    ):
        planner = IndividualPlanner(self.model, identity_policy=identity_policy)
        individuals, assertions, annotations = planner.plan_graph(graph)
        objects = {
            plan.iri: self.model.classes[plan.class_iri]() for plan in individuals
        }
        for assertion in assertions:
            target = objects[assertion.subject]
            value = (
                objects[assertion.value]
                if assertion.mapping.kind == "object"
                else assertion.value
            )
            collection = getattr(target, assertion.mapping.name)
            if assertion.mapping.container == "set":
                collection.add(value)
            elif not any(
                type(value) is type(existing) and value == existing
                for existing in collection
            ):
                collection.append(value)
        return LoadedIndividuals(
            objects,
            {plan.iri: plan.asserted_types for plan in individuals},
            annotations,
            source_sha256,
            graph,
            identity_policy=identity_policy,
            distinct_groups=planner.identities_graph(graph).groups,
            source_bytes=content,
            source_format=source_format,
        )


@dataclass
class IndividualPlanner:
    """Validate an A-Box without creating symbols or mutating their graph."""

    model: GeneratedModel
    identity_policy: str = "reject"
    schema_types: frozenset[URIRef] = field(
        default_factory=lambda: frozenset(
            (
                OWL.Class,
                RDFS.Class,
                OWL.Ontology,
                OWL.ObjectProperty,
                OWL.DatatypeProperty,
                OWL.AnnotationProperty,
                OWL.TransitiveProperty,
                OWL.SymmetricProperty,
                OWL.Restriction,
                RDF.Property,
                RDF.List,
                OWL.FunctionalProperty,
                OWL.InverseFunctionalProperty,
                OWL.ReflexiveProperty,
                OWL.IrreflexiveProperty,
                OWL.AsymmetricProperty,
                OWL.AllDifferent,
                OWL.AllDisjointClasses,
                OWL.AllDisjointProperties,
                OWL.Axiom,
                OWL.NegativePropertyAssertion,
            )
        )
    )

    def plan(self, document: OntologyDocument) -> tuple[
        tuple[IndividualPlan, ...],
        tuple[AssertionPlan, ...],
        dict[str, dict[str, tuple[Any, ...]]],
    ]:
        """Resolve explicit types, references, signatures and literal conversions."""
        return self.plan_graph(document.graph)

    def plan_graph(self, graph):
        """Preflight an existing RDFLib graph without serialization or object construction."""
        if self.identity_policy not in {"reject", "distinct"}:
            raise ValueError("identity_policy must be reject or distinct")
        forbidden = (OWL.sameAs, OWL.imports) + (
            (OWL.differentFrom,) if self.identity_policy == "reject" else ()
        )
        if self.identity_policy == "reject" and any(
            graph.subjects(RDF.type, OWL.AllDifferent)
        ):
            self.fail(
                "identity_or_import_unsupported",
                "AllDifferent requires identity_policy=distinct",
            )
        for predicate in forbidden:
            if any(graph.triples((None, predicate, None))):
                self.fail(
                    "identity_or_import_unsupported",
                    "Identity axioms and ontology imports are unsupported by this loader",
                    str(predicate),
                )
        candidates = {
            subject
            for subject, kind in graph.subject_objects(RDF.type)
            if kind not in self.schema_types
        }
        candidates.update(
            subject
            for subject, predicate, _ in graph
            if str(predicate) in self.model.properties
        )
        plans = {}
        for subject in sorted(candidates, key=str):
            if not isinstance(subject, URIRef):
                self.fail(
                    "anonymous_individual",
                    "Only named individuals are supported",
                    str(subject),
                )
            if (
                str(subject) in self.model.classes
                or str(subject) in self.model.properties
            ):
                self.fail(
                    "punning_unsupported",
                    "Schema terms cannot also be individuals in this profile",
                    str(subject),
                )
            types = tuple(
                sorted(
                    str(kind)
                    for kind in graph.objects(subject, RDF.type)
                    if kind != OWL.NamedIndividual
                )
            )
            if not types:
                self.fail(
                    "missing_type",
                    "Individual requires an explicit generated class type",
                    str(subject),
                )
            if any(kind not in self.model.classes for kind in types):
                self.fail(
                    "unknown_type",
                    "Individual type is absent from the generated model",
                    str(subject),
                )
            choices = [
                kind
                for kind in types
                if all(
                    issubclass(self.model.classes[kind], self.model.classes[parent])
                    for parent in types
                )
            ]
            if len(choices) != 1:
                self.fail(
                    "unrelated_types",
                    "Multiple types must form a generated inheritance chain",
                    str(subject),
                )
            plans[str(subject)] = IndividualPlan(str(subject), choices[0], types)
        if self.identity_policy == "distinct":
            IdentityConstraints.from_graph(graph, self.fail, registered=plans)
        assertions = []
        annotations = {}
        allowed_annotations = self.model.annotation_properties | {
            str(RDFS.label),
            str(RDFS.comment),
        }
        for iri, plan in sorted(plans.items()):
            for predicate, value in sorted(
                graph.predicate_objects(URIRef(iri)),
                key=lambda pair: (str(pair[0]), pair[1].n3()),
            ):
                if predicate == RDF.type or (
                    predicate == OWL.differentFrom
                    and self.identity_policy == "distinct"
                ):
                    continue
                if str(predicate) in allowed_annotations:
                    if isinstance(value, BNode):
                        self.fail(
                            "anonymous_annotation",
                            "Blank-node annotations are unsupported",
                            iri,
                        )
                    values = annotations.setdefault(iri, {}).setdefault(
                        str(predicate), []
                    )
                    values.append(value)
                    continue
                mapping = self.model.mapping_for(str(predicate), plan.class_iri)
                if mapping is None:
                    self.fail(
                        "unknown_property",
                        "Individual assertion has no generated field binding on this class",
                        str(predicate),
                    )
                declared_domain = self.model.properties[mapping.iri].domain_iri
                if declared_domain and not issubclass(
                    self.model.classes[plan.class_iri],
                    self.model.classes[declared_domain],
                ):
                    self.fail(
                        "domain_mismatch",
                        "Assertion subject does not have the declared property domain",
                        iri,
                    )
                if mapping.kind == "object":
                    if not isinstance(value, URIRef) or str(value) not in plans:
                        self.fail(
                            "unresolved_reference",
                            "Object target must be a named, explicitly typed individual in this load",
                            str(value),
                        )
                    if mapping.range_iri and not issubclass(
                        self.model.classes[plans[str(value)].class_iri],
                        self.model.classes[mapping.range_iri],
                    ):
                        self.fail(
                            "range_mismatch",
                            "Object target does not have the declared property range",
                            str(value),
                        )
                    converted = str(value)
                else:
                    converted = self.literal(value, mapping, iri)
                assertions.append(AssertionPlan(iri, mapping, converted))
        frozen_annotations = {
            iri: {predicate: tuple(values) for predicate, values in predicates.items()}
            for iri, predicates in annotations.items()
        }
        return tuple(plans.values()), tuple(assertions), frozen_annotations

    def identities(self, document, *, registered=None):
        return self.identities_graph(document.graph, registered=registered)

    def identities_graph(self, graph, *, registered=None):
        return (
            IdentityConstraints.from_graph(graph, self.fail, registered=registered)
            if self.identity_policy == "distinct"
            else IdentityConstraints()
        )

    def literal(self, value: Any, mapping: FieldMapping, subject: str) -> Any:
        """Use RDFLib conversion with exact declared datatype matching."""
        if not isinstance(value, Literal):
            self.fail(
                "literal_required", "Datatype properties require RDF literals", subject
            )
        datatype = value.datatype or XSD.string
        if mapping.range_iri and (value.language or str(datatype) != mapping.range_iri):
            self.fail(
                "datatype_mismatch",
                "Literal datatype must match the declared range; language-tagged field values are unsupported",
                subject,
            )
        converted = value.toPython()
        if value.ill_typed or isinstance(converted, Literal):
            self.fail(
                "invalid_literal",
                "RDFLib could not convert the literal lexical value",
                subject,
            )
        return converted

    @staticmethod
    def fail(code: str, message: str, iri: str | None = None) -> NoReturn:
        raise IndividualLoadingError(code, message, iri)
