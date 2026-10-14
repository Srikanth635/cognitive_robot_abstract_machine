"""Translate a documented structural subset without claiming OWL entailment."""

from __future__ import annotations

import keyword
import typing
from dataclasses import dataclass, replace
from graphlib import CycleError, TopologicalSorter
from typing import NoReturn
from urllib.parse import unquote, urlsplit

from krrood.code_generation.naming import to_camel_case, to_snake_case
from krrood.entity_query_language.predicate import Symbol
from rdflib import OWL, RDF, RDFS, XSD, URIRef

from krrood_ontology.expressions import ExpressionTranslator
from krrood_ontology.inspection import OntologyReport
from krrood_ontology.model import (
    ClassDefinition,
    CompilationDiagnostic,
    ModelDefinition,
    OntologyTranslationError,
    PropertyDefinition,
    PropertyKind,
)
from krrood_ontology.ontology import OntologyDocument
from krrood_ontology.semantics import RelationSemantics


@dataclass
class OntologyTranslator:
    """Resolve declared classes, named parents, and explicit property signatures."""

    def translate(
        self,
        document: OntologyDocument,
        report: OntologyReport,
        structural_only: bool = False,
    ) -> ModelDefinition:
        """Return a generation plan or fail before producing partial models."""
        diagnostics = self._semantic_diagnostics(document, report, structural_only)
        if diagnostics and not structural_only:
            raise OntologyTranslationError(diagnostics)
        graph = document.graph
        names = {iri: self._name(iri, is_class=True) for iri in report.classes}
        if not names:
            self._fail("no_classes", "No explicitly declared named classes were found")
        self._check_class_names(names)
        for term in graph.subjects(RDFS.subClassOf):
            if isinstance(term, URIRef) and str(term) not in names:
                self._fail(
                    "undeclared_class",
                    "Subclass subjects must be explicitly declared named classes",
                    str(term),
                )
        declared_properties = set(
            report.object_properties
            + report.datatype_properties
            + report.annotation_properties
        )
        property_terms = (
            set(graph.subjects(RDFS.domain))
            | set(graph.subjects(RDFS.range))
            | set(graph.subjects(RDF.type, RDF.Property))
        )
        for term in sorted(property_terms, key=str):
            if str(term) not in declared_properties:
                self._fail(
                    "undeclared_property",
                    "Property signatures require an explicit object, datatype, or annotation property declaration",
                    str(term),
                )
        parents: dict[str, str | None] = {}
        for iri in names:
            declared = set(graph.objects(URIRef(iri), RDFS.subClassOf)) - {OWL.Thing}
            if len(declared) > 1:
                self._fail(
                    "multiple_parents", "Only one named superclass is supported", iri
                )
            parent = next(iter(declared), None)
            if parent is not None and (
                not isinstance(parent, URIRef) or str(parent) not in names
            ):
                self._fail(
                    "unsupported_parent",
                    "Superclass must be a declared named class",
                    iri,
                )
            parents[iri] = str(parent) if parent is not None else None
        hierarchy = {
            iri: (parent,) if parent else () for iri, parent in parents.items()
        }
        try:
            order = tuple(TopologicalSorter(hierarchy).static_order())
        except CycleError as error:
            raise OntologyTranslationError(
                (
                    CompilationDiagnostic(
                        "inheritance_cycle",
                        "Cyclic subclass declarations cannot become Python inheritance",
                    ),
                )
            ) from error
        both_kinds = set(report.object_properties) & set(report.datatype_properties)
        if both_kinds:
            self._fail(
                "property_kind_conflict",
                "Property is declared as both object and datatype property",
                sorted(both_kinds)[0],
            )
        properties = tuple(
            self._property(document, iri, kind, names)
            for kind, iris in (
                (PropertyKind.OBJECT, report.object_properties),
                (PropertyKind.DATATYPE, report.datatype_properties),
            )
            for iri in iris
        )
        classes: list[ClassDefinition] = []
        inherited: dict[str, dict[str, str]] = {}
        for iri in order:
            owned = tuple(
                sorted(
                    (prop for prop in properties if prop.domain_iri == iri),
                    key=lambda item: item.iri,
                )
            )
            field_names = dict(inherited[parents[iri]]) if parents[iri] else {}
            for prop in owned:
                if prop.name in field_names:
                    self._fail(
                        "field_collision",
                        f"Field {prop.name!r} also maps {field_names[prop.name]}",
                        prop.iri,
                    )
                field_names[prop.name] = prop.iri
            inherited[iri] = field_names
            classes.append(
                ClassDefinition(
                    iri,
                    names[iri],
                    names[parents[iri]] if parents[iri] else None,
                    owned,
                )
            )
        if report.individuals:
            diagnostics += (
                CompilationDiagnostic(
                    "individuals_deferred",
                    "Individual assertions are inventoried; use the generated load_individuals API to instantiate them",
                ),
            )
        model = ModelDefinition(tuple(classes), properties, diagnostics)
        if structural_only:
            return model
        model = RelationSemantics().translate(document, model)
        return replace(
            model, expressions=ExpressionTranslator().translate(document, model)
        )

    def _semantic_diagnostics(
        self,
        document: OntologyDocument,
        report: OntologyReport,
        structural_only: bool = False,
    ) -> tuple[CompilationDiagnostic, ...]:
        """Detect excluded OWL features and unsupported RDFS property axioms."""
        allowed = {
            OWL.Ontology,
            OWL.Class,
            OWL.ObjectProperty,
            OWL.DatatypeProperty,
            OWL.AnnotationProperty,
            OWL.NamedIndividual,
            OWL.versionInfo,
            OWL.versionIRI,
            OWL.priorVersion,
            OWL.backwardCompatibleWith,
            OWL.incompatibleWith,
            OWL.deprecated,
        }
        if not structural_only:
            allowed.update(
                (
                    OWL.inverseOf,
                    OWL.TransitiveProperty,
                    OWL.SymmetricProperty,
                )
            )
            if not any(document.graph.triples((None, OWL.imports, None))):
                allowed.update(
                    (
                        OWL.equivalentClass,
                        OWL.intersectionOf,
                        OWL.unionOf,
                        OWL.oneOf,
                        OWL.Restriction,
                        OWL.onProperty,
                        OWL.hasValue,
                        OWL.someValuesFrom,
                    )
                )
        unsupported = {URIRef(iri) for iri in report.owl_features} - allowed
        graph = document.graph
        unsupported.update(
            predicate
            for predicate in (RDFS.subPropertyOf,)
            if structural_only and any(graph.triples((None, predicate, None)))
        )
        if OWL.Nothing in set(graph.all_nodes()):
            unsupported.add(OWL.Nothing)
        return tuple(
            CompilationDiagnostic(
                "semantic_feature_deferred",
                "This feature is omitted from the selected profile; --structural-only records semantic omissions",
                str(term),
            )
            for term in sorted(unsupported, key=str)
        )

    def _property(
        self,
        document: OntologyDocument,
        iri: str,
        kind: PropertyKind,
        names: dict[str, str],
    ) -> PropertyDefinition:
        """Require an unambiguous domain and a supported named range."""
        graph = document.graph
        term = URIRef(iri)
        domains = set(graph.objects(term, RDFS.domain))
        ranges = set(graph.objects(term, RDFS.range))
        if len(domains) != 1 or len(ranges) != 1:
            self._fail(
                "property_signature",
                "Exactly one explicit domain and range are required",
                iri,
            )
        domain = next(iter(domains))
        range_term = next(iter(ranges))
        if not isinstance(domain, URIRef) or str(domain) not in names:
            self._fail("property_domain", "Domain must be a declared named class", iri)
        if not isinstance(range_term, URIRef):
            self._fail(
                "property_range", "Anonymous property ranges are unsupported", iri
            )
        if kind is PropertyKind.OBJECT:
            if str(range_term) not in names:
                self._fail(
                    "property_range", "Object range must be a declared named class", iri
                )
            range_type = names[str(range_term)]
        else:
            range_type = self._datatype_type(range_term)
        return PropertyDefinition(
            iri,
            self._name(iri, is_class=False),
            str(domain),
            names[str(domain)],
            str(range_term),
            range_type,
            kind,
        )

    def _datatype_type(self, range_term: URIRef) -> str:
        """Share the supported RDF-to-Python literal table across profiles."""
        datatypes = {
            XSD.string: "str",
            XSD.boolean: "bool",
            XSD.integer: "int",
            XSD.decimal: "_Decimal",
            XSD.double: "float",
            XSD.float: "float",
            XSD.date: "_date",
            XSD.dateTime: "_datetime",
            XSD.time: "_time",
        }
        if range_term not in datatypes:
            self._fail(
                "datatype_range",
                "Datatype has no supported Python mapping",
                str(range_term),
            )
        return datatypes[range_term]

    def _name(self, iri: str, is_class: bool) -> str:
        """Reuse KRROOD case conversion and reject unsafe identifiers."""
        parsed = urlsplit(iri)
        local = unquote(parsed.fragment or parsed.path.rstrip("/").rsplit("/", 1)[-1])
        if ":" in local:
            local = local.rsplit(":", 1)[-1]
        name = to_camel_case(to_snake_case(local)) if is_class else to_snake_case(local)
        if keyword.iskeyword(name) or (is_class and name in typing.__all__):
            name += "_"
        if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_"):
            self._fail(
                "invalid_identifier",
                f"Cannot use {name!r} as a public Python identifier",
                iri,
            )
        if not is_class and name in dir(Symbol):
            self._fail(
                "reserved_field", f"Field {name!r} conflicts with KRROOD Symbol", iri
            )
        return name

    def _check_class_names(self, names: dict[str, str]) -> None:
        """Reject distinct IRIs that would produce the same exported class."""
        seen: dict[str, str] = {}
        for iri, name in names.items():
            if name in seen:
                self._fail(
                    "class_collision", f"Class {name!r} also maps {seen[name]}", iri
                )
            if name in {
                "CLASS_BY_IRI",
                "PROPERTY_FIELD_NAMES",
                "PROPERTY_DESCRIPTORS",
                "CLASS_IRIS",
                "OBJECT_PROPERTY_IRIS",
                "DATATYPE_PROPERTY_IRIS",
                "INDIVIDUAL_IRIS",
                "ANNOTATION_PROPERTY_IRIS",
                "ONTOLOGY_IMPORT_IRIS",
                "SOURCE_SHA256",
                "MODEL",
                "SCHEMA_AXIOMS",
                "PROPERTY_BINDINGS",
                "CLASS_EXPRESSIONS",
                "model_queries",
                "load_individuals",
                "RELATION_AXIOMS",
                "UPDATE_POLICY",
                "RETRACTION_SUPPORTED",
            }:
                self._fail(
                    "reserved_class",
                    "Class name conflicts with a generated export",
                    iri,
                )
            seen[name] = iri

    @staticmethod
    def _fail(code: str, message: str, iri: str | None = None) -> NoReturn:
        raise OntologyTranslationError((CompilationDiagnostic(code, message, iri),))
