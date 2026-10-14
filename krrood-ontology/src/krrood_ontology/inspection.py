"""Inventory explicitly declared ontology vocabulary and OWL features."""

from __future__ import annotations

from dataclasses import dataclass

from rdflib import OWL, RDF, RDFS, URIRef

from krrood_ontology.ontology import OntologyDocument


@dataclass(frozen=True)
class OntologyReport:
    """A sorted inventory of declarations, with no inference claims."""

    source_path: str
    source_sha256: str
    rdf_format: str
    triple_count: int
    classes: tuple[str, ...]
    object_properties: tuple[str, ...]
    datatype_properties: tuple[str, ...]
    annotation_properties: tuple[str, ...]
    individuals: tuple[str, ...]
    imports: tuple[str, ...]
    owl_features: tuple[str, ...]


@dataclass
class OntologyInspector:
    """Describe the asserted graph while preserving full IRIs."""

    def inspect(self, document: OntologyDocument) -> OntologyReport:
        """List named declarations and the OWL terms present in the file."""
        graph = document.graph

        def declared(*types: URIRef) -> set[URIRef]:
            return {
                subject
                for declared_type in types
                for subject in graph.subjects(RDF.type, declared_type)
                if isinstance(subject, URIRef)
            }

        def iris(terms: set[URIRef]) -> tuple[str, ...]:
            return tuple(sorted(map(str, terms)))

        classes = declared(OWL.Class, RDFS.Class)
        individuals = declared(OWL.NamedIndividual, *classes)
        imports = {
            value
            for value in graph.objects(None, OWL.imports)
            if isinstance(value, URIRef)
        }
        features = {
            term
            for _, predicate, value in graph
            for term in (predicate, value if predicate == RDF.type else None)
            if isinstance(term, URIRef) and str(term).startswith(str(OWL))
        }
        return OntologyReport(
            source_path=str(document.source.path),
            source_sha256=document.sha256,
            rdf_format=document.format.value,
            triple_count=len(graph),
            classes=iris(classes),
            object_properties=iris(declared(OWL.ObjectProperty)),
            datatype_properties=iris(declared(OWL.DatatypeProperty)),
            annotation_properties=iris(declared(OWL.AnnotationProperty)),
            individuals=iris(individuals),
            imports=iris(imports),
            owl_features=iris(features),
        )
