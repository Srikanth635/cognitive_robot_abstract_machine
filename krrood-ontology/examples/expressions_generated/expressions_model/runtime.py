"""Load named ontology individuals through the shared Phase 4 runtime."""

from pathlib import Path

from krrood_ontology.config import OntologyInput
from krrood_ontology.individuals import IndividualLoader
from krrood_ontology.runtime import FieldMapping, GeneratedModel, LoadedIndividuals

from .classes import CLASS_BY_IRI
from .vocabulary import ANNOTATION_PROPERTY_IRIS

MODEL = GeneratedModel(
    classes=CLASS_BY_IRI,
    properties={
        "https://example.org/expressions#hasEmployee": FieldMapping(
            iri="https://example.org/expressions#hasEmployee",
            name="has_employee",
            domain_iri="https://example.org/expressions#Company",
            range_iri="https://example.org/expressions#Person",
            kind="object",
            container="set",
        ),
        "https://example.org/expressions#worksFor": FieldMapping(
            iri="https://example.org/expressions#worksFor",
            name="works_for",
            domain_iri="https://example.org/expressions#Person",
            range_iri="https://example.org/expressions#Company",
            kind="object",
            container="set",
        ),
        "https://example.org/expressions#name": FieldMapping(
            iri="https://example.org/expressions#name",
            name="name",
            domain_iri="https://example.org/expressions#Person",
            range_iri="http://www.w3.org/2001/XMLSchema#string",
            kind="datatype",
            container="list",
        ),
    },
    annotation_properties=frozenset(ANNOTATION_PROPERTY_IRIS),
)


def load_individuals(source: OntologyInput | Path | str) -> LoadedIndividuals:
    """Create a fresh IRI registry and populate its asserted individuals."""
    return IndividualLoader(MODEL).load(source)
