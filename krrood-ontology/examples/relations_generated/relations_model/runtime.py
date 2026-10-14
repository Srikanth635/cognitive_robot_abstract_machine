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
        "https://example.org/relations#colleagueOf": FieldMapping(
            iri="https://example.org/relations#colleagueOf",
            name="colleague_of",
            domain_iri="https://example.org/relations#Person",
            range_iri="https://example.org/relations#Person",
            kind="object",
            container="set",
        ),
        "https://example.org/relations#containsPart": FieldMapping(
            iri="https://example.org/relations#containsPart",
            name="contains_part",
            domain_iri="https://example.org/relations#Organization",
            range_iri="https://example.org/relations#Organization",
            kind="object",
            container="set",
        ),
        "https://example.org/relations#employeeOf": FieldMapping(
            iri="https://example.org/relations#employeeOf",
            name="employee_of",
            domain_iri="https://example.org/relations#Person",
            range_iri="https://example.org/relations#Company",
            kind="object",
            container="set",
        ),
        "https://example.org/relations#hasEmployee": FieldMapping(
            iri="https://example.org/relations#hasEmployee",
            name="has_employee",
            domain_iri="https://example.org/relations#Company",
            range_iri="https://example.org/relations#Person",
            kind="object",
            container="set",
        ),
        "https://example.org/relations#headsOf": FieldMapping(
            iri="https://example.org/relations#headsOf",
            name="heads_of",
            domain_iri="https://example.org/relations#Person",
            range_iri="https://example.org/relations#Company",
            kind="object",
            container="set",
        ),
        "https://example.org/relations#knows": FieldMapping(
            iri="https://example.org/relations#knows",
            name="knows",
            domain_iri="https://example.org/relations#Person",
            range_iri="https://example.org/relations#Person",
            kind="object",
            container="set",
        ),
        "https://example.org/relations#partOf": FieldMapping(
            iri="https://example.org/relations#partOf",
            name="part_of",
            domain_iri="https://example.org/relations#Organization",
            range_iri="https://example.org/relations#Organization",
            kind="object",
            container="set",
        ),
        "https://example.org/relations#worksFor": FieldMapping(
            iri="https://example.org/relations#worksFor",
            name="works_for",
            domain_iri="https://example.org/relations#Person",
            range_iri="https://example.org/relations#Company",
            kind="object",
            container="set",
        ),
        "https://example.org/relations#name": FieldMapping(
            iri="https://example.org/relations#name",
            name="name",
            domain_iri="https://example.org/relations#Person",
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
