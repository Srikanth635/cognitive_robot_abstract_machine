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
        "https://example.org/validation#parts": FieldMapping(
            iri="https://example.org/validation#parts",
            name="parts",
            domain_iri="",
            range_iri="",
            kind="object",
            container="list",
        ),
        "https://example.org/validation#primary": FieldMapping(
            iri="https://example.org/validation#primary",
            name="primary",
            domain_iri="https://example.org/validation#Assembly",
            range_iri="https://example.org/validation#Entity",
            kind="object",
            container="list",
        ),
        "https://example.org/validation#flag": FieldMapping(
            iri="https://example.org/validation#flag",
            name="flag",
            domain_iri="",
            range_iri="",
            kind="datatype",
            container="list",
        ),
        "https://example.org/validation#tag": FieldMapping(
            iri="https://example.org/validation#tag",
            name="tag",
            domain_iri="",
            range_iri="",
            kind="datatype",
            container="list",
        ),
    },
    bindings=(
        FieldMapping(
            "https://example.org/validation#parts",
            "parts",
            "https://example.org/validation#Bound",
            "",
            "object",
            "list",
        ),
        FieldMapping(
            "https://example.org/validation#flag",
            "flag",
            "https://example.org/validation#Flag",
            "",
            "datatype",
            "list",
        ),
        FieldMapping(
            "https://example.org/validation#parts",
            "parts",
            "https://example.org/validation#UnionOwner",
            "",
            "object",
            "list",
        ),
        FieldMapping(
            "https://example.org/validation#parts",
            "parts",
            "https://example.org/validation#Assembly",
            "",
            "object",
            "list",
        ),
        FieldMapping(
            "https://example.org/validation#primary",
            "primary",
            "https://example.org/validation#Assembly",
            "https://example.org/validation#Entity",
            "object",
            "list",
        ),
        FieldMapping(
            "https://example.org/validation#tag",
            "tag",
            "https://example.org/validation#Assembly",
            "",
            "datatype",
            "list",
        ),
        FieldMapping(
            "https://example.org/validation#parts",
            "parts",
            "https://example.org/validation#Maximum",
            "",
            "object",
            "list",
        ),
        FieldMapping(
            "https://example.org/validation#parts",
            "parts",
            "https://example.org/validation#Minimum",
            "",
            "object",
            "list",
        ),
    ),
    binding_policy="local-usage",
    annotation_properties=frozenset(ANNOTATION_PROPERTY_IRIS),
)


def load_individuals(source: OntologyInput | Path | str) -> LoadedIndividuals:
    """Create a fresh IRI registry and populate its asserted individuals."""
    return IndividualLoader(MODEL).load(source)
