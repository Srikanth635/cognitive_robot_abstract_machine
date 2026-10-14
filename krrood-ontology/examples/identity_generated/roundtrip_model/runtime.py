"""Load named ontology individuals through the shared Phase 4 runtime."""

from pathlib import Path

from krrood_ontology.config import OntologyInput
from krrood_ontology.individuals import IndividualLoader
from krrood_ontology.export import IndividualExporter
from krrood_ontology.runtime import FieldMapping, GeneratedModel, LoadedIndividuals

from .classes import CLASS_BY_IRI
from .classes import _OntologyEntity
from .vocabulary import ANNOTATION_PROPERTY_IRIS

MODEL = GeneratedModel(
    classes=CLASS_BY_IRI,
    entity_base=_OntologyEntity,
    properties={
        "https://example.org/roundtrip#links": FieldMapping(
            iri="https://example.org/roundtrip#links",
            name="links",
            domain_iri="https://example.org/roundtrip#Entity",
            range_iri="https://example.org/roundtrip#Entity",
            kind="object",
            container="set",
        ),
        "https://example.org/roundtrip#amount": FieldMapping(
            iri="https://example.org/roundtrip#amount",
            name="amount",
            domain_iri="https://example.org/roundtrip#Entity",
            range_iri="http://www.w3.org/2001/XMLSchema#decimal",
            kind="datatype",
            container="list",
        ),
        "https://example.org/roundtrip#loose": FieldMapping(
            iri="https://example.org/roundtrip#loose",
            name="loose",
            domain_iri="https://example.org/roundtrip#Entity",
            range_iri="",
            kind="datatype",
            container="list",
        ),
    },
    bindings=(
        FieldMapping(
            "https://example.org/roundtrip#amount",
            "amount",
            "https://example.org/roundtrip#Entity",
            "http://www.w3.org/2001/XMLSchema#decimal",
            "datatype",
            "list",
        ),
        FieldMapping(
            "https://example.org/roundtrip#loose",
            "loose",
            "https://example.org/roundtrip#Entity",
            "",
            "datatype",
            "list",
        ),
        FieldMapping(
            "https://example.org/roundtrip#links",
            "links",
            "",
            "https://example.org/roundtrip#Entity",
            "object",
            "set",
        ),
    ),
    binding_policy="local-usage",
    annotation_properties=frozenset(ANNOTATION_PROPERTY_IRIS),
)


def load_individuals(
    source: OntologyInput | Path | str, *, identity_policy: str = "distinct"
) -> LoadedIndividuals:
    """Create a fresh IRI registry and populate its asserted individuals."""
    return IndividualLoader(MODEL).load(source, identity_policy=identity_policy)


model_exporter = IndividualExporter(MODEL)
