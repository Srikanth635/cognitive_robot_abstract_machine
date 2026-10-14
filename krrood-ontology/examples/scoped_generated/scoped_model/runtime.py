"""Load named ontology individuals through the shared Phase 4 runtime."""

from pathlib import Path

from krrood_ontology.config import OntologyInput
from krrood_ontology.individuals import IndividualLoader
from krrood_ontology.runtime import FieldMapping, GeneratedModel, LoadedIndividuals

from .classes import CLASS_BY_IRI
from .classes import _OntologyEntity
from .vocabulary import ANNOTATION_PROPERTY_IRIS

MODEL = GeneratedModel(
    classes=CLASS_BY_IRI,
    entity_base=_OntologyEntity,
    properties={
        "https://example.org/scoped#before": FieldMapping(
            iri="https://example.org/scoped#before",
            name="before",
            domain_iri="",
            range_iri="",
            kind="object",
            container="set",
        ),
        "https://example.org/scoped#connected": FieldMapping(
            iri="https://example.org/scoped#connected",
            name="connected",
            domain_iri="",
            range_iri="",
            kind="object",
            container="set",
        ),
        "https://example.org/scoped#link": FieldMapping(
            iri="https://example.org/scoped#link",
            name="link",
            domain_iri="",
            range_iri="",
            kind="object",
            container="list",
        ),
        "https://example.org/scoped#flag": FieldMapping(
            iri="https://example.org/scoped#flag",
            name="flag",
            domain_iri="",
            range_iri="",
            kind="datatype",
            container="list",
        ),
    },
    bindings=(
        FieldMapping(
            "https://example.org/scoped#link",
            "link",
            "https://example.org/scoped#Carrier",
            "",
            "object",
            "list",
        ),
        FieldMapping(
            "https://example.org/scoped#flag",
            "flag",
            "https://example.org/scoped#FlagCarrier",
            "",
            "datatype",
            "list",
        ),
        FieldMapping(
            "https://example.org/scoped#link",
            "link",
            "https://example.org/scoped#Linked",
            "",
            "object",
            "list",
        ),
        FieldMapping(
            "https://example.org/scoped#flag",
            "flag",
            "https://example.org/scoped#Marked",
            "",
            "datatype",
            "list",
        ),
        FieldMapping(
            "https://example.org/scoped#link",
            "link",
            "https://example.org/scoped#Nested",
            "",
            "object",
            "list",
        ),
        FieldMapping(
            "https://example.org/scoped#flag",
            "flag",
            "https://example.org/scoped#InheritedFlagAlias",
            "",
            "datatype",
            "list",
        ),
        FieldMapping(
            "https://example.org/scoped#flag",
            "flag",
            "https://example.org/scoped#MarkedAlias",
            "",
            "datatype",
            "list",
        ),
        FieldMapping(
            "https://example.org/scoped#flag",
            "flag",
            "https://example.org/scoped#ReverseMarked",
            "",
            "datatype",
            "list",
        ),
        FieldMapping(
            "https://example.org/scoped#before", "before", "", "", "object", "set"
        ),
        FieldMapping(
            "https://example.org/scoped#connected", "connected", "", "", "object", "set"
        ),
    ),
    binding_policy="local-usage",
    annotation_properties=frozenset(ANNOTATION_PROPERTY_IRIS),
)


def load_individuals(source: OntologyInput | Path | str) -> LoadedIndividuals:
    """Create a fresh IRI registry and populate its asserted individuals."""
    return IndividualLoader(MODEL).load(source)
