"""Structural ontology classes; generated fields do not validate OWL constraints."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from decimal import Decimal as _Decimal
from typing import ClassVar as _ClassVar
from typing import Any as _Any

from krrood.entity_query_language.predicate import Symbol as _KrroodSymbol


@_dataclass(eq=False, kw_only=True)
class _OntologyEntity(_KrroodSymbol):
    """Private storage base for shared native relation associations."""

    links: set[_OntologyEntity] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "https://example.org/roundtrip#links",
            "ontology_range": "https://example.org/roundtrip#Entity",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Entity(_OntologyEntity):
    """Structural representation of Entity."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/roundtrip#Entity"
    amount: list[_Decimal] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/roundtrip#amount",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#decimal",
            "property_kind": "datatype",
        },
    )
    loose: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/roundtrip#loose",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Pick(_OntologyEntity):
    """Structural representation of Pick."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/roundtrip#Pick"


@_dataclass(eq=False, kw_only=True)
class Special(Entity):
    """Structural representation of Special."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/roundtrip#Special"


CLASS_BY_IRI = {
    "https://example.org/roundtrip#Entity": Entity,
    "https://example.org/roundtrip#Pick": Pick,
    "https://example.org/roundtrip#Special": Special,
}
PROPERTY_FIELD_NAMES = {
    "https://example.org/roundtrip#links": "links",
    "https://example.org/roundtrip#amount": "amount",
    "https://example.org/roundtrip#loose": "loose",
}
