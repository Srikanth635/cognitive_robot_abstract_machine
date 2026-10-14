"""Structural ontology classes; generated fields do not validate OWL constraints."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from typing import ClassVar as _ClassVar
from typing import Any as _Any

from krrood.entity_query_language.predicate import Symbol as _KrroodSymbol


@_dataclass(eq=False, kw_only=True)
class Entity(_KrroodSymbol):
    """Structural representation of Entity."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/validation#Entity"


@_dataclass(eq=False, kw_only=True)
class Bound(_KrroodSymbol):
    """Structural representation of Bound."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/validation#Bound"
    parts: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/validation#parts",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Flag(_KrroodSymbol):
    """Structural representation of Flag."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/validation#Flag"
    flag: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/validation#flag",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class UnionOwner(_KrroodSymbol):
    """Structural representation of UnionOwner."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/validation#UnionOwner"
    parts: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/validation#parts",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Assembly(Entity):
    """Structural representation of Assembly."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/validation#Assembly"
    parts: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/validation#parts",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    primary: list[Entity] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/validation#primary",
            "ontology_range": "https://example.org/validation#Entity",
            "property_kind": "object",
        },
    )
    tag: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/validation#tag",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Maximum(Entity):
    """Structural representation of Maximum."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/validation#Maximum"
    parts: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/validation#parts",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Minimum(Entity):
    """Structural representation of Minimum."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/validation#Minimum"
    parts: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/validation#parts",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Other(Entity):
    """Structural representation of Other."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/validation#Other"


@_dataclass(eq=False, kw_only=True)
class Part(Entity):
    """Structural representation of Part."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/validation#Part"


@_dataclass(eq=False, kw_only=True)
class Child(Assembly):
    """Structural representation of Child."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/validation#Child"


@_dataclass(eq=False, kw_only=True)
class Conflict(Part, Other):
    """Structural representation of Conflict."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/validation#Conflict"


CLASS_BY_IRI = {
    "https://example.org/validation#Entity": Entity,
    "https://example.org/validation#Bound": Bound,
    "https://example.org/validation#Flag": Flag,
    "https://example.org/validation#UnionOwner": UnionOwner,
    "https://example.org/validation#Assembly": Assembly,
    "https://example.org/validation#Maximum": Maximum,
    "https://example.org/validation#Minimum": Minimum,
    "https://example.org/validation#Other": Other,
    "https://example.org/validation#Part": Part,
    "https://example.org/validation#Child": Child,
    "https://example.org/validation#Conflict": Conflict,
}
PROPERTY_FIELD_NAMES = {
    "https://example.org/validation#parts": "parts",
    "https://example.org/validation#primary": "primary",
    "https://example.org/validation#flag": "flag",
    "https://example.org/validation#tag": "tag",
}
