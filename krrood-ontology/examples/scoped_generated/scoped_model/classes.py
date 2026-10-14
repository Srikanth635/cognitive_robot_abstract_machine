"""Structural ontology classes; generated fields do not validate OWL constraints."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from typing import ClassVar as _ClassVar
from typing import Any as _Any

from krrood.entity_query_language.predicate import Symbol as _KrroodSymbol


@_dataclass(eq=False, kw_only=True)
class _OntologyEntity(_KrroodSymbol):
    """Private storage base for shared native relation associations."""

    before: set[_OntologyEntity] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "https://example.org/scoped#before",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    connected: set[_OntologyEntity] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "https://example.org/scoped#connected",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Alias(_OntologyEntity):
    """Structural representation of Alias."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/scoped#Alias"


@_dataclass(eq=False, kw_only=True)
class Entity(_OntologyEntity):
    """Structural representation of Entity."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/scoped#Entity"


@_dataclass(eq=False, kw_only=True)
class Carrier(_OntologyEntity):
    """Structural representation of Carrier."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/scoped#Carrier"
    link: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/scoped#link",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Choice(_OntologyEntity):
    """Structural representation of Choice."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/scoped#Choice"


@_dataclass(eq=False, kw_only=True)
class Either(_OntologyEntity):
    """Structural representation of Either."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/scoped#Either"


@_dataclass(eq=False, kw_only=True)
class FlagCarrier(_OntologyEntity):
    """Structural representation of FlagCarrier."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/scoped#FlagCarrier"
    flag: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/scoped#flag",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class InheritedFlagAlias(_OntologyEntity):
    """Structural representation of InheritedFlagAlias."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/scoped#InheritedFlagAlias"
    flag: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/scoped#flag",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Linked(_OntologyEntity):
    """Structural representation of Linked."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/scoped#Linked"
    link: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/scoped#link",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Marked(_OntologyEntity):
    """Structural representation of Marked."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/scoped#Marked"
    flag: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/scoped#flag",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class MarkedAlias(_OntologyEntity):
    """Structural representation of MarkedAlias."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/scoped#MarkedAlias"
    flag: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/scoped#flag",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Nested(_OntologyEntity):
    """Structural representation of Nested."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/scoped#Nested"
    link: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/scoped#link",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ReverseMarked(_OntologyEntity):
    """Structural representation of ReverseMarked."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/scoped#ReverseMarked"
    flag: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/scoped#flag",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Alpha(Entity):
    """Structural representation of Alpha."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/scoped#Alpha"


@_dataclass(eq=False, kw_only=True)
class Beta(Entity):
    """Structural representation of Beta."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/scoped#Beta"


@_dataclass(eq=False, kw_only=True)
class FlagChild(FlagCarrier):
    """Structural representation of FlagChild."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/scoped#FlagChild"


CLASS_BY_IRI = {
    "https://example.org/scoped#Alias": Alias,
    "https://example.org/scoped#Entity": Entity,
    "https://example.org/scoped#Carrier": Carrier,
    "https://example.org/scoped#Choice": Choice,
    "https://example.org/scoped#Either": Either,
    "https://example.org/scoped#FlagCarrier": FlagCarrier,
    "https://example.org/scoped#InheritedFlagAlias": InheritedFlagAlias,
    "https://example.org/scoped#Linked": Linked,
    "https://example.org/scoped#Marked": Marked,
    "https://example.org/scoped#MarkedAlias": MarkedAlias,
    "https://example.org/scoped#Nested": Nested,
    "https://example.org/scoped#ReverseMarked": ReverseMarked,
    "https://example.org/scoped#Alpha": Alpha,
    "https://example.org/scoped#Beta": Beta,
    "https://example.org/scoped#FlagChild": FlagChild,
}
PROPERTY_FIELD_NAMES = {
    "https://example.org/scoped#before": "before",
    "https://example.org/scoped#connected": "connected",
    "https://example.org/scoped#link": "link",
    "https://example.org/scoped#flag": "flag",
}
