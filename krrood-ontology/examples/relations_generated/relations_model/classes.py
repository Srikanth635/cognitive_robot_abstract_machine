"""Structural ontology classes; generated fields do not validate OWL constraints."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from typing import ClassVar as _ClassVar

from krrood.entity_query_language.predicate import Symbol as _KrroodSymbol


@_dataclass(eq=False, kw_only=True)
class Organization(_KrroodSymbol):
    """Structural representation of Organization."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/relations#Organization"
    contains_part: set[Organization] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "https://example.org/relations#containsPart",
            "ontology_range": "https://example.org/relations#Organization",
            "property_kind": "object",
        },
    )
    part_of: set[Organization] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "https://example.org/relations#partOf",
            "ontology_range": "https://example.org/relations#Organization",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Person(_KrroodSymbol):
    """Structural representation of Person."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/relations#Person"
    colleague_of: set[Person] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "https://example.org/relations#colleagueOf",
            "ontology_range": "https://example.org/relations#Person",
            "property_kind": "object",
        },
    )
    employee_of: set[Company] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "https://example.org/relations#employeeOf",
            "ontology_range": "https://example.org/relations#Company",
            "property_kind": "object",
        },
    )
    heads_of: set[Company] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "https://example.org/relations#headsOf",
            "ontology_range": "https://example.org/relations#Company",
            "property_kind": "object",
        },
    )
    knows: set[Person] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "https://example.org/relations#knows",
            "ontology_range": "https://example.org/relations#Person",
            "property_kind": "object",
        },
    )
    name: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/relations#name",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )
    works_for: set[Company] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "https://example.org/relations#worksFor",
            "ontology_range": "https://example.org/relations#Company",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Company(Organization):
    """Structural representation of Company."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/relations#Company"
    has_employee: set[Person] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "https://example.org/relations#hasEmployee",
            "ontology_range": "https://example.org/relations#Person",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Manager(Person):
    """Structural representation of Manager."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/relations#Manager"


CLASS_BY_IRI = {
    "https://example.org/relations#Organization": Organization,
    "https://example.org/relations#Person": Person,
    "https://example.org/relations#Company": Company,
    "https://example.org/relations#Manager": Manager,
}
PROPERTY_FIELD_NAMES = {
    "https://example.org/relations#colleagueOf": "colleague_of",
    "https://example.org/relations#containsPart": "contains_part",
    "https://example.org/relations#employeeOf": "employee_of",
    "https://example.org/relations#hasEmployee": "has_employee",
    "https://example.org/relations#headsOf": "heads_of",
    "https://example.org/relations#knows": "knows",
    "https://example.org/relations#partOf": "part_of",
    "https://example.org/relations#worksFor": "works_for",
    "https://example.org/relations#name": "name",
}
