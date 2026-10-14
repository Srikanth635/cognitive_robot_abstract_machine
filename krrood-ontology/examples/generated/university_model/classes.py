"""Structural ontology classes; generated fields do not validate OWL constraints."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from typing import ClassVar as _ClassVar

from krrood.entity_query_language.predicate import Symbol as _KrroodSymbol


@_dataclass(eq=False, kw_only=True)
class Course(_KrroodSymbol):
    """Structural representation of Course."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/university#Course"


@_dataclass(eq=False, kw_only=True)
class Person(_KrroodSymbol):
    """Structural representation of Person."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/university#Person"
    name: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/university#name",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Student(Person):
    """Structural representation of Student."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/university#Student"
    takes_course: list[Course] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/university#takesCourse",
            "ontology_range": "https://example.org/university#Course",
            "property_kind": "object",
        },
    )


CLASS_BY_IRI = {
    "https://example.org/university#Course": Course,
    "https://example.org/university#Person": Person,
    "https://example.org/university#Student": Student,
}
PROPERTY_FIELD_NAMES = {
    "https://example.org/university#takesCourse": "takes_course",
    "https://example.org/university#name": "name",
}
