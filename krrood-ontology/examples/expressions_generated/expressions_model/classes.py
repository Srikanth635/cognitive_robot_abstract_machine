"""Structural ontology classes; generated fields do not validate OWL constraints."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from typing import ClassVar as _ClassVar

from krrood.entity_query_language.predicate import Symbol as _KrroodSymbol


@_dataclass(eq=False, kw_only=True)
class Person(_KrroodSymbol):
    """Structural representation of Person."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/expressions#Person"
    name: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "https://example.org/expressions#name",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )
    works_for: set[Company] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "https://example.org/expressions#worksFor",
            "ontology_range": "https://example.org/expressions#Company",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Company(_KrroodSymbol):
    """Structural representation of Company."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/expressions#Company"
    has_employee: set[Person] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "https://example.org/expressions#hasEmployee",
            "ontology_range": "https://example.org/expressions#Person",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PersonOrCompany(_KrroodSymbol):
    """Structural representation of PersonOrCompany."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/expressions#PersonOrCompany"


@_dataclass(eq=False, kw_only=True)
class SelectedPerson(_KrroodSymbol):
    """Structural representation of SelectedPerson."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/expressions#SelectedPerson"


@_dataclass(eq=False, kw_only=True)
class WorkingOrSelected(_KrroodSymbol):
    """Structural representation of WorkingOrSelected."""

    __ontology_iri__: _ClassVar[str] = (
        "https://example.org/expressions#WorkingOrSelected"
    )


@_dataclass(eq=False, kw_only=True)
class AcmeWorker(Person):
    """Structural representation of AcmeWorker."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/expressions#AcmeWorker"


@_dataclass(eq=False, kw_only=True)
class Manager(Person):
    """Structural representation of Manager."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/expressions#Manager"


@_dataclass(eq=False, kw_only=True)
class NamedAlice(Person):
    """Structural representation of NamedAlice."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/expressions#NamedAlice"


@_dataclass(eq=False, kw_only=True)
class WorkingPerson(Person):
    """Structural representation of WorkingPerson."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/expressions#WorkingPerson"


@_dataclass(eq=False, kw_only=True)
class EmployerOfAlice(Company):
    """Structural representation of EmployerOfAlice."""

    __ontology_iri__: _ClassVar[str] = "https://example.org/expressions#EmployerOfAlice"


@_dataclass(eq=False, kw_only=True)
class EmployerOfManager(Company):
    """Structural representation of EmployerOfManager."""

    __ontology_iri__: _ClassVar[str] = (
        "https://example.org/expressions#EmployerOfManager"
    )


@_dataclass(eq=False, kw_only=True)
class EmployerOfWorker(Company):
    """Structural representation of EmployerOfWorker."""

    __ontology_iri__: _ClassVar[str] = (
        "https://example.org/expressions#EmployerOfWorker"
    )


CLASS_BY_IRI = {
    "https://example.org/expressions#Person": Person,
    "https://example.org/expressions#Company": Company,
    "https://example.org/expressions#PersonOrCompany": PersonOrCompany,
    "https://example.org/expressions#SelectedPerson": SelectedPerson,
    "https://example.org/expressions#WorkingOrSelected": WorkingOrSelected,
    "https://example.org/expressions#AcmeWorker": AcmeWorker,
    "https://example.org/expressions#Manager": Manager,
    "https://example.org/expressions#NamedAlice": NamedAlice,
    "https://example.org/expressions#WorkingPerson": WorkingPerson,
    "https://example.org/expressions#EmployerOfAlice": EmployerOfAlice,
    "https://example.org/expressions#EmployerOfManager": EmployerOfManager,
    "https://example.org/expressions#EmployerOfWorker": EmployerOfWorker,
}
PROPERTY_FIELD_NAMES = {
    "https://example.org/expressions#hasEmployee": "has_employee",
    "https://example.org/expressions#worksFor": "works_for",
    "https://example.org/expressions#name": "name",
}
