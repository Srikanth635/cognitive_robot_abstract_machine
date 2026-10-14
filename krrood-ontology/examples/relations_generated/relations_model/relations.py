"""Object relation tracking delegated to KRROOD property descriptors."""

from __future__ import annotations

from typing import ClassVar as _ClassVar

from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyDescriptor as _PropertyDescriptor,
)
from krrood.ontomatic.property_descriptor.mixins import (
    HasInverseProperty as _HasInverseProperty,
)
from krrood.ontomatic.property_descriptor.mixins import (
    TransitiveProperty as _TransitiveProperty,
)

from . import classes as _classes


class _Property5(_PropertyDescriptor, _HasInverseProperty):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/relations#knows"
    _cache_instances_: _ClassVar[bool] = False

    @classmethod
    def get_inverse(cls) -> type[_PropertyDescriptor]:
        return _Property5


class _Property1(_PropertyDescriptor, _HasInverseProperty, _TransitiveProperty):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/relations#containsPart"
    _cache_instances_: _ClassVar[bool] = False

    @classmethod
    def get_inverse(cls) -> type[_PropertyDescriptor]:
        return _Property6


class _Property7(_PropertyDescriptor, _HasInverseProperty):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/relations#worksFor"
    _cache_instances_: _ClassVar[bool] = False

    @classmethod
    def get_inverse(cls) -> type[_PropertyDescriptor]:
        return _Property3


class _Property3(_PropertyDescriptor, _HasInverseProperty):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/relations#hasEmployee"
    _cache_instances_: _ClassVar[bool] = False

    @classmethod
    def get_inverse(cls) -> type[_PropertyDescriptor]:
        return _Property7


class _Property6(_PropertyDescriptor, _HasInverseProperty):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/relations#partOf"
    _cache_instances_: _ClassVar[bool] = False

    @classmethod
    def get_inverse(cls) -> type[_PropertyDescriptor]:
        return _Property1


class _Property0(_Property5):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/relations#colleagueOf"
    _cache_instances_: _ClassVar[bool] = False


class _Property2(_Property7):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/relations#employeeOf"
    _cache_instances_: _ClassVar[bool] = False


class _Property4(_Property2):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/relations#headsOf"
    _cache_instances_: _ClassVar[bool] = False


_descriptor0 = _Property5(_classes.Person, "knows")
_classes.Person.knows = _descriptor0

_descriptor1 = _Property1(_classes.Organization, "contains_part")
_classes.Organization.contains_part = _descriptor1

_descriptor2 = _Property7(_classes.Person, "works_for")
_classes.Person.works_for = _descriptor2

_descriptor3 = _Property3(_classes.Company, "has_employee")
_classes.Company.has_employee = _descriptor3

_descriptor4 = _Property6(_classes.Organization, "part_of")
_classes.Organization.part_of = _descriptor4

_descriptor5 = _Property0(_classes.Person, "colleague_of")
_classes.Person.colleague_of = _descriptor5

_descriptor6 = _Property2(_classes.Person, "employee_of")
_classes.Person.employee_of = _descriptor6

_descriptor7 = _Property4(_classes.Person, "heads_of")
_classes.Person.heads_of = _descriptor7

PROPERTY_DESCRIPTORS = {
    "https://example.org/relations#knows": _descriptor0,
    "https://example.org/relations#containsPart": _descriptor1,
    "https://example.org/relations#worksFor": _descriptor2,
    "https://example.org/relations#hasEmployee": _descriptor3,
    "https://example.org/relations#partOf": _descriptor4,
    "https://example.org/relations#colleagueOf": _descriptor5,
    "https://example.org/relations#employeeOf": _descriptor6,
    "https://example.org/relations#headsOf": _descriptor7,
}
