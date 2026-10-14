"""Object relation tracking delegated to KRROOD property descriptors."""

from __future__ import annotations

from typing import ClassVar as _ClassVar

from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyDescriptor as _PropertyDescriptor,
)
from krrood.ontomatic.property_descriptor.mixins import (
    HasInverseProperty as _HasInverseProperty,
)

from . import classes as _classes


class _Property0(_PropertyDescriptor, _HasInverseProperty):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/expressions#hasEmployee"
    _cache_instances_: _ClassVar[bool] = False

    @classmethod
    def get_inverse(cls) -> type[_PropertyDescriptor]:
        return _Property1


class _Property1(_PropertyDescriptor, _HasInverseProperty):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/expressions#worksFor"
    _cache_instances_: _ClassVar[bool] = False

    @classmethod
    def get_inverse(cls) -> type[_PropertyDescriptor]:
        return _Property0


_descriptor0 = _Property0(_classes.Company, "has_employee")
_classes.Company.has_employee = _descriptor0

_descriptor1 = _Property1(_classes.Person, "works_for")
_classes.Person.works_for = _descriptor1

PROPERTY_DESCRIPTORS = {
    "https://example.org/expressions#hasEmployee": _descriptor0,
    "https://example.org/expressions#worksFor": _descriptor1,
}
