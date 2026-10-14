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


class _Property0(_PropertyDescriptor):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/scoped#before"
    _cache_instances_: _ClassVar[bool] = False


class _Property1(_PropertyDescriptor, _HasInverseProperty, _TransitiveProperty):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/scoped#connected"
    _cache_instances_: _ClassVar[bool] = False

    @classmethod
    def get_inverse(cls) -> type[_PropertyDescriptor]:
        return _Property1


class _Property2(_PropertyDescriptor):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/scoped#link"
    _cache_instances_: _ClassVar[bool] = False


PROPERTY_BINDINGS = {}
_descriptor0 = _Property2(_classes.Carrier, "link")
_classes.Carrier.link = _descriptor0
PROPERTY_BINDINGS[
    ("https://example.org/scoped#link", "https://example.org/scoped#Carrier")
] = _descriptor0
_descriptor1 = _Property2(_classes.Linked, "link")
_classes.Linked.link = _descriptor1
PROPERTY_BINDINGS[
    ("https://example.org/scoped#link", "https://example.org/scoped#Linked")
] = _descriptor1
_descriptor2 = _Property2(_classes.Nested, "link")
_classes.Nested.link = _descriptor2
PROPERTY_BINDINGS[
    ("https://example.org/scoped#link", "https://example.org/scoped#Nested")
] = _descriptor2
_descriptor3 = _Property0(_classes._OntologyEntity, "before")
_classes._OntologyEntity.before = _descriptor3
PROPERTY_BINDINGS[("https://example.org/scoped#before", "")] = _descriptor3
_descriptor4 = _Property1(_classes._OntologyEntity, "connected")
_classes._OntologyEntity.connected = _descriptor4
PROPERTY_BINDINGS[("https://example.org/scoped#connected", "")] = _descriptor4
PROPERTY_DESCRIPTORS = {
    "https://example.org/scoped#before": PROPERTY_BINDINGS[
        ("https://example.org/scoped#before", "")
    ],
    "https://example.org/scoped#connected": PROPERTY_BINDINGS[
        ("https://example.org/scoped#connected", "")
    ],
    "https://example.org/scoped#link": PROPERTY_BINDINGS[
        ("https://example.org/scoped#link", "https://example.org/scoped#Carrier")
    ],
}
