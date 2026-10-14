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


class _Property0(_PropertyDescriptor, _HasInverseProperty, _TransitiveProperty):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/roundtrip#links"
    _cache_instances_: _ClassVar[bool] = False

    @classmethod
    def get_inverse(cls) -> type[_PropertyDescriptor]:
        return _Property0


PROPERTY_BINDINGS = {}
_descriptor0 = _Property0(_classes._OntologyEntity, "links")
_classes._OntologyEntity.links = _descriptor0
PROPERTY_BINDINGS[("https://example.org/roundtrip#links", "")] = _descriptor0
PROPERTY_DESCRIPTORS = {
    "https://example.org/roundtrip#links": PROPERTY_BINDINGS[
        ("https://example.org/roundtrip#links", "")
    ],
}
