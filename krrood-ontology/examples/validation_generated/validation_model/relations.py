"""Object relation tracking delegated to KRROOD property descriptors."""

from __future__ import annotations

from typing import ClassVar as _ClassVar

from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyDescriptor as _PropertyDescriptor,
)

from . import classes as _classes


class _Property0(_PropertyDescriptor):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/validation#parts"
    _cache_instances_: _ClassVar[bool] = False


class _Property1(_PropertyDescriptor):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/validation#primary"
    _cache_instances_: _ClassVar[bool] = False


PROPERTY_BINDINGS = {}
_descriptor0 = _Property0(_classes.Bound, "parts")
_classes.Bound.parts = _descriptor0
PROPERTY_BINDINGS[
    ("https://example.org/validation#parts", "https://example.org/validation#Bound")
] = _descriptor0
_descriptor1 = _Property0(_classes.UnionOwner, "parts")
_classes.UnionOwner.parts = _descriptor1
PROPERTY_BINDINGS[
    (
        "https://example.org/validation#parts",
        "https://example.org/validation#UnionOwner",
    )
] = _descriptor1
_descriptor2 = _Property0(_classes.Assembly, "parts")
_classes.Assembly.parts = _descriptor2
PROPERTY_BINDINGS[
    ("https://example.org/validation#parts", "https://example.org/validation#Assembly")
] = _descriptor2
_descriptor3 = _Property1(_classes.Assembly, "primary")
_classes.Assembly.primary = _descriptor3
PROPERTY_BINDINGS[
    (
        "https://example.org/validation#primary",
        "https://example.org/validation#Assembly",
    )
] = _descriptor3
_descriptor4 = _Property0(_classes.Maximum, "parts")
_classes.Maximum.parts = _descriptor4
PROPERTY_BINDINGS[
    ("https://example.org/validation#parts", "https://example.org/validation#Maximum")
] = _descriptor4
_descriptor5 = _Property0(_classes.Minimum, "parts")
_classes.Minimum.parts = _descriptor5
PROPERTY_BINDINGS[
    ("https://example.org/validation#parts", "https://example.org/validation#Minimum")
] = _descriptor5
PROPERTY_DESCRIPTORS = {
    "https://example.org/validation#parts": PROPERTY_BINDINGS[
        ("https://example.org/validation#parts", "https://example.org/validation#Bound")
    ],
    "https://example.org/validation#primary": PROPERTY_BINDINGS[
        (
            "https://example.org/validation#primary",
            "https://example.org/validation#Assembly",
        )
    ],
}
