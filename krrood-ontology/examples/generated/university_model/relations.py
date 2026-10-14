"""Object relation tracking delegated to KRROOD property descriptors."""

from __future__ import annotations

from typing import ClassVar as _ClassVar

from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyDescriptor as _PropertyDescriptor,
)

from . import classes as _classes


class _Property0(_PropertyDescriptor):
    """Object-property descriptor generated from an ontology declaration."""

    __ontology_iri__ = "https://example.org/university#takesCourse"
    _cache_instances_: _ClassVar[bool] = False


_descriptor0 = _Property0(_classes.Student, "takes_course")
_classes.Student.takes_course = _descriptor0

PROPERTY_DESCRIPTORS = {
    "https://example.org/university#takesCourse": _descriptor0,
}
