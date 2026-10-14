"""Verify an ontology containing a class and no properties."""

from dataclasses import is_dataclass

from krrood.entity_query_language.predicate import Symbol
from minimal_model import (
    CLASS_BY_IRI,
    PROPERTY_DESCRIPTORS,
    PROPERTY_FIELD_NAMES,
    Person,
)

assert is_dataclass(Person)
assert isinstance(Person(), Symbol)
assert CLASS_BY_IRI["https://example.org/minimal#Person"] is Person
assert PROPERTY_DESCRIPTORS == {}
assert PROPERTY_FIELD_NAMES == {}
print("Minimal model passed")
