"""Exercise generated dataclasses and KRROOD relation tracking."""

from dataclasses import fields, is_dataclass

from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.entity_query_language.predicate import Symbol
from krrood.ontomatic.property_descriptor.property_descriptor import PropertyDescriptor
from krrood.symbol_graph.symbol_graph import SymbolGraph
from university_model import CLASS_BY_IRI, PROPERTY_DESCRIPTORS, Course, Person, Student

base = "https://example.org/university#"
assert is_dataclass(Student)
assert issubclass(Student, Person)
assert issubclass(Person, Symbol)
assert CLASS_BY_IRI[base + "Student"] is Student
assert get_type_hints_of_object(Student)["takes_course"] == list[Course]
assert get_type_hints_of_object(Person)["name"] == list[str]
name_field = next(item for item in fields(Student) if item.name == "name")
assert name_field.metadata["ontology_iri"] == base + "name"
assert isinstance(Student.takes_course, PropertyDescriptor)
assert PROPERTY_DESCRIPTORS[base + "takesCourse"] is Student.takes_course

alice = Student(name=["Alice"])
bob = Student(name=["Bob"])
course = Course()
alice.takes_course.append(course)
assert isinstance(alice, Person)
assert alice.name == ["Alice"]
assert list(alice.takes_course) == [course]
assert list(bob.takes_course) == []
assert alice.name is not bob.name
assert any(
    relation.source.instance is alice and relation.target.instance is course
    for relation in SymbolGraph().relations()
)
alice.takes_course.remove(course)
assert list(alice.takes_course) == []
print(
    "Phase 2 passed: dataclasses, inheritance, typed properties, and KRROOD object relations"
)
