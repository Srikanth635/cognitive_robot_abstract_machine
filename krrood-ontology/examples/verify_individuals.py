"""Verify Phase 4 loading, IRI identity, literals and native relation implications."""

from pathlib import Path

from krrood.entity_query_language.factories import contains, entity, variable
from rdflib import RDFS, URIRef
from relations_model import Manager, Person, load_individuals
from university_model import Student
from university_model import load_individuals as load_university

examples = Path(__file__).resolve().parent
university = load_university(examples / "university.ttl")
alice_student = university["https://example.org/university#Alice"]
course = university["https://example.org/university#Course42"]
assert isinstance(alice_student, Student)
assert alice_student.name == ["Alice"]
assert alice_student.takes_course[0] is course

loaded = load_individuals(examples / "individuals.ttl")
base = "https://example.org/relations#"
alice, bob, acme, root, branch, office = [
    loaded[base + name]
    for name in ("Alice", "Bob", "Acme", "Headquarters", "Branch", "Office")
]
assert len(loaded.by_iri) == 6
assert isinstance(bob, Manager)
assert isinstance(bob, Person)
assert loaded.asserted_types[base + "Bob"] == (base + "Manager", base + "Person")
assert alice.name == ["Alice"]
assert acme in alice.works_for
assert alice in acme.has_employee
assert bob in acme.has_employee
assert acme in bob.employee_of
assert alice in bob.knows
assert office in root.contains_part
assert root in office.part_of
assert (
    URIRef(base + "Headquarters"),
    URIRef(base + "containsPart"),
    URIRef(base + "Office"),
) not in loaded.asserted_graph
assert len(acme.has_employee) == 2
assert next(iter(alice.works_for)) is loaded[base + "Acme"]
label = loaded.annotations[base + "Alice"][str(RDFS.label)][0]
assert str(label) == "Alice" and label.language == "en"
people = variable(Person, domain=[alice, bob])
assert set(entity(people).where(contains(people.works_for, acme)).evaluate()) == {
    alice,
    bob,
}

second = load_individuals(examples / "individuals.ttl")
assert second[base + "Alice"] is not alice
assert next(iter(second[base + "Alice"].works_for)) is second[base + "Acme"]
assert loaded.source_sha256 == second.source_sha256
print(
    "Phase 4 passed: individual loading, IRI identity, inherited types, literals, relations, annotations, and EQL"
)
