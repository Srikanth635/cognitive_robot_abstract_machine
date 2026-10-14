"""Exercise Phase 3 native KRROOD relation implications."""

from dataclasses import fields

from krrood.entity_query_language.factories import contains, entity, variable
from krrood.ontomatic.property_descriptor.mixins import (
    HasInverseProperty,
    TransitiveProperty,
)
from krrood.symbol_graph.symbol_graph import SymbolGraph
from relations_model import (
    RELATION_AXIOMS,
    RETRACTION_SUPPORTED,
    UPDATE_POLICY,
    Company,
    Manager,
    Organization,
    Person,
)

assert isinstance(Person.works_for, HasInverseProperty)
assert isinstance(Organization.contains_part, TransitiveProperty)
assert UPDATE_POLICY == "additions_after_construction"
assert RETRACTION_SUPPORTED is False
assert RELATION_AXIOMS
assert next(item for item in fields(Person) if item.name == "works_for").init is False

alice = Person(name=["Alice"])
bob = Manager(name=["Bob"])
acme = Company()
alice.works_for.add(acme)
assert acme in alice.works_for
assert alice in acme.has_employee
assert len(alice.works_for) == 1
assert len(acme.has_employee) == 1

acme.has_employee.add(bob)
assert acme in bob.works_for
assert not bob.employee_of
assert not bob.heads_of

manager = Manager(name=["Manager"])
manager.heads_of.add(acme)
assert acme in manager.employee_of
assert acme in manager.works_for
assert manager in acme.has_employee

alice.knows.add(bob)
assert alice in bob.knows
assert len(alice.knows) == 1
bob.knows.add(bob)
assert bob in bob.knows

colleague = Person(name=["Colleague"])
colleague.colleague_of.add(alice)
assert alice in colleague.knows
assert colleague in alice.knows
assert colleague not in alice.colleague_of

root, branch, leaf = Organization(), Company(), Organization()
root.contains_part.add(branch)
branch.contains_part.add(leaf)
assert leaf in root.contains_part
assert root in leaf.part_of
assert branch in leaf.part_of

outer, middle, inner = Organization(), Organization(), Organization()
middle.contains_part.add(inner)
outer.contains_part.add(middle)
assert inner in outer.contains_part

first, second, third = Organization(), Organization(), Organization()
first.part_of.add(second)
second.part_of.add(third)
assert third in first.part_of
assert first in third.contains_part

people = variable(Person, domain=[alice, bob, manager, colleague])
employees = entity(people).where(contains(people.works_for, acme)).evaluate()
assert set(employees) == {alice, bob, manager}

regions = variable(Organization, domain=[root, branch, leaf])
containers = entity(regions).where(contains(regions.contains_part, leaf)).evaluate()
assert set(containers) == {root, branch}

inferred = [edge for edge in SymbolGraph().relations() if edge.inferred]
assert any(
    edge.source.instance is root and edge.target.instance is leaf for edge in inferred
)
assert any(
    edge.source.instance is acme and edge.target.instance is alice for edge in inferred
)
print(
    "Phase 3 passed: inverse, transitive, symmetric, subproperty implications, and EQL queries"
)
