"""Exercise positive class-expression queries using the generated Phase 6 model."""

from dataclasses import replace
from pathlib import Path

from expressions_model import (
    CLASS_EXPRESSIONS,
    WorkingPerson,
    load_individuals,
    model_queries,
)
from krrood.entity_query_language.query.query import Query

loaded = load_individuals(Path(__file__).with_name("expressions.ttl"))
base = "https://example.org/expressions#"
alice, bob, charlie, acme, empty, explicit = [
    loaded[base + name]
    for name in ("Alice", "Bob", "Charlie", "Acme", "EmptyCompany", "ExplicitWorker")
]
expected = {
    "WorkingPerson": {alice, bob, explicit},
    "AcmeWorker": {alice, bob},
    "NamedAlice": {alice},
    "SelectedPerson": {alice, charlie},
    "PersonOrCompany": {alice, bob, charlie, acme, empty, explicit},
    "WorkingOrSelected": {alice, bob, charlie, explicit},
    "EmployerOfManager": {acme},
    "EmployerOfWorker": {acme},
    "EmployerOfAlice": {acme},
}
for name, members in expected.items():
    assert set(model_queries.match(base + name, loaded)) == members, name
assert set(
    model_queries.match(base + "WorkingPerson", loaded, candidates=[alice, charlie])
) == {alice}
assert model_queries.match(base + "WorkingPerson", loaded, candidates=[]) == []
query = model_queries.query(base + "WorkingPerson", loaded)
assert isinstance(query, Query)
assert not isinstance(alice, WorkingPerson)
assert isinstance(explicit, WorkingPerson)
assert len(CLASS_EXPRESSIONS) == len(expected)
assert not explicit.works_for
before = set(loaded.asserted_graph)
model_queries.match(base + "EmployerOfAlice", loaded)
assert set(loaded.asserted_graph) == before
partial = replace(
    loaded, by_iri={iri: obj for iri, obj in loaded.by_iri.items() if obj is not acme}
)
assert set(model_queries.match(base + "WorkingPerson", partial)) == {explicit}
try:
    model_queries.match(base + "AcmeWorker", partial)
except ValueError as error:
    assert "absent from the supplied registry" in str(error)
else:
    raise AssertionError("Missing expression individuals must fail explicitly")

second = load_individuals(Path(__file__).with_name("expressions.ttl"))
assert set(model_queries.match(base + "AcmeWorker", second)) == {
    second[base + "Alice"],
    second[base + "Bob"],
}
assert alice not in model_queries.match(base + "AcmeWorker", second)
try:
    model_queries.match(
        base + "WorkingPerson", loaded, candidates=[second[base + "Alice"]]
    )
except ValueError:
    pass
else:
    raise AssertionError("Foreign-registry candidates must fail")
print(
    "Phase 6 passed: intersections, unions, enumerations, value restrictions, existentials, nested queries, and scoped identity"
)
