"""Run the Phase 9 generic example after compiling scoped_semantics.ttl."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import scoped_model as m

base = "https://example.org/scoped#"
a = m.Alpha()
b = m.Beta()
inner = m.Carrier()
outer = m.Carrier()
inner.link.append(a)
outer.link.append(inner)
scope = {base + "one": a, base + "two": b, "urn:inner": inner, "urn:outer": outer}
assert m.model_queries.match(base + "Either", scope) == [a, b]
assert m.model_queries.match(base + "Alias", scope) == [a]
assert m.model_queries.match(base + "Nested", scope) == [outer]
assert m.model_queries.match(base + "Choice", {base + "one": a}) == [a]
seed = m.model_relations.initialize_reflexivity(scope)
assert seed.added == 4
assert m.model_relations.initialize_reflexivity(scope).added == 0
a.connected.add(b)
b.connected.add(inner)
assert inner in a.connected and a in inner.connected
facts = m.model_relations.snapshot(scope)
assert any(
    f.subject_iri == base + "one" and f.object_iri == "urn:inner" and f.inferred
    for f in facts
)
a.before.add(b)
b.before.add(a)
report = m.model_validator.validate(scope, complete=True, identity_policy="python")
assert any(c.kind == "asymmetric" for c in report.violations)
print(
    json.dumps(
        {
            "positive_definitions": len(m.CLASS_EXPRESSIONS),
            "queries": "union, alias, nested restriction and partial enumeration passed",
            "native_relations": "cross-type symmetric and transitive additions passed",
            "reflexive_seeds": seed.added,
            "asymmetric_conflict_detected": True,
            "python_types_unchanged": type(outer) is m.Carrier,
        },
        indent=2,
    )
)
