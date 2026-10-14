"""AICOR is an integration fixture; translation code has no ontology-specific branches."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import aicor_model as m

from krrood_ontology.model import ExpressionKind

start = time.monotonic()
checked = 0
for iri, expression in m.CLASS_EXPRESSIONS.items():
    if expression.kind == ExpressionKind.UNION:
        leaf = expression
        while leaf.kind == ExpressionKind.UNION:
            leaf = leaf.operands[0]
        assert leaf.kind == ExpressionKind.CLASS
        obj = m.CLASS_BY_IRI[leaf.iri]()
        assert m.model_queries.match(iri, {"urn:witness": obj}) == [obj]
    elif expression.kind == ExpressionKind.ENUMERATION:
        # Identity is caller-supplied here; AllDifferent A-Box loading belongs to Phase 10.
        obj = next(iter(m.CLASS_BY_IRI.values()))()
        assert m.model_queries.match(iri, {expression.individuals[0]: obj}) == [obj]
    else:
        raise AssertionError(expression)
    checked += 1
base = "http://aicor.knowledge/l2-ontology.owl#"
a = m.CLASS_BY_IRI[base + "AccessibilityCritic"]()
b = m.CLASS_BY_IRI[base + "ManipulationControlProgram"]()
c = m.CLASS_BY_IRI[base + "ManipulationExecutiveOperatingSystem"]()
scope = {"urn:a": a, "urn:b": b, "urn:c": c}
seed = m.model_relations.initialize_reflexivity(scope)
assert seed.added == 3
name = m.MODEL.properties[base + "coRefersWith"].name
getattr(a, name).add(b)
getattr(b, name).add(c)
assert c in getattr(a, name) and a in getattr(c, name)
assert m.model_relations.initialize_reflexivity(scope).added == 0
facts = m.model_relations.snapshot(scope)
assert any(
    f.subject_iri == "urn:a" and f.object_iri == "urn:c" and f.inferred for f in facts
)
name = m.MODEL.properties[base + "forceDynamicallyPriorTo"].name
getattr(a, name).add(b)
getattr(b, name).add(a)
report = m.model_validator.validate(scope, complete=True, identity_policy="python")
assert any(v.kind == "asymmetric" for v in report.violations)
result = {
    "classes": len(m.CLASS_BY_IRI),
    "properties": len(m.MODEL.properties),
    "field_bindings": len(m.MODEL.bindings),
    "positive_definitions_checked": checked,
    "shared_native_properties": sum(not b.domain_iri for b in m.MODEL.bindings),
    "reflexive_seeds": seed.added,
    "scoped_relation_facts": len(facts),
    "asymmetric_conflict_detected": True,
    "runtime_check_seconds": round(time.monotonic() - start, 3),
}
print(json.dumps(result, indent=2))
