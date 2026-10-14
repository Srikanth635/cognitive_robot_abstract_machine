import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import aicor_model as m

v = m.model_validator
base = "http://aicor.knowledge/l2-ontology.owl#"
critic = m.CLASS_BY_IRI[base + "AccessibilityCritic"]()
start = time.monotonic()
scope = {"urn:critic": critic}
r = v.validate(scope, complete=True, identity_policy="python")
target = [
    c
    for c in r.checks
    if c.class_iri == base + "AccessibilityCritic"
    and c.property_iri == base + "criticDimension"
]
assert len(target) == 1 and target[0].status == "violation", target
critic.critic_dimension.append("accessibility")
r = v.validate(scope, complete=True, identity_policy="python")
assert (
    next(
        c
        for c in r.checks
        if c.class_iri == base + "AccessibilityCritic"
        and c.property_iri == base + "criticDimension"
    ).status
    == "satisfied"
)
executive = m.CLASS_BY_IRI[base + "AICORReferenceManipulationExecutive"]()
program = m.CLASS_BY_IRI[base + "ManipulationControlProgram"]()
os = m.CLASS_BY_IRI[base + "ManipulationExecutiveOperatingSystem"]()
executive.has_proper_part.extend([program, os])
scope = {"urn:exec": executive, "urn:program": program, "urn:os": os}
r = v.validate(scope, complete=True, identity_policy="python")
exact = [
    c
    for c in r.checks
    if c.class_iri == base + "AICORReferenceManipulationExecutive" and c.kind == "exact"
]
assert len(exact) == 2 and all(
    c.status == "satisfied" and c.observed_count == 1 for c in exact
), exact
result = {
    "coverage": v.coverage(),
    "critic_required_value": "missing violation, present satisfied",
    "qualified_parts": "one operating system and one control program satisfy the two separate exact-cardinality constraints",
    "checks_in_executive_scope": len(r.checks),
    "execution_seconds": round(time.monotonic() - start, 3),
}
print(json.dumps(result, indent=2))
