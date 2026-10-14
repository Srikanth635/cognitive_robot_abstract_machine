"""Exercise the generated generic scoped validator against deliberate changes."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from validation_model import Child, Other, Part, model_validator

assembly = Child()
part = Part()
scope = {"urn:assembly": assembly, "urn:part": part}
open_report = model_validator.validate(scope)
assert open_report.valid is None
assert open_report.unknown

assembly.parts.append(part)
assembly.primary.append(part)
assembly.tag.append("ready")
complete_report = model_validator.validate(
    scope, complete=True, identity_policy="python"
)
assert complete_report.valid is True, complete_report.to_dict()

other = Other()
assembly.parts.append(other)
scope["urn:other"] = other
bad_report = model_validator.validate(scope, complete=True, identity_policy="python")
assert any(c.kind == "all" and c.status == "violation" for c in bad_report.checks)
assert next(c for c in bad_report.checks if c.kind == "exact").status == "satisfied"

assembly.tag.append("conflicting second tag")
functional_report = model_validator.validate(
    scope, complete=True, identity_policy="python"
)
assert any(c.kind == "functional" for c in functional_report.violations)

print(
    json.dumps(
        {
            "incomplete_missing_values": open_report.to_dict()["summary"],
            "complete_valid_values": complete_report.to_dict()["summary"],
            "wrong_filler_detected": True,
            "qualified_count_excludes_other_type": True,
            "functional_conflict_detected": True,
        },
        indent=2,
    )
)
