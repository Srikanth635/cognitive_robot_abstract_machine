"""Verify a locally resolved ontology package and its preserved source closure."""

import json
import sys
from pathlib import Path

from imported_model import Person, Student, load_individuals

output = Path(sys.argv[1])
manifest = json.loads((output / "manifest.json").read_text())
assert manifest["imports_loaded"] is True
assert manifest["resolved_import_iris"] == ["https://example.org/core"]
assert manifest["verification"]["isolated_import"] == "passed"
assert manifest["verification"]["individual_preflight"] == "passed"
assert len(manifest["source_documents"]) == 2
assert (output / "features.json").is_file()
loaded = load_individuals(output / manifest["runtime_ontology"])
alice = loaded["https://example.org/imported#Alice"]
course = loaded["https://example.org/imported#Course42"]
assert isinstance(alice, Student) and isinstance(alice, Person)
assert alice.name == ["Alice"]
assert alice.takes_course[0] is course
print(
    "Phase 5 passed: local imports, isolated verification, provenance, coverage, and runtime loading"
)
