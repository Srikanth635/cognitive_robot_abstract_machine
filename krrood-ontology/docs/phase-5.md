# Phase 5: verified compiler output and local imports

This document records Phase 5. Release 0.6.0 adds the positive expression query
profile in [phase-6.md](phase-6.md), retaining these compiler verification and
local-import APIs.

The compiler now publishes output only after a fresh interpreter verifies the
generated package. Strict compilation also preflights the source individuals
against that package. Local imports are resolved through explicitly supplied
files, and output includes source snapshots, a portable asserted closure,
feature treatment and an environment record.

This hardens the Phases 1–4 profile. It does not add new OWL inference rules or
claim complete OWL conformance.

## Install and regenerate

Use the same working KRROOD environment:

```bash
cd krrood-ontology
python -m pip install --no-deps -e .
python -m krrood_ontology --version
# krrood-ontology 0.5.0

python -m krrood_ontology compile examples/imports/university.ttl \
    --import https://example.org/core=examples/imports/core.ttl \
    --output generated_phase5 --package imported_model

PYTHONPATH=generated_phase5 python examples/verify_imports.py generated_phase5
```

Expected result:

```text
Phase 5 passed: local imports, isolated verification, provenance, coverage, and runtime loading
```

Use a fresh destination. The supplied sample also runs directly:

```bash
PYTHONPATH=examples/imports_generated \
    python examples/verify_imports.py examples/imports_generated
```

The existing university and relation examples still compile with the same
commands. They also receive the new verification and reporting artifacts.

## Modules in build order

| Module | Responsibility | Reuse |
| --- | --- | --- |
| `imports.py` | Resolve the explicit local import closure; terminate cycles; reject missing/duplicate/unused mappings | Existing RDFLib loader and translation diagnostic records |
| `verification_worker.py` | Import the staged package, check class/field metadata and object descriptor registration, preflight individuals | Python imports, dataclass introspection, existing Phase 4 planner |
| `verification.py` | Launch the worker in an isolated interpreter with a timeout and actionable errors | Python subprocess API |
| `features.py` | Inventory observed predicates/type terms and report their treatment | Existing RDFLib graph and compilation diagnostics |
| `compiler.py`, `config.py`, `cli.py` | Integrate resolution, snapshots, verification and publication | Earlier generation pipeline and KRROOD backend |

## Local import contract

Each `--import IRI=PATH` maps the exact RDF import IRI to a supplied local file.
Repeat it for transitive imports. All reachable imports need mappings when
resolution is requested. The resolver never dereferences an IRI or downloads an
ontology. Duplicate mappings and unused mappings fail, helping catch typos.
Graphs are merged with RDFLib, documents are tracked by resolved local path,
and import cycles terminate. Structural/semantic mapping then applies to the
whole resolved closure. Conflicting declarations are subject to the same
signature, hierarchy and collision checks as a single ontology.

No automatic catalog discovery, ontology-version selection or proof that a
file's ontology identifier matches its mapped IRI is provided. The explicit
mapping is the user's choice of import source. Relative RDF IRIs resolve against
the supplied file's URI, which is recorded in provenance. Use the Python API
when a mapped file needs an explicit serialization:

```python
from pathlib import Path
from krrood_ontology import (
    CompilationOptions, LocalImport, OntologyCompiler, OntologyInput, RDFFormat,
)

result = OntologyCompiler().compile(
    OntologyInput(Path("examples/imports/university.ttl")),
    CompilationOptions(
        Path("generated_phase5_api"), "imported_model",
        local_imports=(LocalImport(
            "https://example.org/core",
            OntologyInput(Path("examples/imports/core.ttl"), RDFFormat.TURTLE),
        ),),
    ),
)
```

Without mappings, strict mode rejects unresolved imports. `--structural-only`
with no mappings explicitly omits their semantics and records the omission;
structural declarations in the root must still be independently representable.
Providing some mappings does not consent to silently skipping other imports.

## Output

```text
generated_phase5/
├── manifest.json
├── features.json
├── source/
│   ├── closure.ttl
│   └── documents/
│       ├── 000.turtle
│       └── 001.turtle
└── imported_model/
    ├── __init__.py
    ├── classes.py
    ├── relations.py
    ├── vocabulary.py
    ├── axioms.py
    └── runtime.py
```

Original source bytes are copied with hashes, formats and original base IRIs.
The canonical closure uses RDFLib's canonical graph and N-Triples serializer;
N-Triples is valid Turtle, so `closure.ttl` contains explicit absolute IRIs and
stable sorted statements. Resolved `owl:imports` triples are removed from this
runtime closure; the original imports remain in the source snapshots and
manifest. Unresolved structural-only imports remain asserted in the closure.
The Phase 4 loader intentionally rejects such unresolved imports.

Load the resolved snapshot to instantiate individuals across imported files:

```python
from imported_model import load_individuals

loaded = load_individuals("generated_phase5/source/closure.ttl")
alice = loaded["https://example.org/imported#Alice"]
assert alice.name == ["Alice"]
```

The source snapshots preserve original content, including unsupported source
metadata. They are not a round-trip exporter for later object mutations.

## Verification and feature reporting

The worker uses the active Python interpreter with `-I -B`, explicit paths to
the staged output and installed wrapper, and a 45-second default timeout. It
checks artifact origin, importability, dataclass metadata, field IRIs and
descriptor attachment. Strict mode runs Phase 4 individual preflight without
constructing objects. Structural-only mode omits that preflight and records it.
No source individuals are instantiated during compilation; native implication
execution remains covered by the runtime integration tests.

The isolated process avoids inherited application symbols, model registries,
`PYTHONPATH` and working-directory import state. It is an import verification
process, not an operating-system sandbox or a test of every possible mutation.
Normal compiler failures clean up staged output; successful checks precede the
final directory rename. Existing destinations remain protected. Crash recovery,
concurrent writers and atomic rollback of runtime object updates are outside
this contract.

`features.json` reports predicate counts, RDF type terms and whether each is
structurally mapped, mapped to native relations, subject to runtime preflight,
retained as annotations/source content, or explicitly omitted. A retained
source term is not a claim of executable semantic support. Unknown properties
on recognized individuals fail strict preflight; arbitrary schema metadata can
remain source-only. The report explicitly sets `complete_owl_conformance: false`.

The manifest records the Python and wrapper/KRROOD/RDFLib/Jinja/Black versions,
all source fingerprints, the closure fingerprint, resolved import IRIs and
verification results. This is an environment record rather than a complete
dependency lock. Generated code and the closure are reproducible in the tested
same-source, same-environment cases; source paths and environment versions are
deliberately included in provenance, so cross-machine manifests can differ.

## Remaining improvements

The five original milestones are delivered for the documented subset. Near
1-to-1 translation of broader OWL still needs explicit support for unrelated
multiple types, anonymous class expressions, restrictions, classification,
identity normalization, retraction and export of object changes. Continue to
reuse KRROOD mechanisms where their contracts fit, and preserve unsupported
axioms without describing them as implemented.
