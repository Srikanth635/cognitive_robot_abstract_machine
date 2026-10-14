# Phase 13: isolated Python–RDF construction

Phase 13 turns a successful Phase 12 candidate into actual generated dataclass instances inside a fresh worker process. It reuses KRROOD Symbols, property descriptors, monitored collections and native relation implications. The worker verifies the asserted RDF round trip, returns immutable object views and relation facts, and exits. The parent native graph and accepted RDF are unchanged.

## Install and test

Extract the ZIP and run these commands from its `krrood-ontology` directory with your KRROOD environment active:

```bash
python -m pip install --no-deps -e .
python -m krrood_ontology --version
python -m pytest tests/test_construction.py tests/test_graph_loading.py -q
python examples/verify_phase13.py examples/generated university_model
python examples/verify_phase13.py examples/aicor_generated aicor_model
python -m pytest tests -q
```

Expected version: `0.13.0`. The focused command covers 26 new tests (21 construction and 5 graph-loading cases); the full suite passed all 211 tests. Saved demonstration, wheel and test results are included in `docs/phase-13-*-results.json`. A fresh environment can use `python -m pip install -e '.[dev]'` to install the pinned KRROOD dependency and pytest. The module name uses the underscore: `krrood_ontology`.

The demonstration uses the same registry-driven class/property selection for both ontologies. It requires an operational object property. The library API also supports candidates containing only classes or literals. Expected results include:

```json
{
  "preflight": "pass",
  "construction": "pass",
  "construction_profile": "isolated_generated_objects_v1",
  "worker_process_isolated": true,
  "worker_native_objects_retained": false,
  "one_object_per_iri": true,
  "shared_reference_identity_verified": true,
  "asserted_rdf_roundtrip_equal": true,
  "candidate_added_triples": 3,
  "old_artifact_invalidated": true,
  "parent_native_graph_unchanged": true,
  "accepted_rdf_unchanged": true,
  "local_validation": "unsupported",
  "reasoner": "unsupported",
  "acceptance_blocked": true,
  "facts_committed": false
}
```

The included university demonstration constructs four objects: two source individuals and two proposed individuals. AICOR constructs six: four source individuals and two proposed individuals. These are instances, not counts of schema classes. The worker's native object count is separate from the unchanged parent object count.

## Files and reuse

| File | Responsibility |
| --- | --- |
| `src/krrood_ontology/construction.py` | Constructor/checker adapter, worker lifecycle, revision-bound immutable results |
| `src/krrood_ontology/construction_worker.py` | Import the verified generated package, run native construction and inspect its results |
| `tests/test_construction.py` | Real native construction, lifecycle, failure, timeout and compatibility coverage |
| `tests/test_graph_loading.py` | Graph loading, literal spelling, defensive copies and explicit-domain export regression |
| `examples/verify_phase13.py` | Metadata-driven demonstration for compatible generated models |

`IndividualLoader.load_graph()` adds graph input to the existing loader. File loading and graph loading share one two-pass materializer: preflight, allocate one object per IRI, then populate native collections and shared references. Literal conversion remains RDFLib's existing conversion. Identity handling remains `IdentityConstraints`; the worker uses the existing `distinct` profile. No new class generator, relation inference engine, identity merger, datatype converter or OWL engine was implemented.

`IndividualExporter` continues to project native values into a copy of asserted RDF, keeping schema axioms, annotations and identity statements. For explicit bindings, it now reads a property field only on classes in that field's domain. The planner still sees the original signature and still reports invalid domains. Compiler verification recognizes the included Phase 6 packages while retaining the required validation registration for Phase 8 and newer packages.

Generated modules do **not** need regeneration for this phase. Compiler feature manifests remain Phase 10 for fresh compilations; the service package release is `0.13.0`. Packages used by this bridge must have a manifest with verified closure provenance.

## API example

```python
import sys
from pathlib import Path

directory = Path("examples/generated").resolve()
sys.path.insert(0, str(directory))
import university_model as generated

from krrood_ontology.construction import CandidateConstructor
from krrood_ontology.contracts import (
    CreateEntity, EntityReference, ObjectAssertion, Proposal,
)
from krrood_ontology.preparation import CandidatePreparer
from krrood_ontology.sessions import OntologySystem

preparer = CandidatePreparer.from_manifest(
    generated.MODEL,
    directory / "manifest.json",
    schema_id="urn:my-project:ontology",
    base_revision="accepted-0",
)
constructor = CandidateConstructor(preparer, directory, "university_model")
base = "https://example.org/university#"  # This example's vocabulary only.
proposal = Proposal((
    CreateEntity("student-create", "student", (base + "Student",)),
    CreateEntity("course-create", "course", (base + "Course",)),
    ObjectAssertion(
        "link", EntityReference(local_id="student"),
        base + "takesCourse", EntityReference(local_id="course"),
    ),
))
session = OntologySystem(preparer.context, checker=constructor).begin_candidate(proposal)
result = constructor.construct(session.snapshot)
print(result.report.to_dict())
if result.artifact is not None:
    result.artifact.require_current(session.snapshot)
    student = result.artifact.object_for_local_id("student")
    print(student.python_type, student.iri)
    print(student.field(base + "takesCourse").values)
    rdf = result.artifact.graph()  # Defensive copy of verified candidate RDF.

# Stores a new revision-bound report by executing a separate fresh worker:
review = session.check()
# session.prepare_acceptance() remains blocked by the default policy.
```

The returned `student` is a frozen `ObjectView`. The actual `generated.Student` instance existed in the worker and was discarded when it exited. No live native object, proxy, mutable collection or permanent worker is returned. This phase establishes a safe candidate construction/checking boundary; keeping accepted native objects alive belongs to Phase 16's synchronization policy.

`construct()` returns both report and artifact. `check()` implements the Phase 11 checker protocol and returns only a report. Each call runs independently in a fresh process; there is no shared last-artifact cache. Retain the `ConstructionResult` when object/RDF evidence is needed.

`ObjectView.identity_index` is an ordinal inside one artifact, not a persistent Python address or accepted identity. References in object fields are resolved ontology IRIs. Converted literal views describe the native values; the `PreparedCandidate` retains original RDF literal spellings. For example, asserted `001.2300` can become native `Decimal('1.2300')` while the RDF round trip still retains `001.2300`. Private worker transfer uses a per-pickler RDFLib literal hook to preserve spelling without modifying RDFLib's global normalization flag.

## Construction boundary and feedback

KRROOD registers Symbols in a process-global `SymbolGraph` during allocation. A constructor failure can therefore leave partially registered objects. The adapter owns a fresh interpreter for each request and lets process exit discard the whole candidate native graph. It does not clear the parent's graph, replace KRROOD singletons, or implement rollback in native relation code.

The adapter validates the generated manifest/closure and imports through the existing `PackageVerifier` during setup. It fingerprints generated Python files, the manifest and closure before/after verification and construction. The worker compares its imported model registry with the preflight registry. These checks pin trusted artifacts; they are not a semantic proof that arbitrary Python implements OWL correctly. Generated packages and their dependencies are trusted server code. A subprocess is not a filesystem or security sandbox.

The worker runs with `-I -B` and a configurable construction deadline, default 45 seconds. `subprocess.run()` stops and waits for the direct worker on timeout, and the private transfer directory is removed. The setup verifier has its own existing 45-second deadline. The package does not manage subprocess trees created by arbitrary generated code. Request/preflight time is outside the native worker deadline.

| Condition | Construction report |
| --- | --- |
| Preflight failed or unsupported | `unknown`, `preflight_required`; no worker and no artifact |
| Successful native load and exact asserted projection | `pass`; immutable artifact |
| Existing exporter cannot represent a value | `unsupported`, `projection_unsupported`; no artifact |
| Native loader rejects the candidate | `fail` with its existing diagnostic code; no artifact |
| Constructor/descriptor exception | `error`, `construction_failed`; no private exception text or partial artifact |
| Deadline exceeded | `error`, `construction_timeout`; no artifact |
| Generated artifact changed | `error`, `generated_artifact_changed`; no artifact |
| Imported registry differs from preflight | `error`, `model_registry_changed`; no artifact |
| Worker result belongs to another candidate | `error`, `construction_protocol`; no artifact |

These are execution and object-model checks. They are not proofs of OWL inconsistency. Successful native relation implications are recorded through `ModelRelations.snapshot()`; inferred edge flags describe first insertion and do not provide proof histories. Inferred-only relations are excluded from the asserted RDF round-trip comparison.

Reflexive seeding is optional with `seed_reflexivity=True`. It delegates to the generated `model_relations.initialize_reflexivity()` helper and records the seed count. Older generated packages without that helper report `reflexivity_unavailable`. The default is false. Native descriptors apply their existing implications without a second closure algorithm.

## Revision and scope limits

Call `artifact.require_current()` with the actual current server-side session snapshot before using retained evidence. Editing a proposal invalidates its stored review and old artifact. Comparing an artifact with an old saved snapshot does not detect a later edit. There is no accepted store or atomic commit yet.

The existing loader's supported projection remains in force: named individuals, explicit compatible types, operational bindings and supported literal conversions. `sameAs` merging, unrelated multiple types, anonymous individuals, reclassification and retraction are not added here. Schema axioms can remain in candidate RDF without being executed as Python behavior.

Local validation and reasoning still report `unsupported`; default acceptance stays blocked. No accepted facts are committed. Phase 14 will add an explicit reasoner adapter over candidate RDF; Phase 15 will combine actionable findings; Phase 16 will define acceptance and synchronized accepted-object lifetime. Phase 17 will expose the schema and stable interfaces to the agent system.
