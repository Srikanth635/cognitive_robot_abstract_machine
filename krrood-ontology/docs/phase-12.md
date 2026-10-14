# Phase 12: candidate RDF and ontology-aware preflight

Phase 12 connects Phase 11 proposals to the existing generated model and RDF closure. A successful preparation returns a separate RDF candidate containing the authoritative ontology, its original individuals, captured additional accepted facts and the proposal's new assertions. It creates no native KRROOD objects, runs no reasoner and commits no facts.

## Install and test

From the extracted `krrood-ontology` directory, with your KRROOD environment active:

```bash
python -m pip install --no-deps -e .
python -m krrood_ontology --version
python -m pytest tests/test_preparation.py tests/test_contracts_sessions.py -q
python examples/verify_phase12.py examples/generated university_model
python examples/verify_phase12.py examples/aicor_generated aicor_model
python -m pytest tests -q
```

Expected package version: `0.12.0`. There are 31 Phase 12 tests, 55 tests in the focused command above, and 185 tests in the full suite. Pytest must be installed. A fresh environment can use `python -m pip install -e '.[dev]'`, which also installs the pinned KRROOD dependency. The module command uses the underscore: `krrood_ontology`.

The same demonstration selects an operational property and compatible explicit types from each model's metadata. No ontology names are embedded in the shared implementation or demonstration selection logic. Expected results include:

```json
{
  "preflight": "pass",
  "preflight_profile": "asserted_named_individuals_v1",
  "candidate_entities": 2,
  "candidate_added_triples": 3,
  "invalid_reference_rejected": true,
  "revision": 2,
  "old_review_invalidated": true,
  "provisional_iris_stable_across_revision": true,
  "accepted_rdf_unchanged": true,
  "acceptance_blocked": true,
  "reasoner": "unsupported; not executed",
  "facts_committed": false,
  "native_symbols_created": 0,
  "native_relations_created": 0
}
```

AICOR additionally reports 1,017 classes and 966 properties. Class/property counts and execution time vary with the supplied model. This demonstration checks model admissibility, not OWL consistency.

## New files and reuse

| File | Purpose |
| --- | --- |
| `src/krrood_ontology/preparation.py` | Candidate preparation, immutable RDF result, preflight checker adapter |
| `tests/test_preparation.py` | Generic generated-model, isolation, failure and revision tests |
| `examples/verify_phase12.py` | Same metadata-driven verification for different ontologies |

`CandidatePreparer` reuses `IndividualPlanner.plan_graph()` for explicit class selection, inherited type compatibility, operational property binding, declared domains/ranges, named target resolution, literal conversion and existing identity handling. It reuses RDFLib for RDF terms/graphs and Phase 11 for requests, reports and revisions. The new code supplies only the service glue: resolving proposal IDs, allocating provisional IRIs, assembling a candidate, capturing its baseline and translating planner errors to feedback.

KRROOD class generation, Symbols, descriptors, relation implications, monitored collections and EQL are unchanged. No new dependency was added. Generated modules need not be regenerated. Compiler manifests retain their existing compiler feature level (the current compiler is Phase 10); the package/service release is separately `0.12.0`.

## Using the adapter

```python
import sys
sys.path.insert(0, "examples/generated")
import university_model as model

from krrood_ontology.contracts import (
    CreateEntity, EntityReference, ObjectAssertion, Proposal,
)
from krrood_ontology.preparation import CandidatePreparer
from krrood_ontology.sessions import OntologySystem

adapter = CandidatePreparer.from_manifest(
    model.MODEL,
    "examples/generated/manifest.json",
    schema_id="urn:my-project:ontology",
    base_revision="accepted-0",
)
service = OntologySystem(adapter.context, checker=adapter)
base = "https://example.org/university#"  # Example ontology vocabulary only.
proposal = Proposal((
    CreateEntity("create-student", "student", (base + "Student",)),
    CreateEntity("create-course", "course", (base + "Course",)),
    ObjectAssertion(
        "takes-course", EntityReference(local_id="student"),
        base + "takesCourse", EntityReference(local_id="course"),
    ),
))
session = service.begin_candidate(proposal)
prepared = adapter.prepare(session.snapshot)
print(prepared.report.to_dict())
if prepared.candidate is not None:
    prepared.candidate.require_current(session.snapshot)
    rdf = prepared.candidate.graph()          # Defensive copy; schema + A-Box.
    additions = prepared.candidate.delta_graph()  # Only new proposed triples.
    individual_iri = prepared.candidate.iri_for("student")

# Store a revision-bound review in the Phase 11 lifecycle:
review = session.check()
# session.prepare_acceptance() remains blocked by the default policy.
```

`prepare()` returns both the RDF artifact and its report. `check()` implements Phase 11's `CandidateChecker` protocol and returns only a report. It does not keep a mutable "last prepared graph" cache shared between sessions. They are independent calls; clients needing the RDF artifact retain their `PreparationResult` explicitly.

After editing with `session.replace_proposal(..., expected_revision=...)`, the old report is invalidated. Call `prepare()` again. Before using a retained artifact, call `require_current()` with the actual current server-side session snapshot. Checking a saved old snapshot against itself does not detect a subsequent edit. Later construction/acceptance adapters must enforce this boundary themselves; possession of a prepared record does not authorize acceptance.

## Authority and baseline contract

The ontology system supplies the generated `MODEL`, its matching manifest, a stable schema ID and the accepted base revision. These are trusted server configuration, not fields chosen by the agent. The adapter:

- Verifies the closure SHA-256 against the manifest and confines the path to the generated directory.
- Checks that registry classes and property kinds are declared in that closure.
- Captures schema and additional accepted RDF facts so later caller mutations do not alter its baseline.
- Rejects proposals whose schema fingerprint, schema ID or accepted revision differs from the adapter's context.
- Verifies the proposal digest, including snapshots constructed directly instead of through `CandidateSnapshot.build()`.

These checks do not prove that every generated Python implementation detail matches the ontology. Generated classes, inheritance and descriptors remain trusted compiler output. The fingerprint identifies exact closure bytes, not semantic RDF equivalence.

`accepted_graph=` is optional and supplies **additional A-Box facts**, rather than another copy of the source ontology. Individuals already in the compiler closure are automatically included. Pass a separate graph of accepted instance assertions when needed. A graph containing schema declarations is rejected with `accepted_schema`; this prevents duplicating blank-node schema structures by merging two separately parsed copies of the ontology. Blank nodes in additional A-Box metadata are copied into a separate identity scope during RDF merge.

`base_revision` is checked against this captured adapter context. There is no accepted graph store or atomic concurrency check yet. Recreate the adapter when accepted state changes. Phase 16 must verify the current stored revision at commit time.

## Supported request behavior

| Request or condition | Phase 12 behavior |
| --- | --- |
| Create named entity with a generated class | Add explicit RDF type assertions |
| Multiple asserted types | Existing planner requires a unique most-derived generated type compatible with every asserted type; unrelated types are unsupported |
| Local reference | Resolve within the same complete proposal, including forward references |
| IRI reference | Must identify an explicitly typed baseline individual; use a local reference for a new proposal entity |
| Explicit new IRI | Reject collision with any baseline named term or another proposed entity |
| Omitted new IRI | Allocate `urn:krrood-ontology:candidate:<session-id>:<local-id>` |
| Object assertion | Check property kind, field binding, explicit subject domain and explicit target range |
| Literal assertion | Check property kind, field binding, declared datatype and RDFLib lexical conversion |
| Duplicate RDF assertion | Graph-set semantics; an assertion already in the baseline is not an addition |
| Existing identities | Reuse the `distinct` planner profile for declared `differentFrom` and `AllDifferent`; no unique-name assumption or `sameAs` merging |
| Unsupported or invalid baseline | Report it rather than silently skipping existing facts |
| Successful preflight | Return immutable candidate triples, new triples and local-ID/IRI pairs |
| Failed/unsupported preflight | Return feedback and **no** prepared candidate |

Provisional IRIs are stable for a given local ID within a session across revisions, and separate between different sessions. They are not a permanent accepted-IRI allocation policy. An explicit IRI is included in the proposal digest. A later accepted-identity policy must preserve or explicitly remap identities and references.

Literal lexical forms use RDFLib `normalize=False`, so a valid integer such as `"001"^^xsd:integer` retains that spelling in candidate RDF while preflight checks its Python conversion. Language-tagged field values remain subject to the existing loader's limitations. Phase 12 introduces no annotation creation operation, new identity operation, deletion or retraction request.

## Meaning of pass and feedback

Preflight is an **object-model admissibility** check with profile `asserted_named_individuals_v1`. In OWL, a domain/range assertion can infer a type; this planner instead requires compatible explicit types for construction. A rejection is therefore not necessarily proof that the OWL ontology is inconsistent.

Cardinality, disjointness, functional/asymmetric conflicts, existential witnesses, inferred types and broader OWL consistency are not executed by this adapter. The previous `ModelValidator` still exists through its earlier API but is not automatically run here. Construction, local validation and reasoning each report `unsupported`/`not_available`. The default Phase 11 acceptance policy continues to block acceptance. A trusted custom policy requiring only preflight can prepare an intent, but cannot commit anything and must not be described as reasoner approval.

An ontology property with no applicable generated binding is reported as unsupported in this projection. Request failures carry stable codes and JSON-pointer locations when known. Planner feedback is fail-fast. Because the existing planner identifies an IRI rather than an exact assertion, operation attribution is supplied only when unambiguous; findings caused by the baseline or several possible operations have an empty location. The original planner diagnostic still includes the relevant IRI. This phase does not guess an offending request or claim a complete logical explanation. Unexpected execution failures are converted by `CandidateSession.check()` to Phase 11 execution feedback; direct `prepare()` calls allow unexpected backend exceptions to propagate.

## Next phase

Phase 13 will materialize the validated candidate's Python objects and maintain their RDF identity correspondence in an isolated construction boundary. It must reuse the existing loader/monitored collections/native descriptors while addressing KRROOD's global graph lifecycle. Reasoner execution remains Phase 14, and accepted-state mutation remains Phase 16.
