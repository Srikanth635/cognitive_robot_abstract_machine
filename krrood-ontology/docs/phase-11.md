# Phase 11: ontology-system contracts and candidate lifecycle

Phase 11 introduces a generic boundary between an agent and the ontology system. It provides immutable proposal records, schema/base-revision context, check reports, acceptance requirements and revision-aware candidate sessions. These records are independent of ontology names, LLM providers, reasoners and graph stores.

This phase is groundwork. It does not instantiate generated entities, produce candidate RDF, execute local/OWL validation or commit accepted facts. Existing Phase 1–10 functionality continues to work through its current APIs. Request preflight, construction and reasoning will be wired to the new boundary in later phases.

## Install and test

From the extracted `krrood-ontology` directory, with your KRROOD environment active:

```bash
python -m pip install --no-deps -e .
python -m krrood_ontology --version
python -m pytest tests/test_contracts_sessions.py -q
python examples/verify_phase11.py
python examples/verify_phase11.py examples/aicor_generated
python -m pytest tests -q
```

Expected version: `0.11.0`. The focused suite has 24 tests. The complete suite has 154 tests. Install pytest separately if necessary; a fresh environment can use `python -m pip install -e '.[dev]'` with the existing pinned KRROOD dependency.

The example's passing adapter is named `ContractFixtureChecker` and reports `contract_fixture` profiles. It exercises the lifecycle only. Its output explicitly says no ontology reasoning occurred and no facts were committed. The default adapter reports all execution stages as unsupported and blocks acceptance preparation.

The optional generated-output argument verifies the compiler closure fingerprint. It does not import the generated classes or check that proposal terms belong to that ontology. The example remains a contract test even when pointed at AICOR.

## New modules and dependencies

| Module | Role |
| --- | --- |
| `contracts.py` | Versioned immutable requests, references, literal metadata, review records and acceptance requirements |
| `sessions.py` | Service boundary, candidate revision lifecycle and trusted checker protocol |
| `examples/verify_phase11.py` | Explicitly labelled contract-only lifecycle demonstration |
| `tests/test_contracts_sessions.py` | Record validation, stale-review, isolation and acceptance-blocking tests |

The contracts use standard Python dataclasses, enums, hashing, JSON-ready dictionaries and locks. They do not duplicate KRROOD object storage, inference, query or serialization engines. No new dependency is required. Future construction adapters will reuse the existing planner, mappings, descriptors and RDF export. Reasoner adapters will use an existing reasoner.

Generated classes and templates were not changed. Regeneration is unnecessary to use these shared modules. The compiler's manifest `phase` remains **10**, recording the unchanged compiler feature level; distribution `0.11.0` adds the independent service-contract layer. Earlier generated examples keep their recorded generation environment.

## Authority context

```python
from krrood_ontology.contracts import CandidateContext

context = CandidateContext.from_manifest(
    "examples/aicor_generated/manifest.json",
    schema_id="urn:my-project:ontology",
    base_revision="accepted-0",
)
```

`schema_id` is an explicit absolute IRI supplied by the ontology system. It need not be derived from Python package names. The helper reads the existing compiler manifest, checks that the RDF closure stays inside the generated output directory, and verifies its SHA-256 against the recorded provenance.

`schema_sha256` identifies those exact closure bytes. It is not an RDF semantic-equivalence hash; blank-label/serialization changes can produce another fingerprint. `base_revision` identifies the accepted state on which the proposal depends. Phase 11 carries this value but does not load or verify an accepted graph store. Phase 16 must compare it against the actual accepted state before committing.

## Proposal records

```python
from krrood_ontology.contracts import (
    CreateEntity, EntityReference, LiteralAssertion,
    LiteralValue, ObjectAssertion, Proposal,
)

proposal = Proposal((
    CreateEntity("create-person", "person", ("urn:example:Person",)),
    CreateEntity("create-place", "place", ("urn:example:Place",)),
    LiteralAssertion(
        "set-name", EntityReference(local_id="person"),
        "urn:example:name", LiteralValue("Alice"),
    ),
    ObjectAssertion(
        "lives-at", EntityReference(local_id="person"),
        "urn:example:livesAt", EntityReference(local_id="place"),
    ),
))
```

Every operation has a unique `request_id`, usable in diagnostics. `CreateEntity` has a proposal-local identifier, requested class IRIs and an optional desired individual IRI. Allocation and collision checks belong to the later entity-creation policy. A reference selects exactly one local identifier or existing individual IRI.

`LiteralValue` stores lexical text and optional datatype or language metadata. It preserves text such as `"01"`; this phase does not convert or validate datatype lexical values. Omitting datatype/language describes a plain literal for the later RDF adapter. Multiple class IRIs can be represented, but their compatibility is not established by these records.

Record checks cover tuple immutability, basic absolute-IRI shape, identifier uniqueness, reference shape, metadata compatibility and supported contract version **1.0**. Ontology membership, reference existence, signatures, cardinality, class compatibility and reasoner consistency remain unchecked here. Unknown terms can form a syntactically valid proposal; the default checker reports unsupported execution.

`proposal.to_dict()` and `report.to_dict()` provide JSON-ready envelope dictionaries through the standard dataclass conversion. Untrusted JSON parsing, constrained-output schema generation and transport adapters are Phase 17 work. The API currently accepts typed Python records.

## Candidate lifecycle

```python
from krrood_ontology.sessions import OntologySystem

system = OntologySystem(context)
session = system.begin_candidate(proposal)
report = session.check()  # explicit unsupported feedback with the default adapter

revised = Proposal(proposal.operations[:3])
snapshot = session.replace_proposal(revised, expected_revision=1)
assert snapshot.token.revision == 2
assert session.report is None
```

A candidate begins at revision 1. Its token contains a unique session ID, revision and digest of the immutable proposal plus schema/base context. Proposal replacement always creates a new revision, even if supplied content is identical. Operation order and literal lexical forms participate in the digest; no semantic canonicalization is claimed.

| State | Meaning |
| --- | --- |
| `draft` | No retained review for the current candidate |
| `reviewed` | The current revision has a report, including failed/unknown/unsupported outcomes |
| `ready_for_acceptance` | Configured review conditions passed and an acceptance intent was prepared |
| `closed` | Further checking and revision are prohibited |

Every edit invalidates the retained review. A new check also invalidates prior success before execution. The adapter runs outside the session lock, so a caller can replace or close a candidate while checking is in progress. A late result for an edited/closed/rechecked candidate is rejected with `stale_review`; a foreign-session report is rejected with `review_token`.

Execution exceptions become generic execution feedback without exposing raw backend error details. Malformed or foreign reports raise `LifecycleError`. Phase 15/17 adapters will translate these outcomes into the full service response/transport policy. The session retains the current snapshot/report, not a durable revision history.

Sessions do not own accepted graphs or create native Symbols. The tested isolation is **contract-state isolation**, not a proven transaction boundary for future KRROOD construction. Phase 13 must verify the appropriate native isolation strategy.

## Checker and acceptance contracts

`CandidateChecker.check(snapshot)` is a trusted, configured ontology-system adapter. It returns a `CheckReport` bound to the supplied token. The agent does not submit its own successful review or select the checker through proposal fields.

Four stages are identified:

1. `preflight` — request and ontology-mapping compatibility.
2. `construction` — supported Python/RDF materialization.
3. `local_validation` — checks under a stated local validation profile.
4. `reasoner` — review under a stated ontology reasoning profile.

Each result includes `pass`, `fail`, `unknown`, `unsupported` or `error`, a profile identifier and structured feedback. A reasoner-stage pass is intended to mean consistency under the configured profile; Phase 11 itself does not verify such a claim. The trusted reasoner adapter and capability controls are Phase 14 work. A passing stage cannot carry error-severity feedback.

Feedback distinguishes request, local-validation, reasoner, unsupported, execution and lifecycle categories. It can identify a JSON-pointer location, entity reference, property IRI and axiom identifier when available. This layer does not invent reasoner explanations.

The default acceptance policy requires all four stages under a placeholder configured profile. An installed service must specify actual profiles deliberately. The fixture example uses an explicit `contract_fixture` policy, so it cannot accidentally satisfy the default production policy.

```python
# With a real, configured adapter in a later phase:
report = session.check()
intent = session.prepare_acceptance()
```

`prepare_acceptance()` blocks missing, wrong-profile, nonpassing required checks and error feedback. The resulting `AcceptanceRequest` contains the exact token, authority context and required profiles. **It is an intent, not a commit or signed authorization token.** There is no `accept()` implementation in this phase.

The future acceptance handler must recheck the session's current report/state, configured policy, revision and accepted base revision. An old intent must not be accepted after replacement, closing, a failed recheck or a changed accepted graph. Existing facts remain untouched until that later explicit acceptance implementation.

## Next phase

Phase 12 will prepare candidate RDF and perform ontology-aware request preflight using these contracts and existing mappings/planner facilities. Phase 13 will materialize Python objects through KRROOD. Phase 14 will invoke a configured reasoner; Phases 15–16 will complete refinement and acceptance. The agent-facing catalogue, constrained schemas and service transport remain Phase 17.
