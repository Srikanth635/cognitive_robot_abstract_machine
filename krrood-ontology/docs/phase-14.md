# Phase 14: ontology reasoner adapter

Phase 14 connects successful Phase 13 construction to an explicitly selected ontology reasoner. The checker reviews the same revision-bound candidate RDF, returns structured consistency/capability feedback, and leaves accepted RDF and the parent's native graph unchanged. No ontology names appear in the shared implementation.

## Installation and testing

Run from the extracted `krrood-ontology` directory. With your existing KRROOD environment active:

```bash
python -m pip install --no-deps -e .
python -m pip install 'owlready2==0.51' 'JPype1==1.6.0'
java -version
python -m krrood_ontology --version
python -m pytest tests/test_reasoning.py -q
python examples/verify_phase14.py examples/generated university_model --backend hermit
python examples/verify_phase14.py examples/aicor_generated aicor_model --backend hermit
python examples/verify_phase14.py examples/generated university_model --backend owlrl
python -m pytest tests -q
```

Expected package version: `0.14.0`. All 247 tests passed in the verified environment, including 36 Phase 14 tests and real HermiT/JVM execution. A fresh environment can use `python -m pip install -e '.[dev,reasoner]'`, which installs the pinned KRROOD source, pytest, and the optional reasoner dependencies. A Java runtime is also required for HermiT; this release was tested with Java 17. On Debian/Ubuntu, an available `openjdk-17-jre-headless` installation supplies that runtime. JPype discovers the JVM shared library; `JAVA_HOME` or the backend's `jvm_path=` can select it when discovery fails. `jvm_path` names the shared library, not the `java` command.

The `reasoner` extra is optional. Existing Phases 1–13 and `owlrl` diagnostics work through the original KRROOD dependency. Real HermiT tests skip when its optional Python packages or a discoverable JVM are absent; installation alone is not proof that reasoning ran. Inspect the reported status and profile.

Generated modules do not need regeneration. Fresh compiler manifests still identify compiler Phase 10; the service package release is separately `0.14.0`.

## What is reused

| Component | Existing machinery used |
| --- | --- |
| Proposal, revision and review | Phase 11 contracts and sessions |
| Candidate authority and asserted RDF | Phase 12 `CandidatePreparer` and compiler import closure |
| Actual Python objects and RDF correspondence | Phase 13 `CandidateConstructor`, original loader/export, native KRROOD Symbols/descriptors/collections |
| Rule diagnostics | `owlrl.OWLRL_Semantics`, already a KRROOD dependency |
| OWL 2 DL profile checking | OWLAPI's `OWL2DLProfile.checkOntology()` |
| OWL consistency | Existing HermiT `Reasoner.isConsistent()` |
| Java integration | Existing JPype JVM bridge |

KRROOD does not expose a full OWL 2 DL consistency checker in the pinned package inspected for this release. Its native relation implications and object checks remain useful and unchanged. The optional backend uses the HermiT/OWLAPI JAR supplied by Owlready2; it does **not** create a second Owlready2 Python object model or call its reclassification API. Candidate RDF is serialized with RDFLib and loaded directly by OWLAPI. No new OWL rule engine, tableau algorithm, profile checker, or explanation engine was written.

New implementation files are `reasoning.py` (capabilities, immutable evidence, isolated backends and checker adapter) and `reasoning_worker.py` (calls into existing engines). `tests/test_reasoning.py` and its generic RDF fixtures cover actual reasoning and lifecycle failures. `examples/verify_phase14.py` selects concepts/properties through existing model metadata for any compatible generated package.

## Profiles and status meaning

| Backend/profile | Result meaning |
| --- | --- |
| Default `UnavailableReasoner`, `not_available` | `unsupported`: no engine selected or executed |
| `ReasonerKind.OWL_RL`, `owlrl_diagnostics_v1` | Existing OWL RL/RDF rule diagnostics; reported clash gives `fail`; no clashes gives `unknown` |
| `ReasonerKind.HERMIT`, `hermit_owl2_dl_consistency_v1` | OWLAPI-loaded axioms pass its OWL 2 DL profile check and HermiT returns an OWL 2 DL consistency result |

The HermiT profile uses open-world semantics and no unique-name assumption. Two different IRIs do not automatically mean two logically different individuals. Functional properties or maximum cardinality one can allow equality between targets; adding explicit `differentFrom` can make the same assertions inconsistent. An existential requirement does not fail merely because no witness was asserted: a witness may exist without being named in the supplied graph. The tests verify both behaviors.

An unsatisfiable class with no instances can exist in a consistent ontology. This phase checks ontology consistency, not a requirement that every named class be satisfiable. It does not return inferred classifications, create witnesses, merge Python identities, or materialize reasoner inferences. Native KRROOD objects remain in their generated asserted types.

The `owlrl` backend deliberately cannot return a consistency `pass`. Its existing implementation has a bounded rule/ datatype profile, and a clash-free run is not a complete proof for arbitrary ontology input. There is no silent fallback from HermiT to `owlrl`. A missing backend, Java runtime, unsupported datatype, deadline or execution failure produces its own non-passing outcome.

## Service use

```python
import sys
from pathlib import Path

directory = Path("examples/generated").resolve()
sys.path.insert(0, str(directory))
import university_model as generated

from krrood_ontology.construction import CandidateConstructor
from krrood_ontology.contracts import CreateEntity, Proposal
from krrood_ontology.preparation import CandidatePreparer
from krrood_ontology.reasoning import CandidateReasoner, IsolatedReasoner, ReasonerKind
from krrood_ontology.sessions import OntologySystem

preparer = CandidatePreparer.from_manifest(
    generated.MODEL, directory / "manifest.json",
    schema_id="urn:my-project:ontology", base_revision="accepted-0",
)
constructor = CandidateConstructor(preparer, directory, "university_model")
backend = IsolatedReasoner(
    ReasonerKind.HERMIT, timeout_seconds=90, max_heap_mb=512,
)
checker = CandidateReasoner(constructor, backend)
service = OntologySystem(preparer.context, checker=checker)
session = service.begin_candidate(Proposal((
    CreateEntity("create-student", "student", (
        "https://example.org/university#Student",
    )),
)))
result = checker.review(session.snapshot)
print(result.report.to_dict())
if result.evidence is not None:
    result.evidence.require_current(session.snapshot)
    print(result.evidence.status, result.evidence.consistent)
    print(result.evidence.profile_findings)
    print(result.evidence.unparsed_triples)
    print(result.evidence.engine_version)

# Stores a fresh report through the existing candidate lifecycle:
review = session.check()
# The default policy continues to block session.prepare_acceptance().
```

Only this example names university vocabulary. Service implementation and the demonstration's selection logic use the captured model registry, not ontology-specific classes.

`review()` constructs exactly once for that call, then gives the backend the construction artifact's exact `PreparedCandidate`. It returns `ReasoningResult(report, construction, evidence)`. `check()` implements the existing `CandidateChecker` protocol and returns a report. Calling it again executes fresh construction and reasoning; no mutable last-result cache is retained. Direct `backend.evaluate(prepared)` is a trusted server-level API, not an agent upload endpoint. It does not authorize arbitrary RDF or prove authority for manually fabricated prepared records.

`ReasonerEvidence` retains immutable RDF terms, typed capabilities, status, optional consistency boolean, feedback, engine/provider/JVM version, HermiT JAR SHA-256, worker PID, loaded axiom count, OWLAPI profile findings and reported unparsed RDF triples. Graph access is defensive. A `pass` requires a true consistency conclusion and a backend that can certify consistency under its declared semantics; a diagnostic backend cannot claim this.

The new evidence belongs to one candidate token/context. Editing a proposal invalidates its review and old evidence. Call `require_current()` with the actual current server-side session snapshot before using retained evidence. Checking against an old saved snapshot cannot detect a later edit. The existing session rejects reviews completed after a concurrent revision change.

## Failure and capability feedback

| Situation | Outcome/code |
| --- | --- |
| Preflight or construction did not pass | `unknown`, `construction_required`; reasoner not started |
| Backend not configured | `unsupported`, `reasoner_not_configured` |
| Optional Python packages missing | `unsupported`, `reasoner_dependency_missing` |
| HermiT bundle unavailable | `unsupported`, `reasoner_bundle_missing` |
| JVM shared library missing | `unsupported`, `reasoner_java_missing` |
| Unresolved `owl:imports` in input | `unsupported`, `reasoner_unresolved_import`; no engine started |
| OWLAPI profile violation | `unsupported`, `owl2_dl_profile`; original profile finding retained |
| OWLAPI reports unparsed RDF | `unsupported`, `reasoner_unparsed_rdf`; those triples retained as diagnostic strings |
| HermiT cannot evaluate a datatype | `unsupported`, `reasoner_datatype_unsupported`; no ignore-datatype option enabled |
| HermiT reports inconsistency | `fail`, `ontology_inconsistent`, `consistent=false` |
| Deadline | `error`, `reasoner_timeout`, no consistency conclusion |
| Engine/process failure | `error`, `reasoner_failed`, `reasoner_execution` or `reasoner_worker` |
| Foreign revision/profile returned | `error`, `reasoner_protocol` |

Input must contain the already-resolved local import closure. Reasoner execution does not retrieve imports from the network. A test confirms that disjointness from a supplied local import participates in the consistency check.

HermiT is embedded through JPype **inside** the reasoner worker. The parent process starts no JVM and retains no Java objects. The existing subprocess deadline kills and waits for the whole worker/JVM process. A fault fixture starts a real JVM, blocks, and verifies that the worker PID no longer exists after timeout. Private transfer directories are cleaned up. The reasoner deadline covers the worker's dependency loading, RDF parsing, profile check and reasoning; preflight/construction have their own boundaries. Generated code and reasoner dependencies remain trusted server software; subprocess isolation owns state and time limits rather than providing an untrusted-code security sandbox.

Worker IPC reuses Phase 13's private transfer helper, including its per-pickler RDFLib literal hook. RDF literal spelling is preserved in asserted candidate RDF. RDFLib serialization and OWLAPI's parser supply the engine projection; source RDF is never overwritten.

## Demonstration results and AICOR findings

The university demonstration reports `preflight=pass`, `construction=pass`, `reasoner=pass`, `consistency=true` and 18 loaded axioms for the included candidate. Its `owlrl` demonstration reports `reasoner=unknown`, `consistency=null`, with `owlrl_no_clashes`.

For the supplied AICOR model, object construction still passes. The HermiT adapter reports `reasoner=unsupported`, `consistency=null`, with two existing-input findings:

1. OWLAPI's profile checker reports `InsufficientOperands` for:

   ```text
   EquivalentClasses(
     <http://aicor.knowledge/l2-ontology.owl#PlanTransformation>
     ObjectUnionOf(<http://aicor.knowledge/l2-ontology.owl#CodeReplace>)
   )
   ```

   This is a single-member union, rather than a union with at least two operands. The adapter preserves and reports the axiom. It does not rewrite it into another axiom.

2. The bundled OWLAPI parser reports an unparsed blank-node `rdf:type owl:AllDifferent` marker. This is a loader diagnostic, not a proof of logical inconsistency. Its generated blank-node label can vary between runs. The adapter conservatively withholds approval instead of discarding it silently.

Neither finding is evidence that the ontology is logically inconsistent. They prevent this backend from certifying this exact input. The same generic checks found them; no AICOR-specific logic was added. Native construction and source round-trip success in earlier phases remain separate accomplishments from OWL consistency approval. Saved actual results are in `docs/phase-14-*-results.json`.

## Limits and next steps

A HermiT `pass` concerns OWLAPI's loaded OWL 2 DL axioms. OWLAPI's parser/profile check is not a comprehensive validator of arbitrary raw RDF schema syntax: malformed structures can be normalized or omitted without every omission appearing in residual-triple metadata. Zero reported unparsed triples alone is not proof of exact semantic translation for arbitrary RDF. This adapter is intended for valid OWL RDF input; raw RDF-to-OWL mapping conformance and broader loader interoperability remain explicit hardening work. No alternate parser, ontology repair algorithm or OWL Full consistency guarantee is introduced here.

The HermiT profile returns a whole-candidate consistency result. It does not supply a minimal inconsistent axiom set or explain which operation caused a conflict. `ontology_inconsistent` therefore carries no invented request location. Profile findings include functional-syntax axiom text and a hash-derived OWLAPI axiom identifier when available; these are not the compiler's existing RDF axiom IDs. They must be mapped deliberately before attributing source axioms or agent requests. `owlrl` diagnostic messages are engine findings, not HermiT explanations.

Local object validation is still not automatically executed by this adapter. It remains `unsupported` in the report, and the default acceptance policy remains blocked even after HermiT passes. No facts, inferences, accepted identities or Python reclassification are committed.

Phase 15 will combine existing native validation, profile/parser diagnostics and reasoner feedback into actionable revision feedback, with explicit attribution limits. Source-authorized schema fixes or a compatible loader can then address AICOR's findings without hardcoding its vocabulary. Phase 16 will implement exact-revision acceptance and accepted object/RDF synchronization. Phase 19 hardening will include malformed RDF schema cases and RDF-to-OWL parser conformance. Agent-facing constrained schemas remain Phase 17.

## Primary references

- [Owlready2 reasoning documentation](https://owlready2.readthedocs.io/en/latest/reasoning.html): bundled HermiT engine and Java requirement.
- [OWLAPI OWL2DLProfile](https://owlcs.github.io/owlapi/apidocs_3/org/semanticweb/owlapi/profiles/OWL2DLProfile.html): existing ontology profile checker.
- [JPype API](https://jpype.readthedocs.io/en/latest/api.html): JVM discovery and embedding.
- [OWL-RL closure documentation](https://owl-rl.readthedocs.io/en/latest/Closure.html): existing rule engine and error messages.
