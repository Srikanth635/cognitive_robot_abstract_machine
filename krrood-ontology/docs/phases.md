# Phased implementation

The final goal is a local ontology file as input and a usable Python domain
package as output, with a loader for individuals and clearly specified semantic
behavior. Phases 1–10 are implemented for their documented profiles. Phase 2 produces real
dataclasses and KRROOD object-property registration for the documented structural
subset. Phase 3 implements a bounded additive relation profile through KRROOD.
Phase 4 loads named, explicitly typed individuals. Phase 5 verifies generated
imports and individual preflight, resolves supplied local imports and reports
feature treatment and provenance. Phase 6 maps supported anonymous class
expressions to positive scoped EQL queries. Broader OWL remains future work.

| Phase | Accomplishment / acceptance criterion | Modules to add | Reuse |
| --- | --- | --- | --- |
| 1 — Input and baseline | Parse local RDF, inspect assertions, produce a package that imports and a provenance report | Delivered `config`, `ontology`, `inspection`, `backend`, `bootstrap`, `cli` | RDFLib; KRROOD source rendering and formatting |
| 2 — Structural object model (complete) | Generate instantiable domain classes and typed fields for a documented supported subset; preserve IRI identity; import and exercise generated objects | Delivered `model.py`, `translation.py`, `generation.py`, `compiler.py`, class/property templates | KRROOD `Symbol`, naming helpers, `PropertyDescriptor`, generator and formatter |
| 3 — Supported semantics (complete for the documented relation profile) | Native inverse, symmetric, transitive and subproperty implications; EQL queries over generated models; reject excluded mappings | Delivered `semantics.py`, extended diagnostics, relation/axiom templates | KRROOD inverse/transitive mixins, descriptor inheritance, monitored sets and EQL |
| 4 — Individuals (complete for explicit inheritance-chain types) | Load an A-Box into generated types; resolve shared references; convert literals; preserve annotations | Delivered `individuals.py`, `runtime.py`, generated runtime template | RDFLib literal conversion; generated classes; KRROOD descriptors; roles deferred because their distinct identity does not fit this policy |
| 5 — Compiler product (complete for the bounded profile) | Supplied local import resolution, isolated package verification, strict individual preflight, source snapshots and feature/environment reports | Delivered `imports.py`, `features.py`, `verification.py`, `verification_worker.py`, expanded compiler/CLI | Earlier phases; KRROOD formatting; RDFLib canonicalization/serialization; Python import tools |
| 6 — Positive class expressions (complete for the documented query profile) | Intersections, unions, named enumerations, value restrictions and selected object existential restrictions | Delivered `expressions.py`, `queries.py`, expression records and query template | Native KRROOD EQL logic, membership, type predicates and quantified queries |
| 7 — Generic structural compatibility (implemented local-usage profile) | Keyword-safe names; tested multiple inheritance; absent domain/range binding policy; anonymous superclass constraint records; importable full AICOR structural model | Extend existing model, translation, expressions, generation and verification modules | Symbol, native descriptors and their association maps, dataclasses, current generator |
| 8 — Class-local scoped validation (implemented bounded profile) | Inherited existential/value/universal and qualified cardinality constraints; functional and disjointness checks with declared scope | Extend expressions/queries; add validation adapter and reports | Native EQL predicates, quantification and counting where compatible |
| 9 — Definitions and property implications (implemented bounded scoped profile) | AICOR union/enum definition queries; nested fillers; generalized native transitive/symmetric mappings; explicit reflexivity/asymmetry policies | Delivered scoped.py and relation_runtime.py; extended queries/validation and templates | Phase 6 EQL and Phase 3 native relation machinery |
| 10 — Identity, round trip and AICOR acceptance (implemented bounded profile) | Generic differentFrom/AllDifferent preflight; four fixture individuals; source/current RDF export and full-graph round-trip acceptance; unrelated-type views remain deferred | Delivered identity.py, export.py and shared values.py; extended loader, validator and runtime template | Current registry/planner; RDFLib; native KRROOD identity and relation snapshots |

## AICOR-driven priority update

The supplied real ontology changes the order of future work: structural binding
and inheritance must precede validation. See [aicor-assessment.md](aicor-assessment.md)
for measured coverage, gaps, semantic cautions and acceptance criteria. Phases
1–10 are delivered for their documented profiles.
See [phase-10.md](phase-10.md) for identity and source/current RDF export.
See [phase-9.md](phase-9.md) for positive definitions and native relation additions.
See [phase-8.md](phase-8.md) for the scoped validation contract.
See [phase-7.md](phase-7.md) for the implemented contract and known limits.

## Phase 1: completed boundary

```bash
krrood-ontology bootstrap ontology.owl --output generated --package my_model
```

Produces `my_model/__init__.py`, `my_model/vocabulary.py`, and `report.json`.
It proves input handling, real KRROOD source generation, formatting, and output
imports. It does not label vocabulary constants as domain classes.

## Phase 2: structural conversion (complete)

Start with a small, deliberately specified subset:

1. Named classes and acyclic named subclass relationships, initially a single
   named parent per class. Test inheritance rather than assuming every OWL
   subclass statement can be represented as Python inheritance.
2. Explicit named object/data properties with supported named domains and
   ranges. Multiple domains/ranges have intersection semantics in OWL; do not
   reinterpret them as union types. Missing, anonymous, or multiple domain/range
   cases must receive diagnostics until there is a tested policy.
3. A small documented XSD datatype table, with diagnostics for other datatypes.
4. Collection-valued fields as the conservative initial operational shape.
   Neither domain/range axioms nor missing values imply required Python fields.
5. A collision-checked IRI-to-Python-name map. Reuse KRROOD case conversion,
   and add only the identifier/collision policy it does not provide.

`model.py` contains minimal dataclasses describing generation decisions. It is
not a second ontology storage engine: the RDFLib graph remains the asserted
source. `translation.py` derives these records from the graph; `generation.py`
feeds them into our templates through the existing backend.

Generated field shapes and descriptor integration are now tested against real
KRROOD implementations. Classes are generated first, then descriptor declarations
and registration in `relations.py`; package initialization loads the registration
module. Literal fields remain ordinary dataclass fields because the existing
descriptor requires a `Symbol` range.

Acceptance example:

```python
from my_model import Person, Student, Course

alice = Student()
course = Course()
alice.takes_course.append(course)
assert isinstance(alice, Person)
```

This API now works in Phase 2. The complete supported subset and mapping policy
are documented in [phase-2.md](phase-2.md).

## Phase 3: semantic behavior (implemented relation profile)

See [phase-3.md](phase-3.md) for the delivered subset, required signatures and
native runtime limits. Relations participating in the supported axioms use
monitored sets initialized empty and populated after object construction.
Normal Phase 2 properties retain list-valued fields. Existing diagnostic records
are reused; no separate diagnostic mechanism or inference engine was added.

The delivered EQL example queries objects including their derived relationships;
it does not translate arbitrary class axioms into EQL or infer class membership.

Inverse and transitive behavior is demonstrated through object additions and
queries using the upstream implementations. Future query-oriented axiom
translation needs separately defined and tested operational semantics.

EQL query behavior is not automatically OWL open-world entailment. A cardinality
check over known objects must not be described as a complete OWL cardinality
reasoner. Preserve the source axioms and report unsupported cases explicitly.
If OWL RL closure is selected later, use the existing `owlrl` dependency with a
named inference profile; do not describe it as full OWL DL reasoning.

Strict mode must fail when a requested semantic conversion cannot be honored.
Any structural-only mode must report its omissions in the output manifest.
Property chains, complements, unions, anonymous restrictions, `sameAs`, and
other constructs are feature decisions requiring evidence, not inferred support
from the existence of a generic code generator.

## Phase 4: A-Box loading

The delivered API and exact boundary are documented in [phase-4.md](phase-4.md).
One object per IRI is supported for explicit types forming an inheritance chain.
Unrelated types and identity axioms fail with diagnostics. The generated
`load_individuals` function handles both combined ontologies and separate A-Boxes.

The loader preflights assertions, then creates objects and populates references
in two passes. Supported literal conversion delegates to RDFLib. KRROOD roles
were inspected and deferred for unrelated types because their composition and
distinct identities do not satisfy this loader's contract. Identity normalization,
anonymous individuals and broader multiple-type cases need additional policies
and tests.

Keep the loader separate from T-Box source generation. Large A-Boxes should
be data loaded through this API, not millions of generated Python assignments.

## Phase 5: final black box

The delivered contract and test commands are in [phase-5.md](phase-5.md).
Coverage reporting describes the observed terms and their treatment; it does
not claim complete OWL profile conformance.

The structural command already works; Phase 5 hardens it for the supported
semantic/runtime scope:

```bash
krrood-ontology compile ontology.owl --output generated --package my_model
```

The delivered output layout is:

```text
generated/
├── manifest.json
├── features.json
├── source/closure.ttl
├── source/documents/
└── my_model/
    ├── __init__.py
    ├── vocabulary.py
    ├── classes.py
    ├── relations.py
    ├── axioms.py
    └── runtime.py
```

Isolated import verification, same-environment reproducibility checks,
supported-subset fixtures, an environment record and explicit local-file import
resolution are now implemented. Feature reporting distinguishes executable
mapping from retained source content and explicit omissions. Broader OWL
conformance and a complete dependency lock remain additional work.

ORMatic persistence can be a subsequent optional integration. It is not required
to establish the ontology-to-module boundary, and should reuse KRROOD's DAO
generation rather than introduce a second persistence mechanism.

## Remaining improvements after Phase 10

The documented profiles now support full-source retention and a bounded operational projection. Further work includes `sameAs` normalization with explicit alias semantics, unrelated multiple-type views, new/deleted-individual lifecycle policies, anonymous individuals, property chains and broader OWL classification. None of these are implied by round-trip source preservation. Native KRROOD additions still do not provide a general retraction/recomputation contract.

## Ontology-system service plan: Phases 11–19

The ontology is authoritative. A separate agent requests entity/relation creation and receives review feedback. Concept-selection accuracy is outside the current implementation scope. Phases 1–10 remain the compiler/runtime foundation; Phase 11 is the service-contract foundation.

| Phase | Accomplishment | Dependencies / reuse |
| --- | --- | --- |
| 11 — Contracts and candidate lifecycle (implemented) | Immutable versioned proposals and feedback, schema/base context, revision-bound checks and acceptance intents; explicit unsupported execution defaults | Standard dataclasses; existing compiler closure provenance; see phase-11.md |
| 12 — Candidate graph and preflight (implemented) | Prepare revision-bound candidate RDF, resolve references and check object-model admissibility without native or accepted-state mutation; see phase-12.md | Phase 11 records; existing model mappings, individual planner and RDFLib |
| 13 — Python–RDF construction bridge (implemented) | Materialize actual typed objects in an ephemeral worker, preserve shared references, verify asserted RDF and return immutable revision-bound evidence; see phase-13.md | Native KRROOD Symbols, descriptors, monitored collections; existing loader/export; no accepted-state mutation |
| 14 — Reasoner adapter (implemented) | Review exact constructed candidate RDF with explicit `owlrl` diagnostics or optional HermiT/OWLAPI; report consistency, profile/parser limits, failures and timeouts; see phase-14.md | Existing OWL engines, OWLAPI profile checker, JPype; Phase 13 isolation and local closure; no acceptance |
| 15 — Unified feedback and revision (planned) | Combine native validation, profile/parser diagnostics and reasoner findings into actionable candidate/source feedback with explicit attribution limits | Phase 11 revisions; existing validation reports; Phase 14 evidence and AICOR profile/loader findings |
| 16 — Acceptance and synchronization (planned) | Accept only the exact reviewed revision against the current accepted base; maintain object/RDF projections | Current session/report checks; explicit acceptance handler and provenance |
| 17 — Agent-facing schema and interface (planned) | Catalogue, constrained-output schemas and service/client adapters | Existing vocabulary/mappings and stable requests; no reasoner internals in agents |
| 18 — Persistence and querying (planned) | Durable accepted graphs, reconstruction, supported queries and provenance retrieval | RDFLib, native EQL and replaceable storage adapters |
| 19 — End-to-end acceptance and hardening (planned) | Cross-ontology creation, feedback, repair, acceptance and retrieval; malformed RDF schema and RDF-to-OWL parser conformance cases | Earlier contracts/adapters; generic fixtures plus AICOR integration; explicit supported profiles |

Phase 11 does not claim to execute ontology checks or commit facts. Generated compiler manifests continue to identify Phase 10's compiler profile. The service contract is versioned separately from the package release.
