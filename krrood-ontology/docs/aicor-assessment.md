# AICOR ontology assessment and revised roadmap

Assessed `cram-aicor-ontology.ttl` against wrapper version 0.6.0 (Phases 1–6), using the supplied file and the local KRROOD source. This is an implementation assessment, not an OWL consistency or satisfiability proof. The ontology has not been modified. Upcoming phases below are planned, not implemented.

## Result

The existing compiler cannot yet generate this ontology. A strict compile rejects deferred semantics. A structural-only compile stops at the class IRI ending in `#None`, which is a Python keyword. Correcting that name alone will not resolve the property-signature, multiple-inheritance and anonymous-superclass gaps.

The first practical milestone should therefore be an importable structural AICOR package with faithful constraint metadata. Validation follows that milestone. The absence of an explicit domain/range is valid OWL and must not be treated as an ontology error.

## Inventory

Counts refer to distinct graph nodes or asserted triples as specified; they are not percentages of semantic coverage.

| Construct | Observed |
| --- | ---: |
| RDF triples | 11,596 |
| Declared named classes | 1,017 |
| Object properties | 811 |
| Datatype properties | 155 |
| Restriction nodes | 1,341 |
| Classes with multiple named parents, excluding owl:Thing | 259 |
| Classes with anonymous superclass expressions | 528 |
| Anonymous superclass links | 1,342 |
| Named individuals | 4 |
| owl:imports | 0 |

All 811 object properties lack explicit domains and ranges. All 155 datatype properties lack domains; 61 also lack ranges. The other 94 datatype ranges are already supported types: boolean (64), string (23), integer (5), decimal (2). Restrictions mention 776 object properties and 146 datatype properties. The remaining properties still need an IRI registry and an explicit operational access policy.

## What Phases 1–6 contribute

| Delivered phase | Reusable achievement | Limit exposed by AICOR |
| --- | --- | --- |
| 1: input and provenance | RDF parsing, inspection, names/IRIs and source records | Parsing succeeds; parsing does not mean semantic translation succeeds |
| 2: dataclasses | Symbol-backed classes, supported literal types, native object-property registration | One named parent and exactly one named domain/range per property are too restrictive; `None` needs a stable name override |
| 3: relations | Native symmetric/transitive/inverse/subproperty behavior for the bounded signature profile | Four transitive properties and one symmetric property exist, but lack the signatures required by the current adapter; functionality, asymmetry and reflexivity are deferred |
| 4: individuals | Two-pass identity-preserving loader and RDFLib literal conversion | Four individuals have the supported explicit type shape; the surrounding schema still cannot compile, and AllDifferent is unsupported |
| 5: compiler product | Feature diagnostics, isolated verification, snapshots, local imports | Valuable unchanged infrastructure; no imports are needed for this file, and structural-only cannot bypass structural incompatibilities |
| 6: expressions/EQL | Positive unions/intersections/oneOf/hasValue and selected someValuesFrom queries for anonymous equivalent-class definitions | All 1,341 restrictions here are subclass conditions, outside the admitted placement; 13 union definitions and one enumeration have reusable expression shapes after structural blockers are resolved |

There is no end-to-end AICOR generated model yet. Existing functionality is useful machinery, not evidence that the ontology's restrictions already execute.

## Semantic constructs requiring work

| Construct | Count | Planned treatment |
| --- | ---: | --- |
| someValuesFrom | 1,022 | Class-local necessary-condition metadata and scoped witness checks |
| hasValue | 242 | Required-value checks; no silent conversion into asserted dataclass defaults |
| qualifiedCardinality | 58 | Count qualifying fillers under an explicit validation scope |
| minQualifiedCardinality / maxQualifiedCardinality | 6 / 12 | Qualified minimum/maximum checks with identity policy |
| allValuesFrom | 1 | Check known fillers; cannot prove universal OWL entailment from incomplete data |
| unionOf / intersectionOf | 50 / 1 | Preserve expression structure and context; 36 unions occur as existential fillers, 13 as equivalent-class definitions, one as a superclass |
| equivalentClass | 14 | Thirteen union definitions and the TaskRelativeClass enumeration |
| FunctionalProperty | 119 | 25 object and 94 datatype properties; preserve axioms and implement scoped checks |
| disjointWith | 135 | Pairwise class disjointness metadata and membership conflict checks |
| AllDisjointClasses | 12 groups | Group disjointness checks |
| AllDifferent | 1 group | Preserve declared distinctness of the four named individuals |
| AsymmetricProperty / ReflexiveProperty | 1 / 1 | Asymmetry checks and separately specified reflexive implications |

Examples clarify why context matters. `AICORReferenceManipulationExecutive` has qualified cardinality 1 on `hasProperPart` separately for `ManipulationExecutiveOperatingSystem` and `ManipulationControlProgram`. This does not imply one total part. `AccessibilityCritic` requires `criticDimension` to include the literal `accessibility`; setting it automatically would create inferred data, not faithfully load asserted data. `coRefersWith` is transitive, symmetric and reflexive; existing transitive/symmetric support does not supply reflexivity. `TaskRelativeClass` is equivalent to an enumeration of `solution`, `repaired`, `prohibited`, and `nonachieving`, whose mutual distinctness is explicitly declared.

## Revised upcoming phases

### Phase 7 — AICOR structural compatibility

7A normalizes the schema: deterministic identifier overrides (`None` can become `None_` while retaining its IRI), collisions, named parent edges, and anonymous superclass expressions stored separately with their owning classes. Preserve every asserted parent even if Python base ordering or redundant ancestor removal is needed. Detect cycles and incompatible Python MROs explicitly; do not discard a parent to make imports pass.

7B expands property bindings and tests multiple inheritance. Separate one OWL property identity from its Python field associations. The native descriptor already has a domain/range map; investigate and test compatible multiple associations rather than introducing another descriptor engine. Reuse Symbol and ordinary dataclass literal fields. Missing signatures require an explicit storage/access policy; local restriction usage can guide ergonomic class fields but does not establish global OWL domains or ranges. A generic Symbol storage range, if chosen, is an adapter policy, not an inferred OWL range. Do not generate 966 eager fields on every class by default.

Acceptance: all 1,017 class IRIs have an importable generated mapping or a precise unresolved diagnostic; the target is no unresolved mappings for this file. All 966 property IRIs are represented; all 1,341 restrictions and remaining anonymous expressions have source-linked structured records. Representative multiply inherited classes and property access run against KRROOD. Structural-only output clearly records semantics not yet executable.

### Phase 8 — Class-local constraints and scoped validation

Translate subclass restriction records into inherited necessary conditions. Reuse Phase 6 expression parsing/query construction and native EQL predicates, quantification and counting where their APIs fit. Add qualified cardinality, universal restrictions, functionality and class disjointness. Keep normalization/translation in the wrapper; do not build a parallel inference engine.

Acceptance: tests from this ontology cover missing/present existential witnesses, required values, qualified exact/min/max counts, the ManipulationCase universal restriction, functional conflicts, and both disjointness forms. Reports distinguish violations in a declared complete application scope from unknown cases under open-world semantics. Object functionality can imply equality in OWL; two Python references alone are not a proven logical inconsistency without known distinctness. Counting Python identities alone is likewise not OWL cardinality reasoning.

### Phase 9 — Definitions and property implications

Enable the 13 union definitions and TaskRelativeClass enumeration on the expanded bindings. Support union fillers and the anonymous superclass union without confusing subclass conditions with sufficient membership definitions. Retest all four transitive properties, and symmetric coRefersWith, through native relation machinery. Add a precisely scoped reflexivity policy and asymmetry checks; retain inferred/asserted provenance separately. Handle owl:Thing explicitly in expressions.

Acceptance: positive definition queries and nested existential fillers return documented scoped results; coRefersWith has tested symmetric/transitive/reflexive behavior under the selected scope. No automatic reclassification or inferred default values without an explicit policy.

### Phase 10 — Identity, round trip and AICOR acceptance

Load the four TaskRelativeClass individuals with one registry entry per IRI and preserve AllDifferent. Add unrelated multiple-type views only when an A-Box needs them; the supplied individuals do not currently require that feature. Existing KRROOD roles have distinct identities, so they cannot silently replace the loader's identity contract. Test asserted-source export separately from optional inferred-fact export, and measure generation/import/runtime performance at this ontology's actual size.

Acceptance: compile the unchanged ontology, import its package in isolation, load the four individuals, run representative Phase 8–9 cases, and produce source-preservation plus omission/behavior reports. This establishes the chosen AICOR operational profile, not complete OWL DL equivalence.

## Reproducibility and boundaries

The included inventory JSON records counts, examples and the actual compile failures. Source SHA-256: `a33fb7af8c28792825542b1c663f233102e19ae076bbf747fdd16e4297ef0cda`. Counts of blank-node expressions are reproducible, but RDFLib blank-node labels in examples are parse-local identifiers. The assessment does not claim that the ontology is consistent, that every multiple-inheritance arrangement is Python-compatible, or that every native EQL operator needed by upcoming validators has already been integrated. Those are explicit phase acceptance checks.
