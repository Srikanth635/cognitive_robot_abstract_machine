# Competency-question review and consolidation — 0.10.0

> Historical review/catalog; its fixture versions and counts retain their original scope. Current coverage and consolidation findings: [0.21.0 review](coverage-0.21.0/REVIEW_REPORT.md).

## Result

The fifteen aspect families provide a useful shared description vocabulary. A first competency review now makes their capabilities and limits explicit through **24 runnable SPARQL questions**, two existing synthetic action records, and **six additional reasoning/completeness questions**. The suite has **93 passing assertions**, including cross-record contamination and unknown-information checks. Passing these checks establishes the documented behavior of these fixtures; it does not establish real-world action competence or complete domain coverage.

One evidence-backed schema consolidation is implemented: a common capability-requirement superproperty. No new aspect or integration ontology was necessary. The package still has 24 registered ontology documents, including 15 aspect modules and four semantic bridges. Existing example records, DUL, and both original input ontologies are preserved.

## Scope and method

- Inputs: the local 0.9.0 package, all example graphs, the generic import closure and a bounded review of the unified AICOR schema. This review did not re-audit the live CRAM repository or its current Python interfaces.
- Positive fixtures: `pourRecordV7` and `graspRecordV7`; V6 is checked for version separation. Neither is a real episode or scientifically validated action-domain module.
- Queries explicitly bind one record. They retrieve descriptions and included claims. They do not select a globally “latest” version, interpret a missing triple as false, or authorize execution.
- Added test data exists only in temporary in-memory graphs. Foreign claims/evidence must not enter answers until explicit record membership is added.
- All current V7 claims are `Proposed`, with `CuratedExample` evidence. This is the status of the illustrative statements, not a statement that every referenced model is an established fact.
- OWL import/classification checks, SHACL checks, bounded Python semantics and SPARQL retrieval are separate forms of validation. None substitutes for the others.

See `COMPETENCY_QUESTIONS.md` for every question, the machine-readable JSON catalog for the 24 query contracts, and `reports/competency-review.json` for exact RDF answers and input hashes.

## Consolidation implemented

New canonical property:

`http://aicor.knowledge/ontologies/action-description-core.owl#requiresCapability`

Label: **Requires Capability**. Domain: `dul:Description`. Range: `ac:CapabilityRequirement`.

| Existing property | Meaning retained | Consolidation |
|---|---|---|
| `ac:hasCapabilityRequirement` | Requirement attached directly to an action description | Subproperty of `ac:requiresCapability` |
| `cs:requiresCapability` | Requirement grouped in a capability profile | Same parent |
| `fd:requiresCapability` | Requirement of a recovery option | Same parent |
| `mr:requiresCapability` | Requirement of an observation description | Same parent |
| `ema:requiresCapability` | Requirement of a conditional affordance | Same parent |
| `sm:requiresCapability` | Requirement of a sensory expectation | Same parent |
| `sm:observationRequirement` | Observation capability needed at a control point | Same parent |

Why the parent uses `dul:Description`: reusing `ac:hasCapabilityRequirement` as the parent would infer that every profile, affordance and observation requirement is an `ActionDescription`, due to that property's domain. The new broad parent avoids that inference and preserves the more specific domains of the existing properties.

`cs:fulfillsRequirement` and `cs:assessmentOutcome` are deliberately outside this hierarchy. Intended fulfillment and an assessed outcome have different meanings from requiring a capability. No equivalence, satisfaction inference or property chain silently promotes a nested requirement into an action-level guarantee.

CQ12 follows the declared structural routes and returns the requiring node and specialized predicate. For pouring it exposes **three distinct requirements across 12 owner/property links**: manipulation, verification, and quantity estimation. The existing profile alone lists only the first two. Grasping similarly exposes relation assessment alongside manipulation and verification. Whether a capability profile is intended to be exhaustive is still unspecified; it must not be used as an exhaustive checklist without a separate completeness contract.

## Shared vocabulary decisions

| Area | Decision | Reason |
|---|---|---|
| Conditions | Keep `ac:ConditionSpecification` and the bounded `co:Expression` language | Goal, guard, invariant and model assumption are different roles for conditions; their properties should not all become equivalent. |
| Parameters | Keep `ac:ParameterSpecification` as the common anchor | Model variables, objective metrics, controller settings and retry limits already point to it. Units and context still need stronger semantics. |
| Capabilities | Add the common requirement parent above | Enables cross-aspect discovery while preserving the requiring context. |
| Entities | Keep `ac:EntitySpecification`; retain action-attached entity facets | Avoid creating duplicate objects per perspective or modifying earlier shared entities with later material facets. |
| Failure versus cause | Keep `FailureSpecification`, `DiagnosticHypothesis`, and `CausalLink` distinct | A failure category, a candidate explanation, and a causal relation are not interchangeable. Their explicit connection is a remaining gap. |
| Evidence versus description | Keep separate | An expected observation or proposed simulation study is not evidence that an event occurred. |
| AICOR versus generic core | Keep the optional adapter boundary | Similar labels do not establish equivalent semantics or compatible class restrictions. |

`shared-vocabulary.csv` inventories properties with the exact shared condition, parameter, capability and entity ranges. It is an inventory of declared ranges, not a proof that all relationships with similar labels mean the same thing.

## Findings and unresolved questions

| ID | Finding | Evidence | Classification and next action |
|---|---|---|---|
| F1 | Capabilities were fragmented across predicates and owners | CQ12 reveals requirements beyond the action profile | **Implemented:** shared parent plus scoped query. Specify profile completeness separately before provider matching. |
| F2 | Direct action-description links lack exact statement provenance | CQ23 reports 37/37 direct non-type links in pouring V7 and 33/33 in grasping V7 without an exact included subject/predicate/object statement | **Provenance gap:** these counts cover root links only; many internal component statements do have claims. Define which structural links need evidence and which are bookkeeping, then provide statement-level coverage for substantive content. No evidence was fabricated. |
| F3 | Quantitative pouring success does not explicitly test the intended destination | The success expression depends on delivered amount and spill quantities, with no relation leaf for receiving-container containment | **Example/domain-contract gap:** the vocabulary can express containment, but the quantity lacks an explicit destination referent in this template. A future sourced domain pattern should bind transfer amounts to source, destination and substance and specify partial versus complete transfer. |
| F4 | Context is only partly explicit | `PreState`, `PostState`, `CurrentState`, `TaskContext` and external bindings exist, but evaluator keys contain no episode ID, observation time or freshness | **Context-model gap:** design qualified evaluations/measurements and their action/episode context before running concurrent instances. Current evaluation must receive one isolated context per call. |
| F5 | Diagnostic and causal descriptions are disconnected | `pourSuspectedCause` differs from the causal link's `pourCauseCondition`; no dedicated diagnostic-to-causal-link relation is supplied | **Cross-aspect gap:** define a scoped, evidence-bearing association when domain examples establish the needed meaning. Do not equate a diagnosis with a proven causal link. |
| F6 | Physics/control are descriptive and partial | CQ08/CQ11 and the controller inspector reveal unresolved equations, values and timing; CQ09 finds no included model-validation evidence | **Knowledge and evaluator gaps:** sourced parameters, model applicability, dimensional checks and solver/simulation evidence are still needed. OWL alone does not perform these calculations. |
| F7 | Recovery is structurally expressible but not operationally bounded | CQ15 finds budget specifications with no numeric values; phase graphs may contain retry loops | **Data and execution-contract gaps:** no unlimited-retry default is permitted. Termination, stopping conditions and scheduling belong to later operational contracts. |
| F8 | A record is not an immutable storage snapshot merely because it has a version string | Current examples reuse component and claim IRIs across versions | **Storage/context gap:** the tests prevent several query-join leaks but cannot prevent later edits to a shared IRI. Specify immutable content, named graphs or another versioned storage discipline before multi-annotator persistence. |
| F9 | Explicit unknowns are present but domain knowledge is sparse | Force tendencies/strength/results and material state remain Unknown; geometry is unspecified; current frame artifacts are for V6 | **Intentional missing knowledge:** avoid filling it with invented constants or silently reusing old projections. |
| F10 | Similar AICOR names are not ready for automatic merging | Reviewed class parents/restriction counts differ from generic descriptive concepts | **Alignment-review boundary:** preserve the bridge. Full AICOR satisfiability and alignment to current code remain unverified. |

The uniform superproperty solves F1's discovery problem; it does not solve all the remaining findings. Those are explicit review outcomes, not silently counted as implemented features.

## AICOR consolidation boundary

The bounded comparison in `aicor-overlap.json` covers action/designator, entity/object, parameter, task/action-kind, process-module and plan/strategy concepts. It reports asserted named parents and counts anonymous parent expressions from the actual unified ontology.

Examples of distinctions retained:

- `ac:ActionDescription` describes action knowledge; an AICOR `ActionDesignator` has CRAM-specific restrictions and adapter implications.
- `ac:EntitySpecification` is a description of a possible participant; it is not an AICOR physical object.
- The example `ex:Pouring` is an individual of `ac:ActionKind`; AICOR `Pouring` is an OWL class. A class-to-concept bridge needs an explicit design, not a same-name equivalence.
- An abstract process requirement does not identify a runtime `ProcessModule`; an execution strategy description does not automatically become an AICOR `Plan`/control program.

## Validation and limits

Fresh results for this release:

- Competency review: **93/93 assertions**, 24 queries over two V7 records plus boundary/version/contamination tests.
- Consolidation audit: **81/81 checks** against the 0.9.0 ZIP; exactly six schema files changed. After removing the declared property additions and version metadata, their graphs are isomorphic to the baseline. **42 other ontology/example files are byte-identical**, including DUL, original inputs, candidate ontologies and examples.
- Offline imports/catalogs: **34/34 checks**.
- Combined SHACL and last-aspect integrity/negative tests: **22/22 checks**.
- HermiT classification: **20 modular closures consistent, with no unsatisfiable named classes reported**.
- Combined generic + AICOR + examples: **consistency passed**; see `reports/strategy-integration.json`. This is not full AICOR classification.

The fixtures exercise pouring and grasping. They do not establish adequacy for assembly, navigation, communication, concurrent multi-agent work or every action family. EQL, SDT, CoraPlex and Isaac Sim integrations were not implemented or exercised here.

## Recommended next ontology work

Address **statement provenance and evaluation context** next, using a few explicit cases: two action instances with different targets, two sources that disagree about one scoped parameter, an outdated observation, and a source-backed partial-transfer goal. This gives a concrete basis for claim context, freshness, conflict handling and immutability requirements. After that, use a small reusable material-transfer/manipulation domain module to test the generic aspects against sourced action knowledge. Working projections remain deferred.
