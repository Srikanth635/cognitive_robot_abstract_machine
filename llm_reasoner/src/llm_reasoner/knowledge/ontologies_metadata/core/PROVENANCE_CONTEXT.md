# Provenance and context — 0.11.0

The new `knowledge-context.ttl` module imports the action core and DUL. Its ontology URI is `http://aicor.knowledge/ontologies/knowledge-context.owl`. The generic assembly imports it; individual aspect imports are unchanged. There are now 25 registered ontology documents and fifteen aspect modules.

## Representation

| Concept | Purpose |
|---|---|
| Knowledge Context | A scope for one action description, with a context key and optional externally injected context IRI. Use distinct context IRIs for separate instances/evaluations. Keys are human-facing identifiers; they do not imply identity. |
| Claim Assessment | A record-specific assessment of an exact claim in a context, with status, assessment time, half-open validity interval and evidence links. |
| Source Snapshot | A source locator, explicit version and selector; optional exact excerpt and SHA-256 digest. |
| Observation Evidence | Evidence supplied externally with an observation timestamp. |
| Record Snapshot | A digest of exact exported dataset bytes under a named serialization profile. |

Existing `ac:Statement` remains the exact subject/predicate/value representation. A claim's original `ac:hasStatus` stays intact. A new `kc:assessedStatus` qualifies one assessment in one context. Consumers must explicitly choose whether to use contextual assessments; old queries continue to expose the original status. There is no implicit override or global latest assessment.

For example, two hypothetical contexts may supply target values 0.3 and 0.8 for the same parameter specification. These values exist in reified statements and their assessments, rather than as competing global `ac:numericValue` assertions. Two sources proposing 0.3 and 0.4 in the same context are returned as potential alternatives. Different values only constitute a contradiction when the domain establishes that the predicate is single valued and the claims have compatible units and referents.

## Validation and reference inspection

`knowledge-context-shapes.ttl` checks required fields, exact record membership, description/context alignment, status vocabulary, interval ordering, timezone-bearing timestamps, source version/selector and digest syntax. Supported assessments require explicit evidence. Evidence presence does not establish reliability or acceptance.

The reference inspector in `tools/context_review.py` requires an explicit record, context, evaluation timestamp and maximum observation age. It returns active assessments with fresh/stale/future evidence classifications and potential alternative values. It keeps all returned assessments unaccepted. Non-observation evidence is reported separately and has no inferred freshness.

Intervals use `[validFrom, validUntil)`. Assessments dated after the evaluation instant are excluded. Observation age equal to the maximum age is fresh. Future-dated observations are flagged. No clock or default freshness policy is read from the machine.

The inspector expects SHACL validation first. It performs additional boundary checks but is not a complete validator. No automatic value conversion, observation acquisition, reasoning over negated propositions, confidence aggregation, trust selection, causal identification or robot execution is performed.

Run from the package root with requirements installed:

```bash
python tools/validate_context.py
python tools/review_competency.py
python tools/validate_imports.py
python tools/validate_aspect_completion.py
python tools/reason_imports.py
python tools/check_strategy_integration.py
```

`validate_context.py` validates its fixtures against BOTH core and context shapes, exercises 26 cases, and writes `reports/context-validation.json`. Fixture RDF is in `review/fixtures/context-provenance.ttl`, kept apart from the historical action examples. The synthetic observation evidence is labelled as a test fixture and does not imply sensor acquisition. Original action examples are not retroactively assigned evidence.

## Immutability and provenance boundaries

Record versions and assessment revisions are a modelling contract. An RDF store can still overwrite shared IRIs. Use new assessment IRIs and `kc:revisesAssessment` for revisions; retain old assertions. Do not assume an assessment is current solely because it has no successor. Branches and competing assessments remain possible.

A SHA-256 digest detects changes only to the exact bytes submitted for verification. Producers must export the complete intended dataset, including referenced components, claims and context, using a declared deterministic serialization profile. This release provides an exact-byte verifier and tests mutation detection. It does not implement RDF dataset canonicalization, signatures, graph locking or a persistent immutable store. A matching digest does not prove that all relevant knowledge was included or that it is true.

Source snapshots locate supporting material. Exact excerpts are preserved as supplied; independent source retrieval and excerpt verification remain future ingestion work. A selector may be a page/section, offsets or another explicitly documented format. The ontology does not parse arbitrary selectors.

Old descriptive triples are still not exhaustively covered by exact claims. The 0.10 review's root-link provenance gaps remain visible. This release supplies contextual assessment and source structures for future annotation, rather than fabricating historical justifications. A complete annotation pipeline must define which substantive triples require claims, populate them from real sources, and enforce coverage before acceptance.

## Next work

1. Add a source-backed example whose transfer quantity explicitly identifies substance, source and destination, and whose success condition checks that destination.
2. Define evidence-acceptance policies, including units, uncertain observations, freshness per modality and source conflicts.
3. Implement deterministic dataset exports and persistent immutable records before multi-annotator deployment.
4. Add explicit links between diagnostic hypotheses and causal explanations when sourced examples clarify their intended semantics.

Projection development remains deferred.
