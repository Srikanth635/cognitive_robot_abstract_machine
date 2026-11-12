# Frame perspective and projection contract — 0.8.0

This release introduces frame-perspective.ttl, importing only the generic core and DUL. The comprehensive assembly imports it as its eleventh aspect. Frames use HTTP IRIs under http://aicor.knowledge/ontologies/frame-perspective.owl# and English labels.

This is an explicitly bounded Minsky-inspired slot/default perspective. Its inheritance/default algorithm is defined here and implemented in Python; it is not a claim to implement a complete historical frame language or a particular published frame formalism. OWL class inheritance does not perform default reasoning.

## Representation

FrameDefinition specializes the core PerspectiveDescription. Its SlotDefinitions are DUL Concepts, connected through a specialization of DUL defines. Slots have stable keys, minimum/optional maximum cardinalities, optional source mappings and at most one literal/entity default.

FrameProjectionSpecification specializes the existing ProjectionSpecification. It declares the frame, evidence policy, projection key and missing-information policy. The source record is supplied when applying the contract, rather than copied into a second action knowledge base.

SlotMapping has a finite, contiguous, one-based sequence of PathSteps. Each step contains an absolute predicate IRI literal. Traversal starts at the source record's expressed ActionDescription. Paths follow forward RDF edges only. They do not evaluate code, SPARQL, external lookups or procedural attachments. Output entity fillers must have stable IRIs; literals preserve datatype/language. Anonymous nodes may be traversed internally, but cannot become output fillers.

## Inheritance, defaults and exceptions

The current profile supports one parent per frame. Slots are inherited by identity/key. Duplicate keys, inheritance cycles, ambiguous overrides and malformed cardinalities are rejected. Overrides target an inherited slot and either suppress its default or replace it with an explicit literal. Multiple inheritance, overriding mappings/cardinalities and executable slot attachments are not implemented.

Source content takes precedence over defaults. A default is used only when the source path has no candidate value, and is marked default-unverified. It does not become a source fact or an accepted physical value. A value excluded by the evidence policy is never silently replaced by a default. In strict evidence mode, unverified defaults are excluded altogether.

The examples use only an exploratory metadata default, request-evidence, not a mass, force limit, pouring angle or controller setting. Pouring explicitly suppresses this inherited default; grasping retains it. Neither value is executed as a command.

## Evidence and missing information

Two evidence policies are implemented:

- retain-with-status: preserve template content with attached source-record claims/statuses/evidence. Content without an associated claim is explicitly unassessed-template-content.
- require-allowed-claim: keep a value only if the source record contains an associated claim with an explicitly allowed status. Any Rejected or Disputed associated claim vetoes the value. No implicit acceptance threshold or confidence is invented.

A claim is associated by aboutComponent matching the selected filler and by membership in the exact source record. Claims elsewhere in the graph are not borrowed. This is a bounded component-level provenance rule; it does not prove a claim entails every property of a complex filler. Multiple/disputed claims still require a richer acceptance policy in future. Allowing Proposed in a strict contract explicitly permits proposals; it does not verify them.

Missing-information policies are executable contract behavior: omit-and-report leaves missing values empty with issues; retain-unknown adds an explicit JSON unknown marker without manufacturing a filler; reject raises an error on missing/cardinality/evidence issues. Maximum-cardinality conflicts are reported rather than silently truncated. Cardinality is view structure, not physical truth or execution readiness. Unknown markers are JSON metadata rather than fabricated RDF entities.

## Artifacts and versioning

FrameProjectionArtifact specializes the core ProjectionArtifact. It records derivedFromRecord, conformsToSpecification, projectedFrame and includesClaim links. FrameSlotValue records the filler, slot, source/default origin and associated source claims. Default values have no fabricated claim provenance.

JSON output preserves source record/version, contract/frame identifiers, filler IRIs or typed literals, claim status/evidence, issues and default-origin frame. Complete default interpretation uses the referenced immutable contract/frame. Contract and frame identifiers must be versioned or treated as immutable after projection publication. Artifact IDs derive from record/contract identities, so reusing mutable identifiers would invalidate reproducibility.

The source KnowledgeRecord and description are not mutated. Frames can be projected directly from existing v6 pouring/grasping records. No separately generated action content is needed to fill participants or goals.

## Examples

GenericActionFrame defines action-kind, participants, goals and exploration-policy slots. PouringFrame and GraspingFrame inherit those slots. Their contracts generate views from pourRecordV6 and graspRecordV6. The participants/goal fillers are the same conceptual resources used by the shared action representation, not copies of physical objects or events.

The default exploratory contracts show original Proposed claims where present and explicitly mark other template content unassessed. They produce inspectable frames, not executable action designators. Additional slots for phases, causal models or controllers can use the same finite path mechanism; the initial contracts deliberately demonstrate a small reusable subset.

## Run and validate

```bash
python tools/project_frames.py
python tools/validate_frames.py
python tools/validate_imports.py
python tools/reason_imports.py
python tools/check_strategy_integration.py
```

The reusable API is frame_runtime.project_frame(graph, record_iri, contract_iri), returning a structured result; artifact_graph(result) returns its RDF provenance graph. Validate input record/contracts with SHACL and the frame profile before projection. Runtime checks additionally enforce inheritance, paths and policies. No new interpretation of a physical condition is performed by this projection.

Read examples/frame-projections.json and examples/frame-projection-artifacts.ttl. Templates/contracts are examples/frame-contracts.ttl. Reports are reports/frame-*. The full import classifier now covers eleven aspects, four semantic bridges and the assembly. Combined AICOR checking establishes consistency only, not full class satisfiability.

## Remaining boundaries

Procedure attachments, multiple inheritance, general nonmonotonic logic, deep statement-level acceptance, extracted frame defaults, live world binding and EQL access remain future work. Next perspective modules can cover force dynamics and image schemas while referencing the same shared entity/condition identities.

## Recorded results

All 30 frame checks and all 30 current import/reference/catalog checks passed. Sixteen aspect/bridge/assembly closures passed full classification with no unsatisfiable named classes. Combined AICOR/examples/artifact consistency passed; full AICOR classification remains unverified. Thirty-five prior Turtle files were verified unchanged, with the assembly's added import as the only modification to an earlier ontology.
