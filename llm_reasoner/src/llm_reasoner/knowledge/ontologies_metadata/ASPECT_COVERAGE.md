# Aspect coverage and improvement boundaries — 0.12.0

> Historical coverage snapshot. Current cross-release assessment and next steps: [review/coverage-0.21.0/REVIEW_REPORT.md](review/coverage-0.21.0/REVIEW_REPORT.md). Preserve the dated statements below as historical context.

All fifteen planned aspect families now have a first ontology representation. This closes the initial aspect-inventory stage, not all research, modelling or runtime work.

| Aspect | Representation now available | Important improvement boundary |
|---|---|---|
| Participants/relations | Roles, entity specifications, predicate signatures | Richer relation semantics and role constraints |
| Conditions/outcomes | Boolean/quantity expressions and intended transitions | Quantification, temporal logic and dimensional semantics |
| Phases/transitions | Guarded branching graphs and recovery paths | Concurrency, hierarchical phases and operational timing |
| Monitoring | Observation dependencies and unknown responses | Freshness, uncertainty and temporal detection |
| Causal mechanisms | Scoped causal hypotheses and intervention descriptions | Grounded causal laws, identification and counterfactual reasoning |
| Physical models | Properties, equations, variables, artifacts and validation records | Equation semantics, calibration and validity assessment |
| Objectives/trade-offs | Explicit policy, weights/ranks and hard constraints | Metric normalization and actual optimization |
| Dynamics/control | Vector assignments and controller requirements | Full models, synthesis and stability/feasibility analysis |
| Capabilities/strategies | Abstract interfaces, assessments and ordered steps | Provider matching, grounding and scheduling |
| Failure/diagnosis | Candidate explanations and recovery options | Diagnosis from episodes, selection and bounded execution |
| Frames | Slots, bounded defaults/inheritance and contracts | Additional nonmonotonic semantics and broader frame coverage |
| Force dynamics | Qualitative roles, tendency, strength and result | Evidence-backed domain patterns; quantitative linkage |
| Image schemas | Containment/path/support/contact structures | Additional schemas, transformations and spatial grounding |
| Entity/material/affordance | Record-scoped facets and contextual descriptions | Detailed geometry/material composition and affordance semantics |
| Sensorimotor events | Boundaries, expectations, modalities and control points | Signal semantics, prediction validation and event detection |

## What remains before relying on this as action knowledge

1. Extend the first competency review in review/REVIEW_REPORT.md as new domains are introduced. The initial review and capability-requirement consolidation are complete; context, provenance and explicit causal/diagnostic connections remain unresolved. DUL alignment and consistency do not prove the conceptualization is adequate.
2. Develop reusable action-domain modules (e.g. material transfer, pouring, grasping) and populate them from explicit sources. Current examples are synthetic templates.
3. Strengthen statement-level provenance, source/conflict acceptance policies and context/version semantics. Not all template triples currently carry individual claim provenance.
4. Improve selected semantics based on concrete competency questions. Important unresolved scope includes logical quantification, units, temporal context, concurrency, detailed material/geometry models and operational recovery policies.
5. Revisit full AICOR satisfiability/code alignment. Generic modular classification has passed, while the full AICOR classification remains unverified.

No ontology is expected to encode every possible detail of every action. Unsupported content should remain unknown. The coverage table is a draft modelling inventory, not a declaration that all aspects are scientifically or operationally complete.

## Deferred implementation work

Projection development is paused as requested. Later work includes richer frame/force/image-schema projection contracts, EQL/krrood integration, ontology-guided LLM annotation, persistent multi-annotator records, SDT/world bindings, CoraPlex execution adapters and Isaac Sim evidence acquisition. These are integrations, not additional aspect inventories to disguise as empty classes.

## Recommended ontology-only next step

The first competency review is complete: see **review/COMPETENCY_QUESTIONS.md** and **review/REVIEW_REPORT.md**. Prioritize statement-level provenance and evaluation context, tested with conflicting scoped claims, concurrent action instances and stale observations. Then validate a small sourced action-family module against those contracts. Projection development remains deferred.

## Pouring/diagnosis/model improvements in 0.12.0

See **aspects/POURING_DIAGNOSIS_MODEL_REVIEW.md**. The new V8 example binds transfer quantities to participants, requires destination containment for success, connects a diagnostic hypothesis to a causal hypothesis, and supplies bounded quantity/equation contracts. The prior review still documents V7. Physical validity, complete-transfer semantics, general unit conversion, calibrated dynamics and control synthesis remain unresolved.
