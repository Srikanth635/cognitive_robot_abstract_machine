# Specialized tool interactions — 0.20.0

The refinement ontology IRI is `http://aicor.knowledge/ontologies/specialized-tool-interactions.owl`. It imports tool interaction and knowledge context, with DUL inherited through the generic action core. Its four description classes specialize the existing ToolUseDescription rather than copying its operator/tool/working-part/target/effect vocabulary.

| Bounded profile | Additional success requirements |
|---|---|
| Cutting | SpecifiedSeparationAchieved for the intended target and effect |
| Wiping | SpecifiedSurfaceEffectAchieved, interval-qualified SweepCoverageMaintained and terminal working-part contact |
| Maintained pressing | SpecifiedActuationAchieved and contact at the assessed actuation state |
| Insertion-to-seat | SpecifiedSeatingAchieved, AxisAlignmentSatisfied and absolute depth error within the declared tolerance |

All retain generic ToolHeld, ToolFunctional and EffectObserved conditions, plus the original tool/context/working-part integrity checks. Each has an explicit TechniqueReady TaskContext prerequisite. This is an externally assessed applicability/readiness predicate, not a tool-selection algorithm, verified affordance or proof of a valid physical model.

The examples remain synthetic Proposed descriptions. No cutting force, material hardness, friction, pressure, insertion depth, safe threshold, sensor observation or accepted belief is fabricated. Intended effects are descriptive specifications; their predicates require supplied evidence rather than asserting a physical event.

## Technique boundaries

Cutting's generic contact policy remains unspecified; the specific outcome requires the requested separation rather than contact alone. This bounded profile does not define cut geometry, full/partial separation alternatives, sharpness, cutting trajectories, fracture mechanics or contact-force bounds. Domain evidence must establish what its requested effect means for the material and tool.

Wiping describes an externally qualified requested surface effect and coverage over the declared operation interval. It does not prove sterilization, microbiological safety, contaminant removal completeness or an arbitrary cleaning standard. Without an explicit matching interval, SweepCoverageMaintained is withheld and cannot establish success. Interval bounds do not verify sampling coverage or measurement correctness. Continuous contact, multiple strokes, fluid transfer, material compatibility and changing targets need additional explicit contracts.

The pressing fixture is a **maintained-contact** actuation profile. A momentary press followed by release or a toggle that changes back later is a different temporal contract. The ontology does not infer that contact actuated a button or that an intermediate actuator state persists until the end of a longer recipe.

Insertion-to-seat is a requested aligned seating profile, not all possible insertion behavior. Requested depth, absolute depth error and tolerance are three distinct parameters in canonical metres. The error explicitly references the target depth and the tool description's measurement frame. Known parameters must be finite and nonnegative; no numerical values are provided in the fixtures. The success expression requires the bound PostState error <= TaskContext tolerance, in addition to aligned seating evidence. A seated label alone cannot substitute for the metric. No axis transformation, penetration calculation, compliance controller, jamming detector or force-limited insertion is implemented.

These inherit a held-tool use pattern. Supported/mounted, multi-tool or multi-actor alternatives require explicit profiles rather than assuming this first set is exhaustive.

## Inspection

Load the ontology closure with `examples/specialized-tool-profiles.ttl`, validate core/domain/refinement shapes, and run `specialized_errors` before assessment. `assess(graph, action, context, facts=None, quantities=None, interval=None)` requires the exact TI.evaluationContext. It evaluates the refined criterion through the existing tool evaluator, preserving the distinction between preconditions, success and explicit fault.

Wiping requires a timezone-qualified, nonzero interval with the same context to use its aggregate coverage fact. The other three profiles inspect their declared state/metric conditions without claiming continuous execution validity. Supplied quantity keys use the existing `(scope, parameter IRI)` convention and exact unit IRIs; no conversion occurs. Facts use `(scope, relation IRI, ordered entity tuple)` keys. Inputs are hypothetical until an external evidence policy accepts them. `acceptedObservations` is zero and executed/executionReadinessAssessed are false.

`tools/build_specialized_tool_examples.py` clones the previous tool-use fixture into new identities, specializes its success/goal and adds exact Proposed claims. It regenerates only this new example file and does not rewrite the original tool domain or examples. `queries/specialized-tool-targets.rq` requires an explicit selected record and returns the intended technique, tool, working part, target, effect and optional depth specification; it does not validate or accept observed effects.

Reproduce checks using `tools/validate_recipes_and_specialized_tools.py`, `tools/validate_recipe_tool_queries.py` and `tools/reason_recipes_specialized.py`. Current counts and reasoning scope are indexed in `reports/release-0.20.0.json`.

Next improvements should be source-backed effect definitions, material/geometry/force models, quantitative observation and coverage contracts, temporal variants, scoped simulation validation and capability-provider matching. This increment provides semantic refinements and reference inspection, not new working projections or controllers.

## Recorded release results

149 combined new checks, 23 scoped query cases and 62 import/catalog checks passed. Transport (159), tool/material-transfer (81) and historical competency (93) regressions passed. Three generic ontology closures classified with no unsatisfiable named classes. Combined generic examples/demo consistency passed with a 7168 MB heap. Optional AICOR consistency timed out at 180 seconds and remains unverified. Of 81 prior RDF files, 80 are byte-identical; only the combined assembly changes.
