# Ordered action recipes — 0.20.0

The ontology IRI is `http://aicor.knowledge/ontologies/action-recipes.owl`. It imports the existing transport/carrying composition and specialized tool refinements. DUL, the generic action core, object manipulation and generic tool interaction remain their vocabulary owners. There is no CoraPlex, SDT, robot, live-world, planner or simulator import.

## Initial families and four examples

| Family | Ordered semantic stages | Synthetic examples |
|---|---|---|
| Pick–carry–place | Pickup, Carry, Place | Destination carrying; route/orientation-constrained carrying |
| Acquire–use–return tool | AcquireTool, PositionTool, EngageTool, UseTool, WithdrawTool, ReturnTool | Cutting cycle; wiping cycle |

RecipeStep is an owned indexed occurrence referencing an action description, semantic stage role and explicit context. Indices are contiguous and unique. Each stage uses a distinct action/context occurrence; the parent uses another context. The recipe's subaction set must match its stages. Stage occurrences and handoff requirements have one owning recipe. The fixed first profiles reject reordered, missing, duplicated or incompatible stage families rather than silently skipping them.

The pickup/carry/place family preserves the same manipulator and payload. The carry stage also matches the parent carrier and destination. Pickup's source support and placement's destination support/region are explicit. The placement region has a descriptive withinDestination link to the carry destination; this link is a Proposed compatibility description, not a computed geometric containment proof. RecipeResourcesReady and component prerequisites still need qualified evidence. Navigation's region may be broader than the precise placement region; they are not silently equated.

The tool cycle acquires a grasp, positions/engages the tool, performs one declared specialized technique, withdraws, and returns it through supported placement. Tool stages share operator, tool/payload, working part, target and intended effect. Their working parts must belong to that tool, and the tool stages use a common declared frame in this bounded profile. The final support equals the declared original support; the parent precondition explicitly describes the tool initially supported there. The final placement region is explicit. No navigation to the tool or across rooms is inserted; workspace reachability and required motions remain externally qualified requirements.

Generic ToolHeld and manipulation Holds retain their separate vocabularies. They are not made owl:equivalentProperty. A later stage's own prerequisites must be supplied explicitly; the handoff contract qualifies continuity rather than silently rewriting one domain's PostState predicates into another's PreState facts.

## Qualified outcomes rather than one flat state

Pickup may report Holds=True in its PostState and placement may report Holds=False in its PostState. These are historical conditions in different contexts/windows, not a contradiction. The recipe must not flatten them into a single fact dictionary or a single ordinary PostState conjunction.

RecipeSuccessContract is a machine-readable specialization of the core SuccessCriterion. It names every required stage success, every required stage precondition, every handoff and the final stage. The surrounding core conditions/goal use the declared `ordered-recipe-contract-v1` lexical contract; this is **not** arbitrary executable text or a new general logic engine. The recipe inspector supplies the bounded interpretation. Generic condition evaluation should not be applied to this contract as though it had one flat expression.

The recipe's conforming-success result requires its local prerequisites, every stage precondition and success, every handoff condition, and the complete ordered interval binding. A favorable stage outcome does not waive that stage's prerequisite. The final intended placement/return is the final stage's qualified outcome. Tool effects are established at their use stage; persistence of a surface/actuator/material state through later stages is not inferred. A task needing persistence until final completion needs an additional sourced invariant/goal contract.

Precondition is separately reported from the first stage and recipe-local prerequisites. Failure is the disjunction of explicit component faults and an explicit RecipeFault. Missing history or an unresolved goal is not automatically an action-failure event. Concurrent conforming-success and fault reports remain visible; there is no hidden arbitration or authorization.

## Sequential intervals and handoffs

The inspector consumes an ordered prefix of stage inputs. Foreign stages, missing earlier stages with later outcomes, overlapping windows and reversed timing are errors. Each supplied stage input must name its stage context; an optional stage interval must have that context, timezone-bearing timestamps and positive duration. Stage intervals require an explicit parent interval. Every known stage window must lie within it. A complete binding starts at the first stage boundary and ends at the final stage boundary.

Gaps between stages are allowed, but do not preserve a grasp or load support by default. Exactly one HandoffRequirement joins each adjacent pair. Its positive ControlContinuity predicate refers to the same operator/payload; pick/carry/place handoffs additionally require LoadSupportContinuity for the carrier/load. These predicates mean externally assessed continuity over the exact selected gap or boundary, not an inferred instantaneous contact or a guarantee from a stage label.

Each supplied handoff record names parent context, fromStep, toStep, start, end and its fact dictionary. Bounds must be exactly previous-stage end to next-stage start. A zero-width shared boundary is permitted, but still requires explicit boundary continuity evidence; it is not treated as a positive-duration observation interval. Missing handoff evidence stays Unknown even when both neighboring stages independently report success. Incorrectly scoped or partially bounded supplied handoff records are rejected.

Interval metadata scopes evidence, not its adequacy, calibration, freshness or sampling coverage. Current reference inputs do not acquire or accept observations and carry no implicit trusted-source policy. All supplied inputs in validation are synthetic.

## Nested carrying and wiping

The carry stage calls the existing transport inspector at the explicitly selected description within the recipe's shared record. The earlier record-based transport API remains available and delegates to `assess_description`; its regression suite is preserved. No duplicate independently mutable child record is manufactured merely to call the inspector.

The carry stage's holding and mobility windows must remain coextensive with that stage, while pickup and placement lie before/after it. A missing nested holding window is not borrowed from the carry or whole-recipe window. Orientation constraints refer to interval maximum deviation during carrying, not pickup/place orientations.

The wiping stage similarly needs its own explicit operation interval to use coverage evidence. The recipe's parent interval does not automatically qualify that aggregate. Generic tool contact/effect evidence cannot substitute for the specialized surface outcome or requested coverage.

## API and input shape

Load the import closure and example RDF; validate shapes and `recipe_errors` first. Then use `assess(graph, record, parent_context, stages=..., handoffs=..., interval=..., facts=...)` from `tools/action_recipes.py`.

Stage inputs are keyed by RecipeStep IRIs, not just action classes:

```python
stages = {
    step_iri: {
        "context": "<that stage context>",
        "interval": {"context": "<that stage context>",
                     "start": "<timezone-qualified start>",
                     "end": "<timezone-qualified end>"},
        "facts": {...}, "quantities": {...},
        # Carry stages additionally use the existing transport `components` map.
    }
}
handoffs = {
    handoff_iri: {
        "context": "<recipe context>",
        "fromStep": "<previous step IRI>", "toStep": "<next step IRI>",
        "start": "<previous stage end>", "end": "<next stage start>",
        "facts": {...},
    }
}
```

Facts/quantities use the existing domain conventions and remain separate per stage/handoff. The inspector reports stage results, qualified handoff truth, ordered-interval binding, precondition/success/failure and simultaneous-success/fault reports. Executed and executionReadinessAssessed remain false, acceptedObservations is zero and selectedNextStage is null. The model does not start a stage, move a robot, consume resources, decrement retries or perform recovery.

## Provenance, checks and extension boundary

All four recipes are synthetic Proposed snapshots. Reachable substantive links have exact included statements/evidence. There are no physical parameter defaults, actual world bindings, accepted observations or performed action occurrences. Earlier fixtures and the comprehensive pouring demo remain unchanged. `tools/build_action_recipe_examples.py` clones existing templates into new occurrence identities and regenerates only the new recipe file.

`queries/recipe-stages.rq` and `queries/recipe-handoffs.rq` require an explicitly selected record and expose the ordered descriptions and continuity requirements. Retrieval does not evaluate truth or accept evidence.

Run `tools/validate_recipes_and_specialized_tools.py`, `tools/validate_recipe_tool_queries.py`, `tools/reason_recipes_specialized.py`, `tools/validate_transport_carrying.py`, `tools/validate_tool_material_transfer.py`, `tools/validate_imports.py` and `tools/review_competency.py`. Reasoning scope and limits are indexed in `reports/release-0.20.0.json`. Generic classification/consistency is separate from optional AICOR checking; full AICOR named-class satisfiability remains unverified.

This first recipe profile is a bounded total order. It is not a temporal planner, concurrent scheduler, complete PDDL/behavior-tree formalism or terminating executable plan. Retries, timeouts, recovery, alternate branches, process-provider selection, resource locks, actual pose transforms and task-specific feasibility require further contracts. Later recipes may compose open-access-manipulate-close, fetch/deliver, pouring or assembly, but must define their identity, context, stage and boundary semantics explicitly. Working projections and CoraPlex execution adapters remain deferred.

## Recorded release results

149 combined new checks, 23 scoped query cases and 62 import/catalog checks passed. Transport (159), tool/material-transfer (81) and historical competency (93) regressions passed. Three generic ontology closures classified with no unsatisfiable named classes. Combined generic examples/demo consistency passed with a 7168 MB heap. Optional AICOR consistency timed out at 180 seconds and remains unverified. Of 81 prior RDF files, 80 are byte-identical; only the combined assembly changes.
