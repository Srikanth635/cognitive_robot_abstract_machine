# Pouring, pickup, place and navigation — 0.22.0

This increment strengthens the existing action modules. It adds **no ontology module, import, action family, robot binding or execution adapter**. All existing example records and their claims remain unchanged. Read this guide alongside the relevant ontology definitions and selected examples when constructing LLM generation context.

## Shared generation contract

Generate a KnowledgeRecord describing an action, not an asserted real event. Identify the record/version, action kind, entity specifications and role bindings. Reuse each entity specification consistently across direct fields, predicates, phases, conditions, models and monitoring. Preserve `PreState`, `CurrentState`, `PostState` and `TaskContext`; a held object before placement and a released object afterward are compatible facts.

Use one binding per declared single-valued field. The three domain ontologies now state maximum cardinalities for their principal bindings. OWL maxima preserve partial descriptions and can infer equality; they do not replace SHACL required-field counts, distinct-binding checks or ordinary input validation. No unique-name assumption or closed-world completion is introduced.

Goals and success conditions have different jobs. A goal may describe only the desired outcome while success adds clearance, support, spill or other conditions. In the bounded conjunction profile, every goal conjunct must also be established by success. Do not silently rewrite a goal as the entire success criterion. Arbitrary entailment, disjunctive plans and physical counterfactuals are outside these inspectors.

Reject a conjunction containing both an atom and its negation for the **same** scope and ordered arguments. The shared signature inspector also rejects an exact pair of complementary comparisons with identical operands/scopes. This is a bounded syntactic consistency check, not a general satisfiability or interval-arithmetic solver. Contrasts across different state scopes remain valid.

Descriptions and generated claims remain Proposed unless an explicit assessment policy provides another status. A filled field, logical consistency or a True supplied-input expression does not establish an accepted observation. Keep source support, evidence grade, workflow status and evidence acceptance separate.

## Action-specific contracts

| Action | Prerequisites in the current bounded profile | Success | Fault evidence |
|---|---|---|---|
| Pickup | Intended object is graspable; intended initial support supports that object in PreState | Intended manipulator holds object; original support no longer supports it; object is clear of that support | Current HoldLost refers to the same manipulator/object; acquisition faults remain separately described by phase guards |
| Place | Intended manipulator holds intended object in PreState | Object at intended target, stably supported by destination support, no longer held, manipulator clear | Current SupportUnstable refers to that object and destination support |
| Navigate | Scoped capability availability and navigability for intended robot, target and footprint; route/level variants retain their additional prerequisites | Intended-region arrival and explicit position/heading tolerance comparisons; variants retain route, drift, level and cabin-exit requirements | Current NavigationFault refers to the same robot and target |
| Pour | Either TransferReady binds intended source/destination/material, or explicit PreState material-in-source containment and destination accessibility are supplied | Intended transferred portion in intended destination; delivered amount at least target; spill no greater than tolerance | Current TransferFault for those endpoints, or explicit post-state spill greater than the declared tolerance |

These are engineering description profiles, not universal physical laws or exhaustive conditions for safe performance. A closed gripper/contact alone is not pickup, release alone is not placement, endpoint arrival alone is not route compliance, and a tilted container alone is not completed pouring.

The simple pouring profile remains partial transfer. It does not require source emptiness. Complete transfer retains its explicit residual-amount/tolerance contract; mixing incompatible concrete transfer profiles requires an explicit extension rather than silently dropping conditions. Quantity basis is volume or mass, with the existing canonical cubic-metre/kilogram checks. Equal numerical values do not imply equal quantity roles. No capacity, foam, safe-force, spill-tolerance or controller value is invented by this release.

## Context, frames and roles

Navigation and material-transfer contexts must scope the selected action. Holding/fixation retain their mandatory interval contexts. Pickup/place may additionally declare a context; when present it must scope that action and be unique. The older context-free pickup/place templates remain supported, but callers must isolate their input facts. Historical record identity alone does not freeze a mutable graph.

A robot, target region and footprint must appear consistently in navigation's participant roles. Manipulator/object and support/object aliases are rejected in the current bounded manipulation profile. Transfer source/destination and transferred-portion/material-description aliases are rejected. These constraints concern these descriptive roles; they do not establish grounded physical identities.

Navigation target/route poses retain explicit reference-frame agreement. A manipulation target may use its existing pose specification; never invent a frame transform, grasp offset or object-to-manipulator pose correspondence. Frame transformations, geometry and observed object identity remain injected responsibilities.

## Assessing supplied inputs

The existing inspectors retain their old success/precondition/fault fields. All three now also report:

- `criteriaSatisfied`: three-valued `precondition AND success AND NOT fault` on supplied inputs.
- `successAndFaultReported`: simultaneous reported outcome and fault remain visible.
- `acceptedObservations = 0`, `worldUpdated = False`, `executionReadinessAssessed = False`, `executed = False`.

The transfer inspector retains `fault` and adds the consistent `failure` alias. An unknown fault is not a false fault. Outcome True with missing prerequisite or fault evidence can therefore leave `criteriaSatisfied` Unknown; a false prerequisite or a reported fault makes it False. It is never an execution certificate or proof an action occurred.

Holding/fixation, route navigation and facing retain their existing interval-evidence rules. This release does not impose an invented universal sampling/freshness threshold. Source correctness, observation acceptance, temporal coverage and physical feasibility remain separate checks.

## Validation before accepting a generated candidate

1. Resolve imports using the local registry and retain the selected ontology/content versions.
2. Validate required structure, cardinalities, basis and context/identity constraints with the existing shapes.
3. Run generic expression/phase integrity and the relevant domain checker. A nonempty error list, unsupported profile or ValueError requires repair or explicit abstention; it must not be converted into an observation that the action failed.
4. Check exact statement provenance and keep generated/synthetic evidence explicit.
5. Use a bounded inspector only with appropriately isolated supplied facts/quantities. Report unsupported questions and Unknown inputs directly.

`generation/primary-action-templates.json` contains structural templates extracted from the four canonical examples: symbolic participant/parameter roles, scoped predicates/comparisons and canonical unit declarations. They are generation aids, not a new serialization standard or guaranteed complete output schema. They contain no physical scalar defaults. `root_expression_type` records the source expression kind; a single atomic clause remains an AtomicCondition, rather than a one-operand Conjunction. `all-clauses` describes logical combination, not a requirement to instantiate a Conjunction node. Symbolic entity bindings include frame specifications and need not all be physical participant roles. Existing fixture IRIs identify their sources; use a fresh record and mutable annotation/context identities, and reuse entity/parameter descriptions across records only through an explicit versioned sharing policy. Keep identities consistent within each description.

Run from the package root with requirements installed:

```bash
python tools/validate_primary_actions.py
python tools/validate_object_manipulation.py
python tools/validate_mobility_navigation.py
python tools/validate_tool_material_transfer.py
```

Current release checks, preserved inputs, reasoning results and remaining limitations are indexed in `reports/release-0.22.0.json` and `HANDOFF_UPDATE_0.22.0.txt`. The historical 0.21 coverage matrices retain their dated scope; missing CoraPlex action families remain explicitly deferred.
