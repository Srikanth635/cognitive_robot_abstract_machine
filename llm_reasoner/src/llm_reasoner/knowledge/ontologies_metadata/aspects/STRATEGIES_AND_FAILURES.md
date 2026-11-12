# Capabilities, strategies, failure diagnosis and recovery — 0.7.0

Two modules extend the comprehensive shared description. Both import only action-description-core and its DUL foundation:

- capabilities-strategies.ttl (`cs:`): grouped requirements, strategy steps, abstract process interfaces and contextual capability assessments.
- failure-diagnosis.ttl (`fd:`): failure kinds/specifications, candidate diagnostic causes and guarded recovery options.

The optional integrations/recovery-strategy-bridge.ttl declares fd:usesStrategy, linking a RecoveryOption to an ExecutionStrategy. This link is defined in the bridge from its first release; neither base module requires the other. The comprehensive all-action-aspects.ttl assembly includes both and the bridge. All new terms have HTTP IRIs and English labels.

## Capabilities and strategies

CapabilityProfile groups the existing CapabilityRequirement concepts. It describes needs rather than actual robot abilities. StrategyStep links a conceptual ActionDescription, process-interface requirements and optional conditions or phase references. ExecutionStrategy owns a partial order of steps. The profile rejects cycles, foreign ordered steps and shared step ownership. Unordered steps remain unordered; computed precedence layers do not authorize concurrent execution.

ProcessModuleRequirement has an abstract interface key, intended capability requirements and optional parameter inputs/outputs. These are contract fields, not implemented APIs or runtime providers. Strategy process requirements must be listed in the action's capability profile. Identity-based structural coverage is not semantic capability matching, type compatibility, resource allocation, or proof that a robot can execute the plan.

CapabilityAssessment records requirement, applicability context and Met/Unmet/Undetermined. Met and Unmet require explicit EvidenceRecord links. Evidence existence does not prove that the assessment is sound; acceptance is an application policy. The examples use Undetermined and contain no actual capability provider.

## Diagnosis and recovery

FailureSpecification describes a failure kind and a reusable FailureCriterion. It is not an observed episode. DiagnosticHypothesis links a candidate cause condition to failures, with optional supporting/refuting evidence and alternative explanations. Contradictory evidence may coexist; no scoring, ranking, root-cause inference or automatic status promotion is performed. Epistemic status stays on the associated KnowledgeClaim, not on a diagnostic description through the core's claim-only hasStatus property.

RecoveryOption specializes the core RecoverySpecification. It declares failures addressed, an applicability guard, intended restored condition and required capabilities. Retry budget is optional and unknown by default; explicit numeric budgets must be nonnegative integers. Zero means no retries. Budget units are governed by the parameter contract; the structural checker does not perform dimensional reasoning. No budget is decremented and no retry is performed.

The bridge can add a descriptive recovery strategy. This does not associate it automatically with a phase-graph recovery transition; such binding needs an explicit future adapter/profile. Guards and restored conditions may be descriptive rather than machine-evaluable expressions. Their existence is not a successful guard evaluation or a guaranteed recovery outcome.

## Examples

New pourRecordV6 and graspRecordV6 reference new action descriptions; earlier example data remains unchanged. Each has:

- Two-step perform/verify strategy, with abstract process-interface keys.
- Manipulation and verification capability requirements; an Undetermined assessment.
- One existing failure criterion reused as a described failure mode.
- A Proposed diagnostic explanation, without fabricated diagnostic evidence.
- A stabilization/reassessment recovery option with unspecified guard details and retry budget, plus a single-step recovery strategy.

The strategies are schematic. They omit grounding, timing, parameter transformations, resource locks and detailed guards. The diagnostics are synthetic schema examples, not source-backed explanations of actual spills or lost grasps. The intended recovery condition is stable task-reassessment conditions, not reversal of historical spillage or a claim of guaranteed safety.

Eight new Proposed claims record selected capability/strategy/hypothesis/recovery associations and synthetic authorship. As in earlier increments, provenance is illustrative rather than statement-complete for every structural triple.

## Validation and use

```bash
python tools/validate_strategy.py
python tools/project_strategy.py
python tools/validate_imports.py
python tools/reason_imports.py
python tools/check_strategy_integration.py
```

The readable output is examples/execution-recovery-projections.json. Reports use reports/strategy-*. The import classifier now covers ten aspects, four semantic bridges and the complete assembly: fifteen independently resolved closures. The larger AICOR/example check establishes ontology consistency only; full AICOR class satisfiability remains unverified.

See COVERAGE_AND_NEXT_STEPS.md at the package root for remaining ontology work and runtime integrations.

## Recorded results

All 23 new strategy/diagnosis checks passed. All 29 current import/reference/catalog checks passed. All 15 actual aspect/bridge/assembly import closures passed full classification with no unsatisfiable named classes. Combined AICOR/example consistency passed; full AICOR classification remains unverified. Thirty earlier Turtle files were verified unchanged; only the assembly adds the new module imports.

SHACL checks use the local action-module hierarchy; DUL's axioms are checked separately through full HermiT import-closure classification, avoiding unnecessary DUL expansion during repeated shape validation.
