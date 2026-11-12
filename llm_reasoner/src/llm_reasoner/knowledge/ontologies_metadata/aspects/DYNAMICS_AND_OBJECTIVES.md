> Import architecture updated in release 0.6.0: see `MODULAR_IMPORTS.md` at the package root. Earlier import-chain/open-file instructions below are superseded. Vocabulary descriptions still apply.

# Dynamics/control and objectives/trade-offs — 0.5.0

This increment adds two reusable modules. `planning/objectives-tradeoffs.ttl` imports physical-models; `physics-control/dynamics-control.ttl` imports objectives-tradeoffs. Open dynamics-control.ttl using the adjacent XML catalog to inspect this stack. The monitoring module remains a sibling; validation includes all eight aspects. HTTP IRIs and English labels identify all new terms.

## Objectives and trade-offs

ObjectiveSet contains ObjectiveTerms and one declared aggregation policy. Each term references a metric ParameterSpecification and a Minimize/Maximize direction. Hard constraints are separate links to the existing ConstraintSpecification with a structured condition expression; a preference weight cannot override them.

Supported descriptive policies:

- Lexicographic: unique explicit ranks, smaller ranks first. No weights are accepted on these terms.
- WeightedSum: every term declares a weight parameter and normalization specification. Known weights must be nonnegative and cannot all be zero. Normalization scales must be positive when known. Numeric weights and scales can remain unspecified.
- Pareto: no weights or ranks; this describes a multiobjective policy without supplying a selection procedure.

Normalization refers to a reference scale, conventionally metric/scale; metric and scale unit compatibility is an adapter responsibility. Direction, rank and weight serve distinct purposes. No default equal weights, penalty coefficients, optimizer, Pareto-front computation or unit conversion is supplied.

## Dynamics and control

DynamicalModelSpecification specializes the existing PhysicalModelSpecification. It reuses ModelVariable, ModelEquation, assumptions and applicability. VariableAssignments declare State, Input, Output or Disturbance roles with one-based entry positions and explicit dimensions. Positions order vector entries; dimension counts scalar components of an entry. The same variable may appear in different roles with consistent dimension. A variable cannot appear twice within one role. Total vector dimensions are calculated from assignments.

State, input and output roles are required by the current ingestion profile. Dynamics can remain partial: evolution and output equations are optional descriptive fields, but missing equations are surfaced by the completeness inspector. When provided, such equations must also be declared by the model and reference its own variables. No dynamics equation, Jacobian, equilibrium, observability, controllability or stability result is computed.

ControllerSpecification specializes the core ControlSpecification. It declares a controlled model, controller family, sampling-period parameter and ObjectiveSet. Optional prediction horizons, state/input constraints and parameter adaptations can be described. Known time values must be positive; temporal units and sampling/horizon compatibility require downstream validation. No robot DOF, torque limits, rates or horizons are invented.

AdaptationSpecification links an update parameter and a validity condition. The checker requires the parameter to belong to the dynamic model or the directly referenced plant model. It does not implement online identification or execute parameter updates. Validity conditions can remain descriptive; they are not assumed true.

## Examples and shared identity

`examples/control-actions.ttl` adds new v5 action descriptions and records. Existing participants, conditions, causal/physical models and temporal structures remain referenced. Earlier records and files are unchanged. Eight new claims stay Proposed with synthetic-example evidence.

Pouring reuses the delivered-volume variable as a scalar state and output, and declares an abstract transfer-command input. Its objectives minimize spill before completion time using explicit lexicographic ranks. The existing current-spill expression is reused in a hard constraint. This is a declared model preference, not a validated task policy.

Grasping declares a conceptual hold metric and abstract hold-command input. It describes a weighted trade-off between maximizing that metric and minimizing effort. Weights, normalization scales, sampling periods and evolution equations remain unknown. Hold-metric meaning, measurement definition and unit semantics require further domain work. Both models are partial examples; no real controller, control law or prediction is provided.

`examples/control-projections.json` shows objective policies, vector dimensions, unresolved fields and provenance. `executionReadinessAssessed` is always false. Filling the listed missing fields does not establish executable readiness; the list is a limited completeness inspection, not a comprehensive controller certification.

## Validation and commands

```bash
python tools/validate_control.py
python tools/project_control.py
python tools/reason_control.py
python tools/check_control_integration.py
```

SHACL and structural checks cover vector membership/ordering/dimensions, objective aggregation settings, constraints, time positivity and adaptation targets. HermiT classification covers eight aspect modules, core and DUL. Combined AICOR validation checks ontology consistency only; full combined classification remains unverified.

Reports are under `reports/control-*`. Equation parsing, dimensional analysis, control synthesis, trajectory generation, optimization, simulation acquisition, LLM population and EQL/runtime adapters remain future integrations. Next aspects can address capabilities, execution strategies and richer failure diagnosis, followed by perspective-specific schemas and projections.

Recorded result: all 21 new checks passed. Full classification of eight aspects/core/DUL passed with no unsatisfiable named classes. Combined AICOR consistency passed. All prior Turtle files were verified unchanged.
