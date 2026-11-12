# Pouring success, diagnosis and physical/control contracts — 0.12.0

## Changes

This release addresses three findings of the competency review through two reusable extension modules and one optional semantic bridge. Their foundation remains DUL through the action core. The extensions are generic enough for other material-transfer actions and scalar model contracts; they do not contain robot/world dependencies.

| Module | Direct imports | Meaning |
|---|---|---|
| transfer-contracts | Conditions/outcomes | Explicit source, destination, substance, transferred portion, quantity referents and success criterion |
| model-contracts | Dynamics/control | Scalar quantity dimensions/units and bounded additive equation contracts |
| diagnostic-causal-bridge | Failure diagnosis + Causal mechanisms | Scoped association between a candidate diagnosis and a causal hypothesis |

The assembly now imports these modules. Registry and all catalogs resolve them locally. There are 28 registered ontology documents: seventeen aspect modules, five semantic bridges, the assembly, DUL, action core, provenance/context core, AICOR and the CRAM bridge.

## Pouring revision

`examples/pouring-contracts-v8.ttl` adds `pourRecordV8` and `pourActionV8`. Prior records remain unchanged. The canonical `ex:Pouring` action-kind concept is reused across records. The reachable descriptive components are copied to revision-specific IRIs so the modified success expression also propagates consistently to phase guards, verification monitoring, failure branches and strategy completion references. V7 and its meaning remain available to historical queries. The revised example is a synthetic specification, not sourced domain knowledge.

The transfer contract binds:

- Source and destination participant specifications.
- Substance specification and the specification of the transferred portion.
- Delivered amount, target amount, spilled amount and allowed spillage.
- A success criterion and an explicit partial-transfer mode.

Success requires all of:

1. Delivered amount reaches the requested target.
2. Spillage remains within tolerance.
3. The transferred portion is contained in the intended destination in the supplied post-state.

All numerical thresholds remain unresolved in the RDF example. Tests inject hypothetical values to exercise the expression semantics. Missing destination evidence yields Unknown; evidence for a different container cannot substitute for evidence about the intended destination. Negative destination evidence or an unmet quantitative criterion yields False under the existing three-valued expression evaluator.

Partial pouring does not require emptying the source. The quantity goal remains an amount subgoal, while the success criterion expresses the combined conditions. The vocabulary reserves a complete-transfer mode, but this reference profile does not define or verify complete emptying: that needs an explicit residual-source criterion and tolerance. No inference makes an entire substance specification occur in two physical containers; all nodes are descriptions, and the transferred portion is described separately.

`transfer_contracts.py` checks participant bindings, distinct source/destination, quantity references and the post-state destination relation. Its link checks are structural. The tests evaluate the actual V8 Boolean expression as well; presence of an atom somewhere in an arbitrary expression is not a general logical proof that success entails it.

## Diagnosis

A `DiagnosticCausalAssociation` connects a diagnostic hypothesis, its candidate cause condition, a causal link and an applicability specification. The cause condition must match both the diagnostic candidate and causal antecedent. The Python integrity check also requires that the causal consequent match a criterion detecting a failure the hypothesis explains.

The example adds a separate proposed spill causal hypothesis. It does not repurpose the existing normal-transfer mechanism or assert that it identified a real failure's root cause. All statements about the association and the new causal link remain Proposed and carry explicit synthetic-example evidence. Real diagnostic acceptance requires episode evidence, competing explanations and an explicit acceptance policy.

## Physics and control

A `QuantityContract` binds a scalar parameter to:

- A seven-entry SI dimension vector, ordered length, mass, time, electric current, temperature, amount of substance, luminous intensity.
- One exact unit IRI.

The dimensional vector is different from the existing controller vector-entry dimension, which counts scalar components. Volume uses `3 0 0 0 0 0 0`. The example selects cubic metres as a contract unit; no measured parameter value is asserted and no unit library is fetched.

An `AdditiveEquationContract` gives a machine-readable scalar interpretation of the example's existing volume accounting equation:

`source_after = source_before - delivered - spilled`

Each signed term points to a bound parameter; the contract references the original model-equation description and its applicability. Coefficients are dimensionless. This is a bounded arithmetic representation, not parsing of arbitrary mathematical text. It inherits the example's assumption that source losses are accounted as delivered or spilled volume. Internal retention, evaporation, compressibility and other unmodelled effects may invalidate that assumption in a real application.

The reference checker reports:

- Unknown when required quantity declarations or supplied values are absent.
- Incompatible dimensions when terms cannot be added dimensionally.
- Conversion required when dimensions agree but exact unit identifiers differ.
- An accounting residual and comparison against an explicit absolute tolerance when all values and units are supplied.

The default arithmetic tolerance is exact zero; a caller can supply a finite nonnegative tolerance in the contract unit. No implicit unit conversion occurs. Dimensionally consistent accounting does not establish a physically adequate model, valid fluid behavior, nonnegative volumes or empirical validity.

The control inspector now reports missing quantity/unit contracts for assigned variables, checks declared sampling/horizon parameters for time dimensions, and retains the existing missing-equation, timing and objective-setting diagnostics. The current controller still lacks state-evolution/output equations and numerical settings. The release does not synthesize a controller, optimize a policy, verify stability, simulate liquid, calibrate parameters or authorize execution.

## Provenance

Every one of the 435 descriptive non-type/non-label links in the new revision has an exact included `ac:Statement` and Proposed claim. Claims refer to a revision bundle as their descriptive component; the statements retain exact subjects, predicates and values. Type declarations and labels are excluded from this coverage measure. Evidence explicitly identifies a synthetic contract example. Existing records' provenance is not rewritten.

This supplies concrete exact-statement coverage for the new example, while scientific source acquisition, contextual acceptance assessments and deterministic immutable storage remain separate work. The existing context module can qualify assessments of these claims in future records.

## Validation

Run from the package root with requirements installed:

```bash
python tools/validate_pouring_contracts.py
python tools/review_competency.py
python tools/validate_context.py
python tools/validate_imports.py
python tools/reason_imports.py
python tools/check_strategy_integration.py
```

`validate_pouring_contracts.py` runs the new contract cases, loads all example graphs, validates all aspect shapes plus the core shapes, checks expression structure and phase/monitor dependencies, and writes `reports/pouring-contract-validation.json`. It tests intended/wrong/missing destinations, partial transfer, spillage, mismatched causal links, unit/dimension failures, unknown model inputs, arithmetic residuals, controller diagnostics, and exact statement coverage. The prior competency questions remain pinned to V7 so they continue to document that historical record's original gaps.

Fresh results: 31 new contract checks, 93 historical competency assertions, 26 context checks, four scoped query cases, 38 import checks, and 23 modular classifications passed. Combined AICOR/examples consistency passed. Results are indexed in `reports/release-0.12.0.json`. Full AICOR named-class satisfiability remains unverified; the combined integration check is consistency only. Generic modular classification and syntactic/runtime contract checks do not establish physical validity.

## Remaining work

Next, populate a small sourced material-transfer domain profile and define model selection, calibration and simulation-validation requirements. An Isaac Sim adapter must document whether and how it models the relevant fluid phenomena before its results can support these claims. Complete-transfer semantics, uncertainty propagation, general dimensional algebra, unit conversion, equation solving and actual control synthesis remain explicit limitations. Working projections remain deferred.
