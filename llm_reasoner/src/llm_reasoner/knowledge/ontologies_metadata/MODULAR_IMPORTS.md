> Historical release 0.6.0 refactor report. See ARCHITECTURE.md for the current dependency table and counts.

# Modular ontology imports — release 0.6.0

This release refactors dependencies. It adds no action concepts or world facts. All concept/property IRIs and example files are preserved. Complete-domain axioms were compared with 0.5.0 and are unchanged after excluding ontology-header/import/version metadata.

## Direct imports

| Ontology | Direct dependencies |
|---|---|
| Action-description core | DUL |
| Participants–relations | Core |
| Conditions–outcomes | Participants–relations |
| Phases–transitions | Conditions–outcomes |
| Monitoring requirements | Conditions–outcomes |
| Causal mechanisms | Core |
| Physical models | Core |
| Objectives–trade-offs | Core |
| Dynamics–control | Physical models + Objectives–trade-offs |
| Unified AICOR | DUL |
| CRAM action bridge | Core + Unified AICOR |

Indirect imports supply core/DUL and any necessary participants vocabulary. Dependencies follow the terms actually referenced by the ontology. Monitoring's base vocabulary attaches checks to core descriptive components; checking that a PhaseEntry monitor is attached to a phase is a profile-level constraint in the temporal validator, not a requirement to import phase topology for every monitor.

Objective constraints reference the core's ConstraintSpecification. If an application uses the structured expression language to fill those conditions, it must load conditions–outcomes too. The full assembly includes it. Importing a vocabulary does not by itself load all SHACL profiles or Python execution semantics.

## Three optional semantic bridges

These files are under `integrations/`. Declarations retain their previous property IRIs; their defining axioms moved to the bridge, so existing data needs no URI migration.

| Bridge | Preserved property | Imports |
|---|---|---|
| causal-outcome-bridge.ttl | `cm:intendedOutcome` | Causal mechanisms + Conditions–outcomes |
| physical-outcome-bridge.ttl | `pm:testsOutcome` | Physical models + Conditions–outcomes |
| causal-physical-bridge.ttl | `pm:realizesMechanism` | Causal mechanisms + Physical models |

For example, a physical-property consumer can load physical-models.ttl without causal or phase concepts. A consumer needing the model–mechanism relationship loads causal-physical-bridge.ttl as well. The intervention and simulation-study classes remain available in their base modules; outcome links gain their semantics when the corresponding bridge is loaded.

Preserving property namespaces while moving their declarations is intentional. The namespace identifies a stable term; the bridge is its new defining module. Merely mentioning a property in example data does not load its defining ontology automatically.

## What to open

**For the comprehensive designator model, open `integrations/all-action-aspects.ttl`.** It intentionally imports all eight aspects and all three optional bridges. It includes monitoring, which is a sibling dependency, rather than relying on one long chain. It contains only ontology metadata and imports, with no additional domain axioms.

For a smaller consumer, open the individual ontology and any bridges it needs. The existing CRAM bridge remains optional and separate from this generic assembly. To inspect the complete generic/AICOR combination, load both the generic assembly and the CRAM bridge in an editor or use the integration checker.

No new flattened duplicate is required. The root, core, aspects and integrations folders have XML catalogs mapping ontology IRIs to local files. `ontology-registry.json` supplies the same mappings to Python. HTTP IRIs are identifiers; hosting is not configured.

## Verification

- Every package-local term reference resolves within its actual imported closure.
- Every local catalog entry points to a file declaring the requested ontology IRI.
- Independent loading checks confirm that objectives, physical models and causal mechanisms do not drag in unrelated aspects, and monitoring does not load phase topology.
- The combined eight-aspect/three-bridge axiom graph is isomorphic to the previous eight-aspect graph after removing ontology metadata. Moving declarations therefore preserves the complete model.
- Each of the eight aspects, three semantic bridges and the full assembly was separately classified with its actual import closure: all 12 passed, with no unsatisfiable named classes.
- Existing physical-model and control validation suites were rerun, as was combined AICOR/example consistency. Full AICOR class satisfiability remains unverified.

Reports: `reports/import-refactor.json`, `import-validation.json`, `import-classification.json`, plus the refreshed physical/control validation and integration reports. Older release reports retain their original historical scope.

## Reproduce

With the package requirements installed and Java available:

```bash
python tools/validate_imports.py
python tools/reason_imports.py
python tools/validate_physical.py
python tools/validate_control.py
python tools/check_control_integration.py
```

To repeat the old/new axiom comparison, pass the previous 0.5.0 ZIP as the sole argument to validate_imports.py. Without it, the script explicitly skips that one historical comparison; it still validates the current registry, closures, independent loading and catalogs. The recorded release comparison used the original 0.5.0 archive.

`tools/ontology_imports.py` resolves registered imports locally and fails on unresolved imports. `reason_imports.py` validates the real closures rather than flattening an arbitrary file list. Independent closure classification covers schema consistency/class satisfiability, not the soundness of physical theories or executable action readiness.
