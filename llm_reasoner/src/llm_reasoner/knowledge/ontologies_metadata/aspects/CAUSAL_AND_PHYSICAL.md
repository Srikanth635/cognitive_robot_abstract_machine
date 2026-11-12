> Import architecture updated in release 0.6.0: see `MODULAR_IMPORTS.md` at the package root. Earlier import-chain/open-file instructions below are superseded. Vocabulary descriptions still apply.

# Causal/mechanistic and physical-model aspects — 0.4.0

This release adds causal-mechanisms.ttl (`cm:`) and physical-models.ttl (`pm:`). Both use labelled HTTP IRIs under http://aicor.knowledge/ontologies/. Open physical-models.ttl to load its imports through the adjacent local catalog. Physical models import causal mechanisms, which import phases/transitions, conditions/outcomes, participants/relations, the core and DUL. The monitoring module is a sibling; it is included in validation but is not imported by this new chain.

## Causal structure

CausalModel specializes the core CausalSpecification. It contains CausalLinks, each relating an antecedent condition and consequent condition through a MechanismSpecification. Mechanisms require explicit assumptions and applicability descriptions in the ingestion profile. Enabling conditions and phase context are optional. A chain can order links with precedesLink; the validator rejects foreign links and cycles in this finite explanatory-chain profile. Feedback needs a later temporal model.

InterventionSpecification reuses an intended OutcomeSpecification. SuppressionSpecification relates an intervention to a link it is proposed to suppress. These describe hypotheses; they do not implement causal discovery, intervention calculus, counterfactual inference, or transitive causation. Correlation is not implicitly promoted to causality. No causal probability is generated.

## Physical descriptions

PhysicalPropertySpecification links an EntitySpecification, a PropertyKind concept and a reusable ParameterSpecification. Property kinds include mass, density, viscosity, geometry, friction coefficient and capacity. These vocabulary individuals represent kinds, not measurements. Unknown values stay absent. Geometry assets can be referenced through ModelArtifact, but geometry and material composition schemas are not fully developed in this increment.

PhysicalModelSpecification specializes the core's physical-model extension point. A model declares variables, assumptions and applicability, with optional equations and external artifacts. ModelVariable binds a symbol to an existing parameter. Symbols must be unique per model, and an equation can reference only variables declared by that model. Equations declare expression text and language; no equation parser, dimensional analysis, unit conversion, solver or stability analysis is implemented. Variable declarations can describe a partial model without equations.

ModelArtifact records an external identifier and version or digest; it is not downloaded or executed. SimulationStudySpecification describes a proposed study of a model and an outcome. ModelValidationRecord specializes EvidenceRecord and requires an external validation artifact plus scope. The inherited evidence rules still apply. Its existence records an evaluation, not that the model passed or applies universally. Simulation study specifications are not simulation-result evidence.

## Connection to the shared description

ActionDescription uses the existing hasCausalModel and hasPhysicalModel links. hasPhysicalProperty adds property descriptions. Models can reference the mechanism they intend to represent. Claims retain Proposed/Supported/etc. status and evidence independently of the structural model. No claim is promoted by a model link or a valid OWL classification.

Examples create pourRecordV4 and graspRecordV4 with new descriptions and previousVersion links. Earlier condition claims and temporal structures are reused. Eight new Proposed claims document the main model/mechanism/property/study associations. Annotation records identify synthetic schema authorship. Provenance is illustrative, not statement-complete for every template triple.

## Example meaning and limitations

The pouring example has a scoped volume-accounting model with the text equation:

`V_after = V_before - V_delivered - V_spilled`

It assumes an incompressible material and a boundary accounting for all source losses; evaporation, foam expansion and additional streams are excluded. This conservation accounting does not explain flow initiation, predict a trajectory or establish a causal law. Its causal link is a schematic proposed explanation context, pending grounding in physical mechanisms and sources. The density parameter remains unspecified and is not used by this volume equation.

The grasping example describes a proposed frictional contact mechanism under quasi-static assumptions, with an unspecified friction parameter. It has no force equation or computed grasp-stability result. Friction is context/contact dependent; the property description is not a universal property of the named object specification. Neither example supplies robot commands, numerical safety limits, simulator settings or empirical evidence.

Assumption and applicability content is currently human-readable descriptions, supported by the core lexical condition facilities. They are queryable nodes but are not automatically decidable expressions. A later author can attach structured expressions where the existing language is adequate; richer physical predicates will require explicit semantics.

## Validation and use

Run from the package root with requirements installed:

```bash
python tools/validate_physical.py
python tools/project_physical.py
python tools/reason_physical.py
python tools/check_physical_integration.py
```

The readable output is examples/causal-physical-projections.json. Reports are reports/physical-*. SHACL and additional checks cover required fields, variable ownership, symbol uniqueness and causal-chain topology. HermiT classifies all six aspect modules with the core and DUL. The larger AICOR integration is checked for ontology consistency only; full combined classification remains unverified.

This increment adds representation and validation contracts. Isaac Sim acquisition, equation execution, causal verification and evidence acceptance remain future integrations. Dynamic state-space and control models can subsequently specialize these variables and artifacts rather than introduce separate parameter identities.

Recorded result: all 17 new checks passed; full classification of six aspects/core/DUL passed with no unsatisfiable named classes; combined AICOR consistency passed. Prior Turtle files were verified unchanged.
