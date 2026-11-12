# Aspect foundation completion/review — 0.9.0

This release completes the first representation pass across the planned aspect families. It does not claim complete action knowledge, full scientific theories or validated robotic competence. No new working projections are introduced. Earlier projection demonstrations remain available but are deferred for further development.

## New aspect modules

| Module | Imports | Scope |
|---|---|---|
| force-dynamics.ttl | Core | Agonist/antagonist participants, qualitative tendency, relative strength, described resultant and applicability |
| image-schemas.ttl | Core | Containment, source–path–goal, support, contact and ordered symbolic path elements |
| entities-materials-affordances.ttl | Physical models | Record-specific entity descriptions, materials/composition/state, geometry/openings/assets, physical-property links and conditional affordances |
| sensorimotor-events.ttl | Core | Phase-boundary conditions, sensory expectations/modalities, control points and mismatch specifications |

The four modules have labelled HTTP IRIs. The optional AICOR model is not a dependency. Entity/material descriptions intentionally import physical models to reuse PhysicalPropertySpecification instead of duplicating that concept. Sensorimotor boundaries reference the core PhaseSpecification, avoiding an unnecessary phase-topology import.

## Force-dynamic scope

The qualitative schema distinguishes focal/opposing participants, tendency, comparative strength and a described outcome. These conceptual distinctions are informed by Talmy's force-dynamic account [1]. This engineering profile is deliberately bounded: roles can reference shared entity specifications; strength/tendency/result can be Unknown. Balanced is an explicit profile extension, not a claim that every category is a faithful exhaustive Talmy inventory.

There is no inference from qualitative strength to Newtonian force or actual motion. No force magnitudes, probabilities or resultant motion are fabricated. The sample force assignments are candidate descriptive patterns under Proposed claims; domain evidence is needed to justify them. Physical object identity can be shared by distinct role nodes, but the agonist/antagonist role nodes must be distinct in the current validation profile.

## Image-schema scope

These are explicitly defined schematic representations, not an exhaustive implementation of image-schema theory. Schema role fillers reuse EntitySpecification IRIs. Source and goal are conceptual entity/location specifications, not measured poses. A path may be unspecified; if path elements are provided their positive positions must be contiguous and unique. Support does not imply stability, contact does not imply force, and containment does not imply a numerical fill level.

Broader schemas such as balance, linkage, center/periphery and schema transformations are outside this first profile. No spatial-reasoner inference is supplied.

## Entity and material semantics

EntityDescription is a record-specific facet referencing one shared entity specification through describesEntity. Its material, geometry and affordance descriptions attach to the facet, not globally to the old entity node. This supports later enrichment without changing earlier records' shared entity structures.

MaterialSpecification describes state and optional constituents. Composition cycles are rejected. Detailed chemical mixtures, proportions and constitutive material laws are not modelled. GeometrySpecification can reference dimensions through existing parameters and external assets through identifier/version. OpeningSpecification provides a geometric specialization; detailed boundary representations, meshes and aperture shapes remain external or future extensions. Cyclic opening descriptions and unversioned external assets are rejected.

Physical-property links reuse the physical model vocabulary and must describe the same entity as the facet. AffordanceSpecification is relative to an action-kind concept, capability requirements and context conditions. Its existence is not an assertion that an arbitrary agent can perform that action. Condition evaluation and capability-provider matching remain external.

## Sensorimotor scope

Flanagan, Bowman and Johansson describe manipulation phases organized around sensory-signalled events and comparison between predicted and actual sensory consequences [2]. This module represents the associated descriptive distinctions: a boundary condition between two different phases, expected sensory conditions/modalities, and a control point at which a mismatch criterion can be applied.

All nodes are descriptions. An expectation is not an observation or a completed event. No sensor data is created, and no mismatch is diagnosed without supplied evidence and a specified comparison rule. Observations are described as abstract capability requirements rather than attached to a particular sensor API. Mismatch conditions may remain lexical descriptions; there is no temporal detector or signal-processing implementation.

The validator checks that control-point boundaries attached to an action refer to that action's declared phases. It does not establish temporal occurrence, continuous invariant preservation, correct sensory predictions or a sufficient controller.

## Examples and epistemic status

examples/remaining-aspect-knowledge.ttl adds new action descriptions and pourRecordV7/graspRecordV7. Earlier v6 records and example files remain untouched. The new records reuse entity, capability, phase and condition identities, add record-specific facets and attach candidate perspectives. Eight new claims remain Proposed with synthetic-example evidence. Material state and force-strength/result are explicitly Unknown; no material measurement or world observation is inferred.

This is schema-test knowledge, not automatic LLM generation or source-validated action facts. Provenance demonstrates selected associations and remains incomplete at individual structural-triple granularity.

## Validation/review

Run:

```bash
python tools/validate_aspect_completion.py
python tools/validate_imports.py
python tools/reason_imports.py
python tools/check_strategy_integration.py
```

New SHACL and integrity checks cover required roles, property/context references, composition cycles, path ordering, assets and boundary consistency. Full HermiT classification covers all actual aspect/bridge/assembly import closures. Combined AICOR checking remains ontology consistency only; full AICOR class satisfiability remains unverified.

## Theory sources and implementation ownership

[1] Leonard Talmy, Force Dynamics in Language and Cognition, author-hosted chapter: https://www.acsu.buffalo.edu/~talmy/talmyweb/Volume1/chap7.pdf

[2] J. Randall Flanagan, Miles C. Bowman, Roland S. Johansson, Control strategies in object manipulation tasks (2006), DOI 10.1016/j.conb.2006.10.005: https://robots-that-learn.github.io/resources/FlanaganBowmanJohansson06.pdf

The ontology decomposition, validation rules, unknown categories, record facets and engineering restrictions are design choices of this package. Sources motivate distinctions; they do not certify our axioms or examples. Primary-source content is paraphrased only; no article text is bundled.

Recorded result: all 22 new aspect checks and 34 import/reference/catalog checks passed. All 20 actual aspect/bridge/assembly closures passed full classification with no unsatisfiable named classes. Combined AICOR/example consistency passed; full AICOR classification remains unverified. Earlier Turtle files were verified unchanged except for the assembly import additions.
