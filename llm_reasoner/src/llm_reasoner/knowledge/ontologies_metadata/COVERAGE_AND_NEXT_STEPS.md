> Superseded by ASPECT_COVERAGE.md for the current aspect inventory and remaining work. This is the earlier planning record.

> Historical coverage snapshot. Current cross-release assessment and next steps: [review/coverage-0.21.0/REVIEW_REPORT.md](review/coverage-0.21.0/REVIEW_REPORT.md). Preserve the dated statements below as historical context.

# Coverage and remaining work — release 0.8.0

We have eleven generic knowledge-aspect vocabularies, a shared core, DUL alignment, optional bridges and conceptual examples. These are draft representation/validation capabilities; they do not constitute fully populated knowledge or verified robot competence.

| Area | Current representation | Still needed |
|---|---|---|
| Participants and relations | Roles, entity specifications, predicate signatures | Detailed materials, geometry, substance state and affordance schemas |
| Conditions and outcomes | Boolean expressions, quantity comparisons, intended transitions | Quantifiers, richer temporal/relational expressions and context semantics |
| Phases and transitions | Branching phase graphs and recovery paths | Hierarchical/concurrent phases, durations, bounded operational retry semantics |
| Monitoring | Observation dependencies, explicit true/false/unknown responses | Temporal event detection, freshness, uncertainty and actual observation adapters |
| Causal mechanisms | Scoped hypotheses, chains, interventions/suppression | Source-grounded mechanisms, causal identification and counterfactual semantics |
| Physical models | Properties, variables, equation/artifact descriptions | Rich material/geometry knowledge, parsed equations, unit/dimension validation and calibrated models |
| Objectives | Priorities, weighted/Pareto policies and hard constraints | Operational policy selection, normalized metric definitions and optimization |
| Dynamics/control | State/input/output roles and controller descriptions | Full dynamical models, hybrid modes, model identification, synthesis and validation |
| Capabilities/strategies | Abstract contracts, partial-order steps, contextual assessments | Provider matching, parameter bindings, scheduling, process adapters and actual planning |
| Failure/diagnosis/recovery | Criteria, candidate causes, evidence links and recovery options | Episode-based diagnosis, ranking, recovery selection and bounded execution |

## Aspect families not yet modelled in detail

1. **Advanced frame semantics:** a bounded frame/projection module now exists with slots, single inheritance, defaults, explicit exceptions and provenance. Multiple inheritance and procedural attachments remain absent.
2. **Force dynamics:** force participants, tendencies, resistance, enabling/preventing and balance patterns. Reuse causal and physical identities; qualitative patterns must not silently imply measured forces.
3. **Image schemas and spatial transformations:** containment, source–path–goal, support, contact and transformations as structured perspectives, beyond the current predicate vocabulary.
4. **Richer entity/material/affordance knowledge:** compositional geometry, openings, capacity, material behavior, substance properties and action-dependent affordances.
5. **Sensorimotor event/control-point knowledge:** meaningful event boundaries, expected sensory changes and links between phases and control, beyond generic condition monitoring.

Some families overlap. Start with competency questions and reuse existing concepts before adding subclasses. Optional cross-aspect links belong in bridges when they would otherwise create unnecessary dependencies.

## Reusable action-domain modules

Not yet built: formal action-family modules such as material-transfer, contact manipulation, pouring and grasping. Current action examples are conceptual test data, not mature domain ontologies. Domain modules should specialize/reuse the generic aspects rather than copy them into separate incompatible action representations.

## Infrastructure still required

- Ontology-guided LLM annotation/extraction with sources, validation, abstention and statement-level provenance.
- Persistent versioned shared records, multi-annotator updates and evidence acceptance/conflict policy.
- Full perspective projection schemas and transformations; existing JSON views demonstrate inspection only.
- EQL/krrood query integration and competence-level query evaluation. Bloom's taxonomy guides evaluation, not independent copies of action knowledge.
- SDT/world injection, entity binding resolution and capability-provider matching.
- CoraPlex/CRAM executable projection contracts and process adapters.
- Isaac Sim study execution, scoped run evidence, model calibration and evaluation.
- Full AICOR satisfiability review and code alignment; current combined consistency is a weaker check.

## Recommended next increment

The first frame perspective and working projection now demonstrate reuse of the pouring/grasping shared records. Next develop force dynamics, followed by image schemas, preserving participant and condition identities. Before claiming complete action knowledge, create action-domain modules and populate/evaluate them against explicit competency questions and evidence.
