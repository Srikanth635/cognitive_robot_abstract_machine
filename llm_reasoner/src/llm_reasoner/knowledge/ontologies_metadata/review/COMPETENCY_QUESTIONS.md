# Competency questions — 0.10.0

> Historical review/catalog; its fixture versions and counts retain their original scope. Current coverage and consolidation findings: [0.21.0 review](coverage-0.21.0/REVIEW_REPORT.md).

Questions are about the content and reasoning limits of a selected knowledge record. A retrieved description is not an accepted world fact. The 24 SPARQL questions below are run against both pouring and grasping V7 fixtures. Detailed answers, RDF terms and tests are in `reports/competency-review.json`.

| ID | Question | What the answer means |
|---|---|---|
| [CQ01](../queries/competency/cq01.rq) | Which action kind, participant roles and entity specifications are described? | Role fillers are specifications, not grounded objects. |
| [CQ02](../queries/competency/cq02.rq) | Which goals, preconditions, success and failure criteria have formal expressions? | Unbound expression means no expression supplied, not an unsatisfiable condition. |
| [CQ03](../queries/competency/cq03.rq) | Which scoped facts or quantities does each action-level condition depend on? | State labels are relative scopes; episode/time identity is not encoded here. |
| [CQ04](../queries/competency/cq04.rq) | Which directly attached action parameters lack an explicit scalar value? | Bounded inventory of direct action parameters only; absence is unspecified, never zero. Nested model/controller parameters appear in CQ08/CQ11. |
| [CQ05](../queries/competency/cq05.rq) | Which guarded phase transitions belong to this description? | Topology and declared guards; no transition execution or termination proof. |
| [CQ06](../queries/competency/cq06.rq) | What must be observed and what response is specified for unknown evidence? | Declared monitoring dependencies; no sensor readings. |
| [CQ07](../queries/competency/cq07.rq) | Which causal links, mechanisms and enabling conditions are proposed? | Candidate explanatory structure; no learned causal probability or established law. |
| [CQ08](../queries/competency/cq08.rq) | Which equations, variables, parameters and validity scopes describe the physical models? | Equation text and bindings are retrievable; symbolic correctness is not checked. Equation/variable columns are independent inventories, not usesVariable assertions. |
| [CQ09](../queries/competency/cq09.rq) | Which model-validation evidence is explicitly included by a claim in this record? | Empty in current examples: no recorded validation evidence. A proposed simulation study is not a validation result. |
| [CQ10](../queries/competency/cq10.rq) | What objective policy, priorities and hard constraints are declared? | Priorities are descriptions, not a solved optimum. |
| [CQ11](../queries/competency/cq11.rq) | Which control-model roles, dimensions and timing parameters are specified? | Dimensions are representation sizes; missing sampling values or dynamics prevent interpreting this as a working controller. |
| [CQ12](../queries/competency/cq12.rq) | Which capabilities are required across profiles, recovery, affordances and sensing? | One shared requirement relation, retaining each specialized owner and property. This enumerates declared routes, not recursive requirements of arbitrary future subactions. |
| [CQ13](../queries/competency/cq13.rq) | Which strategy steps and abstract process interfaces are declared? | An interface intended to fulfill a requirement is not a matched implementation. |
| [CQ14](../queries/competency/cq14.rq) | Which failures and candidate diagnostic causes are described? | Candidate diagnosis; this does not identify an actual root cause. |
| [CQ15](../queries/competency/cq15.rq) | Which recovery options, guards, intended effects and retry budgets are described? | An absent retry value is not permission for unlimited retry; recovery is descriptive. |
| [CQ16](../queries/competency/cq16.rq) | Is a frame projection already derived from this exact record? | Current V7 has none; existing V6 artifacts must not silently be presented as V7. This review creates no projections. |
| [CQ17](../queries/competency/cq17.rq) | What qualitative force roles, tendencies and relative strength are supplied? | UnknownStrength/Tendency/Result are explicit unknowns, not a physical prediction. |
| [CQ18](../queries/competency/cq18.rq) | Which image-schema structures refer to the shared entity specifications? | Structural image-schema links only; no inferred spatial facts. |
| [CQ19](../queries/competency/cq19.rq) | Which entity facets, material states and geometry descriptions belong to this action record? | Action-specific facets share entity identities; unspecified material and geometry stay unspecified. |
| [CQ20](../queries/competency/cq20.rq) | Under which conditions and capability requirements is an affordance described? | An affordance specification is conditional, not an assertion that this robot can act now. |
| [CQ21](../queries/competency/cq21.rq) | How do sensorimotor boundaries connect phases, expected conditions and modalities? | Expected evidence is separate from observed evidence. |
| [CQ22](../queries/competency/cq22.rq) | What status, exact statement and source accompany claims included in this record? | Status is workflow metadata; proposed template claims do not become true through query retrieval. |
| [CQ23](../queries/competency/cq23.rq) | Which direct description links lack exact statement-level provenance in this record? | A bounded provenance-coverage audit of direct action triples. Missing a claim is not evidence that a triple is false. |
| [CQ24](../queries/competency/cq24.rq) | Which entities and parameter objects are reused across goals, models and perspectives? | Identity joins demonstrate reuse; they do not establish physical identity in a world. |

## Additional reasoning and completeness questions

| ID | Question | Current answer and review method |
|---|---|---|
| CQ25 | Does the success condition hold given supplied facts and quantities? | Bounded evaluation supports True/False/Unknown; absent data and incompatible units remain Unknown. Five explicit tests exercise evaluation and non-selection of unknown transitions. No observations are acquired. |
| CQ26 | Does satisfying the current pouring criterion establish transfer to the intended destination? | Not explicitly. Dependency inspection finds no destination-containment relation in the success expression. Existing condition/pattern vocabulary can express one; the template and quantity-referent contract need improvement. |
| CQ27 | Which causal link supports a diagnostic explanation of failure? | The current candidates are disconnected. Existing cause-condition identities do not join, and no dedicated diagnosis-to-causal-link association is supplied. An explicit association needs a defined meaning and provenance. |
| CQ28 | Can two concurrent action instances, conflicting sources and stale observations be distinguished reliably? | Not as a complete runtime contract. Record bindings and relative state scopes exist, but occurrence/time/freshness and immutable statement-status scoping are incomplete. This is a model/architecture review finding, not a passed runtime capability. |
| CQ29 | What would changing tilt, viscosity or force do, and which model is physically trustworthy? | Model descriptions can identify equations, assumptions and external artifacts; validated counterfactual computation, dimensional checking, model selection and physical evidence are not provided. No solver test was run. |
| CQ30 | Can a specified robot safely execute this action in its current world? | Not established. Assembly independence and absence of example world bindings are checked. Feasibility needs injected world/robot context, accepted requirements, process matching and physical validation. |

## Relationship to competence levels

Retrieval questions primarily exercise naming and describing. Identity joins, dependencies and comparative policies support structural analysis. CQ25 is bounded application of a supplied expression. Evaluating physical explanations and constructing executable controllers need accepted domain knowledge and external reasoning tools. Bloom-style labels are useful for organizing questions, but the ontology does not establish a competence level merely by having fields with those names.
