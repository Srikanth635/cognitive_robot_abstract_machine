# Coverage and consolidation review — 0.21.0

Review date: 12 November 2026. Ontology baseline: cumulative 0.20.0. This is a review and query/documentation consolidation release, with **40 registered ontology documents unchanged**. It does not add another knowledge aspect or domain.

**Assessment:** the architecture and initial broad aspect inventory match the original design. They are ready for targeted refinement, sourced population and integration contracts. They do **not** yet establish complete CoraPlex action coverage, all scientific competency answers, or executable and physically validated comprehensive designators. The largest remaining work is semantic detail, evidence and integration, rather than another broad layer of ontologies.

## What was compared

1. The eight original objectives in this conversation: parallel LLM/EQL generation, common-sense adaptation, unknown resolution, comprehensive multiperspective knowledge, physics/simulation, EQL/KRROOD access, DUL/logic, and world independence with injected SDT information.
2. The package's historical 24 descriptive query questions and six additional boundary questions, **LOCAL-CQ01–30**. Their original catalog and immutable V7 fixtures remain historical evidence, not a description of every current profile.
3. The repository's separate **CQ-001–065** scientific contract. In this review these are called **REPO-CQ-001–065** to avoid conflating them with the local numbers. This more demanding contract includes detailed sensorimotor, learning and perceptual questions beyond the original Lisp illustrations.
4. Every registered ontology, its role and suggested improvement; all canonical CoraPlex action/motion source files in the pinned revision; selected implementation tests; repository competency, architecture and status documents.

The upstream repository is [Srikanth635/cognitive_robot_abstract_machine](https://github.com/Srikanth635/cognitive_robot_abstract_machine), branch `claude/cram-aicor-domain-model`, pinned to commit `4cf1da42be36c752834b6ed325faa092b7f5a1b8` dated 7 November 2026. The recursive source tree had 5,162 entries and was not truncated. The included source manifest records file links and hashes. This is static source analysis: no upstream ROS, SDT, Giskard, robot or simulator execution was performed.

The canonical subtree contains **28 locally concrete action candidates and 13 locally concrete motion candidates**. The AST extractor excludes locally abstract bases, helper classes and commented-out `SearchAction`. Its abstract-method analysis resolves inspected classes only; external mixin contracts, dynamic registrations and runtime availability are not proven. Experiment-only actions are outside this inventory. Inherited fields from inspected bases are included; external mixin fields still need adapter review.

## Coverage is assessed at separate levels

| Level | What the package currently establishes |
|---|---|
| Representation | Shared action descriptions, entity/parameter identities, 17 aspect/contract modules, seven primitive domains, compositions and refinements |
| Knowledge population | Synthetic Proposed templates and a fictional numerical pouring illustration; these are not a source-backed action knowledge base |
| Structural and bounded reasoning | Offline imports, shape/integrity checks, generic OWL results from 0.20.0, and bounded inspectors/tests on supplied inputs |
| Evidence acceptance | Exact claim/provenance and context structures; selected inspectors have bounded freshness/scope policies, but no general evidence acceptance or world-belief publication |
| Physical and executable competence | Not established for this standalone ontology package; suitable models, measurements, bindings, provider mappings and adapters remain necessary |

These levels should stay separate in documentation, extraction schemas and future answer APIs. A named concept is not populated knowledge. A Proposed statement is not an observation. A consistent graph is not physically correct. An action family match is not an executable adapter.

## Original objectives

The complete eight-row assessment is in `original-objectives.csv` and JSON.

The strongest completed objective is architectural: a shared DUL-based, world-independent representation with reusable domain knowledge and separate optional integration modules. Shared references support an executable subset, a frame, force dynamics and other future projections without maintaining unrelated versions of the same action.

The LLM can propose knowledge without task-specific training, but adaptation to unseen objects, robots and environments remains a hypothesis to evaluate. The package does not yet demonstrate accuracy, transfer improvement or calibrated uncertainty relative to a trained probabilistic backend. Preserve both backend contracts and measure held-out performance, grounding failures and appropriate abstention.

The standalone EQL/KRROOD integration, injected world binding, provider matching, accepted answer production and Isaac Sim validation are deferred. The upstream `llm_reasoner` already documents its own resolver, EQL, storage and measurement work; this review does not claim those features are absent from the repository. It also does not establish that those components currently consume this separate ontology package.

## CoraPlex action coverage findings

`coraplex-action-coverage.csv` gives a row for all 41 candidates, including source location, available field inventory, nearest ontology family, mapping gap and runtime-verification status. None is labelled a verified executable ontology mapping.

| Finding | Existing coverage | Required change |
|---|---|---|
| Mixing | Generic tool interaction only | Add a bounded mixing refinement: material/containment, working part, technique, intended circulation/homogeneity or other specified effect, monitoring and termination. Do not assert that every mixing action achieves uniformity. |
| Pick-and-place without navigation | Pickup and placement domains | Add a no-navigation recipe variant. The current pick/carry/place recipe must not impose base travel on this source action. |
| Full transport | Holding plus mobility composition and pick/carry/place recipe | Map navigation, facing/looking and arm-parking wrappers. `TransportAction` is broader than sustained carrying. |
| Move-and-act and face/look wrappers | Constituent mobility, perception and manipulation domains | Add bounded compositions for the actual wrappers, preserving phase contexts and control handoffs. |
| Elevator navigation | Already has inter-level/elevator profiles, door/cabin guards, entry/ride/exit and explicit cabin exit | Map source frame-parent reattachment and monitor conventions; refine timeout/recovery and facility-provider contracts. A new elevator domain is unnecessary. |
| Open/close | Articulated mechanisms and manipulation | Map grasp-handle, mechanism motion, release and retreat. Source position cutoffs are implementation conventions, not universal physical thresholds. |
| Cutting/wiping/pouring | Specific semantic profiles exist | Map fields and distinguish motion completion from intended physical outcome and evidence of that outcome. |
| Pressing/insertion/bounded search | Useful ontology extensions | No matching named active canonical action was found in the scoped action inventory; commented search is not counted as active. This is not a claim about every experiment or external executor. |

The pouring source builds a pre-pour TCP pose and tilted TCP pose. That class alone does not measure delivered volume, spilling, destination containment or foam quality. Wiping can tolerate a no-progress condition when the final waypoint is sufficiently close; that does not demonstrate the ontology's surface effect or interval-qualified coverage. Cutting's tool path does not itself prove material separation. Broader repository measurement consumers may check additional effects; they must be explicitly connected and their result scope preserved.

Source `OpeningMotion`/`ClosingMotion` have motion-chart construction paths. A no-op `perform` body alone would be insufficient evidence that a motion is unavailable; this review does not make that inference. Likewise, reading source tests establishes intended checks, not that those tests passed on this machine.

## Competency-question review

### Historical local questions

The historical suite still passes **93 assertions** on V7 pouring/grasping snapshots. Several assertions deliberately preserve old gaps, including missing destination containment and disconnected diagnostic causes. Later V8 contract work addresses these structural points. Rewriting V7 or treating its gap assertions as current capability would both misrepresent the evidence.

`local-competency-coverage.csv` assesses all 30 local questions and records these distinctions. Existing per-domain inspectors and scoped queries provide further bounded coverage, but there was no single current release-wide registry linking every question, fixture, query and validator.

Two query improvements are included:

- `queries/current-action-conditions.rq` follows explicit subactions and owned recipe-step descriptions and exposes both formal-expression and lexical-contract fields. An unbound `co:hasExpression` on an ordered recipe is not proof that its condition has no structured semantics.
- `queries/current-action-capabilities.rq` retrieves requirements attached directly to those actions or their capability profiles, retaining action/owner identity. It handles nested action descendants but does not claim to enumerate every monitoring, recovery or affordance owner. Recipe-context output is only present where the action explicitly declares that property. The query enumerates the eight currently reviewed requirement properties and verifies their generic requirement hierarchy; new properties need an explicit query review.

These are descriptive queries, not new projection engines or arbitrary lexical evaluators. The selected-record wrapper rejects unknown and ambiguous roots. Regression cases check unrelated-record/owner leakage, cycles, nested stages and current exact statement coverage. They do not turn descriptions into accepted evidence.

### Scientific repository questions

`repository-scientific-competencies.csv` retains the wording of all 65 questions, maps relevant modules and identifies remaining work. These are **semantic review judgments**, not 65 automated tests or 65 implemented answers. No numerical completion percentage is reported because the questions require different depths of representation and evidence.

Broad roles, phases, conditions, sensorimotor event slots, material transfer, failure/recovery and provenance have useful representations. Important remaining details include:

- Initial-state-conditioned optional phases, event comparison across occurrence/time/location/magnitude/direction, and typed correction/continuation semantics.
- Estimation memory, cue-dependent updates, efference copies, model activation while grasping, and the distinction between discovering a mapping and adapting it.
- Grip/load temporal coupling, unexpected-weight lift-off profiles and predictive response to external impact.
- Proactive gaze timing/termination, actor–observer modality differences, learning stages and perceptual attenuation.
- Source-backed regime-specific flow, continuous feedback, changing contents/COM/torque/grip predictions and alternative transfer termination conditions.
- Evidence grades per answer part, source-claim alignment, runtime module-selection traces and applicability to accepted participant/material/initial-state bindings.

Some can be populated using existing vocabulary; some need small aspect extensions. A generic equation or observation slot is only an extension point for detailed scientific mechanisms. Do not create a large new domain for every question. First decide which scientific questions are required for this project's next demonstrator, then add the smallest tested semantics that those questions need.

## Consolidation decisions

1. Keep the 40-module import architecture and existing entity identities. All **87 prior RDF/OWL files are byte-identical** to 0.20.0. No classes were renamed, merged speculatively or added during this review.
2. Keep generic aspects, domain specializations, compositions, refinements, bridges and assemblies distinct. Integration modules express cross-module semantics and optional dependencies; they are not duplicate domain models. The historical AICOR overlap review deliberately withheld equivalence for superficially similar designator/action concepts with different or unverified restrictions; no label-based merge is justified by this review.
3. Keep historical reports and fixture versions. Add a current review index and dated supersession notices instead of silently changing old meaning. Module documentation remains scoped to its release; later chronology is in the current index and handoff updates.
4. Treat workflow status, evidence grade, source support, observed outcome and runtime acceptance as separate concepts. In particular, `Supported` must not be translated automatically to `Measured`.
5. Preserve stage and occurrence contexts. Pickup's held-object outcome and placement's released-object outcome can both be valid at different stages. Never flatten all historical PostState facts into one state.
6. Document bounded evaluator dispatch. Recipes intentionally use a qualified contract instead of a flattened Boolean PostState expression. Generic expression inspectors must report unsupported contract types explicitly.
7. Keep optional AICOR compatibility unresolved. The latest 0.20.0 optional combined reasoning attempt timed out; an earlier SceneNode normalization error remains unexplained. Passing generic checks does not repair the optional legacy ontology.

`module-consolidation.csv` provides an improvement recommendation for every registered module. This includes reusing existing elevator semantics and perception freshness policies, rather than rebuilding capabilities that are already present.

## Recommended next work, with acceptance criteria

| Priority | Work item | Acceptance criterion |
|---|---|---|
| 1 | Close concrete action-coverage gaps | Mixing refinement plus no-navigation pick/place and selected move-and-act/face/look recipes; valid/invalid/unknown cases; explicit existing-source field correspondence; generic imports stay world-independent. |
| 2 | Consolidate the current CQ registry | Each required question identifies namespace/version, record/profile, query or evaluator, expected evidence level and executable positive/negative fixture case. Keep scientific conceptual review distinct from tested answers. |
| 3 | Define evidence and answer contracts | Per-part Defined/Generated/Sourced/Derived/Measured grades, exact source claim/selector, context and derivation dependencies; no automatic promotion into accepted beliefs; immutable snapshot/storage policy explicit. |
| 4 | Populate a small sourced knowledge slice | One pouring material regime and one contrast action such as pickup/mixing; sources retained exactly, conflicting claims visible, unknown numerical limits left unspecified. |
| 5 | Add justified scientific aspect detail | Select a bounded subset of the remaining 65 CQs; define estimator/event-error/gaze semantics only as needed, with source-backed examples and counterexamples. |
| 6 | Isolate optional AICOR reasoning failures | Reproduce/minimize the problematic closure and determine normalization vs timeout vs actual consistency/class-satisfiability issues; preserve original source axioms. |
| Later | Working projections and runtime adapters | Versioned mapping to CoraPlex/SDT/EQL; compare task, code completion, physical outcome and evidence acceptance separately; validate suitable simulations/measurements before executable claims. |

Priorities 1–5 can proceed within the user's ontology-first plan. The review does not start the deferred working projections or add speculative executor implementations.

## Verification and reproduction

Fresh results are indexed in `reports/release-0.21.0.json`: **59 current-record checks, 93 historical competency assertions, 23 recipe-query checks, 62 import/catalog checks and 31 V8 pouring-contract checks all passed (268 total)**. Historical V7 competency assertions, current nested-query checks, existing recipe query checks offline import/catalog checks and V8 pouring-contract regressions were rerun. The 0.20.0 classification/consistency results are cited as baseline evidence, **not rerun in this review**, because no RDF/OWL input changed. This release makes no new physical, numerical-control or upstream runtime-validation claim.

From the extracted package root, with `requirements.txt` installed:

```bash
python tools/review_current_coverage.py
python tools/review_competency.py
python tools/validate_recipe_tool_queries.py
python tools/validate_imports.py
python tools/validate_pouring_contracts.py
```

To regenerate static class facts, obtain the pinned upstream source tree and run:

```bash
python tools/extract_coraplex_inventory.py /path/to/pinned/repository /tmp/coraplex-static-inventory.json
```

The source tree is not bundled. The included AST snapshot, source URLs/blob hashes and SHA-256 values preserve what was inspected. Coverage classifications are curated review decisions and do not automatically change when a new source inventory is generated.

The package remains a standalone artifact. No GitHub push, migration into `llm_reasoner`, accepted observation, simulated run or robot execution is established by this release.
