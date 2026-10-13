# `llm_reasoner` status

The single record of where the work stands: what is done, what is in progress, what was
decided, and what is proposed next. Read it with [`architecture.md`](architecture.md),
which describes how the package is built. Update both in the same change whenever the
work moves on.

Last updated: 2026-10-12; code at commit `6743d6116`.

## 1. State at a glance

| Component | State |
|---|---|
| World injector (`world/`) | Built and tested; eight tools; world-access contract 1.0 with declared answer shapes |
| Resolver (`resolver/`) | Built and tested; structured prompts; verified live on a PR2 kitchen (28 of 28 cases as expected) |
| Clients (`clients/`) | Anthropic tool loop with caching and usage; OpenAI plain completion; scripted client |
| Knowledge component | Built and tested (ontology, generation, admission, storage, views), reorganised into `knowledge/` with one public module; consulted by the resolver before every request to its model (R1–R7, verified live), but its hypotheses are **not yet read** by the resolver's request (knowledge tools are next) |
| Offline tests | `python -m pytest test/llm_reasoner_test`: 385 passed, 2 skipped |

## 2. Completed

### World injector
- Questions about the live world from krrood alone: kinds, individuals, descriptions,
  holders, field paths, predicates (`645765fd5`), as model tools (`ca31d29c1`), with
  listings of what the world contains and token usage per conversation (`347cef65b`).
- One search can name what a path reaches from each individual found (`c075ec000`).
- Search conditions checked before the search: a missing field or a comparison its
  values cannot take is an error naming the fields to continue to; a condition may
  compare with an individual by handle (`707ead8f0`).
- A guessed handle's error names the search that gives such handles (`eb07186f4`).
- Placeholders (`<Kind>#<n>`, `<field>.<field>`) instead of one world's names in what
  the model reads (`cd3f2e107`).
- The world-access contract: `world/access.py` (`WorldAccess`, version 1.0), the
  resolver importing nothing else of the world injector, declared answer shapes for
  every tool and one error shape, all checked by tests (`709d0ef8a`).

### Resolver
- Candidates described by walking their fields (`4e534d505`); tests, static situation
  provider removed (`5344ec818`); world tools and handles (`3c2bc8a49`); the text-answer
  rule and caching (`df54039fa`).
- Verdicts: no fitting candidate, equally fitting candidates (`68858bb84`); refusals
  kept to open slots (`a25df41c0`, `ea301abb9`, which reverts `ab148b25b`).
- The full source of the action and of its same-module parents (`5d1f32cd6`,
  `d5f427678`).
- Defaults and given values written as an answer writes them (`1b8d64f2c`,
  `346c62f5a`); `null` for fields defaulting to `None` (`8db3f5c36`).
- Answers breaking a where-condition repaired with the broken conditions named; the
  `unsatisfiable` verdict (`9776c26f6`).
- A one-sentence `why` in every answer (`abab4e8bb`); fewer candidates when fewer are
  good (`fda0221e0`).
- Structured prompts: tagged sections, JSON data, named rules (`6126cdd3a`), with each
  accepted answer's JSON type stated and verdicts kept to candidate slots (`64d294e37`).
- Choose a candidate only once known; never vary only to differ; ties between values
  chosen and explained (`6743d6116`).

### Clients
- The tool interface and Anthropic tool loop (`ca31d29c1`, `df54039fa`); tool
  descriptions as labelled parts and errors as JSON (`6126cdd3a`); a turn calling no
  tool ends the conversation instead of sending an empty message (`ea44fdbac`).

### Knowledge component
- The predictive-manipulation ontology (`core`, `object-dynamics`, `prehension`,
  `pouring`, `gaze`) linked to CRAM-AICOR and DUL, and the Python vocabulary generated
  from it with a drift test (`fb9ca0830`).
- The pipeline: generation as fragments, closed-world acceptance, open-world HermiT
  consistency with contradiction localisation, records and the store, views and the
  graph projection (`fb9ca0830`, `199245d89`).
- Acceptance moved from hand-written rules into the ontology's restrictions and SHACL
  shapes (`172ae748e`, `4419248ce`, `cf72116a6`, `d0a5c1bd3`, `243359ff0`); calibrated
  against a complete reference structure (`64795fc29`); the full pouring phase sequence
  required (`09af95ccc`); a guarded grasp required in every holding phase (`43f8db702`).
- Repairs by patch on the answer with the fewest problems (`a89518ca6`); structured
  completion with Sonnet 5.5 (`da4d20a33`); the type catalogue cached in the system
  prompt (`25b0b9269`); concept references checked against the linked ontologies
  (`aa3f6dc3a`).
- cram_viz: Hypothesis Graph and Ontology Graph tabs, the open-world check when the
  linked ontologies are configured (`5846a6fa5`, `7e84563bc`).

## 3. Live testing

Live runs use Claude Sonnet 5.5 on the PR2 in the IAI apartment (the repository's
`_apartment_world_setup`) with a bowl, a pouring cup, bread and a cutting knife added,
and optionally SDT's furniture annotations (109 annotations). Until 2026-10-12 the run
scripts were kept outside the repository; the cases now live in
`experiments/llm_reasoner_kitchen` and run as tests with `LLM_REASONER_LIVE=1`
(`architecture.md` §5), on the repository's own PR2 in the apartment rather than
cram_viz's PR2 scene.

**The suite** (C = `CuttingAction`, P = `PouringAction`, T = with a task):

| Level | Cases | What it checks |
|---|---|---|
| 1 Almost given | C1, P1 | Values only, one candidate each |
| 2 Small lists | C2, P2 | The right individual among 3 listed |
| 3 The whole world | C3, P3 | Finding individuals among 229 bodies with the tools |
| 4 Conditions on values | C4, C5, P4, P5 | `<`, `in_`, `not_`, `and_` range, `or_` |
| 5 Conditions on individuals | C6, C7, P6, P7 | `contains` on `name.name`, `exists` over a Food's root, linked slots |
| 6 Everything open, several answers | C8, P8, C9, P9 | All optional fields open; `an(...)` with limit 3 |
| 7 Conflicts | C10, P10, C11, C12 | A condition forcing an unsuitable tool; nothing to cut; contradictory bounds |
| 8 With a task | T1–T6 | Halving, thick slices under a limit, slowly, milk, sandwiches, the sink among furniture |

**Final run (2026-10-11, at `6743d6116`):** 28 of 28 as expected, every case in one
conversation (C12's repair is designed), $0.98 in total. The previous full run had 23
of 28, two repairs and $1.04; its findings led to `ea44fdbac` and `6743d6116`.

**Observed in the final run, none changing a result:**
- Unforced choices vary between runs (SAW or SLICE without a task; "slowly" read as
  tilts from 1.2 to 1.85; PouringAction has no speed field).
- The arm is an arbitrary choice in about 20 of 28 answers, as the model says, because
  the world injector cannot yet say where things are or which gripper holds what.
- One guessed handle in 28 runs (it was about 4 in 10 before the placeholders); one
  `name` search corrected to `name.name`; five runs listed many bodies before searching
  sensibly; eight opened with `list_kinds`.

**The knowledge request (R7, 2026-10-12, at `9a0b09687`):** one `PouringAction` from the
cup into the bowl, arm and tilt open, six cases; hypotheses in a scratch store, HermiT not
configured (closed-world acceptance only).

| Case | Task | Situation chosen | Hypothesis |
|---|---|---|---|
| K1 | none | both unknown | generated, 1 attempt |
| K2 | "Pour the milk from the cup into the bowl." | LIQUID | generated, 1 attempt |
| K3 | "Pour the cereal …" | GRANULAR | generated, 1 attempt |
| K4 | "… milk … until the bowl is half full." | LIQUID, TARGET_QUANTITY_ATTAINED | generated, 2 attempts |
| K5 | "Pour all the cereal …" | GRANULAR, SOURCE_EXHAUSTED | generated, 2 attempts |
| K6 | as K2 | LIQUID | served from the store (K2's record), 9 s, $0.04 |

All six as expected; $2.02 in total (a generation $0.30–0.41 and about 3 minutes, a
situation pass $0.005–0.016, the request $0.04–0.06); five stored hypotheses, one per
combination. The situation pass named its evidence and left the rest `null` ("nothing
… says what ends the transfer, so the termination reason stays null").

The first run ($1.93) found two problems, both fixed before the rerun: every hypothesis
failed to be built because krrood's symbol graph, created with the kitchen, did not know
the vocabulary (§7; `bf6c786d8` now refuses before asking the model), and the situation
pass filled features by convention ("I chose the usual pouring case") where nothing
established them (`9a0b09687`: open values as facts).

**Reading the knowledge (I5, 2026-10-12, at `c44d3dd7a`):** the same cases K1–K5, the
hypotheses served from R7's store, in a process holding the kitchen without rebuilding
krrood's symbol graph (knowledge no longer needs it). Every request read
`knowledge_overview` first; $0.29 in total ($0.04–0.07 a case, nothing generated).

| Case | Knowledge read | Result | Grounding stated |
|---|---|---|---|
| K1 | overview | RIGHT, 1.85 | "the knowledge's LiquidPouringStructure and its ph_tilt phase set no other angle" |
| K2 | overview | RIGHT, 1.85 | follows the tilt-to-flow phase (ph_tilt) |
| K3 | overview, `find_fragments` (condition requirements) | RIGHT, 1.85 | the granular structure gives no tilt value |
| K4 (half full) | overview and 7 descriptions (sg_tilt, ph_tilt, sp_tilt, th_flow, r_tilt, r_flow_th, role_eff) | RIGHT, **1.5** | inside sp_tilt's 30–120° setpoint range and above th_flow's 40–110° flow onset, "so the milk flows but gently, which suits stopping at half full" |
| K5 (all of it) | overview | RIGHT, 1.85 | GRANULAR, SOURCE_EXHAUSTED: tilt until the cereal has flowed out |

The knowledge changed a choice only where the task asked for nuance (K4), and every
answer cited the fragments it relied on. The overview names phases and subgoals but no
values, so numeric ranges are found only by describing fragments. The arm stays an
arbitrary choice: neither the world nor the knowledge says which gripper holds the cup.

**Grounds and values in the overview (2026-10-12, at `1d75290bd`):** K1–K5 again,
$0.32 in total. Every answer named its grounds, all checked: the bowl's and the cup's
handles and one or two knowledge fragments (the tilt phase, the flow-onset threshold).
With the ranges in the overview, no request described a fragment for them; each quoted
the range of the hypothesis it read ("about 106°, inside the 30–120° tilt range") and
kept the default tilt of 1.85 rad, K4 included, which in the previous run had described
seven fragments and chosen 1.5. One run each, so some of the difference is run-to-run
variation; the lasting limit is that the knowledge's tilt range is wide and holds the
default, while the parameters it states more narrowly (tilt rate 5–30°/s, transferred
quantity) have no field on coraplex's `PouringAction` to choose.

**Structured generation prompts (2026-10-12):** K8 ("…until I tell you to stop") chose
LIQUID and INSTRUCTED_STOP, a combination not yet stored, and the hypothesis generated
with the structured prompts was admitted on its second attempt, like K4 and K5 under
the prose prompts; the answer and patch were of the same size (about 30k characters,
then a patch of a few hundred). Generation cost $0.51 against $0.30–0.41 before: the
system prompt carries the types as indented JSON, 130k characters against 81k, on every
attempt. It is paid once per situation, since the hypothesis is then stored. K7 (honey)
was judged LIQUID and served the stored liquid hypothesis, so it did not exercise the
generation.

**The ported suite (2026-10-12):** `experiments/llm_reasoner_kitchen` live, all 34
tests passed in 10 minutes (26 resolutions, 2 refusals, 6 knowledge consultations), on
the repository's PR2 in the apartment and with R7's stored hypotheses; $1.81 for the
resolvers' conversations, pouring cases $0.05–0.09 with their consultation, cutting cases
$0.03–0.05. One hypothesis was generated, which the per-case costs leave out: T6 ("Empty
the cup into the sink") chose LIQUID and SOURCE_EXHAUSTED; emptying is stated, the liquid
is inferred from the sink rather than stated.

**cram_viz resolving its designators (2026-10-12):** in the viewer's simple_pouring
scene, "pour from the vessel into the bowl" resolved live in about a minute for $0.075:
source and target given, RIGHT and 1.85 rad chosen, resting on the bowl's and the cup's
handles and the knowledge's flow-regulation phase ("inside the 90–120° regulation range
of the pouring knowledge"); the knowledge was cram_viz's stored plain pouring hypothesis.
The Designator tab showed the resolved slots, the explanation and the grounds.

**cram_viz's designators as match queries (2026-10-12):** the chips show the krrood
match queries the resolver receives, each read from a file of its own, in place of
natural-language labels and tasks. Two remain: `objects_given` binds the pouring cup and
the bowl's body directly and leaves the arm and the tilt open; `objects_open` leaves every
slot open. A chip is offered when the scene binds every name its query reads and has a
candidate for every slot it leaves to choose. Live, `objects_given` resolved in
simple_pouring for $0.068 (RIGHT, 1.85 rad, citing the tilt phase and its set-point);
the situation pass read the bound `Static_MilkPitcher` as liquid, so a liquid pouring
hypothesis was generated once, at a cost not recorded.

**Situation from the world, runs kept apart (2026-10-12):** the situation pass saw only
the values a query binds, so `objects_open` (the cup only a candidate) was served the
plain pouring knowledge and `objects_given` the liquid one in the same scene. Its
`<action>` section now also lists the action's open slots with their candidates, and the
facts rule has the model read the world about the individuals the action involves. In
cram_viz every resolution is a run of its own (`cram_viz.resolution.runs`): the scene's
world built anew, the knowledge consulted over a copy of the viewer's store in
`~/.cram_viz/runs/<run>/hypotheses`, and the resolution kept beside it as
`resolution.json`; no run reads what another generated. Live, each from the same nine
stored hypotheses: both designators now read the pitcher in the world as liquid; each run
generated its own liquid pouring hypothesis (about 3.5 minutes, cost not recorded), and
the two differ (a tilt range of 0–120° against 40–120°, 176 against 172 typed nodes),
while both resolved to the cup, the bowl, RIGHT and 1.85 rad; the resolvers cost $0.103
(`objects_given`) and $0.078 (`objects_open`).

**cram_viz: Generate and Execute (2026-10-13):** a chip copies its match query into the
search bar and Generate resolves whatever query the search bar holds. Execute performs
the resolved action in coraplex's simulation, in a process under the live bridge, from
the run's environment: the scene's world (its robot now on a drive over the surroundings
at the recording's first base pose, annotated as the robot class kept for its name),
park arms, torso LOW, navigate to face the source at the recorded 0.63 m, pick up with
the resolved arm, pour. With the torso HIGH or MID the PR2 cannot reach the pitcher on
the counter, and coraplex's `giskard_reachability_location` found no standing pose in
the scene, so the standing pose is computed directly. The fixed demo stays behind a Demo
button, but it no longer imports with the current semantic_digital_twin (`BoundingBox`).
Live in the viewer: the `objects_given` chip copied into the search bar, Generate
resolved it ($0.101, LIQUID read from the world, this run's own hypothesis generated, the
objects given, RIGHT and 1.85 rad chosen), and Execute performed the run under the live
bridge without a failed step, the pitcher held and tilted 106°. Measured as the
recording demo measures it, though, the pitcher's mouth ends 0.188 m from the bowl's rim
centre (rim radius 0.070 m, mouth 0.087 m above the rim): with its default pour
geometry (`pour_side_offset`, `pour_approach_offset`, `pour_height`), coraplex's
`PouringAction` parks the pitcher beside the bowl rather than over it. A third
predefined designator, `pour_geometry_open`, gives the cup and the bowl and leaves the
arm, the tilt and the pour geometry (`pour_side`, `pour_side_offset`,
`pour_approach_offset`, `pour_height`) to the resolver, to test whether the resolver's
choice puts the mouth over the bowl. Live, it kept every default (offsets 0, height
0.13 m, the side of the arm, "nothing in the request calls for changing them"), so the
executed pour ended exactly as before, 0.188 m off: nothing the resolver reads says
where the pitcher's mouth ends for a given geometry ($0.100, LIQUID, its own hypothesis
generated).

**Scene geometry in the prompt (2026-10-13):** every resolution prompt now states, as a
fact of the situation, each annotated object's origin, world-frame bounding box and top
centre and where each robot base stands and faces (`cram_viz.resolution.scene_geometry`,
through llm_reasoner's context-provider seam). Live, `pour_geometry_open` then reasoned
from the positions (the pitcher on the robot's right, so the right arm and side; 0.13 m
clears the rim) but, reading `PouringAction`'s source, trusted its built-in
`sin(tilt) × mouth height` side offset and kept the extra offsets at 0, so the pour
again ended 0.188 m off ($0.091 and $0.087 in two runs). Probed without the model, a
`pour_side_offset` of −0.188 m puts the mouth 0.005 m from the bowl's centre, 0.089 m
above its rim: coraplex's offset moves the tool sideways while the tilt does not bring
the mouth back. Choosing the cancelling offset needs either that fixed in coraplex or
feedback to the resolver from a simulated pour.

**Simulated pours fed back to the resolver (2026-10-13):** the resolver may now be given
an `OutcomeCheck` (`resolver/outcomes.py`) trying each answer before it is accepted; a
refusal (`OutcomeFellShort`) goes back through the repair round as a structured
rejection holding the outcome. cram_viz's `SimulatedPouring` performs every pouring
answer from the scene's initial world in a process of its own and measures where the
pitcher's mouth ends against the bowl's opening, ahead and right as seen from the robot,
up to three trials. Live, `pour_geometry_open`: the first answer kept the defaults and
missed (0.188 m, 0.184 m right of the opening's centre); reading that, the resolver set
`pour_side_offset` to −0.188 m ("a pour_side_offset of -0.188 cancels it"), and the
second pour landed 0.005 m from the centre. $0.139 for the resolver, about four minutes
with the run's knowledge generation and both trials. Switching scenes now removes the previous scene's world
from krrood's symbol graph (collected, then `remove_dead_instances`), so an open
`variable(Tool)` never offers another scene's objects.

**Seen but not yet acted on (R7):** the request resolved every case alike (RIGHT, the default
tilt of 1.85 rad), whatever the task, since it does not read the hypothesis yet; and
the world does not say what the cup holds, so the material came from the task alone.

## 4. Decisions

| Date | Decision |
|---|---|
| 2026-10-09 | Three components: world injector, resolver, knowledge component; the resolver takes krrood queries, not instructions |
| 2026-10-09 | `llm_reasoner` stays independent of `semantic_digital_twin`; the world is read through krrood only |
| 2026-10-11 | The resolver returns a resolved designator; executability and preconditions are not its job |
| 2026-10-11 | Option A: a where-condition leaving one candidate decides the slot, even an unsuitable one |
| 2026-10-11 | Option i: the query's slot type decides what kind of thing fills a slot; the task only chooses among candidates |
| 2026-10-11 | Prompts are structured (tagged sections, JSON data, named rules), with no prose outside content |
| 2026-10-11 | Accuracy and structure over token cost; compact JSON in prompts was tried and rejected |
| 2026-10-11 | The resolver reaches the world through `WorldAccess`; an MCP server waits for a second consumer |
| 2026-10-12 | The knowledge component provides stereotypical action knowledge from several perspectives (FrameNet, Minsky frames, image schemas, Flanagan), giving the resolver an understanding of the action and explicit grounds for its choices, not slot values directly |
| 2026-10-12 | The resolver reaches knowledge as tools it pulls from, not as a prompt section, because hypotheses will grow |
| 2026-10-12 | Knowledge requirements: generated by a language-model agent, structured and ontology-backed, consistent under HermiT, and queryable with krrood's EQL as first-class krrood objects |
| 2026-10-12 | The ontology is the single source of truth for knowledge; krrood objects are its working form (EQL, closed world) and RDF its reasoning form (HermiT, open world); both are generated or derived from the ontology, never hand-written |
| 2026-10-12 | The knowledge component is reorganised and redesigned first, with the Flanagan representation only; cram_viz is updated along with it |
| 2026-10-12 | Naming: a *hypothesis* is one generated, admitted structure; *knowledge* is what the component provides and its contract |
| 2026-10-12 | Option A for K4: the knowledge component declares the situation features it accepts in ontology terms, and the resolver derives them from the world |
| 2026-10-12 | The knowledge request (`architecture.md` §3.6): D1 a krrood query on the knowledge vocabulary with the action concept and situation features fixed, no instruction; D2 designator → knowledge by a mapping kept as data; D3 situation features derived from the ontology's defined subclasses; D4 the resolver fills them in a first resolution with its world tools and the task; D5 the store keyed by root class, fixed values and ontology versions, generating once per feature combination on a miss |
| 2026-10-12 | The resolver consults the knowledge component before every request to its model (served from the store, else generated); `Reasoner` assembles the system with it always on, while the bare backend keeps it optional |
| 2026-10-12 | A situation feature is a fact: the first pass answers only what the action, the given values, the task, the situation or the world establishes, and `null` otherwise (`OpenValueMeaning.FACT`) |
| 2026-10-12 | Live checks with Sonnet 5.5 may run when needed, without asking per run; their outcome and cost are recorded in §3 |
| 2026-10-12 | Two graphs: krrood's symbol graph is the universal one (the world, created once); the knowledge component keeps its own local graph, independent of krrood's; the resolver answers a query from both, the world through `WorldAccess` and the knowledge through `KnowledgeAccess`. Knowledge objects stop being krrood `Symbol`s, and the two closed-world inferences krrood made for them (a sub-property's values in its super-property, symmetric properties) move into the knowledge component's own property descriptors. K2 reworded accordingly |
| 2026-10-12 | Reorganisation first, without new functionality: one `knowledge/` package (`ontologies`, `ontology_translation`, `acceptance`, `storage`, `admission`, `generation`, `hypotheses`) with one public module, moved lowest layer first; stored hypotheses (the cram_viz sample and `~/.cram_viz/hypotheses`, backed up) are migrated with each move that changes a stored module path |

## 5. In progress

- The knowledge request, steps R1–R7: all done; R7's results are in §3.
- Knowledge in its own graph, steps I1–I5: done, except I4's domains for slots of a
  knowledge class (waiting for a use and a decision on whose fragments form the domain).
- Reorganising the knowledge component (decision of 2026-10-12). Done: the pure moves
  into `knowledge/`, lowest layer first, each committed with both test suites passing
  (`ebc93d9fa` ontologies, `274542e2c` ontology translation, `5149984fe` acceptance,
  `b22e1f7e9` storage split from admission, `52c0273f3` generation, `9dca1ed56`
  hypotheses, `5da11b5d5` the public `access.py`), and the cycle between packages broken
  with a new `representation/` layer and a test of the layers (`bf7858d06`). Stored hypotheses were migrated: the
  cram_viz sample and the 4 current records in `~/.cram_viz/hypotheses` (backup in
  `~/.cram_viz/hypotheses.backup-2026-10-12`; 5 older records there use paths of code
  removed on 2026-10-07 and were left as they were). Then `HypothesisBackend` split by
  responsibility, its interface unchanged: `HypothesisRequest` (`c041e3c5b`),
  `HypothesisAssessor` (`b9cd77467`), `GenerationDialogue` (`3fee61653`) and
  `HypothesisRecorder` with the backend left to orchestrate (`c4bb49ec9`). The
  reorganisation adds no functionality; next is designing the request from the resolver
  (K3) with situation features (option A), the knowledge tools and structured
  generation prompts.
- `doc/drafts/object_qualities/` is an uncommitted draft on object qualities in the
  ontology (affordances, dispositions after DUL). It waits for decisions on where the
  module lives and on changes to `core.ttl`.

### Planned: knowledge independent of krrood's symbol graph (two graphs), lowest layer first

Each step test-first and committed with both test suites passing.

| Step | What | Where | Changes for others |
|---|---|---|---|
| I1 | **Done** — The knowledge component's own property descriptors, applying when a field is set what krrood's did (a sub-property's values are also its super-property's; a symmetric property holds both ways), generated by the ontology translator in place of krrood's; `DescriptionEntity` no longer a krrood `Symbol`; the vocabulary regenerated. The symbol-graph guard and `Reasoner`'s exclusion of the knowledge package become unnecessary and go | knowledge, `ontologies/`, `ontology_translation/`; `reasoner.py` | none expected: cram_viz already queries knowledge over explicit domains |
| I2 | **Done** (`da2a3051b`: `KnowledgeGraph` in `hypotheses/knowledge_graph.py`, exported by `access.py`; `KnowledgeService` reads hypotheses through it, so each is read once and its fragments keep their identity) — The knowledge component's local graph: the admitted hypotheses' fragments, loaded from the store, giving EQL its domains (every fragment of a class) and the hypothesis each belongs to | knowledge | none |
| I3 | **Done** (option B, knowledge-shaped tools, chosen 2026-10-12; `architecture.md` §3.7) — `KnowledgeAccess`, the contract the resolver reads the local graph through: domains for knowledge classes and knowledge tools for the model, declared like the world tools | knowledge, `access.py` | none |
| I4 | **Done in part** — the request's model gets the knowledge tools over the consulted hypothesis beside the world tools, with the `knowledge` rule, when the backend asks with tools (`LanguageModelBackend.knowledge_tools`, `request_completion`); the requester depends on `KnowledgeAccess`. **Not built:** slots of a knowledge class taking their domain from the knowledge graph, which no action designator has today and which needs a decision on whose fragments form the domain (every stored hypothesis, or the consulted one). Planned: The resolver answers a query from both graphs: a slot or variable of a knowledge class takes its domain from `KnowledgeAccess`, one of a world class from `WorldAccess`; the model gets the knowledge tools beside the world tools, so the request reads the hypothesis the consultation obtained | resolver, `reasoner.py` | none |
| I5 | Live check: the R7 cases, the request now reading the knowledge — **done** (§3): every request read the knowledge, K4 chose its tilt from it | — | — |

### Done: the knowledge request (D1–D5), lowest layer first

Each step test-first and committed with both test suites passing; live runs only with
approval.

| Step | What | Where | Changes for others |
|---|---|---|---|
| R1 | Declare the situation features of a root class, derived from the ontology's defined subclasses (property, allowed values, meaning) — **done** (`7d74e9f99`, `ontologies/situation.py`; pouring has two: material regime, termination reason) | knowledge, `ontologies/` | none |
| R2 | Verify how a fixed feature's variant is enforced — **done**: closed-world acceptance already classifies a structure by its feature values into every variant they define and applies its restrictions, HermiT adds the open-world check when configured, and the generation prompt states each variant's definition and requirements; acceptance needs no change, two tests pin the behaviour | knowledge, `acceptance/` | none |
| R3 | Key the store by root class, fixed values and ontology versions; the instruction becomes optional provenance (`KnowledgeService.view_for` takes the query first) — **done** | knowledge, `storage/`, `hypotheses/` | cram_viz: one hypothesis per concept and features, no longer per instruction; the 4 records in `~/.cram_viz/hypotheses` all share the plain pouring key, so the newest is served for every pouring instruction |
| R4 | The designator → knowledge mapping as data, and the lookup of the knowledge describing a designator class — **done** (`ontologies/described_actions.json` and `.py`; one entry: coraplex's `PouringAction` → `aicor:Pouring`, `PouringPhaseStructure`, matched by identifier and structure IRI) | knowledge | none |
| R5 | The knowledge component's entry for a request: serve the stored hypothesis or generate it, in its public module — **done**: `KnowledgeService.serve(query, instruction="")` returns the record identifier (`view_for` is `view(serve(...))`), and `access.py` exports `DescribedActions` and `SituationFeatures` for the resolver | knowledge, `access.py` | none |
| R6 | The resolver's first pass: map the designator, build the feature query, resolve it with world tools and the task, and send the request through the public module only | resolver | the resolver depends on the knowledge component's public module |
| R6a | The world's symbol-graph source excludes packages, so the knowledge component's symbols are not shown as the world — **done** (`d54c36db7`, `SymbolGraphSource.excluded_packages`) | world | none |
| R6b | `KnowledgeRequester` and `KnowledgeConsultation` (`resolver/knowledge.py`), the `<action>` section (`ActionContextProvider`), and the knowledge side's `DescribedAction.query` and `SituationFeature.value_in` (`f83690eb9`) — **done**; a test checks the resolver imports only `knowledge.access` | resolver, knowledge | none |
| R6c | Always on, as the user asked on 2026-10-12: the resolver consults the knowledge component before every request to its model, served from the store when the situation was met before, generated otherwise — **done**: `LanguageModelBackend.knowledge`, `consultation`, `situation_backend` and `situation_backend_for`; `reasoner.py`'s `Reasoner` assembles the system, its world leaving out the knowledge package, and always gives the backend a knowledge requester. The bare backend keeps the requester optional, so it stays usable on its own; the main request does not read the hypothesis until the knowledge tools exist | resolver, `reasoner.py` | none: the requester is a new optional setting |
| R7 | Live check (with approval): pouring from the cup of milk (Liquid) and of cereal (Granular), with and without a stopping task; the features chosen and their reasons, and one generation per combination — **done** (§3): six of six as expected after two fixes | — | — |

## 6. Proposed and deferred

| Item | Why | State |
|---|---|---|
| Computed world values: poses, reachability, contact, which gripper holds what | Grounds the arm and geometry choices the model now calls arbitrary | Deferred; options: (a) a registry in krrood's `@symbolic_function`, (b) SDT moves the functions to class form, (c) SDT marks its properties with krrood field metadata, (d) `llm_reasoner` reads zero-argument properties generically. No choice yet |
| Linking the knowledge component to the resolver and the world injector | Lets the resolver use action knowledge | The request is built (R1–R6, `architecture.md` §3.6): the resolver obtains the hypothesis before every request to its model. Reading it, as knowledge tools (`architecture.md` §2.3), is next after R7 |
| Representing knowledge for tools | Several perspectives, organised by action concept, addressable fragments, objectives and consequences first-class | Requirements K1–K4 recorded below; design next |
| Structured prompts for the knowledge component | Its system, initial and repair prompts were prose built in Python, unlike the resolver's tagged sections | **Done** 2026-10-12 (`architecture.md` §3.2): tagged sections, JSON data and named rules; the system prompt grew from about 80k to 130k characters (the types as indented JSON) |
| Designator slots supplied from the SDT world (containers, arms) | Replaces hand-written bindings | Deferred |
| Choosing smaller pouring structures from the world state | The full 12-phase pouring structure is always required | Deferred |
| An MCP server over the world-access contract | For consumers outside the resolver's process | When a second consumer needs it |
| Live tests in the repository | The live suite lived in scripts outside the repository | **Done** 2026-10-12 (`architecture.md` §5): `experiments/llm_reasoner_kitchen` with its tests; the live tests run only when opted in, five cases have scripted twins run with the experiments suite. `llm_reasoner`'s own tests are not yet in CI (the user's decision of 2026-10-12) |
| Fewer wasted calls (opening `list_kinds`, listing many bodies, `name` before `name.name`) | Cost only; results are correct | Open |
| An explicit "either arm" answer | Arbitrary arm choices are stated only in `why` | Open; would change the action interface |
| Research direction: language-model commonsense checked against the ontology for designator resolution | Long-term | Open |

### Knowledge component requirements (2026-10-12)

| # | Requirement | Against today |
|---|---|---|
| K1 | Generation backed by an ontology linked to CRAM-AICOR, so that language-model knowledge stays consistent, structured and reasonable | Met: the Flanagan ontology links CRAM-AICOR and DUL; acceptance and HermiT enforce it |
| K2 | Knowledge as objects generated from the ontology, queryable with EQL over the knowledge component's own graph, independent of krrood's symbol graph (reworded 2026-10-12; first stated as "first-class krrood symbols") | Met: independent of krrood's symbol graph since I1; `KnowledgeGraph.instances_of` gives EQL its domains since I2 |
| K3 | Requests come from the resolver, not as a natural-language instruction; knowledge is generated for that input | Not met: `KnowledgeService.view_for(instruction, query)` takes an instruction, which also keys the store |
| K4 | The knowledge component is independent of the world; situational facts it needs (such as a material's flow regime, whether a target is open, sizes, or poses as values) reach it only through the resolver, in the ontology's terms | Met for independence (it imports no world); not met for receiving situational facts |

K3 supersedes the earlier proposal that the resolver only reads stored knowledge: the
resolver's request now triggers generation. K4 was first stated as the knowledge
component reading the world itself, and corrected the same day: the resolver is the
only component that reads the world.

### Open questions for linking the knowledge component

Answered 2026-10-12: *how the resolver consumes knowledge* (as tools) and *the scope
beyond the Flanagan microworld* (several perspectives); see §4.

1. **The knowledge representation.** How knowledge is organised for tools: by action
   concept and perspective rather than by instruction, perspectives as linked ontology
   modules, objectives and consequences first-class. The user's requirements are to come.
2. **The `KnowledgeAccess` contract.** Its tools, identifiers and answer shapes, and how
   much of the world injector's generic machinery it reuses over a knowledge source.
   *Answered 2026-10-12 (I3):* three knowledge-shaped tools over the consulted
   hypothesis (`knowledge_overview`, `find_fragments`, `describe_fragment`), fragments
   named by their citations, following the world tools' conventions but not reusing
   their machinery, whose words (individuals, holders, predicates) are the world's.
3. **Separate sources.** Keeping knowledge fragments, which are krrood symbols too, out
   of what the world injector shows as the world. *Answered by R6a and R6c:* the
   symbol-graph source leaves out excluded packages, and `Reasoner` excludes the
   knowledge package.
4. **Which knowledge answers which query.** How a resolver query on an action designator
   (a coraplex `PouringAction`) reaches the knowledge about its action concept
   (`aicor:Pouring`): a mapping kept as data, or the action declaring its concept; and a
   store lookup by concept rather than by instruction. *Answered by R3 and R4:* a mapping
   kept as data, and the store keyed by root class, fixed values and ontology versions.
5. **Roles and slots.** Whether ontology roles (source, target, the pouring lip) map onto
   designator slots, as the cram_viz Designator tab sketches.
6. **Situation features (from K4).** Proposed: the knowledge component declares the
   situation features it accepts, in ontology terms (a material's flow regime, an open
   or narrow target, a fill level, sizes or poses as values), and the resolver derives
   them from the world; knowledge is then cached by action concept and features. The
   alternatives were an abstract situation port implemented by the resolver, or passing
   the world tools (rejected: world handles would leak into the knowledge). The
   resolver, not the knowledge component, grounds roles in world individuals.
7. **The request from the resolver (from K3).** What the resolver sends: the action
   designator's class, its given and open slots, the query's conditions, the optional
   task; and what of it is generated per request versus reused.
8. **Reasons as data.** Whether the resolver's `why` becomes reasons citing knowledge
   identifiers and world handles. *Answered 2026-10-12:* each answer element carries
   `grounds`, checked against the world and the knowledge, kept as
   `LanguageModelBackend.justifications`.
9. **Inferred knowledge.** HermiT checks consistency only; whether facts it infers (such
   as a fragment's inferred types) are written back so EQL queries see them.

## 7. Known issues

**In `llm_reasoner`:**
- `clients/anthropic_client.py` reads `stop_reason` with `getattr` in one place, which
  the code rules forbid.

- cram_viz's `kb.build_symbol_graph` still rebuilds krrood's symbol graph and gives
  the knowledge vocabulary as its reason, which no longer holds since I1; whether
  cram_viz's own symbols still need the rebuild is unchecked.

**Outside `llm_reasoner`** (to report to their owners, not to change here):
- krrood's verbalizer writes contradictory bounds as "between 0.01 and 0.005" and enum
  lists as `[<CuttingTechnique.SLICE: 1>, …]`.
- krrood does not enforce an `exists` condition over a variable declared outside the
  match; conditions must use `query.variable.…`.
- coraplex: `ToolMotionAction.arm` and `.tool` have no docstrings; `PouringAction` has
  no speed or duration field; `composite/searching.py` fails to import
  (`PerceptionObjectNotFound` is missing).
- The test kitchen gives the bread's body two `Connection6DoF` connections as child.

## 8. Continuing the work

1. Read `architecture.md` for how things fit, then this file for where they stand.
2. Run the offline tests: `python -m pytest test/llm_reasoner_test`, and
   `scripts/format_docstrings.py` on changed files.
3. Live checks with Sonnet 5.5 may run when needed (decision of 2026-10-12); record
   each one's outcome and cost in §3.
4. Record decisions in §4 and move items between §5 and §6 as they change.
