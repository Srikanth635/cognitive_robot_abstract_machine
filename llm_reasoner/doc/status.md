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
| Knowledge component | Built and tested on its own (ontology, generation, admission, storage, views), reorganised into `knowledge/` with one public module; **not yet linked** to the resolver |
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
and optionally SDT's furniture annotations (109 annotations). The run scripts are kept
outside the repository; every run needs the user's approval.

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
| 2026-10-12 | Reorganisation first, without new functionality: one `knowledge/` package (`ontologies`, `ontology_translation`, `acceptance`, `storage`, `admission`, `generation`, `hypotheses`) with one public module, moved lowest layer first; stored hypotheses (the cram_viz sample and `~/.cram_viz/hypotheses`, backed up) are migrated with each move that changes a stored module path |

## 5. In progress

- The knowledge request, steps R1–R7 (planned below): R1–R6 done; R7, the live check, waits for approval.
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

### Planned: the knowledge request (D1–D5), lowest layer first

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
| R7 | Live check (with approval): pouring from the cup of milk (Liquid) and of cereal (Granular), with and without a stopping task; the features chosen and their reasons, and one generation per combination | — | — |

## 6. Proposed and deferred

| Item | Why | State |
|---|---|---|
| Computed world values: poses, reachability, contact, which gripper holds what | Grounds the arm and geometry choices the model now calls arbitrary | Deferred; options: (a) a registry in krrood's `@symbolic_function`, (b) SDT moves the functions to class form, (c) SDT marks its properties with krrood field metadata, (d) `llm_reasoner` reads zero-argument properties generically. No choice yet |
| Linking the knowledge component to the resolver and the world injector | Lets the resolver use action knowledge | The request is built (R1–R6, `architecture.md` §3.6): the resolver obtains the hypothesis before every request to its model. Reading it, as knowledge tools (`architecture.md` §2.3), is next after R7 |
| Representing knowledge for tools | Several perspectives, organised by action concept, addressable fragments, objectives and consequences first-class | Requirements K1–K4 recorded below; design next |
| Structured prompts for the knowledge component | Its system, initial and repair prompts are prose built in Python, unlike the resolver's tagged sections | Proposed |
| Designator slots supplied from the SDT world (containers, arms) | Replaces hand-written bindings | Deferred |
| Choosing smaller pouring structures from the world state | The full 12-phase pouring structure is always required | Deferred |
| An MCP server over the world-access contract | For consumers outside the resolver's process | When a second consumer needs it |
| Live tests in the repository | The live suite now lives in scripts outside the repository | Proposed: skipped in CI, each paired with a scripted test |
| Fewer wasted calls (opening `list_kinds`, listing many bodies, `name` before `name.name`) | Cost only; results are correct | Open |
| An explicit "either arm" answer | Arbitrary arm choices are stated only in `why` | Open; would change the action interface |
| Research direction: language-model commonsense checked against the ontology for designator resolution | Long-term | Open |

### Knowledge component requirements (2026-10-12)

| # | Requirement | Against today |
|---|---|---|
| K1 | Generation backed by an ontology linked to CRAM-AICOR, so that language-model knowledge stays consistent, structured and reasonable | Met: the Flanagan ontology links CRAM-AICOR and DUL; acceptance and HermiT enforce it |
| K2 | Knowledge as first-class krrood symbols, queryable with EQL | Met: the vocabulary classes are krrood `Symbol` dataclasses |
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
   identifiers and world handles.
9. **Inferred knowledge.** HermiT checks consistency only; whether facts it infers (such
   as a fragment's inferred types) are written back so EQL queries see them.

## 7. Known issues

**In `llm_reasoner`:**
- `clients/anthropic_client.py` reads `stop_reason` with `getattr` in one place, which
  the code rules forbid.

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
3. Ask before any live run; record its outcome and cost in §3.
4. Record decisions in §4 and move items between §5 and §6 as they change.
