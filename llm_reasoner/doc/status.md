# `llm_reasoner` status

The single record of where the work stands: what is done, what is in progress, what was
decided, and what is proposed next. Read it with [`architecture.md`](architecture.md),
which describes how the package is built. Update both in the same change whenever the
work moves on.

Last updated: 2026-10-21; code on the branch `feature/llm_reasoner`, which carries `origin/main` of that day.

## 1. State at a glance

| Component | State |
|---|---|
| World injector (`world/`) | Built and tested; eight tools; world-access contract 1.0 with declared answer shapes |
| Resolver (`resolver/`) | Built and tested; structured prompts; verified live on a PR2 kitchen (28 of 28 cases as expected) |
| Clients (`clients/`) | Anthropic tool loop with caching and usage; OpenAI plain completion; scripted client |
| Observation (`observation/`) | Events, sinks, wrappers for the client, tools, world and knowledge component, and hooks in the resolver and the knowledge generation built and tested; cram_viz logs every run; the viewer (`llm_reasoner_viz`) serves the logs and shows them as a map, a timeline, an inspector, an attempts strip with a plan of where the pours ended, filters and a replay |
| Knowledge component | Built and tested (ontology, generation, admission, storage, views), reorganised into `knowledge/` with one public module; consulted by the resolver before every request to its model, which reads the hypothesis through three knowledge tools and cites its fragments as grounds (R1–R7 and I1–I5, verified live). Two vocabulary families: the Flanagan predictive-control modules and the action perspectives (action core and action frame, phases 0–4 of `knowledge_perspectives.md`, offered opt-in). A pour's predicted end and the flow-onset tilt are checked on the resolver side (`pour_prediction.md`) |
| Offline tests | `python -m pytest test/llm_reasoner_test`: 834 passed, 6 skipped, 29 deselected as slow (2026-10-19) |

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
- Functions declared with `symbolic_function` are predicates of the world: found in the
  loaded modules by krrood's wrapper, offered when everything they require is a plain
  value or an individual the world holds, evaluated through `evaluate_predicate`, and a
  failing one answered as `PredicateFailed`. In the pouring scene the resolver's world
  offers `contact`, `is_supported_by`, `is_supporting` and `stable` (which fails, saying
  it needs a simulation). The robot and camera functions appear once the scene's robot
  is annotated and their module, `semantic_digital_twin.reasoning.robot_predicates`, is
  loaded.
- The modules of a world model are named in `SymbolGraphSource(world_model_modules=...)`
  and imported when the source is made. With the robot predicates' module named, the
  simple_pouring world with its robot annotated offers 14 predicates instead of 8, the
  six robot ones answering through the tools.
- The ontology of a code base (2026-10-15): `world/class_ontology/` writes the classes a
  selection file names as OWL, read from the live classes as the world injector reads them
  (`DocumentedClass`, which now also gives whole class and field docstrings, not only their
  first lines): each class with its documentation and the described classes it specialises
  (mixins left out, their fields kept), each field as a property scoped to the class that
  adds it, with its documentation, type (a plain value, an individual of a class, a member of
  an enumeration, optional, several), default and the type as written, and each enumeration
  with its members. A type variable reads as what the class's generic bases fix it to, else
  its bound. It names no package: `knowledge/ontologies/cram_classes/cram-classes.json`
  selects CRAM's action and motion designators, semantic annotations and robots, with
  coraplex's data structures in full and `Designator`'s bookkeeping left out, and
  `cram-classes.ttl` is the result (392 classes, 13 enumerations, 300 KB; the hand-made
  `cram_tbox.ttl` was 9.8 MB). It replaces `cram_tbox.ttl` as the designator side of the
  knowledge perspectives' roles-to-slots mapping (`knowledge_perspectives.md` §4.1). 29
  fields keep only their written type: unions of several classes, a type variable nothing
  fixes, `Type[...]`. One coraplex module cannot be imported (`composite/searching.py`) and
  is named in the file. A test in `test/experiments_test/llm_reasoner_cram_classes_test`
  fails when the committed file no longer matches the code; regenerate with
  `python -m llm_reasoner.world.class_ontology` and the two paths. Writing it twice gives
  the same text, and it is the same whatever else the process imported (the test found
  that krrood's resolved hints differ by process, so the generator works type variables
  out itself and records the type as written in the source).

### Resolver
- Whole field docstrings (2026-10-15): a slot's meaning in `<slots>`, and a field's meaning in
  the world tools, are now the whole docstring below the field, not its first line, which cut
  meanings mid-sentence (`position_threshold` read "… ``None`` falls back to"). One reader,
  `llm_reasoner/attribute_docstrings.py`, serves the resolver, the world injector and the
  knowledge component (whose vocabulary docstrings are single lines, so its prompts do not
  change). A slot also finds the docstring of a field a mixin declares, which it lacked before
  (`PouringAction.position_threshold` and `orientation_threshold` had no meaning at all).
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

### Observation
- Events about the system's calls (`ModelAsked`, `ModelAnswered`, `ToolCalled`, …) as plain
  JSON, sinks (none, memory, an append-only JSON-lines log shared by several processes), and
  wrappers reporting a client's completions and a model's tool calls from outside
  (`ObservedClient`, `ObservedToolClient`, `ObservedTool`); `AnthropicClient` reports the
  tokens of its plain and structured completions per thread (`UsageReporting`), so their
  cost can be told. Step V1 of the viewer (`c7fa16fa8`).
- Wrappers reporting the calls between components (`ObservedWorld`, `ObservedKnowledge`),
  the tool calls the tool box refuses, the resolver's start, consultation of the
  knowledge, attempts, rejections and end (`LanguageModelBackend.observer`), the knowledge
  generation's request, attempts, failure and hypothesis served
  (`HypothesisBackend.observer`), and `Reasoner(sink=...)` assembling them. Step V2
  (`41d06eef4`, `45555d376`, `f297aee73`).
- Every cram_viz run logs its work to `events.jsonl` in its directory: the run starting
  and ending or failing, the resolver's and the knowledge component's questions to the
  model, the world and knowledge calls, the attempts, the hypothesis generated, and each
  pour tried in the simulation (`SimulatedPouring.observer`). A replayed resolution writes
  about 36 events, 210 KB, the prompts being most of it. Step V3, the log (first part).
- Execute logs too: that the run started to be performed and how it ended, and each step
  of the plan when its status changes (`StepChanged`, from the plan snapshots the bridge
  takes, `PlanStepReporter`). Checked on a real Execute of a stored run (12 s): park arms
  +2.0 s, torso +4.3 s, navigate +4.3 s, pick up +4.6 s, pour +7.8 s, ended +12.3 s. It
  needed two corrections to the bridge, which also give the Plan tab the plan of an
  executed run: a plan performed through its root node, as `PouringExecution` does, never
  reached the bridge (`install_root_node_hook`), and coraplex's `NOT_STARTED` and `PAUSED`
  were not the `CREATED` and `PAUSE` the bridge derives progress from (`STATUS_ALIASES`),
  so no node ever showed progress. An action still reads `RUNNING` until the plan ends,
  its conditions not being evaluated in these runs. Step V3, the log (second part).
- The answer panel links to the viewer of the run (`paths.viewer_url`, `http://localhost:8721`
  or `LLM_REASONER_VIZ_URL`); the server returns the address with the resolution. Step V3
  complete.
- The viewer, step V4: the package `llm_reasoner_viz` with its server on port 8721 (this
  machine only), listing the runs that have a log, newest first, each summed up (query,
  scene, how resolving and performing went, events, model calls, tokens, cost), the events
  of a run as a batch, and a stream following a log as it grows, resumable by offset; a first
  page showing each event as a row, coloured by the component it belongs to. Checked in a
  browser on the replayed resolution, on a real Execute's log, and on a log written slowly
  while the page followed it. Its tests are not in CI, like those of `llm_reasoner` and
  `cram_viz`.
- The viewer, step V5: the screen. The page is ES modules (`web/*.mjs`, no build step); the
  logic lives in modules with node tests (`prompts` parses a prompt into its tagged
  sections, `model` builds spans from events one at a time, `layout` packs spans into rows,
  `inspector` makes the view of a span, `map` counts calls per connection), and the
  elements are drawn by `*_view` modules checked in a browser. The screen has a top bar (run
  picker, status, totals, connection), the map of the components on the left (which
  component is at work and what it does, which connection was just used), the timeline in
  the middle (a row per component, bars that grow with the run, zoom, fit, follow), and the
  inspector on the right (prompts as sections, a trail up through what a span happened
  during, links down to what happened during it); a drawer lists the raw events. A span has
  an address (`?run=…&span=…`), so a link returns to it. Checked in a browser on the replayed
  resolution (every kind of span opened), a real Execute's log, a log written in real time
  while the page followed it (bars grow, the run settles to the same spans as loading the
  finished log), light and dark themes and a narrow window.
- The viewer, step V6a: replay. `replay.mjs` (node tests) says what a run looked like at any
  moment (the events written by then, rebuilt into spans, so what was going on is open) and
  how a replay moves (play, pause, step to the next or previous event, speeds 1/4/16/64,
  clamped to the run, stopping at the end, starting over from the end); `replay_view.mjs`
  draws the controls. While replaying, the three views show the run as of the replay's moment
  and the timeline's axis keeps the length of the whole run. Not available while a run is going
  on. Checked in a browser on the 13-second stretched replay: scrubbing, stepping, speeds,
  the end, leaving the replay.
- The viewer, step V6b: filters. `filters.mjs` (node tests) picks the spans that say some
  words (name and everything their events hold, case-insensitive) and those that were
  rejected or failed, alone or together, and finds the next match in time, going round;
  `filter_view.mjs` draws a search box, *problems only*, the count and *next*. The timeline
  fades the bars that do not match. Checked in a browser on the replayed resolution.
- The viewer, step V6c: the attempts strip and the pour-trial view. `attempts.mjs` (node
  tests) makes a card per attempt of a resolution (what it chose, from the answer it was
  rejected with or the last the model gave; why; what it was rejected for; the pour tried)
  and says which resolution to show for the chosen span; `pour.mjs` makes a plan of where
  pours ended from the outcomes (the opening as a circle of its radius, ahead up, right
  right, only pours that were measured, scaled to hold the furthest) and lists the pours tried
  during a resolution. The strip sits at the bottom with the plan beside the cards; the
  inspector shows the plan for a pour (that one marked) and for a resolution. A card, a point
  and a bar open one another. Checked in a browser on a replayed resolution whose two trials
  carry measured positions (the scratch copy of a real log; a real one comes with the live
  check). The outcome keys are in one place, `OUTCOME` in `model.mjs`.
- The viewer, step V6d, cram-viz's half: a run kept earlier can be reopened. The resolver's
  `reopen(name)` returns the run's resolution with the knowledge component over the run's own
  store; `GET /api/designator/run?name=` answers as a resolution just made does (resolution,
  consulted hypothesis, link to the viewer); the EQL panel, opened with `?run=<run>`, shows it
  after booting (query in the search bar, hypothesis on the bus, Execute for that run) without
  resolving. Opening needs the same credentials as resolving, the resolver being built the
  same way. Checked on the real run of 13 October (which predates logs) against a sandbox
  server.
- The viewer, step V6d, the viewer's half: `GET /api/config` says where cram-viz is
  (`CRAM_VIZ_URL`, default `http://localhost:8711`), and the top bar's *open in cram-viz* link
  (`links.mjs`, node tests) opens cram-viz at the run's scene and run. Checked in a browser with
  both servers: the link, then the reopened run in cram-viz with its link back. The sandbox
  machine has no `pouring` scene bundle, so cram-viz's designator menu and 3D scene showed their
  errors there, which reopening does not depend on.
- NEEMs of the simulations (14 Oct). The plan every simulation performs is kept as a NEEM, saved
  through coraplex's ORM into one shared database (`~/.cram_viz/neems.sqlite`,
  `CRAM_VIZ_NEEM_DATABASE`): each pour tried while a resolution is made (the trial process is
  told the database, the run's log and the trial through options, and keeps the plan whether or
  not the motion finished) and the run Execute performs. `cram_viz/neem.py`: a store (SQLite,
  tables made once under a lock), recorders that report `NeemRecorded` or `NeemNotRecorded` to the
  run's log without hiding how the performance went, and `NeemRecording` which tells another
  process where to record. Measured on a real pour: the first NEEM into a new database takes about
  40 seconds (the 1,724 tables), each later one 5 to 9 seconds and about 0.8 MB, so a resolution
  of three pours takes around twenty seconds longer. The viewer marks bars that have a NEEM, shows
  the plan id, database and a `sqlite3` command in the inspector, and counts them in the top bar.
  Checked with the real trial subprocess (a normal pour and one whose motion fails, both kept),
  the real Execute on a scratch copy of a run, and the viewer in a browser. One test against the
  real store is marked `slow`. Not yet: a switch to turn the recording off, and recording from
  `cram-viz-live` demo scripts.
- Every node of the plan Execute performs is logged (15 Oct): `PlanNodeChanged` in the run's
  `events.jsonl`, from the same bridge snapshots as `StepChanged` (`PlanNodeReporter`): when a
  node first appears (the tree, which grows as an action is expanded) and whenever its status
  changes, with its parent, kind, label and whether the bridge derived the status. Done so the
  viewer can show the per-step status and timing that a saved NEEM lacks (it keeps a real status
  for the root only). Checked on a real Execute of a copy of a stored run (bridge on its own
  port, no NEEM kept): 65 events, 15 KB; each motion's start and end show, e.g. the tilting
  motion of the pour from +8.6 s to +12.3 s. Known from the bridge, not changed: an action
  never reaches `SUCCEEDED` (its conditions are not evaluated, so it reads `RUNNING` until the
  plan ends) and the conditions of most steps stay `CREATED`; the motions and the root do end.
  The viewer does not show these events yet (it ignores events it does not know).
- The NEEM browser of the viewer, steps 1 and 2 (15 Oct; the screen follows). The viewer opens the
  shared NEEM database read-only (never changes it; reported as busy while a simulation writes;
  refused with what it lacks when it is no NEEM database). `GET /api/neems` lists the plans, each
  joined to the run whose log says it was kept and to how that trial or execution went (`?run=`
  keeps one run's); `GET /api/neems/<id>` gives one plan as a tree of steps, each with what it was
  asked to do (`PouringAction`, ...), its values, the things it refers to by the names the world
  gives them (`bowl.stl`), poses with their frame, and the rows it is kept in. The reader follows
  the ORM's one-table-per-level inheritance through the foreign keys, so it needs no list of
  action classes. Facts found on the real database: only the root of a plan has a real status and
  times (every step below reads `NOT_STARTED`, so the screen must not show a status per step);
  bytes columns (world states) are left out; a step's references to other steps and to what its
  execution recorded are left out as bookkeeping. Read a real 33-step plan in 0.01 s (38 KB).
  Tests use SQL fixtures built from the real tables (`test/llm_reasoner_viz_test/neems`).
  Step 3, the *neems* button (`neems.mjs` with node tests, `neems_view.mjs`): there only when
  the open run saved NEEMs; its list starts with that run's NEEMs and widens to all of them,
  filtered to trials or executions, each row saying how it ended and going to the bar of its
  trial or execution in the run that saved it. Checked in a browser against a scratch database
  (a trial and an execution, in two runs) and the real one: found and fixed a menu that closed
  when a button in it was clicked (the list is redrawn, so the clicked element had left it
  before the outside-click check ran), and the address now keeps the span chosen by a row.
  Step 4, one NEEM (`neem.mjs` with node tests on fixtures that the Python tests keep equal to
  what the server sends, `neem_view.mjs`): a row of the list opens `?neem=<id>` in place of the
  run's panels, with the tree of the steps, what the chosen step was asked to do (reusing the
  inspector's blocks), a link to the bar of its trial or execution in the run that saved it, and
  a link back from the inspector of that bar. Checked in a browser on the real database: a
  33-step plan with a pick-up (its grasp description), a re-attach (the pitcher moved to the
  gripper) and poses in the frame of the pitcher.
  Step 5, comparing (`neem_compare.mjs`, node tests): *compare with…* in the header sets the NEEM
  against another one. Steps are matched by their names from the root down (the first of two
  same-named siblings with the first), steps whose values or references differ are marked, and
  the chosen step lists each differing value as `here → there`; steps only one plan has are
  marked or listed. The other NEEM is in the address (`compare=`), so a comparison can be
  reloaded. Checked in a browser: a pour against a navigate-and-pour plan, reload, clearing.
- The viewer, step V6e: a live check. One real resolution with Sonnet 5.5 (the designator
  giving the objects, in `simple_pouring`) through a cram-viz of its own (its own data directory,
  its store a copy of an earlier run's, so no knowledge was generated) and a viewer of its own,
  watched as it ran: bars growing, tokens and cost climbing, the simulation's pour. It failed
  after 40 s, which the viewer showed from the model call up to the run: attempt 1's pour missed
  (0.188 m from an opening of radius 0.069 m) and attempt 2's answer was cut off at cram_viz's
  `RESOLVER_MAX_TOKENS` (2048 output tokens). Cost reported $0.0946 for the two answered calls;
  the cut-off call carries no usage (only answers do), so its cost, roughly 3–8 cents, is in no
  event. Found and fixed by it: a prompt that comes from and goes to the cache reports those
  tokens apart from `input_tokens`, which the viewer had summed alone (6 shown for 21,323 read);
  all three are now input. The run, shortened, is the fixture `failed_run.json`; its tests cover
  the failure chain, the cards of the attempts, the plan of a real pour and the tokens.
  `RESOLVER_MAX_TOKENS` was then raised from 2048 to 8192 (a per-turn ceiling, paid for only as
  far as the model writes; the SDK accepts it without streaming). The cut-off is intermittent, so
  this is not proven until a few live runs pass. A cut-off answer is a `ResponseTruncated`
  naming the limit and holding what the model had written (2026-10-20), and the resolver now
  sends it back for repair as an `AnswerTruncated` rejection (2026-10-21, after a run in cram_viz
  failed on one); still open: a failed call carries no usage or cost, and what the model writes
  in a turn that runs to 8192 tokens has not been looked at.

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
- The action core (2026-10-15, phase 0 of `knowledge_perspectives.md`): a second family of
  ontology modules, the action perspectives (`ontologies/action_perspectives/`, namespace
  `…/llm_reasoner/action-perspectives/`), whose first module, `action-core.ttl`, is the hub
  every perspective attaches to. It states an action's frame (`ActionFrame`, a DUL
  description; `ManipulationActionFrame`, also CRAM-AICOR's `ManipulationFrame`, so a frame
  of navigating need not be one), the slots its participants fill (`AgentSlot`,
  `SourceSlot`, `SubstanceSlot`, `GoalSlot`, `InstrumentSlot`, `PatientSlot`, each a
  CRAM-AICOR `RoleSlot` specialising the matching CRAM-AICOR participant role, and
  `PartSlot` for a part of a participant, such as a lip or an opening), what it is for
  (`Objective`, a CRAM-AICOR `Telos`), what it brings about (`Consequence`, a CRAM-AICOR
  `Effect`) and the choices it leaves open (`ParameterDimension`, a DUL parameter in
  CRAM-AICOR's `ActionParameterSpace`, with a `QuantityKind`). `PouringFrame` frames
  `aicor:Pouring`: an agent, a source, a substance and a goal, a part of the source and of
  the goal, an objective aiming at `aicor:AccumulatedState` (as `aicor:PouringTelos` does),
  a consequence and a parameter dimension. Written by hand, not proposed by the model. The
  vocabulary generator now reads both families in dependency order (the action core first)
  and writes each module into its own package; the Flanagan modules regenerate
  byte-identically. HermiT finds the core consistent with CRAM-AICOR and DUL and every class
  of it satisfiable (22 s). Nothing uses it yet: hypotheses, the store key, admission and the
  knowledge tools still read only the predictive manipulation modules (phase 2).
- The vocabulary proposal (2026-10-15, phase 0 of `knowledge_perspectives.md`):
  `knowledge/vocabulary_proposal/` asks a model for the vocabulary of one perspective
  (`python -m llm_reasoner.knowledge.vocabulary_proposal action-frame --action aicor:Pouring
  --frame core:PouringFrame --output <directory>`). The request shows the perspective's
  description, the core's terms and the terms of CRAM-AICOR and DUL; an answer is checked
  for names, definitions, terms that exist and are of the right kind, classes grounded in
  the core, CRAM-AICOR or DUL, attachment to the core, the Python generator and, with
  HermiT, consistency with the core, CRAM-AICOR and DUL, and is returned for repair up to
  twice. The first live run failed because HermiT tried to fetch the proposal's imports;
  the checker now drops them, since HermiT is given the files. The second run (Sonnet 5.5,
  96k input and 12k output tokens, $0.31) passed on the first answer: 9 classes
  (`SlotExpectation`, `SlotCondition` with feature, affordance, state and posture
  conditions, `ActionDefault`, `AdditionalConsequence`, `DimensionRationale`), 23
  properties and 2 value partitions (`ConditionPolarity`, `ChangeDirection`), in
  `ontologies/action_perspectives/proposals/action-frame.ttl`. In review two gaps were
  fixed by hand on the answer, which then passed the rules and HermiT again: a posture
  condition can now name the posture (`Posture`: `Upright`, `Tilted`, `Inverted`, a DUL
  region of the core's `Orientation`), and an affordance condition names the CRAM-AICOR
  action the filler affords (`conditionAffordsAction`, under `aicor:affordsAction`) or
  the state it can be brought into (`conditionAffordsState`, such as
  `aicor:ContainedState`), since CRAM-AICOR's `Affordance` has no kinds to name. Now 25
  properties and 3 value partitions. A property whose values are CRAM-AICOR concepts now
  names them by IRI with no range, as the core's `framesAction` and `aimsAtState` do: a
  proposal may leave a property's range out, a rule tells the model so, and the frame's
  `conditionHasState`, `conditionAffordsState`, `conditionAffordsAction` and
  `normallyFilledBy` follow it. The core's `hasQuantityKind` is no longer reserved for
  parameter dimensions, so the frame reuses it for a posture condition and an additional
  consequence instead of declaring two copies of it; the core's Python is unchanged but
  for the field's docstring, and HermiT still finds the core consistent. Now 23
  properties.
- The action frame registered (2026-10-15, phase 1 of `knowledge_perspectives.md`): the
  proposal writer writes one block per term in the proposal's order, so an edit to one
  term changes only its lines; `action-frame.ttl` is the second action perspective module
  (`ActionPerspectiveFile.ACTION_FRAME`), with its generated `action_frame.py` and tests
  like the core's, and its proposal record stays under `proposals/`. On registering, its
  statements about a frame moved onto a new `FrameStereotype` (`stereotypeOf` a core
  frame), since with the core's `ActionFrame` as their domain they became fields of the
  core's Python class; and its `ConditionPolarity` became `SlotConditionPolarity`, the
  Flanagan core having a class of that name. HermiT finds the frame consistent with the
  core, CRAM-AICOR and DUL and every class satisfiable. Nothing uses it yet (phase 2).
- Knowledge in the action perspective vocabularies, step 1 (2026-10-15, phase 2 of
  `knowledge_perspectives.md`, §11): the pipeline generates, checks, stores and reads a
  core hypothesis, `the(PouringFrame)(frames_action=aicor:Pouring)`, as it does a
  Flanagan one. `Vocabulary` and the RDF reader cover both families; each family states
  the modules a hypothesis rooted in it may use, and the schema keeps to them, both for
  the types offered and for the types a field accepts; a hypothesis's open-world check
  covers the modules it uses, and its record keeps only their versions, so stored
  Flanagan records stay valid; the prompt's role and content rules come from the root's
  module (`generation/briefs.py`), an action perspective's from its description beside the
  modules. The Flanagan prompts are byte-identical. Tested with a scripted answer, also
  admitted by HermiT against the real CRAM-AICOR and DUL.
- Step 2 (2026-10-15): a perspective hypothesis on top of a stored core, such as
  `the(FrameStereotype)(stereotype_of=<citation of the core frame>)`. A fixed value may cite
  a stored fragment, keyed by its IRI and told to the model by its name; the cited
  records' fragments are shown in a `<given>` section and referred to by name, and an
  answer restating one or naming a fragment like one is returned for repair; the
  open-world check sees them with the new fragments; the record lists what it
  `builds_on`, and its Turtle refers to their IRIs without stating them, which the store
  and the knowledge graph read back as the same fragments. A perspective's brief adds the
  rule about the given core. Tested with scripted core and frame answers, both also
  admitted by HermiT.
- Step 3 (2026-10-15): the knowledge tools read several records as one body of
  knowledge, the records they build on included (knowledge-access contract 1.2:
  `tools(*record_ids)`); an action perspective hypothesis's fragments are named after its
  module, such as `action-core/source`. `knowledge_overview` answers as before for a phase
  structure and adds, for an action core, its frame (slots, objectives, consequences,
  parameter dimensions) and the perspectives read with it; describing a slot lists the
  expectations linking to it. A call for one record is logged as before, several as
  `records`.
- Step 4 (2026-10-15): the resolver consults the action core and the perspectives its
  requester names (`KnowledgeRequester.perspectives`, empty by default, so runs and
  cram_viz are unchanged) beside the Flanagan hypothesis: the core is served for the
  described action's frame (`described_actions.json` gains `"frame"`), each perspective
  on the stored core, named by its record (`StoredRoot`) and rooted where
  `PERSPECTIVE_ROOTS` says. The knowledge tools and the grounds read all of the
  consultation's records; the `frame_knowledge` rule is stated when perspectives are
  offered; `KnowledgeConsulted` reports their records. Tested with scripted answers.
- Step 5, live (2026-10-15, Sonnet 5.5, HermiT against CRAM-AICOR and DUL, linked terms
  checked): the core of pouring (13 fragments: six slots with the lip and the opening as
  parts, the objective aiming at `AccumulatedState`, two consequences, three parameter
  dimensions: amount, the source's orientation, the lip's position relative to the goal)
  and the action frame on it (19 fragments: what the source, goal and substance are
  normally filled by and must have or afford, such as a lip, `aicor:Tilting` and
  `aicor:Grasping`, and `ContainedState` with the substance; the source grasped and
  upright before pouring; pouring until the goal holds the wanted amount rather than
  until the source is empty; why each dimension matters; the source becoming lighter).
  Each admitted on its first answer, no warnings, $0.065 together (14k tokens written to
  the cache, 2.7k output). Two weak spots: most conditions leave their polarity out,
  which the vocabulary allows, and "the source is empty" was stated without the state it
  denies.
- The two weak spots fixed (2026-10-16, action frame 0.2.0): every slot condition must state
  its polarity, and a state condition its state (`conditionHasState some owl:Thing`, as
  the Flanagan core says "has some value"); the state condition's definition now gives
  the negated example, the source empty as the substance not contained in it. An answer
  missing either is returned for repair.
  Live again on the stored core ($0.044, first answer, consistent): every condition now
  states its polarity, and the source being empty is `ContainedState` not holding of the
  source with the substance.
- Roles to slots (2026-10-16, phase 3 of `knowledge_perspectives.md`, §12): a designator's
  fields mapped to the parts of the action core's frame in the core's vocabulary
  (`described_actions.json` `"frame_fields"`; `ontologies/frame_fields.py`): a slot by its
  class, a choice by its quantity kind, the slot it is about and the slot it is measured
  from, a part of a slot counting as the slot. The mapping is checked against the code
  ontology (`cram-classes.ttl`: an object field fills a slot, a number or enumeration a
  choice, every field exists) in a test, and matched to the consulted core by the
  knowledge component (`frame_parts_filled`, knowledge-access contract 1.3). When
  perspectives are offered, each open slot's entry in `<slots>` names the frame parts it
  fills (`"frame"`), and the frame rule says to read those first. The `PouringAction`
  mapping is written by hand for review; against the live core, five of its six fields
  find their part, `pour_height` none. Reviewed: kept; the core's description now names
  pouring's usual choices, height above the goal among them, and a core generated with it
  fills all six fields. A description is not part of a module's version, so stored cores
  are still served.
- Measuring the action frame, the simulated pour (2026-10-16, phase 4 of
  `knowledge_perspectives.md`, §13; `experiments.frame_comparison`, Sonnet 5.5): cram_viz's
  `pour_geometry_open` in `simple_pouring`, three runs per arm, both arms reading the same
  stored knowledge (the liquid pouring hypothesis, and the core and frame with the height
  choice), nothing generated, $1.03 in total.

  | | Without the frame | With the frame |
  |---|---|---|
  | First pour lands | 0 of 3 (0.188 m off each) | 0 of 3 (0.188 m off each) |
  | Pours tried | 2, 2, 2 | 2, 2, 2 |
  | Final distance from the opening's centre | 0.018, 0.005, 0.014 m | 0.019, 0.005, 0.019 m |
  | Knowledge the answer rests on | the phase structure, in one run | the core's lip position, tilt and height, in every run |
  | Fragments described | none | 1, 9 and none |
  | Resolver cost per run | $0.083–0.133 | $0.193–0.291 |

  The frame changed what the answers rest on and what was read, not what was chosen: every
  first answer kept the default geometry, the frame arm's with the reason "the code
  already shifts the pour point sideways by sin(tilt) times the mouth height, so both extra
  offsets stay at 0", after reading the frame's rationale that the lip's position decides
  whether the substance lands in the goal. Knowing *what* matters did not say *where* the
  lip ends for given values; the simulated pour's feedback corrected it in both arms. The
  comparison showed cram_viz naming a knowledge ground by its fragment name alone, so its
  records counted no ground on the frame; a shown ground now keeps its citation
  (`39b555192`), and this run's grounds were classified afterwards against each run's
  stored records.
- Measuring the action frame, the kitchen (2026-10-16): K1–K5 once per arm, both reading
  the same store, into which the first arm generated the three situations not yet stored
  (granular; liquid until the target quantity; granular until the source is exhausted),
  their cost not recorded by the kitchen's live harness (about $0.3–0.5 each by earlier
  runs). Every case chose the same in both arms: the right arm, the default tilt of
  1.85 rad and the default pour geometry. With the frame, four of five answers rested on
  the core's tilt choice (K3 on the phase structure only), each read three to six
  knowledge tools against one or two, and the resolver cost $0.098 a case against $0.075
  ($0.49 against $0.37 for the five). K4's "until the bowl is half full" meets the frame's
  default of pouring until the goal holds the wanted amount, but `PouringAction` has no
  amount to choose.

  Phase 4's verdict, both measurements: the action frame changes what an answer rests on
  and what the model reads, not what it chooses, at 30 % (kitchen) to 100 % (simulated
  pour) more resolver cost. Where the pouring cases fail, they fail on where the lip
  ends for given values, which the simulated pour's feedback corrects in both arms.
- Measuring the action frame where the containers are open (2026-10-16, twice per arm,
  $2.14): the kitchen's P2 (the bowl among the bowl, the bread and a spoon), P3 (every
  body a candidate) and P8 (every optional field open), and cram_viz's `objects_open` (the
  pitcher and the bowl beside a milk carton and a cereal box). Both arms chose the cup or
  the pitcher and the bowl in every run; the kitchen's expectations were met in all 12.
  With the frame, every kitchen answer rested on the frame's expectations of the goal and
  the source (`exp_goal`, `exp_source`; once also the goal's opening and the source being
  held), where without it the answers rested on the tilt phase only; the model read five
  to eight knowledge tools against one or two, for about 20 % more ($0.100 a case
  against $0.080). `objects_open` missed every pour in both arms, its pour geometry not
  open to correct the 0.188 m miss, and its later tilts failed the motion ($0.27 against
  $0.26 a run).

  Phase 4, all three measurements: the frame does not change a choice on the pouring
  cases measured, the baseline already choosing well where a choice was open; it makes
  the answers rest on explicit expectations of the participants, which is the grounding
  the knowledge component is for, at 20–100 % more resolver cost. Off by default it costs
  nothing.
- Predicting where a pour ends (2026-10-16, `pour_prediction.md`): coraplex computes the
  pour's tool poses when its plan is built, before the pitcher is picked up, so its
  sideways compensation misses by sin(tilt) × the mouth's height; those poses, computed
  in the scene's world before anything is performed and carried with how the gripper
  holds the pitcher after a simulated pick-up, put the mouth within 5 mm of every
  recorded simulated pour. llm_reasoner gains `OutcomePrediction` (tools predicting an
  answer's outcome, offered beside the world tools with their rule); cram_viz's
  `PourPrediction` offers `predict_pour_end`, the hold measured once per pitcher and arm
  in a process of its own (`hold_measurement`), a pick-up the arm cannot make predicted as
  a failed motion; `DesignatorResolver.predicts_pours`, off by default. Live on
  `pour_geometry_open`, three runs: the first pour landed in every run (none did without
  the prediction), one pour each, the final prediction within 5 mm of the simulated pour,
  for $0.062 a run against $0.114. coraplex is unchanged. Kept off by default
  (decided 2026-10-18) until the open-container cases (P2, P3, P8) and the trap cases
  are measured with it. On `objects_open` (2026-10-18, `pour_prediction.md` §5), where
  only the tilt moves the mouth, it did not help: no first pour landed in either
  arm; the tilts it predicted to land (2.8–3.07 rad) failed as motions, since the
  pour's own motion is not predicted, and the one landing pour, at 0.2 rad, would
  not flow. The kitchen's P2, P3 and P8 try no pour, so the prediction does not
  apply to them.
  The simulated check now refuses a landing pour tilted less than the flow onset the
  stored knowledge states (2026-10-19, `pour_prediction.md` §6); tests pass. Live on
  `objects_open` no pour landed in four runs ($1.04), so it had nothing to refuse.
  The prediction now also says when the arm cannot move to the pour pose (inverse
  kinematics from the pick-up's end, `pour_prediction.md` §7): refused at 2.8 and 3.0 rad,
  accepted where the simulation finishes, not yet measured live.

- The image schemas and the failures (2026-10-19, phases 5 and 6 of `knowledge_perspectives.md`,
  `4`+`5` live proposals $0.53 and $0.50): two more action perspective modules, `image-schemas.ttl`
  and `failures.ttl`, with generated Python, scripted-knowledge tests and HermiT checks, offered
  opt-in through `KnowledgeRequester.perspectives` and as comparison arms. Measured live
  ($3.34, `knowledge_perspectives.md` §13.6): they change what the answers rest on, not what
  they choose, for 20–70 % more resolver cost; the failures arm lost 2 of 5 runs to the
  resolver's final turn being cut off at cram_viz's 8192 output tokens (a cut-off answer is now
  sent back for repair instead of failing the run).

- The first trap case (2026-10-20, `knowledge_perspectives.md` §13.7): the scene's only bowl
  upside down, its up axis now stated in the scene geometry. All four arms poured into it
  (7 of 7 runs that finished, $2.24; one run cut off at 8192 output tokens): the model does not
  read an up direction as an opening facing down, with or without the perspectives. With the
  opening direction stated (`opening_faces`, §13.7) 5 of 6 perspective runs left the bowl and
  the default arm poured into it twice ($2.28), but a repetition with the final harness
  (ontology-based opening, $1.77) had 7 of 8 runs refuse, the default arm included: two runs per
  arm do not show the perspectives changing the choice.

- The frame-knowledge rule is no longer hand-written (2026-10-20): `KnowledgeRequester.frame_rule`
  builds it from the first paragraph (`ActionPerspectiveModule.summary`) of the description of
  the core and of each perspective offered, so it states only what is offered; it used to name
  the action frame's expectations and the image schemas whichever were offered. Earlier
  comparison runs saw the old wording.

- The viewer shows all the knowledge a resolution consulted (2026-10-20): when perspectives are
  offered, `activate_resolution` draws the action core's frame and each perspective on it beside the
  Flanagan phase structure in the Hypothesis Graph tab, each root titled by its module (Action Core,
  Action Frame, Image Schemas, Failures) and hanging off the instruction, a perspective's fragments
  linking the core's own slot and dimension nodes. `HypothesisGraphProjector.project_together` draws
  several hypotheses in one graph (`project` is the one-hypothesis case); cram_viz's
  `ConsultedHypotheses` attaches them after the phase overlay, which is untouched, so the `pouring_*`
  EQL variables, highlighting and the Designator tab behave as before. The Ontology Graph tab draws
  the classes of all of them (`OntologyClassGraph.of(*views)`), reading descriptions from the action
  perspective modules and `object-kinds.ttl`, attributing terms to `TermSource.ACTION_PERSPECTIVE` and
  `OBJECT_KINDS`; CRAM-AICOR's container carries the requirement "has opening: true". A ground citing
  a core or perspective fragment is named by its fragment and is a node of the Hypothesis Graph.
  Checked in the browser on the recorded scripted core and perspectives of pouring (the machine has
  no stored run with perspectives); the EQL variables still range over the phase structure only.

- cram_viz's Generate can consult the action core and chosen perspectives (2026-10-21): below
  Generate, one box per perspective the knowledge component registers (`PERSPECTIVE_ROOTS`:
  action frame, image schemas, failures), all ticked by default (the user's choice after a
  run showed only the phase structure with them off); a ticked one is sent with the
  resolve request and consulted, with the core, beside the phase structure, so the Hypothesis and
  Ontology tabs then show it. A request naming an unregistered perspective is refused
  (`UnknownPerspective`). Before this the viewer never consulted them, so the tabs showed only
  the phase structure in normal use. Checked in the browser that the boxes appear; not run live.

- No more accepting a pour the simulation showed failing (2026-10-21): a user's run resolved
  to the left arm, whose pick-up had failed in both of its trials, because `SimulatedPouring`
  accepted its last trial whatever it brought about; Execute then failed at the pick-up. Now
  every pour that does not land is refused, and an `OutcomeFellShort` carries a
  `fallback_rank` (how far the mouth ended from the opening, for a pour that was performed and
  flows; none for a failed motion or one tilted too little to flow). When the repair budget
  runs out (`POUR_REPAIRS`, 3) the resolver keeps the best-ranked answer, with its own reason,
  and fails only when none was acceptable. In that run it would have kept the right-arm pour
  that missed by 0.188 m. Tests only, not run live.

- The phase structure is a choice too (2026-10-21): `KnowledgeRequester.consults_phase_structure`
  (and `Reasoner`'s, `DesignatorResolver`'s), on by default; off, no situation is chosen and only
  the action core and the perspectives are served. cram_viz offers it as a box beside the
  perspectives ("action phase structure", ticked), sent as `phase_structure: false` when
  unticked. Without it the tabs show the core first, the phase-hierarchy overlay and the
  phase-based EQL variables and Designator roles left empty, and the pour's flow-onset check
  has nothing to read. With every box unticked nothing is consulted and the tabs keep showing
  the previous hypothesis. Tests and a browser check; not run live.

- Knowledge is reused across cram_viz runs (2026-10-21): every run consults a copy of the
  baseline store as before, and when it ends, however its resolution ended, the hypotheses it
  admitted are added to the baseline (`HypothesisStore.adopt_new_records`, each record copied
  under a hidden name and renamed when complete), so the next run is served them instead of
  generating again. Found in an audit: the last six runs had all generated their knowledge
  afresh (`from_store: false`) and the baseline had not been written since 12 October, so every
  Generate paid for a phase-structure generation (about 3 minutes, $0.3–0.5). A run's own copy is
  unchanged, so reopening and Execute use what it consulted; records under older module versions
  are still never served. Tests only, not run live.

- The two families are linked at the level of classes (2026-10-20): `ontologies/slot-roles.ttl`
  puts each Flanagan role under the action core's slot whose CRAM-AICOR role it already
  specialises (`Actor` ⊑ `AgentSlot`, `Manipulandum` ⊑ `PatientSlot`, `GraspingEffector` ⊑
  `InstrumentSlot`, `SourceContainer` ⊑ `SourceSlot`, `TargetReceptacle` ⊑ `GoalSlot`,
  `PouredSubstance` ⊑ `SubstanceSlot`, `PouringLip` ⊑ `PartSlot` of the source). Neither family
  is edited, the file declares no term, and HermiT finds both families, the file, CRAM-AICOR and
  DUL consistent with every class satisfiable. `SlotRoles` reads it and
  `DescribedAction.field_roles` gives the role each designator field mapped to a slot fills, so
  cram_viz's Designator tab derives its roles instead of keeping a hand-written table.
  Generation and prompts are unchanged; instances in the two families' hypotheses are not yet
  linked to each other.

- One condition polarity for both families (2026-10-20): the action core's `ConditionPolarity`
  (`Holds`, `DoesNotHold`) replaced the Flanagan core's `ConditionPolarity` (`MustHold`,
  `MustNotHold`), the action frame's `SlotConditionPolarity` and the failures' `StatePresence`
  (`Holds`, `Missing`); whether the value is demanded or observed is said by each property. The
  Flanagan core imports the action core, the only import between the families. Versions moved
  (action core 0.2.0, action frame 0.3.0, failures 0.2.0, Flanagan core 0.5.0), so every stored
  hypothesis using them is generated again on its next use (about $0.3–0.5 each); cram_viz's
  hand-authored sample was rewritten to the new terms. Answers name the members `HOLDS` and
  `DOES_NOT_HOLD`. Tests and HermiT only, not run live.

- One direction of change (2026-10-20): the action core's `ChangeDirection` (`Increases`,
  `Decreases`) replaced the frame's, and the failures' `ParameterChangeDirection` and the
  Flanagan core's `CorrectionDirection` and `EstimateUpdateDirection` list its two members in
  place of their own up and down (`Increase`/`Decrease`, `RaiseSetpoint`/`LowerSetpoint`,
  `RaiseEstimate`/`LowerEstimate`), keeping their other values. The Flanagan two became DUL
  concepts, as a shared member cannot be both a region and a concept. Since several
  enumerations now list one individual, reading a hypothesis and filling entailed values give
  the member of the field's own enumeration (`Vocabulary.held_by_field`). Answers name the
  shared members `INCREASES` and `DECREASES`. Versions moved (action core 0.3.0, action frame
  0.4.0, failures 0.3.0, Flanagan core 0.6.0, prehension 0.5.0); cram_viz's sample was
  rewritten. Tests and HermiT only, not run live.

- One condition on a slot (2026-10-20): the action core's `SlotCondition` (`conditionOnSlot`,
  `conditionInvolvesSlot`, `hasPolarity`) and `StateCondition` (`conditionHasState`) replaced
  the frame's two classes and the failures' `ObservableCondition` with its three properties.
  The frame keeps its feature, affordance and posture conditions under the core's class. A
  perspective may state these core classes, so the decoder refuses another module's type only
  when a given hypothesis states it (`FragmentDecoder.given_types`); restating a core slot is
  still refused. Versions moved (action core 0.4.0, action frame 0.5.0, failures 0.4.0), so
  stored frame and failures knowledge is generated again. Tests and HermiT only, not run live.

- One quantity about a slot (2026-10-21): the action core's `SlotQuantity` (`hasQuantityKind`,
  `aboutSlot`, `relativeToSlot`, a DUL concept) is the class above `ParameterDimension`, the
  image schemas' `DecidingQuantity` (adding its aspect) and the failures' `Indicator` (adding
  its form and noticer); `quantityOfSlot` and `measuredOnSlot` are gone. Versions moved (action
  core 0.5.0, image schemas 0.2.0, failures 0.5.0), so every stored action-perspective
  hypothesis is generated again. With this, the four duplicated concepts the audit named are
  merged. Tests and HermiT only, not run live.

- Shapes for the perspective answers (2026-10-21): `StructureShapes` loads the SHACL shapes of
  every module of either family, and `action_perspectives/shapes/` holds 12 for the core, the
  frame, the image schemas and the failures (9 errors, 3 warnings; listed in
  `knowledge_perspectives.md` §9 phase 7), so perspective answers are checked across
  fragments and told the rules as the Flanagan ones are. The scripted failures answer broke
  one (a change towards a target naming none) and was corrected. No vocabulary changed. Tests
  only, not run live.

- Perspectives keyed by situation (2026-10-21): the material regime and the termination
  reason moved from the Flanagan pouring module into the action core, `PouringFrame` takes
  both with a defined variant per value, and the situation chosen once keys both the phase
  structure and the core's frame (on the frame alone when the phase structure is left out; no
  pass at all when nothing is consulted). Each perspective cites the core of its situation, so
  granular pouring gets its own core, frame, image schemas and failures. Versions moved
  (action core 0.6.0, pouring 0.7.0), so every stored hypothesis is generated again; up to 12
  situations × 4 hypotheses, each on first use. Tests and HermiT only, not run live.

- Live granular pour (2026-10-21, `knowledge_perspectives.md` §13.8): `objects_given` with the
  task "Pour the cereal from the cup into the bowl." added by the run's script, every
  perspective consulted, Sonnet 5.5, two runs ($1.09 and $1.30). Granular was chosen and all
  five hypotheses were generated for it; no pour landed, as the designator leaves no geometry
  slot to correct the 0.188 m side miss. The first run exposed a prompt defect from the
  quantity merge (a base no field ranges over was never described, so the model missed
  `about_slot` and `has_quantity_kind`); a type's entry now takes in such bases, and the second
  run generated every hypothesis on the first attempt. With the geometry open
  (`pour_geometry_open`, knowledge served from the store) the second pour landed 5 mm from the
  opening's centre, for $0.14.

- The robot faces the target before pouring (2026-10-21, `pour_prediction.md` §8): a cram_viz
  run of `objects_given` failed its third pour (2.9 rad, `MotionDidNotFinish`) and had its
  fourth answer cut off. Measured in the simulation, turning the base in place after the
  pick-up lets 2.8 and 2.9 rad finish and land (0.049 and 0.035 m) though no base position
  moves the sideways miss of natural tilts, and reach there is jagged (3.0 rad fails at the
  exact facing turn, 2.6 rad finishes without it). The plan (`steps_before_pour`, so trials,
  Execute and the pour prediction) now ends the pick-up with a turn in place to face the
  target container (`facing_pose`). The reach check's solver budget became
  `ApproachTolerances.max_solver_iterations` (400): from the turned base 200 iterations
  refused two recorded pours the simulation performs. The NEEM sqlite slow test fails in this
  environment with or without the change (no `json_msgs` module).
  Live re-run of `objects_given` (every perspective, no prediction): still no landing. The
  resolver tried right 1.85, left 1.85 (failed), right 1.57 and right 0.6 rad, kept 0.6 rad
  (0.123 m off) and did not try the near-upside-down tilts that now land; $0.60, 4.4 minutes.
  With the pour prediction on it refused instead ($0.47, 4.7 minutes): the prediction told it
  2.0 to 2.8 rad are out of reach, though the simulation performs 2.0, 2.8 and 2.9 rad, because
  the reach check followed an interpolated path the simulation does not (`pour_prediction.md`
  §8; not the base, which does not move, nor the tolerances). The check now tries the pour
  pose from every state it reached: 2.8 and 2.9 rad are accepted and the simulation's
  failures at 2.2–2.6, 3.0 and 3.07 stay refused; 2.0 rad is still wrongly refused (pinned).
  Live re-run with the corrected check: refused again ($0.34, 3.1 minutes); it searched 0.1 to
  2.4 rad, stopped after the refusals at 2.0–2.4 and never asked about 2.5–3.0, and named the
  real cause itself (`pour_side_offset` is not an open slot of `objects_given`).

- Slots an outcome check opens (2026-10-21, `architecture.md` §3 step 7): an `OutcomeCheck`
  may hold back fields of the action the query leaves out (`withheld_slots`) and open them in
  a refusal (`OutcomeFellShort.opens`); the model is shown them in an `<opened_slots>` section
  and answers for them from the next attempt on, the action's defaults standing until then.
  cram_viz's `SimulatedPouring` holds back `pour_side_offset`, `pour_approach_offset` and
  `pour_height` and opens only the one a pour's miss shows: sideways or along the approach
  beyond the opening's radius, or the mouth not above the opening. A box "adjust the pour
  geometry", ticked by default, turns it off (`adjust_geometry: false`). The resolved action
  lists an opened slot as chosen. This relaxes the rule that the query alone decides the open
  slots, as the user decided. Live (Sonnet 5.5, `objects_given`, every perspective, the liquid
  knowledge served from the store, no pour prediction): the first pour (right arm, 1.85 rad)
  missed 0.187 m, the refusal opened `pour_side_offset` alone, and the second pour set it to
  -0.184 m and landed 4 mm from the opening's centre; $0.12, 0.6 minutes, two pours. No
  turning of the base was involved in that pour's reach (the plan turns to face the bowl
  after the pick-up, as before).

## 2a. Moved onto the upstream main (2026-10-21)

`origin/main` had moved 1347 commits since the last common commit (2026-09-22). `krrood`,
`semantic_digital_twin` and `coraplex` are taken as upstream has them, with only the 19
lines of `SubClassSafeGeneric` that were ours on top; `llm_reasoner`, `llm_reasoner_viz`,
`cram_viz` and the experiments of ours were adapted. Upstream's `cramera` is the viewer
`cram_viz` grew into and stays beside it; moving `cram_viz`'s functions over is later work.

What upstream changed that reached us, and how it is met:
- **`Match` internals sit behind the underscore convention** (`_kwargs_`, `_variable_`,
  `_type_`, `_parent_`, `_matches_with_variables_`). The old names read as symbolic
  attributes instead of failing, so every read of one was found by its tests and renamed.
- **`Pose` is read through properties** (`position`, `quaternion`, `yaw`,
  `homogeneous_matrix`) instead of `to_position()` and the like.
- **`Arms` is gone: an arm is a robot part of the world** (`PouringAction.arm: Arm`). The
  query now reads `arm=variable(Arm)`, the model picks among the arms the scene's robot
  holds (`PR2LeftArm#1`, `PR2RightArm#1`), a run keeps the arm by its own name
  (`PR2RightArm`, not `RIGHT`), and a scene needs a recognised robot to offer the slot.
  Runs kept before this hold `"arm": "RIGHT"` and no longer execute.
- **A pick-up takes a grasp the object offers**: cram_viz picks the grasp whose approach
  runs most nearly the way the robot faces. The hold after the pick-up matches the old
  one to 0.01 rad in every arm joint.
- **A simulated run is given up on after 5 s without approaching its goal.** The pour's
  tilt keeps the torso rising slowly while the arm waits, which the rule gave up on, so
  the plan raises the torso right after the pick-up. Pours at 0.9 to 2.0 and 2.9 rad
  perform; **2.7 and 2.8 rad no longer do** (2.8 did before). Upstream's `MotionDidNotFinish`
  is `MotionMadeNoProgress`.
- `cram-classes.ttl` was written again from the classes as they are now.

Checked on the merged tree: `llm_reasoner_test`, `llm_reasoner_viz_test` and
`cram_viz_test` (1849 passed, 2 skipped, 1 expected failure, as before), our experiments'
tests, and upstream's `cramera_test` (1066 passed).

## 2b. Moving cram_viz's features onto cramera (spike, 2026-10-21)

Goal: every feature of `cram_viz` available in `cramera`, with neither package edited.
`cramera` has no place to add a route or a graph tab (its handler and tab list are fixed),
so the features come as an add-on package, `cramera_resolution`, on the branch
`feature/cramera_resolution`. It subclasses `cramera`'s handler and server, serves its own
panels under `/addon/`, and inserts their scripts into `cramera`'s own `index.html` as it is
served. What is `cram_viz`'s alone and to be moved: `resolution/`, `hypothesis_knowledge/`,
the designator and NEEM code, five API routes and the designator, hypothesis and ontology
views.

The spike shows:
- **The mechanism works.** A test panel defined by the add-on mounts beside `cramera`'s EQL
  and graph panels in a real browser, and `cramera`'s own routes and files are answered
  exactly as `cramera` alone answers them (12 tests, comparing against `cramera`'s own
  server).
- **`cramera` reads `cram_viz`'s scene bundles** (same `<name>/scene.json` layout; either
  `CRAMERA_SCENES` or `CRAM_VIZ_SCENES` can name them) with two data differences: a bundle's
  `placeTarget` holds `pos` where `cramera` reads `position` (the knowledge views fail on
  `KeyError: 'position'`), and the PR2 URDFs hold a `<transmission>` without `type`, which
  `urdf_parser_py` refuses (the kinematics tab fails on the two PR2 scenes). With both
  adapted, every graph tab works on all three scenes.
- Not checked: the 3D scene, as the browser used lacks WebGL.
- Both viewers default to port 8711.

Phase 1 (2026-10-21) moved the backend. The add-on now holds copies of `resolution/`,
`hypothesis_knowledge/` and the designator, NEEM and event code (imports renamed, otherwise
as `cram_viz` has them, so they stay comparable with it), and serves six routes under
`/api/resolution/` (see its README). Its own code follows the current `AGENTS.md`: the hypothesis,
ontology and designator payloads are built in `views.py` instead of inside `kb.py`; the state
the routes share is held by one `ResolutionService` instead of module-level caches; Execute
starts `python -m cramera_resolution.live --run <run>`, which performs the run and serves the
world and plan through `cramera`'s own live bridge.

Phase 2 (2026-10-21) added the Designators panel (`web/addon/designator_panel.js`, mounted
through `cramera`'s panel registry; styles and the markup helper beside it) and the route
`GET /api/resolution/runs` for its run list. It does what `cram_viz`'s EQL panel did for
designators: the chips, the choices of knowledge, Resolve, the answer with its trials, the
runs, `?run=` and Execute. Page assets are inserted into `cramera`'s own `index.html` as it
is served. 61 node tests cover the panel and its markup, and run in the suite.

Checked: 439 add-on tests (ported from `cram_viz` where the code is shared, plus the route and
launcher tests), the routes against `cram_viz/scenes/simple_pouring` itself (all three designators
available, the designator and hypothesis views answer), and a kept run performed through the
live module (the pour performed in 7 s; the bridge reported the PR2 and its four objects and
published a 45-node plan). Not moved: the panels, plan-step events in a run's log, EQL access to
the knowledge, the hardcoded demo.

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

**Robot predicates offered in the viewer (2026-10-13):** the resolver's world now has
the scene's robot annotated when a robot class is kept for it (a scene whose robot has
none, such as the garmi apartment, is resolved as before, without the robot predicates;
Execute still refuses it), the resolver names `semantic_digital_twin.reasoning.robot_predicates`
in `WORLD_MODEL_MODULES` so `llm_reasoner` loads it, and the prompt's scene geometry
leaves the robot and its parts out of `objects` (annotated, they made it fail on bodies
without a shape, and they are stated by `robot_bases`). Checked on the simple_pouring
scene with a scripted model through the viewer's resolver, no live call: `list_predicates`
now offers `robot_holds_body`, `is_body_in_gripper`, `is_gripper_holding_something`,
`bodies_in_gripper`, `is_body_gripped`, `robot_in_collision` beside `contact`, the support
and camera functions, and the class predicates (`IsReachableBy`, `GripperIsFree`, …);
`robot_holds_body` answered `false` for the pitcher, and a call that fails answered as
`PredicateFailed`. Whether a live model uses them is not yet checked.

**Seen in R7, since acted on:** the request resolved every case alike (RIGHT, the default
tilt of 1.85 rad), whatever the task, since it did not read the hypothesis yet; I3–I5 added
the knowledge tools (above). Still true: the world does not say what the cup holds, so the
material came from the task alone, unless the scene geometry and world reads establish it.

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
| 2026-10-13 | Functions declared with `symbolic_function` are predicates of the world, found by the world injector alone (krrood's wrapper recognised among the loaded modules; no change to krrood, SDT or coraplex), offered when each required argument is a plain value or an individual the world holds; a failing one is answered as `PredicateFailed`, at `WorldFunction.call`, the one place the injector catches an error |
| 2026-10-13 | The modules a world model's functions are declared in are named by the caller (`SymbolGraphSource(world_model_modules=...)`), never by `llm_reasoner`, and imported when the source is made |
| 2026-10-14 | A viewer for the system at work is a package of its own (`llm_reasoner_viz`, port 8721), read-only at first; it reads a log of events per run that every process working on the run appends to, so it can follow a run live and replay a finished one; the components report through wrappers and optional hooks and do not know the viewer; past runs have no events; prompts are shown in full |
| 2026-10-14 | The untracked `knowledge_updated` prototype (a typed-Python rewrite of the knowledge domain, `ontological/` and `microworlds/flanagan/`, with 47 passing test files) is kept off the main line on the local branch `archive/knowledge-updated` (`d6fec2059`) and was removed from the working tree: nothing in the product imports it, and `llm_reasoner.knowledge` is the knowledge component. Bytecode-only folders left under `knowledge/` by an older layout (`hypothesis`, `microworlds`, `ontological`) were removed |
| 2026-10-14 | The simulations keep what they perform as NEEMs (the executed plan, saved through coraplex's ORM) in one shared database: every pour tried and the performed run, failed motions included; the viewer links to them by showing which plan is which and how to read it |
| 2026-10-21 | Runs keep their own copy of the knowledge, but what a run admitted is added to the baseline store when it ends, so knowledge is generated once per situation and then served (replaces "no run reads what another generated", 2026-10-12) |
| 2026-10-14 | `.env` files are ignored by the repository (`.gitignore`), so that a blanket `git add` cannot commit the model API key. The untracked notebooks of `llm_reasoner/notebooks/` (written against a layout of the knowledge package that no longer exists) and the earlier `README (1).md`, which no committed README matches, are kept off the main line on the local branch `archive/llm-reasoner-notebooks-and-readme` (`946a60c53`, 12 MB, never to be pushed) and are out of the working tree |

## 5. In progress

- A viewer for the system at work, in a package of its own (`llm_reasoner_viz`, port
  8721, read-only), fed by a log of events per run (`~/.cram_viz/runs/<run>/events.jsonl`)
  that the cram-viz server, the trial process and the live runner append to, and linked
  to cram-viz by run id. Steps, each test-first and committed: V1 events, sinks and the
  client and tool wrappers (done); V2 wrappers for the world and knowledge calls, and
  hooks for the resolver's attempts and rejections and the knowledge generation (done); V3 the
  log in cram-viz with trial and Execute events and a link from the answer panel (done); V4 the
  viewer server (done); V5 the screen: map, timeline, inspector (done); V6 replay, filters, attempts strip,
  pour-trial view, links both ways and a live check (done).
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
  reorganisation added no functionality; the request from the resolver (K3, situation
  features), the knowledge tools and the structured generation prompts followed and are
  done (the next subsection, §6).
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
| Computed world values: poses, reachability, contact, which gripper holds what | Grounds the arm and geometry choices the model now calls arbitrary | Functions: met for those taking individuals and plain values (contact, support, and the robot and camera ones, which the viewer's resolver now has: it annotates the robot and names their module); not met for those taking a pose, a vector, an array or a list. Properties (poses, bounding boxes): deferred; options (c) SDT marks its properties with krrood field metadata, (d) `llm_reasoner` reads zero-argument properties generically. No choice yet |
| Linking the knowledge component to the resolver and the world injector | Lets the resolver use action knowledge | **Done** (R1–R7, I1–I5; `architecture.md` §3.6, §3.7): the resolver obtains the hypothesis before every request to its model and reads it with the knowledge tools |
| Representing knowledge for tools | Several perspectives, organised by action concept, addressable fragments, objectives and consequences first-class | Requirements K1–K4 recorded below. Designed in [`knowledge_perspectives.md`](knowledge_perspectives.md): an action core and one module per perspective, each importing `cram-aicor-ontology.ttl`; the model proposes each perspective's vocabulary, which is checked and reviewed before use. Built: the core and the action frame, with knowledge, tools, roles-to-slots and the phase 4 measurement (phases 0–4, §2). Not built: image schemas (5), failures and corrections (6), the Flanagan modules on the core's slots (7), more actions (8). Trap cases come before any further perspective |
| Structured prompts for the knowledge component | Its system, initial and repair prompts were prose built in Python, unlike the resolver's tagged sections | **Done** 2026-10-12 (`architecture.md` §3.2): tagged sections, JSON data and named rules; the system prompt grew from about 80k to 130k characters (the types as indented JSON) |
| Designator slots supplied from the SDT world (containers, arms) | Replaces hand-written bindings | Deferred |
| Choosing smaller pouring structures from the world state | The full 12-phase pouring structure is always required | Deferred |
| An MCP server over the world-access contract | For consumers outside the resolver's process | When a second consumer needs it |
| Live tests in the repository | The live suite lived in scripts outside the repository | **Done** 2026-10-12 (`architecture.md` §5): `experiments/llm_reasoner_kitchen` with its tests; the live tests run only when opted in, five cases have scripted twins run with the experiments suite. `llm_reasoner`'s own tests are not yet in CI (the user's decision of 2026-10-12) |
| Making the model aware of the predicates the world offers and what they do | Nothing in the resolver's prompt names a predicate; the model finds one only by calling `list_predicates` from its tool description, so it may never use `contact` or the support checks | Open, long-term; options: a prompt rule telling it to look for a predicate when it checks a relation, or stating the predicates offered for the kinds a request involves, as the candidates of a slot are |
| Loading the modules that declare a world model's functions (such as `semantic_digital_twin.reasoning.robot_predicates`) | The function scan reads only loaded modules, so a function in a module nobody imported is never offered | **Done** 2026-10-13: `SymbolGraphSource(world_model_modules=...)`, named by cram_viz's resolver (§3) |
| Fewer wasted calls (opening `list_kinds`, listing many bodies, `name` before `name.name`) | Cost only; results are correct | Open |
| An explicit "either arm" answer | Arbitrary arm choices are stated only in `why` | Open; would change the action interface |
| Research direction: language-model commonsense checked against the ontology for designator resolution | Long-term | Open |

### Knowledge component requirements (2026-10-12)

| # | Requirement | Against today |
|---|---|---|
| K1 | Generation backed by an ontology linked to CRAM-AICOR, so that language-model knowledge stays consistent, structured and reasonable | Met: the Flanagan ontology links CRAM-AICOR and DUL; acceptance and HermiT enforce it |
| K2 | Knowledge as objects generated from the ontology, queryable with EQL over the knowledge component's own graph, independent of krrood's symbol graph (reworded 2026-10-12; first stated as "first-class krrood symbols") | Met: independent of krrood's symbol graph since I1; `KnowledgeGraph.instances_of` gives EQL its domains since I2 |
| K3 | Requests come from the resolver, not as a natural-language instruction; knowledge is generated for that input | Met (R3–R6): the query comes first, the store is keyed by root class, fixed values and ontology versions, and the instruction is provenance only |
| K4 | The knowledge component is independent of the world; situational facts it needs (such as a material's flow regime, whether a target is open, sizes, or poses as values) reach it only through the resolver, in the ontology's terms | Met for independence (it imports no world) and, for the material regime and termination reason, for receiving situational facts (R1–R6, the situation features); sizes and poses are not features |

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
   designator slots, as the cram_viz Designator tab sketches. *Answered 2026-10-16
   (phase 3 of `knowledge_perspectives.md`):* a mapping kept in `described_actions.json`,
   checked automatically and reviewed as a diff.
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
- cram_viz's `kb.build_symbol_graph` still rebuilds krrood's symbol graph and gives
  the knowledge vocabulary as its reason, which no longer holds since I1; whether
  cram_viz's own symbols still need the rebuild is unchecked.

**Outside `llm_reasoner`** (to report to their owners, not to change here):
- krrood's verbalizer writes contradictory bounds as "between 0.01 and 0.005" and enum
  lists as `[<CuttingTechnique.SLICE: 1>, …]`.
- krrood does not enforce an `exists` condition over a variable declared outside the
  match; conditions must use `query._variable_.…`.
- coraplex: `ToolMotionAction.arm` and `.tool` have no docstrings; `PouringAction` has
  no speed or duration field; `composite/searching.py` fails to import
  (`PerceptionObjectNotFound` is missing).
- The test kitchen gives the bread's body two `Connection6DoF` connections as child.
- The experiments' tests rebuild every package's ORM interface, and the experiments' control
  loop needs ROS's `control_msgs`, which this machine lacks; they run with
  `--orm-build=never` here.
- `krrood/test/dataset` tracks a generated `ormatic_interface.*.py.tmp` file of 3851 lines.

## 8. Continuing the work

1. Read `architecture.md` for how things fit, then this file for where they stand.
2. Run the offline tests: `python -m pytest test/llm_reasoner_test`, and
   `scripts/format_docstrings.py` on changed files.
3. Live checks with Sonnet 5.5 may run when needed (decision of 2026-10-12); record
   each one's outcome and cost in §3.
4. Record decisions in §4 and move items between §5 and §6 as they change.
