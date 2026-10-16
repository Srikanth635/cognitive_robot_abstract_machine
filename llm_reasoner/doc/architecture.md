# `llm_reasoner` architecture

This is the single description of how `llm_reasoner` is built: its components, what each
one is for, what goes in and comes out, and how the components talk to each other. The
state of the work — what is done, in progress and proposed — is kept separately in
[`status.md`](status.md). The scientific contract of the predictive-manipulation
ontology is [`flanagan_scientific_contract.md`](flanagan_scientific_contract.md).

## 1. Principles

- **A base layer on krrood.** `llm_reasoner` belongs to the krrood layer of CRAM. It
  imports krrood (and `random_events` / `probabilistic_model` through it) and never a
  world model, robot or plan package such as `semantic_digital_twin`, `coraplex` or
  `cram_viz`. It learns about a world only through krrood's symbol graph and the
  dataclasses and docstrings of the classes it finds there.
- **Components with one responsibility each**, connected by declared contracts, not by
  reaching into each other's internals (dependency inversion).
- **Structured prompts.** Everything a language model reads is tagged sections holding
  data (JSON) and named rules; the only English is content: a query as krrood words it,
  a caller's task, docstrings and source.
- **Generic behaviour.** Nothing is written for one action, object or world; behaviour
  follows from types, fields, defaults, docstrings and the query.
- **Accuracy over token cost.** A little extra cost is preferred to any loss of
  accuracy, precision or structure that would cause repairs or reruns.
- **The ontology is the single source of truth for knowledge.** Knowledge has a working
  form, Python objects that EQL queries in the closed world, and a reasoning form, RDF
  that HermiT checks in the open world; both are generated or derived from the
  ontology, never written by hand.
- **Two graphs.** krrood's symbol graph is the universal one: the live world, created
  once per process. The knowledge component keeps its own local graph of admitted
  knowledge, which has no part in krrood's. The resolver is the component that reads
  both: it answers a query by retrieving what it needs from the world (through
  `WorldAccess`) and from the knowledge (through `KnowledgeAccess`). Decided
  2026-10-12; being built (`status.md`, steps I1–I5).

## 2. Components

```mermaid
flowchart LR
    caller["Caller<br/>(plan, cram_viz, test)"] -- "krrood query<br/>+ optional task" --> resolver
    subgraph llm_reasoner
        resolver["Resolver<br/>resolver/"]
        world["World injector<br/>world/"]
        knowledge["Knowledge component<br/>knowledge/"]
        clients["Clients<br/>clients/"]
    end
    resolver -- "WorldAccess<br/>(handles, holders, tools)" --> world
    resolver -- "prompts / tool loop" --> clients
    knowledge -- "structured completion" --> clients
    clients -- "API" --> llm["Language model"]
    world -- "reads" --> graph["krrood SymbolGraph<br/>(the live world)"]
    resolver -- "knowledge request<br/>(access.py only)" --> knowledge
```

| Component | Package | Role in one sentence |
|---|---|---|
| World injector | `world/` | Answers questions about the live world, using krrood alone |
| Resolver | `resolver/` | Turns an underspecified krrood query into a resolved instance, choosing what the query leaves open |
| Knowledge component | `knowledge/` | Generates and stores ontology-backed, world-independent action knowledge |
| Clients | `clients/` | Talk to language models, and define the vendor-neutral tool interface |

`reasoner.py` assembles the system in one place: `Reasoner(client, knowledge)` holds the
resolver's model, a `KnowledgeService` and a world (by default a `WorldView` whose
symbol-graph source leaves out `krrood` and `llm_reasoner.knowledge`), and
`Reasoner.backend(**settings)` gives a backend with the world tools that consults the
knowledge component about every action it resolves.

### 2.1 World injector (`world/`)

**Role.** The world's data and state for any component that needs it: which kinds of
thing exist, which individuals, what they are, what holds them, what their fields reach,
and what the world model's own predicates say. It is useful to the resolver, to the
knowledge component and to any later component.

**Does not:** change the world, plan, or know any world model by name.

**Input.** The live symbol graph (`sources.py`, `SymbolGraphSource`; every loaded
`Symbol` class outside krrood is a kind, every live symbol an individual) and the loaded
modules, whose functions declared with krrood's `symbolic_function` are the world
model's functions.

**Output.** Typed answers (dataclasses in `values.py`, `description.py`, `kinds.py`,
`finding.py`, `holders.py`, `predicates.py`), and the same answers as JSON for a model.

| Module | Responsibility |
|---|---|
| `access.py` | **The contract**: `WorldAccess`, `WORLD_ACCESS_VERSION`, shared names (see §3.3) |
| `view.py` | `WorldView`, the façade answering every question and implementing `WorldAccess` |
| `naming.py` | `HandleRegistry`: handles `Kind#n`, stable for the world's lifetime, held weakly |
| `classes.py` | `DocumentedClass`: summary, field meanings, content fields, lineage of a class |
| `describer.py`, `value_forms.py`, `values.py`, `description.py`, `rendering.py` | Walking an individual's fields into described values, and writing them as text |
| `kinds.py` | Listing, describing and resolving kinds by identifier or unique name |
| `finding.py`, `paths.py` | Searches as krrood EQL queries with conditions; following field paths |
| `holders.py` | Which individuals hold an individual, and in which field |
| `predicates.py` | Listing, resolving and evaluating the predicates a world model declares: krrood `Predicate` / `SymbolicFunction` classes and functions (`DeclaredPredicate`) |
| `functions.py` | `WorldFunction`: a function declared with `symbolic_function`, what it takes and says, and calling it |
| `tools.py`, `tool_definitions/` | The eight questions as tools for a model (§3.3) |
| `encoding.py`, `exceptions.py` | Answers as JSON; `WorldQuestionError` and its kinds |
| `class_ontology/` | The ontology of a code base: the classes a selection file names, written as OWL from the live classes (`python -m llm_reasoner.world.class_ontology <selection.json> <ontology.ttl>`); not used by the injector's tools. CRAM's selection and ontology are kept in `knowledge/ontologies/cram_classes/` |

**Errors.** A question naming something the world lacks, or a function of the world
model failing, raises a `WorldQuestionError` (`UnknownKind`, `AmbiguousKind`,
`UnknownHandle`, `ExpiredHandle`, `UnknownField`, `InapplicableComparison`,
`NotAPredicate`, `AmbiguousPredicate`, `UnknownArgument`, `MissingArgument`,
`PredicateFailed`). As a tool, it becomes an error answer the model can correct or ask
around, never a crash.

**Functions of the world model.** krrood keeps no record of the functions declared with
`symbolic_function`, so the source recognises krrood's wrapper among the members of the
loaded modules outside the excluded packages, the way it finds kinds among loaded
classes. A function is a predicate of the world when each argument it must be passed is
a plain value (`bool`, `int`, `float`, `str`) or an individual of a kind the world holds
or of a kind those specialise, judged by the type name its annotation writes. Others are
neither listed nor evaluated: a function taking a pose, an array or a list, one taking a
camera while the world holds none. They appear as the world gains such individuals. A
function that raises is reported as `PredicateFailed` with the error and what it said;
`WorldFunction.call` is the one place the injector catches an error, since it runs the
world model's code there.

A function is found only once its module is loaded, and the source loads nothing by
itself. The caller names the modules of its world model in
`SymbolGraphSource(world_model_modules=(...))`, which imports them when the source is
made (an unknown name raises `ModuleNotFoundError` then, not during a conversation).
`llm_reasoner` names no world model; imports are process-wide, and a module also brings
the kinds it declares.

**Limits.** It reads dataclass fields, krrood predicate classes and the functions above.
Values a world model computes in properties (poses, bounding boxes) are not reachable
yet, and a function appears only once its module is loaded (see `status.md`).

### 2.2 Resolver (`resolver/`)

**Role.** Resolve an underspecified krrood query — a `Match` such as
`the(CuttingAction)(object_to_cut=variable(Body), arm=..., …)` — into instances of the
queried type, letting a language model choose what the query leaves open and grounding
every chosen individual in the live world.

**Does not:** take natural-language instructions as the request (a task is optional
context only), check preconditions or executability (navigation, opening a drawer, …
belong to the plan), or construct the action itself (krrood does).

**Input.**

| Input | Form |
|---|---|
| The query | A krrood `Match`; `variable(Kind)` marks a slot choosing an individual, `...` a value to generate; `where(...)` adds conditions; `an(...)` with `expression.limit(n)` asks for up to n alternatives |
| Optional task | `TaskContextProvider(goal=…, constraints=…)`, intent, never fact |
| The world | A `WorldAccess` (§3.3), given through `LanguageModelBackend.with_world_tools(client, world, …)` |
| The knowledge component | A `KnowledgeRequester` (§3.6), given as the `knowledge` setting; `Reasoner.backend` always gives one |

**Output.** The resolved instances of the queried type, constructed by krrood and
meeting the query's `where`-conditions; or one of these outcomes:

| Outcome | Meaning |
|---|---|
| `NoFittingReferent` | No candidate of an open slot is what the request asks for |
| `AmbiguousReferent` | Several candidates fit equally and nothing prefers one (carries them) |
| `UnsatisfiableRequest` | The query's conditions cannot all hold together |
| `RepairBudgetExhausted` | The answers stayed unusable after the repair (last error attached) |
| `ReferentSlotHasNoDomain`, `NoCandidateReferents` | Refused before any model call |

**How it works** (`backend.py`, `LanguageModelBackend`, a krrood `GenerativeBackend`):

1. **Reflect the slots** (`reflection.py`): each field of the query is `BOUND` (given),
   `VALUE` (`...`: a number, text, boolean or enum), `REFERENT` (an individual of a
   domain) or `SORTAL` (a kind). Optional fields defaulting to `None` may be left empty.
2. **Consult the knowledge component** (`knowledge.py`; §3.6), when the backend has a
   knowledge requester and the model is to be asked: a first resolution chooses the
   situation features, and the knowledge component serves the hypothesis for them,
   generating it on a store miss. The consultation is kept on the backend
   (`consultation`, `situation_backend`), and the knowledge tools over the hypothesis
   (`knowledge_tools`, §3.7) are offered to the request below beside the world tools
   when the backend asks with tools, with the `knowledge` rule saying how to read and
   cite them. A backend asking without tools is offered no knowledge.
3. **Build the prompts** (`prompts.py`, `prompt_format.py`; §3.2).
4. **Ask** through a `Completion` (`completion.py`): `PlainCompletion` (one completion,
   candidates listed by integer token) or `ToolCompletion` (a conversation in which the
   model may call the world tools, candidates named by handle).
5. **Decode** the JSON answer: handles back to live individuals through `WorldAccess`,
   enum names to members, `null` where allowed; verdicts become the outcomes above.
6. **Construct and check**: values are written into the query, krrood constructs the
   instances, and the query's `where`-conditions are checked.
7. **Try** (`outcomes.py`), when the backend has an `OutcomeCheck`: the caller performs
   what the answer constructs, in a simulation for instance, and refuses the answer with
   `OutcomeFellShort`, holding the outcome as data, when it falls short. The situation
   pass is never checked.
8. **Repair once** (`max_repairs = 1`) when an answer is unusable, breaks a condition or
   falls short when tried: a fresh conversation with the rejected answer and the
   rejection as data, its kind naming the failure.

| Module | Responsibility |
|---|---|
| `backend.py` | Orchestration: ask, decode, construct, validate, repair; `log`, `conversations`, `rules` |
| `reflection.py` | Slots of a query: kind, type, docstring, default, domain |
| `prompts.py` | System prompt, request, repair; the general rules |
| `prompt_format.py` | The resolver's prompt vocabulary: its sections, rules and data keys, written in the shared structure of `llm_reasoner/prompt_structure.py` (`PromptBlock`, `PromptRule`, `render_blocks`) |
| `referents.py` | How candidates are named and answered: `IntegerTokens`, `WorldHandles`; verdict rules |
| `context.py` | Context providers (task, situation, composite) and candidate renderers |
| `action_context.py` | `ActionContextProvider`: states the action a resolution serves, as its own `<request>`, `<slots>` (its open slots with their candidates in the world) and `<given>` in an `<action>` section |
| `knowledge.py` | `KnowledgeRequester`: consults the knowledge component, through its public module only, about the action a query asks for (§3.6); `KnowledgeConsultation`: the designator, the situation chosen and the identifier of the hypothesis served |
| `completion.py` | Plain and tool-using completion; the text-answer rule |
| `outcomes.py` | `OutcomeCheck`: tries the instances an answer constructs before they are accepted |
| `exceptions.py` | Retryable failures and outcomes |

**Settings** (`LanguageModelBackend`): `number_of_candidates` (3 unless the query sets a
limit), `max_repairs` (1), `source_detail` (`FULL`: the source of the queried type and
of its parents declared in the same module; `SIGNATURE`: docstrings; `NONE`),
`context_provider` (none by default), `candidate_renderer` (structural description),
`open_values` (`CHOICE`: open values are chosen to serve the request, the default;
`FACT`: only what the action, the given values, the task, the situation or the world
establishes, `null` otherwise), `knowledge` (a `KnowledgeRequester`; none by default,
always one from `Reasoner`).

### 2.3 Knowledge component (`knowledge/`)

**Role.** Generate commonsense knowledge about *kinds* of action with a language model,
and keep it consistent and queryable through an ontology: what an action normally
involves — its phases, roles, conditions, forces, sensory predictions, failures and
corrections — in the terms of the predictive-manipulation framework of Johansson and
Flanagan, linked to CRAM-AICOR and DUL (the scientific contract is
[`flanagan_scientific_contract.md`](flanagan_scientific_contract.md)).

**Does not:** name individuals of a world, robot or episode (a hypothesis says what
pouring involves, never which bowl or which arm), resolve designators, or read the live
world.

**Input.**

| Input | Form |
|---|---|
| An instruction (optional) | Text naming what the knowledge is for, such as "pour milk from the carton into the bowl"; stated to the model and recorded as provenance, not part of the store's key |
| A query | A krrood `Match` on a generated vocabulary class, such as `the(pouring.PouringPhaseStructure)(decomposes_task=ConceptReference(iri=CRAM_AICOR_NAMESPACE + "Pouring"))`; the fields it fixes are given, the rest is generated |

**Output.** An admitted *hypothesis*: instances of the generated vocabulary (one root
structure and the fragments making it up), stored as Turtle with a record of how it was
made, and read through `HypothesisView`. `HypothesisRejected` (with the remaining
problems and every exchange) when no answer is admitted within the repair budget.

**The pipeline.**

```
instruction + query ──► 1 generation ──► 2 acceptance ──► 3 admission ──► 4 store ──► 5 reading
                         (fragments)      (closed world)    (open world)    (Turtle +   (HypothesisView,
                              ▲                 │                │            record)     graph)
                              └──── patch repair (up to 2) ◄─────┘
```

| Stage | Package | What happens |
|---|---|---|
| 0 Vocabulary | `ontologies/`, `ontology_translation/` | The Turtle modules of two families, the action perspectives (`action-core`, `action-frame`; see [`knowledge_perspectives.md`](knowledge_perspectives.md)) and the predictive manipulation microworld (`core`, `object-dynamics`, `prehension`, `pouring`, `gaze`), are the source of truth; Python vocabulary modules are generated from them with krrood's code generator (`python -m llm_reasoner.knowledge.ontology_translation`) and checked in; a test fails when they differ from the ontology. Every class carries its IRI and its OWL restrictions |
| 1 Generation | `generation/` | `HypothesisBackend`, a krrood `GenerativeBackend` like the resolver, reads the request (`HypothesisRequest`) and runs a `GenerationDialogue`. `GenerationSchema` reads every vocabulary class reachable from the queried type; the model answers with a structured completion (`complete_structured`) shaped by that schema; `FragmentDecoder` turns the answer into vocabulary instances |
| 2 Acceptance | `acceptance/` | Closed-world checks — what is missing is an error, since a model tends to generate too little: `restrictions_hold` (each fragment meets its classes' OWL restrictions), `ranges_hold` (every linked value has the type its property allows), `shapes_hold` (SHACL shapes for rules across fragments, such as transitions connecting all phases), `components_declared` (everything a structure reaches, it declares), plus the values the query fixes. Violations are errors or warnings; warnings are kept with the hypothesis |
| 3 Admission | `admission/`, `representation/` | Optionally (`consistency=HermitCheck(...)`), the hypothesis is written as RDF and checked in the open world against the ontology, CRAM-AICOR and DUL with HermiT; `ContradictionLocaliser` names the fragments that together contradict them. Concept references are checked against the linked ontologies (`linked_terms`) |
| Repair | `generation/` | Errors from stages 1–3 are shown to the model, which answers with a *patch* (fragments added or changed, identifiers removed) applied to the answer with the fewest problems so far (`max_repairs = 2`) |
| 4 Store | `storage/` | `HypothesisRecord` (instruction, root type, fixed values, client, every exchange, warnings, consistency verdict, the versions of the ontology modules its vocabulary uses, the records it builds on) and the hypothesis as Turtle, one directory per hypothesis in `HypothesisStore`. A request is answered from the store when a record has the same root type and fixed values (situation features included) under the current module versions, whatever instruction it was generated for; otherwise it is generated |
| 5 Reading | `hypotheses/` | `KnowledgeService.view_for(query, instruction)` returns the stored or newly generated hypothesis as a `HypothesisView`, which answers questions crossing fragments: `phases`, `next_phases`, `requirements_of`, `world_assumptions`, `gaze_fixations_for`, `force_couplings_in`, `failure_events`, `corrections_at`, `force_dynamics`. Every fragment is cited by the IRI it is stored under (`Citation`). `HypothesisGraphProjector` projects a hypothesis into the graph cram_viz shows |

**The answer format.** A flat list of fragments that refer to each other by
identifier, since phases and events form cycles that nesting cannot express:

```json
{"fragments": [
  {"id": "pouring", "type": "PouringPhaseStructure", "decomposes_task": "aicor:Pouring",
   "defines_component": ["tilting", "flow_onset", …], …},
  {"id": "tilting", "type": "ActionPhase", "initiated_by": "flow_onset",
   "is_optional": false, …}]}
```

A field holds, by its kind: the `id` of another fragment, an enumeration member, a
concept as a prefixed term (`aicor:<Name>`, `dul:<Name>`, or `hypothesis:<Name>` for a
concept CRAM-AICOR and DUL lack), a number, a truth value or text.

A repair answers `{"fragments": [<added or changed>], "remove": [<identifiers>]}`.

A request may fix a field to a fragment of a stored hypothesis (a `Citation`); the
fragments of the cited hypotheses are then shown in a `<given>` section, written in this
format, and the answer refers to them by their names without writing them again.
Fragments are created first and linked second; a malformed answer is not an exception
but a list of problems shown to the model.

**Components inside, by responsibility** (all under `knowledge/`, lowest layer first;
each package imports only the packages above it in this table, which a test checks):

| Package | Modules | Responsibility |
|---|---|---|
| `ontologies/` | `__init__.py`, `terms.py`, `properties.py`, `linked.py`, `vocabulary.py`, `situation.py`, `described_actions.py` and `.json`, `action_perspectives/`, `predictive_manipulation/`, `cram_classes/` | The ontology files, the shared terms, the descriptors applying the object properties' closed-world entailments, the linked ontologies (CRAM-AICOR, DUL), the generated vocabulary and the lookup from IRIs to its classes, the situation features a structure accepts, and which structure describes which action designator |
| `ontology_translation/` | `reading.py`, `declarations.py`, `writing.py`, `templates/` | Ontology → declarations → Python modules |
| `representation/` | `hypothesis.py`, `rdf.py` | A hypothesis in memory as vocabulary instances, and as RDF in the ontology's terms (written and read back) |
| `acceptance/` | `rules.py`, `restrictions.py`, `derived.py`, `shapes.py`, `core.py`, `suite.py` | Rules and violations, closed-world class membership, the microworld's suite |
| `admission/` | `consistency.py`, `hermit.py` | Open-world consistency with HermiT, and locating contradictions |
| `vocabulary_proposal/` | `perspective.py`, `catalogue.py`, `proposal.py`, `checks.py`, `writing.py`, `prompts.py`, `proposer.py`, `__main__.py`, `perspectives/` | A model proposes the vocabulary of one action perspective (stage 1 of [`knowledge_perspectives.md`](knowledge_perspectives.md) §6); the rules, the Python generator and HermiT check it, and it is written as a module under `ontologies/action_perspectives/proposals/` for a person's review |
| `storage/` | `record.py`, `store.py`, `citation.py` | Records of admitted hypotheses, the store, citations of stored fragments |
| `generation/` | `request.py`, `schema.py`, `briefs.py`, `prompts.py`, `wire.py`, `assessment.py`, `dialogue.py`, `recording.py`, `backend.py` | The request a query makes; output shape, what the model is asked to describe for the root's module, prompts, decoding and merging patches; judging an answer (`HypothesisAssessor`); asking and repairing (`GenerationDialogue`); recording and storing (`HypothesisRecorder`); `HypothesisBackend` orchestrating them as a krrood generative backend |
| `hypotheses/` | `service.py`, `knowledge_access.py`, `questions.py`, `tools.py` and `tool_definitions/`, `knowledge_graph.py`, `view.py`, `graph.py` | The service, which fulfils the knowledge-access contract (`KnowledgeAccess`, version 1.0: `serve`, `tools(record_id)`, `instances_of`, `is_vocabulary_class`); the knowledge tools (§3.7); the knowledge component's own graph (`KnowledgeGraph`: every stored hypothesis read once, so fragments keep their identity, and `instances_of(kind)` as EQL's domain over knowledge); cross-fragment questions (`HypothesisView`, with `instances_of` within one hypothesis); the graph projection drawn by cram_viz |
| `access.py` | — | **The public module**: every name another package may use; cram_viz imports only from it (a test checks) |

Records store Python module paths (the record class and the root type), so moving a
module that a record names requires migrating stored records.

**How it relates to the resolver.** Both are krrood generative backends answering an
underspecified query with a model, checks and repairs. The resolver produces one
resolved instance grounded in the live world; the knowledge component produces reusable
description knowledge checked by an ontology.

#### Intended role and design (agreed 2026-10-12; not built yet)

**Role.** The knowledge component provides the *stereotypical* knowledge associated with
an action, from several perspectives — FrameNet frames, Minsky frames, image schemas,
and the Flanagan predictive-control structure that exists today. Together they state an
action's objectives, consequences, roles and their expectations, preconditions, defaults
and failure modes. Not all of it maps onto a designator's parameters; its purpose is to
give the resolving agent a comprehensive understanding of the action and something
explicit to ground its choices on, so that it can say *why* it decided a parameter.

| Component | Knows | Kind of truth |
|---|---|---|
| World injector | What *is* in this world now | Fact |
| Knowledge component | What an action *normally* involves, given situation features in its own terms | Stereotype, independent of any world |
| Resolver | What to do *here* for one query | A decision with its reasons |

| Perspective | What it contributes (pouring as the example) |
|---|---|
| FrameNet frame | Frame elements: the material, the source, the goal, the manner |
| Minsky frame | Slots with default expectations: typically a liquid, an upright open container, poured until the desired amount |
| Image schemas | Underlying structure: SOURCE–PATH–GOAL, CONTAINMENT, VERTICALITY |
| Flanagan predictive control | Phases, events, sensory predictions, failures and corrections |

**Communication: knowledge as tools.** The resolver pulls the parts it needs, as it does
from the world, rather than receiving knowledge pushed into its prompt, because a
hypothesis will grow beyond what a prompt should carry:

```
                    ┌── world tools ──────► world injector     "what is here?"
resolver (agent) ───┤
                    └── knowledge tools ──► knowledge component "what does this action
                                                                  normally involve?"
answer: slot values and reasons citing both (world handles, knowledge identifiers)
```

**Requirements on the knowledge itself.**

| Requirement | How it is met |
|---|---|
| Generated by a language-model agent | As today, by a krrood generative backend with checks and repairs |
| Structured and backed by an ontology | The ontology is the source of truth; the Python vocabulary is generated from it |
| Consistent under ontological reasoning (HermiT) | Each hypothesis written as RDF in the ontology's terms and checked in the open world |
| Queryable with krrood's EQL | Vocabulary classes are dataclasses generated from the ontology (`DescriptionEntity`), outside krrood's symbol graph (I1); EQL queries them over domains from the knowledge component's own graph, such as `variable(core.ActionPhase, domain=graph.instances_of(core.ActionPhase))` (I2) |
| Requested by the resolver, not by a natural-language instruction | To build: the knowledge component answers the resolver's request for an action and generates for it |
| Independent of the world | Kept: it imports and calls no world; situational facts it needs come in the resolver's request, in ontology terms |

The resolver is the only component reading the world, and the only one grounding roles
in world individuals:

```
world injector  ◄── resolver ──►  knowledge component
 (WorldAccess)      (reads the world;   (no world; accepts situation
                     grounds roles)      features in its own terms)
```

Which situation features the knowledge accepts, and how the resolver derives them, is
open (`status.md`).

**Design points this implies** (to be designed with the representation requirements):

- A `KnowledgeAccess` contract mirroring `WorldAccess`: versioned, its own tools, and
  stable identifiers for knowledge fragments so they can be cited.
- Tools shaped around the agent's questions (which perspectives exist for an action;
  objectives and consequences; roles and expectations; preconditions and failure modes;
  what an element connects to), not around storage. Much of the world injector's
  machinery — walking fields, handles, holders, paths, searches, answer shapes — is
  generic over its source and could serve knowledge through a knowledge source.
- **Separate sources for world and knowledge.** Knowledge fragments are krrood symbols
  too; loaded into the same process, the world injector's symbol-graph source would show
  them as if they were things in the world. *What is* and *what normally is* must never
  mix.
- Knowledge organised by action concept (such as `aicor:Pouring`) and perspective, not
  by instruction text; perspectives as linked ontology modules about the same concept;
  objectives and consequences as first-class content.
- Reasons as data: the resolver's `why` growing into reasons that cite knowledge
  identifiers and world handles.

### 2.4 Clients (`clients/`)

**Role.** Shared infrastructure: talking to language models, and the vendor-neutral
tool interface the world injector implements and the clients run.

| Type | Contract |
|---|---|
| `LanguageModelClient` | `complete(system, user) -> str` |
| `ToolUsingClient` | `complete_with_tools(system, user, tools) -> ToolConversation` (answer, every `ToolCall`, `TokenUsage`) |
| `AnthropicClient` | Both, and `UsageReporting`; a written-out tool loop (`max_tool_turns = 10`), prompt caching of the system prompt and of the growing conversation |
| `UsageReporting` | `latest_usage`: the tokens of the latest plain or structured completion made in the calling thread, which a completion returning only text cannot carry |
| `OpenAIClient` | `complete` only |
| `ScriptedClient` | Replays recorded answers and tool requests, for tests |
| `Tool`, `ToolDefinition`, `ToolDescription`, `ToolBox`, `ToolOutcome` | The tool interface (§3.3) |

### 2.5 Observation (`observation/`)

**Role.** Let a viewer follow the system at work, without the components knowing one is
there: who asked what of whom, and what came back, as events written as plain JSON.

**Does not:** change what a component does. Nothing here is imported by the components;
a wrapper reports their calls from outside, and the default is to report nowhere.

| Module | Responsibility |
|---|---|
| `events.py` | `Event` and its kinds: `ModelAsked`, `ModelAnswered`, `ModelFailed`, `ToolCalled`, `ToolReturned`, `ToolFailed`, `ToolRefused`, `ComponentCalled`, `ComponentReturned`, `ComponentFailed`; `Component`, `CompletionStyle`. A call's two events share an identifier. |
| `sinks.py` | `EventSink`: `NullSink`, `MemorySink`, and `JsonLinesSink`, which appends one line per event to a log with a single write so writers in several processes share it |
| `observed.py` | `ObservedClient` and `ObservedToolClient` wrap a `LanguageModelClient` and report each completion with its prompts, schema, answer, duration, tokens and cost; `ObservedTool` reports each call a model makes of a tool, and the calls the tool box refused before running (unknown tool, input breaking its schema) are reported after the conversation, as `ToolRefused` just before the call run next. A wrapper has the identity and abilities of what it wraps, so a client that cannot let a model call tools still cannot. |
| `components.py` | `ObservedWorld` and `ObservedKnowledge` wrap `WorldAccess` and `KnowledgeAccess` and report each call between components with the operation, what it asks about and what it answers, an individual or a collection reported by its kind or its size |

The resolver reports its own steps through an optional `observer` (a sink, `NullSink` by
default): `ResolutionStarted` with the slots to fill and, for a slot chosen from
individuals, how many there are; `KnowledgeConsulted`; `AttemptStarted`;
`AttemptRejected` with the failure's name, what it found and the rejected answer (so a
pour that fell short in the simulation shows its outcome); and `ResolutionFinished`,
with why each instance was chosen and what that rests on, or `ResolutionFailed`. A
resolution has a purpose: the action asked for, or the choice of the situation features
of the knowledge describing it (`ResolutionPurpose`).

The knowledge generation reports the same way (`HypothesisBackend.observer`, passed on to
the dialogue): `HypothesisRequested` with the vocabulary class and the values the request
fixes, `GenerationAttemptStarted`, `GenerationAttemptRejected` with the problems found,
and `GenerationFailed`, or `HypothesisServed` saying whether the store held the
hypothesis or generating it took so many answers, and with what warnings it was admitted.

`Reasoner(sink=..., prices=...)` assembles all of it: the resolver's model, world and
knowledge component wrapped to report to the sink as the resolver's, the backend's
observer set, and each tool of the model attributed to the world injector or the
knowledge component. Without a sink nothing is wrapped. The knowledge component's own
model client is not reachable from there; whoever builds it wraps its client as the
knowledge component's (`ObservedClient.of(client, sink, Component.KNOWLEDGE)`) and sets
its backend's observer.

An event holds the prompts and answers in full and never a credential: the wrappers see
only what a client is given and returns.

The viewer that reads these logs is the package `llm_reasoner_viz` (`llm-reasoner-viz`, port
8721, read-only, standard library only): a server listing the runs that have a log and
following one as it grows, as a batch or as server-sent events whose ids are offsets in the
log, so a reader resumes where it stopped. Its page builds *spans* from the events (a
model call, a tool call, a call between components, a resolution and its attempts, a
hypothesis and its generation attempts, a pour tried in the simulation, a step of the
plan), each with a start, an end and how it ended, nested in what it happened during, and
shows them three ways: a map of the components with the number of calls over each
connection and what each is doing, a timeline with a row per component whose bars grow
while the run goes on, and an inspector with the prompts of a model call as the tagged
sections they are made of, the tools offered and called, answers, slots, why an attempt was
rejected, and the outcome of a pour. cram_viz's answer panel links to it by run
(`LLM_REASONER_VIZ_URL`). The viewer spells the events' names as the writers do, which its
tests check.

## 3. How the components communicate

### 3.1 Caller → resolver

```python
backend = LanguageModelBackend.with_world_tools(
    client=AnthropicClient(model="claude-sonnet-5-5"),
    world=WorldView(),                      # any WorldAccess
    context_provider=TaskContextProvider(goal="Cut the bread into thin slices."),
)
query = the(CuttingAction)(object_to_cut=variable(Body), tool=variable(Tool),
                           arm=..., technique=..., slice_thickness=...)
query.where(query.variable.slice_thickness < 0.015)      # optional conditions
actions = list(query.evaluate(backend=backend))           # resolved CuttingActions
```

A condition must be written through the match's own variable (`query.variable.…`);
krrood does not enforce one written over a variable declared outside the match.

### 3.2 Resolver → language model (the prompt protocol)

Every prompt of `llm_reasoner`, the resolver's and the knowledge component's, is tagged
sections only (`<section attribute="…">body</section>`), written with the shared
`prompt_structure.py` and checked by tests that no text lies outside them.

The knowledge component's generation prompts (`knowledge/generation/prompts.py`) have
their own sections and rules (`GenerationSection`, `GenerationRuleName`): a system
prompt of `<role>`, `<inputs>`, `<rules>` (content, roles, quantities, root,
answer_format, fragment_keys, references, types_and_fields, specific_type,
defined_type, enumerations, concepts, absent_fields, fixed_values, repair), `<types>`
(each type as JSON: name, kind_of, meaning, defined, requires, and its fields with what
each takes — a fragment type, an enum, a concept or a literal — whether it takes
several values and its meaning), `<structure_rules>` (JSON) and `<output>` (the answer's
JSON schema); a request of `<instruction>` and `<fixed_values>` (JSON); and a repair
adding `<rejected_answer>` and `<problems>` (JSON).

| Prompt | Sections |
|---|---|
| System | `<role>`, `<inputs>` (what each request section means), `<rules>` (one named rule per line, `- name: text`) |
| Request | `<source type="…">` per type (or `<description>`), `<request>` (the query in English; `Ellipsis` marks an open slot), `<task>` / `<situation>` / `<action>` (from context providers; `<action>` holds the action a resolution serves as its own `<request>`, `<slots>` and `<given>`), `<slots>` (JSON), `<given>` (JSON, values the query fixes), `<output>` (JSON schema of the answer) |
| Repair | The request, then `<rejected_answer>` and `<rejection kind="…">` (the failure's fields as JSON) |

A slot entry in `<slots>`:

```json
{"path": "CuttingAction.technique", "type": "CuttingTechnique",
 "meaning": "The cutting technique to use.",
 "accepts": [{"enum": ["SLICE", "SAW", "HALVING"]}], "default": "SAW"}
{"path": "CuttingAction.tool", "type": "Tool",
 "accepts": [{"type": "string", "holds": "handle"}],
 "candidates": {"count": 2, "kind": "…Tool", "listed": [
   {"handle": "CuttingKnife#1", "kind": "CuttingKnife", "held_as": ["…"]}]}}
```

Up to 10 candidates are listed (`LISTED_CANDIDATES`); more are counted, and the model
searches them with the world tools.

**The rules** (`RuleName`): `answer_format`, `element_keys`, `value_slot`,
`established_values` (only when open values are facts, `OpenValueMeaning.FACT`), `default`,
`union`, `kind_slot`, `referent_slot`, `finding_candidates`, `known_candidate`,
`meaning_holder` (the last three with world tools only), `no_fitting_candidate`,
`equally_fitting`, `unsatisfiable`, `conditions`, `given`, `candidates`, `why`,
`repair`, `text_answer` (with tools only), `knowledge` (only when the knowledge tools
are offered: read `knowledge_overview` first, follow what bears on the open slots, ground
the choices on it and name the fragments relied on in `why`).

**The answer** is one JSON array of at most N elements, each holding every slot path,
`"why"` (one sentence) and, when there is a source to name, `"grounds"`: what the
choices rest on, each as `{"world": "<handle>"}` (when candidates are named by world
handles) or `{"knowledge": "<fragment name>"}` (when the knowledge tools are offered).
Each ground is checked against its source (`grounds.py`: `WorldGrounds` by
`WorldAccess.is_live`, `KnowledgeGrounds` by `KnowledgeAccess.citation`); one naming
nothing is sent back for repair (`UnknownGround`). The backend keeps a `Justification`
per accepted instance in `justifications`: the instance, the `why`, and its `Ground`s,
each a source and a stable identifier (a handle, or the IRI citing a fragment).

| Slot | Answer |
|---|---|
| Value slot | An enum member's name, a JSON number/string/boolean, `null` where accepted |
| Referent slot | A handle string (`"Body#2"`) or an integer token; or `{"no_fitting_candidate": "<why>"}`; or `{"equally_fitting": [...], "why": "<why>"}` |
| Kind slot | The integer token of a listed kind |
| Whole request | `[{"unsatisfiable": "<why>"}]` when its conditions cannot all hold |

### 3.3 Resolver ↔ world injector (the world-access contract, version 1.0)

The resolver imports nothing of the world injector but `world/access.py` (a test checks
its imports). The contract has two parts.

**Direct calls** (`WorldAccess`, implemented by `WorldView`):

| Operation | Use |
|---|---|
| `handle_of(individual) -> str` | Naming listed candidates |
| `individual_of(handle)`, `is_live(handle)` | Decoding an answer's handle back to the live individual |
| `holders(handle) -> List[Holder]` | Saying what holds a listed candidate |
| `tools() -> List[Tool]` | The tools offered to the model |
| `version` | The contract version |

Shared names: `WorldToolName`, `HANDLE_SEPARATOR` (`Kind#n`), `Holder`,
`is_individual`, `identifier_of`, `text_description`.

**The tools** (asked by the model during a tool conversation). Each is a JSON file in
`world/tool_definitions/` with `name`, `description` (`purpose`, `returns`, `use_when`,
`avoid`), `input_schema` and `output_schema`; the value forms its answers share are in
`shared_definitions.json`. Inputs are validated before a tool runs; answers are checked
against the output schemas by tests; `output_schema` is not sent to the model.

| Tool | Input | Answer |
|---|---|---|
| `list_kinds` | `within`, `listing` (`instantiated` default, `present`, `all`) | Kinds with identifier, summary, individual counts |
| `describe_kind` | `kind` | Summary, ancestors, specialisations, field definitions |
| `find_individuals` | `kind`, `conditions` (`path`, `comparison`, `value` or `{"handle": …}`), `limit` (≤ 50), `path` | The search in English, total, counts per kind, handles (and what `path` reaches) |
| `describe_individual` | `handle`, `depth` | The individual as readable text, holders first |
| `holders_of` | `handle` | Holders with field and summary |
| `follow_path` | `handle`, `path` | The value reached, with its form |
| `list_predicates` | `about` | Predicates and value functions (krrood classes and `symbolic_function`s) with their arguments |
| `evaluate_predicate` | `predicate`, `arguments` | What it says |

Comparisons: `equal`, `not equal`, `less`, `less or equal`, `greater`,
`greater or equal`, `contains`. A condition is checked against the values its path
reaches before the search runs, so a wrong path or an inapplicable comparison is an
error answer rather than an empty result.

Values carry a `"form"`: `PlainValue`, `NumericValue`, `WorldStateDependentValue`
(variables and current numbers), `IndividualReference` (a handle), `NestedIndividual`,
`CollectionValue`, `UndescribedValue`.

**Errors** are JSON following `clients/tool_error_schema.json`:
`{"error": "<kind>", …fields, "message": "<one sentence>"}` — world errors
(`UnknownHandle` with the kind whose search gives such handles, `InapplicableComparison`
with the fields the path may continue to, `PredicateFailed` with the error a function of
the world model raised, …) and tool-box errors (`UnknownTool`,
`InputBreaksSchema`).

**Why not MCP yet.** The tools are shaped like MCP tools (name, description, input and
output schema, error flag). The resolver must stay in the world's process, because it
turns handles back into live objects; an MCP server would be an extra door for other
consumers and is a thin adapter over this contract when one is needed.

### 3.4 Clients ↔ language model

`AnthropicClient` sends the system prompt with a cache breakpoint and caches the
conversation as it grows; each tool turn's calls run through `ToolBox` and their
outcomes return as `tool_result` blocks. A turn that calls no tool ends the
conversation with its text as the answer.

### 3.5 Knowledge component ↔ the others

| Link | State |
|---|---|
| Knowledge component → clients | `LanguageModelClient.complete_structured(system, user, json_schema)`; no tools |
| Caller → knowledge component | `KnowledgeService.serve(query, instruction)` returns the record identifier of the hypothesis answering a request, generating it first when none is stored; `view_for(query, instruction)` returns its view. cram_viz's Hypothesis Graph and Designator tabs read it |
| cram_viz → the system | cram_viz shows the system at work: its predefined pouring designators (`cram_viz.resolution`), each an underspecified krrood match query shown as written, are resolved by a `Reasoner` over a semantic_digital_twin world built from the selected scene bundle, its robot annotated when a robot class is kept for it, with the world model's robot predicates loaded by name (`WORLD_MODEL_MODULES`), consulting cram_viz's knowledge store; the answer panel shows the chosen slots, the explanation and the grounds, and the consulted hypothesis is displayed with the resolution in the Hypothesis Graph, Ontology Graph and Designator tabs. Each run logs what it does to `events.jsonl` in its directory (`cram_viz.events`: the run starting and ending, each pour tried in the simulation, beside the events of §2.5, the knowledge component's model client wrapped as the knowledge component's). A chip copies its query into the search bar, Generate resolves the query there, and Execute performs the resolved action in coraplex's simulation from the run's own initial environment. Before a pouring resolution is accepted, `SimulatedPouring` (an `OutcomeCheck`) performs it in a process of its own and sends a pour whose mouth misses the target's opening back to the resolver with where it ended, up to three trials |
| Resolver → knowledge component | Through `knowledge/access.py` only (a test checks this): the knowledge request (§3.6) before every request to the model, serving or generating the hypothesis for the action's situation, and the knowledge-access contract (§3.7), whose tools the request's model reads that hypothesis with |
| Knowledge component ↔ world injector | None, by design: world facts reach the knowledge only through the resolver; the two keep separate sources |

### 3.6 Resolver → knowledge component: the knowledge request (designed and built 2026-10-12)

The resolver asks the knowledge component with a krrood query on the knowledge
vocabulary, never with an instruction:

```
resolver query  the(PouringAction)(…)  [+ optional task]
  ① mapping kept as data: PouringAction → aicor:Pouring, pouring.PouringPhaseStructure
  ② situation features the knowledge accepts, derived from the ontology:
       has_material_regime ∈ {Liquid, Granular, Cohesive}, has_termination_reason ∈ {…}
  ③ the resolver fills them from the world and the task, in a first short resolution of
       the(PouringPhaseStructure)(decomposes_task=aicor:Pouring,
                                  has_material_regime=..., has_termination_reason=...)
  ④ the knowledge request: that query with the features fixed
  ⑤ the knowledge component serves the stored hypothesis for (root class, fixed values,
     ontology versions), or generates, admits and stores it once
```

| Part | Design |
|---|---|
| Request | A krrood `Match` on the knowledge vocabulary: the action concept and the situation features as fixed values; an instruction is optional provenance only |
| Designator → knowledge | A mapping kept as data, by identifiers (strings), so `llm_reasoner` imports no designator package; later possibly in the ontology or on the action |
| Situation features | Derived from the ontology: the root class's functional properties whose values define its subclasses (such as `GranularPouringStructure ≡ PouringPhaseStructure ⊓ hasMaterialRegime value Granular`); no hand-written list |
| Filling them | The resolver, in a first resolution of the feature query with its world tools and the optional task, each choice with its reason; a feature left empty is unknown |
| Enforcing the variant | Already in place: closed-world acceptance classifies a structure by its feature values into the variants they define (such as `GranularPouringStructure`) and applies their restrictions; HermiT checks them open-world when configured; the generation prompt states each variant's definition and requirements |
| Store key | Root class, fixed values (features included) and ontology module versions; the instruction is no longer part of the key |
| Generation | On a miss, once per feature combination, then served from the store |
| Not in the request | The designator's slots and the task text; the resolver keeps them |
| Action perspectives | When the requester names perspectives (`KnowledgeRequester.perspectives`, none by default), the core's frame of the action (`DescribedAction.frame_query`) and each perspective on it (`PERSPECTIVE_ROOTS`, naming the stored core by its record with `StoredRoot`) are served too; the tools and grounds read every record, and the `frame_knowledge` rule is stated ([`knowledge_perspectives.md`](knowledge_perspectives.md) §11.6) |

**As built.** `KnowledgeRequester.consult(action, situation_backend)` in
`resolver/knowledge.py`: ① `DescribedActions.describing(identifier_of(action.type))`,
and without knowledge for the designator a consultation without a hypothesis; ②
`SituationFeatures.of(structure)`; ③ `DescribedAction.query({feature: ...})` resolved by
the backend given, one structure, each feature read back with
`SituationFeature.value_in`; ④ `DescribedAction.query(situation)` with the unknown
features left out; ⑤ `KnowledgeService.serve(request, instruction=<designator>)`. The
backend filling the features is `LanguageModelBackend.situation_backend_for(action)`:
the resolver's own model, tools and context, the action stated first in an `<action>`
section, `SourceDetail.SIGNATURE`, one candidate, `OpenValueMeaning.FACT` (a feature
nothing establishes is `null`, never the usual case) and no consultation of its own.
The features come from the world: `<action>` lists the action's open slots with their
candidates, written as the action's own resolution writes them, and the
`established_values` rule has the model read the world about the individuals the action
involves, the ones its `<given>` names and its slots' candidates, before answering. A
query binding an object and a query leaving it open to the only candidate therefore see
the same facts. The
resolver consults before every request to its model; a query leaving nothing open asks
no model and consults nothing.

**Knowledge needs no symbol graph.** Vocabulary classes are not krrood `Symbol`s, and
their object properties are managed by the knowledge component's own
`ObjectPropertyDescriptor`s (`ontologies/properties.py`), which apply the closed-world
entailments as fields are set and read: a sub-property's values are values of its
super-properties (`defines_component` holds every declared component), and a symmetric
property (`matched_with`) holds both ways. A hypothesis is therefore built, admitted and
read in any process, whatever krrood's symbol graph knows, including one that loaded a
world first (I1; the first R7 run failed on exactly that).

### 3.7 Resolver → knowledge component: the knowledge-access contract (version 1.3)

The resolver reads the knowledge component's own graph through `KnowledgeAccess`
(`hypotheses/knowledge_access.py`, exported by `access.py`), as it reads the world
through `WorldAccess`. `KnowledgeService` fulfils it.

| Operation | Answers |
|---|---|
| `serve(query, instruction="")` | The record identifier of the hypothesis for a request (§3.6) |
| `tools(*record_ids)` | The knowledge tools over those hypotheses and the ones they build on, read as one body of knowledge, named by `KnowledgeToolName` (several records since version 1.2) |
| `frame_parts_filled(record_id, frame_fields)` | The names of the parts of a stored core frame each field of a designator fills, given the designator's mapping (`described_actions.json` `"frame_fields"`); none from knowledge without frames (version 1.3) |
| `citation(record_id, name)` | The IRI citing a fragment of the hypothesis by the name the tools gave it, `None` when it has none (version 1.1) |
| `instances_of(kind)` | The stored fragments of a vocabulary class: an EQL domain |
| `is_vocabulary_class(kind)` | Whether a class's instances are knowledge rather than parts of the world |

The knowledge tools follow the world tools' conventions: one JSON definition each
(`hypotheses/tool_definitions/`, a description with purpose, returns, use_when and
avoid, an input schema and a declared answer shape sharing `shared_definitions.json`),
answers as JSON, and a question naming something the hypothesis lacks answered as an
error in the shared error shape. Fragments are named as their citations name them, those
of an action perspective hypothesis after its module (`action-core/source`), so names
stay apart across the hypotheses read.

| Tool | Input | Answer |
|---|---|---|
| `knowledge_overview` | none | The action concept's IRI; for a phase structure, the structure's class and meaning; the situation (feature → value name, unknown features absent); the phases in order, each with name, class and meaning, label, the subgoal it aims for, and its parameters and requirements, each summarised with its values (a linked fragment holding only values, such as a range, written in place) and the names of its other links; the roles; for an action core, its frame with the slots, objectives, consequences and parameter dimensions, and the perspectives read with it (version 1.2) |
| `find_fragments` | `kind`: a vocabulary class name | The fragments of that class and its subclasses, each with name, class and label |
| `describe_fragment` | `name`: a fragment name | Its class and meaning, label, a statement per ontology property it gives a value for (field, meaning, and linked fragment names or its value), and the fragments linking to it with their field |

## 4. Decisions that shape behaviour

| Decision | Consequence |
|---|---|
| The resolver resolves; it does not check executability | Preconditions and navigation belong to the plan |
| The query decides what fills a slot | A where-condition leaving one candidate decides it; the slot's type decides what kind of thing goes there; a task only chooses among candidates |
| Refusals only for open slots | `no_fitting_candidate` when no candidate is what the request asks for or can do what the slot is for |
| A chosen individual must be known | The model learns what a candidate is (holders or description) before choosing it |
| Several answers only when each is good | Elements are never varied only to differ; fewer are given when fewer are good |
| A tie between values is chosen and explained | The value is given and the tie stated in `why` |

## 5. Tests

`test/llm_reasoner_test` runs offline with mocked and scripted clients
(`python -m pytest test/llm_reasoner_test`). Mimic world classes live in
`world_mimics.py`; scripted answers in `resolver_cases/`; prompt sections are read back
as data with `prompt_sections.py`.

The kitchen cases live in the `experiments` package (`experiments/llm_reasoner_kitchen`:
the breakfast scene, the 28 resolver cases and 6 knowledge cases, each with what a
correct resolution satisfies, and a resolver as a CRAM process would run it), since they
need SDT and coraplex and `llm_reasoner`'s own tests must not. Their tests are in
`test/experiments_test/llm_reasoner_kitchen_test`: scripted twins replaying recorded
answers (`recordings/`) run with the experiments suite, and the live tests run only with
`LLM_REASONER_LIVE=1` and an API key (`LLM_REASONER_KNOWLEDGE_STORE` names a store whose
hypotheses later runs reuse). Live outcomes and costs are recorded in `status.md`.
