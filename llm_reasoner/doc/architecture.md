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
  form, krrood objects that EQL queries in the closed world, and a reasoning form, RDF
  that HermiT checks in the open world; both are generated or derived from the
  ontology, never written by hand.

## 2. Components

```mermaid
flowchart LR
    caller["Caller<br/>(plan, cram_viz, test)"] -- "krrood query<br/>+ optional task" --> resolver
    subgraph llm_reasoner
        resolver["Resolver<br/>resolver/"]
        world["World injector<br/>world/"]
        knowledge["Knowledge component<br/>ontologies/ … hypotheses/"]
        clients["Clients<br/>clients/"]
    end
    resolver -- "WorldAccess<br/>(handles, holders, tools)" --> world
    resolver -- "prompts / tool loop" --> clients
    knowledge -- "structured completion" --> clients
    clients -- "API" --> llm["Language model"]
    world -- "reads" --> graph["krrood SymbolGraph<br/>(the live world)"]
    knowledge -. "link deferred" .- resolver
```

| Component | Package | Role in one sentence |
|---|---|---|
| World injector | `world/` | Answers questions about the live world, using krrood alone |
| Resolver | `resolver/` | Turns an underspecified krrood query into a resolved instance, choosing what the query leaves open |
| Knowledge component | `ontologies/`, `ontology_translation/`, `acceptance/`, `generation/`, `admission/`, `hypotheses/` | Generates and stores ontology-backed, world-independent action knowledge |
| Clients | `clients/` | Talk to language models, and define the vendor-neutral tool interface |

### 2.1 World injector (`world/`)

**Role.** The world's data and state for any component that needs it: which kinds of
thing exist, which individuals, what they are, what holds them, what their fields reach,
and what the world model's own predicates say. It is useful to the resolver, to the
knowledge component and to any later component.

**Does not:** change the world, plan, or know any world model by name.

**Input.** The live symbol graph (`sources.py`, `SymbolGraphSource`; every loaded
`Symbol` class outside krrood is a kind, every live symbol an individual).

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
| `predicates.py` | Listing and evaluating krrood `Predicate` / `SymbolicFunction` classes |
| `tools.py`, `tool_definitions/` | The eight questions as tools for a model (§3.3) |
| `encoding.py`, `exceptions.py` | Answers as JSON; `WorldQuestionError` and its kinds |

**Errors.** A question naming something the world lacks raises a `WorldQuestionError`
(`UnknownKind`, `AmbiguousKind`, `UnknownHandle`, `ExpiredHandle`, `UnknownField`,
`InapplicableComparison`, `NotAPredicate`, `UnknownArgument`, `MissingArgument`). As a
tool, it becomes an error answer the model can correct, never a crash.

**Limits.** It reads dataclass fields and krrood predicate classes. Values a world model
computes in properties or plain `@symbolic_function`s (poses, reachability, contact) are
not reachable yet (see `status.md`).

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
2. **Build the prompts** (`prompts.py`, `prompt_format.py`; §3.2).
3. **Ask** through a `Completion` (`completion.py`): `PlainCompletion` (one completion,
   candidates listed by integer token) or `ToolCompletion` (a conversation in which the
   model may call the world tools, candidates named by handle).
4. **Decode** the JSON answer: handles back to live individuals through `WorldAccess`,
   enum names to members, `null` where allowed; verdicts become the outcomes above.
5. **Construct and check**: values are written into the query, krrood constructs the
   instances, and the query's `where`-conditions are checked.
6. **Repair once** (`max_repairs = 1`) when an answer is unusable or breaks a
   condition: a fresh conversation with the rejected answer and the rejection as data.

| Module | Responsibility |
|---|---|
| `backend.py` | Orchestration: ask, decode, construct, validate, repair; `log`, `conversations`, `rules` |
| `reflection.py` | Slots of a query: kind, type, docstring, default, domain |
| `prompts.py` | System prompt, request, repair; the general rules |
| `prompt_format.py` | The prompt structure: sections, rules, data keys |
| `referents.py` | How candidates are named and answered: `IntegerTokens`, `WorldHandles`; verdict rules |
| `context.py` | Context providers (task, situation, composite) and candidate renderers |
| `completion.py` | Plain and tool-using completion; the text-answer rule |
| `exceptions.py` | Retryable failures and outcomes |

**Settings** (`LanguageModelBackend`): `number_of_candidates` (3 unless the query sets a
limit), `max_repairs` (1), `source_detail` (`FULL`: the source of the queried type and
of its parents declared in the same module; `SIGNATURE`: docstrings; `NONE`),
`context_provider` (none by default), `candidate_renderer` (structural description).

### 2.3 Knowledge component (`ontologies/` … `hypotheses/`)

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
| An instruction | Text naming what the knowledge is for, such as "pour milk from the carton into the bowl"; it keys the stored hypothesis |
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
| 0 Vocabulary | `ontologies/`, `ontology_translation/` | The Turtle modules (`core`, `object-dynamics`, `prehension`, `pouring`, `gaze`) are the source of truth; Python vocabulary modules are generated from them with krrood's code generator (`python -m llm_reasoner.ontology_translation`) and checked in; a test fails when they differ from the ontology. Every class carries its IRI and its OWL restrictions |
| 1 Generation | `generation/` | `HypothesisBackend`, a krrood `GenerativeBackend` like the resolver. `GenerationSchema` reads every vocabulary class reachable from the queried type; the model answers with a structured completion (`complete_structured`) shaped by that schema; `FragmentDecoder` turns the answer into vocabulary instances |
| 2 Acceptance | `acceptance/` | Closed-world checks — what is missing is an error, since a model tends to generate too little: `restrictions_hold` (each fragment meets its classes' OWL restrictions), `ranges_hold` (every linked value has the type its property allows), `shapes_hold` (SHACL shapes for rules across fragments, such as transitions connecting all phases), `components_declared` (everything a structure reaches, it declares), plus the values the query fixes. Violations are errors or warnings; warnings are kept with the hypothesis |
| 3 Admission | `admission/` | Optionally (`consistency=HermitCheck(...)`), the hypothesis is written as RDF and checked in the open world against the ontology, CRAM-AICOR and DUL with HermiT; `ContradictionLocaliser` names the fragments that together contradict them. Concept references are checked against the linked ontologies (`linked_terms`) |
| Repair | `generation/` | Errors from stages 1–3 are shown to the model, which answers with a *patch* (fragments added or changed, identifiers removed) applied to the answer with the fewest problems so far (`max_repairs = 2`) |
| 4 Store | `admission/` | `HypothesisRecord` (instruction, root type, fixed values, client, every exchange, warnings, consistency verdict, ontology module versions) and the hypothesis as Turtle, one directory per hypothesis in `HypothesisStore`. A request is answered from the store when a record has the same instruction, root type and fixed values under the current module versions; otherwise it is generated |
| 5 Reading | `hypotheses/` | `KnowledgeService.view_for(instruction, query)` returns the stored or newly generated hypothesis as a `HypothesisView`, which answers questions crossing fragments: `phases`, `next_phases`, `requirements_of`, `world_assumptions`, `gaze_fixations_for`, `force_couplings_in`, `failure_events`, `corrections_at`, `force_dynamics`. Every fragment is cited by the IRI it is stored under (`Citation`). `HypothesisGraphProjector` projects a hypothesis into the graph cram_viz shows |

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
Fragments are created first and linked second; a malformed answer is not an exception
but a list of problems shown to the model.

**Components inside, by responsibility.**

| Module | Responsibility |
|---|---|
| `ontologies/__init__.py`, `linked.py`, `terms.py` | The ontology files, the linked ontologies (CRAM-AICOR, DUL) and concept references |
| `ontology_translation/reading.py`, `declarations.py`, `writing.py` | Ontology → declarations → Python modules |
| `generation/schema.py`, `prompts.py`, `wire.py`, `backend.py` | Output shape, prompts, decoding and merging patches, the generation loop |
| `acceptance/rules.py`, `restrictions.py`, `derived.py`, `shapes.py`, `core.py`, `suite.py`, `vocabulary.py` | Rules and violations, closed-world class membership, the microworld's suite |
| `admission/rdf.py`, `consistency.py`, `hermit.py`, `record.py`, `store.py`, `citation.py` | RDF in and out, consistency, records, the store, citations |
| `hypotheses/service.py`, `view.py`, `graph.py` | The service, cross-fragment questions, the graph projection |

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
| Queryable with krrood's EQL | Vocabulary classes are krrood `Symbol` dataclasses (`DescriptionEntity`), so loaded knowledge is first-class krrood objects that EQL queries directly |
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
| `AnthropicClient` | Both; a written-out tool loop (`max_tool_turns = 10`), prompt caching of the system prompt and of the growing conversation |
| `OpenAIClient` | `complete` only |
| `ScriptedClient` | Replays recorded answers and tool requests, for tests |
| `Tool`, `ToolDefinition`, `ToolDescription`, `ToolBox`, `ToolOutcome` | The tool interface (§3.3) |

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

Every prompt is tagged sections only (`<section attribute="…">body</section>`), checked
by a test that no text lies outside them.

| Prompt | Sections |
|---|---|
| System | `<role>`, `<inputs>` (what each request section means), `<rules>` (one named rule per line, `- name: text`) |
| Request | `<source type="…">` per type (or `<description>`), `<request>` (the query in English; `Ellipsis` marks an open slot), `<task>` / `<situation>` (from context providers), `<slots>` (JSON), `<given>` (JSON, values the query fixes), `<output>` (JSON schema of the answer) |
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

**The rules** (`RuleName`): `answer_format`, `element_keys`, `value_slot`, `default`,
`union`, `kind_slot`, `referent_slot`, `finding_candidates`, `known_candidate`,
`meaning_holder` (the last three with world tools only), `no_fitting_candidate`,
`equally_fitting`, `unsatisfiable`, `conditions`, `given`, `candidates`, `why`,
`repair`, `text_answer` (with tools only).

**The answer** is one JSON array of at most N elements, each holding every slot path
and `"why"` (one sentence, logged, never decoded):

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
| `list_predicates` | `about` | Predicates and value functions with their arguments |
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
with the fields the path may continue to, …) and tool-box errors (`UnknownTool`,
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
| Caller → knowledge component | `KnowledgeService.view_for(instruction, query)`; cram_viz's Hypothesis Graph and Designator tabs read it |
| Knowledge component ↔ resolver | Not connected; to be knowledge tools through a `KnowledgeAccess` contract (§2.3) |
| Knowledge component ↔ world injector | None, by design: world facts reach the knowledge only through the resolver; the two keep separate sources |

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
as data with `prompt_sections.py`. Live runs against a model are made only with the
user's approval and are recorded in `status.md`.
