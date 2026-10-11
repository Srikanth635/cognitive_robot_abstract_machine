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

**Role.** Generate and keep description-level knowledge about kinds of action — what an
action normally requires, its phases, forces, failures and corrections — written in the
terms of the predictive-manipulation ontology and independent of any world or episode.

**Input.** An instruction and a krrood query over the ontology's generated classes.
**Output.** A hypothesis admitted only if it meets the acceptance rules (EQL and SHACL)
and, optionally, is consistent under HermiT; stored and read through `HypothesisView`.

| Package | Responsibility |
|---|---|
| `ontologies/` | The ontology modules (Turtle) and the Python vocabulary generated from them |
| `ontology_translation/` | Generating that vocabulary with krrood's code generator |
| `acceptance/` | Closed-world acceptance rules and shapes |
| `generation/` | `HypothesisBackend`: a model answers a query; answers breaking the rules are repaired |
| `admission/` | RDF, HermiT consistency (`HermitCheck`), records, `HypothesisStore` |
| `hypotheses/` | `KnowledgeService`, `HypothesisView`, the graph projection |

How it is used is shown in the [README](../README.md). **Its link to the resolver and
the world injector is deferred** (see `status.md`).

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

Not connected yet. The knowledge component uses `clients` for structured completions;
how the resolver will consult its knowledge, and how it will read the world through
`WorldAccess`, is open (see `status.md`).

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
