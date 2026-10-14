# llm_reasoner

A **language model generative backend** for krrood's entity query language.

Resolves underspecified `Match` queries by asking a language model, while depending on nothing but
`krrood` and the standard library. Knowledge about the world reaches it through injected seams
rather than imports, which is what lets it resolve a `coraplex` designator referring to a
`semantic_digital_twin` individual without importing either package.

```python
match = a(GraspDescription)(
    approach_direction=...,                                    # generated
    vertical_alignment=...,
    end_effector=variable(EndEffector, world.semantic_annotations),  # selected
    rotate_gripper=...,
    manipulation_offset=...,
)
match.where(match.variable.manipulation_offset > 0.0)

for grasp in language_model_backend_for_world(world, AnthropicClient()).evaluate(match):
    print(grasp)
```

---

## Where it sits

```
krrood  ←  llm_reasoner  ←  coraplex.querying.language_model  →  semantic_digital_twin
```

The world-aware seams live in `coraplex/querying/language_model.py`. Putting them here would make
every consumer of a language model backend depend on `semantic_digital_twin`, which is the
coupling this package exists to avoid.

---

## Getting started

```bash
uv sync
jupyter lab doc/language_model_backend.ipynb
```

The notebook runs end to end with no API key and no network: a `ScriptedClient` replays fixed
responses, which exercises reflection, prompt assembly, decoding, write-back and constraint
enforcement — most of the package. The real client goes in at the last cell.

For the same thing against a real world and a real `GraspDescription`, see
`coraplex/doc/language_model_backend.ipynb`.

```bash
pytest test/llm_reasoner_test          # CI-safe, no credentials
```

---

## What it is

| | |
|---|---|
| **What it knows** | Typed, path-named slots reflected from the query, the query's constraints in English, and whatever the context provider supplies. No ontology, no world model. |
| **What it enables** | Abductive parameter filling — plausible completions of a partial description, using generic knowledge the query does not contain. |
| **What it refuses** | Minting individuals. |
| **How it stays sound** | It does not. Where-conditions are stated in the prompt *and* applied afterwards as a filter. |

### Slot kinds

A model can produce the string `"the left gripper"`. It cannot produce an `EndEffector` with a
tool frame and a place in the kinematic structure. That distinction is the type system of this
package:

| kind | what is missing | what is allowed |
|---|---|---|
| `VALUE` | a self-identifying value | generation |
| `REFERENT` | an individual | selection from an extension |
| `SORTAL` | a kind, declared as `Type[T]` | selection from `T` and its subclasses |
| `BOUND` | nothing | context for the other slots |

`SORTAL` exists because choosing a *kind* is abduction over categories -- "something to
drink from" is a cup, a mug or a glass -- which a model does well, and unlike an individual a
kind can be named from the class hierarchy alone. Its extension comes from reflection, so no
world is needed.

krrood unwraps `Type[T]` to `T` when it builds a variable, so reflection consults the raw
annotation to tell a sortal slot apart from one asking for an individual of the same type.

`end_effector=...` raises `ReferentSlotHasNoDomain`. The fix is `variable(EndEffector)`, whose
extension comes from krrood's symbol graph, or `variable(EndEffector, domain)`.

---

## How it stays decoupled

| mechanism | source |
|---|---|
| schema — annotations and attribute docstrings | `assigned_variable._type_`, `_attribute_docstrings` |
| vocabulary — dotted access paths | `name_from_variable_access_path` |
| extension — individuals arrive as query data | `variable(T)` via the symbol graph |
| identity — individuals travel as integer tokens | this package |
| construction — the class is a callable handed in | `HasFactoryAndKwargs.construct_instance` |
| world knowledge — injected | `PromptContextProvider`, `CandidateRenderer` |

`SchemaOnlyContextProvider` adds nothing at all, which makes it both the default for environments
with no world and a useful uninformative baseline.

---

## Layout

```
src/llm_reasoner/
  backend.py     LanguageModelBackend(GenerativeBackend)
  reflection.py  SlotReflector: Match -> List[Slot]
  prompts.py     PromptBuilder: schema / constraints / situation
  context.py     PromptContextProvider, CandidateRenderer
  exceptions.py  retryable and terminal failures
  clients/       LanguageModelClient, ScriptedClient, AnthropicClient
examples/kitchen_domain.py   a world-free domain for the notebook
doc/language_model_backend.ipynb
```

### Call budget

One completion per `evaluate()`, plus at most `max_repairs`. All slots are asked about jointly,
because they are correlated and per-slot questions would discard that structure.

---

## Failures

| exception | kind | response |
|---|---|---|
| `ModelResponseUnparseable`, `UnknownSlotInResponse`, `MissingSlotInResponse`, `SlotValueRejected` | retryable | repair prompt |
| `RepairBudgetExhausted` | terminal | give up |
| `NoValidCandidates` | terminal | every candidate failed the filter |
| `ReferentSlotHasNoDomain` | terminal | rewrite the slot as `variable(T)` |
| `NoCandidateReferents` | terminal | the world contains no such thing — replan |
| `ReferentNotModelled` | terminal | nobody has looked yet — perceive |

The last two must not collapse into each other; they demand opposite responses.

---

## Provenance

Every exchange is retained on `backend.log` — prompt, raw response, model identity, attempt.
For any resolved object you can recover exactly how it was produced, which the probabilistic
backend cannot offer: its only justification for a candidate is a likelihood number with no
derivation trace.

---

## What reaches the model from outside the query

Two seams, both taking and returning ordinary Python values, so an adapter for a different world
is one class each and no change here.

`PromptContextProvider.context_for(expression)` receives the match, so a provider can inspect what
is already bound and describe accordingly rather than emitting a fixed dump. Each provider names
the section its text appears under, and `CompositeContextProvider` presents several as one:

```python
CompositeContextProvider(providers=[
    TaskContextProvider(goal="Lay the table for one."),
    SituationContextProvider(world=world, robot=robot),
])
```

`CandidateRenderer.render(candidate, expression)` receives it too, so a candidate can be described
*relative to the task*: what distinguishes two otherwise identical grippers is usually their
fitness for this particular query, not their intrinsic properties.

### Intent is not state

Some parameters are settled by the world -- which gripper is free, which face of the object is
turned toward the robot. Others are settled only by the goal: how thickly to slice, how long to
stir, which cutting technique to use. Identical worlds give different right answers depending on
what the caller wants, so no amount of situational detail settles them.

`TaskContextProvider` is the only route by which that knowledge arrives, and it renders under its
own heading. The system prompt tells the model to read *The task* as intent to be served and
*Situation* as fact -- collapsing the two invites it to treat a goal as something already true.

---

## How much of the target type goes in the prompt

`SourceDetail` controls it, defaulting to `SIGNATURE`:

| value | what is included |
|---|---|
| `NONE` | nothing; the slot table alone |
| `SIGNATURE` | the class docstring only *(default)* |
| `FULL` | `inspect.getsource` of the class |

`FULL` was the original default and turned out to be mostly noise: the slot table already
carries every field's name, type, docstring and default, so the full definition repeats all of
it and buries the repetition in method bodies describing how the type is *used* rather than how
to choose its values. On `coraplex.GraspDescription` that was roughly 2,800 tokens for about 60
tokens of signal.

Reach for `FULL` when behaviour defined in methods constrains which field combinations make
sense.

---

## Union-typed slots

krrood narrows unions away for the probabilistic path, because a random variable cannot span
`Union[int, str]`. This package keeps the branches instead: the prompt offers every branch, and
decoding tries them in declaration order. Choosing the branch from context is something a model
can do and a distribution cannot.

`Optional[T]` arrives already narrowed to `T`, so only genuine unions need this.

---

## krrood behaviour to be aware of

**Bind where-conditions through `match.variable`.**

```python
match.where(match.variable.manipulation_offset > 0.0)    # correct
match.where(match.expression.manipulation_offset > 0.0)  # silently matches nothing
```

`match.variable` is the subject being built; `match.expression` is the query wrapping it. The
second form compiles, verbalizes correctly, and returns an empty set — with the shipped
`EntityQueryLanguageGenerativeBackend` too, so it is not specific to this package.

**The symbol graph holds members weakly, by design.** An individual that nothing else references
is collected and drops out of `variable(T)`'s extension, which surfaces later as
`NoCandidateReferents`. Whoever creates individuals keeps them alive; in a robot system the world
does.

**`_attribute_docstrings` is private.** `reflection.py` depends on it; promoting it to public API
would remove the only private-symbol dependency here.

**krrood verbalization renders expressions, not individuals.** There is no path from a live world
entity to English, which is why `CandidateRenderer` exists as a separate seam rather than
delegating to the verbalizer.

---

## Not implemented yet

**Referents with no extension.** Phase 2 is describe, ground, select: the model proposes a sortal
description, the description is compiled into EQL conditions and evaluated against the world by a
selective backend, and only the survivors are offered for selection.

**Composition.** A `ParameterResolver` seam of shape `Match -> Match` would let this backend
resolve the referent and hand the geometry to `ProbabilisticBackend`, which treats the chosen
object as conditioning evidence.

**Ambiguity.** When several candidates are equally good and the consequences differ, a coin flip is
the wrong answer. `krrood.ripple_down_rules.experts.Expert` is the existing escalation path.