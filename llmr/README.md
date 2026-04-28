# llmr

**LLM-powered `GenerativeBackend` for KRROOD.** Resolves underspecified PyCRAM action `Match` expressions into executable plans using an LLM.

## Install

```bash
pip install ./llmr
```

## Quick start

### Natural language → executable plan

```python
from llmr import plan_from_instruction, sequential_plan_from_instruction
from llmr.reasoning.llm_provider import LLMProvider, make_llm

llm = make_llm(LLMProvider.OPENAI, model="gpt-4o")

# Single instruction
plan_from_instruction(
    "pick up the milk from the table",
    context=ctx,
    llm=llm,
    symbol_type=Body,
).perform()

# Multi-step (auto-decomposed)
for plan in sequential_plan_from_instruction(
    "go to the kitchen, pick up the milk, place it in the fridge",
    context=ctx,
    llm=llm,
    symbol_type=Body,
):
    plan.perform()
```

### Pre-built Match → resolved action or plan

```python
from krrood.entity_query_language.query.match import Match
from llmr import instance_from_match, plan_from_match

match = Match(PickUpAction)(object_designator=..., arm=..., grasp_description=...)

action = instance_from_match(
    match,
    llm=llm,
    instruction="pick up the milk",
    symbol_type=Body,
)
plan = plan_from_match(match, context=ctx, llm=llm, symbol_type=Body)
plan.perform()
```

### Add pluggable reasoners and a hypothesis graph

```python
from llmr import instance_from_match
from llmr.hypotheses import (
    FrameNetFamily,
    FrameNetGraphView,
    HypothesisGraph,
)
from llmr.reasoning.framenet_reasoner import FrameNetReasoner

graph = HypothesisGraph()
manager = FrameNetFamily.make_manager(graph)
view = FrameNetFamily.make_view(graph)

action = instance_from_match(
    match,
    llm=llm,
    instruction="pick up the milk from the table",
    symbol_type=Body,
    reasoners=[FrameNetReasoner(llm=llm)],
    hypothesis_graph_manager=manager,
)

print(action)
print(view.frames())  # Frame-level hypotheses
print(view.roles())   # Flattened role hypotheses
```

## How it works

```
NL instruction ─► infer_action_class ─► underspecified_match(action_cls)
                                              │
                                              ▼
                                        LLMBackend._evaluate
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
               fill_slots             EntityGrounding              enum / primitive
              (LLM prompt)             (SymbolGraph)                  coercion
                    │                         │                         │
                    └────────────► bind resolved values back ◄─────────┘
                                              │
                                              ▼
                                     construct_action → action
                                              │
                                              ▼
                                      execute_single → PlanNode
```

With pluggable reasoners and the hypothesis graph enabled:

```
underspecified Match + instruction
        │
        ▼
   LLMBackend._evaluate
        │
        ├─► slot filling + grounding
        ├─► reasoners annotate ActionAnnotationBundle sidecars
        │      └─► FrameNetReasoner → semantics.frames
        ├─► construct_action → action
        └─► ProjectionOrchestrator
               └─► FrameNetFamily / FrameNetProjector
                      └─► HypothesisGraph
                              ├─► FrameHypothesisNode
                              ├─► FrameRoleHypothesisNode
                              └─► support / grounding evidence nodes
```

## Hypothesis graph

`llmr.hypotheses` is an epistemic layer for LLM-generated interpretations.

- `SymbolGraph` models world entities and their relations.
- `HypothesisGraph` models reasoner-produced claims about instructions and actions.
- Hypothesis nodes and edges always carry provenance and epistemic metadata.
- Grounded claims are still hypotheses; grounding means "linked to structured system state", not "verified fact".

For the current FrameNet use case the graph contains:

- `InstructionNode`
- `ActionNode`
- `ReasonerRunNode`
- `FrameHypothesisNode`
- `FrameRoleHypothesisNode`
- `SlotBindingEvidenceNode`
- `SymbolGroundingEvidenceNode`

The current `FrameNetProjector` is conservative:

- it projects `ActionAnnotationBundle.frames`
- it flattens core and peripheral FrameNet roles into one role node each
- it marks a role as `SUPPORTED` only when the filler aligns with already resolved action slots
- it marks a role as `SYMBOL_GROUNDED` only when that aligned slot value is symbol-like

## End-to-end FrameNet example

See [examples/framenet_hypothesis_graph.py](examples/framenet_hypothesis_graph.py) for a complete setup using:

- `FrameNetFamily`
- `HypothesisGraph`
- `instance_from_match(...)`

The example also shows how to query the resulting graph for grounded `theme` roles.

## Package layout

| Module | Purpose |
|---|---|
| `backend.py` | `LLMBackend(GenerativeBackend)` — main evaluation pipeline |
| `factory.py` | `plan_from_instruction`, `sequential_plan_from_instruction`, `plan_from_match`, `instance_from_match` |
| `resolution/grounder.py` | `EntityGrounding` — entity lookup against `SymbolGraph` |
| `resolution/resolver.py` | Resolve LLM slot outputs by `FieldKind` |
| `bridge/introspect.py` | `ActionFieldIntrospector`, `FieldKind`, `ActionSpec` |
| `bridge/match_reader.py` | `MatchSnapshot` / `MatchField` — `snapshot_match`, `bind_slot_value`, `construct_action`, `underspecified_match` |
| `bridge/world_reader.py` | `render_world_context`, symbol/body helpers |
| `pycram/adapter.py` | `discover_action_classes`, `execute_single` — single PyCRAM boundary |
| `reasoning/slot_filler.py` | `infer_action_class`, `fill_slots` — LLM prompts from introspection |
| `reasoning/decomposer.py` | `TaskDecomposer` — compound NL → atomic steps with deps |
| `reasoning/llm_provider.py` | `make_llm`, `LLMProvider` |
| `reasoning/framenet_reasoner.py` | `FrameNetReasoner` — FrameNet sidecar annotation |
| `reasoning/flanagan_reasoner.py` | `FlanaganReasoner` — motion-phase sidecar annotation |
| `hypotheses/graph.py` | `HypothesisGraph` — epistemic graph of reasoner-produced claims |
| `hypotheses/projection.py` | projector contracts, registry, and orchestrator |
| `hypotheses/projectors/framenet/projector.py` | `FrameNetProjector` — FrameNet sidecar → graph projection |
| `schemas.py` | `EntityDescription`, `SlotValue`, `ActionClassificationResult`, `ActionAnnotationBundle`, FrameNet and Flanagan schemas |

## Field kinds

| Kind | Resolved by |
|---|---|
| `ENTITY` | `EntityGrounder` → `Symbol` instance in `SymbolGraph` |
| `POSE` | `EntityGrounder` + `.global_pose` |
| `ENUM` | String → `Enum` member coercion |
| `PRIMITIVE` | Direct coercion (`bool`, `int`, `float`, `str`) |
| `TYPE_REF` | `resolve_symbol_class` → `Symbol` subclass |
| `COMPLEX` | Recursed as a nested KRROOD `Match` leaf |

## Testing

```bash
pytest test/llmr_test --confcutdir=test/llmr_test
```

The suite uses `ScriptedLLM` (a deterministic `BaseChatModel`) — no API key, no network. Live-LLM smoke tests live in `test/llmr_test/live/` and activate only with `LLMR_LIVE_TESTS=1` and a valid `OPENAI_API_KEY`.

## Design invariants

- **Single krrood boundary.** All krrood access is funneled through `llmr.bridge.*`.
- **Single pycram boundary.** All pycram imports live in `llmr.pycram.*`.
- **No world-package imports.** World context is derived from `SymbolGraph`, not from a concrete world object.
- **One-way dependency.** `llmr → krrood`; no circular imports.
- **Raw reasoners, projected graph.** Reasoners write sidecars to `ActionAnnotationBundle`; projectors normalize those sidecars into `HypothesisGraph`.
- **Epistemic separation.** `HypothesisGraph` does not replace `SymbolGraph`; it stores interpretations, support, and grounding metadata.
