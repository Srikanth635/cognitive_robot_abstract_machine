# agentic_llmr, KRROOD, SymbolGraph, and World Discussion

Date: 2026-05-26

This note captures the main conclusions from the discussion about `agentic_llmr`,
its relationship to KRROOD, and why it currently uses an explicit active world
context in addition to KRROOD's `SymbolGraph`.

## agentic_llmr Purpose and Architecture

`agentic_llmr` is an experimental agentic action-resolution backend for robot
tasks. It subclasses KRROOD's `GenerativeBackend` through:

- `agentic_llmr/src/agentic_llmr/backend.py`
- `AgenticLLMBackend`

Its purpose is to take either:

- a raw natural-language command, or
- a KRROOD `Match` expression,

and resolve it into executable action parameters or action instances using an
LLM-driven ReAct loop.

The high-level structure is:

```text
AgenticLLMBackend
  -> ReActAgent / Orchestrator
      -> SceneQueryAgent as tool
          -> SDT scene tools
      -> KinematicsAgent as tool
          -> reachability / grasp / robot-state tools
      -> PlanningAgent as tool
          -> PyCRAM schema / simulation tools
```

The package uses LangGraph's `create_react_agent()` at two levels:

- the top-level orchestrator,
- the specialist sub-agents.

The agentic behavior is therefore hierarchical and tool-driven, not a fully
hand-written planner loop.

## Main Runtime Flow

1. A world is loaded externally.
2. `set_active_world(world, robot_view)` stores the active runtime context.
3. `AgenticLLMBackend(llm=...)` creates the orchestrator.
4. `_evaluate()` receives a raw instruction or KRROOD `Match`.
5. If a `Match` is provided, `snapshot_match()` extracts fixed and free fields.
6. The action schema/context is rendered for the LLM.
7. The ReAct orchestrator calls specialist tools.
8. The final JSON designator is parsed.
9. Values are hydrated back into KRROOD/PyCRAM objects.
10. The resolved action instance is yielded.

## KRROOD SymbolGraph and World Access

KRROOD itself is not directly world-aware.

Its central runtime registry is the singleton `SymbolGraph`:

- `krrood/src/krrood/symbol_graph/symbol_graph.py`

`SymbolGraph` stores weak references to instances of `Symbol` subclasses.
It tracks symbolic objects and relations, but it does not store a current SDT
world, a simulator state, or a physical root frame.

KRROOD's `variable(type_, domain=None)` can infer domains from `SymbolGraph`
when `type_` is a `Symbol` subclass:

```python
elif domain is None and issubclass(type_, Symbol):
    domain = SymbolGraph().get_instances_of_type(type_)
```

So KRROOD can access symbolic instances such as SDT `Body`,
`SemanticAnnotation`, robot annotations, etc., because those inherit from
`Symbol`.

However, KRROOD does not itself provide:

- a current world handle,
- world root frame,
- FK computation,
- collision checking,
- simulation mutation context,
- current robot view.

## Can World Be Reached Through a Symbol?

Yes, for SDT symbols that carry a `_world` backreference.

For example:

```python
world = body._world
root = world.root
```

or:

```python
world = annotation._world
root = world.root
```

This works if the symbol is known to belong to the intended current world.

The problem is selecting the correct symbol from a global `SymbolGraph` in a
long-running process.

## Multiple-World Issue

`SymbolGraph` is a process-global singleton. It can contain symbols from
multiple loaded worlds in the same Python process, especially in notebooks,
tests, and demos.

Example failure pattern:

1. Load world A.
2. `SymbolGraph` stores bodies and annotations from world A.
3. Load world B.
4. `SymbolGraph` also stores bodies and annotations from world B.
5. Agent asks for "milk".
6. A naive lookup might return milk from world A.
7. Tool computes FK using world B:

```python
world_B.compute_forward_kinematics_np(world_B.root, milk_from_world_A)
```

That is inconsistent and can fail or produce wrong reasoning.

`agentic_llmr` avoids this by storing an explicit active world and filtering:

```python
if active_world is not None and getattr(inst, "_world", None) is not active_world:
    continue
```

So it still uses `SymbolGraph`, but it only considers symbols belonging to the
active world.

## Does SymbolGraph Have a Root?

No, not in the SDT-world sense.

`SymbolGraph` has graph nodes and relation edges, but no canonical physical
root frame.

SDT `World` has:

```python
world.root
```

That root is the kinematic root frame used for operations such as:

```python
world.compute_forward_kinematics_np(world.root, body)
```

`SymbolGraph` cannot replace this because it is an object/relation registry,
not a coordinate-frame tree.

## What agentic_llmr Uses From World That SymbolGraph Does Not Provide

`agentic_llmr` can get symbolic objects from `SymbolGraph`, but it needs world
services for live robot reasoning.

World-dependent capabilities include:

- forward kinematics:

```python
world.compute_forward_kinematics_np(world.root, body)
```

- collision checks:

```python
world.collision_manager.compute_collisions()
```

- simulation / mutation context:

```python
with world.modify_world():
    action_instance.execute()
```

- world state and joint state access,
- kinematic tree traversal,
- world entity lookup by name,
- robot view and arm/manipulator state,
- current-world filtering of SymbolGraph symbols.

Some of these can be reached through `symbol._world`, but they are still world
operations. `SymbolGraph` itself does not provide them.

## Does KRROOD's ProbabilisticBackend Access World?

No.

`ProbabilisticBackend` is implemented in:

- `krrood/src/krrood/entity_query_language/backends.py`

It operates on a KRROOD `Match`, extracts variables using
`UnderspecifiedParameters`, samples from a probabilistic model, writes sampled
values back into mapped variables, and constructs instances.

It does not call:

- `World`,
- `_world`,
- `SymbolGraph()`.

It may inherit multiple-world problems only if the input `Match` already used
SymbolGraph-derived domains, for example:

```python
variable(Body, domain=None)
```

But the probabilistic backend itself is match/model based and does not perform
world discovery.

## Design Concern

The main architectural concern is that `agentic_llmr` subclasses KRROOD's
`GenerativeBackend`, but currently relies on a global active-world manager.

This makes it feel less like a pure KRROOD backend and more like a robotics
runtime extension.

The better boundary would be dependency injection:

```python
AgenticLLMBackend(llm=llm, scene_context=context)
```

where `scene_context` is an adapter/protocol that exposes scene operations:

```python
class SceneContext:
    def symbols(...): ...
    def world_for(symbol): ...
    def get_pose(body): ...
    def get_robot(): ...
```

Then an SDT implementation could use:

```text
SymbolGraph + World + robot_view
```

This keeps KRROOD generic while making `agentic_llmr`'s domain-specific runtime
requirements explicit.

## Current Accurate Contract

`agentic_llmr` does not formally type its world as
`semantic_digital_twin.world.World`.

In `world_manager.py`, the world is stored as `Any`.

However, the tools structurally require an SDT-compatible world object. The
object passed to `set_active_world(world, robot_view)` must behave like an SDT
world and provide methods/fields such as:

```python
world.root
world.bodies
world.semantic_annotations
world.compute_forward_kinematics_np(...)
world.compute_forward_kinematics(...)
world.modify_world()
world.collision_manager
world.get_body_by_name(...)
```

Likewise, `robot_view` must behave like an SDT robot annotation:

```python
robot_view.arms
robot_view.left_arm
robot_view.right_arm
arm.root
arm.manipulator
manipulator.tool_frame
```

So the dependency is structural / duck-typed, not nominal:

```text
Not: isinstance(world, World)
But: world behaves like an SDT World
```

## Final Position

`agentic_llmr` should probably not make KRROOD itself world-aware.

But `agentic_llmr` legitimately needs access to live scene/runtime services for
robot action grounding. The clean design is to make that dependency explicit via
an injected context or adapter, rather than hiding it behind a global active
world.

