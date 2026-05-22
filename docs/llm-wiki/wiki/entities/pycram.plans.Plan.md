---
id: pycram.plans.Plan
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/plans/plan.py
    lines: [46, 388]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.plans.PlanEntity
  - pycram.plans.PlanNode
  - pycram.plans.DesignatorNode
  - pycram.plans.Designator
  - pycram.plans.plan_callbacks.PlanCallback
  - pycram.datastructures.Context
  - concept.designator
used_by:
  - pycram.plans.PlanEntity
  - pycram.plans.PlanNode
  - pycram.plans.factories.make_node
  - pycram.plans.factories
  - pycram.plans.Designator
  - pycram.robot_plans.ActionDescription
status: stable
tags: [plan, graph, executor, container]
last_ingest: 2026-05-18
---

_The executable plan container: a `rustworkx` directed graph of [[pycram.plans.PlanNode|PlanNodes]] traversed in depth-first order, plus a `Context` that supplies world/robot/configuration._

## Purpose

`Plan` is the **container** that holds a graph of plan nodes and drives their
execution. It is a dataclass with a `rustworkx.PyDiGraph[PlanNode]` as its backing
store, a `Context` (which itself is a [[pycram.plans.PlanEntity]]) that supplies the
runtime world/robot, and an optional `initial_world` snapshot (deepcopy taken before
the first `perform`).

The plan is expected to be a **tree** (verified by `validate()`: `edges == nodes - 1`
and exactly one root). Nodes are added bare via `add_node` (no parent) and connected
via `add_edge(source, target)`, which assigns `target.layer_index` to preserve
left-to-right child order (rustworkx doesn't preserve order natively).

## When to use / read

- Read when building a custom plan construction flow, debugging plan structure, or
  understanding execution semantics (DFS via the root's `perform`).
- Don't construct directly in user-facing plan code — use the combinators in
  [[pycram.plans.factories.make_node|`pycram.plans.factories`]] (Phase 3 ingest).
- Subclassing is not idiomatic; extend behavior via `PlanCallback`s instead
  (stub: [[pycram.plans.plan_callbacks.PlanCallback]]).

## Construction

```python
plan = Plan(context=some_context)        # __post_init__ binds context.plan = plan
plan.add_node(root_node)
plan.add_edge(root_node, child_node)
plan.validate()                          # raises ValueError if not a tree
plan.perform()                           # deepcopies world → initial_world, runs root
```

## Key attributes

| Name | Type | Notes |
|---|---|---|
| `context` | `Optional[Context]` | The runtime context. Bound to this plan in `__post_init__` via `add_plan_entity`. |
| `initial_world` | `Optional[World]` | A deepcopy of the world snapshotted at the start of `perform()`. Used by `replay()` / `re_perform()`. |
| `node_callbacks` | `List[PlanCallback]` | Called on node start/end. |
| `plan_graph` | `rx.PyDiGraph[PlanNode]` | The graph itself; `multigraph=False`. `init=False`. |

## Key methods (selection)

| Name | Returns | Role |
|---|---|---|
| `validate()` | None | Asserts the plan is a tree. |
| `root` *(property)* | `PlanNode` | The unique node with no parent. |
| `world` / `robot` *(properties)* | `World` / `AbstractRobot` | Delegated to `context.world` / `context.robot`. |
| `nodes` / `all_nodes` *(properties)* | `List[PlanNode]` | DFS from root / every node in the graph (including orphans). |
| `actions` *(property)* | `List[ActionNode]` | Filtered DFS for `ActionNode` instances. |
| `layers` *(property)* | `List[List[PlanNode]]` | BFS layers from the root, each layer sorted by `layer_index`. |
| `add_plan_entity(e)` / `remove_plan_entity(e)` | None | Set/null `e.plan` (works for PlanNode and Context). |
| `add_node(node)` | None | Inserts a node into the graph and binds `node.plan = self`. |
| `add_edge(src, tgt, target_index=None)` | None | Connects two nodes; assigns `tgt.layer_index`; shifts siblings right when inserting. |
| `add_edges_from(edges)` / `add_nodes_from(nodes)` | None | Bulk variants. |
| `insert_below(insert_node, insert_below)` | None | `add_edge(insert_below, insert_node)` shorthand. |
| `merge_nodes(node1, node2)` | None | Reparents `node2`'s children under `node1` and removes `node2`. |
| `remove_node(node)` | None | Removes from graph; nulls `index`, `layer_index`, `plan`, `world`. |
| `perform()` | `Any` | `initial_world = deepcopy(self.world)`; `return self.root.perform()`. |
| `re_perform()` | None | Re-runs every leaf descendant of the root. |
| `_migrate_nodes_from_plan(other)` | `PlanNode` | Steals every node and edge from `other`, returns the migrated root. Used by `mount_subplan`. |
| `get_nodes_by_designator_type(*types)` | `List[DesignatorNode]` | Filter `DesignatorNode`s by the type of their wrapped designator. |
| `simplify()` | None | Walks BFS-reversed and calls each node's `simplify()` (mostly used by `pycram.language` combinators). |
| `bfs_layout(scale, align)` / `plot_plan_structure(...)` / `prepare_for_replay()` / `replay()` | varied | Visualization and replay helpers (not core to execution). |
| `__repr__` | str | `Plan({nodes…})` style. |

## Related

- Container of: [[pycram.plans.PlanNode]] (and its concrete subclasses).
- Owns: [[pycram.datastructures.Context]] (stub) as a `PlanEntity`.
- Cross-package via `context`: `semantic_digital_twin.world.World`,
  `semantic_digital_twin.robots.abstract_robot.AbstractRobot` — to be bridged in
  Phase 5.
- Constructed indirectly by: [[pycram.plans.factories.make_node]] and its sibling
  combinators (Phase 3 ingest).
- Concept: [[concept.designator]] (designators are what plan nodes wrap).

## Open questions

- `remove_node` sets `node_for_removal.world = None` — but `PlanNode` does not
  declare a `world` attribute (only the `Designator` properties delegate through
  `plan_node.plan.world`). Either there is dynamic state being attached elsewhere,
  or this is a latent bug. Worth checking on the Phase 2 ingest.
- `Plan.world`/`Plan.robot` will raise `AttributeError` if `self.context is None`
  (no guard). Either contexts are mandatory in practice, or this is undocumented
  precondition.

## Provenance

- `pycram/src/pycram/plans/plan.py:46-388` at commit `0528d8cf3` — full class
  (388-line file; the body extends to the end with plotting/replay helpers).
- `plan.py:75-77` — `__post_init__` binds context to the plan.
