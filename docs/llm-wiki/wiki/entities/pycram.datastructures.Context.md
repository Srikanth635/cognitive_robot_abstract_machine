---
id: pycram.datastructures.Context
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/datastructures/dataclasses.py
    lines: [27, 96]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.plans.PlanEntity
  - sdt.world.World
  - sdt.robots.abstract_robot.AbstractRobot
  - krrood.entity_query_language.backends.QueryBackend
used_by:
  - bridge.pycram-sdt
  - concept.krrood-eql
  - pycram.plans.Plan
  - pycram.plans.Designator
  - pycram.plans.factories
  - pycram.robot_plans.ActionDescription
  - pycram.robot_plans.actions.core.container
  - pycram.robot_plans.actions.core.robot_body
status: stable
tags: [context, runtime, plan-entity, query-backend]
last_ingest: 2026-05-18
---

_Runtime configuration object attached to a [[pycram.plans.Plan]]; supplies the world, robot, ROS node, and condition-evaluation policy for every action executed under that plan._

## Purpose

`Context` is the "environment" thread that passes implicitly through an entire plan execution. It holds the `World` reference that action nodes mutate, the `AbstractRobot` annotation that motion designators resolve arms/grippers from, and flags that let test scenarios skip pre/post-condition evaluation. Because it extends `PlanEntity`, `Plan.add_plan_entity(context)` wires it into the plan graph automatically.

## When to use

- **Creating a plan execution environment:** construct a `Context` (or use `Context.from_world`) before building a `Plan`, then attach it.
- **Accessing the robot or world from inside a designator:** `designator.context.world` / `designator.context.robot`.
- **Disabling condition evaluation in tests:** set `evaluate_conditions=False`.
- **Not for:** storing per-node execution results — use [[pycram.datastructures.ExecutionData]] instead.

## Construction / dependencies

```python
ctx = Context(
    world=world,          # sdt.world.World
    robot=robot,          # sdt.robots.abstract_robot.AbstractRobot
    ros_node=None,        # Optional rclpy.node.Node
    evaluate_conditions=True,
    query_backend=EntityQueryLanguageBackend(),
)
# convenience: derive robot from world automatically
ctx = Context.from_world(world, plan=my_plan)
```

`from_world` calls `world.get_semantic_annotations_by_type(AbstractRobot)[0]` to discover the robot and optionally calls `plan.add_plan_entity(result)`.

## Key attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `world` | `World` | SDT kinematic world instance |
| `robot` | `AbstractRobot` | Semantic robot annotation |
| `ros_node` | `rclpy.node.Node \| None` | ROS 2 node; required when `debug=True` |
| `evaluate_conditions` | `bool` | If `False`, skip pre/post-condition checks on all actions |
| `query_backend` | `QueryBackend` | EQL backend used for underspecified action resolution |
| `_debug` | `bool` | Activates DEBUG logging; requires `ros_node` to be set |

## Related

**Uses:** [[sdt.world.World]], [[sdt.robots.abstract_robot.AbstractRobot]], [[pycram.plans.PlanEntity]], [[krrood.entity_query_language.backends.QueryBackend]]

**Used by:** [[pycram.plans.Plan]], [[pycram.plans.Designator]], [[pycram.robot_plans.ActionDescription]], [[pycram.robot_plans.actions.core.container]], [[pycram.robot_plans.actions.core.robot_body]]

## Provenance

- `pycram/src/pycram/datastructures/dataclasses.py` lines 27–96 (commit `0528d8cf3`) — full `Context` class definition.
