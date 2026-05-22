---
id: sdt.datastructures.joint_state.JointState
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/datastructures/joint_state.py
    lines: [1, 159]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.connections
used_by:
  - sdt.robots.abstract_robot.AbstractRobot
status: stable
tags: [sdt, joint-state, arm, gripper, torso, named-config, robot]
last_ingest: 2026-05-17
---

_Named target configuration: a list of `ActiveConnection1DOF` → target float value pairs, optionally typed (`JointStateType`) and named (`PrefixedName`)._

## Purpose

`JointState` represents a desired joint configuration for a robot subset — e.g. an arm pose, a gripper open/close state, or a torso height. It is distinct from `WorldState`: `JointState` is a *goal* specification, not the live state bus. `is_achieved()` checks whether the current world positions match the targets within 1 cm / 0.01 rad tolerance.

Three type aliases exist for readability: `GripperState = JointState`, `ArmState = JointState`, `TorsoState = JointState`.

## When to use

- Use as the target for `SetGripperAction`, `MoveTorsoAction`, `ParkArmsAction`.
- Construct from a `{connection_name: float}` dict via `from_str_dict(mapping, world)`.
- Construct from a `{connection: float}` dict via `from_mapping(mapping)`.

## Key attributes

| Attribute | Type | Notes |
|---|---|---|
| `connections` | `List[ActiveConnection1DOF]` | Ordered connection refs |
| `target_values` | `List[float]` | Corresponding target positions (same order) |
| `state_type` | `Optional[JointStateType]` | Semantic type tag (e.g. OPEN, CLOSED, PARKED) |
| `name` | `PrefixedName` | Default: `"JointState"` |

## Key methods

| Method | Returns | Notes |
|---|---|---|
| `is_achieved()` | `bool` | `np.allclose(connection.position, target, atol=1e-2)` for all |
| `items()` | `zip` | `(connection, target_value)` pairs — matches `dict.items()` API |
| `assign_to_robot(robot)` | — | Ensures state belongs to exactly one robot |
| `copy_for_world(world)` | `JointState` | Re-resolves connections in a new world (needed after world copy) |

## Related

- **Uses:** [[sdt.world_description.connections]] (`ActiveConnection1DOF` references)
- **Used by:** [[sdt.robots.abstract_robot.AbstractRobot]] (gripper/arm/torso state members)

## Provenance

- `joint_state.py:1-159` — `JointState`, `is_achieved`, factory classmethods, `copy_for_world`, `GripperState`/`ArmState`/`TorsoState` aliases.
