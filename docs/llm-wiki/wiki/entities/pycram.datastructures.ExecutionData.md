---
id: pycram.datastructures.ExecutionData
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/datastructures/execution_data.py
    lines: [13, 44]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.spatial_types.Pose
used_by:
  - pycram.plans.ActionNode
status: stable
tags: [dataclass, execution, telemetry, snapshot]
last_ingest: 2026-05-17
---

_Pre/post execution snapshot recorded by [[pycram.plans.ActionNode]]: robot root pose, world state (numpy array), and accumulated world-modification blocks._

## Purpose

`ExecutionData` captures the state of the world and robot at the start and end of an action. This lets callers inspect what changed during a `perform()` call, replay modifications, or detect failures by comparing start/end state. The `added_world_modifications` list collects `WorldModelModificationBlock` entries appended during execution.

## Key attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `execution_start_pose` | `Pose` | Robot root pose at action start |
| `execution_start_world_state` | `np.ndarray` | World state vector at action start |
| `execution_end_pose` | `Pose \| None` | Robot root pose at action end (None if not reached) |
| `execution_end_world_state` | `np.ndarray \| None` | World state vector at action end |
| `added_world_modifications` | `List[WorldModelModificationBlock]` | World modifications accumulated during execution |

## Construction

Created directly as a dataclass by `ActionNode` at the start of `perform()`:
```python
data = ExecutionData(
    execution_start_pose=context.robot.root.global_pose,
    execution_start_world_state=world.get_state(),
)
```
End-of-execution fields are filled in after the action completes.

## Related

**Uses:** [[sdt.spatial_types.Pose]]

**Used by:** [[pycram.plans.ActionNode]]

## Provenance

- `pycram/src/pycram/datastructures/execution_data.py` lines 13–44 (commit `0528d8cf3`) — full `ExecutionData` dataclass definition.
