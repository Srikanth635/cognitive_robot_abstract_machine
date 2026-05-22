---
id: pycram.robot_plans.actions.core.NavigateAction
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/robot_plans/actions/core/navigation.py
    lines: [23, 91]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.robot_plans.ActionDescription
  - sdt.spatial_types.Pose
  - pycram.plans.factories
  - pycram.robot_plans.motions.gripper
  - pycram.robot_plans.motions.navigation
used_by:
  - pycram.robot_plans.actions.composite
fields:
  target_location:
    type: sdt.spatial_types.Pose
    description: World pose to navigate to.
  keep_joint_states:
    type: bool
    description: Keep arm joints fixed during navigation; value read from ActionConfig.navigate_keep_joint_states at runtime.
status: stable
tags: [action, navigation, look-at, mobile]
last_ingest: 2026-05-19
---

_Navigates the robot to a target `Pose`. Also in this module: `LookAtAction` (directs a camera toward a target pose)._

## NavigateAction

The simplest concrete action: wraps a single `MoveMotion` in an `execute_single` plan.

```python
def execute(self) -> None:
    self.add_subplan(
        execute_single(MoveMotion(self.target_location, self.keep_joint_states))
    ).perform()
```

### Key fields

| Field | Type | Notes |
|---|---|---|
| `target_location` | `Pose` | World pose to navigate to. |
| `keep_joint_states` | `bool` | Default from `ActionConfig.navigate_keep_joint_states`. Keep arm joints fixed during navigation. |

### Pre / post conditions

- **Pre:** robot has a `drive` component AND `is_pose_free_for_robot(robot, target_location)`.
- **Post:** `allclose(robot.root.global_pose, target_location, atol=0.03)` — within 3 cm.

`MoveMotion` (in `pycram.robot_plans.motions.navigation`) is the corresponding
`BaseMotion` — not yet on its own wiki page (Phase 5 target with navigation module).

---

## LookAtAction

Directs the robot's camera toward a target `Pose`.

```python
def execute(self) -> None:
    camera = self.camera or self.robot.get_default_camera()
    self.add_subplan(
        execute_single(LookingMotion(target=self.target, camera=camera))
    ).perform()
```

### Key fields

| Field | Type | Notes |
|---|---|---|
| `target` | `Pose` | Pose to look at (6D). |
| `camera` | `Camera` | Optional; falls back to `robot.get_default_camera()`. |

No pre/post conditions defined in source.

---

## Related

- Base class: [[pycram.robot_plans.ActionDescription]]
- Target type: [[sdt.spatial_types.Pose]]
- Combinator: [[pycram.plans.factories]]

## Open questions

- `keep_joint_states` is read from `ActionConfig.navigate_keep_joint_states` — a
  configuration singleton. Its default value and override mechanism are not yet in the
  wiki. Flag for a future config ingest.
- `LookAtAction` has no pre/post conditions. Whether symbolic planning can reason
  about gaze direction is unclear.

## Provenance

- `pycram/src/pycram/robot_plans/actions/core/navigation.py:23-91` at commit `0528d8cf3`.
