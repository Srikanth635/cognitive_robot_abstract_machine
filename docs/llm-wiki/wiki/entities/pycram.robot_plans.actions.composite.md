---
id: pycram.robot_plans.actions.composite
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/robot_plans/actions/composite/searching.py
    lines: [1, 91]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: pycram/src/pycram/robot_plans/actions/composite/facing.py
    lines: [1, 69]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: pycram/src/pycram/robot_plans/actions/composite/transporting.py
    lines: [1, 406]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: pycram/src/pycram/robot_plans/actions/composite/tool_based.py
    lines: [1, 254]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.robot_plans.ActionDescription
  - pycram.robot_plans.actions.core.PickUpAction
  - pycram.robot_plans.actions.core.PlaceAction
  - pycram.robot_plans.actions.core.NavigateAction
  - pycram.robot_plans.actions.core.container
  - pycram.robot_plans.actions.core.robot_body
  - pycram.robot_plans.actions.core.misc
  - pycram.datastructures.grasp.GraspDescription
  - pycram.locations.locations.CostmapLocation
  - sdt.world_description.world_entity.Body
  - sdt.spatial_types.Pose
  - pycram.datastructures.enums.Arms
used_by: []
fields:
  FaceAtAction:
    pose:
      type: sdt.spatial_types.Pose
      description: Target pose to face toward; robot rotates chassis to align heading, then runs LookAtAction.
    keep_joint_states:
      type: bool
      description: Preserve arm and head joints during the chassis rotation.
  SearchAction:
    target_location:
      type: sdt.spatial_types.Pose
      description: Centre of the search area; robot navigates to a visible location nearby, then sweeps three gaze directions.
    object_sem_annotation:
      type: sdt.semantic_annotations.SemanticAnnotations
      description: SDT annotation type to search for (e.g. Cup, Drawer).
  TransportAction:
    object_designator:
      type: sdt.world_description.world_entity.Body
      description: Object to transport.
    target_location:
      type: sdt.spatial_types.Pose
      description: Place destination.
    arm:
      type: pycram.datastructures.enums.Arms
      domain: [LEFT, RIGHT]
      description: Arm to use; optional — if None, any reachable arm is selected via CostmapLocation.
    grasp_description:
      type: pycram.datastructures.grasp.GraspDescription
      description: Grasp override; optional — if None, computed automatically via calculate_grasp_descriptions.
status: stable
tags: [action, composite, transport, search, navigate, plan-composition]
last_ingest: 2026-05-19
---

_Composite `ActionDescription` subclasses that compose core actions and motions into multi-step behaviours: `FaceAtAction`, `SearchAction`, `TransportAction`, `PickAndPlaceAction`, `MoveAndPlaceAction`, `MoveAndPickUpAction`._

## FaceAtAction

Rotates the robot chassis to face a target pose, then performs `LookAtAction`.

| Field | Type | Description |
|-------|------|-------------|
| `pose` | `Pose` | Target to face |
| `keep_joint_states` | `bool` | Preserve arm/head joints during navigation |

**execute()**: computes heading angle with `arctan2(robot_y - pose_y, robot_x - pose_x) + π`, creates a new robot pose with that yaw, then runs:
```
sequential([NavigateAction(new_robot_pose), LookAtAction(pose)])
```

## SearchAction

Searches for objects of a given semantic type around a target location by sweeping three look-angles.

| Field | Type | Description |
|-------|------|-------------|
| `target_location` | `Pose` | Centre of the search area |
| `object_sem_annotation` | `Type[SemanticAnnotation]` | SDT annotation type to find |

**execute()**:
1. `NavigateAction(CostmapLocation(target, visible=True))` — moves to a location with line-of-sight.
2. `try_in_order([sequential([LookAt(target), Detect(TYPES, annotation)]) for target in [centre, left−0.5m, right+0.5m]])` — sweeps three candidate gaze directions.
3. Diffs `world.semantic_annotations` before/after — if no new IDs, raises `PerceptionObjectNotFound(self)`.

## TransportAction

Full pick-and-place with obstacle awareness: opens containers if needed, navigates to the pick-up location, picks up, parks, raises torso, then navigates to place and places.

| Field | Type | Description |
|-------|------|-------------|
| `object_designator` | `Body` | Object to transport |
| `target_location` | `Pose` | Place destination |
| `arm` | `Arms` | Arm to use (None = any) |
| `grasp_description` | `GraspDescription \| None` | Override grasp; `None` = auto-compute |

**execute()**:
1. For each containing body with `InsideOf > 0.9`: `OpenAction(container_drawer_handle, arm)`.
2. `ParkArmsAction(BOTH)`.
3. `CostmapLocation(object.global_pose, reachable_arm, reachable=True, grasp_description).ground()` — raises `BodyUnfetchable` if no reachable pose found.
4. `NavigateAction + PickUpAction + ParkArmsAction + MoveTorsoAction(HIGH)`.
5. Place plan: `underspecified(NavigateAction)` (krrood resolves place navigation pose) + `PlaceAction + ParkArmsAction`.

`_make_navigate_action_for_placing` wraps `NavigateAction` in krrood's `underspecified(...)` with a `variable(Pose, domain=CostmapLocation(...))` — the navigation target is grounded lazily at execution time.

## PickAndPlaceAction

Pick-up and place **without moving the robot base** — assumes the robot is already in reach of both positions.

`execute()`: `ParkArmsAction → PickUpAction → ParkArmsAction → PlaceAction → ParkArmsAction`.

`validate()`: checks `object.pose == target_location`; raises `ValueError` if not.

## MoveAndPlaceAction / MoveAndPickUpAction

Navigate to a fixed standing position, face the target, then place/pick-up.

```
MoveAndPlaceAction:   NavigateAction(standing_position) → FaceAtAction(target) → PlaceAction
MoveAndPickUpAction:  NavigateAction(standing_position) → FaceAtAction(obj_pose) → PickUpAction
```

## `EfficientTransportAction` (legacy / non-functional)

Located in `transporting.py:325-406`. Selects the closest free arm via Euclidean distance, then runs `ParkArmsAction → PickUpAction → ParkArmsAction → PlaceAction → ParkArmsAction`.

**Non-functional at this commit.** Uses `BelieveObject(names=[...]).resolve()` (pycram-bullet `BelieveObject` wrapper), `RobotDescription.current_robot_description.get_arm_chain()`, `robot.get_link_position()`, and `robot._attached_objects` — all legacy pycram-bullet APIs that do not exist in the SDT-based architecture. Also uses `ParkArmsActionDescription` / `PickUpActionDescription` / `PlaceActionDescription` (old naming convention with `Description` suffix).

## Tool-based actions in `tool_based.py` (legacy / non-functional)

`MixingAction`, `PouringAction`, and `CuttingAction` all use:
- `LocalTransformer()` — pycram-bullet ROS coordinate transformer, not imported
- `World.current_world` — old pycram-bullet world singleton
- `RobotDescription.current_robot_description` — old robot description API
- `MovementType.CARTESIAN` — old motion enum

Additionally, `CuttingAction.execute()` is **dead code** at this commit — it computes `slice_poses` and `lift_pose` in a loop but never calls `.perform()` on any motion. The loop body ends with `lift_pose = new_pose.copy(); lift_pose.pose.position.z += height` and does nothing further.

Only `MoveToolCenterPointMotion` (from current pycram) is imported successfully. None of these three actions are usable in the current SDT-based codebase.

## Open questions

- `TransportAction.open_container` uses krrood EQL `an(entity(drawer := variable(Drawer, domain=...)).where(...))` — the EQL query mechanism is documented at a high level in [[pycram.plans.UnderspecifiedNode]] but not yet covered in a dedicated concept page (planned for Phase 10).
- `EfficientTransportAction._choose_best_arm` references `robot._attached_objects` — a private dict that does not exist on SDT's `AbstractRobot`. Whether this class is intended for future migration or is dead is unclear.

## Related

**Uses:** [[pycram.robot_plans.ActionDescription]], [[pycram.robot_plans.actions.core.PickUpAction]], [[pycram.robot_plans.actions.core.PlaceAction]], [[pycram.robot_plans.actions.core.NavigateAction]], [[pycram.robot_plans.actions.core.container]], [[pycram.robot_plans.actions.core.robot_body]], [[pycram.robot_plans.actions.core.misc]], [[pycram.datastructures.grasp.GraspDescription]], [[pycram.locations.locations.CostmapLocation]], [[sdt.world_description.world_entity.Body]], [[sdt.spatial_types.Pose]]

## Provenance

- `pycram/src/pycram/robot_plans/actions/composite/facing.py` lines 1–69 — `FaceAtAction`.
- `pycram/src/pycram/robot_plans/actions/composite/searching.py` lines 1–91 — `SearchAction`.
- `pycram/src/pycram/robot_plans/actions/composite/transporting.py` lines 1–406 — `TransportAction` (1–270), `PickAndPlaceAction`, `MoveAndPlaceAction`, `MoveAndPickUpAction` (271–324), `EfficientTransportAction` (325–406, non-functional legacy).
- `pycram/src/pycram/robot_plans/actions/composite/tool_based.py` lines 1–254 — `MixingAction`, `PouringAction`, `CuttingAction` (all non-functional legacy; `CuttingAction.execute()` is dead code).
