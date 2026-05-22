---
id: pycram.pose_validator
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/pose_validator.py
    lines: [1, 213]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world.World
  - sdt.robots.abstract_robot.AbstractRobot
  - giskardpy.qp.qp_controller
  - giskardpy.motion_statechart.context
  - pycram.alternative_motion_mapping.AlternativeMotion
used_by:
  - pycram.locations.locations.CostmapLocation
  - pycram.locations.locations.AccessingLocation
  - pycram.robot_plans.actions.core.container
status: stable
tags: [pycram, pose-validator, reachability, visibility, collision, giskardpy, msc, world-state, executor]
last_ingest: 2026-05-17
---

_Module of robot-pose validation functions: `visibility_validator` (single ray-test), `reachability_validator` / `pose_sequence_reachability_validator` (full giskardpy MSC execution with world-state save/restore), and `collision_check` (FCL proximity list)._

## Purpose

`pycram.pose_validator` is called by `CostmapLocation` and `AccessingLocation` during their
candidate-iteration loops to reject base poses that are in collision, unreachable, or invisible
before the robot commits to moving there. All validators leave the live world state unchanged
after execution — world mutations are either avoided or explicitly rolled back.

## `visibility_validator(robot, object_or_pose, world) → bool`

Casts a single ray from the robot's camera to the target and returns `True` if the first
non-robot body hit is the target.

```python
visibility_validator(
    robot: AbstractRobot,
    object_or_pose: Body | Pose,
    world: World,
) -> bool
```

**Behavior:**
- If the input is a `Pose` (not a `Body`), creates a temporary `Body` with `Connection6DoF`,
  shoots the ray, then immediately removes the body.
- Uses `world.ray_tracer.ray_test(camera_origin, target_position)` — single point-to-point ray.

**Differs from `sdt.reasoning.predicates.visible()`:** `predicates.visible()` uses
`RayTracer.create_segmentation_mask()` at 256-pixel resolution with camera orientation forced
to identity. `visibility_validator` sends exactly one ray and checks for occlusion by other
bodies between the camera and target. The two can disagree on borderline cases.

## `reachability_validator(target_pose, tip_link, robot_view, world, use_fullbody_ik=False) → bool`

Thin `@symbolic_function`-decorated wrapper; delegates immediately to
`pose_sequence_reachability_validator([target_pose], ...)`.

## `pose_sequence_reachability_validator(target_sequence, tip_link, robot_view, world, use_fullbody_ik=False) → bool`

The primary validator. Builds and executes a full giskardpy MSC on the live world with
explicit state save and restore in all code paths.

```python
@symbolic_function
def pose_sequence_reachability_validator(
    target_sequence: List[Pose],
    tip_link: KinematicStructureEntity,
    robot_view: AbstractRobot,
    world: World,
    use_fullbody_ik: bool = False,
) -> bool
```

**Algorithm:**

1. **Save state**: `old_state = deepcopy(world.state._data)`.
2. **AlternativeMotion check**: calls `check_for_alternative(robot_view, tip_link, ...)`. If an
   `AlternativeMotion` override is registered for this robot/execution type, its goal sequence
   replaces the default `CartesianPose` targets.
3. **Build MSC**: constructs a `Sequence` of `CartesianPose` nodes (one per target pose), adds
   `ExternalCollisionAvoidance` constraint, adds `EndMotion.when_true(sequence_node)`.
4. **Create executor**:
   ```python
   executor = Executor(MotionStatechartContext(
       world=world,
       qp_controller_config=QPControllerConfig(
           target_frequency=50, prediction_horizon=4, verbose=False
       )
   ))
   executor.compile(msc)
   ```
5. **Run**: `executor.tick_until_end()`.
6. **Failure**: catches `TimeoutError` → returns `False`.
7. **`finally` block (always runs)**:
   ```python
   world.state._data[:] = old_state
   world.notify_state_change()
   ```
   World joint positions are restored to their pre-call values regardless of success or failure.
8. Returns `True` on success.

**Key design point:** the executor modifies the live `world` (not a deepcopy) while running,
but the `finally` block restores `world.state._data` in-place. This means FK and joint positions
are temporarily modified mid-call but callers observe no net change. Only the world *state* is
restored — structural world mutations (attach/detach) would not be rolled back, but standard MSC
execution does not cause structural changes.

**Same mechanism as `GiskardLocation`:** `GiskardLocation.__iter__` constructs its own `Executor`
directly using the same pattern. `CostmapLocation` and `AccessingLocation` both delegate to
`pose_sequence_reachability_validator` from this module; `GiskardLocation` is the exception.

## `collision_check(robot, world) → List[ClosestPoints]`

```python
collision_check(robot: AbstractRobot, world: World) -> List[ClosestPoints]
```

1. Adds `AllowSelfCollisions` collision rule to the robot's rule stack.
2. Calls `collision_manager.update_collision_matrix(buffer=0.0)`.
3. Returns all `ClosestPoints` pairs with `distance <= 0` (actual inter-penetrations).

Used by `CostmapLocation` and `AccessingLocation` as the cheap first filter before the more
expensive `pose_sequence_reachability_validator` call. A non-empty return list → skip the candidate.

## Related

**Uses:** [[sdt.world.World]], [[sdt.robots.abstract_robot.AbstractRobot]], [[giskardpy.qp.qp_controller]], [[giskardpy.motion_statechart.context]], [[pycram.alternative_motion_mapping.AlternativeMotion]]

**Used by:** [[pycram.locations.locations.CostmapLocation]], [[pycram.locations.locations.AccessingLocation]], [[pycram.robot_plans.actions.core.container]]

**See also:** [[pycram.locations.locations.GiskardLocation]] — constructs its own `Executor` directly with the same pattern instead of delegating here; [[sdt.reasoning.predicates]] — provides `visible()` using a different mechanism (segmentation mask, not ray-test)

## Open questions

- `pose_sequence_reachability_validator` runs the MSC on the live world; only `world.state._data`
  is restored. If MSC execution triggers structural world changes (body attachment/detachment),
  those would persist across the call. Standard MSC evaluation does not cause structural changes,
  but this assumption is not formally guaranteed.
- `use_fullbody_ik` controls root-link selection but exact behavior is robot-dependent and not
  documented in the source.
- The `target_frequency=50, prediction_horizon=4` config is hardcoded — shorter horizon than
  the standard `prediction_horizon=7`. Whether this is intentional for speed or a latent tuning
  issue is undocumented.

## Provenance

- `pycram/src/pycram/pose_validator.py:1-213` — `visibility_validator`, `reachability_validator`,
  `pose_sequence_reachability_validator`, `collision_check`.
