---
id: sdt.reasoning.robot_predicates
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/reasoning/robot_predicates.py
    lines: [1, 205]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.robots.abstract_robot.AbstractRobot
  - sdt.world_description.world_entity.Body
  - sdt.world_description.geometry.BoundingBox
  - sdt.collision_checking
  - sdt.reasoning.predicates
  - sdt.spatial_computations.raytracer
used_by:
  - pycram.robot_plans.actions.core.container
status: stable
tags: [sdt, reasoning, predicate, symbolic-function, gripper, collision, ray-sampling, robot, blocking]
last_ingest: 2026-05-18
---

_Module of `@symbolic_function`-decorated robot-centric predicates: gripper occupancy via ray-sampling, collision checking with configurable rule sets, blocking-body detection, and footprint collision._

## Purpose

`sdt.reasoning.robot_predicates` extends `sdt.reasoning.predicates` with robot-specific inference. The key design choice is the ray-sampling approach to gripper occupancy: rather than relying on kinematic attachment, `bodies_in_gripper` and `is_body_in_gripper` cast rays between the actual gripper finger meshes in the trimesh scene and return bodies physically located between the fingers — even before a pick-up attachment has been registered.

## `is_body_in_gripper(body, gripper, sample_size=100) → float`

```python
@symbolic_function
def is_body_in_gripper(body: Body, gripper: Manipulator, sample_size: int = 100) -> float
```

Returns the fraction of rays between the gripper fingers that hit `body`:
`len([b for b in bodies_in_gripper(gripper, sample_size) if b == body]) / sample_size`.

Range: `[0.0, 1.0]`. Used by `OpenAction.post_condition` with threshold `> 0.9`.

## `bodies_in_gripper(gripper, sample_size=100) → List[Body]`

```python
@symbolic_function
def bodies_in_gripper(gripper: ParallelGripper, sample_size: int = 100) -> List[Body]
```

**Algorithm:**
1. Copies `gripper.thumb.tip.collision.combined_mesh` and `gripper.finger.tip.collision.combined_mesh`.
2. Applies each body's global transform to get world-frame meshes.
3. Samples `sample_size` surface points from each finger mesh via `trimesh.sample.sample_surface`.
4. Calls `gripper._world.ray_tracer.ray_test(finger_points, thumb_points)` — each finger-point → thumb-point ray represents the closing direction.
5. Returns `set(bodies_hit) - set(finger.bodies) - set(thumb.bodies)` — excludes the gripper's own geometry.

Note: requires a `ParallelGripper` (two-fingered); does not generalize to `HumanoidGripper`.

## `robot_in_collision(robot, ignore_collision_with=None, threshold=0.001) → List[ClosestPoints]`

```python
@symbolic_function
def robot_in_collision(robot, ignore_collision_with=None, threshold=0.001) -> List[ClosestPoints]
```

Temporarily configures the collision manager with three rules:
1. `AvoidExternalCollisions(buffer_zone_distance=threshold, robot=robot)` — flag robot–environment pairs.
2. `AllowSelfCollisions(robot=robot)` — suppress robot self-pairs.
3. `AllowCollisionBetweenGroups(robot.bodies, ignore_collision_with)` — exclude specified bodies.

Uses `world.modify_world()` for rule changes, then calls `collision_manager.compute_collisions()` and returns `contacts`.

## `robot_holds_body(robot, body) → bool`

Queries all `ParallelGripper` annotations whose `_robot == robot` via EQL, then checks `is_body_in_gripper(body, gripper) > 0.0` for each. Returns `True` if any gripper registers a positive score.

## `blocking(pose, root, tip) → List[ClosestPoints]`

**Algorithm:**
1. Calls `world.compute_inverse_kinematics(root, tip, pose, max_iterations=1000)` to reach the target.
2. Writes the resulting joint positions directly into `world.state`.
3. Queries the robot owning `tip` via EQL (`contains(r.bodies, tip)`).
4. Returns `robot_in_collision(robot, [])`.

Note: unlike `pycram.pose_validator.pose_sequence_reachability_validator`, this function does NOT restore world state after the IK writes. It mutates `world.state` permanently.

## `is_gripper_holding_something(gripper) → bool`

Kinematic-structure check: `len(gripper._world.get_kinematic_structure_entities_of_branch(gripper.tool_frame)) > 0`. Equivalent to `GripperIsNotFree` from `pycram.querying.predicates` but without the `Predicate` base class.

## `is_pose_free_for_robot(robot, pose) → bool`

Gets `robot.base.bounding_box`, transforms it to the target `pose` frame, and calls `is_place_occupied(target_bb, robot._world, robot.bodies_with_collision)` from `sdt.reasoning.predicates`. Returns `True` if the robot's footprint at `pose` would not collide with any world body.

## Design observations

- `bodies_in_gripper` uses rays from **surface samples** (100 points per finger), not a dense grid. Very thin objects may not register unless `sample_size` is increased.
- `blocking()` permanently mutates world state via IK writes — callers must save/restore `world.state._data` manually if needed. Contrast with `pose_sequence_reachability_validator` which always restores state.
- `robot_holds_body` queries `ParallelGripper` only; `HumanoidGripper` robots will never pass this check.

## Related

**Uses:** [[sdt.robots.abstract_robot.AbstractRobot]], [[sdt.world_description.world_entity.Body]], [[sdt.world_description.geometry.BoundingBox]], [[sdt.collision_checking]], [[sdt.reasoning.predicates]], [[sdt.spatial_computations.raytracer]]

**Used by:** [[pycram.robot_plans.actions.core.container]] (`is_body_in_gripper` in `OpenAction.post_condition`)

**See also:** [[pycram.querying.predicates]] — kinematic-attachment-based gripper check (no ray sampling); [[pycram.pose_validator]] — reachability validation with state save/restore

## Open questions

- `blocking()` mutates `world.state` via IK writes without restoring it. Any caller must handle state cleanup explicitly. Whether any action in pycram calls `blocking()` and handles this is unverified.
- `bodies_in_gripper` assumes the gripper has a `thumb` and `finger` attribute (i.e. `ParallelGripper`). Non-parallel gripper types are untested and likely unsupported.

## Provenance

- `semantic_digital_twin/src/semantic_digital_twin/reasoning/robot_predicates.py:1-205` —
  `robot_in_collision`, `robot_holds_body`, `blocking`, `bodies_in_gripper`, `is_body_in_gripper`,
  `is_gripper_holding_something`, `is_pose_free_for_robot`.
