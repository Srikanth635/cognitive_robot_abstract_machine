---
id: pycram.locations.locations.AccessingLocation
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/locations/locations.py
    lines: [321, 524]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world.World
  - sdt.world_description.connections
  - sdt.robots.abstract_robot.AbstractRobot
  - pycram.datastructures.grasp.GraspDescription
  - pycram.pose_validator
  - pycram.locations.costmaps
used_by: []
status: stable
tags: [location, accessing, container, drawer, door, non-fixed-connection, reachability, grasp]
last_ingest: 2026-05-17
---

_Location sampler for articulated containers (drawers, doors): traverses the kinematic chain to find the first non-`FixedConnection` joint, computes a 3-pose open/close sequence, and IK-validates each candidate across all arm chains._

## Purpose

`AccessingLocation` specialises location sampling for articulated containers. Rather than
building a visibility or ring costmap, it uses an `OccupancyCostmap + GaussianCostmap`
centred on the handle and validates each candidate by testing whether any arm can reach the
full open/close pose sequence (init → half-open → fully open).

Used by `OpenAction` and `CloseAction` to determine where the robot must stand before
grasping and actuating a drawer or door handle.

## Construction

```python
AccessingLocation(
    handle=handle_body,           # sdt.world_description.world_entity.Body
    arm=Arms.RIGHT,               # arm to use
    prepose_distance=None,        # float or None; defaults to ActionConfig.grasping_prepose_distance
    context=ctx,
)
```

`prepose_distance` defaults to `ActionConfig.grasping_prepose_distance` in `__post_init__`.

## `create_target_sequence()`

```python
def create_target_sequence(self) -> List[Pose]:
```

1. `world.compute_chain_of_connections(world.root, handle)` — full path from root to handle.
2. Reverse to get handle-to-root direction.
3. Filter: `filter(lambda c: not isinstance(c, FixedConnection), ...)` → **first non-FixedConnection** = the container's movable joint.
4. Compute three poses:
   - `init_pose` — handle at DOF lower limit (closed).
   - `half_pose` — handle at `upper_limit / 1.5` (roughly 2/3 open).
   - `goal_pose` — handle at DOF upper limit (fully open).
5. Returns `[init_pose, half_pose, goal_pose]`.

## `setup_costmaps(handle)`

`OccupancyCostmap(robot, distance_to_obstacle=robot_footprint_radius, origin=handle.global_pose @z=0)`  
`+ GaussianCostmap(mean=200, sigma=15, origin=handle_pose)`  
→ Gaussian pulls robot toward handle; occupancy excludes obstacles.
`adjust_map_for_drawer_opening` can then zero out cells in the drawer's sweep path.

## Iteration algorithm (`__iter__`)

1. **Sandbox world**: `test_world = deepcopy(self.world)` — IK tests run in isolation.
2. `test_robot = self.robot.from_world(test_world)`.
3. `final_map.number_of_samples = 600`; orientation generator points toward `half_pose`.
4. For each candidate:
   - `collision_check(test_robot, test_world)` — skip on `RobotInCollision`.
   - For **each arm chain** in `test_robot.manipulator_chains`:
     - Compute FRONT/NoAlignment `GraspDescription` for the arm's manipulator.
     - Apply grasp rotation to the 3-pose sequence.
     - `pose_sequence_reachability_validator(current_target_sequence, arm_chain.manipulator.tool_frame, test_robot, test_world, use_fullbody_ik=...)`.
     - If reachable → `yield pose_candidate`.

Note: **all arm chains are tried**, not just `self.arm` — any arm that can reach yields the pose.

## `adjust_map_for_drawer_opening()` (static)

Removes costmap cells in the rectangular region swept by the drawer between `init_pose` and
`goal_pose` (width = 0.2 m default). Prevents robot from standing directly in the drawer's
path.

## Related

**Uses:** [[sdt.world.World]], [[sdt.world_description.connections]], [[sdt.robots.abstract_robot.AbstractRobot]], [[pycram.datastructures.grasp.GraspDescription]], [[pycram.pose_validator]], [[pycram.locations.costmaps]]

**Used by:** (none confirmed — composite actions navigate to objects via CostmapLocation; OpenAction/CloseAction do not import AccessingLocation directly)

**See also:** [[pycram.locations.locations.CostmapLocation]], [[pycram.robot_plans.motions.container]]

## Open questions

- `AccessingLocation` yields raw `pose_candidate` — not an `AccessPose(pose, connection)`.
  Callers do not receive the identified movable connection; they must re-discover it
  during `OpeningMotion`/`ClosingMotion` execution.
- Only the first non-FixedConnection on the path from root to handle is used. Articulated
  structures with multiple movable joints (e.g. a cabinet with a door and inner drawer)
  would only address the outermost movable joint.

## Provenance

- `pycram/src/pycram/locations/locations.py:321-524` — `AccessingLocation`, `create_target_sequence`,
  `setup_costmaps`, `adjust_map_for_drawer_opening`, `__iter__`.
