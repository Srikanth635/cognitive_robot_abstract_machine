---
id: pycram.locations.locations.GiskardLocation
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/locations/locations.py
    lines: [526, 697]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - giskardpy.motion_statechart.context
  - giskardpy.motion_statechart.graph_node.EndMotion
  - giskardpy.qp.qp_controller
  - sdt.world.World
  - pycram.datastructures.grasp.GraspDescription
  - sdt.robots.abstract_robot.AbstractRobot
used_by: []
status: stable
tags: [location, giskard, reachability, motion-planning, msc, executor, full-motion-test]
last_ingest: 2026-05-17
---

_Location sampler that tests reachability by running a full giskardpy MSC tick rather than point IK — more accurate than `CostmapLocation` but substantially slower; samples only **10 candidates** per call._

## Purpose

`GiskardLocation` answers the same question as `CostmapLocation` ("can the robot's TCP reach
this target from this base pose?") but uses *motion planning feasibility* as the acceptance
criterion. It constructs a `CartesianPose` sequence → `MotionStatechart`, compiles it with
a fresh `Executor`, ticks to completion, and accepts a pose only if the TCP reaches within
0.02 m of the final target.

Appropriate when IK feasibility is insufficient — e.g. when collision along the path,
velocity limits, or whole-body constraints could invalidate an IK-reachable pose.

## Construction

```python
GiskardLocation(
    target_pose=target_pose,          # Pose: TCP target
    arm=Arms.RIGHT,                   # Arm to use
    grasp_description=None,           # optional GraspDescription for pose sequence
    threshold=0.02,                   # m: TCP distance threshold for success (default 0.02)
    context=ctx,
)
```

## Iteration algorithm (`__iter__`)

1. **Costmap**: `OccupancyCostmap + GaussianCostmap` centred on target; `number_of_samples = 10`
   (much less than CostmapLocation's 600).
2. **Sandbox world**: `test_world = deepcopy(self.world)`; `test_robot = robot.__class__.from_world(test_world)`.
3. For each candidate and grasp_description:
   a. `test_robot.root.parent_connection.origin = candidate`
   b. `executor = setup_giskard_executor(target_sequence, test_world, test_robot, test_ee)`
   c. `executor.tick_until_end()` — catches `TimeoutError` and `InfeasibleException` (skip).
   d. `dist = test_ee.global_pose.to_position().euclidean_distance(target_sequence[-1].to_position())`
   e. If `dist > 0.02` → skip.
4. **Yield**: `GraspPose.from_pose(test_robot.root.global_pose, grasp_desc, arm=self.arm)` —
   note: yields the robot's **root global pose** (after FK converged), not the raw candidate.

## `setup_giskard_executor()`

Builds the giskardpy MSC directly — **not** via `pycram.motion_executor.MotionExecutor`:

```python
pose_seq = Sequence(nodes=[CartesianPose(root_link, tip_link, goal_pose) for pose in sequence])
msc = MotionStatechart()
msc.add_nodes([pose_seq, ExternalCollisionAvoidance(robot=robot_view, ...)])
msc.add_node(EndMotion.when_true(pose_seq))
executor = Executor(
    MotionStatechartContext(
        world=world,
        qp_controller_config=QPControllerConfig(target_frequency=50, prediction_horizon=4),
    )
)
executor.compile(msc)
return executor
```

`Executor` here is `giskardpy.executor.Executor` — the internal giskardpy tick driver, not
pycram's `MotionExecutor` wrapper.

## Key differences from `CostmapLocation`

| Property | CostmapLocation | GiskardLocation |
|----------|-----------------|-----------------|
| Candidate count | 600 | **10** |
| Reachability test | IK solver | Full MSC tick |
| Failure mode | IK unreachable | TimeoutError / InfeasibleException / dist > 0.02 |
| Speed | Fast (IK only) | Slow (compile + tick per candidate) |
| Yielded pose | candidate base pose | TCP-verified root global pose |

## Related

**Uses:** [[giskardpy.motion_statechart.context]], [[giskardpy.qp.qp_controller]], [[sdt.world.World]], [[pycram.datastructures.grasp.GraspDescription]]

**See also:** [[pycram.locations.locations.CostmapLocation]], [[pycram.locations.locations.AccessingLocation]]

## Open questions

- Which action(s) currently use `GiskardLocation` vs `CostmapLocation` in practice is not
  confirmed in source — it may be experimental or used only in specific robot configurations.
- `executor.tick_until_end()` has no explicit iteration limit shown in this path (unlike
  MotionExecutor's 2000-tick cap). Whether Executor has its own limit is unknown.

## Provenance

- `pycram/src/pycram/locations/locations.py:526-697` — `GiskardLocation`, `setup_costmap`,
  `setup_giskard_executor`, `__iter__`.
