---
id: pycram.locations.locations.CostmapLocation
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/locations/locations.py
    lines: [1, 697]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world.World
  - pycram.datastructures.grasp.GraspDescription
  - pycram.pose_validator
  - pycram.locations.costmaps
used_by:
  - pycram.robot_plans.actions.composite
status: stable
tags: [location, costmap, reachability, visibility, base-pose, grasp, navigation]
last_ingest: 2026-05-17
---

_Samples robot base poses satisfying reachability, visibility, and grasp constraints by merging occupancy, visibility, and ring costmaps and IK-validating up to 600 candidates in a sandbox world copy._

## Purpose

`CostmapLocation` answers "where can the robot stand to perform this action?" It is the
primary location resolver used by manipulation actions (`TransportAction`, `SearchAction`,
`OpenAction`). The sampler is purely spatial: it deepcopies the SDT world for collision
isolation, builds a merged costmap, and IK-validates sampled poses against a sandbox copy
without modifying the live world.

## Construction

```python
CostmapLocation(
    target=target_pose,
    reachable=True,
    visible=False,
    reachable_arm=Arms.RIGHT,
    context=ctx,
    grasp_description=gd,
)
```

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `target` | `Pose` | — | Target pose the robot must reach or see |
| `reachable` | `bool` | `False` | Enable IK reachability validation |
| `visible` | `bool` | `False` | Enable visibility (ray-cast) validation |
| `reachable_arm` | `Arms \| None` | `None` | Arm to use; activates `RingCostmap` when set |
| `context` | `Context \| None` | `None` | Provides world + robot |
| `grasp_description` | `GraspDescription \| None` | `None` | Passed through to yielded `GraspPose` |

## Iteration algorithm (`__iter__`)

1. **Sandbox world**: `world = deepcopy(context.world)` — all collision/IK tests run against
   this copy without modifying the live world.

2. **Costmap construction** (`setup_costmaps()`):
   - `OccupancyCostmap(world)` — always; marks grid cells blocked by world bodies.
   - `+ VisibilityCostmap(target, world)` — if `visible=True`; cells with line-of-sight.
   - `+ RingCostmap(target, distance=0.4)` — if `reachable=True`; ring at **0.4 m
     radius (hardcoded)** around target to keep robot at a useful manipulation distance.
   - Three-component merge → `MergedCostmap`; a candidate must pass all active components.

3. **Sampling**: up to **600 candidates** drawn from the merged costmap (hardcoded limit).

4. **Per-candidate validation**:
   - Collision: place robot at candidate in sandbox world → check no collision.
   - Visibility: if `visible`, cast a ray from camera frame to target.
   - Reachability: if `reachable`, call `pose_sequence_reachability_validator([target_T_grasp], tip_link, robot, world)`.
     Returns `False` on `TimeoutError` → skip candidate. Uses full giskardpy MSC execution,
     not the standalone `InverseKinematicsSolver`. World state is save/restored by the validator.

5. **Yield**: `GraspPose(candidate_pose, grasp_description)` per valid candidate.

6. **Raise**: `LocationNotFound` if all 600 candidates exhaust without a valid one.

## Design observations

- The **0.4 m ring radius** and **600 candidate limit** are both hardcoded. No per-call
  API to adjust them.
- Failure raises `LocationNotFound` with no breakdown of how many candidates failed per
  criterion — collision vs. IK vs. visibility indistinguishable.
- The sandbox `deepcopy` uses `sdt.world.World.__deepcopy__` (structured clone, not
  `copy.deepcopy`); semantic annotation and actuator copy semantics are an open question
  (see [[sdt.world.World]] open questions).

## Related

**Uses:** [[sdt.world.World]], [[pycram.datastructures.grasp.GraspDescription]], [[pycram.pose_validator]], [[pycram.locations.costmaps]]

**Used by:** [[pycram.robot_plans.actions.composite]], [[pycram.robot_plans.actions.core.container]]

**See also:** [[pycram.locations.locations.AccessingLocation]], [[pycram.locations.locations.GiskardLocation]]

## Open questions

- Whether `VisibilityCostmap` and `RingCostmap` are ever combined (`visible=True` AND
  `reachable_arm` set) is unknown — the code path exists but no action currently
  exercises it.

## Provenance

- `pycram/src/pycram/locations/locations.py:1-697` — `Location` ABC, `CostmapLocation`,
  `AccessingLocation`, `GiskardLocation`.
