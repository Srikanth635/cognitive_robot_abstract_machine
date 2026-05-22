---
id: pycram.datastructures.grasp.GraspDescription
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/datastructures/grasp.py
    lines: [1, 433]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: pycram/src/pycram/datastructures/rotations.py
    lines: [1, 37]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: pycram/src/pycram/datastructures/enums.py
    lines: [63, 167]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.robots.abstract_robot.Manipulator
  - sdt.robots.abstract_robot.AbstractRobot
  - sdt.world_description.world_entity.Body
  - sdt.spatial_types.Pose
  - sdt.spatial_types.spatial_types
  - pycram.datastructures.enums.Arms
  - pycram.datastructures.enums.ApproachDirection
  - pycram.datastructures.enums.VerticalAlignment
  - pycram.datastructures.enums.AxisIdentifier
used_by:
  - pycram.robot_plans.actions.core.PickUpAction
  - pycram.robot_plans.actions.core.PlaceAction
  - pycram.robot_plans.motions.gripper
  - pycram.locations.locations.CostmapLocation
  - pycram.locations.locations.AccessingLocation
  - pycram.locations.locations.GiskardLocation
  - pycram.robot_plans.actions.composite
  - pycram.robot_plans.actions.core.container
status: stable
tags: [grasp, geometry, manipulation, pose-sequence, approach-direction, rotations]
last_ingest: 2026-05-19
fields:
  approach_direction:
    type: pycram.datastructures.enums.ApproachDirection
    description: Face of the object's x-y bounding box from which the gripper approaches.
  vertical_alignment:
    type: pycram.datastructures.enums.VerticalAlignment
    description: Whether to approach from above, below, or keep a side grasp.
  manipulator:
    type: sdt.robots.abstract_robot.Manipulator
    description: SDT manipulator annotation; provides tool_frame and front_facing_orientation.
  rotate_gripper:
    type: bool
    default: false
    description: 90-degree roll around the gripper X axis; useful for objects that fit only sideways.
  manipulation_offset:
    type: float
    default: 0.05
    description: Pre-grasp standoff in metres and lift height; one constant controls both.
---

_Bundled page: `GraspDescription` (grasp geometry), the `Rotations` quaternion tables (`SIDE_ROTATIONS` / `VERTICAL_ROTATIONS` / `HORIZONTAL_ROTATIONS`), the related enums (`ApproachDirection`, `VerticalAlignment`, `AxisIdentifier`, `Arms`), and the helper types `PreferredGraspAlignment` and `GraspPose`._

## Purpose

`GraspDescription` is the geometry layer between a high-level "pick up the cup" action and the TCP waypoints that `MoveToolCenterPointMotion` / `MoveTCPWaypointsMotion` actually execute. It encodes a grasp strategy as four fields (approach direction, vertical alignment, manipulator reference, optional 90° gripper roll), and from that derives:

1. A grasp orientation (`grasp_orientation()`) — a `Quaternion`
2. A three-pose sequence (`grasp_pose_sequence(body)`) — pre-grasp, grasp, lift
3. Object-to-robot face matching (`calculate_grasp_descriptions`) — ranked list of plausible grasps

The class is the central artifact that LLM-based grounding (Q6 in the design conversation) and EQL-based grounding both produce when filling in the `grasp_description` slot of a `PickUpAction` or `PlaceAction`.

## When to use

- **PickUpAction / PlaceAction** — both call `grasp_pose_sequence(body)` or `place_pose_sequence(pose)` to generate waypoints for the `ReachMotion` / `MoveGripperMotion` chain.
- **Automatic grasp enumeration** — `GraspDescription.calculate_grasp_descriptions(manipulator, pose)` ranks all valid approach × vertical combinations by angular proximity to the robot.
- **Computing the held-object's pose during place** — `place_pose_sequence(target_pose)` calls `_pose_sequence` with `reverse=True` (retract order), reading the held body from `manipulator.tool_frame.child_kinematic_structure_entities[0]`.
- **Not for:** directly commanding the robot — pass the resulting `List[Pose]` to a motion designator.

## `GraspDescription` — the dataclass

```python
@dataclass
class GraspDescription:
    approach_direction: ApproachDirection      # FRONT/BACK/LEFT/RIGHT
    vertical_alignment: VerticalAlignment      # TOP/BOTTOM/NoAlignment
    manipulator: Manipulator                   # SDT manipulator annotation
    rotate_gripper: bool = False               # 90° roll around x
    manipulation_offset: float = 0.05          # metres — standoff + lift height
```

`__hash__ = id(self)` — instances are distinguished by identity, not by field values.

### Field semantics

| Field | Type | Effect |
|---|---|---|
| `approach_direction` | `ApproachDirection` | Picks one of four faces of the object's x-y bounding box to approach from. |
| `vertical_alignment` | `VerticalAlignment` | TOP/BOTTOM tilts the gripper to grasp from above/below; NoAlignment keeps a side grasp. |
| `manipulator` | `Manipulator` | Provides `front_facing_orientation` (4th rotation factor) and `tool_frame`. See [[sdt.robots.abstract_robot.Manipulator]]. |
| `rotate_gripper` | `bool` | When True, rolls the gripper 90° around its X axis (useful for objects that fit only sideways). |
| `manipulation_offset` | `float` (m) | Distance from the object surface for the pre-grasp pose, AND the +z displacement for the lift pose. Same constant is reused for both standoff and lift height. Default 0.05 m. |

## Pose sequence algorithm

`_pose_sequence(target_T_grasp_pose, body, reverse) → List[Pose]` produces three poses:

```
1. pre_pose    = translate(grasp_pose,  along manipulation_axis,  -(bbox_half + manipulation_offset))
2. grasp_pose  = target_T_grasp_pose with orientation = grasp_orientation()
3. lift_pose   = same position as grasp_pose, translated +manipulation_offset in MAP z-axis
                 (then transformed back into the target frame)
```

If `body` is `None`, the pre-pose offset collapses to 0 (no bounding-box knowledge). The `lift_pose` is always computed in **map frame** before being transformed back — this guarantees lift is global +z regardless of how the object frame is oriented.

`reverse=True` reverses the sequence to `[lift_pose, grasp_pose, pre_pose]` — used by `PlaceAction` to retract.

### `grasp_pose_sequence(body)` vs. `place_pose_sequence(pose)`

| Call | What it does |
|---|---|
| `grasp_pose_sequence(body)` | Calls `_pose_sequence(Pose(reference_frame=body), body)` — pose sequence anchored at object origin |
| `place_pose_sequence(target_pose)` | Identifies held body via `manipulator.tool_frame.child_kinematic_structure_entities[0]`; calls `_pose_sequence(target_pose, held_body, reverse=True)` |

## `grasp_orientation()` — the quaternion product

Combines four rotations in order:

```
rotation = SIDE_ROTATIONS[approach_direction]                   # face selection
         × VERTICAL_ROTATIONS[vertical_alignment]               # top/bottom tilt
         × HORIZONTAL_ROTATIONS[rotate_gripper]                 # 90° roll
         × manipulator.front_facing_orientation                 # robot-specific orientation
result = normalize(rotation)                                    # unit quaternion
```

The result is a `Quaternion` representing the gripper's world-frame orientation at grasp.

### `Rotations` lookup tables (concrete quaternion values)

Source: `pycram/datastructures/rotations.py`. Quaternions stored as `[x, y, z, w]`.

```python
SIDE_ROTATIONS = {
    ApproachDirection.FRONT:  [0, 0,  0,            1],          # identity
    ApproachDirection.BACK:   [0, 0,  1,            0],          # 180° around Z
    ApproachDirection.LEFT:   [0, 0, -√2/2,         √2/2],       # -90° around Z
    ApproachDirection.RIGHT:  [0, 0,  √2/2,         √2/2],       # +90° around Z
}

VERTICAL_ROTATIONS = {
    VerticalAlignment.NoAlignment: [0,  0,    0, 1],             # identity
    VerticalAlignment.TOP:         [0,  √2/2, 0, √2/2],          # +90° around Y
    VerticalAlignment.BOTTOM:      [0, -√2/2, 0, √2/2],          # -90° around Y
}

HORIZONTAL_ROTATIONS = {
    False: [0,    0, 0, 1],                                       # identity
    True:  [√2/2, 0, 0, √2/2],                                    # 90° around X
}
```

## Enum sources

### `ApproachDirection(Grasp, Enum)`

`(axis: AxisIdentifier, direction: ±1)` pairs:

| Value | Encoding | Meaning |
|---|---|---|
| `FRONT` | `(AxisIdentifier.X, -1)` | Approach from -X face of object |
| `BACK`  | `(AxisIdentifier.X, +1)` | From +X face |
| `RIGHT` | `(AxisIdentifier.Y, -1)` | From -Y face |
| `LEFT`  | `(AxisIdentifier.Y, +1)` | From +Y face |

### `VerticalAlignment(Grasp, Enum)`

| Value | Encoding |
|---|---|
| `NoAlignment` | `(AxisIdentifier.Undefined, 0)` — side grasp, no vertical tilt |
| `TOP`         | `(AxisIdentifier.Z, -1)` — grasp from above |
| `BOTTOM`      | `(AxisIdentifier.Z, +1)` — grasp from below |

### `AxisIdentifier(Enum)`

| Value | Tuple |
|---|---|
| `X` | `(1, 0, 0)` |
| `Y` | `(0, 1, 0)` |
| `Z` | `(0, 0, 1)` |
| `Undefined` | `(0, 0, 0)` |

### `Arms(IntEnum)`

| Value | Int |
|---|---|
| `LEFT` | 0 |
| `RIGHT` | 1 |
| `BOTH` | 2 |

## `calculate_grasp_descriptions(manipulator, pose, grasp_alignment=None)` — classmethod

Returns a `List[GraspDescription]` — all plausible grasp configurations, ranked by proximity to the robot. Pseudocode:

```
1. map_T_object  = transform target pose into world.root frame
2. map_V_r→o     = robot_root.global_pose - object.position
3. object_R_map  = inverse rotation of object frame
4. object_V_r    = object_R_map @ map_V_r→o          # robot direction in object frame

5. vector_side     = Vector3(object_V_r.x, object_V_r.y, NaN)     # x-y component only
   side_faces      = calculate_closest_faces(vector_side, optional preferred_axis)

6. if grasp_alignment.with_vertical_alignment:
       vector_vertical = Vector3(NaN, NaN, object_V_r.z)
       vertical_faces  = calculate_closest_faces(vector_vertical)
   else:
       vertical_faces  = [VerticalAlignment.NoAlignment]

7. return [GraspDescription(side, top, rotate_gripper, manipulator)
           for side in side_faces
           for top  in vertical_faces]
```

The result is two faces × two verticals = up to 4 grasps per call (or 2 if `vertical=False`). They are returned in proximity order (closest face first).

### `calculate_closest_faces(pose_to_robot_vector, specified_grasp_axis)` — staticmethod

Returns a `(primary_face, secondary_face)` tuple. Algorithm:

1. Pick the valid axes (those whose component in the vector is not NaN), or use the explicitly specified axis.
2. Sort by absolute magnitude of the vector component — biggest component is the primary face.
3. The sign of the component determines which side of that axis (FRONT vs BACK, LEFT vs RIGHT, TOP vs BOTTOM).
4. The secondary face is the next-largest axis if there are ≥2 valid axes; otherwise it's the opposite side of the primary.

Each face is produced via `ApproachDirection.from_axis_direction(axis, sign)` or `VerticalAlignment.from_axis_direction(...)`.

## Additional geometric methods

| Method | Returns | Used for |
|---|---|---|
| `manipulation_axis()` | `List[float]` (3D) | Axis along which approach happens. Calls `calculate_manipulator_axis(AxisIdentifier.X)`. |
| `lift_axis()` | `List[float]` (3D) | Axis along which lift happens. Calls `calculate_manipulator_axis(AxisIdentifier.Z)`. |
| `calculate_manipulator_axis(axis)` | `List[float]` (3D) | Transforms a body-frame axis vector through the manipulator's `front_facing_orientation` to get the corresponding gripper-frame axis. |
| `edge_offset(body)` | `float` | Half the bounding-box dimension along the approach axis. Used by `grasp_pose(grasp_edge=True)` for edge-grasping. |
| `grasp_pose(body, grasp_edge=False)` | `Pose` | A single grasp pose in the body frame — `(edge_offset, 0, 0)` translated, oriented by `grasp_orientation()`. |

## `PreferredGraspAlignment` — optional pinning

```python
@dataclass
class PreferredGraspAlignment:
    preferred_axis: Optional[AxisIdentifier]   # X, Y, Z, or None
    with_vertical_alignment: bool              # enumerate vertical or not
    with_rotated_gripper: bool                 # set rotate_gripper=True on all returned grasps
```

Passed to `calculate_grasp_descriptions` to constrain enumeration: pin the approach to a specific axis, request top/bottom grasps, or force the 90° roll. When None, defaults are `(Undefined, False, False)`.

## `GraspPose(Pose)` — pose annotated with arm and grasp

```python
@dataclass(eq=False, init=False)
class GraspPose(Pose):
    arm: Arms
    grasp_description: GraspDescription
```

A `Pose` enriched with the `Arms` enum (which arm performs the grasp) and the `GraspDescription` (how). Returned from grasp-planning helpers that produce both pose and metadata in one object.

`from_pose(pose, arm, grasp_description)` is a convenience factory.

## Related

- **Uses:** [[sdt.robots.abstract_robot.Manipulator]] (primary consumer of `front_facing_orientation` and `tool_frame`), [[sdt.robots.abstract_robot.AbstractRobot]] (via `manipulator._robot.root.global_pose`), [[sdt.world_description.world_entity.Body]] (bounding box), [[sdt.spatial_types.Pose]], [[sdt.spatial_types.spatial_types]]
- **Used by:** [[pycram.robot_plans.actions.core.PickUpAction]], [[pycram.robot_plans.actions.core.PlaceAction]], [[pycram.robot_plans.motions.gripper]] (passed into `ReachMotion`), [[pycram.locations.locations.CostmapLocation]] / [[pycram.locations.locations.AccessingLocation]] / [[pycram.locations.locations.GiskardLocation]] (for reachability testing)

## Open questions

- `PickUpAction.execute` calls `grasp_description.grasp_pose_sequence(object)` while `ReachAction` calls `grasp_description._pose_sequence(target_pose, body, reverse)`. Public vs. private names; different signatures. Likely the public method is object-centric (compute target pose internally) and the private one is pose-centric (target pose supplied). Worth formalising.
- `place_pose_sequence` reads `manipulator.tool_frame.child_kinematic_structure_entities[0]` — silently assumes exactly one held body. Multi-body grasps (tray with items, two cubes) fail here. Same caveat is logged on the [[sdt.robots.abstract_robot.Manipulator]] page.
- `manipulation_offset` controls **both** the pre-grasp standoff and the lift height. There is no way to set them independently. For tall fragile objects you might want a small standoff but a large lift, which is impossible without subclassing or post-modifying poses.
- `__hash__ = id(self)` means two `GraspDescription` instances with identical fields are non-equal — relevant when caching grasp-enumeration results.

## Provenance

- `pycram/src/pycram/datastructures/grasp.py:1-433` — `GraspDescription`, `PreferredGraspAlignment`, `GraspPose`, all derivation methods.
- `pycram/src/pycram/datastructures/rotations.py:1-37` — `Rotations` class with `SIDE_ROTATIONS`, `VERTICAL_ROTATIONS`, `HORIZONTAL_ROTATIONS` quaternion tables.
- `pycram/src/pycram/datastructures/enums.py:63-167` — `Arms`, `AxisIdentifier`, `ApproachDirection`, `VerticalAlignment` enums.
