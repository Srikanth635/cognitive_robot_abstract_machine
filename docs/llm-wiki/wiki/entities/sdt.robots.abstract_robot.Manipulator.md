---
id: sdt.robots.abstract_robot.Manipulator
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/robots/abstract_robot.py
    lines: [214, 329]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.world_entity.Body
  - sdt.spatial_types.spatial_types
used_by:
  - sdt.robots.abstract_robot.AbstractRobot
  - sdt.robots.concrete
  - pycram.datastructures.grasp.GraspDescription
status: stable
tags: [sdt, robot, manipulator, gripper, parallel-gripper, humanoid-gripper, finger, tool-frame]
last_ingest: 2026-05-19
---

_Bundled page: end-effector semantic annotations — `Manipulator` (abstract base), `ParallelGripper`, `HumanoidGripper`, and the related `Finger` kinematic chain._

## Purpose

A `Manipulator` is the SDT semantic annotation that wraps a robot's end-effector. It is **not** a kinematic chain itself — it sits adjacent to one (the `Arm` chain whose tip is the manipulator's mount), and carries the geometric metadata needed to grasp objects: the tool frame, the front-facing orientation, and the forward-facing axis. Every action involving "the gripper" — `MoveGripperMotion`, `GraspDescription`, `is_body_in_gripper` — routes through a `Manipulator` reference.

## When to use

- **As a grasp target** — `GraspDescription` requires a `Manipulator` to compute the grasp orientation (uses `front_facing_orientation`) and to identify the held body (via `tool_frame.child_kinematic_structure_entities[0]`).
- **To enumerate the robot's end-effectors** — `AbstractRobot.manipulators` returns a `List[Manipulator]`. Each has a parent `Arm` reachable via the manipulator's mount body's kinematic chain.
- **To resolve TCP frame** — `manipulator.tool_frame` is the `Body` whose pose is what motion goals like `CartesianPose(root, tip=manipulator.tool_frame, goal)` target.
- **Not for joint commands** — opening/closing the gripper goes through `JointState`s attached to the parent `Arm` (`arm.get_joint_state_by_type(GripperState.OPEN)`), not through Manipulator directly.

## `Manipulator` — abstract base

```python
@dataclass
class Manipulator(SemanticRobotAnnotation, ABC):
    tool_frame: Body                          # kw_only — the TCP link
    front_facing_orientation: Quaternion      # kw_only — orientation of forward axis
    front_facing_axis: Vector3                # kw_only — axis of forward motion
```

Inherits `SemanticRobotAnnotation`, which provides `_world`, `_robot` (back-reference), `joint_states` (fixed pre-defined joint configurations like OPEN/CLOSE), `add_joint_state(js)`, and `get_joint_state_by_type(state_type)`.

`assign_to_robot(robot)` is idempotent and raises if reassigned to a different robot.

`__hash__` = `hash((name, root, tool_frame))` — manipulators with the same name+root+tool_frame are equal.

### Key fields

| Field | Type | Purpose |
|---|---|---|
| `tool_frame` | `Body` | The TCP link. Used by motion goals as `tip` and by GraspDescription to identify the held body. |
| `front_facing_orientation` | `Quaternion` | Orientation of the manipulator's natural forward-facing pose. Used as the fourth term in `GraspDescription.grasp_orientation()`. |
| `front_facing_axis` | `Vector3` | Unit vector of the front-facing direction (typically X). Used by collision and approach-direction reasoning. |
| `joint_states` | `List[JointState]` | Inherited from `SemanticRobotAnnotation`. Holds named configurations (OPEN/CLOSE/MEDIUM). |
| `_robot` | `AbstractRobot` | Back-reference to the owning robot; `None` until `assign_to_robot`. |

## `ParallelGripper(Manipulator)` — two-fingered

```python
@dataclass
class ParallelGripper(Manipulator):
    finger: Finger     # one finger
    thumb: Finger      # the thumb — always touches object during grasp
```

The `thumb` is a distinguished finger guaranteed to contact the object during a grasp; this is how the geometry validates "stable grasp" downstream. Both `finger` and `thumb` are `Finger` (a `KinematicChain`) — they have their own root/tip and may carry sensors.

Used by Tiago, Panda, and most parallel-jaw robots — see [[sdt.robots.concrete]].

## `HumanoidGripper(Manipulator)` — multi-fingered

```python
@dataclass
class HumanoidGripper(Manipulator):
    fingers: List[Finger]    # arbitrary number of fingers
    thumb: Finger            # privileged finger
```

Same `thumb` semantics as ParallelGripper — the thumb is the one guaranteed-contact finger. The other fingers are just `fingers` (a list).

## `Finger(KinematicChain)`

A `Finger` is a `KinematicChain` (root, tip) that may contain sensors (e.g. tactile pads). Its tip is the fingertip. Fingers do **not** carry a `manipulator` field themselves — they're owned by the gripper, not by an arm.

```python
@dataclass
class Finger(KinematicChain):
    # inherits: root, tip, manipulator (always None here), sensors, joint_states
    ...
```

## Subclass overview

```
SemanticRobotAnnotation
└── Manipulator              ← this page
    ├── ParallelGripper       (Tiago, Panda, PR2, …)
    └── HumanoidGripper       (HSR-B humanoid variants, Atlas, …)

SemanticRobotAnnotation
└── KinematicChain
    └── Finger                ← this page (related to manipulators)
```

## How `GraspDescription` consumes a Manipulator

1. `manipulator.front_facing_orientation` is the fourth quaternion factor in `grasp_orientation()`:
   ```
   SIDE_ROTATIONS × VERTICAL_ROTATIONS × HORIZONTAL_ROTATIONS × manipulator.front_facing_orientation
   ```
2. `manipulator._world` is read to access the kinematic structure for FK transforms.
3. `manipulator._robot.root.global_pose` is read for the robot-to-object vector during `calculate_grasp_descriptions`.
4. `manipulator.tool_frame.child_kinematic_structure_entities[0]` is read in `place_pose_sequence` to identify the held body — assumes exactly one held body per manipulator.

## How `MoveGripperMotion` uses a Manipulator (indirectly)

`MoveGripperMotion` takes an `Arms` enum, resolves the corresponding `Arm` from the robot, and calls `arm.get_joint_state_by_type(motion)` to get the gripper open/close `JointState`. The `Manipulator` itself isn't passed — the open/close states are attached to the `Arm`, not the `Manipulator`. The Manipulator participates indirectly: the `Arm.manipulator` field is what makes the chain an "arm with gripper" versus a sensor-only chain.

## Per-robot wiring

Concrete robots wire up Manipulators in `_setup_semantic_annotations`. For Tiago this looks roughly like:

```python
right_finger = Finger(name=..., root=right_finger_root, tip=right_fingertip)
right_thumb  = Finger(name=..., root=right_thumb_root,  tip=right_thumbtip)
right_gripper = ParallelGripper(
    name=...,
    tool_frame=right_tool_link,
    front_facing_orientation=Quaternion(0, 0, 0, 1),  # robot-specific
    front_facing_axis=Vector3(1, 0, 0),
    finger=right_finger, thumb=right_thumb,
)
right_arm = Arm(name=..., root=shoulder, tip=right_tool_link, manipulator=right_gripper)
robot.add_kinematic_chain(right_arm)   # → robot.manipulator_chains + robot.manipulators
```

This is why `robot.manipulators` and the arm's `arm.manipulator` point to the same `Manipulator` instance — single source of truth.

## Related

- **Uses:** [[sdt.world_description.world_entity.Body]], [[sdt.spatial_types.spatial_types]]
- **Used by:** [[sdt.robots.abstract_robot.AbstractRobot]] (owns the list), [[sdt.robots.concrete]] (wires per-robot), [[pycram.datastructures.grasp.GraspDescription]] (primary consumer), [[pycram.robot_plans.actions.core.PickUpAction]], [[pycram.robot_plans.actions.core.PlaceAction]], [[pycram.robot_plans.motions.gripper]]

## Open questions

- The `place_pose_sequence` method in `GraspDescription` reads `manipulator.tool_frame.child_kinematic_structure_entities[0]` to find the held body — silently assumes exactly one child. Multi-body grasps (two cubes held with one gripper, or a tray with items on top) are unrepresentable. Whether this is a hard architectural limit or a TODO is unclear.
- The `front_facing_orientation` quaternion is set per-robot in concrete subclasses but its sign convention (which direction does "front" face in the URDF) is not codified anywhere. Two robots can legitimately disagree on what "front" means; grasps written for one may invert when ported to another.

## Provenance

- `semantic_digital_twin/src/semantic_digital_twin/robots/abstract_robot.py:214-329` at commit `0528d8cf3` — `Manipulator` abstract base, `ParallelGripper`, `HumanoidGripper`, and the related `Finger` class.
