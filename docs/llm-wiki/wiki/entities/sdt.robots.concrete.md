---
id: sdt.robots.concrete
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/robots/tiago.py
    lines: [1, 120]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: semantic_digital_twin/src/semantic_digital_twin/robots/panda.py
    lines: [1, 80]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.robots.abstract_robot.AbstractRobot
  - sdt.robots.abstract_robot.Manipulator
  - sdt.world.World
  - sdt.collision_checking
  - sdt.adapters
used_by: []
status: stable
tags: [sdt, robot, tiago, panda, mobile-manipulator, arm, gripper, concrete]
last_ingest: 2026-05-19
---

_Bundled overview of concrete `AbstractRobot` subclasses; uses `Tiago` (mobile manipulator) and `Panda` (fixed-arm) as representative examples of the two main robot archetypes._

## Robot inventory (20 models)

`armar`, `armar7`, `boxy`, `donbot`, `hsrb`, `icub3`, `justin`, `kevin`, `minimal_robot`, `panda`, `pr2`, `stretch`, `tiago`, `tiago_mujoco`, `tracy`, `turtlebot`, `unitree_g1`, `ur5`, `ur5e_controlled`.

## Construction pattern

All concrete robots follow the same two-step pattern:

1. **`from_world(world)`** (or `_init_empty_robot(world)` + `_setup_semantic_annotations()` for two-phase init): finds bodies by name in the world, wires `Arm` / `ParallelGripper` / `Finger` / `Camera` / `Torso` / `Base` chains.
2. Everything happens inside `world.modify_world()` to ensure atomicity.

```python
with world.modify_world():
    robot = Tiago._init_empty_robot(world)
    robot._setup_semantic_annotations()
```

## `Tiago` — mobile manipulator

Extends `AbstractRobot, SpecifiesLeftRightArm, HasNeck`.

- Two arms (left + right), each with `ParallelGripper(Finger × 2)`.
- Arm root: `torso_lift_link`; arm tip: `arm_{side}_tool_link`.
- Gripper `front_facing_axis = Vector3(1, 0, 0)`, `front_facing_orientation = Quaternion(0,0,0,1)`.
- Torso: `PrismaticConnection`-based lift joint.
- Base: `OmniDrive` connection; `full_body_controlled=False` (base not included in whole-body QP by default).
- Mixins supply `SpecifiesLeftRightArm.left_arm` / `right_arm` properties and `HasNeck.neck`.

## `Panda` — fixed-arm manipulator

Extends `AbstractRobot, HasArms`.

- Single arm; root `"panda"` body, tip `"link7"`.
- `ParallelGripper` with `front_facing_axis = Vector3(0, 0, 1)`.
- `from_world(world)` factory; arm park state loaded via `JointState.from_mapping(...)`.

## `robot_mixins.py` traits

| Mixin | Provides |
|---|---|
| `SpecifiesLeftRightArm` | `left_arm`, `right_arm` properties by convention |
| `HasArms` | `arms: List[Arm]` + `add_arm(arm)` |
| `HasNeck` | `neck: Neck` field |

## Collision setup

Each robot file configures its collision matrix on construction:
- `AvoidSelfCollisions` — avoid all self-collision pairs.
- `AvoidExternalCollisions` — avoid collisions with the environment.
- `AllowCollisionForAdjacentPairs` — suppress adjacent-link collisions (always in contact).
- `MaxAvoidedCollisionsOverride` — per-robot cap on how many collision pairs are tracked.

## Related

- **Uses:** [[sdt.robots.abstract_robot.AbstractRobot]], [[sdt.world.World]]
- **Used by:** [[pycram.datastructures.Context]] (`from_world()` discovers the robot annotation)

## Provenance

- `tiago.py:1-120` — `Tiago._init_empty_robot`, `_setup_semantic_annotations`, arm/gripper wiring.
- `panda.py:1-80` — `Panda.from_world`, arm wiring, park state.
