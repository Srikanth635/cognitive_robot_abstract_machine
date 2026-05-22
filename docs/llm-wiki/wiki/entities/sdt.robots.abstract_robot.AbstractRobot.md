---
id: sdt.robots.abstract_robot.AbstractRobot
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/robots/abstract_robot.py
    lines: [1, 676]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world.World
  - sdt.datastructures.joint_state.JointState
used_by:
  - pycram.datastructures.Context
  - bridge.pycram-sdt
  - giskardpy.model.world_config
  - pycram.perception.PerceptionQuery
  - pycram.locations.locations.AccessingLocation
  - pycram.locations.locations.GiskardLocation
  - pycram.pose_validator
  - sdt.reasoning.predicates
  - sdt.reasoning.robot_predicates
  - pycram.querying.predicates
  - pycram.alternative_motion_mapping.AlternativeMotion
  - pycram.datastructures.grasp.GraspDescription
  - pycram.view_manager.ViewManager
  - pycram.robot_plans.actions.core.robot_body
  - pycram.robot_plans.motions.robot_body
  - sdt.robots.abstract_robot.Manipulator
  - sdt.robots.concrete
status: stable
tags: [sdt, robot, abstract, kinematic-chain, manipulator, semantic-annotation, arm]
last_ingest: 2026-05-19
---

_Abstract base for all robots in SDT: composed of `SemanticRobotAnnotation` chains (arms, torso, base, grippers, cameras) discovered from the World._

## Purpose

`AbstractRobot` is the entry point for all robot-specific reasoning in SDT. It wraps a set of
`SemanticRobotAnnotation` instances — typed kinematic chains (`Arm`, `Torso`, `Base`) and
end-effectors (`Manipulator`: `ParallelGripper`, `HumanoidGripper`) — that are discovered from
the `World`'s kinematic structure via `from_world()`.

pycram accesses the robot through `Context.robot`. Properties like `robot.full_body_controlled`
and `robot.drive` are used by motion designators to select the correct motion frame and root link.

## Key attributes

| Name | Kind | Notes |
|---|---|---|
| `torso` | `Torso` | Torso kinematic chain. |
| `base` | `Base` | Mobile base kinematic chain. |
| `manipulators` | `List[Manipulator]` | All arm+gripper pairs. |
| `sensors` | `List[Sensor]` | Cameras and other sensors. |
| `manipulator_chains` | `List[KinematicChain]` | All chains that are manipulators. |
| `full_body_controlled` | `bool` | True for mobile manipulators where the whole-body is jointly controlled. Controls `MoveToolCenterPointMotion` root frame selection: `world.root` if True, `robot.root` if False. |

## Key methods

| Name | Notes |
|---|---|
| `from_world(cls, world)` | Factory classmethod. Calls `_init_empty_robot`, `_setup_semantic_annotations`, `_setup_collision_rules`, `_setup_velocity_limits`, `_setup_hardware_interfaces`, `_setup_joint_states`. Concrete robot classes implement one or more of these. |
| `drive` | Property. Returns `OmniDrive` or `DifferentialDrive` if the robot's root body has a drive annotation; else `None`. |

## SemanticRobotAnnotation hierarchy

```
SemanticRobotAnnotation
├── KinematicChain
│   ├── Arm
│   ├── Neck
│   ├── Torso
│   ├── Base                 (with `main_axis: Vector3` and `bounding_box` property)
│   └── Finger
├── Manipulator              → see [[sdt.robots.abstract_robot.Manipulator]]
│   ├── ParallelGripper      (finger + thumb)
│   └── HumanoidGripper      (fingers list + thumb)
└── Sensor
    └── Camera               (with FieldOfView, default_camera flag)
```

`KinematicChain.assign_to_robot(robot)` is idempotent — calling it twice with the same robot
is safe; calling it with a different robot raises.

## Key collections (for grounding)

When grounding agents enumerate the robot's resources they read these `AbstractRobot` fields:

| Collection | Type | Use |
|---|---|---|
| `manipulators` | `List[Manipulator]` | Every gripper. Indexed by appending — order is wiring-dependent. |
| `manipulator_chains` | `List[KinematicChain]` | Every chain whose `manipulator` is non-None — i.e. every arm-with-gripper. |
| `sensor_chains` | `List[KinematicChain]` | Every chain whose `sensors` list is non-empty — typically neck-with-camera. |
| `sensors` | `List[Sensor]` | All sensors (flat). |
| `controlled_connections` | `Set[ActiveConnection]` | Intersection of world's controlled connections and this robot's connections. The QP-actuable joints. |
| `degrees_of_freedom_with_hardware_interface` | `List[DegreeOfFreedom]` | DOFs that have a real hardware actuator (vs simulation-only). |

The mapping `Arms.LEFT/RIGHT/BOTH` → concrete `Arm` and `Manipulator` is **not** in `AbstractRobot` itself — it's resolved per-robot in concrete subclasses (see [[sdt.robots.concrete]]). For grounding to look up "the right gripper", it must go via the concrete robot's API (e.g. `tiago.right_arm.manipulator`).

## Relation to the legacy `RobotDescription`

The pycram codebase also contains `RobotDescription.current_robot_description` (in `pycram.robot_description` — not ingested) used by legacy actions: `pycram.robot_plans.actions.core.robot_body`, `pycram.robot_plans.actions.composite.tool_based`, and `pycram.robot_plans.actions.composite.transporting`. This is the pre-SDT robot description that `AbstractRobot` replaces.

Treat `RobotDescription` as **deprecated**:
- All new code uses `Context.robot: AbstractRobot`.
- Actions that still reference `RobotDescription.current_robot_description` are flagged as legacy/non-functional in their own wiki pages (e.g. `EfficientTransportAction`, `MixingAction`, `PouringAction`, `CuttingAction`).
- No `RobotDescription` page exists in this wiki by design — the modern replacement is `AbstractRobot`. If you need to understand a legacy action, read the action's wiki page; the `RobotDescription` calls inside it are not load-bearing for the new pipeline.

## Related

- World source: [[sdt.world.World]]
- Manipulator detail: [[sdt.robots.abstract_robot.Manipulator]] (`Manipulator`, `ParallelGripper`, `HumanoidGripper`, `Finger`)
- Concrete robots: [[sdt.robots.concrete]] (Tiago, Panda, PR2, HSR-B, …)
- Context carrier: [[pycram.datastructures.Context]]
- Bridge: [[bridge.pycram-sdt]]
- Consuming motion: [[pycram.robot_plans.motions.gripper]]

## Open questions

- Concrete `AbstractRobot` subclasses (e.g. `PR2Robot`, `HSRBRobot`) and their per-robot wiring of arms/manipulators are documented in [[sdt.robots.concrete]] as a bundled overview, but the full mapping `Arms.LEFT/RIGHT/BOTH` → concrete `Arm` per robot is implicit in robot-specific naming conventions, not in a shared API.
- `_setup_velocity_limits` defaults to 1.0 rad/s for every 1DOF connection — this is the only velocity-limit knob exposed on the abstract base. Per-robot limits live in concrete subclasses.
- The legacy `pycram.robot_description.RobotDescription` is still imported by `robot_body.py`, `tool_based.py`, `transporting.py`. Whether these actions get migrated to `AbstractRobot` or are formally deprecated has not been declared.

## Provenance

- `semantic_digital_twin/src/semantic_digital_twin/robots/abstract_robot.py:1-676` at commit
  `0528d8cf3` — full module: `AbstractRobot`, `SemanticRobotAnnotation`, `KinematicChain`,
  `Manipulator`, `Sensor`, `Arm`, `ParallelGripper`, `HumanoidGripper`, `Camera`.
