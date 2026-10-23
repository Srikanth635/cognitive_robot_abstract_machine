# Isaac Sim backend

Running the physics viewer's scenes and plans under NVIDIA Isaac Sim (PhysX) instead of
MuJoCo, the same way they run under MuJoCo today: in the same Python process, stepped in
lockstep with the controller.

## What exists

`semantic_digital_twin/adapters/multi_sim.py` separates a simulator-independent layer from
its only implementation, MuJoCo:

| Layer | Generic | MuJoCo |
| --- | --- | --- |
| Write the world to a scene file | `MultiSimBuilder` | `MujocoBuilder` (MJCF) |
| Load and step the scene | `physics_simulators.BaseSimulator` | `MujocoSimulator` |
| Keep world and simulator in step | `MultiSimSynchronizer` | `MujocoSynchronizer` |
| Tie the three together | `MultiSim` | `MujocoSim` |

The stepped API that coraplex and giskard use is not part of the generic layer:
`start_stepped_simulation` and `step_simulation` exist only on `MujocoSim`, and
`SteppedSimulationPacer.wait_for_standstill` reads servo positions through
`MujocoSimulator.get_actuators_lengths`. coraplex (`PhysicallySimulatedRobot`,
`GiskardExecutable`) and giskard (`SteppedSimulationPacer`) are typed to `MujocoSim`.

## Plan

1. **A stepped-simulation interface.** Declare what a controller needs from a physically
   simulated world (start stepped, step by a duration, read the servo positions) as an
   interface `MujocoSim` implements, and type coraplex and giskard against it. Behaviour
   under MuJoCo is unchanged; this is testable now.
2. **USD builder.** `IsaacBuilder(MultiSimBuilder)` writes the world as a USD stage with
   `pxr` (UsdGeom, UsdPhysics, PhysxSchema). Testable with `usd-core`, without Isaac.
3. **Isaac simulator.** `IsaacSimulator(BaseSimulator)` in `physics_simulators` loads the
   stage into Isaac Sim and steps PhysX; needs Isaac Sim (Python 3.12, 6.x).
4. **Isaac synchronizer and `IsaacSim(MultiSim)`**, implementing the interface of step 1.
5. **Physics viewer under Isaac**: the scenes and the action plans are reused; the window,
   pacing and simulation settings get an Isaac version. The grasps are tuned anew for PhysX.

## Mapping the world to USD

| World | USD / PhysX |
| --- | --- |
| `Body` | `Xform` with `UsdPhysics.RigidBodyAPI` (when it can move) and `MassAPI` from its inertial |
| `Box`, `Sphere`, `Cylinder` | `UsdGeom.Cube` (scaled), `Sphere`, `Cylinder` with `CollisionAPI` |
| `Mesh` | `UsdGeom.Mesh` with `CollisionAPI` and `MeshCollisionAPI` (convex hull, or decomposition for concave furniture) |
| `RevoluteConnection`, `PrismaticConnection` | `UsdPhysics.RevoluteJoint`, `PrismaticJoint`, limits from the degree of freedom |
| `FixedConnection` | `UsdPhysics.FixedJoint`, or the child under its parent's rigid body |
| `Connection6DoF` | a free rigid body, no joint |
| A degree of freedom shared by several joints (mimic) | `PhysxSchema.PhysxMimicJointAPI` |
| `PositionServo` gains | `UsdPhysics.DriveAPI`: stiffness, damping, max force |
| Robot | `UsdPhysics.ArticulationRootAPI` on its root |
| `ContactParameters` friction | `UsdPhysics.MaterialAPI` |

USD angular joints are in degrees: limits, positions, and the stiffness and damping of an
angular drive are per degree, so radians are converted at the boundary.

## Status

- **Step 1 done.** `SteppedSimulation` (in `multi_sim.py`) declares
  `start_stepped_simulation`, `step_simulation` and `servo_positions`; `MujocoSim` implements
  it, and coraplex (`PhysicallySimulatedRobot`, `GiskardExecutable`) and giskard
  (`SteppedSimulationPacer`) depend on it instead of `MujocoSim`.
- **Step 2 done for what the demos need.** `UsdBuilder` (`adapters/usd/builder.py`) writes
  bodies, boxes, spheres, cylinders and meshes, revolute, prismatic and fixed joints, free
  bodies, position and velocity servos as drives, and a servoed wheeled drive as a slide, a
  slide and a hinge through two light bodies of its own. The PR2 at the counter builds in
  under three seconds; written as `.usdc`, the scene is about 5 MB.
- **Not yet written** (needs `PhysxSchema`, which comes with Isaac Sim):
  - a degree of freedom shared by several joints (the PR2 gripper's fingers) is written
    as independent joints, only the one named after it driven; it needs PhysX's mimic
    joint API;
  - robots are written as joints between rigid bodies, not as articulations
    (`ArticulationRootAPI`);
  - joint armature, friction and damping, and contact stiffness and friction.

## Open points

- The Python environment Isaac Sim 6.x is installed in, and how the repository's packages
  are installed next to it (Isaac Sim pins its own numpy, torch and USD).
- `usd-core` (in semantic_digital_twin's `datasets` extra) for builder tests without Isaac;
  it must not be installed next to Isaac Sim, which ships its own USD.
- Tests that need Isaac Sim are skipped when it is not installed, so CI runs without it.
