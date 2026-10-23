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
3. **Isaac simulator.** `IsaacSimulator(BaseSimulator)` in `semantic_digital_twin/adapters/isaac`
   loads the stage into Isaac Sim and steps PhysX; needs Isaac Sim (Python 3.12, 6.x).
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
| `GravityCompensation` | `PhysxForceAPI`: an upward acceleration of that fraction of gravity, in the world's frame |
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
- **Articulations.** Moving bodies joined by joints are one articulation, hanging from an
  anchor fixed to the static body they stand on, with self-collisions off; a body on a
  single joint is one too, since Isaac Sim reads and drives joints only through
  articulations. Written as separate rigid bodies instead, the PR2 drifted 4 m in 3 s under Isaac
  Sim 6.0 (its drive's light links against a heavy base); as an articulation its driven
  joints hold within 0.03 of the world's values, and the objects stay on the counter.
- **Joints sharing a degree of freedom** (the PR2 gripper's fingers) mimic the joint named
  after it through `PhysxMimicJointAPI` (deprecated in Isaac Sim 6.0 for the Newton
  schema's mimic API, still read). Commanding only the right gripper's primary finger to
  0.436 rad brings all four finger joints to 0.437 rad.
- **Joint friction and contact categories.** A joint's dry friction, damping and armature
  become `PhysxJointAxisAPI` under the drive axis name (`angular`, `linear`; PhysX ignores
  the per-axis names), and its joint becomes part of an articulation, where PhysX applies
  them. The MuJoCo contact categories of the shapes become collision groups.
- **Bodies with nothing to weigh.** A moving body with neither shapes nor an inertial gets
  the minimum mass and inertia the MuJoCo builder bounds bodies by (1 g, 1e-6 kg m²); 24
  bodies of the counter scene were massless to PhysX before, among them the PR2's camera
  frames.
- **Isaac Sim 6.0's own API**, checked in `isaac-cram-env`: an `Articulation` view of the
  PR2's root lists its 48 joints under the builder's joint names, reads positions and sets
  targets in radians and metres, and `SimulationManager.step` advances 2 s of the counter
  scene in about 4 s on the CPU.
- **Open in the counter scene:**
  - the apartment's two-joint "fancy" doors flip over, or become NaN, within 0.3 s under
    PhysX, whether the coupling between their joints is hard, compliant or removed, and
    whether their first joint has friction, limits or a heavier link; the robot and the
    objects on the counter are unaffected;
  - the furniture friction the physics viewer's scenes give (1.0) holds a drawer under
    MuJoCo but lets `cabinet8_drawer_bottom` creep 13 cm in 3 s under PhysX; a thousand
    times that holds it.
- **Steps 3 and 4 done** in `adapters/isaac`: `IsaacApplication` (the one `SimulationApp`
  of the process), `IsaacSimulator`, `IsaacSynchronizer` and `IsaacSim(MultiSim,
  SteppedSimulation)`. The builder tags each prim with the id of its body and each driven
  joint with the degrees of freedom it measures and is commanded through, which is how the
  synchronizer finds them. As under MuJoCo, servoed joints keep the world's set point, and
  joints without a servo, a drive's odometry and free bodies are read back; the world puts
  a drive's odometry only when the simulation starts. PhysX starts articulations with every
  joint at zero, so the synchronizer puts every joint and free body where the world has it.
- **What made the PR2 behave as under MuJoCo:**
  - gravity compensation: without it the torso servo stopped 17 mm below its set point;
  - a step of 1/200 s, which divides giskard's 10 ms tick; at 1/240 s a tick advanced 2
    steps, 8.3 ms, and the base controller overshot and diverged. `step_simulation` refuses
    a duration that is not a whole number of steps;
  - drive links and anchors of 1 kg: at 10 g, against the PR2's 271 kg, turning the base
    blew it up.
- **Coraplex plans through `IsaacSim`**, against `MujocoSim` on the same plan:

  | Plan | Isaac Sim 6.0 | MuJoCo |
  | --- | --- | --- |
  | hold still, 3 s | mean servo error 0.0030 | 0.0030 |
  | `raise_torso` | torso 0.2887 of 0.2904, 22.2 s simulated | 0.2900, 22.1 s |
  | `go_to_the_counter` | base at x 1.852 of 1.85, 2.8 s simulated | 1.852, 2.8 s |
  | `pick_up_milk` | milk lifted 9.1 cm | 9.5 cm |

  The objects on the counter move less than 7 mm while the robot holds still.
- **Not yet written:** contact stiffness and friction; rendering a window.

## Checking a stage in Isaac Sim

Isaac Sim 6.0 is installed standalone in `~/isaacsim`, with its own Python 3.12. A stage
written in `new-cram-env` is checked with Isaac's Python, headless, started from a clean
environment so that neither ROS nor `new-cram-env` mixes into it:

```bash
env -i HOME=$HOME USER=$USER PATH=/usr/bin:/bin LANG=C.UTF-8 ~/isaacsim/python.sh SCRIPT.py
```

The first start builds Isaac's caches and takes a few minutes; later starts take seconds.

## Open points

- CRAM and Isaac Sim run in one process in `isaac-cram-env`, a copy of `new-cram-env` with
  `isaacsim[all,extscache]==6.0.0.1` pip-installed; `typing_extensions` is raised to 4.16
  over Isaac Sim's pin of 4.12, which CRAM needs.
- The fancy doors and the drawer creep above.
- The physics viewer under Isaac (step 5).
- `usd-core` (in semantic_digital_twin's `datasets` extra) for builder tests without Isaac;
  it must not be installed next to Isaac Sim, which ships its own USD.
- Tests that need Isaac Sim are skipped when it is not installed, so CI runs without it.
