# Isaac Sim backend

Running the physics viewer's scenes and plans under NVIDIA Isaac Sim (PhysX) instead of
MuJoCo, the same way they run under MuJoCo today: in the same Python process, stepped in
lockstep with the controller.

## What exists

`semantic_digital_twin/adapters/multi_sim.py` separates a simulator-independent layer from
its only implementation, MuJoCo:

| Layer | Generic | MuJoCo | Isaac Sim |
| --- | --- | --- | --- |
| Write the world to a scene file | `MultiSimBuilder` | `MujocoBuilder` (MJCF) | `UsdBuilder` (`adapters/usd/builder.py`) |
| Load and step the scene | `physics_simulators.BaseSimulator` | `physics_simulators.mujoco_simulator.MujocoSimulator` | `physics_simulators.isaac_simulator.IsaacSimulator` |
| Keep world and simulator in step | `MultiSimSynchronizer` | `MujocoSynchronizer` | `IsaacSynchronizer` (`adapters/isaac/synchronizer.py`) |
| Tie the three together | `MultiSim` | `MujocoSim` | `IsaacSim` (`adapters/isaac/simulation.py`) |

`physics_simulators.isaac_application.IsaacApplication` holds the one Isaac Sim
application a process runs; every other Isaac module needs it started before it is
imported. `IsaacSimulator` reads and writes joints and bodies with plain methods rather
than `MujocoSimulator`'s `@simulator_callback`s, since it is only stepped in lockstep. The
builder and synchronizer reuse `MujocoConnectionJoints` and `MujocoGeom`, whose drive axes,
servo controls and contact categories are not MuJoCo's alone.

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
3. **Isaac simulator.** `IsaacSimulator(BaseSimulator)` in `physics_simulators/isaac_simulator.py`
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
  articulations. The fixed joint to the static body carries `ArticulationRootAPI`, which
  gives the articulation a fixed base; on the anchor body instead, the base floated, held
  only by that joint, and with 10 g anchors the fancy doors pressed against their cabinet
  flipped or went NaN and the PR2 blew up when it turned. Written as separate rigid bodies,
  the PR2 drifted 4 m in 3 s.
  Since the first joint hangs from the anchor rather than from the static body, the
  articulation's topmost body is kept from colliding with that body by
  `UsdPhysics.FilteredPairsAPI`, as a joint keeps the bodies it joins apart; without it,
  Tracy's left shoulder rubbed on its mount and stopped 0.45 rad short of its set point.
- **Joints sharing a degree of freedom** (the PR2 gripper's fingers) mimic the joint named
  after it through `PhysxMimicJointAPI` (deprecated in Isaac Sim 6.0 for the Newton
  schema's mimic API, still read). Commanding only the right gripper's primary finger to
  0.436 rad brings all four finger joints to 0.437 rad. PhysX reads the gearing in the
  stage's units of both joints, so a sliding joint following a turning one, as the fancy
  doors' frames do, is geared in metres per degree.
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
- **The counter scene's furniture**, with fixed-base articulations: none of its 38 joints
  moves more than 0.1 mm in 3 s, as under MuJoCo, the fancy doors included, with no
  non-finite PhysX result. Dry friction reaches PhysX in newton metres: a hinge holds a flap
  whose weight turns it with half its friction, and lets one fall that turns it with twice.
- **Steps 3 and 4 done**: `IsaacApplication` (the one `SimulationApp` of the process) and
  `IsaacSimulator` in `physics_simulators`, `IsaacSynchronizer` and `IsaacSim(MultiSim,
  SteppedSimulation)` in `adapters/isaac`. The builder tags each prim with the id of its body and each driven
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
  - fixed-base articulations: with a floating base, turning the base blew the PR2 up.
- **Every physics viewer demo through `IsaacSim`**, headless, each in its scene, after 1 s
  of settling as the viewer does, against `MujocoSim` on the same plan: all 24 finish under
  both, with no non-finite value, the same simulated time within 0.4 s, the base where
  MuJoCo puts it within 3 mm, and the same mean servo error within 0.003.

  | Demo | Isaac Sim 6.0 | MuJoCo |
  | --- | --- | --- |
  | `raise_torso` | torso 0.2897 of 0.2904 | 0.2900 |
  | `pick_up_milk`, `move_and_pick_up_milk` | milk lifted 9.3, 9.6 cm | 9.5, 9.7 cm |
  | `place_milk`, `pick_and_place_milk`, `move_and_place_milk`, `transport_milk` | milk moved 19.8 to 19.9 cm along the counter | 18.8 to 19.4 cm |
  | `open_drawer`, `move_and_open_drawer` | drawer open 0.441 m | 0.447, 0.448 m |
  | `close_drawer` | drawer shut | shut |
  | `mix_in_bowl` | the whisk pushes the bowl 13.5 cm | 13.0 cm |
  | `pick_up_cube` (Tracy) | cube lifted 8.4 cm | 8.5 cm |

  `open_grippers`, `close_grippers`, `park_arms`, `move_arm`, `look_at_milk`,
  `face_the_cup`, `face_and_look_at_the_cup`, `go_to_the_counter`, `reach_for_milk`,
  `grasp_milk`, `pour_into_bowl` and `wipe_counter` move only the robot, and its joints and
  base end where MuJoCo's do.
- **Not yet written:** contact stiffness and friction; rendering a window.

## Checking a stage in Isaac Sim

Isaac Sim 6.0 is installed standalone in `~/isaacsim`, with its own Python 3.12. A stage
written in `new-cram-env` is checked with Isaac's Python, headless, started from a clean
environment so that neither ROS nor `new-cram-env` mixes into it:

```bash
env -i HOME=$HOME USER=$USER PATH=/usr/bin:/bin LANG=C.UTF-8 ~/isaacsim/python.sh SCRIPT.py
```

The first start builds Isaac's caches and takes a few minutes; later starts take seconds.

## Watching and recording a run

`isaac_viewer/run.sh` runs a physics viewer scene's action under Isaac Sim with this
checkout's packages, in `isaac-cram-env`:

```bash
isaac_viewer/run.sh view pr2_at_the_counter go_to_the_counter
isaac_viewer/run.sh record pr2_at_the_counter pick_up_milk recordings/pick_up_milk
```

Both light the stage, which the builder writes without lights, and put up the cameras of
`isaac_viewer/scene_cameras.py`: three standing in the room, placed relative to where the
robot starts (behind it to the left and to the right, and beyond the counter facing it),
and one on the robot's default camera, looking where it looks with its field of view.
`view` shows one of them in a window at real-time speed (the window's camera menu switches
between them); `record` renders them headless with Replicator, writes one H.264 video per
camera, and `player.html`, a page that plays the four in step, one at a time or side by
side, at 1, 2, 4 or 8 times speed. Recording `pick_up_milk` from the four cameras at
1280 by 720 and 25 frames a second takes about a minute on an RTX 4070.

## Open points

- CRAM and Isaac Sim run in one process in `isaac-cram-env`, a copy of `new-cram-env` with
  `isaacsim[all,extscache]==6.0.0.1` pip-installed; `typing_extensions` is raised to 4.16
  over Isaac Sim's pin of 4.12, which CRAM needs.
- The physics viewer under Isaac (step 5).
- `usd-core` (in semantic_digital_twin's `datasets` extra) for builder tests without Isaac;
  it must not be installed next to Isaac Sim, which ships its own USD.
- Tests that need Isaac Sim are skipped when it is not installed, so CI runs without it.
