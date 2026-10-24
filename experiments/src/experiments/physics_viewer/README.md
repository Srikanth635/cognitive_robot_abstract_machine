# Physics viewer

Puts a scene under MuJoCo physics in a window of its own, to see how it behaves. The demos
perform plans in a *kinematic* simulation; here the bodies fall, collide and are held by
servos.

```bash
python -m experiments.physics_viewer SCENE [--action ACTION] [--speed FACTOR] [--duration SECONDS] [--headless]
```

| Scene | What it shows |
| --- | --- |
| `floor` | Nothing but the ground |
| `cube` | A cube dropped onto the ground |
| `cups` | A set of cups |
| `apartment` | The apartment, with its walls, floor and furniture |
| `pr2_in_apartment` | The PR2 in the apartment, where the demos have it stand |
| `pr2_at_the_counter` | The PR2 at the kitchen counter, with a milk carton, a cereal box, a bowl and a cup on it, under stiff contacts |
| `pr2_holding_a_cup` | The same, with a cup in the PR2's right hand |
| `pr2_holding_a_whisk` | The same, with a whisk in the PR2's right hand |
| `pr2_holding_a_sponge` | The same, with a sponge in the PR2's right hand |
| `tracy_at_the_table` | Tracy at its table, with a 5 cm cube on it, under stiff contacts |
| `pr2` | The PR2 on the floor |
| `tracy` | The Tracy robot on the floor |

The physics is stepped frame by frame at the pace of the clock (`--speed 0.25` is a quarter of
real time), because MuJoCo free-running steps as fast as the machine can and would show a dropped
cube at rest at once. With a window the run lasts until the window is closed, unless
`--duration` is given; without one (`--headless`) `--duration` is required.

The kitchen and apartment MJCF files of `semantic_digital_twin/resources/mjcf` describe only
the kinematic structure, with no geometry, so MuJoCo shows nothing of them but a black window; the
apartment here is the URDF one. The two URDF kitchens do not compile in MuJoCo (`axis too small in
joint`), so there is no kitchen scene.

A scene's furniture is made to rest: a drawer's collision shape overlaps the cabinet it slides in,
which in physics throws every drawer and door open at the start, so the furniture's shapes do not
collide with each other (only with what is not furniture) and its joints carry a little friction.
A robot is spawned with its arms parked, so its servos do not pull them to a limit.

The URDF parser gives a material's color to boxes, spheres and cylinders and leaves meshes white, so
the viewer colors the meshes of the apartment and the robots with the colors their URDF declares. There
are no textures to show: none of these meshes carries a texture image.

## Actions

`--action ACTION` has the robot perform an action first, under MuJoCo physics: the scene is left
a second to settle, the plan is performed by the robot's own servos, and the window stays open
after it (or for `--duration`, counted from the end of the action):

There is one demonstration of each kind of action coraplex offers, apart from cutting, detecting
objects and taking a lift. Each is a sequence of coraplex actions (the third column) with every
parameter given; the scene is `pr2_at_the_counter` unless the table names another.

| Action | What happens | coraplex actions | Scene |
| --- | --- | --- | --- |
| `raise_torso` | The torso moves up to its highest position | `MoveTorsoAction` | |
| `open_grippers` | The grippers of both arms open | `SetGripperAction` | |
| `close_grippers` | The grippers of both arms close | `SetGripperAction` | |
| `park_arms` | The right hand reaches out, then both arms fold back | `MoveManipulatorAction`, `ParkArmsAction` | |
| `move_arm` | The torso rises and the right hand moves to a point in front of the robot | `MoveTorsoAction`, `MoveManipulatorAction` | |
| `look_at_milk` | The head turns to look at the milk carton | `LookAtAction` | |
| `face_the_cup` | The base turns on the spot to face the cup | `FaceAtAction` | |
| `face_and_look_at_the_cup` | The base turns to face the cup and the head looks at it | `FaceAndLookAtAction` | |
| `go_to_the_counter` | The base drives to the spot in front of the counter (from 1.1 m to 1.85 m) | `NavigateAction` | |
| `reach_for_milk` | The robot drives to the counter and moves its open right hand around the milk | `NavigateAction`, `SetGripperAction`, `ReachAction` | |
| `grasp_milk` | The robot drives to the counter and grasps the milk where it stands | `NavigateAction`, `GraspingAction` | |
| `pick_up_milk` | The robot drives to the counter (to 1.75 m) and picks the milk carton up head-on (see below) | `NavigateAction`, `PickUpAction` | |
| `place_milk` | The robot picks the milk up and puts it down 20 cm along the counter | `NavigateAction`, `PickUpAction`, `PlaceAction` | |
| `pick_and_place_milk` | The same, in one action | `NavigateAction`, `PickAndPlaceAction` | |
| `move_and_pick_up_milk` | The robot drives to the counter, turns towards the milk and picks it up | `MoveAndPickUpAction` | |
| `move_and_place_milk` | The robot picks the milk up, drives along the counter and puts it down there | `MoveAndPickUpAction`, `MoveAndPlaceAction` | |
| `transport_milk` | The same, carrying the milk with the arms parked | `TransportAction` | |
| `open_drawer` | The robot drives to the drawer below the counter and pulls it open | `MoveTorsoAction`, `ParkArmsAction`, `NavigateAction`, `OpenAction` | |
| `close_drawer` | The robot pulls the drawer open and pushes it shut again | the same, then `CloseAction` | |
| `move_and_open_drawer` | The robot drives to the drawer, turns towards its handle and pulls it open | `MoveTorsoAction`, `ParkArmsAction`, `MoveAndOpenAction` | |
| `pour_into_bowl` | The robot pours from the cup in its hand into the bowl (there is no liquid; only the cup moves) | `SetGripperAction`, `ParkArmsAction`, `MoveTorsoAction`, `NavigateAction`, `PouringAction` | `pr2_holding_a_cup` |
| `mix_in_bowl` | The robot stirs in the bowl with the whisk in its hand | the same, ending in `MixingAction` | `pr2_holding_a_whisk` |
| `wipe_counter` | The robot wipes a free stretch of the counter with the sponge in its hand | the same, ending in `WipingAction` | `pr2_holding_a_sponge` |
| `pick_up_cube` | Tracy picks the cube up off its table with its left arm | `PickUpAction` | `tracy_at_the_table` |

```bash
python -m experiments.physics_viewer pr2_at_the_counter --action raise_torso --speed 4
```

```bash
python -m experiments.physics_viewer pr2_holding_a_cup --action pour_into_bowl --speed 4
```

The counter scenes have stiff contacts (an elliptic friction cone and passes against slip, as
Tracy's scene has): without them a held carton creeps down between the pads while it is carried
and slips out, and the taps of the sink turn by themselves over a long run. The tools are fixed
in the hand, as the tool demos mount them, so only the actions are under physics, not the grip on
the tool. Stirring pushes the light bowl about 13 cm along the counter, upright, which is why
the cup stands further back than the bowl.

The physics, and the plan with it, run no faster than the clock times `--speed`. At 1 raising the
torso takes as long as the real torso, about half a minute.

The window opens on a view of the scene where it has one: at the counter, from the side, with the
PR2 and the objects on the counter in view. While an action runs the window says so at the top left,
and when it is done it lists the joints it moved, from where to where (`torso_lift_joint: 0.011 ->
0.290`); the same lines go to the log. Only joints a servo drives are listed. Press Tab in the window
to hide MuJoCo's side panels and give the scene the whole window.

### Picking up the milk

The PR2 drives to 1.75 m in front of the counter and lifts the milk carton about 9 cm, grasping it
head-on (the first of the 12 grasps the carton offers). Two things had to be right for this:

- The integrator. With the Runge-Kutta integrator the simulation used before, the pads touch the
  carton for a few milliseconds at a time with 150 to 170 N, and the carton is squeezed out forward
  and pushed over. With MuJoCo's implicit integrator (`SimulationOptions.integrator`) the same grasp
  holds it.
- The grasp. The grasp the plan took before (the second, diagonal to the carton) needs the gripper's
  full opening, and from 1.85 m it still pushes the carton over. The head-on grasp is not reachable
  from 1.85 m.

What did not help under the Runge-Kutta integrator: grasping 5 cm lower to 3 cm higher, gripper force
limits from 0.2 to 3.0, sliding friction up to 6, a softer or stiffer coupling of the finger joints, a
servo on every finger joint, stiffer arm servos and stiffer contacts. The cereal box, held the same
way, is lifted too with the implicit integrator at the gripper's default force.

### Tracy picks up a cube

Tracy, which is set up for it, does lift the cube under physics:

```bash
python -m experiments.physics_viewer tracy_at_the_table --action pick_up_cube --speed 3
```

The scene is built as the Tracy montessori demo builds its own: every body is as solid as the cube (a
contact time constant of 2 ms), MuJoCo uses the elliptic friction cone, an impedance ratio of 10 and 10
passes against slip, and the arms start parked. The cube is 5 cm, which the 8.5 cm gripper takes with
room to spare, and it stands on the table where Tracy reaches it. Of the 12 grasps the cube offers,
2 to 10 are reachable by the left arm in the kinematic simulation, and of those 3 and 6 lift the
cube under physics (to 0.98 and 0.99 m, from 0.905); the others push it about or miss. The action
takes the sixth, which is a measurement and not a rule.

## Under Isaac Sim

The same scenes and actions run under NVIDIA Isaac Sim's PhysX instead of MuJoCo, in the
Python environment Isaac Sim is installed in beside these packages:

```bash
python -m experiments.physics_viewer.isaac SCENE [--action ACTION] [--speed FACTOR] [--duration SECONDS] [--headless] [--camera CAMERA] [--record DIRECTORY]
```

Three cameras stand in the room (`behind_left`, `behind_right`, `across_the_counter`) and
one rides on the robot's own camera (`robot_camera`); `--camera` picks the one the window
shows first. `--record DIRECTORY` writes a video of every camera and `player.html`, a page
that plays them one at a time or side by side, at 1, 2, 4 or 8 times speed:

```bash
python -m experiments.physics_viewer.isaac pr2_at_the_counter --action pick_up_milk --headless --duration 1.5 --record recordings/pick_up_milk
```

How the world is written and simulated under Isaac Sim is described in
`semantic_digital_twin/adapters/isaac/README.md`.
