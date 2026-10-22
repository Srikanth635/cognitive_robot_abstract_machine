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
| `pr2_at_the_counter` | The PR2 at the kitchen counter, with a milk carton, a cereal box, a bowl and a cup on it |
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

| Action | What happens |
| --- | --- |
| `raise_torso` | The torso moves up to its highest position |
| `open_grippers` | The grippers of both arms open |
| `close_grippers` | The grippers of both arms close |
| `go_to_the_counter` | The base drives to the spot in front of the counter (from 1.1 m to 1.85 m) |
| `pick_up_milk` | The robot drives to the counter and picks the milk carton up with its right arm (`pr2_at_the_counter`; see below) |
| `pick_up_cube` | Tracy picks the cube up off its table with its left arm (`tracy_at_the_table`) |

```bash
python -m experiments.physics_viewer pr2_at_the_counter --action raise_torso --speed 4
```

The physics, and the plan with it, run no faster than the clock times `--speed`. At 1 raising the
torso takes as long as the real torso, about half a minute.

The window opens on a view of the scene where it has one: at the counter, from the side, with the
PR2 and the objects on the counter in view. While an action runs the window says so at the top left,
and when it is done it lists the joints it moved, from where to where (`torso_lift_joint: 0.011 ->
0.290`); the same lines go to the log. Only joints a servo drives are listed. Press Tab in the window
to hide MuJoCo's side panels and give the scene the whole window.

### Picking up the milk

Driving to the counter works under physics: the base ends within a centimetre of the spot. Picking up
the milk is performed, but the carton does not end up in the gripper: it is pushed over (it ends lying
on the counter, 5.6 cm lower), and the bowl is knocked along with it. The window reports the objects
that moved next to the joints. The same plan lifts the carton in the kinematic simulation the demos use.

What the physics shows: the gripper opens to 8.8 cm for a carton 6.4 cm across, so the approach has
1.2 cm to spare on each side, and the arm tracks the plan less precisely than that; the fingers clip
the carton, and it is pushed out of the way as they close. Stiffer arm servos (up to 16 times), a
stiffer contact (a time constant of 2 ms, an elliptic friction cone) and sliding friction up to 6 did not
change this. Of the 12 grasps the milk offers, only 1, 2 and 3 are reachable by the right arm even
kinematically, and the plan turns the wrist by 4.4 rad on the way. semantic_digital_twin and coraplex
set up only the Tracy robot for picking things up under physics.

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
