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

```bash
python -m experiments.physics_viewer pr2_at_the_counter --action raise_torso --speed 4
```

The physics, and the plan with it, run no faster than the clock times `--speed`. At 1 raising the
torso takes as long as the real torso, about half a minute.
