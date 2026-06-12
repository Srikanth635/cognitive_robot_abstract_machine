# MuJoCo resources

This directory contains self-contained MJCF models and the assets MuJoCo opens
while compiling them. Examples must load resources from here instead of relying
on external Multiverse or ROS workspaces.

## Layout

- `environments/`: complete scenes and static surroundings.
- `objects/`: standalone objects and articulated props.
- `robots/`: robot, manipulator, and hand models.

The source-relative layout below each family is preserved because MJCF
`compiler`, `include`, mesh, texture, and height-field paths are relative.

## Environment families

- `apartment`: customized apartment plus the upstream no-visual variant.
- `dlr_kitchen`: articulated DLR kitchen.
- `empty`: empty world.
- `floor`: basic floor scene.
- `table_with_bowling`: table, cube, bowling ball, and composed scene.

## Object families

- `cereal_box` and `milk_box`: customized dynamic household objects.
- `iai_objects/jeroen_cup`: cup, liquid, and combined cup-with-liquid models.
- `montessori_toys`: box, six insert objects, full composition, and small table.
- `others/meta_quest_box`: Meta Quest box set.
- `table`: standalone table.
- `task_board`: articulated board, main board, cable plug, and probe.
- `ycb_objects`: bowl, cracker box, plate, and spoon.

## Robot families

- `pr2`: customized controllable PR2 generated from the IAI PR2 description.
- `dlr/rollin_justin`: Rollin' Justin.
- `franka_emika_panda`: Panda control variants.
- `franka_robotics/fr3`: FR3 without end effector.
- `iit/iCub`: mesh and primitive iCub variants.
- `inspire_hand`: left and right Inspire Hand variants.
- `kinova`: Gen3 with Robotiq 2F-85 variants.
- `shadow_hand`: left and right Shadow Hand control variants.
- `unitree`: B2, B2-W, G1, Go2, Go2-W, H1, and H1-2 variants and scenes.

There are 86 directly compilable MJCF entry points in this bundle. Scene files
and robot-only files are both retained because they serve different composition
workflows.

## Customized resources

The apartment, PR2, milk box, and cereal box differ from their upstream source
versions. They are the canonical SDT copies and must not be overwritten during
resource refreshes.

The apartment's nested `apartment/apartment` directory is required by:

```xml
<compiler meshdir="apartment/meshes/" texturedir="apartment/textures/"/>
```

When regenerating PR2, write the model to `robots/pr2/pr2.xml`; its required
meshes belong in the adjacent `robots/pr2/pr2_meshes` directory.

## Provenance

- Multiverse Resources:
  `Multiverse-Framework/Multiverse-Resources`
  revision `7e63942e786513faf0aabfcc0e5fa32c3db61fc2`
- IAI PR2 ROS 2 resources:
  `code-iai/iai_pr2`
  revision `44009944748bb8c0b9a7ecae73721bc8a6a76f1d`

The ROS 2 tree contains no standalone MJCF entry point. Its relevant PR2 mesh
data is already represented by the generated, self-contained `robots/pr2`
bundle.

The independently maintained `robots/mujoco_menagerie` mirror in Multiverse is
not duplicated here. It is approximately 1.6 GB, has per-model licensing, and
is better consumed from MuJoCo Menagerie directly when a specific model is
needed. The native Multiverse MJCF families listed above are bundled here.

Keep family-specific license files beside imported assets where the source
provides them.
