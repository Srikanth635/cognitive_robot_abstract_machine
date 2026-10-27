# cramera scenes

Scene bundles for cramera, the browser viewer of CRAM worlds and plan execution (see
[cramera's README](../README.md)). Each directory is one self-contained bundle of a coraplex
demo run: URDFs (environment and robot, `package://` resolved), meshes, the recorded
trajectory and the executed plan tree. `index.json` lists the bundles and the default scene.

The bundles were recorded with cram_viz, cramera's predecessor, and converted into the
format cramera reads by `cramera.resolution.scene_conversion`: a `placeTarget` stores
`position`, and the PR2's `<transmission>` elements, which cramera's URDF reader refuses,
are left out. An archive still in cram_viz's format, such as `~/.cram_viz/scenes`, is
converted the same way:

```bash
python -m cramera.resolution.scene_conversion ~/.cram_viz/scenes path/to/converted
```

cramera saves a new recording from its live viewer (see *Record and replay* in its README)
under `CRAMERA_DATA/scenes`, never in this directory. To share one, copy its bundle here and
add it to `index.json`.

## What a recording cannot observe

What a container holds is stated by hand in its object entry of `scene.json`, for example
the pitcher of `simple_pouring`:

```json
"contents": {"materialRegime": "LIQUID", "fillFraction": 0.5}
```

`materialRegime` names a member of the ontology's `MaterialRegime`; `fillFraction` is the
volume held over what the container holds upright when filled to its lowest lip. Converting
an archive keeps the contents; a bundle copied from a new recording has none until they are
stated.

## How the viewer finds these scenes

cramera looks for scene bundles in this order:

1. `CRAMERA_SCENES=/path/to/scenes` (environment variable)
2. this directory, when it holds an `index.json`
3. `CRAMERA_DATA/scenes` (default `~/.cramera/scenes`), where its own recordings are saved

A recording of the same name saved locally is shown instead of the bundle here.

## Scenes

| scene             | demo                                                   |
| ----------------- | ------------------------------------------------------ |
| `pr2_kitchen`     | PR2 transporting milk/bowl/spoon (IAI apartment)       |
| `garmi_apartment` | Garmi in the apartment                                 |
| `simple_pouring`  | PR2 pouring from a pitcher into a bowl (IAI apartment) |
