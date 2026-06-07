# MuJoCo resources

This directory keeps the MJCF models and their required assets together so
examples do not depend on external resource checkouts.

## Layout

- `apartment/apartment.xml`: Multiverse apartment MJCF.
- `apartment/apartment/meshes`: apartment OBJ and STL assets.
- `apartment/apartment/textures`: apartment texture assets.
- `pr2/pr2.xml`: controllable PR2 MJCF.
- `pr2/pr2_meshes`: PR2 STL assets.

The nested `apartment/apartment` path is intentional. The apartment compiler
uses paths relative to `apartment.xml`:

```xml
<compiler meshdir="apartment/meshes/" texturedir="apartment/textures/"/>
```

When regenerating the PR2 model, write the output to `pr2/pr2.xml`. The
generator defaults to creating the required `pr2_meshes` directory beside it.

The apartment MJCF also contains a finite floor and three room walls. The
front is intentionally open so the default MuJoCo viewer camera can see the
kitchen and robot.
- `objects/milk_box/`: textured milk-carton visual mesh, stable box collision geometry, and dynamic MJCF.
- `objects/cereal_box/`: textured cereal-box visual mesh, stable box collision geometry, and dynamic MJCF.
