---
id: giskardpy.model.world_config
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/model/world_config.py
    lines: [1, 159]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world.World
  - sdt.world_description.world_entity.Body
  - sdt.world_description.world_entity.KinematicStructureEntity
  - sdt.world_description.connections
  - sdt.robots.abstract_robot.AbstractRobot
  - sdt.adapters
used_by: []
status: stable
tags: [giskardpy, model, world, urdf, config, robot, standalone]
last_ingest: 2026-05-17
---

_`WorldConfig` and its four concrete subclasses are giskardpy's standalone world initialization API: they create and populate an `sdt.world.World` without pycram. When giskardpy is driven via pycram, this module is bypassed — pycram supplies the world through `Context.world`._

## Purpose

`WorldConfig` is the entry point for using giskardpy outside of pycram — e.g. in the
ROS 2 middleware, in tests, or in standalone scripts. User code creates a subclass
instance, calls `setup_world()`, and passes `config.world` to the giskardpy infrastructure.

**When used via pycram:** `WorldConfig` is NOT used. The world is created externally,
placed in `pycram.datastructures.Context.world`, and flows via:

```
Context.world → plan.world → ActionNode.construct_motion_state_chart()
    → MotionExecutor(motions, self.plan.world, ...)
    → MotionStatechartContext(world=self.world, ...)
```

This means a single `sdt.world.World` object is shared throughout the entire execution stack. No dual-world issues arise in the pycram path.

## Class hierarchy

```
WorldConfig (ABC)
  .world: World = field(default_factory=World)
  .setup_world(*args, **kwargs)   ← abstract; user must call after init

├── EmptyWorld
│     adds a single "map" Body as root
│
├── WorldWithFixedRobot(urdf, root_name, robot_name, urdf_view)
│     parses URDF → attaches robot via FixedConnection under map
│
├── WorldWithOmniDriveRobot(urdf, root_name, odom_body_name, urdf_view)
│     map → Connection6DoF → odom → OmniDrive → robot_root
│
├── WorldWithDiffDriveRobot(urdf, root_name, odom_body_name, urdf_view)
│     map → Connection6DoF → odom → DifferentialDrive → robot_root
│
└── WorldFromDatabaseConfig(primary_key=1)
      loads World from krrood/ormatic ORM database (persisted world)
```

## Key patterns

### URDF-based configs

All three URDF configs follow the same recipe:
1. `URDFParser(urdf=..., prefix="").parse()` → new `World` with robot kinematic tree.
2. `urdf_view.from_world(world_with_robot)` → attaches the `AbstractRobot` view.
3. `self.world.merge_world(world_with_robot, connection)` — merges robot world into the
   config world, stitching them via the connection type.

The difference is the base connection:
- `WorldWithFixedRobot` → `FixedConnection` (no floating base DOF)
- `WorldWithOmniDriveRobot` → `Connection6DoF` (localization) + `OmniDrive` (3-DOF base)
- `WorldWithDiffDriveRobot` → `Connection6DoF` (localization) + `DifferentialDrive` (2-DOF base)

Both mobile configs use `translation_velocity_limits=0.2, rotation_velocity_limits=0.2` as
hardcoded default drive limits.

### Database config

`WorldFromDatabaseConfig` uses `krrood.ormatic` to query the world by primary key from the
SDT ORM. This is the path for persisted worlds (e.g. after learning or simulation). The
`ormatic_world_class` is resolved dynamically via `get_dao_class(World)`.

## Key observations

- `urdf_view` defaults to `MinimalRobot` — a thin wrapper that provides the `AbstractRobot`
  interface with no extra semantics. Users pass richer robot types (e.g. `HSRB`) for
  sensor/manipulator-aware behavior.
- `WorldWithFixedRobot.robot_root` is stored as an `InitVar`-style field — it is set during
  `setup_world()`, not at init time.
- `EmptyWorld` has a commented-out `set_default_limits` call — suggesting default
  velocity/jerk limits were once set globally here but have been removed. Joints currently
  get limits from the URDF or robot-type definitions.

## Related

- **World:** [[sdt.world.World]]
- **Bodies/Connections:** [[sdt.world_description.world_entity.Body]], [[sdt.world_description.connections]]
- **Robot view:** [[sdt.robots.abstract_robot.AbstractRobot]]
- **SDT bridge:** [[bridge.sdt-giskardpy]]

## Open questions

- `WorldFromDatabaseConfig.setup_collision_config()` is a no-op stub — collision configuration
  from database is not implemented. Unknown whether this is intentional or a TODO.

## Provenance

- `giskardpy/src/giskardpy/model/world_config.py:1-159` at commit `0528d8cf3`.
