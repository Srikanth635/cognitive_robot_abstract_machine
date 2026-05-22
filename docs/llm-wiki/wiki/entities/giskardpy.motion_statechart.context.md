---
id: giskardpy.motion_statechart.context
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/motion_statechart/context.py
    lines: [1, 119]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world.World
  - sdt.collision_checking
  - giskardpy.qp.qp_controller
used_by:
  - pycram.motion_executor.MotionExecutor
  - giskardpy.qp.qp_controller
  - bridge.sdt-giskardpy
  - pycram.locations.locations.GiskardLocation
  - pycram.pose_validator
status: stable
tags: [giskardpy, context, world, collision, qp-config, extension]
last_ingest: 2026-05-17
---

_`MotionStatechartContext` is the execution substrate passed to every `build()` and `tick()` call in the MSC. It binds `sdt.world.World`, `QPControllerConfig`, collision infrastructure, and an extension registry into a single object._

## Purpose

`MotionStatechartContext` is the DI container for MSC execution. It is created once before
`Executor.compile(msc)` and passed down to every node's `build()`, `on_start()`, and
`cleanup()` methods. After execution, `cleanup()` must be called to deregister collision
consumers.

## Key fields

| Field | Type | Notes |
|---|---|---|
| `world` | `sdt.world.World` | The kinematic world; FK queries, control command writes |
| `float_variable_data` | `FloatVariableData` | Registry for auxiliary CasADi float variables (FK bindings, etc.) |
| `qp_controller_config` | `QPControllerConfig` | MPC horizon, frequency, DOF weights, solver class |
| `control_cycle_variable` | `FloatVariable` | Tick counter; auto-created in `__post_init__` |
| `extensions` | `Dict[Type, ContextExtension]` | Plugin map for ROS 2 extensions etc. |

## Collision managers (lazy-cached properties)

```python
@cached_property
def self_collision_manager(self) -> SelfCollisionVariableManager:
    manager = SelfCollisionVariableManager(self.float_variable_data)
    self.collision_manager.add_collision_consumer(manager)
    return manager

@cached_property
def external_collision_manager(self) -> ExternalCollisionVariableManager:
    ...
```

Both are lazily created on first access. `self.collision_manager` is a shortcut to
`self.world.collision_manager`. The consumers are registered with the collision manager
so they receive proximity data each tick.

`cleanup()` checks `__dict__` directly (`"self_collision_manager" in self.__dict__`) —
the standard way to test if a `@cached_property` has been materialized — and removes
the consumers before deleting the cached attribute.

## Extension mechanism

```python
context.add_extension(MyRosExtension())
ctx_ext = context.require_extension(MyRosExtension)  # raises MissingContextExtensionError
```

`ContextExtension` is an empty base dataclass. Extensions are keyed by type. The ROS 2
executor adds its extensions before `compile()` so that nodes can call
`context.require_extension(Ros2Extension)` during `build()`.

## Construction by pycram

`pycram.motion_executor.MotionExecutor._execute_for_simulation()` creates it inline:

```python
MotionStatechartContext(
    world=self.world,
    qp_controller_config=QPControllerConfig(
        target_frequency=50, prediction_horizon=4, verbose=False
    ),
)
```

`target_frequency=50` but max loop counter = 2000 → max execution time = 40 s.

## Related

- **World:** [[sdt.world.World]]
- **Collision:** [[sdt.collision_checking]]
- **QP config:** [[giskardpy.qp.qp_controller]]
- **SDT coupling:** [[bridge.sdt-giskardpy]]
- **pycram caller:** [[pycram.motion_executor.MotionExecutor]]

## Provenance

- `giskardpy/src/giskardpy/motion_statechart/context.py:1-119` at commit `0528d8cf3`.
