---
id: pycram.alternative_motion_mapping.AlternativeMotion
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/alternative_motion_mapping.py
    lines: [1, 44]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.robot_plans.BaseMotion
  - pycram.motion_executor.MotionExecutor
  - sdt.robots.abstract_robot.AbstractRobot
used_by:
  - pycram.pose_validator
status: stable
tags: [motion, alternative, robot-specific, execution-type, dispatch]
last_ingest: 2026-05-18
---

_Robot- and execution-type-specific motion override: a `BaseMotion` subclass that replaces another `BaseMotion` at dispatch time if the current robot and execution context match._

## Purpose

`AlternativeMotion` solves the problem of robot-specific or execution-mode-specific motion implementations without conditional logic inside plans. A plan author writes `MoveGripperMotion(...)` uniformly. At runtime, `check_for_alternative` scans `__subclasses__()` and substitutes a specialized implementation when:
- The motion class matches (subclass relation).
- The robot's concrete class matches (`original_class()` — provided by `HasGeneric`).
- `MotionExecutor.execution_type` matches the alternative's declared `execution_type`.

## Class structure

```python
@dataclass
class AlternativeMotion(HasGeneric[AbstractRobotType], ABC):
    execution_type: ClassVar[ExecutionType]   # which execution mode this handles
    perform(self) -> None                     # no-op; same as BaseMotion
```

`HasGeneric[AbstractRobotType]` (from `krrood.ormatic`) provides `original_class()` which returns the concrete `AbstractRobot` subtype bound as the generic parameter.

## check_for_alternative

```python
@staticmethod
def check_for_alternative(robot_view: AbstractRobot, motion: Type[BaseMotion]) -> Optional[Type[BaseMotion]]:
    for alternative in AlternativeMotion.__subclasses__():
        if (
            issubclass(alternative, motion)             # is a specialization of motion
            and alternative.original_class() == robot_view.__class__   # robot type matches
            and MotionExecutor.execution_type == alternative.execution_type  # mode matches
        ):
            return alternative                          # return class, not instance
    return None
```

Returns the **class**, not an instance. The caller is expected to instantiate it with the same constructor arguments as the original motion.

## When to use

- **Defining a robot-specific motion** — subclass `AlternativeMotion` with the target motion class AND parameterize by the robot type:
  ```python
  @dataclass
  class PR2MoveGripperMotion(AlternativeMotion[PR2], MoveGripperMotion):
      execution_type: ClassVar[ExecutionType] = ExecutionType.REAL
  ```
- **Not for:** default behavior — only for overrides. If `check_for_alternative` returns `None`, the original motion class is used.

## Open questions

- `__subclasses__()` only returns direct subclasses, not grandchildren. If alternative motions are themselves subclassed, the dispatch breaks silently.
- No fallback priority ordering — if two alternatives match (e.g., two robot types share a base), an arbitrary one is returned (first in `__subclasses__()` iteration order).
- Where `check_for_alternative` is actually called — whether it's called in `ActionNode`, `MotionNode`, or `MotionExecutor` — is not visible from this file alone.

## Related

**Uses:** [[pycram.robot_plans.BaseMotion]], [[pycram.motion_executor.MotionExecutor]], [[sdt.robots.abstract_robot.AbstractRobot]]

**Used by:** [[pycram.pose_validator]] (`check_for_alternative` is called from `pose_sequence_reachability_validator`)

## Provenance

- `pycram/src/pycram/alternative_motion_mapping.py` lines 1–44 (commit `0528d8cf3`) — full `AlternativeMotion` class.
