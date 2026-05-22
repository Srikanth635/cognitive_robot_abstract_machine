---
id: pycram.datastructures.enums.AxisIdentifier
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/datastructures/enums.py
    lines: [74, 90]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
values:
  - X
  - Y
  - Z
  - Undefined
uses: []
used_by:
  - pycram.datastructures.enums.ApproachDirection
  - pycram.datastructures.enums.VerticalAlignment
  - pycram.datastructures.grasp.GraspDescription
  - pycram.robot_plans.actions.core.robot_body
status: stable
tags: [pycram, enum, axis, geometry, grasp]
last_ingest: 2026-05-19
---

_Enum: identifies a Cartesian axis (X/Y/Z) or the absence of one (`Undefined`); used as the first element of the `(axis, direction)` encoding in `ApproachDirection` and `VerticalAlignment`._

| Value | Tuple | Meaning |
|---|---|---|
| `X` | `(1, 0, 0)` | Positive X axis |
| `Y` | `(0, 1, 0)` | Positive Y axis |
| `Z` | `(0, 0, 1)` | Positive Z axis |
| `Undefined` | `(0, 0, 0)` | No specific axis (used by `VerticalAlignment.NoAlignment`) |

`AxisIdentifier` is also consumed by `GraspDescription.calculate_manipulator_axis(axis)`, which transforms the named axis through the manipulator's `front_facing_orientation` to produce the corresponding gripper-frame direction vector.

## Related

- **Used by:** [[pycram.datastructures.enums.ApproachDirection]] (axis component of encoding), [[pycram.datastructures.enums.VerticalAlignment]] (axis component of encoding), [[pycram.datastructures.grasp.GraspDescription]] (`calculate_manipulator_axis`)

## Provenance

- `pycram/src/pycram/datastructures/enums.py:74-90` — `AxisIdentifier(Enum)` class with `(x, y, z)` tuple values.
