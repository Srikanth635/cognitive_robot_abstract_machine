---
id: pycram.datastructures.enums.ApproachDirection
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/datastructures/enums.py
    lines: [92, 135]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
values:
  - FRONT
  - BACK
  - LEFT
  - RIGHT
uses:
  - pycram.datastructures.enums.AxisIdentifier
used_by:
  - pycram.datastructures.grasp.GraspDescription
status: stable
tags: [pycram, enum, approach-direction, grasp, geometry]
last_ingest: 2026-05-19
---

_Enum: face of the object's x-y bounding box from which the gripper approaches; each value encodes an `(AxisIdentifier, direction: ±1)` pair._

| Value | Encoding | Side approached | `SIDE_ROTATIONS` quaternion `[x,y,z,w]` |
|---|---|---|---|
| `FRONT` | `(X, -1)` | −X face | `[0, 0, 0, 1]` — identity |
| `BACK`  | `(X, +1)` | +X face | `[0, 0, 1, 0]` — 180° around Z |
| `RIGHT` | `(Y, -1)` | −Y face | `[0, 0, √2/2, √2/2]` — +90° around Z |
| `LEFT`  | `(Y, +1)` | +Y face | `[0, 0, −√2/2, √2/2]` — −90° around Z |

`ApproachDirection` subclasses `Grasp` (a base for grasp-related enums). The class method `from_axis_direction(axis, sign)` maps an `(AxisIdentifier, int)` pair back to the matching enum member; used inside `GraspDescription.calculate_closest_faces`.

The `SIDE_ROTATIONS` quaternion table in `pycram.datastructures.rotations` is keyed by these values and is one of four rotation factors in `GraspDescription.grasp_orientation()`.

## Related

- **Uses:** [[pycram.datastructures.enums.AxisIdentifier]]
- **Used by:** [[pycram.datastructures.grasp.GraspDescription]] (`approach_direction` field, `SIDE_ROTATIONS` lookup, `calculate_closest_faces`)

## Provenance

- `pycram/src/pycram/datastructures/enums.py:92-135` — `ApproachDirection(Grasp, Enum)` class with `from_axis_direction` classmethod.
