---
id: pycram.datastructures.enums.VerticalAlignment
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/datastructures/enums.py
    lines: [137, 167]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
values:
  - TOP
  - BOTTOM
  - NoAlignment
uses:
  - pycram.datastructures.enums.AxisIdentifier
used_by:
  - pycram.datastructures.grasp.GraspDescription
status: stable
tags: [pycram, enum, vertical-alignment, grasp, geometry]
last_ingest: 2026-05-19
---

_Enum: whether the gripper tilts to approach from above, below, or keeps a lateral (side-grasp) posture; each value encodes an `(AxisIdentifier, direction: ±1 or 0)` pair._

| Value | Encoding | Meaning | `VERTICAL_ROTATIONS` quaternion `[x,y,z,w]` |
|---|---|---|---|
| `NoAlignment` | `(Undefined, 0)` | Side grasp, no vertical tilt | `[0, 0, 0, 1]` — identity |
| `TOP`         | `(Z, -1)`        | Approach from above (grasp top face) | `[0, √2/2, 0, √2/2]` — +90° around Y |
| `BOTTOM`      | `(Z, +1)`        | Approach from below (grasp bottom face) | `[0, −√2/2, 0, √2/2]` — −90° around Y |

`VerticalAlignment` subclasses `Grasp`. The classmethod `from_axis_direction(axis, sign)` maps `(AxisIdentifier, int)` back to the matching member; used by `GraspDescription.calculate_closest_faces`.

The `VERTICAL_ROTATIONS` quaternion table in `pycram.datastructures.rotations` is keyed by these values and is the second rotation factor in `GraspDescription.grasp_orientation()`.

## Related

- **Uses:** [[pycram.datastructures.enums.AxisIdentifier]]
- **Used by:** [[pycram.datastructures.grasp.GraspDescription]] (`vertical_alignment` field, `VERTICAL_ROTATIONS` lookup, `calculate_closest_faces`)

## Provenance

- `pycram/src/pycram/datastructures/enums.py:137-167` — `VerticalAlignment(Grasp, Enum)` class with `from_axis_direction` classmethod.
