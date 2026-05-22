---
id: sdt.collision_checking
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/collision_checking/collision_manager.py
    lines: [1, 80]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.world_entity.Body
  - sdt.world.World
used_by:
  - bridge.sdt-giskardpy
  - sdt.reasoning.predicates
  - sdt.reasoning.robot_predicates
  - sdt.robots.concrete
  - giskardpy.motion_statechart.context
status: stable
tags: [sdt, collision, fcl, pybullet, trimesh, matrix, rules, manager]
last_ingest: 2026-05-18
---

_`CollisionManager`: primary interface for collision checking; manages rules, owns the detector backend, and dispatches collision results to `CollisionConsumer` observers._

## Purpose

The collision system in SDT answers two questions at runtime: (1) are any bodies in collision right now? (2) which pairs should be checked? `CollisionManager` owns the answer to both. It is a `ModelChangeCallback` — world topology changes automatically trigger a detector scene rebuild and collision matrix update.

## Architecture

```
CollisionManager
  ├── CollisionDetector  (abstract; concrete: BulletCollisionDetector, FCLCollisionDetector)
  ├── CollisionMatrix    (which pairs to check; driven by CollisionRules)
  └── List[CollisionConsumer]  (observers: giskardpy constraint generators)
```

**`CollisionConsumer` observer pattern:** when `CollisionManager.compute_collisions()` runs, it calls `consumer.on_compute_collisions(results)` on all registered consumers. giskardpy collision-avoidance tasks register as consumers so they see collision results each tick without polling.

## Collision rules

`collision_rules.py` defines the priority stack applied when building the `CollisionMatrix`:

| Rule | Effect |
|---|---|
| `AllowCollisionForAdjacentPairs` | Suppress adjacent-link pairs (always touching) |
| `AllowNonRobotCollisions` | Allow object–object collisions (only robot–world matters) |
| `AvoidSelfCollisions` | Mark all robot self-pairs as `AVOID` |
| `AvoidExternalCollisions` | Mark robot–environment pairs as `AVOID` |
| `MaxAvoidedCollisionsOverride` | Cap total `AVOID` pairs per robot (performance) |

Rules are applied in priority order; later rules override earlier ones for the same pair.

## Detector backends

| Class | Library | Notes |
|---|---|---|
| `FCLCollisionDetector` (trimesh) | trimesh + python-fcl | Default; mesh-accurate, fast |
| `BulletCollisionDetector` | pybullet | Alternative; supports soft bodies |

Both implement `CollisionDetector.check_collision_between_bodies(body1, body2) → CollisionCheckingResult`.

## `CollisionCheckingResult`

Contains per-pair results: `distance: float`, contact points, normals. The `contact` predicate in `reasoning/predicates.py` uses `distance < threshold` directly from this result.

## Related

- **Uses:** [[sdt.world_description.world_entity.Body]], [[sdt.world.World]]
- **Used by:** [[sdt.reasoning.predicates]] (`contact()` calls `FCLCollisionDetector.check_collision_between_bodies()`), [[sdt.reasoning.robot_predicates]], [[sdt.robots.concrete]] (robot files register collision rules), [[giskardpy.motion_statechart.context]]

## Open questions

- `MaxAvoidedCollisionsOverride` caps the number of tracked collision pairs — if a robot has more active pairs than the cap, some pairs are silently dropped. The selection criterion for which pairs survive the cap is not documented in the files read.

## Provenance

- `collision_manager.py:1-80` — `CollisionManager`, `CollisionConsumer`, `on_compute_collisions`, `on_world_model_update`, `on_collision_matrix_update`.
