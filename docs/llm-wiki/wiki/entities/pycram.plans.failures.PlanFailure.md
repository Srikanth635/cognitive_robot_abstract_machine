---
id: pycram.plans.failures.PlanFailure
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/plans/failures.py
    lines: [1, 110]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.language.LanguageNode
  - sdt.spatial_types.Pose
  - sdt.world_description.world_entity.Body
used_by:
  - pycram.plans.PlanNode
  - pycram.plans.ActionNode
  - pycram.robot_plans.ActionDescription
  - pycram.language.LanguageNode
status: stable
tags: [exception, failure, plan, dataclass]
last_ingest: 2026-05-17
---

_Base exception dataclass for all plan/node execution failures. Inherits from `krrood.exceptions.DataclassException`. Six subclasses cover the concrete failure modes._

## Purpose

`PlanFailure` is the catch-all exception type for anything that goes wrong during plan
execution. It inherits from `krrood.exceptions.DataclassException` (not Python's
built-in `Exception` directly), making it a dataclass that may carry fields.

`PlanNode.perform()` catches `PlanFailure` (and only `PlanFailure`) to set
`status = FAILED` and re-raise. All plan execution exceptions must be subclasses.

## Hierarchy

```
DataclassException (krrood.exceptions)
└── PlanFailure
    ├── AllChildrenFailed          — TryInOrderNode / TryAllNode exhausted all branches
    ├── PerceptionObjectNotFound   — SearchAction found nothing
    ├── RobotInCollision           — robot intersects environment geometry
    ├── ConfigurationNotReached    — joint configuration not achieved
    ├── NavigationGoalNotReachedError — navigation goal pose not reached
    └── BodyUnfetchable            — arm cannot fetch a specific body
```

## Subclass details

| Class | Extra fields | Raised by |
|---|---|---|
| `AllChildrenFailed` | `language_node: LanguageNode` | `TryInOrderNode._perform()`, `TryAllNode._perform()` |
| `PerceptionObjectNotFound` | `search_action: SearchAction` | perception/search actions (Phase 4) |
| `RobotInCollision` | — | collision-checking code (Phase 4/5) |
| `ConfigurationNotReached` | `goal_validator`, `configuration_type: StaticJointState` | joint execution code |
| `NavigationGoalNotReachedError` | `current_pose: Pose`, `goal_pose: Pose` | navigation actions |
| `BodyUnfetchable` | `body: Body`, `arm: Arms` | grasp/pick-up actions |

## Cross-package dependencies

`failures.py` imports at runtime (not under `TYPE_CHECKING`) from SDT:
- `sdt.spatial_types.spatial_types.Pose` — for `NavigationGoalNotReachedError`
- `sdt.world_description.world_entity.Body` — for `BodyUnfetchable`

This makes `pycram.plans.failures` one of the first coupling points between pycram and
SDT. See stubs [[sdt.spatial_types.Pose]] and [[sdt.world_description.world_entity.Body]]
(Phase 5/6 expansion targets).

`LanguageNode` and `SearchAction` are `TYPE_CHECKING`-only imports — no runtime cycle.

## Related

- Caught by: [[pycram.plans.PlanNode]] (`perform()` sets status FAILED, records `reason`)
- Raised by: [[pycram.language.LanguageNode]] (`TryInOrderNode`, `TryAllNode`)
- Cross-package: [[sdt.spatial_types.Pose]], [[sdt.world_description.world_entity.Body]]

## Open questions

- `AllChildrenFailed.language_node` is annotated as `LanguageNode` but that import is
  `TYPE_CHECKING`-only. The field works at runtime as a forward reference. Worth
  verifying the krrood `DataclassException` handles string annotations correctly.
- No existing subclass covers "goal precondition not met" (e.g. object not reachable
  before attempting pick-up). This may live in `krrood` or be raised differently —
  check during Phase 4 ingest.

## Provenance

- `pycram/src/pycram/plans/failures.py:1-110` at commit `0528d8cf3`.
