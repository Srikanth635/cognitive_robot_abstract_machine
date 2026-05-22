# pycram — entity index

Complete entity listing for the `pycram` package. See [[index]] for Concepts, Bridges, and per-package overview pages.

---

## Plan graph

- [[pycram.plans.Plan]] — entity — Executable plan container: a `rustworkx` graph of PlanNodes with a Context supplying world/robot.
- [[pycram.plans.PlanNode]] — entity — Abstract graph-node base; carries execution status, timing, navigation, lifecycle (`perform`).
- [[pycram.plans.PlanEntity]] — entity — Minimal base for anything owned by a `Plan` (PlanNode and Context).
- [[pycram.plans.Designator]] — entity — Abstract base dataclass for all designators in pycram.
- [[pycram.plans.DesignatorNode]] — entity — Abstract plan-graph node that wraps a Designator; parent of ActionNode and MotionNode.
- [[pycram.plans.ActionNode]] — entity — Concrete node wrapping ActionDescription; drives action execution + Motion State Chart.
- [[pycram.plans.MotionNode]] — entity — Concrete leaf node wrapping BaseMotion; harvested by parent ActionNode for MSC construction.
- [[pycram.plans.UnderspecifiedNode]] — entity — Plan node that grounds a krrood `underspecified(...)` expression into ActionNode children.
- [[pycram.plans.factories.make_node]] — entity — Dispatch function: maps an ActionDescription/BaseMotion/PlanNode to its concrete plan-graph node.
- [[pycram.plans.factories]] — entity — Public combinator functions (sequential, parallel, try_in_order, etc.) that build plan trees.
- [[pycram.language.LanguageNode]] — entity — Abstract plan combinator node; parent of SequentialNode, ParallelNode, RepeatNode, MonitorNode, TryInOrderNode, TryAllNode, CodeNode.

## Designators — actions

- [[pycram.robot_plans.ActionDescription]] — entity — Designator subclass for actions; "builder for plans" with pre/post conditions.
- [[pycram.robot_plans.actions.core.PickUpAction]] — entity — Two-phase pick-up (reach+grasp, then lift+attach); also covers ReachAction and GraspingAction.
- [[pycram.robot_plans.actions.core.PlaceAction]] — entity — Reverse-reach, open, detach body from arm in SDT world model, retract.
- [[pycram.robot_plans.actions.core.NavigateAction]] — entity — Drive robot to a target pose; also covers LookAtAction.
- [[pycram.robot_plans.actions.core.container]] — entity — OpenAction + CloseAction for articulated containers; front-approach grasp + joint drive + release.
- [[pycram.robot_plans.actions.core.robot_body]] — entity — MoveTorsoAction, SetGripperAction, ParkArmsAction, CarryAction, FollowToolCenterPointPathAction.
- [[pycram.robot_plans.actions.core.misc]] — entity — DetectAction: constructs PerceptionQuery and calls from_world(); no MSC involvement.
- [[pycram.robot_plans.actions.composite]] — entity — FaceAtAction, SearchAction, TransportAction, PickAndPlaceAction, MoveAndPlace/PickUp variants.

## Designators — motions

- [[pycram.robot_plans.BaseMotion]] — entity — Designator subclass for motions; "builder for Motion State Charts", creates exactly one goal.
- [[pycram.robot_plans.motions.gripper]] — entity — Four gripper/TCP motions: MoveGripperMotion, MoveToolCenterPointMotion, ReachMotion, MoveTCPWaypointsMotion.
- [[pycram.robot_plans.motions.navigation]] — entity — MoveMotion: drives robot root link to target Pose via CartesianPose.
- [[pycram.robot_plans.motions.container]] — entity — OpeningMotion + ClosingMotion: articulated joint control via giskardpy Open/Close goals.
- [[pycram.robot_plans.motions.robot_body]] — entity — MoveJointsMotion (named joints to positions) + LookingMotion (camera Pointing goal).
- [[pycram.robot_plans.motions.misc]] — entity — DetectingMotion: wraps PerceptionQuery; _motion_chart returns None (no giskardpy task).

## Runtime and datastructures

- [[pycram.datastructures.Context]] — entity — Runtime configuration object attached to a Plan; supplies world/robot/ros_node/flags/query_backend.
- [[pycram.motion_executor.MotionExecutor]] — entity — Builds a MotionStatechart from pycram Tasks, compiles it, and runs the tick loop.
- [[pycram.datastructures.ExecutionData]] — entity — Pre/post execution snapshot (robot pose, world state) recorded by ActionNode.
- [[pycram.plans.plan_callbacks.PlanCallback]] — entity — Lifecycle callback hook invoked on node start/end during plan execution.
- [[pycram.fluent.Fluent]] — entity — Thread-safe reactive value wrapper polled by MonitorNode as a live condition.
- [[pycram.datastructures.grasp.GraspDescription]] — entity — Approach direction + orientation for grasping; provides 3-pose grasp/place sequences.
- [[pycram.view_manager.ViewManager]] — entity — Static lookup for end-effector/arm/camera views from arm enum + robot instance.
- [[pycram.plans.failures.PlanFailure]] — entity — Base exception type for plan/node execution failures; six concrete subclasses.

## Motion dispatch

- [[pycram.alternative_motion_mapping.AlternativeMotion]] — entity — Robot- and execution-type-specific motion override; `check_for_alternative` dispatches at runtime.

## Locations

- [[pycram.locations.costmaps]] — entity — Five costmap types (`Costmap` base, `OccupancyCostmap`, `VisibilityCostmap`, `GaussianCostmap`, `RingCostmap`) + `OrientationGenerator`; spatial probability-distribution machinery underlying all base-pose samplers.
- [[pycram.locations.locations.CostmapLocation]] — entity — Merged-costmap base pose sampler: OccupancyCostmap + optional Visibility/Ring costmaps, 600 candidates, MSC-validated via `pycram.pose_validator`.
- [[pycram.locations.locations.AccessingLocation]] — entity — Base pose sampler for articulated containers: OccupancyCostmap + GaussianCostmap, tries all arm chains, MSC-validates via `pycram.pose_validator`.
- [[pycram.locations.locations.GiskardLocation]] — entity — Full MSC-executor base pose test: constructs its own `Executor` directly; only 10 candidates; yields root global pose.
- [[pycram.pose_validator]] — entity — Pose validation helpers: `visibility_validator` (ray-test), `pose_sequence_reachability_validator` (full MSC, saves/restores world state), `collision_check`.

## krrood EQL (used by pycram)

- [[krrood.entity_query_language.core.variable.Variable]] — entity — EQL atomic variable: typed lazy re-enterable domain iterable; Literal/InstantiatedVariable/ExternallySetVariable subclasses.
- [[krrood.entity_query_language.predicate.Predicate]] — entity — EQL boolean predicate with dual-mode `__new__`: concrete evaluation or symbolic InstantiatedVariable depending on argument type.
- [[krrood.entity_query_language.backends.QueryBackend]] — entity — EQL evaluation strategy; four implementations (EntityQueryLanguage in-process, ProbabilisticBackend, SQLAlchemy, GenerativeBackend ABC).
- [[krrood.entity_query_language.query.Query]] — entity — EQL query DAG node with fluent API (.where/.having/.limit/.group_by); `build()` wires the DAG; `SetOf` vs `Entity` subtypes; build-lock; evaluate() delegates to QueryBackend.

## Perception

- [[pycram.perception.PerceptionQuery]] — entity — Queries existing SDT SemanticAnnotations by type, filters by BoundingBox region and camera visibility; returns matching Body objects. `from_robokudo()` is a stub.

## Gripper predicates

- [[pycram.querying.predicates]] — entity — `GripperIsFree` / `GripperIsNotFree`: kinematic-tree-based gripper occupancy predicates; check bodies attached under TCP in the world graph.

## Enums

- [[pycram.datastructures.enums.Arms]] — entity — `IntEnum`: LEFT / RIGHT / BOTH; arm selector used by PickUpAction, PlaceAction, MoveGripperMotion, GraspPose.
- [[pycram.datastructures.enums.ApproachDirection]] — entity — `Enum`: FRONT / BACK / LEFT / RIGHT; face of object bounding box approached by gripper; encodes (AxisIdentifier, ±1) pair.
- [[pycram.datastructures.enums.VerticalAlignment]] — entity — `Enum`: TOP / BOTTOM / NoAlignment; vertical tilt of gripper during grasp; encodes (AxisIdentifier, ±1 or 0) pair.
- [[pycram.datastructures.enums.AxisIdentifier]] — entity — `Enum`: X / Y / Z / Undefined; Cartesian axis selector used as encoding component in ApproachDirection and VerticalAlignment.

## Stubs

_(no remaining stubs in pycram package)_
