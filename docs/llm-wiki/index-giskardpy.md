# giskardpy — entity index

Complete entity listing for the `giskardpy` package. See [[index]] for Concepts, Bridges, and per-package overview pages.

---

## Model layer

- [[giskardpy.model.world_config]] — entity — `WorldConfig` hierarchy: standalone world init from URDF (`WorldWithFixedRobot`, OmniDrive, DiffDrive) or DB (`WorldFromDatabaseConfig`); bypassed when driven via pycram.

## Motion State Chart

- [[giskardpy.motion_statechart.motion_statechart.MotionStatechart]] — entity — Compiled MSC graph; runs tick loop updating trinary lifecycle/observation states and feeding constraints to QP.
- [[giskardpy.motion_statechart.graph_node.MotionStatechartNode]] — entity — Abstract base for MSC nodes; four `TrinaryCondition` lifecycle transitions; build/tick/start/pause/end/reset hooks.
- [[giskardpy.motion_statechart.graph_node.TrinaryCondition]] — entity — Trinary condition type (TRUE/FALSE/UNKNOWN); symbolic guard for all four lifecycle transitions.
- [[giskardpy.motion_statechart.graph_node.NodeArtifacts]] — entity (stub) — Dataclass returned by `build()`; carries constraints, observation expression, debug expressions.
- [[giskardpy.motion_statechart.graph_node.Goal]] — entity — Composite MSC node; bundles `EndMotion`, `CancelMotion`, `ThreadPayloadMonitor`, `Sequence`, `Parallel`.
- [[giskardpy.motion_statechart.graph_node.EndMotion]] — entity (stub) — Success-termination MSC node; fires when start_condition becomes true; see Goal bundled page.
- [[giskardpy.motion_statechart.graph_node.CancelMotion]] — entity (stub) — Failure-termination MSC node; raises exception inside control loop on RUNNING; see Goal bundled page.
- [[giskardpy.motion_statechart.graph_node.Task]] — entity — MSC leaf node carrying QP motion constraints; returned by every `BaseMotion._motion_chart`.
- [[giskardpy.motion_statechart.monitors]] — entity — All monitor families: convergence (`LocalMinimumReached`), payload (timing/counting), cartesian geometry, joint position.
- [[giskardpy.motion_statechart.goals]] — entity — Concrete Goal subclasses: `DifferentialDriveBaseGoal`, `CartesianPoseStraight`, `Open`/`Close`, collision avoidance.
- [[giskardpy.motion_statechart.tasks]] — entity — Concrete Task subclasses: `CartesianPosition`, `CartesianOrientation`, `CartesianPose`, `JointPositionList`.
- [[giskardpy.motion_statechart.context]] — entity — `MotionStatechartContext`: execution substrate binding SDT World, QPControllerConfig, collision managers, and extension registry.

## QP Layer

- [[giskardpy.qp.constraint]] — entity — Constraint hierarchy: BaseConstraint, Integral (EqualityConstraint/InequalityConstraint), Derivative (DerivativeEquality/DerivativeInequality).
- [[giskardpy.qp.constraint_collection.ConstraintCollection]] — entity — Constraint container; high-level factory methods; lifecycle gating via `link_to_motion_statechart_node`.
- [[giskardpy.qp.adapters]] — entity — `QPDataSymbolic` + `ProblemDataPart` hierarchy: assembles full symbolic QP (H, g, lb, ub, E, A) from DOF list and ConstraintCollection; FK binding.
- [[giskardpy.qp.qp_data]] — entity — `QPDataExplicit` (8-array explicit format) and `QPDataTwoSidedInequality` (flat unified format); `apply_filters()` removes zero-weight slack variables.
- [[giskardpy.qp.pos_in_vel_limits]] — entity — Symbolic velocity profile functions: `shifted_velocity_profile`, `compute_slowdown_asap_vel_profile`, `acc_cap` (Gaussian formula); position-aware MPC velocity bounds.
- [[giskardpy.qp.qp_data_factories]] — entity — `QPDataFactory[T]` + `QPDataExplicitFactory` + `QPDataTwoSidedInequalityFactory`; compile CasADi expressions → evaluate numerically each tick.
- [[giskardpy.qp.qp_controller]] — entity — QPController + QPControllerConfig + QPSolver (PIQP/Gurobi/QPALM/qpSWIFT) + Executor tick loop.
