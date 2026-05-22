# Wiki log

Append-only chronological record of ingest, lint, schema, and query events.
Newest entries at the bottom. Entry format defined in [CLAUDE.md §7](CLAUDE.md).

---

## [2026-05-17] schema | initial CLAUDE.md committed
- **scope:** schema bootstrap
- **created:** [[CLAUDE.md]], [[index.md]], [[log.md]]
- **notes:** Initial schema version. Vault structure: `wiki/{packages,concepts,entities,bridges}` + `raw/{snapshots,notes,pr-digests}`. ID convention is fully-qualified dotted path (e.g. `pycram.plans.Designator`). Frontmatter requires `id`, `kind`, `package`, `status`, optional `source_paths` with commit SHA.

## [2026-05-17] ingest | pycram.designators (first ingest)
- **scope:** the Designator concept in pycram, as it actually lives in the codebase (not in `pycram/designators/`, which is empty)
- **created:** [[pycram]], [[concept.designator]], [[pycram.plans.Designator]], [[pycram.robot_plans.ActionDescription]], [[pycram.robot_plans.BaseMotion]], [[pycram.plans.DesignatorNode]], [[pycram.plans.factories.make_node]]
- **updated:** [[index.md]]
- **findings:**
  - The directory `pycram/src/pycram/designators/` is empty (all files 0 bytes). The actual Designator hierarchy lives in `pycram/src/pycram/plans/designator.py` (the base) and `pycram/src/pycram/robot_plans/{actions,motions}/base.py` (the two main subclasses). This is a non-obvious split likely reflecting a planned-or-in-progress refactor. **Flagged as open question on [[concept.designator]]**.
  - The plan graph wraps Designators in `DesignatorNode` (parent of `ActionNode`, `MotionNode`). Dispatch is via `pycram.plans.factories.make_node`. The Designator class itself does not know about its node; the wrapper sets `designator.plan_node` in `__post_init__`.
  - Designators are dataclasses. Properties `plan`, `robot`, `world`, `context` are *delegated* through `plan_node` — accessing them before the node is attached raises `ContextIsUnavailable`.
  - Cross-package dependencies of the Designator concept: `semantic_digital_twin.world.World`, `semantic_digital_twin.robots.abstract_robot.AbstractRobot`, `giskardpy.motion_statechart.graph_node.Task`, `krrood.entity_query_language` (for symbolic pre/post-conditions on actions). These will be candidates for [[bridge.pycram-sdt]] and [[bridge.pycram-giskardpy]] in later ingests.
- **open questions:**
  - Is `pycram/designators/` intentionally a future home (refactor target) or vestigial? See [[concept.designator]].
  - The `Designator.fields` classmethod uses `@classmethod` + `@property` — this stack only works on Python <3.13. Worth checking whether the project pins Python version. Logged on [[pycram.plans.Designator]].

## [2026-05-17] schema | v2 — navigation, stub policy, size budget, filename rule
- **scope:** CLAUDE.md hardening based on first-ingest experience
- **updated:** [[CLAUDE.md]]
- **changes:**
  - **§4 ID and link conventions** — added deterministic filename rule (`wiki/<kind-plural>/<id>.md`) so any `[[id]]` resolves to a path without searching.
  - **§5 Body sections** — added page-size budget (entity ≤200 lines, concept ≤300, package/bridge ≤150). When a page outgrows it, promote a subsection.
  - **§6 Ingest workflow** — added step 9: scan touched pages for `[[links]]` whose target file is missing, and create stubs in the same ingest.
  - **§12 Navigation / lookup workflow (NEW)** — explicit recipe for an agent *using* the wiki to answer a question. Read CLAUDE → index → start page → follow links on demand → fall through to source only if wiki is silent (and log the gap as a future ingest).
  - **§13 Stub policy (NEW)** — rule that every ID in `uses`/`used_by`/`[[link]]` must have a file. Defines minimal stub frontmatter and body. Stubs participate in `used_by` symmetry from day one.
- **findings:** the first ingest produced 4 dangling references (`PlanNode`, `UnderspecifiedNode`, `Context`, `MotionExecutor` etc.) that I left unhandled. The new §13 stub policy + §6 step 9 closes that loophole.

## [2026-05-17] ingest | pycram.plans core (Phase 1)
- **scope:** the three classes that make up the plan-graph machinery — `Plan`, `PlanNode`, `PlanEntity` — read at commit `0528d8cf3`. Sources: `pycram/src/pycram/plans/{plan,plan_node,plan_entity}.py`.
- **created:** [[pycram.plans.Plan]], [[pycram.plans.PlanNode]], [[pycram.plans.PlanEntity]], [[pycram.datastructures.Context]] (stub), [[pycram.plans.UnderspecifiedNode]] (stub), [[pycram.plans.failures.PlanFailure]] (stub), [[pycram.plans.plan_callbacks.PlanCallback]] (stub)
- **updated:** [[pycram]], [[concept.designator]], [[pycram.plans.Designator]], [[pycram.plans.DesignatorNode]], [[pycram.plans.factories.make_node]], [[pycram.robot_plans.ActionDescription]], [[index.md]] — all to add missing entries in `uses` / `used_by` for the new pages and stubs.
- **findings:**
  - `Plan` is a `rustworkx.PyDiGraph[PlanNode]` with a `Context` that itself is a `PlanEntity` — that's why `add_plan_entity` works for both nodes and the context.
  - The plan is expected to be a **tree** (`validate()`: edges == nodes − 1 + exactly one root). `add_edge` explicitly maintains `target.layer_index` because rustworkx doesn't preserve child order.
  - `PlanNode.perform()` (the public entry) checks every ancestor for `INTERRUPTED` before running — that's how interruption propagates downward.
  - `Plan.actions` is a typed convenience: `[node for node in self.nodes if isinstance(node, ActionNode)]` — `ActionNode` is a peer of `DesignatorNode` (currently bundled there; Phase 2 will split).
  - `Plan.remove_node` sets `node.world = None` even though `PlanNode` has no declared `world` attribute — flagged as **Open question on [[pycram.plans.Plan]]**.
  - `Plan.world` and `Plan.robot` will `AttributeError` if `context is None`; not guarded. Flagged.
- **open questions:**
  - Confirm the `node.world` write in `Plan.remove_node` (latent bug vs. dynamic attr set elsewhere).
  - Confirm `context` is a hard precondition for `Plan.world`/`Plan.robot`.
  - Resolve `TaskStatus.PAUSE` vs `PAUSED` naming (see [[pycram.plans.PlanNode]] Open questions).

## [2026-05-17] ingest | pycram.plans Phase 2 (ActionNode / MotionNode / UnderspecifiedNode)
- **scope:** `pycram/src/pycram/plans/plan_node.py:301-489` — the three concrete `PlanNode` subclasses bundled in the same module. Read at commit `0528d8cf3`.
- **created:** [[pycram.plans.ActionNode]], [[pycram.plans.MotionNode]], [[pycram.motion_executor.MotionExecutor]] (stub), [[pycram.datastructures.ExecutionData]] (stub)
- **updated:** [[pycram.plans.UnderspecifiedNode]] (stub → full page), [[pycram.plans.DesignatorNode]] (restructured to abstract-base only; subclasses now have own pages), [[pycram.plans.PlanNode]] (subclass list confirmed; added `used_by: UnderspecifiedNode`), [[pycram.plans.factories.make_node]] (added `uses: ActionNode, MotionNode`), [[index.md]]
- **findings:**
  - `ActionNode._perform()` is a four-step orchestrator: pre-capture → `action.perform()` → MSC construct+execute → post-capture. The `parent_action_node` guard in `collect_motions()` is what makes nested-action motion ownership unambiguous.
  - `MotionNode` is a passive leaf in normal execution — it is harvested by `ActionNode.collect_motions()` before any individual `_perform()` is called. Its own `_perform()` runs only in isolated replay.
  - `UnderspecifiedNode` lazily initializes its iterator (`query_backend.evaluate(match)`) and persists it across calls. The try-until-success loop creates `ActionNode` children dynamically and attaches them via `add_child`.
  - `MotionExecutor` lives at `pycram.motion_executor` (not under `pycram.plans`) and `ExecutionData` at `pycram.datastructures.execution_data` — both imported at the top of `plan_node.py`.
- **open questions:**
  - `ActionNode` field `_world_modification_block_length_pre_perform` vs runtime attr `_last_world_modification_block_pre_perform_index` — naming inconsistency; check Phase 4.
  - `UnderspecifiedNode._perform()` returns silently on iterator exhaustion (no `PlanFailure`). Caller semantics unclear; check Phase 3 when `pycram.language` is ingested.

## [2026-05-17] ingest | pycram.plans Phase 3 (language nodes, combinators, failures)
- **scope:** `pycram/src/pycram/language.py` (226 lines), `pycram/src/pycram/plans/factories.py` (147 lines), `pycram/src/pycram/plans/failures.py` (110 lines). Read at commit `0528d8cf3`.
- **created:** [[concept.plan-language]], [[pycram.language.LanguageNode]], [[pycram.plans.factories]], [[pycram.fluent.Fluent]] (stub), [[sdt.spatial_types.Pose]] (stub), [[sdt.world_description.world_entity.Body]] (stub)
- **updated:** [[pycram.plans.failures.PlanFailure]] (stub → full page), [[pycram.plans.PlanNode]] (resolved simplify() open question), [[pycram.plans.factories.make_node]] (resolved ActionLike open question; added used_by factories), [[index.md]]
- **findings:**
  - `LanguageNode.simplify()` uses `type(child) != type(self)` — **exact type equality**, not `isinstance`. Only same-type children are flattened; heterogeneous nesting is preserved as-is.
  - `MonitorNode.__post_init__` starts its monitor thread **immediately on construction**, before `perform()` is called. `RESUME` behavior also calls `self.pause()` in `__post_init__`, placing the node in a paused state from the start.
  - `_make_plan_from_type_and_children` uses `mount_subplan` for `LanguageNode` children (migrates the sub-plan's graph) vs `add_child` for leaf nodes — the key difference for nested combinator composition.
  - `failures.py` imports `sdt.spatial_types.Pose` and `sdt.world_description.world_entity.Body` at **runtime** (not under `TYPE_CHECKING`), making it the first hard cross-package coupling point between pycram and SDT in the source. Two stubs created as Phase 5/6 targets.
  - `PlanFailure` inherits from `krrood.exceptions.DataclassException`, not Python's `Exception` — it is a krrood dataclass exception throughout.
- **open questions:**
  - `UnderspecifiedNode` silent exhaustion (from Phase 2): confirmed that `TryInOrderNode` and `TryAllNode` do raise `AllChildrenFailed` on exhaustion, but `UnderspecifiedNode` does not — callers that rely on UnderspecifiedNode failure must check node status explicitly.
  - `AllChildrenFailed.language_node` is a runtime forward reference (`TYPE_CHECKING`-only import); depends on krrood `DataclassException` supporting string annotations correctly.

## [2026-05-17] ingest | pycram.robot_plans Phase 4 (concrete actions + gripper motions)
- **scope:** `actions/core/pick_up.py` (264 lines), `actions/core/placing.py` (134 lines), `motions/gripper.py` (215 lines), `actions/core/navigation.py` (91 lines). Read at commit `0528d8cf3`.
- **created:** [[pycram.robot_plans.actions.core.PickUpAction]], [[pycram.robot_plans.actions.core.PlaceAction]], [[pycram.robot_plans.actions.core.NavigateAction]], [[pycram.robot_plans.motions.gripper]], [[pycram.datastructures.grasp.GraspDescription]] (stub), [[pycram.view_manager.ViewManager]] (stub), [[sdt.datastructures.GripperState]] (stub)
- **updated:** [[pycram.robot_plans.ActionDescription]] (added concrete action subclasses to `used_by`), [[pycram.robot_plans.BaseMotion]] (added gripper motions module to `used_by`), [[index.md]]
- **findings:**
  - `PickUpAction.execute` is two-phase: (1) sequential reach+grasp, (2) world mutation to attach body to end-effector, then lift via TRANSLATION motion. The world mutation happens **inside `execute()`**, between two `.perform()` calls — unusual timing that may affect replay semantics.
  - `PlaceAction` recovers the prior grasp description by traversing plan history via `get_previous_node_by_designator_type(PickUpAction)`. Falls back to a default front-approach if no prior pick-up exists. Detaches the body by replacing its SDT parent connection with a `Connection6DoF` to the world root.
  - All four motions in `gripper.py` override `perform()` as `return` (no-op). The real execution is entirely through `_motion_chart` → MSC. This confirms the invariant from [[pycram.plans.MotionNode]].
  - `MoveToolCenterPointMotion` selects its root link based on `robot.full_body_controlled`: world root for mobile manipulators, robot root for fixed-base robots.
  - `NavigateAction` is the simplest core action: wraps exactly one `MoveMotion` in `execute_single`.
- **open questions:**
  - `GraspDescription.grasp_pose_sequence` vs `._pose_sequence` — two different method signatures with different callers (PickUpAction vs ReachAction). Semantics unclear; check during GraspDescription ingest.
  - `PickUpAction` world mutation inside `execute()`: whether `re_perform()` / `replay()` correctly re-runs the attachment needs verification.

## [2026-05-17] ingest | Phase 5 — cross-package bridges + SDT/giskardpy entry points
- **scope:** `semantic_digital_twin/src/semantic_digital_twin/world.py` (1961 lines), `semantic_digital_twin/src/semantic_digital_twin/robots/abstract_robot.py` (676 lines), `giskardpy/src/giskardpy/motion_statechart/graph_node.py` (lines 320–1038). Read at commit `0528d8cf3`.
- **created:** [[bridge.pycram-sdt]], [[bridge.pycram-giskardpy]], [[sdt.world.World]], [[sdt.robots.abstract_robot.AbstractRobot]], [[giskardpy.motion_statechart.graph_node.Task]], [[giskardpy.motion_statechart.graph_node.MotionStatechartNode]] (stub)
- **updated:** [[pycram.datastructures.Context]] (added sdt.world.World and sdt.robots.abstract_robot.AbstractRobot to uses), [[pycram.robot_plans.BaseMotion]] (added giskardpy.motion_statechart.graph_node.Task to uses), [[sdt.spatial_types.Pose]] (added bridge.pycram-sdt to used_by), [[sdt.world_description.world_entity.Body]] (added bridge.pycram-sdt to used_by), [[index.md]] (restructured into pycram/SDT/giskardpy/bridges sections)
- **findings:**
  - `giskardpy.motion_statechart.graph_node.Task` is a minimal dataclass (adds only `weight: float`) that subclasses the 500-line `MotionStatechartNode`. The lifecycle machinery (four `TrinaryCondition` conditions, `LifeCycleVariable`, `ObservationVariable`, `build()`/`on_tick()`/`on_start()` hooks) lives entirely in `MotionStatechartNode`.
  - `sdt.world.World.move_branch_with_fixed_connection` is decorated `@atomic_world_modification`, so the pycram PickUpAction caller gets atomicity/locking for free — the concern logged in Phase 4 is resolved.
  - `pycram.plans.failures` imports `sdt.spatial_types.Pose` and `sdt.world_description.world_entity.Body` at **module load time** (not `TYPE_CHECKING`). This is the earliest compile-time coupling point between pycram and SDT. Logged on [[bridge.pycram-sdt]].
  - `AbstractRobot.from_world()` follows a 6-step setup protocol (`_init_empty_robot` → `_setup_semantic_annotations` → `_setup_collision_rules` → `_setup_velocity_limits` → `_setup_hardware_interfaces` → `_setup_joint_states`). Concrete robot subclasses override the steps they need.
- **open questions:**
  - Concrete `AbstractRobot` subclasses (robot-model specific) not yet ingested — Phase 6 target.
  - `World.__deepcopy__` structured clone: unclear if `semantic_annotations` and `actuators` are deep or shallow copied. Phase 6 target.
  - `MotionStatechart` graph assembly and QP controller details deferred to Phase 7.

## [2026-05-17] ingest | Phase 6 — SDT package + concept.world
- **scope:** `semantic_digital_twin/src/semantic_digital_twin/world_description/world_entity.py` (lines 1–510, 780–880), `spatial_types/spatial_types.py` (lines 1769–1850), `__init__.py`. Read at commit `0528d8cf3`.
- **created:** [[sdt]] (package), [[concept.world]], [[sdt.world_description.world_entity.Connection]] (stub), [[sdt.world_description.world_entity.KinematicStructureEntity]] (stub)
- **updated:** [[sdt.world_description.world_entity.Body]] (stub → full page), [[sdt.spatial_types.Pose]] (stub → full page), [[sdt.world.World]] (added concept.world to used_by), [[index.md]] (added sdt package, concept.world, promoted Body/Pose from stubs, new KSE/Connection stubs)
- **findings:**
  - `Connection` uses a strict three-transform structure (`parent_T_connection` + `_kinematics` + `connection_T_child`). Both fixed transforms are validated at construction to contain no CasADi free variables — the invariant is enforced eagerly, not lazily.
  - `Pose` is backed by a CasADi SX 4×4 matrix, enabling symbolic FK chains. This is the mechanism that allows giskardpy to differentiate FK expressions in the QP solver — the pose is never just a numpy array at the SDT layer.
  - SDT has 13 subpackages including a full reasoning engine (`reasoning/`), ORM layer (`orm/`), and 17 pre-configured robot models in `robots/`. Coverage so far: `world.py`, `world_description/world_entity.py` (partial), `robots/abstract_robot.py`, `spatial_types/spatial_types.py` (Pose only).
  - `Body` is explicitly a "semantic atom" — the docstring says it cannot be decomposed into meaningful smaller parts. This distinguishes it from `Region` (spatial labelling) and is the semantically correct object to use as the pick/place target in pycram actions.
- **open questions:**
  - `Region` role in FK and world traversal not confirmed.
  - Full FK traversal code lives in `spatial_computations/` — not yet ingested.
  - `World.__deepcopy__` shallow vs deep copy of `semantic_annotations` / `actuators` remains unconfirmed (same open question from Phase 5).

## [2026-05-17] ingest | Phase 7 — giskardpy package + concept.motion-statechart
- **scope:** `giskardpy/src/giskardpy/motion_statechart/motion_statechart.py` (full, ~540 lines), `graph_node.py` (lines 885–1038 for Goal/EndMotion context), `__init__.py`. Read at commit `0528d8cf3`.
- **created:** [[giskardpy]] (package), [[concept.motion-statechart]], [[giskardpy.motion_statechart.motion_statechart.MotionStatechart]]
- **updated:** [[giskardpy.motion_statechart.graph_node.Task]] (added MotionStatechart and concept.motion-statechart to used_by), [[giskardpy.motion_statechart.graph_node.MotionStatechartNode]] (added MotionStatechart and concept.motion-statechart to used_by), [[pycram.motion_executor.MotionExecutor]] (stub updated: added MotionStatechart+Task to uses, improved body), [[index.md]]
- **findings:**
  - `MotionStatechart.compile()` follows a strict five-step pipeline. The compiled CasADi functions are bound to numpy memory views via `bind_args_to_memory_view` — making `tick()` zero-copy with no Python-level data movement per node. This is the key performance design of the MSC control loop.
  - `combine_constraint_collections_of_nodes()` iterates ALL nodes (not just RUNNING ones). Whether inactive nodes produce zero-weight constraints or are genuinely excluded depends on `link_to_motion_statechart_node` behaviour — not yet confirmed from QP layer.
  - `LifeCycleState` and `ObservationState` are both `MutableMapping[MotionStatechartNode, float]` backed by a numpy array indexed by `node.index`. The compiled CasADi function for lifecycle transitions uses `if_eq_cases` (matching on the current lifecycle enum value) to produce the next state vector in one vectorised call.
  - `StateHistory` deduplicates consecutive identical states — only appends when something actually changes. Used for Gantt chart plotting and post-execution debugging.
- **open questions:**
  - QP controller internals (`QPController`, `QPDataFactory`, `QPSolver`) not yet ingested.
  - Whether lifecycle gating is applied automatically to constraint weights in `link_to_motion_statechart_node` is unconfirmed.

## [2026-05-17] schema | v3 — per-package entity index split
- **scope:** `doc/llm-wiki/` vault reorganisation only; no source code read.
- **created:** [[index-pycram]], [[index-sdt]], [[index-giskardpy]]
- **updated:** [[index.md]] (slimmed to Packages/Concepts/Bridges + pointers to per-package indexes), [[CLAUDE.md]] (§4 added per-package index rule; §12 updated lookup step 2 to reference per-package indexes)
- **findings:**
  - `index.md` was growing unwieldy with ~35 entity lines. Three per-package index files now own the entity listings, making it practical for an LLM agent to load only the relevant package index rather than the full entity list.

## [2026-05-17] ingest | Phase 8 — pycram stub closure (7 stubs → full pages)
- **scope:** `pycram/src/pycram/datastructures/dataclasses.py`, `motion_executor.py`, `datastructures/execution_data.py`, `plans/plan_callbacks.py`, `fluent.py`, `datastructures/grasp.py`, `view_manager.py`. All read at commit `0528d8cf3`.
- **created:** (none)
- **updated:** [[pycram.datastructures.Context]] (stub → full), [[pycram.motion_executor.MotionExecutor]] (stub → full), [[pycram.datastructures.ExecutionData]] (stub → full), [[pycram.plans.plan_callbacks.PlanCallback]] (stub → full), [[pycram.fluent.Fluent]] (stub → full), [[pycram.datastructures.grasp.GraspDescription]] (stub → full), [[pycram.view_manager.ViewManager]] (stub → full), [[index-pycram]] (all stubs reflect promoted status)
- **findings:**
  - `MotionExecutor` wraps all Tasks in a giskardpy `Sequence` goal before constructing the MSC, then adds `EndMotion.when_true(sequence_node)`. The `execution_type` class variable is a true global mode switch — `ExecutionEnvironment` context managers save/restore it, enabling nested environments cleanly.
  - `MotionExecutor._execute_for_simulation()` tick loop runs up to 2,000 iterations; `QPControllerConfig` is constructed inline with `target_frequency=50`. This gives a hard 40 s timeout with no per-plan override mechanism visible in the source.
  - `Fluent` uses a `None`-means-falsy contract throughout: `get_value()` returns `None` when the condition is inactive. `wait_for()` blocks until non-`None`. This asymmetry (True vs None instead of True vs False) is intentional — it allows the reactive network to distinguish "not yet evaluated" from "evaluated False".
  - `GraspDescription._pose_sequence` accesses bounding box geometry from `Body.collision` and FK from `World.transform`. The lift pose is computed in the map frame then re-expressed in the target frame — a non-obvious two-step that avoids rotation contamination of the z-lift offset.
  - `PlanCallback` is a 14-line `@dataclass` extending `PlanEntity`; both methods are no-ops (`...`). Simpler than expected — no ABC, no Protocol.
- **open questions:**
  - `MotionExecutor` tick loop: 2,000-iteration / 40 s hard limit has no per-plan override. Is there a mechanism elsewhere (e.g. `PlanNode.interrupt`) that cuts the loop before timeout? Logged for Phase 9.
  - `GraspDescription.place_pose_sequence` reads `manipulator.tool_frame.child_kinematic_structure_entities[0]` — assumes exactly one child. Multi-body grasps would silently use the wrong body.

## [2026-05-17] ingest | Phase 9 — pycram action+motion breadth
- **scope:** `pycram/src/pycram/robot_plans/motions/{navigation,container,robot_body,misc}.py`, `robot_plans/actions/core/{container,robot_body,misc}.py`, `robot_plans/actions/composite/{facing,searching,transporting,tool_based}.py`. All read at commit `0528d8cf3`.
- **created:** [[pycram.robot_plans.motions.navigation]], [[pycram.robot_plans.motions.container]], [[pycram.robot_plans.motions.robot_body]], [[pycram.robot_plans.motions.misc]], [[pycram.robot_plans.actions.core.container]], [[pycram.robot_plans.actions.core.robot_body]], [[pycram.robot_plans.actions.core.misc]], [[pycram.robot_plans.actions.composite]], [[pycram.locations.locations.CostmapLocation]] (stub), [[pycram.perception.PerceptionQuery]] (stub), [[sdt.world_description.world_entity.Region]] (stub)
- **updated:** [[index-pycram]] (new actions/motions sections, stubs added), [[index-sdt]] (Region stub added)
- **findings:**
  - `DetectingMotion._motion_chart` returns `None` (`pass`) — the only motion in the codebase with no giskardpy Task. Detection is driven by `PerceptionQuery.from_world()` in `DetectAction.execute()` directly, not via the QP controller.
  - `TransportAction` uses `underspecified(NavigateAction)` with `variable(Pose, domain=CostmapLocation(...))` for the place-navigation step — the navigation target is grounded lazily via krrood EQL at execution time. This is the same mechanism as `UnderspecifiedNode` but used inline inside an action's `execute()`, not in the plan graph.
  - `SearchAction` detects success by diffing `world.semantic_annotations` IDs before/after the `try_in_order` sweep — it does not check the return value of `DetectAction.execute()`. This means detection results must be side-effected into the world by `PerceptionQuery.from_world()`.
  - `LookingMotion._motion_chart` mutates `camera.forward_facing_axis.reference_frame` as a side effect each time the property is accessed. Concurrent access from two plan threads would race on this field.
  - `EfficientTransportAction` in `tool_based.py` uses `RobotDescription.current_robot_description` and `BelieveObject` — legacy pycram-bullet patterns incompatible with the current SDT architecture. Likely in-migration or vestigial.
- **open questions:**
  - How does `PerceptionQuery.from_world()` populate `world.semantic_annotations`? The call returns a value but `DetectAction.execute()` discards it — the expected side effect must happen inside `from_world()`. Expand in future `pycram.perception` ingest.
  - `CostmapLocation.ground()` returns a `GraspPose` or raises `BodyUnfetchable`. The exact sampling algorithm (costmap type, resolution, collision checks) is unknown until `pycram.locations` is ingested.
  - `MotionExecutor` receives tasks from `MotionNode._motion_chart`. If `DetectingMotion` is used as a `MotionNode` child, `None` would land in `executor.motions`. Is this guarded? (Phase 10 candidate.)

## [2026-05-17] ingest | Phase 10 — krrood EQL concept + AlternativeMotion
- **scope:** `krrood/src/krrood/entity_query_language/{factories.py,backends.py,core/variable.py,predicate.py}`, `pycram/src/pycram/robot_plans/actions/base.py`, `pycram/src/pycram/alternative_motion_mapping.py`. All read at commit `0528d8cf3`.
- **created:** [[concept.krrood-eql]], [[pycram.alternative_motion_mapping.AlternativeMotion]], [[krrood.entity_query_language.core.variable.Variable]] (stub), [[krrood.entity_query_language.predicate.Predicate]] (stub), [[krrood.entity_query_language.backends.QueryBackend]] (stub)
- **updated:** [[index.md]] (concept.krrood-eql added), [[index-pycram.md]] (AlternativeMotion + krrood stubs added)
- **findings:**
  - EQL has two distinct evaluation paths in pycram: **selective** (`variable(T, domain)` filters existing objects) and **generative** (`underspecified(Constructor)` builds new instances from all domain combinations). `EntityQueryLanguageBackend._evaluate_underspecified` enumerates all `variable(...)` domain cross-products and constructs an instance for each, then filters by `.where()` conditions.
  - `ActionDescription.bound_variables` wraps each field value in a singleton-domain `Variable` — so `variable(type(arm), [arm])` where `arm=Arms.RIGHT`. This lets `pre_condition` compose concrete runtime values with symbolic predicates (e.g. `GripperIsFree(manipulator)`) in a uniform EQL expression.
  - `evaluate_condition(condition)` = `any(condition.evaluate())` — it stops at the first satisfying binding. This means pre-condition evaluation is **not exhaustive** — any valid assignment suffices.
  - `AlternativeMotion.__subclasses__()` dispatch has a silent ordering ambiguity: if two alternatives both match, the first in `__subclasses__()` iteration order wins. No priority mechanism is defined.
  - `@symbolic_function` is the bridge between Python functions and EQL: when any argument is a `Variable`, the function call is deferred as an `InstantiatedVariable`; when all arguments are concrete, it runs normally. `ViewManager.get_end_effector_view` is decorated this way, making it usable both in plans and in EQL pre-conditions.
- **open questions:**
  - Where `AlternativeMotion.check_for_alternative` is invoked in the execution path — `ActionNode`, `MotionNode`, or elsewhere — is not confirmed from these files alone.
  - `ProbabilisticBackend` uses `FullyFactorizedRegistry` by default; how priors are provided for `Pose`-typed action parameters is undocumented in pycram.
  - Whether `UnderspecifiedNode` uses `context.query_backend.evaluate()` directly or goes through some intermediate layer (lazy iterator protocol) is worth confirming in a future ingest.

## [2026-05-17] ingest | Phase 11 — SDT kinematic structure depth
- **scope:** `semantic_digital_twin/src/semantic_digital_twin/world_description/{world_entity.py (lines 280-938), connections.py (full ~1148 lines), degree_of_freedom.py, shape_collection.py}`. All read at commit `0528d8cf3`.
- **created:** [[sdt.world_description.connections]], [[sdt.world_description.degree_of_freedom.DegreeOfFreedom]], [[sdt.world_description.shape_collection.ShapeCollection]]
- **updated:** [[sdt.world_description.world_entity.KinematicStructureEntity]] (stub → stable), [[sdt.world_description.world_entity.Connection]] (stub → stable), [[sdt.world_description.world_entity.Region]] (stub → stable), [[index-sdt.md]] (new entries + stub markers removed)
- **findings:**
  - `Connection` splits its transform into three sequential terms: `parent_T_connection` (constant fixed frame offset) + `_kinematics` (variable, set by subclass `add_to_world`) + `connection_T_child` (constant fixed offset). `__post_init__` enforces at construction time that the two constant frames are truly constant (no free CasADi variables). This means you cannot express asymmetric connections whose offsets change at runtime — all variability must flow through `_kinematics`.
  - `DegreeOfFreedom.create_variables()` must be called explicitly after the DOF is added to the world; before it, `variables` is an empty `DerivativeMap`. This is a mandatory two-phase initialisation pattern and an easy source of `AttributeError` if the add-to-world step is bypassed.
  - `OmniDrive` and `DifferentialDrive` both implement `HasUpdateState` with `update_state()` that resets velocity DOF positions to `0` each cycle. This means the drive controller is expected to re-write velocity commands every tick — stale velocity commands are automatically zeroed rather than persisted.
  - `KinematicStructureEntity.get_first_parent_connection_of_type(T)` traverses ancestors by recursing on `parent_connection.parent`, not on `parent_kinematic_structure_entity`. The termination condition is reaching `world.root`; this raises `ValueError` instead of returning `None`, so callers must ensure the target type exists in the ancestor chain.
- **open questions:**
  - `Connection6DoF` exposes both a `dofs` property and seven named UUID fields. Whether the UUID fields are always kept in sync with `world.state` on write paths (e.g. via `origin.setter`) needs verification.
  - `Body.combined_mesh` delegates to `collision.combined_mesh`; `visual` is never used for FK/physics. Whether this is intentional (visual-only geometry is purely cosmetic) or an oversight is not clear from the source.

## [2026-05-17] ingest | Phase 12 — SDT computation + state
- **scope:** `semantic_digital_twin/src/semantic_digital_twin/spatial_computations/{forward_kinematics.py, ik_solver.py, raytracer.py}`, `world_description/world_state.py`, `datastructures/joint_state.py`, `datastructures/definitions.py`. All read at commit `0528d8cf3`.
- **created:** [[concept.forward-kinematics]], [[sdt.spatial_computations.forward_kinematics]], [[sdt.spatial_computations.ik_solver]], [[sdt.spatial_computations.raytracer]], [[sdt.world_description.world_state.WorldState]], [[sdt.datastructures.joint_state.JointState]]
- **updated:** [[sdt.datastructures.GripperState]] (stub → stable; clarified it is a `JointStateType` enum in `definitions.py`, NOT the `JointState` type alias in `joint_state.py`), [[index-sdt.md]] (spatial_computations section added, datastructures expanded), [[index.md]] (concept.forward-kinematics added)
- **findings:**
  - `ForwardKinematicsManager` uses the exact same zero-copy pattern as `MotionStatechart.tick()`: `compiled_all_fks.bind_args_to_memory_view(0, world.state.positions)` at compile time, then pure `evaluate()` each tick with no additional argument passing. This confirms that `WorldState._data` is the single shared memory region for both subsystems.
  - `WorldState.add_degree_of_freedom` is where `DegreeOfFreedom.create_variables()` is actually called — not by the DOF itself and not by the `Connection.add_to_world`. The two-phase DOF init (`add_to_world` → `add_degree_of_freedom`) is split across Connection and WorldState.
  - `sdt.datastructures` has two different things named `GripperState`: an enum in `definitions.py` (command selector: OPEN/CLOSE/MEDIUM) and a type alias `GripperState = JointState` in `joint_state.py` (target configuration). The pycram motion designators use the enum; the robot configuration system uses the alias. **Name collision logged as open question.**
  - The SDT IK solver uses `daqp` directly — the same solver type as giskardpy's QP controller but invoked independently, without any giskardpy infrastructure. This means SDT can solve IK offline (e.g. for base placement planning) without a running giskardpy instance.
  - `RayTracer` has no event registration — it relies on callers invoking `update_scene()` before each render query. The `version` counter on `WorldState` makes this lazy check cheap.
- **open questions:**
  - The `InverseKinematicsSolver` rotates the tip frame by a small angle (`-0.0001` rad) before computing the quaternion error. The docstring does not explain this; it is likely a singularity avoidance hack but the exact failure mode is undocumented.
  - `WorldState._apply_control_commands` integrates downward from the written derivative, but never upward. If giskardpy writes jerk, only jerk is updated. Whether giskardpy always writes at velocity level (row 1) or sometimes at jerk/acceleration level is worth confirming.

## [2026-05-17] ingest | Phase 13 — SDT semantic layer + concrete robots
- **scope:** `semantic_digital_twin/src/semantic_digital_twin/semantic_annotations/{semantic_annotations.py, mixins.py}`, `reasoning/{reasoner.py, world_reasoner.py, predicates.py, queries.py}`, `robots/{tiago.py, panda.py}`, `collision_checking/collision_manager.py`, `world_description/world_entity.py` (SemanticAnnotation lines 555-650). All read at commit `0528d8cf3`.
- **created:** [[concept.semantic-annotation]], [[sdt.semantic_annotations.SemanticAnnotations]], [[sdt.reasoning.WorldReasoner]], [[sdt.robots.concrete]], [[sdt.collision_checking]]
- **updated:** [[index-sdt.md]] (Robots + Semantic layer sections added), [[index.md]] (concept.semantic-annotation added)
- **findings:**
  - `SemanticAnnotation` hash is computed from `type + sorted KSE UUIDs`, not object identity. This means two independently constructed annotation objects with the same type and same bodies are considered equal by the RDR. This is intentional (RDR equality checks) but means that replacing a body while keeping the UUID stable would create a hash collision with an annotation over the original body.
  - `WorldReasoner._update_world_attributes` calls `setattr(world, attr_name, attr_value)` for every RDR result key except `"semantic_annotations"` — the RDR can silently overwrite arbitrary World fields. No schema validation or allowlist is applied. This is a non-obvious side effect of calling `world_reasoner.reason()`.
  - `predicates.py::stable()` raises `NotImplementedError("Needs multiverse")` unconditionally. Any plan that evaluates `stable(obj)` as a pre-condition will fail at runtime until a multiverse physics backend is wired in. This is currently a dead branch in pre-condition evaluation.
  - Concrete robot models all follow the same construction idiom: `_init_empty_robot(world)` or `from_world(world)` → look up bodies by name (`world.get_body_by_name(...)`) → wire `Arm/Gripper/Finger` chain. The body names are hardcoded strings — robot models are tightly coupled to specific URDF link names.
  - `CollisionManager` is a `ModelChangeCallback` and is therefore automatically notified on world topology changes. But it is NOT a `StateChangeCallback` — state changes (joint position updates) do not trigger a re-check. Collision checking must be called explicitly each tick.
- **open questions:**
  - How `WorldReasoner.infer_semantic_annotations()` populates `world.semantic_annotations` is not fully traced — whether it calls `world.add_semantic_annotation()` or writes directly to the field needs verification.
  - `MaxAvoidedCollisionsOverride` caps tracked collision pairs — the selection criterion for which pairs survive (closest? alphabetical? random?) is not documented in the files read.


## [2026-05-17] ingest | Phase 14 — giskardpy MSC node hierarchy depth
- **scope:** `giskardpy/src/giskardpy/motion_statechart/{graph_node.py (full), data_types.py, monitors/monitors.py, monitors/payload_monitors.py, monitors/cartesian_monitors.py, monitors/joint_monitors.py, goals/templates.py, goals/cartesian_goals.py, goals/open_close.py, goals/collision_avoidance.py, tasks/cartesian_tasks.py, tasks/joint_tasks.py}`. All read at commit `0528d8cf3`.
- **created:** [[giskardpy.motion_statechart.graph_node.TrinaryCondition]], [[giskardpy.motion_statechart.graph_node.Goal]], [[giskardpy.motion_statechart.monitors]], [[giskardpy.motion_statechart.goals]], [[giskardpy.motion_statechart.tasks]]
- **updated:** [[giskardpy.motion_statechart.graph_node.MotionStatechartNode]] (stub → stable; full lifecycle/hook tables), [[index-giskardpy.md]] (7 MSC entries expanded from 3)
- **findings:**
  - The MSC lifecycle ancestor OR-chain (`_create_any_ancestor_condition_true`) means any ancestor firing its end/reset condition cascades to ALL descendants simultaneously — a structural invariant that simplifies goal abort logic at the cost of making individual subtree termination impossible without an explicit `CancelMotion`.
  - `ThreadedPayloadMonitor` (abstract in `monitors.py`) and `ThreadPayloadMonitor` (concrete base in `graph_node.py`) are two distinct classes with nearly identical names. The `graph_node.py` version starts a daemon thread in `__post_init__`; the `monitors.py` version is abstract and defers thread management to subclasses via `__call__`. This is a confusing naming duplication.
  - `CartesianTask.binding_policy` defaults to `Bind_on_start` — the goal pose FK is resolved the moment the node enters RUNNING, not at compile time. This is intentional for goals expressed in moving reference frames (e.g. approach relative to a moving object) but means the goal can silently change if `on_start` fires multiple times (after a reset).
  - `DifferentialDriveBaseGoal.expand()` computes intermediate orientation geometrically (vector from current to goal cross Z) and hardwires a three-step Sequence. If the robot is already at the goal position, step1 has a zero-length direction vector — potential degenerate case not guarded.
- **open questions:**
  - `CancelMotion.on_tick` raises the exception directly in the control loop. Whether the MSC's `cleanup()` is guaranteed to run before the exception propagates depends on `MotionExecutor`'s exception handling — needs verification (logged in [[giskardpy.motion_statechart.graph_node.Goal]]).
  - Two `ThreadedPayloadMonitor`-named classes — confirm whether `graph_node.ThreadPayloadMonitor` (no 'd') and `monitors.ThreadedPayloadMonitor` (with 'd') are intentionally distinct or an accidental naming drift.

## [2026-05-17] ingest | Phase 15 — giskardpy QP layer
- **scope:** `giskardpy/src/giskardpy/qp/{constraint.py, constraint_collection.py, qp_controller.py, qp_controller_config.py, solvers/qp_solver.py}`, `executor.py`. All read at commit `0528d8cf3`.
- **created:** [[concept.qp-controller]], [[giskardpy.qp.constraint]], [[giskardpy.qp.constraint_collection.ConstraintCollection]], [[giskardpy.qp.qp_controller]], [[giskardpy.motion_statechart.graph_node.NodeArtifacts]] (stub)
- **updated:** [[concept.motion-statechart]] (open question resolved: lifecycle gating confirmed; TrinaryCondition added to uses; used_by populated), [[bridge.pycram-giskardpy]] (open question resolved: full QP pipeline now documented; uses updated), [[giskardpy.motion_statechart.motion_statechart.MotionStatechart]] (added qp_controller to used_by), [[index.md]] (concept.qp-controller added), [[index-giskardpy.md]] (QP layer section added)
- **findings:**
  - `ConstraintCollection.link_to_motion_statechart_node(node)` multiplies all constraint quadratic_weights by `if_eq(lifecycle_var, RUNNING, 1, 0)`. This means inactive constraints have zero weight in every tick — the QP matrix dimensions never change, avoiding recompilation. This resolves the long-standing open question in concept.motion-statechart.
  - `QPController.xdot_to_control_commands` extracts commands from `xdot[offset:offset+len(active_dofs)]` where `offset = len(active_dofs) * (prediction_horizon - 2)`. Divides by `mpc_dt²`. The "divide by dt²" comes from the jerk→velocity integration across the horizon.
  - `add_rotation_goal_constraints` applies the exact same `-0.0001` rad singularity hack as `sdt.spatial_computations.ik_solver`. Same constant, same rationale — avoiding quaternion flip at zero rotation. The hack appears in two independent codebases, suggesting it was ported or independently discovered.
  - `QPController._set_active_dofs` intersects the DOF variable names with constraint expression free variables — only DOFs that appear in at least one constraint enter the QP. This makes the QP size proportional to task complexity, not world complexity.
- **open questions:**
  - `QPControllerDebugger.are_hard_limits_violated()` is defined but it's not clear if it's called automatically in the tick loop or only in exception handlers. The slack limit check is valuable for diagnosis but silently skipped if never called.

## [2026-05-17] lint | Phase 15 lint pass
- **scope:** Reviewed all pages created or updated in Phases 14-15 (12 pages). Checked symmetry, dangling links, stub coverage, ID hygiene.
- **violations fixed:**
  - `giskardpy.qp.qp_controller.QPController` (non-existent ID) referenced in bridge and concept uses fields — replaced with correct `giskardpy.qp.qp_controller` throughout.
  - `concept.qp-controller.used_by` was empty; populated with `giskardpy.qp.qp_controller`, `giskardpy.qp.constraint_collection.ConstraintCollection`, `bridge.pycram-giskardpy`.
  - `concept.motion-statechart.used_by` was empty; populated with `concept.qp-controller`, `bridge.pycram-giskardpy`. Added `giskardpy.motion_statechart.graph_node.TrinaryCondition` to uses.
  - `giskardpy.motion_statechart.motion_statechart.MotionStatechart.used_by` missing `giskardpy.qp.qp_controller` — added.
  - `giskardpy.motion_statechart.graph_node.NodeArtifacts` was a dangling target in MotionStatechartNode.uses; created stub.
- **remaining stubs (all fresh, 2026-05-17):** krrood stubs (3), pycram.locations.CostmapLocation, pycram.perception.PerceptionQuery, giskardpy.motion_statechart.graph_node.NodeArtifacts.
- **ID hygiene:** PASS — all filenames match id fields.

## [2026-05-17] ingest | Phase 16 — structural gaps (bridge.sdt-giskardpy, QP data layer, context)
- **scope:** `giskardpy/src/giskardpy/motion_statechart/context.py`, `binding_policy.py`, `qp/adapters/qp_adapter.py`, `qp/qp_data.py`, `qp/qp_data_factories.py`, plus grep scan of all giskardpy→SDT imports.
- **created:** [[bridge.sdt-giskardpy]], [[giskardpy.qp.qp_data]], [[giskardpy.qp.qp_data_factories]], [[giskardpy.qp.adapters]], [[giskardpy.motion_statechart.context]]
- **updated:** [[giskardpy.qp.qp_controller]] (adds uses: qp_data, qp_data_factories, adapters, context; used_by: bridge.sdt-giskardpy), [[sdt.world.World]] (used_by: bridge.sdt-giskardpy, context), [[sdt.world_description.degree_of_freedom.DegreeOfFreedom]] (used_by: bridge.sdt-giskardpy, adapters), [[pycram.motion_executor.MotionExecutor]] (uses: context added), [[index-giskardpy.md]] (context + 3 QP data entries added), [[index.md]] (bridge.sdt-giskardpy added)
- **findings:**
  - giskardpy→SDT coupling is pervasive: 10+ files in giskardpy import SDT types. Three primary coupling axes: (1) `DegreeOfFreedom` as QP dimension (qp/adapters), (2) `World` as execution substrate (context), (3) `Derivatives` as the derivative-order type used throughout qp and constraint layers.
  - `QPDataSymbolic.from_giskard()` eliminates acceleration from the optimization variable (the "no_acc" formulation) by expressing acceleration as `vt = vt-1 + 2*vt-2 - jt * dt²`. This reduces QP variable count vs. a full pos/vel/acc/jerk formulation.
  - `ForwardKinematicsBinding.bind()` calls `world.compute_forward_kinematics_np(root, tip)` and flattens the 3×4 rotation+translation submatrix into 12 float variables each tick. Goal constraints reference these float variables symbolically — FK is injected as a parameter, not computed inside the CasADi expression.
  - `QPDataTwoSidedInequalityFactory.evaluate()` stores result in `self.qp_data_raw` (instance attribute) — potential unintentional statefulness in what should be a stateless factory.
- **open questions:**
  - `giskardpy/model/world_config.py` loads SDT World from URDF via `URDFParser` + `AbstractRobot`. How this world is shared with pycram's Context.world (or whether they are the same object) is untraced — potential dual-world risk.
  - `no_acc` derivative link formulation: eliminating acceleration reduces QP variables but it's unclear if this causes numerical issues for high-bandwidth motions. Not documented in source.

## [2026-05-17] ingest | Phase 17 — giskardpy model layer + velocity profile functions
- **scope:** `giskardpy/src/giskardpy/model/world_config.py`, `giskardpy/src/giskardpy/qp/pos_in_vel_limits.py`, `pycram/src/pycram/plans/plan_node.py:395-420` (world flow trace).
- **created:** [[giskardpy.model.world_config]], [[giskardpy.qp.pos_in_vel_limits]]
- **updated:** [[giskardpy.qp.adapters]] (pos_in_vel_limits added to uses), [[sdt.world.World]] (used_by: world_config added), [[sdt.robots.abstract_robot.AbstractRobot]] (used_by: world_config added), [[bridge.sdt-giskardpy]] (WorldConfig open question resolved), [[index-giskardpy.md]] (model layer section + pos_in_vel_limits entry added)
- **findings:**
  - `WorldConfig` is giskardpy's **standalone initialization path** — not used by pycram. When pycram drives giskardpy, world flows via `Context.world → plan.world → MotionExecutor(world=plan.world) → MotionStatechartContext(world=self.world)`. Single `sdt.world.World` object shared throughout — no dual-world risk. Confirmed at `pycram/plans/plan_node.py:406-408`.
  - `WorldFromDatabaseConfig` loads world from krrood/ormatic ORM database via primary key — a fourth initialization path (URDF fixed, URDF omnidrive, URDF diffdrive, ORM database).
  - `acc_cap()` uses inverse Gaussian summation `r_gauss(k) = sqrt(2k + 0.25) - 0.5` to compute maximum reachable acceleration before hitting velocity limit in closed form — avoids iteration. `@substitution_cache` prevents CasADi subgraph duplication.
  - `compute_slowdown_asap_vel_profile` iterates horizon-many applications of `compute_next_vel_and_acc` to produce a symbolic velocity profile. If the projected profile would violate position limits, jerk limits for steps 0–2 are **relaxed** to emergency values — graceful degradation over hard infeasibility.
- **open questions:**
  - `WorldFromDatabaseConfig.setup_collision_config()` is a no-op stub — collision from ORM not implemented.
  - `EmptyWorld` has commented-out `set_default_limits` — unclear if velocity/jerk defaults are set elsewhere or simply absent for empty worlds.

## [2026-05-17] ingest | Phase 18 — krrood EQL stub expansion + pycram.locations
- **scope:** `krrood/src/krrood/entity_query_language/core/variable.py`, `predicate.py`, `backends.py`, `query/query.py`; `pycram/src/pycram/locations/locations.py`.
- **created:** [[krrood.entity_query_language.query.Query]] (stub), [[pycram.locations.locations.AccessingLocation]], [[pycram.locations.locations.GiskardLocation]]
- **updated (stub→full):** [[krrood.entity_query_language.core.variable.Variable]], [[krrood.entity_query_language.predicate.Predicate]], [[krrood.entity_query_language.backends.QueryBackend]], [[pycram.locations.locations.CostmapLocation]]
- **updated (symmetry):** [[sdt.world.World]] (used_by: CostmapLocation, AccessingLocation, GiskardLocation), [[pycram.datastructures.grasp.GraspDescription]] (used_by: CostmapLocation), [[sdt.spatial_computations.ik_solver]] (used_by: CostmapLocation, AccessingLocation), [[pycram.motion_executor.MotionExecutor]] (used_by: GiskardLocation), [[sdt.world_description.connections]] (used_by: AccessingLocation), [[pycram.robot_plans.actions.core.container]] (uses: CostmapLocation, AccessingLocation), [[concept.krrood-eql]] (uses: Query; source_paths: query.py added), [[index-pycram.md]] (Locations + krrood EQL sections added)
- **findings:**
  - `Variable.__new__` and `Predicate.__new__` implement a dual-mode dispatch pattern: calling with `Variable` arguments returns `InstantiatedVariable` (deferred/symbolic), calling with concrete args proceeds normally. This makes ordinary Python classes EQL-composable without any EQL imports.
  - `CostmapLocation` hardcodes two constants: **0.4 m ring radius** and **600 candidate limit** — neither is surfaced in the API. `LocationNotFound` gives no breakdown of why candidates failed.
  - `GiskardLocation` tests actual motion feasibility (full MSC executor tick) rather than point IK. It is substantially slower than `CostmapLocation` and may be experimental (no confirmed action currently uses it).
  - `EntityQueryLanguageBackend._evaluate_underspecified` constructs `T(**grounded_kwargs)` for each Cartesian product combination of Variable domains — the generative grounding loop. Nested `Match` is unverified.
- **open questions:**
  - `GiskardLocation`: which action(s) use it in practice, and whether it runs in `SIMULATED` or a dedicated feasibility mode — unconfirmed.
  - `AccessingLocation`: whether it deepcopies the world for IK testing and the exact `Connection` subtype filter are unverified.
  - `krrood.entity_query_language.query.Query`: full DAG API and `build()` semantics deferred to future ingest.

## [2026-05-17] ingest | Phase 19 — Query expansion + PerceptionQuery + location corrections
- **scope:** `krrood/src/krrood/entity_query_language/query/query.py` (full, 595 lines); `pycram/src/pycram/perception.py` (77 lines); `pycram/src/pycram/locations/locations.py` (full re-read, lines 1-697 for GiskardLocation + AccessingLocation corrections).
- **created:** [[sdt.world_description.geometry.BoundingBox]] (stub)
- **updated (stub→full):** [[krrood.entity_query_language.query.Query]], [[pycram.perception.PerceptionQuery]]
- **updated (corrections):** [[pycram.locations.locations.GiskardLocation]] (major: uses Executor directly not MotionExecutor; only 10 candidates not 600; yields root global pose; catches InfeasibleException), [[pycram.locations.locations.AccessingLocation]] (GaussianCostmap not RingCostmap; yields raw pose not AccessPose; all arm chains tried; uses GraspDescription directly), [[pycram.locations.locations.CostmapLocation]] (ring activated by reachable=True, not reachable_arm)
- **updated (symmetry):** [[pycram.motion_executor.MotionExecutor]] (remove used_by: GiskardLocation — incorrect), [[giskardpy.motion_statechart.context]] (add used_by: GiskardLocation), [[giskardpy.qp.qp_controller]] (add used_by: GiskardLocation — Executor is bundled here), [[sdt.robots.abstract_robot.AbstractRobot]] (add used_by: PerceptionQuery, AccessingLocation, GiskardLocation), [[pycram.datastructures.grasp.GraspDescription]] (add used_by: AccessingLocation, GiskardLocation), [[sdt.spatial_computations.ik_solver]] (remove used_by: AccessingLocation — indirect dep only), [[sdt.world.World]] (add used_by: PerceptionQuery), [[index-pycram.md]] (Perception section added; stubs section cleared), [[index-sdt.md]] (Geometry section + BoundingBox added)
- **findings:**
  - `PerceptionQuery.from_world()` does **not** populate `world.semantic_annotations` — it reads existing annotations. Population is handled upstream by the vision system. The prior open question was based on a wrong assumption.
  - `GiskardLocation` uses `giskardpy.executor.Executor` directly (NOT `pycram.motion_executor.MotionExecutor`) and samples only **10 candidates** (vs CostmapLocation's 600). The MSC is compiled once per candidate.
  - `AccessingLocation` uses `OccupancyCostmap + GaussianCostmap` (not Ring); tries **all arm chains**, not just `self.arm`; yields raw base pose (not AccessPose with connection). The movable connection is re-discovered during OpeningMotion execution, not carried from the location.
  - `Query.build()` wires the DAG lazily: having → grouped_by → where → selected_variables → ordered_by → quantifier. Build-lock prevents modification once wired; `_parent_` setter auto-triggers `build()` as a hot-fix (TODO in source).
- **open questions:**
  - `AccessingLocation` yields raw pose; `OpenAction`/`CloseAction` must re-discover the movable connection during `OpeningMotion`. Whether this is by UUID lookup or re-traversal is unverified.
  - `GiskardLocation.executor.tick_until_end()` — no explicit iteration cap visible; whether Executor has its own limit is unknown.
  - `sdt.world_description.geometry.BoundingBox` stub — `contains()` semantics (inclusive/exclusive bounds, coordinate frame assumptions) unverified.

## [2026-05-17] ingest | Phase 20 — sdt.reasoning.predicates + BoundingBox + pycram.pose_validator
- **scope:** `semantic_digital_twin/src/semantic_digital_twin/reasoning/predicates.py` (615 lines, full); `semantic_digital_twin/src/semantic_digital_twin/world_description/geometry.py` (1242 lines, full); `pycram/src/pycram/pose_validator.py` (213 lines, full).
- **created:** [[sdt.reasoning.predicates]], [[pycram.pose_validator]]
- **updated (stub→full):** [[sdt.world_description.geometry.BoundingBox]]
- **updated (corrections):**
  - [[pycram.locations.locations.CostmapLocation]] — removed `sdt.spatial_computations.ik_solver` from uses (not used); added `pycram.pose_validator`; fixed reachability description (full MSC not standalone IK)
  - [[pycram.locations.locations.AccessingLocation]] — added `pycram.pose_validator` to uses (calls `pose_sequence_reachability_validator`)
  - [[sdt.spatial_computations.ik_solver]] — removed `pycram.locations.locations.CostmapLocation` from used_by
  - [[sdt.reasoning.WorldReasoner]] — added `sdt.reasoning.predicates` to uses; updated EQL predicates section to link to new page; updated provenance note
- **updated (symmetry):**
  - [[sdt.world.World]] (add used_by: pycram.pose_validator)
  - [[sdt.robots.abstract_robot.AbstractRobot]] (add used_by: pycram.pose_validator)
  - [[giskardpy.qp.qp_controller]] (add used_by: pycram.pose_validator)
  - [[giskardpy.motion_statechart.context]] (add used_by: pycram.pose_validator)
  - [[pycram.perception.PerceptionQuery]] (add uses: sdt.reasoning.predicates)
  - [[sdt.collision_checking]] (add used_by: sdt.reasoning.predicates)
  - [[sdt.spatial_computations.raytracer]] (add used_by: sdt.reasoning.predicates)
  - [[index-sdt.md]] (add predicates entry; promote BoundingBox from stub)
  - [[index-pycram.md]] (add pose_validator entry; fix location descriptions; fix Query stub→full)
- **findings:**
  - `predicates.visible()` ignores the camera's orientation entirely — `get_visible_bodies()` sets cam_pose rotation to `np.eye(3)` (identity), using only the camera's world position. The segmentation mask therefore covers all directions from that point at 256-pixel resolution. This differs from `pycram.pose_validator.visibility_validator()` which sends a single point-to-point ray and checks for occlusion between camera and target. **Two different visibility semantics coexist.**
  - `predicates.stable()` raises `NotImplementedError("Needs multiverse")` unconditionally — it is dead code. Any plan relying on physics stability will fail at runtime.
  - `pose_sequence_reachability_validator` modifies the live world state during MSC execution but saves/restores `world.state._data` in a `finally` block. Structural world mutations during MSC would not be rolled back (but standard MSC evaluation does not cause structural changes).
  - `BoundingBox.contains()` transforms the query point into the origin's reference frame before checking — it is frame-aware, not just a min/max comparison. `origin` is excluded from hash/equality, so same-extent boxes in different frames are hash-equal.
- **open questions:**
  - `visible()` ignores orientation — whether intentional ("is the object detectable if the robot rotates?") or a bug is undocumented.
  - `pose_sequence_reachability_validator` hardcodes `target_frequency=50, prediction_horizon=4` (shorter horizon than the standard 7). No API to override it.
  - `BoundingBox` exact line numbers within `geometry.py` (1242 lines) not pinpointed; `source_paths` covers the full file.

## [2026-05-18] ingest | Phase 21 — container actions deep-dive + tool-based audit + robot predicates
- **scope:** `pycram/src/pycram/robot_plans/actions/core/container.py` (171 lines, full re-read); `pycram/src/pycram/robot_plans/actions/composite/tool_based.py` (254 lines, full); `pycram/src/pycram/robot_plans/actions/composite/transporting.py` (lines 265–406, EfficientTransportAction + MoveAndPickUpAction); `pycram/src/pycram/querying/predicates.py` (61 lines); `semantic_digital_twin/src/semantic_digital_twin/reasoning/robot_predicates.py` (205 lines).
- **created:** [[pycram.querying.predicates]], [[sdt.reasoning.robot_predicates]]
- **updated:** [[pycram.robot_plans.actions.core.container]] (corrected uses; expanded pre/post condition; design notes added), [[pycram.robot_plans.actions.composite]] (EfficientTransportAction location corrected; tool_based audit added; provenance fixed), [[pycram.locations.locations.CostmapLocation]] (remove incorrect container from used_by), [[pycram.locations.locations.AccessingLocation]] (remove incorrect container from used_by), [[pycram.pose_validator]] (add container to used_by)
- **updated (symmetry):** [[krrood.entity_query_language.predicate.Predicate]] (add pycram.querying.predicates), [[sdt.robots.abstract_robot.AbstractRobot]] (add robot_predicates, pycram.querying.predicates), [[sdt.world_description.world_entity.Body]] (add robot_predicates, pycram.querying.predicates), [[sdt.world_description.geometry.BoundingBox]] (add robot_predicates), [[sdt.collision_checking]] (add robot_predicates), [[sdt.reasoning.predicates]] (add robot_predicates), [[sdt.spatial_computations.raytracer]] (add robot_predicates), [[index-pycram.md]] (Gripper predicates section), [[index-sdt.md]] (robot_predicates entry)
- **findings:**
  - `EfficientTransportAction` is in `transporting.py:325-406`, NOT in `tool_based.py` — Phase 18 provenance was wrong. It uses `BelieveObject(names=[...]).resolve()` and `RobotDescription.current_robot_description` — non-functional at this commit.
  - All three `tool_based.py` actions (`MixingAction`, `PouringAction`, `CuttingAction`) are non-functional: they use `LocalTransformer()`, `World.current_world`, and `RobotDescription.current_robot_description` (all pycram-bullet). `CuttingAction.execute()` is dead code — it computes slice poses but never performs any motion.
  - `OpenAction.pre_condition` creates a full `deepcopy(context.world)` AND `context.robot.from_world(test_world)` — a complete world+robot deepcopy, not just the state save/restore used by `pose_sequence_reachability_validator`. Both are valid approaches to isolation but with different costs.
  - **Two gripper-occupancy implementations coexist:** `GripperIsFree` (kinematic attachment: is anything connected under TCP?) vs. `is_body_in_gripper` (ray-sampling: is anything physically between the fingers?). `OpenAction.pre_condition` uses `GripperIsFree`; `post_condition` uses `is_body_in_gripper` — checks different things.
  - `container.py` never imports `CostmapLocation` or `AccessingLocation` — those were incorrectly added to container.uses in Phase 18. Corrected.
  - `blocking()` in `sdt.reasoning.robot_predicates` permanently mutates `world.state` via IK writes without restoring — callers must handle cleanup (unlike `pose_sequence_reachability_validator`).
- **open questions:**
  - `EfficientTransportAction` and all `tool_based.py` actions: are these in-migration (planned to be ported to SDT) or dead code? No branch/issue found.
  - `CloseAction` has no `pre_condition` — it will attempt to grasp even if the gripper is already occupied. Whether this is intentional (e.g., robot is already holding the handle from a prior step) is unverified.
  - `blocking()` permanently mutates world state — which callers (if any) invoke it and whether they handle state restoration is unknown.

## [2026-05-18] lint | Phase 22 — full symmetry pass across all ~100 wiki pages
- **scope:** Full `uses` ↔ `used_by` symmetry audit across all entity pages in `wiki/entities/`. ~100 pages checked. 79+ violations found and fixed.
- **created:** (none)
- **updated (symmetry — used_by additions):**
  - [[sdt.world_description.world_entity.Body]] — +11: giskardpy.model.world_config, pycram.datastructures.grasp.GraspDescription, sdt.collision_checking, sdt.semantic_annotations.SemanticAnnotations, sdt.spatial_computations.raytracer, motions.gripper, motions.container, actions.core.container, PickUpAction, PlaceAction, composite
  - [[sdt.world.World]] — +6: pycram.motion_executor.MotionExecutor, AbstractRobot, robots.concrete, sdt.collision_checking, sdt.reasoning.WorldReasoner, sdt.spatial_computations.raytracer
  - [[sdt.robots.abstract_robot.AbstractRobot]] — +6: AlternativeMotion, GraspDescription, ViewManager, actions.core.robot_body, motions.robot_body, sdt.robots.concrete
  - [[sdt.spatial_types.Pose]] — +9: ExecutionData, GraspDescription, NavigateAction, PlaceAction, motions.gripper, motions.navigation, motions.robot_body, composite, giskardpy.qp.adapters
  - [[pycram.robot_plans.BaseMotion]] — +5: AlternativeMotion, motions.container, motions.misc, motions.navigation, motions.robot_body
  - [[pycram.plans.PlanNode]] — +3: MotionExecutor, Designator, PlanCallback
  - [[pycram.plans.PlanEntity]] — +2: Context, PlanCallback
  - [[pycram.robot_plans.ActionDescription]] — +4: composite, core.container, core.misc, core.robot_body
  - [[giskardpy.motion_statechart.graph_node.MotionStatechartNode]] — +4: goals, monitors, tasks, ConstraintCollection
  - [[giskardpy.motion_statechart.graph_node.Task]] — +2: MotionExecutor, giskardpy.motion_statechart.tasks
  - [[sdt.world_description.world_entity.KinematicStructureEntity]] — +6: giskardpy.model.world_config, WorldReasoner, SemanticAnnotations, forward_kinematics, ik_solver, Pose
  - [[pycram.plans.factories]] — +3: NavigateAction, PickUpAction, PlaceAction
  - [[sdt.world_description.connections]] — +4: giskardpy.model.world_config, JointState, SemanticAnnotations, core.container
  - [[pycram.alternative_motion_mapping.AlternativeMotion]] — +1: pycram.pose_validator
  - [[giskardpy.qp.constraint_collection.ConstraintCollection]] — +1: giskardpy.qp.adapters
  - [[sdt.collision_checking]] — +1: giskardpy.motion_statechart.context
  - [[sdt.world_description.degree_of_freedom.DegreeOfFreedom]] — +2: ik_solver, WorldState
  - [[sdt.world_description.world_entity.Connection]] — +1: sdt.spatial_computations.forward_kinematics
  - [[pycram.view_manager.ViewManager]] — +4: motions.robot_body, actions.core.robot_body, motions.container, core.container
  - [[pycram.robot_plans.actions.core.PickUpAction]] — +2: composite, core.container
  - [[pycram.robot_plans.actions.core.PlaceAction]] — +1: composite
  - [[pycram.robot_plans.actions.core.NavigateAction]] — +1: composite
  - [[pycram.robot_plans.motions.gripper]] — +2: core.container, core.robot_body
  - [[pycram.datastructures.grasp.GraspDescription]] — +2: composite, core.container
  - [[pycram.datastructures.Context]] — +2: core.container, core.robot_body
  - [[pycram.plans.Plan]] — +1: pycram.plans.factories
  - [[sdt.world_description.world_entity.Region]] — +1: SemanticAnnotations
  - [[sdt.world_description.shape_collection.ShapeCollection]] — +1: SemanticAnnotations
- **findings:**
  - The largest violation clusters were around high-fan-in infrastructure pages: `sdt.world.World` (6 missing), `sdt.world_description.world_entity.Body` (11 missing), `sdt.spatial_types.Pose` (9 missing), and `sdt.world_description.world_entity.KinematicStructureEntity` (6 missing). These are referenced by many action/motion pages but their `used_by` lists were never updated after those pages were ingested.
  - `pycram.robot_plans.actions.core.PlaceAction` and `pycram.robot_plans.actions.core.NavigateAction` had `used_by: []` despite being used by `composite`.
  - `pycram.alternative_motion_mapping.AlternativeMotion` had `used_by: []` despite being referenced by `pycram.pose_validator`.
- **open questions:**
  - `sdt.datastructures.joint_state.JointState.used_by` lists `AbstractRobot` but `AbstractRobot.uses` does not list `JointState`. This is an inverse-direction violation (used_by claims a consumer that doesn't declare the dependency). Deferred — would require either adding JointState to AbstractRobot.uses or removing AbstractRobot from JointState.used_by after source verification.
  - A follow-up lint pass is recommended after any new Phase ingests to catch incremental drift.

## [2026-05-18] lint | Phase 23 — symmetry audit continuation (deferred pages)
- **scope:** Remaining ~30 entity pages not covered in Phase 22 symmetry pass. Pages checked: `pycram.plans.plan_callbacks.PlanCallback`, `pycram.fluent.Fluent`, `sdt.world_description.connections`, `giskardpy.model.world_config`, `sdt.datastructures.joint_state.JointState`, `sdt.semantic_annotations.SemanticAnnotations`, `pycram.robot_plans.actions.core.container`, `giskardpy.qp.adapters`, `giskardpy.motion_statechart.monitors`, `giskardpy.qp.constraint_collection.ConstraintCollection`, `giskardpy.qp.qp_data`, `giskardpy.qp.qp_data_factories`, `giskardpy.motion_statechart.graph_node.TrinaryCondition`, `giskardpy.qp.qp_controller`, `giskardpy.motion_statechart.graph_node.MotionStatechartNode`, `giskardpy.motion_statechart.graph_node.NodeArtifacts`, `giskardpy.motion_statechart.context`, `sdt.world_description.geometry.BoundingBox`, `sdt.world_description.degree_of_freedom.DegreeOfFreedom`, `sdt.world_description.world_entity.Region`, `sdt.world_description.world_entity.Connection`, `sdt.world_description.shape_collection.ShapeCollection`, `sdt.spatial_computations.ik_solver`, `sdt.world.World`, `sdt.spatial_computations.forward_kinematics`, `sdt.world_description.world_state.WorldState`, `sdt.spatial_types.Pose`, `giskardpy.motion_statechart.goals`, `pycram.pose_validator`, `pycram.perception.PerceptionQuery`, `sdt.reasoning.robot_predicates`, `pycram.alternative_motion_mapping.AlternativeMotion`, `pycram.robot_plans.motions.misc`, `giskardpy.motion_statechart.tasks`, `giskardpy.qp.pos_in_vel_limits`, `giskardpy.qp.constraint`, `pycram.plans.ActionNode`, `pycram.plans.failures.PlanFailure`, `sdt.reasoning.predicates`.
- **created:** (none)
- **updated (frontmatter symmetry fixes):**
  - [[sdt.world.World]] — `uses` +1: `sdt.spatial_computations.ik_solver` (World has `compute_inverse_kinematics` method that delegates to IK solver; ik_solver.used_by claimed World)
  - [[pycram.robot_plans.actions.core.container]] — `uses` +1: `sdt.world_description.connections` (post_condition calls `get_first_parent_connection_of_type(ActiveConnection1DOF)` — uses type from connections.py; connections.used_by already claimed container)
  - [[giskardpy.motion_statechart.monitors]] — `used_by` -1: removed `giskardpy.motion_statechart.goals` (goals.uses does not list monitors; goals compose Tasks via `expand()`, not monitors; monitors are separate MSC node types added alongside goals, not inside them)
- **updated (body/frontmatter text sync — no frontmatter changes):**
  - [[pycram.robot_plans.motions.misc]] — body "Used by" corrected to `(none confirmed)` to match `used_by: []` frontmatter
  - [[pycram.alternative_motion_mapping.AlternativeMotion]] — body "Used by" section added for `pycram.pose_validator`
  - [[sdt.world_description.connections]] — body "Used by" expanded to match all 7 frontmatter `used_by` entries
  - [[sdt.world_description.degree_of_freedom.DegreeOfFreedom]] — body "Used by" expanded to match all 6 frontmatter `used_by` entries
  - [[sdt.world_description.geometry.BoundingBox]] — body "Used by" added `sdt.reasoning.robot_predicates`
  - [[sdt.world_description.world_entity.Region]] — body "Used by" added `sdt.semantic_annotations.SemanticAnnotations`
  - [[sdt.world_description.world_entity.Connection]] — body "Used by" added `sdt.spatial_computations.forward_kinematics`
  - [[sdt.world_description.shape_collection.ShapeCollection]] — body "Used by" corrected: removed `pycram.datastructures.grasp.GraspDescription` and `pycram.robot_plans.actions.core.misc` (not in frontmatter), added `sdt.semantic_annotations.SemanticAnnotations`
  - [[sdt.reasoning.predicates]] — body "Used by" added `sdt.reasoning.robot_predicates`
  - [[pycram.pose_validator]] — body "Used by" added `pycram.robot_plans.actions.core.container`
  - [[giskardpy.qp.qp_controller]] — body "Used by" expanded to match all 5 frontmatter `used_by` entries
  - [[sdt.world.World]] — body Related section added `sdt.spatial_computations.ik_solver`
- **findings:**
  - `pycram.robot_plans.actions.core.container` post_condition directly references `ActiveConnection1DOF` from `sdt.world_description.connections`, confirming the missing `uses` entry.
  - `giskardpy.motion_statechart.monitors.used_by` had a spurious `goals` entry: goals compose Tasks via `expand()`, and monitors are separate MSC participants added alongside goals in the MSC topology — not by goals.py itself.
  - `sdt.world.World` was missing `sdt.spatial_computations.ik_solver` from `uses` despite having a `compute_inverse_kinematics` method and ik_solver.used_by claiming World.
  - Body/frontmatter drift was widespread across SDT data-structure pages (DegreeOfFreedom, Connection, connections, ShapeCollection) where `used_by` frontmatter had grown but body "Used by" sections were not updated.
- **open questions:**
  - `pycram.plans.failures.PlanFailure.used_by` claims `pycram.plans.ActionNode` but `ActionNode.uses` does not include `PlanFailure`. Source verification needed to determine if ActionNode.py imports PlanFailure directly or only raises it via re-throw from PlanNode.
  - `giskardpy.motion_statechart.monitors` may be used by motion designator modules (motions.gripper, motions.navigation) that add monitors as EndMotion conditions — but their `uses` lists don't include monitors.monitors. Deferred.
  - `pycram.datastructures.grasp.GraspDescription` calls `body.visual.as_bounding_box_collection_in_frame(...)` — accessing a `ShapeCollection` method. Whether GraspDescription.py imports ShapeCollection directly (warranting a `uses` entry) is unverified from wiki evidence alone.

## [2026-05-18] lint | Phase 24 — concept/bridge symmetry pass
- **scope:** frontmatter symmetry audit across all concept and bridge pages; 34 violations resolved by adding missing reciprocal entries (UNION strategy — never removing existing claims).
- **created:** none
- **updated:**
  - [[concept.forward-kinematics]] — `used_by` += `sdt.spatial_computations.forward_kinematics`
  - [[concept.motion-statechart]] — `uses` += `concept.forward-kinematics`, `giskardpy.motion_statechart.monitors`
  - [[concept.world]] — `uses` += `concept.forward-kinematics`, `concept.semantic-annotation`, `sdt.spatial_types.Pose`, `sdt.world_description.connections`
  - [[bridge.pycram-giskardpy]] — `uses` += `concept.motion-statechart`
  - [[bridge.pycram-sdt]] — `uses` += `concept.semantic-annotation`, `sdt.world_description.connections`
  - [[sdt.world_description.world_entity.Connection]] — `used_by` += `concept.forward-kinematics`
  - [[sdt.world_description.degree_of_freedom.DegreeOfFreedom]] — `used_by` += `concept.forward-kinematics`
  - [[sdt.spatial_computations.forward_kinematics]] — `used_by` += `concept.forward-kinematics`
  - [[pycram.robot_plans.ActionDescription]] — `used_by` += `concept.krrood-eql`
  - [[pycram.plans.UnderspecifiedNode]] — `used_by` += `concept.krrood-eql`
  - [[pycram.datastructures.Context]] — `used_by` += `bridge.pycram-sdt`, `concept.krrood-eql`, `pycram.plans.factories`
  - [[giskardpy.qp.qp_controller]] — `used_by` += `concept.qp-controller`
  - [[sdt.world_description.world_entity.KinematicStructureEntity]] — `used_by` += `bridge.sdt-giskardpy`, `concept.semantic-annotation`
  - [[pycram.robot_plans.BaseMotion]] — `used_by` += `bridge.pycram-giskardpy`
  - [[pycram.motion_executor.MotionExecutor]] — `used_by` += `bridge.pycram-giskardpy`
  - [[sdt.datastructures.GripperState]] — `used_by` += `bridge.pycram-sdt`
  - [[sdt.spatial_types.Pose]] — `used_by` += `bridge.sdt-giskardpy`
  - [[sdt.collision_checking]] — `used_by` += `bridge.sdt-giskardpy`
  - [[giskardpy.motion_statechart.motion_statechart.MotionStatechart]] — `used_by` += `bridge.sdt-giskardpy`
  - [[giskardpy.qp.constraint_collection.ConstraintCollection]] — `uses` += `concept.qp-controller`
  - [[pycram.robot_plans.motions.gripper]] — `uses` += `giskardpy.motion_statechart.graph_node.Goal`
  - [[pycram.robot_plans.motions.container]] — `uses` += `giskardpy.motion_statechart.graph_node.Goal`
  - [[giskardpy.motion_statechart.graph_node.Task]] — `uses` += `giskardpy.qp.constraint_collection.ConstraintCollection`
  - [[pycram.plans.ActionNode]] — `uses` += `pycram.plans.failures.PlanFailure`
  - [[pycram.plans.PlanEntity]] — `uses` += `pycram.plans.Plan`
  - [[pycram.robot_plans.actions.core.NavigateAction]] — `uses` += `pycram.robot_plans.motions.gripper`
- **findings:**
  - Concept/bridge pages had asymmetric growth: their `uses` lists referenced specific entities (Connection, DegreeOfFreedom, Context, Pose, etc.) but the target entity pages did not reciprocate in `used_by`. Conversely, many entity pages listed concepts/bridges in `used_by` but the concept/bridge had no corresponding `uses` entry — likely a relic of the entity pages being ingested before the concept/bridge frontmatter caught up.
  - The cross-concept link `concept.forward-kinematics ↔ concept.motion-statechart` was asymmetric. Concept-to-concept symmetry is a useful first-class citizen for navigation queries.
  - `bridge.pycram-sdt.uses` lacked `sdt.world_description.connections` despite the body inventory referencing `Connection6DoF.create_with_dofs` — frontmatter drift from PickUpAction/PlaceAction body content.
  - The `concept.qp-controller ↔ ConstraintCollection` symmetry was missing in both directions, even though the concept page extensively documents the lifecycle gating mechanism implemented by ConstraintCollection.
- **open questions:**
  - Package-level pycram.md `used_by` lists 5 entities (ActionDescription, BaseMotion, DesignatorNode, factories, factories.make_node) that don't reciprocate. Skipped per phase scope (pycram.md is a stub). Worth revisiting when pycram package page is promoted from stub.
  - Entity-to-entity asymmetries (outside concept/bridge scope) were not audited in this pass and likely still exist.

## [2026-05-18] lint | Phase 25 — dangling link fixes

- **scope:** All `[[id]]` references across ~99 entity/concept/bridge pages; identified and resolved 3 bad IDs + 2 missing stub pages
- **created:** [[giskardpy.motion_statechart.graph_node.EndMotion]], [[giskardpy.motion_statechart.graph_node.CancelMotion]]
- **updated:**
  - [[concept.qp-controller]] — body "Related" `[[giskardpy.qp.qp_controller.QPController]]` → `[[giskardpy.qp.qp_controller]]`
  - [[concept.motion-statechart]] — body "Resolved notes" same bad ID fixed
  - [[giskardpy.qp.constraint_collection.ConstraintCollection]] — body "Used by" same bad ID fixed
  - [[giskardpy.motion_statechart.motion_statechart.MotionStatechart]] — `uses` += `EndMotion` (is_end_motion() checks EndMotion node type)
  - [[pycram.motion_executor.MotionExecutor]] — `uses` += `EndMotion` (adds EndMotion.when_true sentinel)
  - [[pycram.locations.locations.GiskardLocation]] — `uses` += `EndMotion` (adds EndMotion.when_true)
  - [[giskardpy.motion_statechart.goals]] — `uses` += `CancelMotion` (collision avoidance triggers CancelMotion)
  - [[index-giskardpy.md]] — added EndMotion and CancelMotion stub entries
- **findings:**
  - `[[giskardpy.qp.qp_controller.QPController]]` was a non-existent sub-class ID; the page is a bundled page with ID `giskardpy.qp.qp_controller` (no class name suffix). Appeared in 3 pages.
  - `EndMotion` and `CancelMotion` are documented in the bundled Goal page but were referenced as standalone IDs by `MotionStatechartNode.used_by`; created minimal stubs with correct inheritance (`uses: MotionStatechartNode`) and pointing to the Goal bundled page.
  - `[[List[Body]]` in `pycram.querying.predicates.md` was NOT a dangling link — it's a Python type annotation `Callable[[List[Body]], bool]` inside a code span; no fix needed.
- **open questions:** (none new)

## [2026-05-18] ingest | Phase 26 — new pages: costmaps, spatial_types, adapters, pipeline

- **scope:** `pycram/src/pycram/locations/costmaps.py` (861 lines, full); `semantic_digital_twin/src/semantic_digital_twin/spatial_types/spatial_types.py` (2078 lines, full); `semantic_digital_twin/src/semantic_digital_twin/adapters/urdf.py` (366 lines, full) + adapters subpackage structure (glob); `semantic_digital_twin/src/semantic_digital_twin/pipeline/pipeline.py` (174 lines, full) + `mesh_decomposition/base.py` (81 lines, full). Read at commit `0528d8cf3`.
- **created:** [[pycram.locations.costmaps]], [[sdt.spatial_types.spatial_types]], [[sdt.adapters]], [[sdt.pipeline]]
- **updated:**
  - [[pycram.locations.locations.CostmapLocation]] — `uses` += `pycram.locations.costmaps`; body Related updated
  - [[pycram.locations.locations.AccessingLocation]] — `uses` += `pycram.locations.costmaps`; body Related updated
  - [[sdt.world.World]] — `used_by` += `pycram.locations.costmaps`, `sdt.adapters`, `sdt.pipeline`; body Related updated
  - [[sdt.spatial_computations.raytracer]] — `used_by` += `pycram.locations.costmaps`; body Related updated
  - [[sdt.world_description.world_entity.Body]] — `used_by` += `pycram.locations.costmaps`, `sdt.adapters`, `sdt.pipeline`
  - [[sdt.world_description.connections]] — `used_by` += `sdt.adapters`; body Related updated
  - [[sdt.world_description.degree_of_freedom.DegreeOfFreedom]] — `used_by` += `sdt.adapters`; body Related updated
  - [[sdt.world_description.shape_collection.ShapeCollection]] — `used_by` += `sdt.adapters`, `sdt.pipeline`; body Related updated
  - [[sdt.spatial_types.Pose]] — `used_by` += `sdt.spatial_types.spatial_types`
  - [[sdt.world_description.world_entity.KinematicStructureEntity]] — `used_by` += `sdt.spatial_types.spatial_types`
  - [[sdt.spatial_computations.forward_kinematics]] — `used_by` += `sdt.spatial_types.spatial_types`
  - [[sdt.spatial_computations.ik_solver]] — `used_by` += `sdt.spatial_types.spatial_types`
  - [[giskardpy.model.world_config]] — `uses` += `sdt.adapters`
  - [[sdt.robots.concrete]] — `uses` += `sdt.adapters`
  - [[index-pycram.md]] — added `pycram.locations.costmaps` under Locations
  - [[index-sdt.md]] — added `sdt.spatial_types.spatial_types` under Spatial types; added Adapters and Pipeline sections
- **findings:**
  - `VisibilityCostmap` contains an acknowledged bug (source comment): the quaternion used to rotate between the four 90° depth-image captures is never normalised. The tests pass but the visibility scores for some sectors may be computed against an incorrectly oriented image. The spatial type used is `Quaternion(0, 0, 1, 1)` — invalid but empirically functional in current test suite.
  - `OccupancyCostmap.create_ray_mask_around_origin` uses `self.width` for both dimensions regardless of the `height` attribute — the occupancy grid is always square. The `height` field from `Costmap` base class is effectively unused by `OccupancyCostmap`.
  - `sdt.spatial_types.spatial_types` is one of the largest modules in the codebase (2078 lines). Every spatial type — HTM, RotationMatrix, Point3, Vector3, Quaternion, Pose — lives in a single file. The `@sm.substitution_cache` decorator on `rotation_matrix_to_quaternion` is a key performance guard preventing CasADi subgraph explosion during FK expression construction.
  - `sdt.adapters/` is substantially broader than just URDF loading: MJCF (MuJoCo), USD, FBX, PartNet-Mobility, SAGE-10K, ProcTHOR datasets, and a full ROS 2 adapter layer with TF publishing and world synchronisation.
  - `sdt.pipeline.BodyFactoryReplace` has a hardcoded default `body_condition` matching `dresser_\d+` — dataset-specific logic baked into the generic pipeline mechanism.
  - `MeshDecomposer.apply_to_body` replaces `body.collision` with decomposed visual mesh shapes — comment says "visual" but the target is collision; likely intentional (pre-processing visual for physics) but worth verifying.
- **open questions:**
  - `VisibilityCostmap` quaternion bug: `Quaternion(0, 0, 1, 1)` is passed to `HomogeneousTransformationMatrix.from_point_rotation_matrix(rotation_matrix=q.to_rotation_matrix())` — does `to_rotation_matrix()` silently normalise, masking the bug? Or do the tests just not cover the problematic sectors?
  - `mjcf.py`, `usd.py`, `fbx.py` were not read in full — MJCF especially (MuJoCo actuators, tendons) may introduce new `Connection` subclasses not found in URDF. Phase 27 candidate.
  - `sdt.pipeline.gltf_loader.py` was not read — GLTF/GLB loading path unknown.

## [2026-05-19] ingest | Phase 27 — promote stubs (Manipulator, GraspDescription) + document RobotDescription legacy

- **scope:** Suggestion #1 from design-conversation review: promote the three "stub/missing" pages blocking the LLM-grounding agent pipeline. Read `semantic_digital_twin/src/.../robots/abstract_robot.py` (Manipulator hierarchy), `pycram/src/.../datastructures/grasp.py` (full GraspDescription source), `pycram/src/.../datastructures/rotations.py` (quaternion tables), `pycram/src/.../datastructures/enums.py` (ApproachDirection/VerticalAlignment/Arms/AxisIdentifier).
- **created:** [[sdt.robots.abstract_robot.Manipulator]] — new bundled page covering `Manipulator` (abstract), `ParallelGripper`, `HumanoidGripper`, `Finger`. Documents `tool_frame`, `front_facing_orientation`, `front_facing_axis`, per-robot wiring pattern, and the GraspDescription consumption chain.
- **updated:**
  - [[pycram.datastructures.grasp.GraspDescription]] — expanded from concise overview to full reference: `Rotations` quaternion tables with concrete values, `ApproachDirection`/`VerticalAlignment`/`AxisIdentifier`/`Arms` enum encodings, `calculate_closest_faces` algorithm, `edge_offset`/`grasp_pose`/`manipulation_axis`/`lift_axis`/`calculate_manipulator_axis` methods, `PreferredGraspAlignment` and `GraspPose` helper types. `uses` now lists `Manipulator` (was implicit via AbstractRobot only).
  - [[sdt.robots.abstract_robot.AbstractRobot]] — added: link to new Manipulator page; "Key collections (for grounding)" table enumerating `manipulators` / `manipulator_chains` / `controlled_connections` / `degrees_of_freedom_with_hardware_interface`; explicit "Relation to the legacy `RobotDescription`" section flagging it as deprecated in favour of `AbstractRobot`. `used_by` += Manipulator.
  - [[sdt.robots.concrete]] — `uses` += Manipulator (concrete robots wire `ParallelGripper` etc.).
  - [[index-sdt.md]] — added Manipulator entry under "Robots" section; AbstractRobot description now notes it replaces the legacy `RobotDescription`.
- **findings:**
  - No `RobotDescription` class exists in SDT — it's a pycram-only legacy artifact (`pycram.robot_description.RobotDescription.current_robot_description`) referenced ONLY from already-flagged-legacy actions: `robot_body.py`, `tool_based.py` (MixingAction/PouringAction/CuttingAction), `transporting.py` (EfficientTransportAction). The modern replacement is `AbstractRobot` (via `Context.robot`). Decision: no `RobotDescription` page; document the relationship on `AbstractRobot` and let legacy action pages cite their own legacy patterns.
  - `Manipulator.front_facing_orientation` is the fourth and final term in `GraspDescription.grasp_orientation()`'s quaternion product — making it the single robot-specific piece of grasp geometry. Its sign convention is not codified anywhere; two robots can disagree on which direction is "front" and ports of grasp logic between them silently invert.
  - The `manipulation_offset` field of `GraspDescription` controls BOTH the pre-grasp standoff AND the lift height. There is no API to set them independently.
  - `place_pose_sequence` reads `manipulator.tool_frame.child_kinematic_structure_entities[0]` — implicit assumption that exactly one body is held; multi-body grasps (tray with items, two cubes) fail silently here.
  - `GraspDescription.__hash__ = id(self)` — two instances with identical fields are non-equal. Affects cache strategies for grasp enumeration.
- **open questions:** (added to relevant pages — Manipulator + GraspDescription + AbstractRobot — see "Open questions" sections)

## [2026-05-19] ingest | Phase 28 — enum foundation pages (schema enrichment Phase 1)

- **scope:** `pycram/src/pycram/datastructures/enums.py:63-167` (commit `0528d8cf3`). Phase 1 of 5 in schema enrichment plan: create dedicated enum entity pages with `values:` blocks, enabling agents to enumerate valid field values from frontmatter alone.
- **created:** [[pycram.datastructures.enums.Arms]], [[pycram.datastructures.enums.ApproachDirection]], [[pycram.datastructures.enums.VerticalAlignment]], [[pycram.datastructures.enums.AxisIdentifier]]
- **updated:**
  - [[pycram.datastructures.grasp.GraspDescription]] — `uses` += all four new enum IDs
  - [[pycram.robot_plans.actions.core.PickUpAction]] — `uses` += `Arms`
  - [[pycram.robot_plans.actions.core.PlaceAction]] — `uses` += `Arms`
  - [[pycram.robot_plans.motions.gripper]] — `uses` += `Arms`
  - [[index-pycram.md]] — added "Enums" section listing all four pages
- **findings:**
  - All four enums live in the same 105-line span of `enums.py`. `ApproachDirection` and `VerticalAlignment` encode their values as `(AxisIdentifier, ±1)` tuples, making `AxisIdentifier` a foundational type that all three grasp-enumeration enums depend on.
  - `Arms` is an `IntEnum` (LEFT=0, RIGHT=1, BOTH=2) — integer coercibility is relevant for array indexing in robot_plans code.
  - `ApproachDirection.SIDE_ROTATIONS` and `VerticalAlignment.VERTICAL_ROTATIONS` quaternion lookup tables (in `rotations.py`) are now cross-linked on the enum pages, completing the traceability from enum value → quaternion → `grasp_orientation()`.
- **open questions:** (none new — Phase 28)

## [2026-05-19] ingest | Phase 29 — fields: blocks (schema enrichment Phase 2)

- **scope:** Schema enrichment Phase 2: add `fields:` frontmatter blocks to `PickUpAction` and `GraspDescription` — the vertical slice covering the most common manipulation instruction end-to-end.
- **created:** (none)
- **updated:**
  - [[pycram.robot_plans.actions.core.PickUpAction]] — `fields:` block added: `object_designator` (Body), `arm` (Arms, domain=[LEFT,RIGHT]), `grasp_description` (GraspDescription). `last_ingest` → 2026-05-19.
  - [[pycram.datastructures.grasp.GraspDescription]] — `fields:` block added: `approach_direction` (ApproachDirection), `vertical_alignment` (VerticalAlignment), `manipulator` (Manipulator), `rotate_gripper` (bool, default false), `manipulation_offset` (float, default 0.05).
- **findings:**
  - `arm` domain is `[LEFT, RIGHT]` — `BOTH` is structurally valid (Arms enum member) but semantically inapplicable to a single-arm pick-up; the `domain:` restriction captures this constraint without changing the type.
  - `manipulation_offset` controls two geometrically distinct things (standoff and lift height) via a single scalar. The `description:` explicitly calls this out so an agent generating a GraspDescription knows it cannot tune them independently.
  - `rotate_gripper` has `default: false` because `GraspDescription.__init__` declares `rotate_gripper: bool = False`; an agent can omit it unless the object geometry requires the 90° roll.
  - `GraspDescription.fields` is itself a nested composite — `PickUpAction.fields.grasp_description` points to a type that also has `fields:`. An agent building a PickUpAction instance should recurse into GraspDescription's `fields:` to fill in that sub-object.
- **open questions:** (none new — Phase 29)

## [2026-05-19] ingest | Phase 30 — fields: blocks (schema enrichment Phase 3)

- **scope:** Schema enrichment Phase 3: add `fields:` frontmatter to NavigateAction, PlaceAction, and the container page (OpenAction + CloseAction). Also caught and fixed missing `Arms` entry in container.uses.
- **created:** (none)
- **updated:**
  - [[pycram.robot_plans.actions.core.NavigateAction]] — `fields:` added: `target_location` (Pose), `keep_joint_states` (bool, no default — read from ActionConfig at runtime). `last_ingest` → 2026-05-19.
  - [[pycram.robot_plans.actions.core.PlaceAction]] — `fields:` added: `object_designator` (Body), `target_location` (Pose), `arm` (Arms, domain=[LEFT,RIGHT]), `grasp_description` (GraspDescription, `derived_from: plan_history.PickUpAction.grasp_description`). `last_ingest` → 2026-05-19.
  - [[pycram.robot_plans.actions.core.container]] — `fields:` added: `object_designator` (Body), `arm` (Arms, domain=[LEFT,RIGHT]), `grasping_prepose_distance` (float, `derived_from: ActionConfig.grasping_prepose_distance`). `uses` += `pycram.datastructures.enums.Arms` (was missing despite body tables using Arms). `last_ingest` → 2026-05-19.
  - [[pycram.datastructures.enums.Arms]] — `used_by` += `pycram.robot_plans.actions.core.container` (symmetry fix).
- **findings:**
  - `PlaceAction.grasp_description` is the clearest example of a `derived_from` field in the codebase: it is not passed as an explicit constructor argument but recovered from `plan_history.PickUpAction.grasp_description` at runtime. The `derived_from:` key in the schema captures this without inventing a new "optional with history lookup" concept.
  - OpenAction and CloseAction share identical fields (object_designator, arm, grasping_prepose_distance) — a single `fields:` block on the bundled page covers both. The description clarifies applicability.
  - `keep_joint_states` on NavigateAction has no fixed default in the wiki (it is read from `ActionConfig` — a config singleton not yet ingested). Leaving `default:` absent is the correct signal: the agent must not assume a value.
  - `grasping_prepose_distance` on container actions is also config-driven (`derived_from: ActionConfig.grasping_prepose_distance`) rather than a hard-coded constant. Same treatment as `keep_joint_states`.
  - `container.uses` was missing `pycram.datastructures.enums.Arms` despite the body text and execution code using `Arms` for both `GraspingAction(handle, arm, ...)` and `MoveGripperMotion(GripperState.OPEN, arm, ...)`. Caught during `fields:` authoring.
- **open questions:** (none new — Phase 30)

## [2026-05-19] ingest | Phase 31 — fields: mass expansion (schema enrichment Phase 4)

- **scope:** Schema enrichment Phase 4: added `fields:` frontmatter to all remaining action and motion pages. Also caught three missing `Arms` entries in `uses` (robot_body, composite, motions.container) and one missing `AxisIdentifier` entry (robot_body — CarryAction.tip_axis).
- **created:** (none)
- **updated (fields: added):**
  - [[pycram.robot_plans.actions.core.robot_body]] — nested fields per class: MoveTorsoAction, SetGripperAction, ParkArmsAction, CarryAction, FollowToolCenterPointPathAction. `uses` += Arms, AxisIdentifier.
  - [[pycram.robot_plans.actions.core.misc]] — flat fields: technique (str), state (str), object_sem_annotation (SemanticAnnotations), region (Region).
  - [[pycram.robot_plans.actions.composite]] — nested fields: FaceAtAction, SearchAction, TransportAction. `uses` += Arms.
  - [[pycram.robot_plans.motions.gripper]] — nested fields: MoveGripperMotion, MoveToolCenterPointMotion, ReachMotion, MoveTCPWaypointsMotion.
  - [[pycram.robot_plans.motions.navigation]] — flat fields: target (Pose), keep_joint_states (bool, default false).
  - [[pycram.robot_plans.motions.container]] — nested fields: OpeningMotion, ClosingMotion. `uses` += Arms.
  - [[pycram.robot_plans.motions.robot_body]] — nested fields: MoveJointsMotion, LookingMotion.
  - [[pycram.robot_plans.motions.misc]] — flat fields: query (PerceptionQuery).
- **updated (symmetry):**
  - [[pycram.datastructures.enums.Arms]] — `used_by` += robot_body, composite, motions.container.
  - [[pycram.datastructures.enums.AxisIdentifier]] — `used_by` += robot_body.
- **findings:**
  - **Two-tier field schema emerging**: single-class pages (NavigateAction, motions.navigation, motions.misc) use flat `fields:` maps; bundled pages (robot_body, composite, motions.gripper, motions.container, motions.robot_body) use nested per-class maps. The disambiguation rule for agents: if a value under `fields` has a `type:` sub-key it is a field; otherwise it is a class namespace containing fields. This is consistent across all pages.
  - `motions.navigation.keep_joint_states` has `default: false` (set in source), unlike `NavigateAction.keep_joint_states` (read from ActionConfig — no default). These are different levels: the motion has a code default; the action overrides it with a config value. The `derived_from: ActionConfig` approach correctly captures this asymmetry.
  - `DetectionTechnique` and `DetectionState` enums used by `misc/DetectAction` are not yet wiki pages. `type: str` is used as a placeholder; they are candidates for a future Phase 5 enum expansion.
  - `LookingMotion.camera` is typed as `sdt.robots.abstract_robot.AbstractRobot` in the wiki (closest containing page). The actual type is `Camera` — a nested class within AbstractRobot's hierarchy documented on that page but without its own page ID.
  - `ActionDescription` and `BaseMotion` deliberately excluded from `fields:` — they are abstract bases with no own action-specific fields; their subclasses carry all concrete field definitions.
- **open questions:** (none new — Phase 31)

## [2026-05-19] schema | values: and fields: frontmatter additions

- Added `values:` optional field to §3 schema template: flat list of valid enum member names, present only on enum entity pages. Enables agents to enumerate choices from frontmatter without source access.
- Added `fields:` optional field to §3 schema template: typed field map for action/motion entity pages. Two-tier structure: flat map for single-class pages; nested per-class map for bundled pages (distinguished by absence of `type:` sub-key at the class-name level). Supports `type`, `domain`, `default`, `derived_from`, `description` sub-keys.
- Added rules for both new fields to the Rules section of §3.
- No existing pages need re-ingestion; the new fields are additive.

## [2026-05-19] lint | Phase 32 — symmetry audit for Phases 28–31

- **scope:** Full `uses` ↔ `used_by` symmetry audit across all pages created or modified in Phases 28–31 (Phases 28–31 schema enrichment: 4 enum pages + 13 pages with fields: blocks + 5 symmetry edits).
- **created:** (none)
- **updated:** (none — audit-only pass; all violations were already fixed during the ingest phases)
- **findings:**
  - **All symmetry checks PASS.** Verified pairs:
    - `GraspDescription.uses` ∋ Arms/ApproachDirection/VerticalAlignment/AxisIdentifier ↔ each enum page `used_by` ∋ GraspDescription ✓
    - `PickUpAction.uses` ∋ Arms ↔ Arms.`used_by` ∋ PickUpAction ✓
    - `PlaceAction.uses` ∋ Arms ↔ Arms.`used_by` ∋ PlaceAction ✓
    - `container.uses` ∋ Arms ↔ Arms.`used_by` ∋ container ✓
    - `robot_body.uses` ∋ Arms, AxisIdentifier ↔ Arms.`used_by` ∋ robot_body, AxisIdentifier.`used_by` ∋ robot_body ✓
    - `composite.uses` ∋ Arms ↔ Arms.`used_by` ∋ composite ✓
    - `motions.gripper.uses` ∋ Arms ↔ Arms.`used_by` ∋ motions.gripper ✓
    - `motions.container.uses` ∋ Arms ↔ Arms.`used_by` ∋ motions.container ✓
    - `ApproachDirection.uses` ∋ AxisIdentifier ↔ AxisIdentifier.`used_by` ∋ ApproachDirection ✓
    - `VerticalAlignment.uses` ∋ AxisIdentifier ↔ AxisIdentifier.`used_by` ∋ VerticalAlignment ✓
  - **Dangling links: PASS.** All `[[id]]` links in the 4 new enum pages resolve to existing files. All `type:` values in new `fields:` blocks reference either existing wiki pages or declared primitive types (bool/float/str).
  - **ID hygiene: PASS.** All 4 new enum filenames match their `id:` fields exactly.
  - **`fields:` type coverage gaps (noted, not violations):**
    - `misc/DetectAction.technique` typed as `str` — `DetectionTechnique` enum has no wiki page yet. Candidate for a follow-up enum expansion (same pattern as Phase 28).
    - `misc/DetectAction.state` typed as `str` — `DetectionState` has no wiki page yet.
    - `motions.robot_body/LookingMotion.camera` typed as `sdt.robots.abstract_robot.AbstractRobot` — the actual type is `Camera`, a nested annotation class documented on the AbstractRobot page but without its own ID. Acceptable until Camera earns a standalone page.
  - **Stubs (fresh, all 2026-05-19):** none added in Phases 28–31.
  - **Pre-existing open questions not addressed this phase** (deferred from earlier phases): `JointState.used_by` lists `AbstractRobot` without a reciprocal `uses` entry; `GraspDescription.place_pose_sequence` single-held-body assumption; `LookingMotion` camera mutation side effect.
- **open questions:** (none new — Phase 32)
