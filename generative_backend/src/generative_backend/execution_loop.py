"""Execution loop: runs NL instructions end-to-end with automatic precondition planning.

This is the top-level entry point for automated execution of instruction sequences.
For each instruction the loop:

  1. Runs the LLM pipeline (ActionPipeline) to produce a typed task action.
  2. Uses MotionPreconditionPlanner to compute the required preparatory actions
     (navigate, move torso, park arms) and resolve any spatial parameters
     (e.g. placement pose via SemanticCostmapLocation).
  3. Executes all preparatory actions + the task action as a single SequentialPlan.
  4. Updates the internal ExecutionState so subsequent instructions benefit from
     information gathered so far (e.g. which arm is holding an object).

Adding support for a new action type requires:
  - Registering an ActionHandler in ActionDispatcher.
  - Registering a PreconditionProvider in MotionPreconditionPlanner.
  ExecutionLoop itself requires zero changes.

Usage::

    from generative_backend.execution_loop import ExecutionLoop
    from generative_backend.action_pipeline import ActionPipeline

    pipeline = ActionPipeline.from_world(world)
    loop = ExecutionLoop(
        world=world,
        pipeline=pipeline,
        context=context,
        robot_context=lambda: simulated_robot,   # optional
    )

    results = loop.run([
        "Pick up the milk from the fridge",
        "Place it on the kitchen counter",
    ])
    for r in results:
        print(r.instruction, "→", "OK" if r.success else r.error)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Dict, List, Optional, Set, Tuple

from semantic_digital_twin.world import World

from pycram.datastructures.dataclasses import Context
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.base import ActionDescription

from .pipeline.action_pipeline import ActionPipeline
from .pipeline.clarification import (
    ArmCapacityError,
    ArmCapacityRequest,
    ClarificationNeededError,
    ClarificationRequest,
)
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from .planning.motion_precondition_planner import ExecutionState, MotionPreconditionPlanner, PreconditionResult
from .recovery_handler import RecoveryHandler
from .task_decomposer import DecomposedPlan, TaskDecomposer

logger = logging.getLogger(__name__)

_ROBOT_ARM_COUNT = 2  # PR2 has two arms (LEFT + RIGHT)


# ── Result dataclass ────────────────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    """Outcome of executing a single NL instruction.

    :param instruction: The original natural language instruction.
    :param action: The task action produced by the LLM pipeline (possibly updated
                   by the precondition planner, e.g. resolved target_location).
    :param preconditions: Preparatory actions that were prepended to *action*.
    :param success: Whether the instruction completed without exception.
    :param error: The exception that caused failure, if any.
    """

    instruction: str
    action: Optional[ActionDescription]
    preconditions: List[ActionDescription]
    success: bool
    error: Optional[Exception] = None
    clarification: Optional[ClarificationRequest] = None
    arm_capacity_error: Optional[ArmCapacityRequest] = None
    skipped: bool = False


# ── ExecutionLoop ───────────────────────────────────────────────────────────────


@dataclass
class ExecutionLoop:
    """Automated execution loop: NL instructions → preconditions + actions → robot.

    :param world: The Semantic Digital Twin world instance.
    :param pipeline: Configured ``ActionPipeline`` for NL → action translation.
    :param context: ``Context`` (world + robot) used when constructing SequentialPlans.
    :param robot_context: Optional zero-argument callable that returns a context manager
        to wrap each ``SequentialPlan.perform()`` call.
        Example: ``robot_context=lambda: simulated_robot``.
        When ``None``, plans are executed without an additional context manager.
    :param stop_on_failure: If ``True`` (default), the loop stops at the first failed
        instruction.  Set to ``False`` to attempt all instructions regardless.
    """

    world: World
    pipeline: ActionPipeline
    context: Context
    robot_context: Optional[Callable[[], ContextManager]] = field(default=None)
    stop_on_failure: bool = field(default=True)
    decomposer: Optional[TaskDecomposer] = field(default=None)
    recovery_handler: Optional[RecoveryHandler] = field(default=None)

    # ── Internal state (not passed at construction) ─────────────────────────────
    _planner: MotionPreconditionPlanner = field(init=False)
    _exec_state: ExecutionState = field(init=False)

    def __post_init__(self) -> None:
        self._planner = MotionPreconditionPlanner(self.world, self.context)
        self._exec_state = ExecutionState()

    # ── Public API ──────────────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Reset the cross-instruction execution state (held object, active arm).

        Call this before re-running a sequence on a freshly reset world.
        """
        self._exec_state = ExecutionState()

    def run(self, instructions: List[str]) -> List[ExecutionResult]:
        """Plan all instructions, then execute them as one combined SequentialPlan.

        Running as a single plan is required so that actions like ``PlaceAction``
        can find preceding actions (e.g. ``PickUpAction``) via
        ``plan.get_previous_node_by_designator_type``.

        Planning uses a local ``ExecutionState`` that is updated after each
        instruction is planned so that cross-instruction dependencies (e.g.
        pickup arm → place arm) are resolved correctly before execution starts.

        :param instructions: Natural language instructions in execution order.
        :return: One :class:`ExecutionResult` per instruction.
        """
        # ── Phase 0: decompose compound instructions ──────────────────────────
        # Collect atomic steps and merge per-instruction dependency graphs into
        # a single global dependency map (indices offset per instruction block).
        atomic_instructions: List[str] = []
        global_deps: Dict[int, List[int]] = {}  # global step idx → prerequisite indices

        for instruction in instructions:
            if self.decomposer is not None:
                plan: DecomposedPlan = self.decomposer.decompose(instruction)
                offset = len(atomic_instructions)
                atomic_instructions.extend(plan.steps)
                for step_idx, deps in plan.dependencies.items():
                    global_deps[offset + step_idx] = [offset + d for d in deps]
                logger.debug(
                    "ExecutionLoop: '%s' → %d sub-instructions, deps=%s",
                    instruction, len(plan.steps), plan.dependencies,
                )
            else:
                atomic_instructions.append(instruction)

        # ── Phase 1+2: plan all instructions sequentially ────────────────────
        # Seed planning_state from the real exec_state so prior executions
        # (from previous run() calls) are visible to the LLM during planning.
        planning_state = self._exec_state.copy()

        # plan_steps entries are (instruction, PreconditionResult) for successful
        # steps and (instruction, None) for skipped steps.
        plan_steps: List[Tuple[str, Optional["PreconditionResult"]]] = []
        failed_step_indices: Set[int] = set()

        for i, instruction in enumerate(atomic_instructions):
            # ── Dependency check: skip if any prerequisite failed ─────────────
            blocking_deps = [
                d for d in global_deps.get(i, []) if d in failed_step_indices
            ]
            if blocking_deps:
                logger.info(
                    "ExecutionLoop: skipping '%s' — prerequisite step(s) %s failed.",
                    instruction, blocking_deps,
                )
                plan_steps.append((instruction, None))
                failed_step_indices.add(i)
                continue

            logger.info("ExecutionLoop: planning '%s'", instruction)

            try:
                action = self.pipeline.run(instruction, exec_state=planning_state)
            except ClarificationNeededError as exc:
                logger.warning("Clarification needed for '%s': %s", instruction, exc)
                failed_step_indices.add(i)
                plan_steps.append((instruction, None))
                # Surface clarification immediately — stop planning
                already_planned = [
                    ExecutionResult(
                        instruction=ins,
                        action=pr.action if pr else None,
                        preconditions=pr.preconditions if pr else [],
                        success=False,
                        skipped=(pr is None),
                    )
                    for ins, pr in plan_steps[:-1]
                ]
                return already_planned + [
                    ExecutionResult(
                        instruction=instruction, action=None, preconditions=[],
                        success=False, error=exc, clarification=exc.request,
                    )
                ]
            except Exception as exc:
                logger.error("Pipeline failed for '%s': %s", instruction, exc)
                failed_step_indices.add(i)
                plan_steps.append((instruction, None))
                already_planned = [
                    ExecutionResult(
                        instruction=ins,
                        action=pr.action if pr else None,
                        preconditions=pr.preconditions if pr else [],
                        success=False,
                        skipped=(pr is None),
                    )
                    for ins, pr in plan_steps[:-1]
                ]
                return already_planned + [
                    ExecutionResult(
                        instruction=instruction, action=None, preconditions=[],
                        success=False, error=exc,
                    )
                ]

            # ── Arm capacity check ────────────────────────────────────────────
            if isinstance(action, PickUpAction):
                occupied = {
                    arm: body
                    for arm, body in planning_state.held_objects.items()
                    if body is not None
                }
                if len(occupied) >= _ROBOT_ARM_COUNT:
                    held_names = [
                        str(getattr(getattr(b, "name", None), "name", b))
                        for b in occupied.values()
                    ]
                    exc = ArmCapacityError(
                        ArmCapacityRequest(
                            occupied_arms=[a.name for a in occupied],
                            held_object_names=held_names,
                            message=(
                                f"Cannot pick up: all {_ROBOT_ARM_COUNT} arms are occupied "
                                f"(holding {held_names}). Place an object first."
                            ),
                        )
                    )
                    logger.warning("Arm capacity exceeded for '%s': %s", instruction, exc)
                    failed_step_indices.add(i)
                    plan_steps.append((instruction, None))
                    already_planned = [
                        ExecutionResult(
                            instruction=ins,
                            action=pr.action if pr else None,
                            preconditions=pr.preconditions if pr else [],
                            success=False,
                            skipped=(pr is None),
                        )
                        for ins, pr in plan_steps[:-1]
                    ]
                    return already_planned + [
                        ExecutionResult(
                            instruction=instruction, action=None, preconditions=[],
                            success=False, error=exc, arm_capacity_error=exc.request,
                        )
                    ]

            try:
                plan_result = self._planner.compute(action, planning_state)
            except Exception as exc:
                logger.error("Precondition planning failed for '%s': %s", instruction, exc)
                failed_step_indices.add(i)
                plan_steps.append((instruction, None))
                already_planned = [
                    ExecutionResult(
                        instruction=ins,
                        action=pr.action if pr else None,
                        preconditions=pr.preconditions if pr else [],
                        success=False,
                        skipped=(pr is None),
                    )
                    for ins, pr in plan_steps[:-1]
                ]
                return already_planned + [
                    ExecutionResult(
                        instruction=instruction, action=action, preconditions=[],
                        success=False, error=exc,
                    )
                ]

            # Update planning_state immediately so the next instruction's
            # precondition provider can reference results from this one
            # (e.g. PlaceAction needs to know which arm PickUpAction used).
            self._planner.update_state(plan_result.action, planning_state)
            plan_steps.append((instruction, plan_result))

        if not plan_steps:
            return []

        # ── Phase 3: flatten planned (non-skipped) actions into one combined plan
        all_actions = []
        for _, plan_result in plan_steps:
            if plan_result is None:
                continue
            all_actions.extend(plan_result.preconditions)
            all_actions.append(plan_result.action)

        logger.info(
            "ExecutionLoop: combined plan → [%s]",
            ", ".join(type(a).__name__ for a in all_actions),
        )

        # ── Phase 4: execute combined plan ────────────────────────────────────
        error: Optional[Exception] = None
        if all_actions:
            try:
                self._execute(all_actions)
            except Exception as exc:
                logger.error("Combined execution failed: %s", exc)
                error = exc

        # ── Phase 5: update real exec_state on success ────────────────────────
        if error is None:
            for _, plan_result in plan_steps:
                if plan_result is not None:
                    self._planner.update_state(plan_result.action, self._exec_state)

        return [
            ExecutionResult(
                instruction=instruction,
                action=plan_result.action if plan_result else None,
                preconditions=plan_result.preconditions if plan_result else [],
                success=(plan_result is not None and error is None),
                error=error if plan_result is not None else None,
                skipped=(plan_result is None),
            )
            for instruction, plan_result in plan_steps
        ]

    def run_single(self, instruction: str) -> ExecutionResult:
        """Run a single NL instruction through the full pipeline and execute it.

        Phases:
          1. LLM pipeline  → task action
          2. Precondition planner → preparatory actions + (possibly updated) task action
          3. SequentialPlan.perform()
          4. Update ExecutionState for the next instruction

        :param instruction: Natural language instruction.
        :return: :class:`ExecutionResult` describing what happened.
        """
        logger.info("ExecutionLoop.run_single: '%s'", instruction)

        # ── Phase 1: LLM pipeline ──────────────────────────────────────────────
        try:
            action = self.pipeline.run(instruction, exec_state=self._exec_state)
        except ClarificationNeededError as exc:
            logger.warning("Clarification needed for '%s': %s", instruction, exc)
            return ExecutionResult(
                instruction=instruction,
                action=None,
                preconditions=[],
                success=False,
                error=exc,
                clarification=exc.request,
            )
        except Exception as exc:
            logger.error("Pipeline failed for '%s': %s", instruction, exc)
            return ExecutionResult(
                instruction=instruction,
                action=None,
                preconditions=[],
                success=False,
                error=exc,
            )

        # ── Phase 2: precondition planning ────────────────────────────────────
        try:
            plan_result = self._planner.compute(action, self._exec_state)
        except Exception as exc:
            logger.error(
                "Precondition planning failed for '%s': %s", instruction, exc
            )
            return ExecutionResult(
                instruction=instruction,
                action=action,
                preconditions=[],
                success=False,
                error=exc,
            )

        all_actions = plan_result.preconditions + [plan_result.action]
        logger.info(
            "ExecutionLoop: '%s' → [%s]",
            instruction,
            ", ".join(type(a).__name__ for a in all_actions),
        )

        # ── Phase 3: execute (with optional recovery loop) ────────────────────
        current_action = plan_result.action
        current_preconditions = plan_result.preconditions
        max_attempts = (
            1 + self.recovery_handler.max_retries
            if self.recovery_handler is not None else 1
        )
        last_exc: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                self._execute(current_preconditions + [current_action])
                # ── Phase 4: update state on success ──────────────────────────
                self._planner.update_state(current_action, self._exec_state)
                return ExecutionResult(
                    instruction=instruction,
                    action=current_action,
                    preconditions=current_preconditions,
                    success=True,
                )
            except Exception as exc:
                last_exc = exc
                logger.error(
                    "Execution failed for '%s' (attempt %d/%d): %s",
                    instruction, attempt + 1, max_attempts, exc,
                )
                if self.recovery_handler is None or attempt + 1 >= max_attempts:
                    break
                recovery_result = self.recovery_handler.attempt_recovery(
                    instruction=instruction,
                    failed_action=current_action,
                    error=exc,
                    exec_state=self._exec_state,
                    pipeline=self.pipeline,
                    planner=self._planner,
                    attempt_number=attempt + 1,
                )
                if not recovery_result.success:
                    last_exc = recovery_result.error or exc
                    break
                current_action = recovery_result.action
                current_preconditions = recovery_result.preconditions

        return ExecutionResult(
            instruction=instruction,
            action=current_action,
            preconditions=current_preconditions,
            success=False,
            error=last_exc,
        )

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _execute(self, actions: List[Any]) -> None:
        """Wrap *actions* in a sequential plan node and perform it."""
        node = sequential(actions, context=self.context)
        if self.robot_context is not None:
            with self.robot_context():
                node.perform()
        else:
            node.perform()
