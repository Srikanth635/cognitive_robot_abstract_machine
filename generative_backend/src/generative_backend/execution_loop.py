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
from typing import Callable, ContextManager, List, Optional, Tuple, Union

from semantic_digital_twin.world import World

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.partial_designator import PartialDesignator
from pycram.language import SequentialPlan
from pycram.robot_plans.actions.base import ActionDescription

from .pipeline.action_pipeline import ActionPipeline
from .planning.motion_precondition_planner import ExecutionState, MotionPreconditionPlanner, PreconditionResult
from .task_decomposer import TaskDecomposer

logger = logging.getLogger(__name__)


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

    # ── Internal state (not passed at construction) ─────────────────────────────
    _planner: MotionPreconditionPlanner = field(init=False)
    _exec_state: ExecutionState = field(init=False)

    def __post_init__(self) -> None:
        self._planner = MotionPreconditionPlanner(self.world)
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
        atomic_instructions: List[str] = []
        for instruction in instructions:
            if self.decomposer is not None:
                expanded = self.decomposer.decompose(instruction)
                logger.debug(
                    "ExecutionLoop: '%s' → %s sub-instructions", instruction, len(expanded)
                )
                atomic_instructions.extend(expanded)
            else:
                atomic_instructions.append(instruction)

        # ── Phase 1+2: plan all instructions sequentially ────────────────────
        plan_steps: List[Tuple[str, "PreconditionResult"]] = []
        # Seed planning_state from the real exec_state so prior executions
        # (from previous run() calls) are visible to the LLM during planning.
        planning_state = self._exec_state.copy()

        for instruction in atomic_instructions:
            logger.info("ExecutionLoop: planning '%s'", instruction)

            try:
                action = self.pipeline.run(instruction, exec_state=planning_state)
            except Exception as exc:
                logger.error("Pipeline failed for '%s': %s", instruction, exc)
                failed = ExecutionResult(
                    instruction=instruction, action=None, preconditions=[],
                    success=False, error=exc,
                )
                planned = [
                    ExecutionResult(
                        instruction=i, action=pr.action, preconditions=pr.preconditions,
                        success=False, error=exc,
                    )
                    for i, pr in plan_steps
                ]
                return planned + [failed]

            try:
                plan_result = self._planner.compute(action, planning_state)
            except Exception as exc:
                logger.error("Precondition planning failed for '%s': %s", instruction, exc)
                failed = ExecutionResult(
                    instruction=instruction, action=action, preconditions=[],
                    success=False, error=exc,
                )
                planned = [
                    ExecutionResult(
                        instruction=i, action=pr.action, preconditions=pr.preconditions,
                        success=False, error=exc,
                    )
                    for i, pr in plan_steps
                ]
                return planned + [failed]

            # Update planning_state immediately so the next instruction's
            # precondition provider can reference results from this one
            # (e.g. PlaceAction needs to know which arm PickUpAction used).
            self._planner.update_state(plan_result.action, planning_state)
            plan_steps.append((instruction, plan_result))

        if not plan_steps:
            return []

        # ── Phase 3: flatten all actions into one combined plan ───────────────
        all_actions = []
        for _, plan_result in plan_steps:
            all_actions.extend(plan_result.preconditions)
            all_actions.append(plan_result.action)

        logger.info(
            "ExecutionLoop: combined plan → [%s]",
            ", ".join(type(a).__name__ for a in all_actions),
        )

        # ── Phase 4: execute combined plan ────────────────────────────────────
        error: Optional[Exception] = None
        try:
            self._execute(all_actions)
        except Exception as exc:
            logger.error("Combined execution failed: %s", exc)
            error = exc

        # ── Phase 5: update real exec_state on success ────────────────────────
        if error is None:
            for _, plan_result in plan_steps:
                self._planner.update_state(plan_result.action, self._exec_state)

        return [
            ExecutionResult(
                instruction=instruction,
                action=plan_result.action,
                preconditions=plan_result.preconditions,
                success=(error is None),
                error=error,
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

        # ── Phase 3: execute ───────────────────────────────────────────────────
        try:
            self._execute(all_actions)
        except Exception as exc:
            logger.error("Execution failed for '%s': %s", instruction, exc)
            return ExecutionResult(
                instruction=instruction,
                action=plan_result.action,
                preconditions=plan_result.preconditions,
                success=False,
                error=exc,
            )

        # ── Phase 4: update state ──────────────────────────────────────────────
        self._planner.update_state(plan_result.action, self._exec_state)

        return ExecutionResult(
            instruction=instruction,
            action=plan_result.action,
            preconditions=plan_result.preconditions,
            success=True,
        )

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _execute(self, actions: List[Union[ActionDescription, PartialDesignator]]) -> None:
        """Wrap *actions* in a SequentialPlan and perform it."""
        plan = SequentialPlan(self.context, *actions)
        if self.robot_context is not None:
            with self.robot_context():
                plan.perform()
        else:
            plan.perform()
