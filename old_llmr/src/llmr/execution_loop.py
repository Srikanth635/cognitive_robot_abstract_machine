
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING, Any, Callable, ContextManager, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from llmr.sdt_interfaces import WorldLike

from pycram.datastructures.dataclasses import Context
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.base import ActionDescription

from llmr.pipeline.action_pipeline import ActionPipeline
from llmr.pipeline.clarification import (
    ArmCapacityError,
    ArmCapacityRequest,
    ClarificationNeededError,
    ClarificationRequest,
)
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from llmr.planning.motion_precondition_planner import (
    ExecutionState,
    MotionPreconditionPlanner,
    PreconditionResult,
)
from llmr.recovery_handler import RecoveryHandler
from llmr.task_decomposer import DecomposedPlan, TaskDecomposer

logger = logging.getLogger(__name__)

_ROBOT_ARM_COUNT = 2  # PR2 has two arms (LEFT + RIGHT) # CHANGE NEEDED HERE

_PlanStep = Tuple[str, Optional[PreconditionResult]]


# ── Result dataclass ────────────────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    """Outcome of executing a single NL instruction."""

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
    """Orchestrates NL instructions → decompose → plan → execute."""

    world: "WorldLike"
    pipeline: ActionPipeline
    context: Context
    robot_context: Optional[Callable[[], ContextManager]] = field(default=None)
    stop_on_failure: bool = field(default=True)
    decomposer: Optional[TaskDecomposer] = field(default=None)
    recovery_handler: Optional[RecoveryHandler] = field(default=None)

    _planner: MotionPreconditionPlanner = field(init=False)
    _exec_state: ExecutionState = field(init=False)

    def __post_init__(self) -> None:
        self._planner = MotionPreconditionPlanner(self.world, self.context)
        self._exec_state = ExecutionState()

    # ── Public API ──────────────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Reset cross-instruction state (held objects, active arm)."""
        self._exec_state = ExecutionState()

    def run(self, instructions: List[str]) -> List[ExecutionResult]:
        """Decompose, plan, then execute *instructions* in order.

        :param instructions: Natural language instructions in execution order.
        :return: One :class:`ExecutionResult` per atomic step.
        """
        atomic_instructions, global_deps = self._decompose_all(instructions)
        plan_steps, early_exit = self._plan_all(atomic_instructions, global_deps)
        if early_exit:
            return early_exit
        return self._execute_all(plan_steps)

    # ── Phase helpers ───────────────────────────────────────────────────────────

    def _decompose_all(
        self,
        instructions: List[str],
    ) -> Tuple[List[str], Dict[int, List[int]]]:
        """Decompose each instruction into atomic steps; merge dependency graphs."""
        atomic_instructions: List[str] = []
        global_deps: Dict[int, List[int]] = {}

        for instruction in instructions:
            if self.decomposer is not None:
                plan: DecomposedPlan = self.decomposer.decompose(instruction)
                start_idx = len(atomic_instructions)
                atomic_instructions.extend(plan.steps)
                for step_idx, deps in plan.dependencies.items():
                    global_deps[start_idx + step_idx] = [start_idx + d for d in deps]
                logger.debug(
                    "ExecutionLoop: '%s' → %d sub-instructions, deps=%s",
                    instruction,
                    len(plan.steps),
                    plan.dependencies,
                )
            else:
                atomic_instructions.append(instruction)

        return atomic_instructions, global_deps

    def _plan_all(
        self,
        atomic_instructions: List[str],
        global_deps: Dict[int, List[int]],
    ) -> Tuple[List[_PlanStep], List[ExecutionResult]]:
        """Plan every atomic step sequentially.

        Returns ``(plan_steps, early_exit)``.  ``early_exit`` is non-empty only
        when planning must abort (clarification needed, arm capacity exceeded,
        unrecoverable error).
        """
        planning_state = self._exec_state.copy()
        plan_steps: List[_PlanStep] = []
        failed_indices: Set[int] = set()

        for i, instruction in enumerate(atomic_instructions):
            blocking = [d for d in global_deps.get(i, []) if d in failed_indices]
            if blocking:
                logger.info(
                    "ExecutionLoop: skipping '%s' — prerequisite step(s) %s failed.",
                    instruction, blocking,
                )
                plan_steps.append((instruction, None))
                failed_indices.add(i)
                continue

            logger.info("ExecutionLoop: planning '%s'", instruction)

            try:
                action = self.pipeline.run(instruction, exec_state=planning_state)
            except ClarificationNeededError as exc:
                logger.warning("Clarification needed for '%s': %s", instruction, exc)
                failed_indices.add(i)
                return plan_steps, self._make_early_exit(
                    plan_steps,
                    ExecutionResult(
                        instruction=instruction, action=None, preconditions=[],
                        success=False, error=exc, clarification=exc.request,
                    ),
                )
            except Exception as exc:
                logger.error("Pipeline failed for '%s': %s", instruction, exc)
                failed_indices.add(i)
                return plan_steps, self._make_early_exit(
                    plan_steps,
                    ExecutionResult(
                        instruction=instruction, action=None, preconditions=[],
                        success=False, error=exc,
                    ),
                )

            if isinstance(action, PickUpAction):    # CHANGE NEEDED HERE
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
                    failed_indices.add(i)
                    return plan_steps, self._make_early_exit(
                        plan_steps,
                        ExecutionResult(
                            instruction=instruction, action=None, preconditions=[],
                            success=False, error=exc, arm_capacity_error=exc.request,
                        ),
                    )

            try:
                plan_result = self._planner.compute(action, planning_state)
            except Exception as exc:
                logger.error("Precondition planning failed for '%s': %s", instruction, exc)
                failed_indices.add(i)
                return plan_steps, self._make_early_exit(
                    plan_steps,
                    ExecutionResult(
                        instruction=instruction, action=action, preconditions=[],
                        success=False, error=exc,
                    ),
                )

            self._planner.update_state(plan_result.action, planning_state)
            plan_steps.append((instruction, plan_result))

        return plan_steps, []

    def _execute_all(
        self,
        plan_steps: List[_PlanStep],
    ) -> List[ExecutionResult]:
        """Execute each planned step; honour stop_on_failure."""
        results: List[ExecutionResult] = []
        failed = False

        for instruction, plan_result in plan_steps:
            if plan_result is None:
                results.append(ExecutionResult(
                    instruction=instruction, action=None, preconditions=[],
                    success=False, skipped=True,
                ))
                continue

            if failed and self.stop_on_failure:
                results.append(ExecutionResult(
                    instruction=instruction,
                    action=plan_result.action,
                    preconditions=plan_result.preconditions,
                    success=False, skipped=True,
                ))
                continue

            result = self._execute_with_recovery(plan_result, instruction)
            if result.success:
                self._planner.update_state(plan_result.action, self._exec_state)
            else:
                failed = True
            results.append(result)

        return results

    # ── Per-step execution ──────────────────────────────────────────────────────

    def _execute_with_recovery(
        self,
        plan_result: PreconditionResult,
        instruction: str,
    ) -> ExecutionResult:
        """Execute one planned step with optional recovery retries."""
        current_action = plan_result.action
        current_preconditions = plan_result.preconditions
        max_attempts = (
            1 + self.recovery_handler.max_retries if self.recovery_handler is not None else 1
        )
        last_exc: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                self._execute(current_preconditions + [current_action])
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
                recovery = self.recovery_handler.attempt_recovery(
                    instruction=instruction,
                    failed_action=current_action,
                    error=exc,
                    exec_state=self._exec_state,
                    pipeline=self.pipeline,
                    planner=self._planner,
                    attempt_number=attempt + 1,
                )
                if not recovery.success:
                    last_exc = recovery.error or exc
                    break
                current_action = recovery.action
                current_preconditions = recovery.preconditions

        return ExecutionResult(
            instruction=instruction,
            action=current_action,
            preconditions=current_preconditions,
            success=False,
            error=last_exc,
        )

    # ── Internal helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _make_early_exit(
        plan_steps: List[_PlanStep],
        failed_result: ExecutionResult,
    ) -> List[ExecutionResult]:
        """Return results for all previously planned (unexecuted) steps plus the failed one."""
        results = [
            ExecutionResult(
                instruction=ins,
                action=pr.action if pr else None,
                preconditions=pr.preconditions if pr else [],
                success=False,
                skipped=True,
            )
            for ins, pr in plan_steps
        ]
        results.append(failed_result)
        return results

    def _execute(self, actions: List[Any]) -> None:
        """Wrap *actions* in a sequential plan node and perform it."""
        node = sequential(actions, context=self.context)
        if self.robot_context is not None:
            with self.robot_context():
                node.perform()
        else:
            node.perform()
