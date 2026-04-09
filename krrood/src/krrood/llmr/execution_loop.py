"""Generic execution loop — orchestrates NL → decompose → plan → execute.

No robot, sdt, pycram, or arm-state references.  The executor, precondition
planner, and state tracking are all caller-injectable so the loop works for
any action framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing_extensions import Any, Callable, Dict, List, Optional, Set, Tuple

from krrood.llmr.pipeline.action_pipeline import ActionPipeline
from krrood.llmr.pipeline.clarification import ClarificationNeededError, ClarificationRequest
from krrood.llmr.pipeline.dispatcher import ActionSpec
from krrood.llmr.recovery_handler import RecoveryHandler
from krrood.llmr.task_decomposer import DecomposedPlan, TaskDecomposer

logger = logging.getLogger(__name__)

_PlanStep = Tuple[str, Optional["PlanResult"]]


# ── Plan result ────────────────────────────────────────────────────────────────


@dataclass
class PlanResult:
    """Output of the precondition planner for one action."""

    action: ActionSpec
    preconditions: List[Any] = field(default_factory=list)


# ── Execution result ──────────────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    """Outcome of executing a single NL instruction."""

    instruction: str
    action: Optional[ActionSpec]
    preconditions: List[Any]
    success: bool
    error: Optional[Exception] = None
    clarification: Optional[ClarificationRequest] = None
    skipped: bool = False


# ── ExecutionLoop ─────────────────────────────────────────────────────────────


@dataclass
class ExecutionLoop:
    """Orchestrates NL instructions → decompose → plan → execute.

    All robot/framework specifics are injected by the caller:

    :param pipeline: :class:`ActionPipeline` that converts NL → :class:`ActionSpec`.
    :param executor: Callable that executes a list of actions (preconditions + main).
        Signature: ``(preconditions: List[Any], action: ActionSpec) -> None``.
    :param stop_on_failure: Whether to skip remaining steps after the first failure.
    :param decomposer: Optional :class:`TaskDecomposer` for compound instructions.
    :param recovery_handler: Optional :class:`RecoveryHandler` for retry on failure.
    :param precondition_planner: Optional callable computing preconditions for an
        :class:`ActionSpec`.  Signature: ``(ActionSpec) -> List[Any]``.
    """

    pipeline: ActionPipeline
    executor: Callable[[List[Any], ActionSpec], None]
    stop_on_failure: bool = field(default=True)
    decomposer: Optional[TaskDecomposer] = field(default=None)
    recovery_handler: Optional[RecoveryHandler] = field(default=None)
    precondition_planner: Optional[Callable[[ActionSpec], List[Any]]] = field(default=None)

    # ── Public API ─────────────────────────────────────────────────────────────

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

    # ── Phase helpers ──────────────────────────────────────────────────────────

    def _decompose_all(
        self, instructions: List[str]
    ) -> Tuple[List[str], Dict[int, List[int]]]:
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
                    instruction, len(plan.steps), plan.dependencies,
                )
            else:
                atomic_instructions.append(instruction)

        return atomic_instructions, global_deps

    def _plan_all(
        self,
        atomic_instructions: List[str],
        global_deps: Dict[int, List[int]],
    ) -> Tuple[List[_PlanStep], List[ExecutionResult]]:
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
                action = self.pipeline.run(instruction)
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

            preconditions: List[Any] = []
            if self.precondition_planner is not None:
                try:
                    preconditions = self.precondition_planner(action)
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

            plan_steps.append((instruction, PlanResult(action=action, preconditions=preconditions)))

        return plan_steps, []

    def _execute_all(self, plan_steps: List[_PlanStep]) -> List[ExecutionResult]:
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
                    instruction=instruction, action=plan_result.action,
                    preconditions=plan_result.preconditions, success=False, skipped=True,
                ))
                continue

            result = self._execute_with_recovery(plan_result, instruction)
            if not result.success:
                failed = True
            results.append(result)

        return results

    # ── Per-step execution ─────────────────────────────────────────────────────

    def _execute_with_recovery(self, plan_result: PlanResult, instruction: str) -> ExecutionResult:
        current_action = plan_result.action
        current_preconditions = plan_result.preconditions
        max_attempts = (
            1 + self.recovery_handler.max_retries if self.recovery_handler is not None else 1
        )
        last_exc: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                self.executor(current_preconditions, current_action)
                return ExecutionResult(
                    instruction=instruction, action=current_action,
                    preconditions=current_preconditions, success=True,
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
                    attempt_number=attempt + 1,
                )
                if not recovery.success:
                    last_exc = recovery.error or exc
                    break
                current_action = recovery.action
                current_preconditions = recovery.preconditions

        return ExecutionResult(
            instruction=instruction, action=current_action,
            preconditions=current_preconditions, success=False, error=last_exc,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _make_early_exit(
        plan_steps: List[_PlanStep], failed_result: ExecutionResult
    ) -> List[ExecutionResult]:
        results = [
            ExecutionResult(
                instruction=ins,
                action=pr.action if pr else None,
                preconditions=pr.preconditions if pr else [],
                success=False, skipped=True,
            )
            for ins, pr in plan_steps
        ]
        results.append(failed_result)
        return results
