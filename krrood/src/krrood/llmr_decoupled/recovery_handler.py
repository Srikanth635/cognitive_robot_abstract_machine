"""LLM-driven recovery handler — no robot, sdt, pycram, or WorldLike references.

World context is derived from SymbolGraph via the pipeline's serialiser,
or supplied via the caller's world_context_provider.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing_extensions import Any, List, Optional

from krrood.llmr_decoupled.pipeline.action_pipeline import ActionPipeline
from krrood.llmr_decoupled.pipeline.dispatcher import ActionSpec
from krrood.llmr_decoupled.workflows.nodes.recovery_resolver import run_recovery_resolver
from krrood.llmr_decoupled.workflows.schemas.recovery import RecoverySchema

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class RecoveryAttemptResult:
    """Outcome of a single recovery attempt."""

    success: bool
    action: Optional[ActionSpec]
    preconditions: List[Any]
    recovery_schema: Optional[RecoverySchema]
    attempt_number: int
    error: Optional[Exception] = None


# ── Action serialiser ─────────────────────────────────────────────────────────


def _serialise_failed_action(action: Optional[ActionSpec]) -> str:
    """Produce a human-readable description of a failed action for the LLM."""
    if action is None:
        return "No action was produced (pipeline failed before action construction)."

    lines = [f"Action type: {action.action_type}"]
    for key, val in action.parameters.items():
        lines.append(f"  {key}: {val}")
    for role, entity in action.grounded_entities.items():
        entity_name = getattr(getattr(entity, "name", None), "name", str(entity))
        lines.append(f"  {role}: {entity_name}")
    return "\n".join(lines)


# ── RecoveryHandler ───────────────────────────────────────────────────────────


@dataclass
class RecoveryHandler:
    """Performs LLM-driven replanning after a failed action execution.

    No robot/framework knowledge — uses :class:`ActionPipeline` for replanning
    and derives world context from the pipeline's SymbolGraph serialiser.

    :param pipeline: The :class:`ActionPipeline` used for replanning.
    :param max_retries: Maximum recovery attempts per failed action.
    :param precondition_planner: Optional caller-supplied callable that computes
        preconditions for a recovered :class:`ActionSpec`.  Signature:
        ``(ActionSpec) -> List[Any]``.
    """

    pipeline: ActionPipeline
    max_retries: int = field(default=1)
    precondition_planner: Optional[Any] = field(default=None)

    def attempt_recovery(
        self,
        instruction: str,
        failed_action: Optional[ActionSpec],
        error: Exception,
        attempt_number: int = 1,
    ) -> RecoveryAttemptResult:
        """Perform one recovery attempt for a failed action.

        :param instruction: Original NL instruction that triggered the failure.
        :param failed_action: The :class:`ActionSpec` that failed, or ``None``.
        :param error: The exception raised during execution.
        :param attempt_number: Current attempt count (1-based), for logging.
        :return: :class:`RecoveryAttemptResult` describing the outcome.
        """
        logger.info(
            "RecoveryHandler: attempt %d for '%s' (error: %s)",
            attempt_number, instruction, error,
        )

        world_ctx = self.pipeline._get_world_context()
        failed_action_desc = _serialise_failed_action(failed_action)
        error_message = f"{type(error).__name__}: {error}"

        schema = run_recovery_resolver(
            world_context=world_ctx,
            original_instruction=instruction,
            failed_action_description=failed_action_desc,
            error_message=error_message,
        )

        if schema is None:
            logger.warning("RecoveryHandler: LLM call failed for attempt %d of '%s'.", attempt_number, instruction)
            return RecoveryAttemptResult(
                success=False, action=None, preconditions=[], recovery_schema=None,
                attempt_number=attempt_number,
                error=RuntimeError("Recovery resolver LLM call returned None."),
            )

        logger.info("RecoveryHandler: strategy=%s  diagnosis=%s", schema.recovery_strategy, schema.failure_diagnosis)

        if schema.recovery_strategy == "ABORT":
            logger.warning("RecoveryHandler: ABORT for '%s'. Reasoning: %s", instruction, schema.reasoning)
            return RecoveryAttemptResult(
                success=False, action=None, preconditions=[], recovery_schema=schema,
                attempt_number=attempt_number,
                error=RuntimeError(f"Recovery aborted: {schema.failure_diagnosis}"),
            )

        revised_instruction = schema.revised_instruction
        if not revised_instruction:
            logger.error("RecoveryHandler: REPLAN_FULL but revised_instruction is empty — treating as ABORT.")
            return RecoveryAttemptResult(
                success=False, action=None, preconditions=[], recovery_schema=schema,
                attempt_number=attempt_number,
                error=RuntimeError("Recovery strategy is REPLAN_FULL but revised_instruction is empty."),
            )

        logger.info("RecoveryHandler: replanning '%s' → '%s'", instruction, revised_instruction)

        try:
            new_action = self.pipeline.run(revised_instruction)
        except Exception as pipeline_exc:
            logger.error("RecoveryHandler: pipeline failed for revised instruction '%s': %s", revised_instruction, pipeline_exc)
            return RecoveryAttemptResult(
                success=False, action=None, preconditions=[], recovery_schema=schema,
                attempt_number=attempt_number, error=pipeline_exc,
            )

        preconditions: List[Any] = []
        if self.precondition_planner is not None:
            try:
                preconditions = self.precondition_planner(new_action)
            except Exception as plan_exc:
                logger.error("RecoveryHandler: precondition planning failed: %s", plan_exc)
                return RecoveryAttemptResult(
                    success=False, action=new_action, preconditions=[], recovery_schema=schema,
                    attempt_number=attempt_number, error=plan_exc,
                )

        return RecoveryAttemptResult(
            success=True, action=new_action, preconditions=preconditions,
            recovery_schema=schema, attempt_number=attempt_number,
        )
