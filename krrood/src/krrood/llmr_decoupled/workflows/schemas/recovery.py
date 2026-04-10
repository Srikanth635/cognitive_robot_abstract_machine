"""Pydantic schema for recovery resolution — unchanged from original llmr."""

from __future__ import annotations

from typing_extensions import Literal, Optional

from pydantic import BaseModel, Field

__all__ = ["RecoverySchema"]


class RecoverySchema(BaseModel):
    """LLM output for the recovery resolution node."""

    recovery_strategy: Literal["REPLAN_FULL", "ABORT"] = Field(
        description=(
            "REPLAN_FULL: provide a revised instruction to replan from scratch. "
            "ABORT: the task cannot be recovered; propagate the failure."
        )
    )
    revised_instruction: Optional[str] = Field(
        default=None,
        description=(
            "A rewritten natural-language instruction that avoids the failure. "
            "Required when recovery_strategy is REPLAN_FULL, null otherwise."
        ),
    )
    failure_diagnosis: str = Field(
        description="One or two sentences diagnosing why the action failed."
    )
    reasoning: str = Field(
        description="One or two sentences explaining the chosen recovery strategy."
    )
