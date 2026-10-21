"""
Credential-free evidence from language-model experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from repository_knowledge.model import SourceScope

# %% query provenance


class QueryOrigin(StrEnum):
    """
    Who selected a repository query.
    """

    PREFLIGHT = "preflight"
    MODEL = "model"


@dataclass(frozen=True)
class QueryObservation:
    """
    A repository query receipt without its source payload or provider credentials.
    """

    operation: str
    """
    The observed repository operation.
    """

    origin: QueryOrigin
    """
    Whether the experiment or model requested the query.
    """

    query: str
    """
    The selected search text.
    """
    symbol_identifier: str
    """
    The requested declaration identity.
    """
    seconds: float
    """
    Wall time waiting for repository evidence.
    """

    response_characters: int
    """
    The encoded response size before display.
    """

    is_error: bool
    """
    Whether the MCP server refused the query.
    """


# %% model trial evidence


@dataclass(frozen=True)
class ModelTrial:
    """
    A source-bound model response and reported usage, without a correctness verdict.
    """

    model: str
    """
    The provider-qualified model selected for this trial.
    """

    answer: str
    """
    The returned answer or proposed change, without applying edits.
    """

    stop_reason: str
    """
    The SDK's reason for ending the turn.
    """
    source_commit: str
    """
    The committed baseline checked around the trial.
    """
    snapshot_identifier: str
    """
    The exact source capture checked around the trial.
    """

    source_scope: SourceScope
    """
    Whether evidence includes selected working files.
    """
    seconds: float
    """
    End-to-end trial time, including source checks.
    """
    input_tokens: int | None
    """
    Reported input tokens, or unknown when the SDK supplies none.
    """
    output_tokens: int | None
    """
    Reported output tokens, or unknown when the SDK supplies none.
    """

    model_calls: int | None
    """
    Reported model requests, or unknown when usage is unavailable.
    """

    tool_calls: int
    """
    Model-requested tool attempts, excluding operator source checks.
    """

    estimated_cost: float | None
    """
    A positive SDK cost estimate, never a billing receipt.
    """

    observations: list[QueryObservation]
    """
    Initial source verification and repository calls sent by the model.
    """
