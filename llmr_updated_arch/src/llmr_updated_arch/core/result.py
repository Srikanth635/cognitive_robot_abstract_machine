"""Result objects returned by the action-resolution pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Any, Optional

from llmr_updated_arch.integrations.krrood.match_reader import MatchSnapshot
from llmr_updated_arch.schemas import SemanticBundle


@dataclass
class ActionResolutionResult:
    """Resolved executable action plus all sidecar state produced for it."""

    action: Any
    match_snapshot: MatchSnapshot
    semantic_bundle: SemanticBundle
    grounded_slots: dict[str, Any]
    world_context: str
    projection_result: Optional[Any] = None
