"""Builder contracts for projecting reasoner output into sg_model objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from typing_extensions import Any, Optional

if TYPE_CHECKING:
    from llmr_updated_arch.integrations.krrood.match_reader import MatchSnapshot as MatchData
    from llmr_updated_arch.schemas import ActionAnnotationBundle as ActionSemantics
    from llmr_updated_arch.hypotheses.entities.base import Hypothesis


@dataclass(frozen=True)
class BuildInput:
    """Normalized input consumed by sg_model builders."""

    instruction: Optional[str]
    action: Any
    action_type: str
    semantics: "ActionSemantics"
    match_data: "MatchData"
    resolved_slots: dict[str, Any]
    world_context: str
    symbol_type: type
    llm_model_name: Optional[str] = None


@dataclass(frozen=True)
class BuildResult:
    """Root objects and warnings produced by one builder invocation."""

    roots: list["Hypothesis"]
    warnings: list[str] = field(default_factory=list)
