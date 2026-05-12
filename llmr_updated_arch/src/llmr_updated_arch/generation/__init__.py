"""Semantic generation components and optional post-grounding reasoners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmr_updated_arch.integrations.krrood.match_reader import MatchSnapshot
    from llmr_updated_arch.schemas import SemanticBundle


class Reasoner(ABC):
    """Annotate a semantic bundle with optional post-grounding sidecars."""

    REASONER_NAME = "reasoner"
    PROMPT_VERSION = "v1"

    @abstractmethod
    def annotate(
        self,
        semantics: "SemanticBundle",
        match_data: "MatchSnapshot",
        world_context: str,
    ) -> None:
        """Inspect resolved match data and write findings to *semantics*."""


__all__ = ["Reasoner"]
