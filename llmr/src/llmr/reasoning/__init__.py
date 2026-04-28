"""LLM reasoning modules and the pluggable reasoner contract.

`slot_filler` handles action classification and slot filling.
`decomposer` splits compound instructions into atomic steps.
`llm_provider` builds provider-specific LangChain chat models.

Pluggable reasoners subclass :class:`Reasoner` and annotate the
:class:`~llmr.schemas.ActionAnnotationBundle` sidecar after slot filling.
Their outputs are optional: failures are logged and do not block grounding or
execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmr.bridge.match_reader import MatchSnapshot as MatchData
    from llmr.schemas import ActionAnnotationBundle as ActionSemantics


class Reasoner(ABC):
    """Annotate an :class:`ActionAnnotationBundle` with extra LLM-derived semantics.

    Implementations should be idempotent and limit side effects to writing into
    *semantics*, usually by populating a typed field or `semantics.extra`.
    Projectors can later normalize those raw sidecars into `HypothesisGraph`.
    Backend execution continues even if a reasoner fails.
    """

    REASONER_NAME = "reasoner"
    PROMPT_VERSION = "v1"

    @abstractmethod
    def annotate(
        self,
        semantics: "ActionSemantics",
        match_data: "MatchData",
        world_context: str,
    ) -> None:
        """Inspect *match_data* / *world_context* and write findings to *semantics*."""


__all__ = ["Reasoner"]
