"""
Checking the cases against a live model: only when asked to, since every check costs
money.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from llm_reasoner.clients.anthropic_client import AnthropicClient
from llm_reasoner.knowledge.access import (
    HermitCheck,
    HypothesisBackend,
    HypothesisStore,
    KnowledgeService,
    LinkedOntologyTerms,
    predictive_manipulation_suite,
)
from llm_reasoner.reasoner import Reasoner
from typing_extensions import Mapping

LIVE_MODEL = "claude-sonnet-5-5"
"""
The model the live checks are run with.
"""

RESOLVER_MAX_TOKENS = 2048
"""
The most tokens the resolver's model may answer one turn with.
"""

GENERATION_REPAIRS = 4
"""
How many times a rejected hypothesis is returned for repair, as cram_viz generates.
"""

OPT_IN = "1"
"""
The value of the opt-in variable that runs the live checks.
"""


class LiveCheckVariable(StrEnum):
    """
    The environment variables the live checks read.
    """

    OPT_IN = "LLM_REASONER_LIVE"
    """
    Set to run the live checks.
    """

    API_KEY = "ANTHROPIC_API_KEY"
    """
    The key the model is called with.
    """

    KNOWLEDGE_STORE = "LLM_REASONER_KNOWLEDGE_STORE"
    """
    A directory whose stored hypotheses later runs reuse, so each situation is generated
    once; a fresh directory per run when unset.
    """


@dataclass(frozen=True)
class LiveCheck:
    """
    Whether and how the cases are checked against a live model.
    """

    environment: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    """
    The environment the variables are read from.
    """

    def enabled(self) -> bool:
        """
        :return: Whether the live checks were asked for and can call the model.
        """
        return self.environment.get(LiveCheckVariable.OPT_IN) == OPT_IN and bool(
            self.environment.get(LiveCheckVariable.API_KEY)
        )

    def store_directory(self, fresh: Path) -> Path:
        """
        :param fresh: A fresh directory to use when none is set.
        :return: Where the knowledge component stores the hypotheses it generates.
        """
        return Path(self.environment.get(LiveCheckVariable.KNOWLEDGE_STORE, fresh))

    @staticmethod
    def reasoner(store: HypothesisStore) -> Reasoner:
        """
        :param store: Where hypotheses are stored and looked up.
        :return: The resolver and the knowledge component, both asking the live model.
        """
        return Reasoner(
            client=AnthropicClient(model=LIVE_MODEL, max_tokens=RESOLVER_MAX_TOKENS),
            knowledge=KnowledgeService(
                backend=HypothesisBackend(
                    client=AnthropicClient(model=LIVE_MODEL),
                    suite=predictive_manipulation_suite(),
                    max_repairs=GENERATION_REPAIRS,
                    consistency=HermitCheck.from_environment(),
                    linked_terms=LinkedOntologyTerms.from_environment(),
                    store=store,
                )
            ),
        )
