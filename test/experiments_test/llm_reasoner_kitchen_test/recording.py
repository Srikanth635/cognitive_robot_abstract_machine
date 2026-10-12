"""
A model's recorded answers to one case, replayed in place of the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.clients.tools import ScriptedToolRequest
from llm_reasoner.knowledge.access import (
    AcceptanceSuite,
    HypothesisBackend,
    HypothesisStore,
    KnowledgeService,
)
from llm_reasoner.reasoner import Reasoner
from typing_extensions import Any, List, Mapping

RECORDINGS = Path(__file__).parent / "recordings"
"""
Where the recordings are kept, one JSON file per case.
"""


class RecordingKey(StrEnum):
    """
    The keys of a recording.
    """

    TURNS = "turns"
    """
    The resolver's model's turns, in order.
    """

    ANSWER = "answer"
    """
    A turn's final answer.
    """

    TOOLS = "tools"
    """
    A turn's tool calls.
    """

    NAME = "name"
    """
    A tool call's tool.
    """

    ARGUMENTS = "arguments"
    """
    A tool call's input.
    """

    GENERATIONS = "generations"
    """
    The knowledge component's model's answers, in order.
    """


@dataclass(frozen=True)
class Recording:
    """
    What the models answered in one case.
    """

    turns: List[Mapping[str, Any]]
    """
    The resolver's model's turns: each a final answer or tool calls.
    """

    generations: List[Any] = field(default_factory=list)
    """
    The knowledge component's model's answers.
    """

    @classmethod
    def of(cls, identifier: str) -> Recording:
        """
        :return: The recording of the case with the given identifier.
        """
        stored = json.loads((RECORDINGS / f"{identifier}.json").read_text())
        return cls(
            turns=stored[RecordingKey.TURNS],
            generations=stored.get(RecordingKey.GENERATIONS, []),
        )

    def replayed_turns(self) -> List[Any]:
        """
        :return: The turns as a scripted client replays them.
        """
        return [
            (
                json.dumps(turn[RecordingKey.ANSWER])
                if RecordingKey.ANSWER in turn
                else [
                    ScriptedToolRequest(
                        call[RecordingKey.NAME], call[RecordingKey.ARGUMENTS]
                    )
                    for call in turn[RecordingKey.TOOLS]
                ]
            )
            for turn in self.turns
        ]

    def reasoner(self, store: HypothesisStore) -> Reasoner:
        """
        :param store: Where hypotheses are stored and looked up.
        :return: The resolver and the knowledge component, both replaying this
            recording; a generated hypothesis is admitted without acceptance rules.
        """
        return Reasoner(
            client=ScriptedClient(tool_turns=self.replayed_turns()),
            knowledge=KnowledgeService(
                backend=HypothesisBackend(
                    client=ScriptedClient(
                        responses=[json.dumps(answer) for answer in self.generations]
                    ),
                    suite=AcceptanceSuite(rules=[]),
                    store=store,
                )
            ),
        )
