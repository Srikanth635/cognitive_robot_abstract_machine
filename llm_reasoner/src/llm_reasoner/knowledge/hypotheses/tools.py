"""
The questions about a stored hypothesis offered to a language model as tools, each
defined in a JSON file beside this module.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path

from typing_extensions import Any, Mapping

from llm_reasoner.clients.tools import Tool, ToolDefinition, ToolOutcome
from llm_reasoner.knowledge.hypotheses.knowledge_access import KnowledgeToolName
from llm_reasoner.knowledge.hypotheses.questions import (
    KnowledgeAnswer,
    KnowledgeQuestionError,
    KnowledgeQuestions,
)

DEFINITIONS_DIRECTORY = Path(__file__).parent / "tool_definitions"
"""
Where the definitions of the knowledge tools are stored, one JSON file per tool.
"""

DEFINITION_SUFFIX = ".json"
"""
The suffix of a tool definition file.
"""

SHARED_DEFINITIONS_FILE = (
    DEFINITIONS_DIRECTORY / f"shared_definitions{DEFINITION_SUFFIX}"
)
"""
Where the definitions the answer shapes of several tools share are stored.
"""


class KnowledgeToolKey(StrEnum):
    """
    The keys of the inputs the knowledge tools take and of the answers they give that
    are not answers to the questions.
    """

    KIND = "kind"
    NAME = "name"
    MESSAGE = "message"


@dataclass
class KnowledgeTool(Tool, ABC):
    """
    One question about a stored hypothesis offered to a language model.
    """

    questions: KnowledgeQuestions
    """
    Answers the questions about the hypothesis.
    """

    @property
    @abstractmethod
    def name(self) -> KnowledgeToolName:
        """
        The name of the tool, which names its definition file.
        """

    @property
    def definition(self) -> ToolDefinition:
        return self.definition_named(self.name)

    @staticmethod
    @cache
    def definition_named(name: KnowledgeToolName) -> ToolDefinition:
        """
        :return: The definition stored for a tool, read once, its answer shape holding
            the definitions it shares with other tools.
        """
        return ToolDefinition.from_file(
            DEFINITIONS_DIRECTORY / f"{name}{DEFINITION_SUFFIX}"
        ).sharing(json.loads(SHARED_DEFINITIONS_FILE.read_text()))

    def run(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        """
        :return: The answer as JSON, or, as JSON too, what the hypothesis lacks when the
            question names something it does not have.

        .. note:: A question about something the hypothesis lacks is the model's
            mistake to correct, not a failure of the program, so it is the one error
            turned into an answer.
        """
        try:
            answer = self.answer(arguments)
        except KnowledgeQuestionError as error:
            return ToolOutcome.error(
                type(error).__name__,
                {**asdict(error), KnowledgeToolKey.MESSAGE: str(error)},
            )
        return ToolOutcome(content=json.dumps(answer.as_data()))

    @abstractmethod
    def answer(self, arguments: Mapping[str, Any]) -> KnowledgeAnswer:
        """
        :param arguments: The input of the call, checked against the definition.
        :return: The answer to the question.
        """


@dataclass
class SummariseKnowledge(KnowledgeTool):
    """
    Summarises the action the hypothesis describes.
    """

    @property
    def name(self) -> KnowledgeToolName:
        return KnowledgeToolName.KNOWLEDGE_OVERVIEW

    def answer(self, arguments: Mapping[str, Any]) -> KnowledgeAnswer:
        return self.questions.overview()


@dataclass
class FindFragments(KnowledgeTool):
    """
    Finds the hypothesis's fragments of a vocabulary class.
    """

    @property
    def name(self) -> KnowledgeToolName:
        return KnowledgeToolName.FIND_FRAGMENTS

    def answer(self, arguments: Mapping[str, Any]) -> KnowledgeAnswer:
        return self.questions.find(arguments[KnowledgeToolKey.KIND])


@dataclass
class DescribeFragment(KnowledgeTool):
    """
    Describes one fragment of the hypothesis.
    """

    @property
    def name(self) -> KnowledgeToolName:
        return KnowledgeToolName.DESCRIBE_FRAGMENT

    def answer(self, arguments: Mapping[str, Any]) -> KnowledgeAnswer:
        return self.questions.describe(arguments[KnowledgeToolKey.NAME])


KNOWLEDGE_TOOL_KINDS = (SummariseKnowledge, FindFragments, DescribeFragment)
"""
Every knowledge tool, in the order they are offered.
"""
