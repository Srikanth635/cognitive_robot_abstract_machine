"""
The world injector's questions offered to a language model as tools, each defined in a
JSON file beside this module and answered by the world view.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cache
from pathlib import Path

from typing_extensions import Any, List, Mapping

from llm_reasoner.clients.tools import Tool, ToolDefinition, ToolOutcome
from llm_reasoner.world.encoding import AnswerEncoder
from llm_reasoner.world.exceptions import WorldQuestionError
from llm_reasoner.world.finding import Comparison, Condition
from llm_reasoner.world.predicates import Argument, IndividualArgument, PlainArgument
from llm_reasoner.world.rendering import IndividualRenderer
from llm_reasoner.world.view import FIND_LIMIT, WorldView

DEFINITIONS_DIRECTORY = Path(__file__).parent / "tool_definitions"
"""
Where the definitions of the world tools are stored, one JSON file per tool.
"""

DEFINITION_SUFFIX = ".json"
"""
The suffix of a tool definition file.
"""


class WorldToolName(StrEnum):
    """
    The world injector's tools, each named for the question it answers.
    """

    LIST_KINDS = "list_kinds"
    DESCRIBE_KIND = "describe_kind"
    FIND_INDIVIDUALS = "find_individuals"
    DESCRIBE_INDIVIDUAL = "describe_individual"
    HOLDERS_OF = "holders_of"
    FOLLOW_PATH = "follow_path"
    LIST_PREDICATES = "list_predicates"
    EVALUATE_PREDICATE = "evaluate_predicate"


class WorldToolKey(StrEnum):
    """
    The keys of the inputs the world tools take and of the answers they give that are
    not results of the world view.
    """

    KIND = "kind"
    WITHIN = "within"
    PRESENT_ONLY = "present_only"
    CONDITIONS = "conditions"
    PATH = "path"
    COMPARISON = "comparison"
    VALUE = "value"
    LIMIT = "limit"
    HANDLE = "handle"
    DEPTH = "depth"
    ABOUT = "about"
    PREDICATE = "predicate"
    ARGUMENTS = "arguments"
    DESCRIPTION = "description"


# %% tools


@dataclass
class WorldTool(Tool, ABC):
    """
    One question of the world view offered to a language model.
    """

    view: WorldView
    """
    The world view answering.
    """

    encoder: AnswerEncoder = field(default_factory=AnswerEncoder)
    """
    Writes the answer as JSON.
    """

    @property
    @abstractmethod
    def name(self) -> WorldToolName:
        """
        The name of the tool, which names its definition file.
        """

    @property
    def definition(self) -> ToolDefinition:
        return self.definition_named(self.name)

    @staticmethod
    @cache
    def definition_named(name: WorldToolName) -> ToolDefinition:
        """
        :return: The definition stored for a tool, read once.
        """
        return ToolDefinition.from_file(
            DEFINITIONS_DIRECTORY / f"{name}{DEFINITION_SUFFIX}"
        )

    def run(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        """
        :return: The answer as JSON, or what the world lacks when the question names
            something it does not have.

        .. note:: A question about something the world lacks is the model's mistake to
            correct, not a failure of the program, so it is the one error turned into
            an answer.
        """
        try:
            answer = self.answer(arguments)
        except WorldQuestionError as error:
            return ToolOutcome(content=str(error), is_error=True)
        return ToolOutcome(content=json.dumps(self.encoder.encode(answer)))

    @abstractmethod
    def answer(self, arguments: Mapping[str, Any]) -> Any:
        """
        :param arguments: The input of the call, checked against the definition.
        :return: The world view's answer.
        """


@dataclass
class ListKinds(WorldTool):
    """
    Lists the kinds the world has.
    """

    @property
    def name(self) -> WorldToolName:
        return WorldToolName.LIST_KINDS

    def answer(self, arguments: Mapping[str, Any]) -> Any:
        return self.view.kinds(
            within=arguments.get(WorldToolKey.WITHIN),
            present_only=arguments.get(WorldToolKey.PRESENT_ONLY, False),
        )


@dataclass
class DescribeKind(WorldTool):
    """
    Describes what a kind is and what its individuals hold.
    """

    @property
    def name(self) -> WorldToolName:
        return WorldToolName.DESCRIBE_KIND

    def answer(self, arguments: Mapping[str, Any]) -> Any:
        return self.view.describe_kind(arguments[WorldToolKey.KIND])


@dataclass
class FindIndividuals(WorldTool):
    """
    Finds the individuals of a kind meeting conditions on their fields.
    """

    @property
    def name(self) -> WorldToolName:
        return WorldToolName.FIND_INDIVIDUALS

    def answer(self, arguments: Mapping[str, Any]) -> Any:
        conditions = [
            Condition(
                path=condition[WorldToolKey.PATH],
                comparison=Comparison(condition[WorldToolKey.COMPARISON]),
                value=condition[WorldToolKey.VALUE],
            )
            for condition in arguments.get(WorldToolKey.CONDITIONS, [])
        ]
        return self.view.find(
            arguments[WorldToolKey.KIND],
            conditions,
            limit=arguments.get(WorldToolKey.LIMIT, FIND_LIMIT),
        )


@dataclass
class DescribeIndividual(WorldTool):
    """
    Describes an individual, written for reading.
    """

    renderer: IndividualRenderer = field(default_factory=IndividualRenderer)
    """
    Writes the description.
    """

    @property
    def name(self) -> WorldToolName:
        return WorldToolName.DESCRIBE_INDIVIDUAL

    def answer(self, arguments: Mapping[str, Any]) -> Any:
        handle = arguments[WorldToolKey.HANDLE]
        limits = (
            replace(self.view.limits, depth=arguments[WorldToolKey.DEPTH])
            if WorldToolKey.DEPTH in arguments
            else None
        )
        return {
            WorldToolKey.HANDLE: handle,
            WorldToolKey.DESCRIPTION: self.renderer.render(
                self.view.describe(handle, limits)
            ),
        }


@dataclass
class HoldersOf(WorldTool):
    """
    Names the individuals holding an individual.
    """

    @property
    def name(self) -> WorldToolName:
        return WorldToolName.HOLDERS_OF

    def answer(self, arguments: Mapping[str, Any]) -> Any:
        return self.view.holders(arguments[WorldToolKey.HANDLE])


@dataclass
class FollowPath(WorldTool):
    """
    Follows a path of fields from an individual.
    """

    @property
    def name(self) -> WorldToolName:
        return WorldToolName.FOLLOW_PATH

    def answer(self, arguments: Mapping[str, Any]) -> Any:
        return self.view.follow(
            arguments[WorldToolKey.HANDLE], arguments[WorldToolKey.PATH]
        )


@dataclass
class ListPredicates(WorldTool):
    """
    Lists the predicates and value functions the world model declares.
    """

    @property
    def name(self) -> WorldToolName:
        return WorldToolName.LIST_PREDICATES

    def answer(self, arguments: Mapping[str, Any]) -> Any:
        return self.view.predicates(about=arguments.get(WorldToolKey.ABOUT))


@dataclass
class EvaluatePredicate(WorldTool):
    """
    Evaluates a predicate or value function on given individuals and values.
    """

    @property
    def name(self) -> WorldToolName:
        return WorldToolName.EVALUATE_PREDICATE

    def answer(self, arguments: Mapping[str, Any]) -> Any:
        return self.view.evaluate(
            arguments[WorldToolKey.PREDICATE],
            {
                name: self.argument(given)
                for name, given in arguments[WorldToolKey.ARGUMENTS].items()
            },
        )

    @staticmethod
    def argument(given: Mapping[str, Any]) -> Argument:
        """
        :return: An argument as the world view takes it: an individual by handle, or a
            plain value.
        """
        if WorldToolKey.HANDLE in given:
            return IndividualArgument(handle=given[WorldToolKey.HANDLE])
        return PlainArgument(value=given[WorldToolKey.VALUE])


WORLD_TOOL_KINDS = (
    ListKinds,
    DescribeKind,
    FindIndividuals,
    DescribeIndividual,
    HoldersOf,
    FollowPath,
    ListPredicates,
    EvaluatePredicate,
)
"""
Every world tool, in the order a model is offered them.
"""


def world_tools(view: WorldView) -> List[WorldTool]:
    """
    :return: Every world tool, answering from the given world view.
    """
    return [kind(view=view) for kind in WORLD_TOOL_KINDS]
