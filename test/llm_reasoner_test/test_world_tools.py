"""
Tests of the world injector's questions offered to a language model as tools: their
definitions, what running one returns, how a call that cannot be answered is reported,
and a scripted conversation in which a model asks several of them in turn.
"""

from __future__ import annotations

import json

import pytest

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.clients.tools import ScriptedToolRequest, ToolBox
from llm_reasoner.world.finding import Comparison
from llm_reasoner.world.kinds import KindListing
from llm_reasoner.world.classes import identifier_of
from llm_reasoner.world.rendering import IndividualRenderer
from llm_reasoner.world.tools import WorldToolName, world_tools
from llm_reasoner.world.view import WorldView

from .world_mimics import VesselAnnotation, WorldBody, body_with_geometry

# %% helpers


@pytest.fixture
def view() -> WorldView:
    return WorldView()


@pytest.fixture
def toolbox(view) -> ToolBox:
    return ToolBox(tools=world_tools(view))


def parsed(outcome) -> object:
    assert not outcome.is_error, outcome.content
    return json.loads(outcome.content)


# %% definitions


class TestDefinitions:
    def test_every_world_tool_has_a_definition_named_after_it(self, view):
        assert sorted(tool.definition.name for tool in world_tools(view)) == sorted(
            name.value for name in WorldToolName
        )

    def test_the_comparisons_a_search_offers_are_those_it_understands(self, view):
        (find,) = [
            tool
            for tool in world_tools(view)
            if tool.definition.name == WorldToolName.FIND_INDIVIDUALS
        ]
        conditions = find.definition.input_schema["properties"]["conditions"]
        offered = conditions["items"]["properties"]["comparison"]["enum"]
        assert offered == [comparison.value for comparison in Comparison]

    def test_the_listings_offered_are_those_the_view_understands(self, view):
        (list_kinds,) = [
            tool
            for tool in world_tools(view)
            if tool.definition.name == WorldToolName.LIST_KINDS
        ]
        offered = list_kinds.definition.input_schema["properties"]["listing"]["enum"]
        assert offered == [listing.value for listing in KindListing]


# %% running


class TestRunning:
    def test_kinds_are_listed_by_what_has_individuals_of_its_own(self, toolbox):
        annotation = VesselAnnotation(body=body_with_geometry())
        result = parsed(toolbox.call(WorldToolName.LIST_KINDS, {}))
        assert {kind["name"] for kind in result} == {
            type(annotation).__name__,
            type(annotation.body).__name__,
        }

    def test_a_search_returns_the_handles_found(self, view, toolbox):
        body = body_with_geometry()
        result = parsed(
            toolbox.call(
                WorldToolName.FIND_INDIVIDUALS, {"kind": identifier_of(WorldBody)}
            )
        )
        assert [found["handle"] for found in result["individuals"]] == [
            view.handles.handle_of(body)
        ]

    def test_a_description_is_written_for_reading(self, view, toolbox):
        annotation = VesselAnnotation(body=body_with_geometry())
        handle = view.handles.handle_of(annotation)
        result = parsed(
            toolbox.call(WorldToolName.DESCRIBE_INDIVIDUAL, {"handle": handle})
        )
        assert result["description"] == IndividualRenderer().render(
            view.describe(handle)
        )

    def test_a_predicate_is_evaluated_on_a_handle(self, view, toolbox):
        body = body_with_geometry()
        result = parsed(
            toolbox.call(
                WorldToolName.EVALUATE_PREDICATE,
                {
                    "predicate": "IsHeavierThan",
                    "arguments": {
                        "body": {"handle": view.handles.handle_of(body)},
                        "mass": {"value": body.inertial.mass / 2},
                    },
                },
            )
        )
        assert result["value"] is True


# %% calls that cannot be answered


class TestUnanswerableCalls:
    def test_a_question_about_something_the_world_lacks_is_reported(self, toolbox):
        outcome = toolbox.call(WorldToolName.DESCRIBE_KIND, {"kind": "NoSuchKind"})
        assert outcome.is_error
        assert "NoSuchKind" in outcome.content

    def test_a_call_breaking_its_schema_is_reported_without_running(self, toolbox):
        outcome = toolbox.call(WorldToolName.FOLLOW_PATH, {"handle": "WorldBody#1"})
        assert outcome.is_error
        assert "path" in outcome.content

    def test_a_call_to_a_tool_not_offered_is_reported(self, toolbox):
        outcome = toolbox.call("no_such_tool", {})
        assert outcome.is_error


# %% a conversation


def test_a_scripted_model_asks_the_world_in_turns(view):
    answer = "The jug weighs 0.4 kg."
    body = body_with_geometry()
    handle = view.handles.handle_of(body)
    client = ScriptedClient(
        tool_turns=[
            [ScriptedToolRequest(WorldToolName.HOLDERS_OF, {"handle": handle})],
            [
                ScriptedToolRequest(
                    WorldToolName.FOLLOW_PATH,
                    {"handle": handle, "path": "inertial.mass"},
                )
            ],
            answer,
        ]
    )
    conversation = client.complete_with_tools("system", "user", world_tools(view))
    assert conversation.answer == answer
    assert [call.name for call in conversation.calls] == [
        WorldToolName.HOLDERS_OF,
        WorldToolName.FOLLOW_PATH,
    ]
    assert not any(call.outcome.is_error for call in conversation.calls)
