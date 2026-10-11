"""
Tests of the contract between the world injector and the components using it: the
resolver reaches the world only through the access module, a world view fulfils that
contract, and every tool answers in the shape its definition declares.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import llm_reasoner.resolver
from llm_reasoner.clients.tools import ToolBox, ToolOutcome, tool_error_schema
from llm_reasoner.world import access
from llm_reasoner.world.access import WORLD_ACCESS_VERSION, WorldAccess, WorldToolName
from llm_reasoner.world.classes import identifier_of
from llm_reasoner.world.view import WorldView

from .world_mimics import IsHeavierThan, VesselAnnotation, WorldBody, body_with_geometry

RESOLVER_DIRECTORY = Path(llm_reasoner.resolver.__file__).parent
"""
Where the resolver's modules are.
"""

WORLD_PACKAGE = "llm_reasoner.world"
"""
The package of the world injector.
"""

# %% helpers


def world_modules_imported_by(module: Path) -> list:
    """
    :return: The world injector's modules a module imports.
    """
    tree = ast.parse(module.read_text())
    return [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith(WORLD_PACKAGE)
    ]


def calls_of_every_tool(view: WorldView, annotation: VesselAnnotation) -> dict:
    """
    :return: One call for each world tool, asking about a world holding the annotation
        and its body.
    """
    handle = view.handle_of(annotation.body)
    return {
        WorldToolName.LIST_KINDS: {},
        WorldToolName.DESCRIBE_KIND: {"kind": identifier_of(WorldBody)},
        WorldToolName.FIND_INDIVIDUALS: {
            "kind": identifier_of(VesselAnnotation),
            "path": "body",
        },
        WorldToolName.DESCRIBE_INDIVIDUAL: {"handle": view.handle_of(annotation)},
        WorldToolName.HOLDERS_OF: {"handle": handle},
        WorldToolName.FOLLOW_PATH: {"handle": handle, "path": "inertial.mass"},
        WorldToolName.LIST_PREDICATES: {},
        WorldToolName.EVALUATE_PREDICATE: {
            "predicate": IsHeavierThan.__name__,
            "arguments": {"body": {"handle": handle}, "mass": {"value": 0.1}},
        },
    }


# %% the contract


class TestContract:
    def test_the_resolver_reaches_the_world_only_through_the_access_module(self):
        imported = {
            module
            for path in RESOLVER_DIRECTORY.glob("*.py")
            for module in world_modules_imported_by(path)
        }
        assert imported == {access.__name__}

    def test_a_world_view_fulfils_the_contract(self):
        view = WorldView()
        assert isinstance(view, WorldAccess)
        assert view.version == WORLD_ACCESS_VERSION

    def test_a_handle_given_out_leads_back_to_its_individual(self):
        view = WorldView()
        body = body_with_geometry()
        handle = view.handle_of(body)
        assert view.is_live(handle)
        assert view.individual_of(handle) is body


# %% the shapes of the answers


class TestAnswerShapes:
    def test_every_tool_answers_in_the_shape_its_definition_declares(self):
        view = WorldView()
        annotation = VesselAnnotation(body=body_with_geometry())
        tools = {tool.definition.name: tool for tool in view.tools()}
        calls = calls_of_every_tool(view, annotation)
        assert set(calls) == set(tools)
        for name, arguments in calls.items():
            outcome = ToolBox(tools=list(tools.values())).call(name, arguments)
            assert not outcome.is_error, outcome.content
            Draft202012Validator(tools[name].definition.output_schema).validate(
                json.loads(outcome.content)
            )

    def test_every_error_answers_in_the_shape_errors_share(self):
        view = WorldView()
        toolbox = ToolBox(tools=view.tools())
        outcomes = [
            toolbox.call(WorldToolName.DESCRIBE_KIND, {"kind": "NoSuchKind"}),
            toolbox.call(WorldToolName.FOLLOW_PATH, {"handle": "WorldBody#1"}),
            toolbox.call("no_such_tool", {}),
        ]
        assert all(outcome.is_error for outcome in outcomes)
        for outcome in outcomes:
            Draft202012Validator(tool_error_schema()).validate(
                json.loads(outcome.content)
            )
