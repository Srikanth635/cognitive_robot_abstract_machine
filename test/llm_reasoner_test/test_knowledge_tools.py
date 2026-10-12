"""
Tests of the knowledge component as the resolver reaches it: the knowledge-access
contract, and the tools a language model reads the consulted hypothesis with.
"""

from __future__ import annotations

import inspect
import json

import pytest
from jsonschema import Draft202012Validator

from llm_reasoner.clients.tools import ToolBox, ToolErrorKey, tool_error_schema
from llm_reasoner.knowledge.access import (
    KNOWLEDGE_ACCESS_VERSION,
    HypothesisStore,
    KnowledgeAccess,
    KnowledgeService,
    KnowledgeToolName,
    SituationFeatures,
    core,
    pouring,
)
from llm_reasoner.knowledge.hypotheses.questions import (
    UnknownFragment,
    UnknownVocabularyClass,
)

from .test_generation import GenerationCase
from .test_hypotheses import (
    fragment_named,
    granular_query,
    pouring_query,
    service_answering,
)
from .world_mimics import WorldBody

# %% helpers


@pytest.fixture
def service(tmp_path) -> KnowledgeService:
    """
    A knowledge service whose model answers with the lift-and-pour structure and then
    with a granular pouring structure.
    """
    return service_answering(
        HypothesisStore(directory=tmp_path),
        GenerationCase.LIFT_AND_POUR,
        GenerationCase.GRANULAR_POURING,
    )


def asked(toolbox: ToolBox, name: KnowledgeToolName, arguments: dict) -> dict:
    """
    :return: The answer of a knowledge tool, read from JSON.
    """
    return json.loads(toolbox.call(name, arguments).content)


def names(view, fragments) -> list[str]:
    """
    :return: The names the fragments have in their stored hypothesis.
    """
    return [view.citation_of(fragment).fragment_name for fragment in fragments]


# %% the contract


class TestContract:
    def test_a_knowledge_service_fulfils_the_contract(self, service):
        assert isinstance(service, KnowledgeAccess)
        assert service.version == KNOWLEDGE_ACCESS_VERSION

    def test_every_knowledge_tool_is_offered_under_its_name(self, service):
        identifier = service.serve(pouring_query())
        assert {tool.definition.name for tool in service.tools(identifier)} == set(
            KnowledgeToolName
        )

    def test_every_tool_answers_in_the_shape_its_definition_declares(self, service):
        identifier = service.serve(pouring_query())
        tools = {tool.definition.name: tool for tool in service.tools(identifier)}
        calls = {
            KnowledgeToolName.KNOWLEDGE_OVERVIEW: {},
            KnowledgeToolName.FIND_FRAGMENTS: {"kind": core.ActionPhase.__name__},
            KnowledgeToolName.DESCRIBE_FRAGMENT: {"name": "lift"},
        }
        for name, arguments in calls.items():
            Draft202012Validator(tools[name].definition.output_schema).validate(
                json.loads(tools[name].run(arguments).content)
            )

    def test_every_error_answers_in_the_shape_errors_share(self, service):
        identifier = service.serve(pouring_query())
        toolbox = ToolBox(tools=service.tools(identifier))
        for name, arguments in {
            KnowledgeToolName.FIND_FRAGMENTS: {"kind": "NoSuchClass"},
            KnowledgeToolName.DESCRIBE_FRAGMENT: {"name": "no_such_fragment"},
        }.items():
            outcome = toolbox.call(name, arguments)
            assert outcome.is_error
            Draft202012Validator(tool_error_schema()).validate(
                json.loads(outcome.content)
            )

    def test_the_vocabulary_classes_are_the_knowledge_classes(self, service):
        assert service.is_vocabulary_class(core.ActionPhase)
        assert not service.is_vocabulary_class(WorldBody)

    def test_the_instances_of_a_class_come_from_the_knowledge_graph(self, service):
        service.serve(pouring_query())
        assert service.instances_of(core.ActionPhase) == service.graph.instances_of(
            core.ActionPhase
        )


# %% the overview


class TestOverview:
    def test_the_phases_are_given_in_the_order_the_action_goes_through_them(
        self, service
    ):
        identifier = service.serve(pouring_query())
        view = service.view(identifier)
        overview = asked(
            ToolBox(tools=service.tools(identifier)),
            KnowledgeToolName.KNOWLEDGE_OVERVIEW,
            {},
        )
        assert [phase["name"] for phase in overview["phases"]] == names(
            view, view.phases()
        )

    def test_the_overview_names_the_action_and_the_structure(self, service):
        identifier = service.serve(pouring_query())
        view = service.view(identifier)
        overview = asked(
            ToolBox(tools=service.tools(identifier)),
            KnowledgeToolName.KNOWLEDGE_OVERVIEW,
            {},
        )
        assert overview["action"] == view.root.decomposes_task.iri
        assert overview["structure"]["name"] == type(view.root).__name__

    def test_the_overview_names_the_roles(self, service):
        identifier = service.serve(pouring_query())
        view = service.view(identifier)
        overview = asked(
            ToolBox(tools=service.tools(identifier)),
            KnowledgeToolName.KNOWLEDGE_OVERVIEW,
            {},
        )
        assert [role["name"] for role in overview["roles"]] == names(
            view, view.root.defines_role
        )

    def test_the_overview_states_the_situation_the_knowledge_is_for(self, service):
        service.serve(pouring_query())
        identifier = service.serve(granular_query())
        overview = asked(
            ToolBox(tools=service.tools(identifier)),
            KnowledgeToolName.KNOWLEDGE_OVERVIEW,
            {},
        )
        (material,) = [
            feature
            for feature in SituationFeatures().of(pouring.PouringPhaseStructure)
            if pouring.MaterialRegime.GRANULAR in feature.values
        ]
        assert overview["situation"] == {
            material.field_name: pouring.MaterialRegime.GRANULAR.name
        }

    def test_a_phase_names_the_subgoal_it_aims_for(self, service):
        identifier = service.serve(pouring_query())
        view = service.view(identifier)
        overview = asked(
            ToolBox(tools=service.tools(identifier)),
            KnowledgeToolName.KNOWLEDGE_OVERVIEW,
            {},
        )
        aiming = {phase["name"]: phase.get("aims_for") for phase in overview["phases"]}
        assert aiming == {
            name: (
                view.citation_of(phase.aims_for).fragment_name
                if phase.aims_for is not None
                else None
            )
            for name, phase in zip(names(view, view.phases()), view.phases())
        }


# %% finding and describing fragments


class TestFragments:
    def test_the_fragments_of_a_class_are_found_by_its_name(self, service):
        identifier = service.serve(pouring_query())
        view = service.view(identifier)
        found = asked(
            ToolBox(tools=service.tools(identifier)),
            KnowledgeToolName.FIND_FRAGMENTS,
            {"kind": core.ActionPhase.__name__},
        )
        assert [fragment["name"] for fragment in found["fragments"]] == names(
            view, view.instances_of(core.ActionPhase)
        )

    def test_a_class_no_vocabulary_has_is_reported(self, service):
        identifier = service.serve(pouring_query())
        outcome = ToolBox(tools=service.tools(identifier)).call(
            KnowledgeToolName.FIND_FRAGMENTS, {"kind": "NoSuchClass"}
        )
        assert (
            json.loads(outcome.content)[ToolErrorKey.ERROR]
            == UnknownVocabularyClass.__name__
        )

    def test_a_description_names_the_fragments_a_field_links_to(self, service):
        identifier = service.serve(pouring_query())
        view = service.view(identifier)
        lift = fragment_named(view, "lift")
        described = asked(
            ToolBox(tools=service.tools(identifier)),
            KnowledgeToolName.DESCRIBE_FRAGMENT,
            {"name": "lift"},
        )
        (initiated_by,) = [
            statement
            for statement in described["statements"]
            if statement["field_name"] == "initiated_by"
        ]
        assert initiated_by["fragments"] == names(view, lift.initiated_by)

    def test_a_description_gives_the_meaning_of_the_fragments_class(self, service):
        identifier = service.serve(pouring_query())
        view = service.view(identifier)
        described = asked(
            ToolBox(tools=service.tools(identifier)),
            KnowledgeToolName.DESCRIBE_FRAGMENT,
            {"name": "lift"},
        )
        assert described["kind"]["meaning"] == inspect.getdoc(
            type(fragment_named(view, "lift"))
        )

    def test_a_description_names_the_fragments_linking_to_it(self, service):
        identifier = service.serve(pouring_query())
        described = asked(
            ToolBox(tools=service.tools(identifier)),
            KnowledgeToolName.DESCRIBE_FRAGMENT,
            {"name": "lift"},
        )
        assert {
            (link["name"], link["field_name"]) for link in described["linked_from"]
        } >= {("pour", "defines_phase")}

    def test_a_description_leaves_out_values_taken_from_sub_properties(self, service):
        identifier = service.serve(pouring_query())
        described = asked(
            ToolBox(tools=service.tools(identifier)),
            KnowledgeToolName.DESCRIBE_FRAGMENT,
            {"name": "pour"},
        )
        assert "defines_component" not in {
            statement["field_name"] for statement in described["statements"]
        }

    def test_a_fragment_the_hypothesis_lacks_is_reported(self, service):
        identifier = service.serve(pouring_query())
        outcome = ToolBox(tools=service.tools(identifier)).call(
            KnowledgeToolName.DESCRIBE_FRAGMENT, {"name": "no_such_fragment"}
        )
        assert (
            json.loads(outcome.content)[ToolErrorKey.ERROR] == UnknownFragment.__name__
        )
