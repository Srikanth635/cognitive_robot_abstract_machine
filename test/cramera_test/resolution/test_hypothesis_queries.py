"""
Tests of what the EQL panel can ask of the active hypothesis: a variable per kind of
fragment, the vocabulary its answers name, preset questions, and how its answers read.
"""

from __future__ import annotations

import json

import pytest
from llm_reasoner.knowledge.access import action_core

from cramera.knowledge.query_runner import EqlQueryRunner, RenderResult
from cramera.resolution import hypothesis_knowledge
from cramera.resolution.hypothesis_knowledge.ontology_backing import OntologyBacking
from cramera.resolution.hypothesis_queries import HypothesisQueries, HypothesisVariable

from ..test_server import get_json, posted_answer, server

# %% helpers


def answered(queries: HypothesisQueries, code: str) -> RenderResult:
    """
    :return: The answer to a query over what the queries offer.
    """
    return EqlQueryRunner(
        domains=queries.domains(),
        extra_names=queries.names(),
        renderings=(queries.answers(),),
    ).run_source(code)


def entities_of(result: RenderResult) -> list[str]:
    """
    :return: How each entity of the answer is written, in order.
    """
    return [row["__entity__"] for row in result.rows]


# %% the variables


class TestVariables:
    @pytest.mark.parametrize("variable", list(HypothesisVariable))
    def test_every_variable_ranges_over_its_kind_of_fragment(
        self, sample_hypothesis_knowledge, variable
    ):
        queries = HypothesisQueries.of_active()
        fragments = queries.fragments()[variable]
        result = answered(queries, f"an(entity({variable}))")
        assert sorted(entities_of(result)) == sorted(
            queries.answers().rendered(fragment) for fragment in fragments
        )

    def test_the_phases_are_answered_in_the_order_they_run(
        self, sample_hypothesis_knowledge
    ):
        queries = HypothesisQueries.of_active()
        result = answered(queries, f"an(entity({HypothesisVariable.PHASE}))")
        assert entities_of(result) == [
            queries.answers().rendered(phase)
            for phase in sample_hypothesis_knowledge.phase_structure.phases()
        ]

    def test_the_vocabulary_can_be_asked_about(self, sample_hypothesis_knowledge):
        queries = HypothesisQueries.of_active()
        requirement = HypothesisVariable.CONDITION_REQUIREMENT
        result = answered(
            queries,
            f"an(entity({requirement}).where({requirement}.has_condition_timing"
            " == ConditionTiming.THROUGHOUT))",
        )
        assert result.count == len(
            [
                fragment
                for fragment in queries.fragments()[requirement]
                if fragment.has_condition_timing
                == action_core.ConditionTiming.THROUGHOUT
            ]
        )
        assert result.count > 0


# %% the presets


class TestPresets:
    def test_every_preset_is_answered_by_the_active_hypothesis(
        self, sample_hypothesis_knowledge
    ):
        queries = HypothesisQueries.of_active()
        for preset in queries.presets():
            result = answered(queries, preset.code)
            assert result.ok and result.count > 0, preset.text

    def test_every_preset_says_what_it_asks(self, sample_hypothesis_knowledge):
        assert all(preset.text for preset in HypothesisQueries.of_active().presets())

    def test_the_presets_need_no_running_demo(self, sample_hypothesis_knowledge):
        assert not any(
            preset.requires_live for preset in HypothesisQueries.of_active().presets()
        )


# %% the answers


class TestAnswers:
    def test_a_fragment_is_answered_by_its_name_and_kind_alone(
        self, sample_hypothesis_knowledge
    ):
        result = answered(
            HypothesisQueries.of_active(), f"an(entity({HypothesisVariable.PHASE}))"
        )
        assert set(result.rows[0]) == {"__entity__", "__type__"}

    def test_an_answer_lights_up_its_fragments_in_the_hypothesis_graph(
        self, sample_hypothesis_knowledge
    ):
        result = answered(
            HypothesisQueries.of_active(), f"an(entity({HypothesisVariable.PHASE}))"
        )
        assert set(result.highlight) == {
            hypothesis_knowledge.node_identifier_for(phase)
            for phase in sample_hypothesis_knowledge.phase_structure.phases()
        }

    def test_a_term_of_the_vocabulary_is_written_with_its_prefix(self):
        timing = action_core.ConditionTiming.THROUGHOUT
        assert HypothesisQueries(knowledge=None).answers().rendered(
            timing
        ) == OntologyBacking().prefixed(timing.value)

    def test_a_list_of_fragments_is_written_as_their_names(
        self, sample_hypothesis_knowledge
    ):
        answers = HypothesisQueries.of_active().answers()
        phases = sample_hypothesis_knowledge.phase_structure.phases()
        assert answers.rendered(phases) == [answers.rendered(phase) for phase in phases]

    def test_a_list_of_values_it_does_not_know_is_not_its_own(self):
        assert not HypothesisQueries(knowledge=None).answers().renders([1, 2])


# %% before any knowledge is active


class TestNothingActive:
    def test_no_preset_is_offered(self):
        assert HypothesisQueries.of_active().presets() == []

    def test_a_variable_ranges_over_nothing(self):
        result = answered(
            HypothesisQueries.of_active(), f"an(entity({HypothesisVariable.PHASE}))"
        )
        assert result.count == 0


# %% through the viewer's routes


class TestRoutes:
    def test_the_presets_ask_about_the_hypothesis_consulted(
        self, server, sample_hypothesis_knowledge
    ):
        offered = [
            preset["code"] for preset in get_json(server + "/api/knowledge")["presets"]
        ]
        assert all(
            preset.code in offered for preset in HypothesisQueries.of_active().presets()
        )

    def test_a_query_about_the_hypothesis_is_answered(
        self, server, sample_hypothesis_knowledge
    ):
        answer = posted_answer(
            server + "/api/eql", {"code": f"an(entity({HypothesisVariable.PHASE}))"}
        )
        assert answer["count"] == len(
            sample_hypothesis_knowledge.phase_structure.phases()
        )

    def test_a_question_about_the_hypothesis_is_matched_to_its_preset(
        self, server, sample_hypothesis_knowledge
    ):
        preset, *_ = HypothesisQueries.of_active().presets()
        match = posted_answer(server + "/api/question", {"text": preset.text})
        assert match["preset"]["code"] == preset.code

    def test_the_names_of_the_hypothesis_are_offered_to_complete(
        self, server, sample_hypothesis_knowledge
    ):
        vocabulary = json.dumps(get_json(server + "/api/eql/vocabulary"))
        assert HypothesisVariable.PHASE in vocabulary
