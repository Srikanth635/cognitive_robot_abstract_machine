"""
Tests of knowledge a viewer extension contributes to the queries of a recorded scene:
its variables and names beside the episode's own, and its answers written its own way.
"""

from __future__ import annotations

import pytest

from cramera.knowledge.eql_session import EqlSession

from .dataset.contributed_records import (
    LabelledRecord,
    LabelledRecordContribution,
    LabelledRecordRendering,
)

# %% fixtures


@pytest.fixture
def contribution() -> LabelledRecordContribution:
    """
    Two labelled records.
    """
    return LabelledRecordContribution(
        records=[LabelledRecord("cup", ["a note"]), LabelledRecord("bowl")]
    )


@pytest.fixture
def session(fixture_scene, contribution) -> EqlSession:
    """
    A session over the fixture scene with the records contributed.
    """
    return EqlSession.of_active_scene(contributions=(contribution,))


# %% querying what is contributed


class TestContributedVariables:
    def test_a_contributed_variable_is_queried(self, session, contribution):
        result = session.run("an(entity(labelled_record))")
        assert result.ok
        assert [row["__entity__"] for row in result.rows] == [
            record.label for record in contribution.records
        ]

    def test_the_episodes_own_variables_are_still_queried(self, session):
        result = session.run(
            "the(entity(scene_object).where(scene_object.name == 'milk'))"
        )
        assert result.ok and result.count == 1

    def test_a_contributed_name_can_be_used(self, session, contribution):
        result = session.run(
            "the(entity(labelled_record).where(labelled_record.label == FIRST_LABEL))"
        )
        assert [row["__entity__"] for row in result.rows] == [
            contribution.records[0].label
        ]

    def test_a_session_without_contributions_has_no_contributed_variable(
        self, fixture_scene
    ):
        with pytest.raises(NameError):
            EqlSession.of_active_scene().run("an(entity(labelled_record))")


# %% the answers


class TestContributedAnswers:
    def test_an_answer_about_a_contributed_entity_holds_only_how_it_is_written(
        self, session, contribution
    ):
        result = session.run("an(entity(labelled_record))")
        assert result.rows[0] == {
            "__entity__": LabelledRecordRendering().rendered(contribution.records[0]),
            "__type__": LabelledRecord.__name__,
        }

    def test_an_answer_lights_up_what_its_contribution_says(
        self, session, contribution
    ):
        result = session.run("an(entity(labelled_record))")
        assert result.highlight == sorted(
            LabelledRecordRendering.node_of(record) for record in contribution.records
        )

    def test_a_contributed_value_among_others_is_written_its_own_way(
        self, session, contribution
    ):
        result = session.run("set_of(labelled_record, labelled_record.notes)")
        cup = contribution.records[0]
        assert {row[LabelledRecord.__name__] for row in result.rows} == {
            record.label for record in contribution.records
        }
        assert LabelledRecordRendering.node_of(cup) in result.highlight
