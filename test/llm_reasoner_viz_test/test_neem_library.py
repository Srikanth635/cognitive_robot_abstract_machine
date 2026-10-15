"""
Tests of listing the NEEMs with where each came from: the plans of the database, each
joined to the run whose log says it was kept, and to how that went.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typing_extensions import Any, Dict, List

from llm_reasoner_viz.neem_library import NeemLibrary, NeemOrigin
from llm_reasoner_viz.neems import NeemDatabase, UnknownNeem
from llm_reasoner_viz.runs import RunLibrary
from llm_reasoner_viz.vocabulary import KnownEvent, NeemKind

from .logs import (
    append,
    neem_recorded,
    of_kind,
    started,
    trial_finished,
    write_run,
)
from .neem_files import write_neem_database

MISSED = {"lands_in_opening": False, "mouth_distance_from_opening_centre": 0.189}
LANDED = {"lands_in_opening": True, "mouth_distance_from_opening_centre": 0.005}

# %% helpers


def library_with(tmp_path: Path, runs: Dict[str, List[Dict[str, Any]]]) -> NeemLibrary:
    """
    :return: The NEEMs of a database of two plans, with the logs of the given runs.
    """
    database = write_neem_database(tmp_path / "neems.sqlite")
    for run, events in runs.items():
        write_run(tmp_path / "runs", run, events)
    return NeemLibrary(
        database=NeemDatabase(path=database),
        runs=RunLibrary(directory=tmp_path / "runs"),
    )


def database_of(tmp_path: Path) -> Path:
    return tmp_path / "neems.sqlite"


# %% the entries


class TestEntries:
    def test_every_plan_of_the_database_is_an_entry_newest_first(self, tmp_path):
        entries = library_with(tmp_path, {}).entries()
        assert [entry.plan.neem for entry in entries] == [2, 1]

    def test_a_plan_a_log_says_a_run_kept_has_that_run_for_its_origin(self, tmp_path):
        database = database_of(tmp_path)
        library = library_with(
            tmp_path,
            {
                "run-a": [
                    started(run="run-a"),
                    neem_recorded(110.0, NeemKind.TRIAL, "trial-1", database, 1),
                    trial_finished(111.0, "trial-1", MISSED),
                ]
            },
        )
        (first,) = [entry for entry in library.entries() if entry.plan.neem == 1]
        assert first.origin == NeemOrigin(
            kind=NeemKind.TRIAL,
            episode="trial-1",
            run="run-a",
            recorded_at=110.0,
            outcome=MISSED,
            result=None,
        )

    def test_the_outcome_of_a_trial_is_what_that_trial_ended_with_not_another(
        self, tmp_path
    ):
        database = database_of(tmp_path)
        library = library_with(
            tmp_path,
            {
                "run-a": [
                    neem_recorded(110.0, NeemKind.TRIAL, "trial-1", database, 1),
                    trial_finished(111.0, "trial-2", LANDED),
                    trial_finished(112.0, "trial-1", MISSED),
                ]
            },
        )
        (first,) = [entry for entry in library.entries() if entry.plan.neem == 1]
        assert first.origin.outcome == MISSED

    @pytest.mark.parametrize(
        ("ending", "result"),
        [
            (KnownEvent.EXECUTION_FINISHED, "finished"),
            (KnownEvent.EXECUTION_FAILED, "failed"),
        ],
    )
    def test_the_result_of_an_execution_is_how_that_run_ended(
        self, tmp_path, ending, result
    ):
        database = database_of(tmp_path)
        library = library_with(
            tmp_path,
            {
                "run-a": [
                    neem_recorded(120.0, NeemKind.EXECUTION, "run-a", database, 2),
                    of_kind(ending, 121.0, run="run-a"),
                ]
            },
        )
        (second,) = [entry for entry in library.entries() if entry.plan.neem == 2]
        assert second.origin.kind == NeemKind.EXECUTION
        assert second.origin.result == result
        assert second.origin.outcome is None

    def test_a_plan_no_log_names_has_no_origin(self, tmp_path):
        assert all(
            entry.origin is None for entry in library_with(tmp_path, {}).entries()
        )

    def test_an_event_naming_another_database_is_not_taken_for_a_plan_of_this_one(
        self, tmp_path
    ):
        library = library_with(
            tmp_path,
            {
                "run-a": [
                    neem_recorded(
                        110.0,
                        NeemKind.TRIAL,
                        "trial-1",
                        tmp_path / "elsewhere" / "neems.sqlite",
                        1,
                    )
                ]
            },
        )
        assert all(entry.origin is None for entry in library.entries())

    def test_the_entries_of_one_run_are_those_it_kept(self, tmp_path):
        database = database_of(tmp_path)
        library = library_with(
            tmp_path,
            {
                "run-a": [neem_recorded(110.0, NeemKind.TRIAL, "t-a", database, 1)],
                "run-b": [neem_recorded(120.0, NeemKind.TRIAL, "t-b", database, 2)],
            },
        )
        assert [entry.plan.neem for entry in library.entries(run="run-a")] == [1]
        assert [entry.plan.neem for entry in library.entries(run="run-b")] == [2]

    def test_a_log_that_grows_is_read_again(self, tmp_path):
        database = database_of(tmp_path)
        library = library_with(
            tmp_path,
            {"run-a": [neem_recorded(110.0, NeemKind.TRIAL, "t-1", database, 1)]},
        )
        assert [entry.plan.neem for entry in library.entries(run="run-a")] == [1]
        append(
            tmp_path / "runs" / "run-a" / "events.jsonl",
            [neem_recorded(130.0, NeemKind.EXECUTION, "run-a", database, 2)],
        )
        assert [entry.plan.neem for entry in library.entries(run="run-a")] == [2, 1]

    def test_an_entry_is_written_as_data_for_the_page(self, tmp_path):
        database = database_of(tmp_path)
        library = library_with(
            tmp_path,
            {"run-a": [neem_recorded(110.0, NeemKind.TRIAL, "t-1", database, 1)]},
        )
        (first,) = [entry for entry in library.entries() if entry.plan.neem == 1]
        assert first.to_json() == {
            **first.plan.to_json(),
            "origin": {
                "kind": "trial",
                "episode": "t-1",
                "run": "run-a",
                "recorded_at": 110.0,
                "outcome": None,
                "result": None,
            },
        }
        (second,) = [entry for entry in library.entries() if entry.plan.neem == 2]
        assert second.to_json()["origin"] is None


# %% one NEEM


class TestDetail:
    def test_a_neem_is_its_entry_with_the_steps_of_its_plan(self, tmp_path):
        database = database_of(tmp_path)
        library = library_with(
            tmp_path,
            {"run-a": [neem_recorded(110.0, NeemKind.TRIAL, "t-1", database, 1)]},
        )
        detail = library.detail(1)
        (entry,) = [entry for entry in library.entries() if entry.plan.neem == 1]
        assert detail.entry == entry
        assert detail.nodes == library.database.nodes(1)

    def test_a_neem_no_log_names_has_no_origin(self, tmp_path):
        assert library_with(tmp_path, {}).detail(2).entry.origin is None

    def test_a_neem_the_database_does_not_hold_is_unknown(self, tmp_path):
        with pytest.raises(UnknownNeem):
            library_with(tmp_path, {}).detail(99)

    def test_a_neem_is_written_as_data_for_the_page(self, tmp_path):
        library = library_with(tmp_path, {})
        detail = library.detail(1)
        assert detail.to_json() == {
            "neem": detail.entry.to_json(),
            "plan": [node.to_json() for node in detail.nodes],
        }
