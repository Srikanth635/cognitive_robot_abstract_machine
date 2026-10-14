"""
Tests of reading the logs of resolution runs: the events a log holds and where a reader
stands in it, what a run was and how it went, and which runs there are.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_reasoner_viz.runs import (
    EventLog,
    RunLibrary,
    RunSummary,
    Stage,
    UnknownRun,
)
from llm_reasoner_viz.vocabulary import KnownEvent

from .logs import (
    LOG_FILE,
    answered,
    append,
    failed,
    finished,
    lines_of,
    of_kind,
    started,
    write_run,
)

# %% the events of a log


class TestEventLog:
    def test_every_complete_line_is_an_event_in_order(self, tmp_path: Path):
        events = [started(), answered(), finished()]
        log = EventLog(path=write_run(tmp_path, "a-run", events))
        assert [entry.event for entry in log.read_from(0).entries] == events

    def test_reading_goes_on_from_where_it_stopped(self, tmp_path: Path):
        first, second, third = started(), answered(), finished()
        path = write_run(tmp_path, "a-run", [first, second])
        log = EventLog(path=path)
        stopped_at = log.read_from(0).next_offset
        append(path, [third])
        batch = log.read_from(stopped_at)
        assert [entry.event for entry in batch.entries] == [third]
        assert batch.next_offset == path.stat().st_size

    def test_an_event_says_where_it_ends(self, tmp_path: Path):
        first, second = started(), finished()
        log = EventLog(path=write_run(tmp_path, "a-run", [first, second]))
        entries = log.read_from(0).entries
        assert entries[0].end == len(lines_of([first]).encode())
        assert entries[1].end == len(lines_of([first, second]).encode())

    def test_nothing_new_leaves_the_reader_where_it_is(self, tmp_path: Path):
        path = write_run(tmp_path, "a-run", [started()])
        log = EventLog(path=path)
        size = path.stat().st_size
        batch = log.read_from(size)
        assert batch.entries == []
        assert batch.next_offset == size

    def test_a_line_still_being_written_is_left_for_later(self, tmp_path: Path):
        path = write_run(tmp_path, "a-run", [started()])
        complete = path.stat().st_size
        with path.open("a") as file:
            file.write('{"event": "ModelAsked", "at": 1.0')
        log = EventLog(path=path)
        batch = log.read_from(0)
        assert len(batch.entries) == 1
        assert batch.next_offset == complete
        with path.open("a") as file:
            file.write("}\n")
        assert len(log.read_from(complete).entries) == 1

    def test_a_line_that_is_not_json_is_an_event_saying_so(self, tmp_path: Path):
        path = write_run(tmp_path, "a-run", [started()])
        with path.open("a") as file:
            file.write("not json at all\n")
        entries = EventLog(path=path).read_from(0).entries
        assert entries[-1].event["event"] == KnownEvent.UNREADABLE
        assert entries[-1].event["line"] == "not json at all"

    def test_a_limit_stops_after_that_many_events(self, tmp_path: Path):
        events = [started(), answered(), answered(102.0), finished()]
        log = EventLog(path=write_run(tmp_path, "a-run", events))
        batch = log.read_from(0, limit=2)
        assert [entry.event for entry in batch.entries] == events[:2]
        assert batch.next_offset == batch.entries[-1].end

    def test_the_size_of_the_log_is_reported(self, tmp_path: Path):
        path = write_run(tmp_path, "a-run", [started(), finished()])
        assert EventLog(path=path).read_from(0).size == path.stat().st_size


# %% what a run was


class TestRunSummary:
    def summary(self, tmp_path: Path, events) -> RunSummary:
        write_run(tmp_path, "a-run", events)
        (summary,) = RunLibrary(directory=tmp_path).runs()
        return summary

    def test_a_run_is_summed_up_by_its_query_and_scene(self, tmp_path: Path):
        summary = self.summary(tmp_path, [started(), finished()])
        assert summary.run == "a-run"
        assert summary.query == started()["query"]
        assert summary.scene == started()["scene"]
        assert summary.started_at == started()["at"]

    def test_a_run_not_over_is_resolving(self, tmp_path: Path):
        summary = self.summary(tmp_path, [started(), answered()])
        assert summary.resolution is Stage.RUNNING
        assert summary.execution is None

    def test_a_run_that_finished_is_finished(self, tmp_path: Path):
        summary = self.summary(tmp_path, [started(), finished()])
        assert summary.resolution is Stage.FINISHED

    def test_a_run_that_failed_is_failed(self, tmp_path: Path):
        summary = self.summary(tmp_path, [started(), failed()])
        assert summary.resolution is Stage.FAILED

    def test_a_run_being_performed_in_the_simulation_says_so(self, tmp_path: Path):
        summary = self.summary(
            tmp_path,
            [
                started(),
                finished(),
                of_kind(KnownEvent.EXECUTION_STARTED, 110.0, run="a-run"),
            ],
        )
        assert summary.execution is Stage.RUNNING

    def test_a_run_performed_in_the_simulation_says_it_finished(self, tmp_path: Path):
        summary = self.summary(
            tmp_path,
            [
                started(),
                finished(),
                of_kind(KnownEvent.EXECUTION_STARTED, 110.0, run="a-run"),
                of_kind(
                    KnownEvent.EXECUTION_FINISHED, 122.0, run="a-run", seconds=12.0
                ),
            ],
        )
        assert summary.execution is Stage.FINISHED

    def test_a_run_whose_performing_failed_says_so(self, tmp_path: Path):
        summary = self.summary(
            tmp_path,
            [
                started(),
                finished(),
                of_kind(KnownEvent.EXECUTION_STARTED, 110.0, run="a-run"),
                of_kind(
                    KnownEvent.EXECUTION_FAILED,
                    111.0,
                    run="a-run",
                    error_type="RuntimeError",
                    reason="out of reach",
                    seconds=1.0,
                ),
            ],
        )
        assert summary.execution is Stage.FAILED

    def test_what_the_models_used_and_cost_is_summed(self, tmp_path: Path):
        summary = self.summary(
            tmp_path,
            [started(), answered(cost=0.01), answered(102.0, cost=0.02), finished()],
        )
        assert summary.model_calls == 2
        assert summary.cost == pytest.approx(0.03)
        assert summary.input_tokens == 2000
        assert summary.output_tokens == 400

    def test_the_tokens_read_from_and_written_to_the_cache_are_input_too(
        self, tmp_path: Path
    ):
        answer = answered()
        answer["usage"] = {
            "input_tokens": 6,
            "output_tokens": 389,
            "cache_creation_input_tokens": 13572,
            "cache_read_input_tokens": 7745,
        }
        summary = self.summary(tmp_path, [started(), answer])
        assert summary.input_tokens == 6 + 13572 + 7745
        assert summary.output_tokens == 389

    def test_the_neems_kept_are_counted_and_those_that_could_not_be_are_not(
        self, tmp_path: Path
    ):
        summary = self.summary(
            tmp_path,
            [
                started(),
                of_kind(KnownEvent.NEEM_RECORDED, 101.0, episode="t1"),
                of_kind(KnownEvent.NEEM_RECORDED, 102.0, episode="r"),
                of_kind(KnownEvent.NEEM_NOT_RECORDED, 103.0, episode="t2"),
                finished(),
            ],
        )
        assert summary.neems == 2

    def test_a_run_that_kept_no_neem_has_none(self, tmp_path: Path):
        assert self.summary(tmp_path, [started(), finished()]).neems == 0

    def test_an_answer_without_a_known_cost_adds_none(self, tmp_path: Path):
        summary = self.summary(tmp_path, [started(), answered(cost=None)])
        assert summary.cost == 0.0
        assert summary.model_calls == 1

    def test_the_events_are_counted_and_the_last_one_dated(self, tmp_path: Path):
        summary = self.summary(tmp_path, [started(), answered(), finished(at=107.0)])
        assert summary.events == 3
        assert summary.last_event_at == 107.0


# %% which runs there are


class TestRunLibrary:
    def test_runs_are_listed_newest_first(self, tmp_path: Path):
        write_run(tmp_path, "old", [started(at=100.0, run="old"), finished()])
        write_run(tmp_path, "new", [started(at=200.0, run="new"), finished()])
        assert [run.run for run in RunLibrary(directory=tmp_path).runs()] == [
            "new",
            "old",
        ]

    def test_a_directory_without_a_log_is_no_run(self, tmp_path: Path):
        (tmp_path / "empty").mkdir()
        write_run(tmp_path, "real", [started()])
        assert [run.run for run in RunLibrary(directory=tmp_path).runs()] == ["real"]

    def test_a_directory_that_does_not_exist_holds_no_runs(self, tmp_path: Path):
        assert RunLibrary(directory=tmp_path / "missing").runs() == []

    def test_a_run_that_grew_is_summed_up_again(self, tmp_path: Path):
        path = write_run(tmp_path, "a-run", [started()])
        library = RunLibrary(directory=tmp_path)
        assert library.runs()[0].events == 1
        append(path, [answered(), finished()])
        summary = library.runs()[0]
        assert summary.events == 3
        assert summary.resolution is Stage.FINISHED

    def test_the_log_of_a_run_is_found_by_its_name(self, tmp_path: Path):
        path = write_run(tmp_path, "a-run", [started()])
        assert RunLibrary(directory=tmp_path).log_of("a-run").path == path

    @pytest.mark.parametrize("name", ["", "..", "../elsewhere", "a/b", "missing"])
    def test_a_name_that_is_no_run_is_refused(self, tmp_path: Path, name: str):
        write_run(tmp_path, "a-run", [started()])
        with pytest.raises(UnknownRun):
            RunLibrary(directory=tmp_path).log_of(name)
