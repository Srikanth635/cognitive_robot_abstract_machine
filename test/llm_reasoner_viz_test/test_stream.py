"""
Tests of following a log as it grows: each event sent once, in order, from wherever a
reader asks to start, with a sign of life while nothing happens.
"""

from __future__ import annotations

import json
from pathlib import Path

from typing_extensions import List

from llm_reasoner_viz.runs import EventLog
from llm_reasoner_viz.stream import EventStream

from .logs import answered, append, finished, started, write_run

# %% helpers


def counting(calls: int):
    """
    :return: A function that says to carry on that many times, then to stop.
    """
    remaining = [calls]

    def should_stop() -> bool:
        remaining[0] -= 1
        return remaining[0] < 0

    return should_stop


def stream_of(path: Path, offset: int = 0, **settings) -> EventStream:
    """
    :return: A stream over the log that does not wait between looks.
    """
    return EventStream(
        log=EventLog(path=path), offset=offset, pause=lambda seconds: None, **settings
    )


def data_of(frame: str) -> dict:
    """
    :return: The event a frame carries.
    """
    (line,) = [line for line in frame.splitlines() if line.startswith("data: ")]
    return json.loads(line.removeprefix("data: "))


def id_of(frame: str) -> int:
    """
    :return: The offset a frame says its event ends at.
    """
    (line,) = [line for line in frame.splitlines() if line.startswith("id: ")]
    return int(line.removeprefix("id: "))


# %% frames


class TestFrames:
    def test_each_event_of_the_log_is_a_frame_in_order(self, tmp_path: Path):
        events = [started(), answered(), finished()]
        path = write_run(tmp_path, "a-run", events)
        frames = list(stream_of(path).frames(counting(1)))
        assert [data_of(frame) for frame in frames] == events

    def test_a_frame_says_where_its_event_ends_in_the_log(self, tmp_path: Path):
        path = write_run(tmp_path, "a-run", [started(), finished()])
        frames = list(stream_of(path).frames(counting(1)))
        assert [id_of(frame) for frame in frames] == [
            entry.end for entry in EventLog(path=path).read_from(0).entries
        ]

    def test_a_frame_ends_with_a_blank_line(self, tmp_path: Path):
        path = write_run(tmp_path, "a-run", [started()])
        (frame,) = stream_of(path).frames(counting(1))
        assert frame.endswith("\n\n")

    def test_a_stream_goes_on_from_the_offset_it_is_given(self, tmp_path: Path):
        events = [started(), answered(), finished()]
        path = write_run(tmp_path, "a-run", events)
        first_end = EventLog(path=path).read_from(0).entries[0].end
        frames = list(stream_of(path, offset=first_end).frames(counting(1)))
        assert [data_of(frame) for frame in frames] == events[1:]

    def test_an_event_appended_while_streaming_is_sent_too(self, tmp_path: Path):
        path = write_run(tmp_path, "a-run", [started()])
        later = [answered(), finished()]
        stream = stream_of(path)
        sent: List[dict] = []
        looks = iter(range(3))

        def should_stop() -> bool:
            look = next(looks, None)
            if look == 1:
                append(path, later)
            return look is None

        for frame in stream.frames(should_stop):
            sent.append(data_of(frame))
        assert sent == [started(), *later]

    def test_an_event_is_sent_once(self, tmp_path: Path):
        path = write_run(tmp_path, "a-run", [started()])
        frames = list(stream_of(path).frames(counting(4)))
        assert len(frames) == 1

    def test_a_quiet_log_is_answered_with_a_sign_of_life(self, tmp_path: Path):
        path = write_run(tmp_path, "a-run", [started()])
        frames = list(stream_of(path, keepalive_seconds=0.0).frames(counting(3)))
        assert any(frame.startswith(": ") for frame in frames)
        assert sum(1 for frame in frames if frame.startswith("id: ")) == 1
