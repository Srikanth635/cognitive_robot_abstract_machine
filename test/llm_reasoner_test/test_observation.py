"""
Tests of the events the system reports about its own work, and of the sinks they are
reported to: what an event says, how it is written as plain JSON, and how several
writers share one log.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from llm_reasoner.clients.usage import TokenUsage
from llm_reasoner.observation.events import (
    Component,
    CompletionStyle,
    EventKey,
    ModelAnswered,
    ModelAsked,
)
from llm_reasoner.observation.sinks import JsonLinesSink, MemorySink, NullSink

# %% helpers


def asked(call: str = "first") -> ModelAsked:
    """
    :return: An event of a model being asked.
    """
    return ModelAsked(
        call=call,
        source=Component.RESOLVER,
        style=CompletionStyle.PLAIN,
        identity="scripted",
        system="You resolve queries.",
        user="Choose an arm.",
    )


def answered(call: str = "first") -> ModelAnswered:
    """
    :return: An event of a model answering.
    """
    return ModelAnswered(
        call=call,
        answer="RIGHT",
        seconds=1.5,
        usage=TokenUsage(input_tokens=10, output_tokens=2),
        cost=0.01,
    )


# %% events


class TestEvents:
    def test_an_event_records_when_it_happened(self):
        before = time.time()
        event = asked()
        assert before <= event.at <= time.time()

    def test_an_event_is_written_as_plain_json_naming_its_kind(self):
        written = answered().to_json()
        assert written[EventKey.EVENT] == ModelAnswered.__name__
        assert json.loads(json.dumps(written)) == written

    def test_what_an_event_holds_is_written_by_field(self):
        event = answered()
        written = event.to_json()
        assert written["call"] == event.call
        assert written["answer"] == event.answer
        assert written["seconds"] == event.seconds

    def test_an_enumeration_is_written_as_its_value(self):
        written = asked().to_json()
        assert json.loads(json.dumps(written))["source"] == Component.RESOLVER.value

    def test_a_nested_record_is_written_as_an_object(self):
        event = answered()
        assert event.to_json()["usage"]["input_tokens"] == event.usage.input_tokens


# %% sinks


class TestSinks:
    def test_a_null_sink_takes_any_event_and_keeps_none(self):
        sink = NullSink()
        assert sink.emit(asked()) is None

    def test_a_memory_sink_keeps_the_events_in_order(self):
        sink = MemorySink()
        events = [asked(), answered()]
        for event in events:
            sink.emit(event)
        assert sink.events == events

    def test_a_json_lines_sink_writes_one_line_per_event_in_order(self, tmp_path: Path):
        path = tmp_path / "events.jsonl"
        sink = JsonLinesSink(path=path)
        events = [asked(), answered()]
        for event in events:
            sink.emit(event)
        assert [json.loads(line) for line in path.read_text().splitlines()] == [
            event.to_json() for event in events
        ]

    def test_several_sinks_append_to_one_log(self, tmp_path: Path):
        path = tmp_path / "events.jsonl"
        first, second = JsonLinesSink(path=path), JsonLinesSink(path=path)
        first.emit(asked("one"))
        second.emit(asked("two"))
        first.emit(answered("one"))
        calls = [json.loads(line)["call"] for line in path.read_text().splitlines()]
        assert calls == ["one", "two", "one"]

    def test_a_line_longer_than_a_buffer_stays_whole(self, tmp_path: Path):
        path = tmp_path / "events.jsonl"
        long_prompt = ModelAsked(
            call="long",
            source=Component.KNOWLEDGE,
            style=CompletionStyle.STRUCTURED,
            identity="scripted",
            system="x" * 200_000,
            user="y",
        )
        JsonLinesSink(path=path).emit(long_prompt)
        (line,) = path.read_text().splitlines()
        assert json.loads(line)["system"] == long_prompt.system

    def test_a_value_json_cannot_write_does_not_break_the_log(self, tmp_path: Path):
        path = tmp_path / "events.jsonl"
        event = asked()
        event.schema = {"where": tmp_path}
        JsonLinesSink(path=path).emit(event)
        (line,) = path.read_text().splitlines()
        assert json.loads(line)["schema"]["where"] == str(tmp_path)
