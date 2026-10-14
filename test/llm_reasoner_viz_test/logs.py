"""
Event logs for the tests, written the way the processes working on a run write them: one
JSON line per event, in a directory named for the run.
"""

from __future__ import annotations

import json
from pathlib import Path

from typing_extensions import Any, Dict, List

from llm_reasoner_viz.vocabulary import KnownEvent

LOG_FILE = "events.jsonl"
"""
The name of a run's log in its directory.
"""


def started(at: float = 100.0, run: str = "a-run") -> Dict[str, Any]:
    """
    :return: A run starting.
    """
    return {
        "event": KnownEvent.RUN_STARTED,
        "at": at,
        "run": run,
        "query": "the(PouringAction)(\n    source_container=variable(Tool),\n)\n",
        "scene": "simple_pouring",
    }


def answered(
    at: float = 101.0, cost: float | None = 0.01, call: str = "call"
) -> Dict[str, Any]:
    """
    :return: A model answering, with the tokens it used and what that cost.
    """
    return {
        "event": KnownEvent.MODEL_ANSWERED,
        "at": at,
        "call": call,
        "answer": "RIGHT",
        "seconds": 1.0,
        "usage": {"input_tokens": 1000, "output_tokens": 200},
        "cost": cost,
    }


def finished(at: float = 105.0, run: str = "a-run") -> Dict[str, Any]:
    """
    :return: A run finishing.
    """
    return {"event": KnownEvent.RUN_FINISHED, "at": at, "run": run, "seconds": 5.0}


def failed(at: float = 105.0, run: str = "a-run") -> Dict[str, Any]:
    """
    :return: A run failing.
    """
    return {
        "event": KnownEvent.RUN_FAILED,
        "at": at,
        "run": run,
        "error_type": "ScriptExhausted",
        "reason": "no answer left",
        "seconds": 5.0,
    }


def of_kind(kind: KnownEvent, at: float, **fields: Any) -> Dict[str, Any]:
    """
    :return: An event of a known kind with the given fields.
    """
    return {"event": kind, "at": at, **fields}


def lines_of(events: List[Dict[str, Any]]) -> str:
    """
    :return: The events as the lines of a log.
    """
    return "".join(json.dumps(event) + "\n" for event in events)


def write_run(directory: Path, run: str, events: List[Dict[str, Any]]) -> Path:
    """
    :return: The log of a new run in the directory, holding the events.
    """
    log = directory / run / LOG_FILE
    log.parent.mkdir(parents=True)
    log.write_text(lines_of(events))
    return log


def append(log: Path, events: List[Dict[str, Any]]) -> None:
    """
    Append events to a log, as a process working on the run does.
    """
    with log.open("a") as file:
        file.write(lines_of(events))
