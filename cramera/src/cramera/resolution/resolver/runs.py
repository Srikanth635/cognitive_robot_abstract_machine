"""
Resolution runs kept apart: every run starts from the same knowledge, a copy of the
viewer's store in a directory of the run's own, so what one run generates no other run
reads; the run keeps the environment it started from and its resolution beside its
knowledge, so it can be executed later from that same environment.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from krrood.adapters.json_serializer import (
    SubclassJSONSerializer,
    from_json,
    to_json,
)
from llm_reasoner.knowledge.access import HypothesisStore
from llm_reasoner.observation.sinks import JsonLinesSink
from typing_extensions import Any, Dict, Self


class RunFile(StrEnum):
    """
    What a run's directory holds.
    """

    KNOWLEDGE = "hypotheses"
    """
    The run's copy of the knowledge store, with whatever the run generated.
    """

    RESOLUTION = "resolution.json"
    """
    What the run resolved.
    """

    ENVIRONMENT = "environment.json"
    """
    The environment the run started from.
    """

    EVENTS = "events.jsonl"
    """
    The log of what the run did, one JSON line per event, which every process working on
    the run appends to.
    """

    START_REQUEST = "start_request"
    """
    A marker that is written to ask a performance of the run, which waits for it, to
    begin.
    """


class EnvironmentField(StrEnum):
    """
    The fields of a run's environment as JSON.
    """

    SCENE = "scene"
    BUNDLE = "bundle"


@dataclass
class UnknownRun(LookupError):
    """
    No run of the given name is kept.
    """

    name: str
    """
    The name asked for.
    """

    def __post_init__(self):
        super().__init__(f"no run named '{self.name}' is kept")


@dataclass(frozen=True)
class RunEnvironment(SubclassJSONSerializer):
    """
    The environment a run starts from: a scene, built from its bundle.
    """

    scene: str
    """
    The scene's name.
    """

    bundle: Path
    """
    The scene's bundle directory.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            EnvironmentField.SCENE: self.scene,
            EnvironmentField.BUNDLE: str(self.bundle),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            scene=data[EnvironmentField.SCENE],
            bundle=Path(data[EnvironmentField.BUNDLE]),
        )


@dataclass(frozen=True)
class RunDirectory:
    """
    The directory of one resolution run.
    """

    directory: Path
    """
    The directory, named for the run.
    """

    @property
    def name(self) -> str:
        """
        :return: The run's name.
        """
        return self.directory.name

    @property
    def knowledge_store(self) -> HypothesisStore:
        """
        :return: The run's own knowledge store.
        """
        return HypothesisStore(directory=self.directory / RunFile.KNOWLEDGE)

    @property
    def events(self) -> JsonLinesSink:
        """
        :return: The sink appending to the run's log.
        """
        return JsonLinesSink(path=self.directory / RunFile.EVENTS)

    @property
    def start_requested(self) -> bool:
        """
        :return: Whether a start of the run's performance has been requested.
        """
        return (self.directory / RunFile.START_REQUEST).is_file()

    def request_start(self) -> None:
        """
        Ask the run's performance, if it waits for it, to begin.
        """
        (self.directory / RunFile.START_REQUEST).touch()

    def clear_start_request(self) -> None:
        """
        Forget a start requested earlier.
        """
        (self.directory / RunFile.START_REQUEST).unlink(missing_ok=True)

    def environment(self) -> RunEnvironment:
        """
        :return: The environment the run started from.
        """
        return RunEnvironment.from_json(
            json.loads((self.directory / RunFile.ENVIRONMENT).read_text())
        )

    def resolution(self) -> Any:
        """
        :return: What the run resolved.
        """
        return from_json(json.loads((self.directory / RunFile.RESOLUTION).read_text()))

    def keep(self, resolution: Any) -> None:
        """
        Keep what the run resolved beside its knowledge.

        :param resolution: The resolution, a dataclass krrood's serializer writes.
        """
        (self.directory / RunFile.RESOLUTION).write_text(
            json.dumps(to_json(resolution), indent=2)
        )


@dataclass(frozen=True)
class ResolutionRuns:
    """
    Starts resolution runs, each from the same knowledge store and in a directory of its
    own.
    """

    directory: Path
    """
    Where the runs are kept.
    """

    def start(
        self, baseline: HypothesisStore, environment: RunEnvironment
    ) -> RunDirectory:
        """
        :param baseline: The store every run starts from; a run never writes it while
            it runs, only adds what it admitted once it ends.
        :param environment: The environment the run starts from.
        :return: A new run, its knowledge store a copy of the baseline, keeping the
            environment it started from.
        """
        run = RunDirectory(directory=self.directory / uuid4().hex)
        self.copy_store(baseline, run.knowledge_store)
        (run.directory / RunFile.ENVIRONMENT).write_text(
            json.dumps(environment.to_json(), indent=2)
        )
        return run

    @staticmethod
    def copy_store(baseline: HypothesisStore, copy: HypothesisStore) -> None:
        """
        Make a copy of the baseline store, an empty store when there is none.
        """
        if not baseline.directory.is_dir():
            copy.directory.mkdir(parents=True)
            return
        shutil.copytree(baseline.directory, copy.directory)

    def named(self, name: str) -> RunDirectory:
        """
        :return: The run of the given name.
        :raises UnknownRun: When no run of that name is kept here.
        """
        if not name or Path(name).name != name:
            raise UnknownRun(name=name)
        run = RunDirectory(directory=self.directory / name)
        if not (run.directory / RunFile.RESOLUTION).is_file():
            raise UnknownRun(name=name)
        return run
