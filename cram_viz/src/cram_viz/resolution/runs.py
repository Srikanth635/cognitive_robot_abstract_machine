"""
Resolution runs kept apart: every run starts from the same knowledge, a copy of the
viewer's store in a directory of the run's own, so what one run generates no other run
reads, and the run's resolution is kept beside its knowledge.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from llm_reasoner.knowledge.access import HypothesisStore
from typing_extensions import Any, Dict


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

    def keep(self, resolution: Dict[str, Any]) -> None:
        """
        Keep what the run resolved beside its knowledge.

        :param resolution: The resolution as JSON data.
        """
        (self.directory / RunFile.RESOLUTION).write_text(
            json.dumps(resolution, indent=2)
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

    def start(self, baseline: HypothesisStore) -> RunDirectory:
        """
        :param baseline: The store every run starts from, which no run writes.
        :return: A new run, its knowledge store a copy of the baseline.
        """
        run = RunDirectory(directory=self.directory / uuid4().hex)
        knowledge = run.knowledge_store.directory
        if not baseline.directory.is_dir():
            knowledge.mkdir(parents=True)
            return run
        shutil.copytree(baseline.directory, knowledge)
        return run
