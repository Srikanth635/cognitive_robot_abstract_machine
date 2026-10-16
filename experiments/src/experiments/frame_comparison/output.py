"""
Where a comparison keeps its records, one line per run, and its summary.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from krrood.adapters.json_serializer import from_json, to_json
from typing_extensions import List

from experiments.frame_comparison.records import RunRecord, summarise


class ComparisonVariable(StrEnum):
    """
    The environment variables a live comparison reads.
    """

    OUTPUT = "FRAME_COMPARISON_OUTPUT"
    """
    The directory the comparison keeps its records and summary in; the comparison runs
    only when it is set.
    """


POUR_RUNS_PER_ARM = 3
"""
How many times each arm resolves and tries cram_viz's pour, its runs varying.
"""

KITCHEN_RUNS_PER_ARM = 1
"""
How many times each arm resolves each kitchen case.
"""


class ComparisonFile(StrEnum):
    """
    The files of a comparison's directory.
    """

    RECORDS = "records.jsonl"
    """
    One record per run, in the order the runs ended.
    """

    SUMMARY = "summary.json"
    """
    One summary per measurement, case and arm.
    """


@dataclass(frozen=True)
class ComparisonOutput:
    """
    The directory a comparison keeps its records and summary in.
    """

    directory: Path
    """
    The directory, created when the first record is kept.
    """

    @property
    def records_path(self) -> Path:
        """
        Where the records are kept.
        """
        return self.directory / ComparisonFile.RECORDS

    @property
    def summary_path(self) -> Path:
        """
        Where the summary is written.
        """
        return self.directory / ComparisonFile.SUMMARY

    def add(self, record: RunRecord) -> None:
        """
        Keep the record of a run that ended.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.records_path.open("a") as records:
            records.write(json.dumps(to_json(record)) + "\n")

    def records(self) -> List[RunRecord]:
        """
        :return: Every record kept, in the order the runs ended.
        """
        if not self.records_path.is_file():
            return []
        return [
            from_json(json.loads(line))
            for line in self.records_path.read_text().splitlines()
            if line
        ]

    def write_summary(self) -> None:
        """
        Write the summary of every record kept.
        """
        self.summary_path.write_text(
            json.dumps(
                [asdict(summary) for summary in summarise(self.records())], indent=2
            )
        )
