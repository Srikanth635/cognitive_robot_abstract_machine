"""
A store of admitted hypotheses on disk, one directory per hypothesis holding its record
as JSON and the hypothesis as Turtle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from rdflib import Graph
from typing_extensions import Optional

from llm_reasoner.knowledge.storage.record import HypothesisRecord


class StoredFile(StrEnum):
    """
    The files of one stored hypothesis.
    """

    RECORD = "record.json"
    """
    The record, in krrood's JSON form.
    """

    HYPOTHESIS = "hypothesis.ttl"
    """
    The hypothesis, in Turtle.
    """


@dataclass
class StoredHypothesis:
    """
    A stored hypothesis with its record.
    """

    record: HypothesisRecord
    """
    The provenance of the hypothesis.
    """

    graph: Graph
    """
    The hypothesis, in the ontology's terms.
    """


@dataclass
class HypothesisStore:
    """
    Admitted hypotheses on disk.
    """

    directory: Path
    """
    The directory holding one subdirectory per hypothesis.
    """

    def put(self, record: HypothesisRecord, graph: Graph) -> None:
        """
        Store a hypothesis with its record.
        """
        location = self.directory / record.record_id
        location.mkdir(parents=True, exist_ok=True)
        (location / StoredFile.RECORD).write_text(
            json.dumps(record.to_json(), indent=2)
        )
        graph.serialize(location / StoredFile.HYPOTHESIS, format="turtle")

    def get(self, record_id: str) -> StoredHypothesis:
        """
        :return: The stored hypothesis with the given record identifier.
        """
        location = self.directory / record_id
        return StoredHypothesis(
            record=HypothesisRecord.from_json(
                json.loads((location / StoredFile.RECORD).read_text())
            ),
            graph=Graph().parse(location / StoredFile.HYPOTHESIS, format="turtle"),
        )

    def records(self) -> list[HypothesisRecord]:
        """
        :return: Every stored record, oldest first.
        """
        if not self.directory.is_dir():
            return []
        records = [
            HypothesisRecord.from_json(json.loads(path.read_text()))
            for path in self.directory.glob(f"*/{StoredFile.RECORD}")
        ]
        return sorted(records, key=lambda record: record.created_at)

    def find(
        self, root_type: str, fixed_values: dict[str, str]
    ) -> Optional[StoredHypothesis]:
        """
        :param root_type: The qualified name of the vocabulary class requested.
        :param fixed_values: The values the request fixes, as written.
        :return: The newest stored hypothesis answering the request under the current
            ontology versions, whatever instruction it was generated for, or ``None``
            when there is none.
        """
        answering = [
            record
            for record in self.records()
            if record.answers(root_type, fixed_values)
        ]
        if not answering:
            return None
        return self.get(answering[-1].record_id)
