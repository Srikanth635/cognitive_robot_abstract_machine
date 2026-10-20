"""
A store of admitted hypotheses on disk, one directory per hypothesis holding its record
as JSON and the hypothesis as Turtle.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from krrood.utils import resolve_class_from_full_name
from rdflib import Graph
from typing_extensions import Callable, Iterable, List, Mapping, Optional

from llm_reasoner.knowledge.ontologies.terms import DescriptionEntity
from llm_reasoner.knowledge.representation.hypothesis import GivenFragments
from llm_reasoner.knowledge.representation.rdf import GraphReader, ReadHypothesis
from llm_reasoner.knowledge.storage.citation import Citation
from llm_reasoner.knowledge.storage.record import HypothesisRecord

INCOMING_PREFIX = "."
"""
What a stored hypothesis being copied in is named with until it is complete; a hidden
name, so the records listed never include it.
"""


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

    def adopt_new_records(self, source: HypothesisStore) -> List[str]:
        """
        Copy into this store every hypothesis the source holds and this store lacks,
        each whole, so it is never seen half copied.

        :param source: The store to adopt from.
        :return: The record identifiers adopted, oldest first.
        """
        held = {record.record_id for record in self.records()}
        adopted = [
            record.record_id
            for record in source.records()
            if record.record_id not in held
        ]
        for record_id in adopted:
            self.copy_from(source, record_id)
        return adopted

    def copy_from(self, source: HypothesisStore, record_id: str) -> None:
        """
        Copy one stored hypothesis of the source here, under a temporary name first and
        renamed once complete.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        incoming = self.directory / (INCOMING_PREFIX + record_id)
        shutil.copytree(source.directory / record_id, incoming)
        incoming.rename(self.directory / record_id)

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

    def read(
        self,
        record_id: str,
        read_given: Optional[Callable[[str], Mapping[str, DescriptionEntity]]] = None,
    ) -> ReadHypothesis:
        """
        :param record_id: The record identifier of a stored hypothesis.
        :param read_given: Reads the fragments of a hypothesis this one builds on, by
            IRI; by default each is read afresh from the store.
        :return: The stored hypothesis as vocabulary instances, its references to the
            hypotheses it builds on resolved to their fragments.
        """
        stored = self.get(record_id)
        read_given = read_given or (lambda given: self.read(given).by_iri())
        known = {
            iri: fragment
            for given in stored.record.builds_on
            for iri, fragment in read_given(given).items()
        }
        return GraphReader().read(stored.graph, known)

    def root_citation(self, record_id: str) -> Citation:
        """
        :return: The citation of the fragment a stored hypothesis was generated for.
        """
        root_type = resolve_class_from_full_name(self.get(record_id).record.root_type)
        read = self.read(record_id)
        return Citation.from_iri(read.identifiers[id(read.root_of(root_type))])

    def given(self, record_ids: Iterable[str]) -> GivenFragments:
        """
        :param record_ids: The records of the stored hypotheses a new one builds on.
        :return: Their fragments, each named as its citation names it.
        """
        fragments: list[DescriptionEntity] = []
        names: dict[int, str] = {}
        iris: dict[int, str] = {}
        for record_id in record_ids:
            for iri, fragment in self.read(record_id).by_iri().items():
                fragments.append(fragment)
                names[id(fragment)] = Citation.from_iri(iri).fragment_name
                iris[id(fragment)] = iri
        return GivenFragments(fragments=fragments, names=names, iris=iris)

    def records(self) -> list[HypothesisRecord]:
        """
        :return: Every stored record, oldest first, leaving out any still being
            copied in.
        """
        if not self.directory.is_dir():
            return []
        records = [
            HypothesisRecord.from_json(json.loads(path.read_text()))
            for path in self.directory.glob(f"*/{StoredFile.RECORD}")
            if not path.parent.name.startswith(INCOMING_PREFIX)
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
