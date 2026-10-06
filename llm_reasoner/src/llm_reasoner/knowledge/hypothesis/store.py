"""
Where hypotheses live between generations.

WHAT MAKES CACHING POSSIBLE AT ALL
----------------------------------
A hypothesis is generic by construction: it says what pouring is like, not what
is about to happen. That is precisely why it can be reused across episodes, and
why the cache key carries no episodic component. Had generation been allowed to
see world state, every hypothesis would be about one situation and none would
be worth keeping.

STALE IS NOT THE SAME AS SUPERSEDED
-----------------------------------
A record becomes STALE when the modules it was generated under move: the
ontology changed, so the hypothesis describes a model no longer in force. It
becomes SUPERSEDED only when a replacement actually exists.

The distinction matters because the two call for different behaviour. A stale
record is not served, but it is also not marked -- nothing has replaced it yet,
and marking it would lose the ability to tell "we regenerated this" from "we
noticed this was out of date". Only when a fresh hypothesis is stored does the
old one get marked and linked.

WHY NOTHING IS EVER DELETED
---------------------------
A past resolution cited the hypothesis it leaned on. Deleting that record makes
the citation dangle and the resolution unexplainable, so superseded and refuted
records are retained with links rather than removed. Storage is cheap; an
unexplainable decision is not.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence

from llm_reasoner.knowledge.hypothesis.record import HypothesisRecord
from llm_reasoner.knowledge.hypothesis.serialization import decode, encode
from llm_reasoner.knowledge.hypothesis.status import HypothesisStatus
from llm_reasoner.knowledge.ontological.decorators import REGISTRY, OntologyRegistry


@dataclass(frozen=True)
class Lookup:
    """
    The outcome of asking the store for a hypothesis.

    Distinguishes the three ways a lookup can come back empty, because they call
    for different responses: nothing was ever generated, something was generated
    but the ontology has since moved, or something was generated and found
    wanting. Collapsing them into ``None`` would make a store that keeps
    refuting the same hypothesis look identical to an empty one.
    """

    record: Optional[HypothesisRecord] = None
    stale: tuple[HypothesisRecord, ...] = ()
    """
    Records for this action that were generated under superseded module versions.

    Not served, but worth knowing about: regenerating one is cheaper than generating
    from nothing if the prompt can be seeded with it.
    """

    refuted: tuple[HypothesisRecord, ...] = ()
    """
    Records that failed validation or were defeated in use.

    Evidence that regenerating may fail the same way.
    """

    @property
    def hit(self) -> bool:
        return self.record is not None


class HypothesisStore(ABC):
    """
    Somewhere hypotheses are kept and found again.
    """

    @abstractmethod
    def put(self, record: HypothesisRecord) -> HypothesisRecord:
        """
        Store a record, superseding any it replaces.
        """

    @abstractmethod
    def get(self, record_id: str) -> Optional[HypothesisRecord]:
        """
        Retrieve one record by its own identity, whatever its status.

        By identity rather than by key, because this is what a citation resolves against
        -- and a citation must keep working after the record it names has been
        superseded.
        """

    @abstractmethod
    def lookup(
        self,
        action_type: str,
        module_versions: Mapping[str, str],
        prompt_version: str | None = None,
    ) -> Lookup:
        """
        Find a servable hypothesis for an action under current versions.
        """

    @abstractmethod
    def records_for(self, action_type: str) -> tuple[HypothesisRecord, ...]:
        """
        Every record for an action, in insertion order, whatever its status.
        """

    @abstractmethod
    def __len__(self) -> int: ...

    # -- shared policy -----------------------------------------------------

    @staticmethod
    def _classify(
        candidates: Sequence[HypothesisRecord],
        module_versions: Mapping[str, str],
        prompt_version: str | None = None,
    ) -> Lookup:
        """
        Sort an action's records into servable, stale and refuted.

        Ordering matters: the freshest servable record wins, and the check for
        staleness comes before the check for servability, so a validated record
        generated under an old ontology is reported as stale rather than served.
        """
        servable: list[HypothesisRecord] = []
        stale: list[HypothesisRecord] = []
        refuted: list[HypothesisRecord] = []
        for record in candidates:
            if record.status is HypothesisStatus.SUPERSEDED:
                continue
            elif record.is_stale(module_versions, prompt_version):
                stale.append(record)
            elif record.status is HypothesisStatus.REFUTED:
                refuted.append(record)
            elif record.servable:
                servable.append(record)
        best = max(servable, key=lambda r: r.provenance.generated_at, default=None)
        return Lookup(record=best, stale=tuple(stale), refuted=tuple(refuted))


@dataclass
class InMemoryStore(HypothesisStore):
    """
    A store for one process.

    Loses everything on exit.
    """

    _by_id: dict[str, HypothesisRecord] = field(default_factory=dict, repr=False)
    _by_action: dict[str, list[str]] = field(default_factory=dict, repr=False)

    def put(self, record: HypothesisRecord) -> HypothesisRecord:
        action = record.action_type.iri
        for existing in self.records_for(action):
            if (
                existing.record_id != record.record_id
                and existing.status is HypothesisStatus.VALIDATED
                and existing.cache_key.action_type == record.cache_key.action_type
                and existing.cache_key != record.cache_key
            ):
                existing.supersede(record)
        self._by_id[record.record_id] = record
        ids = self._by_action.setdefault(action, [])
        if record.record_id not in ids:
            ids.append(record.record_id)
        return record

    def get(self, record_id: str) -> Optional[HypothesisRecord]:
        return self._by_id.get(record_id)

    def records_for(self, action_type: str) -> tuple[HypothesisRecord, ...]:
        return tuple(
            self._by_id[i]
            for i in self._by_action.get(action_type, ())
            if i in self._by_id
        )

    def lookup(
        self,
        action_type: str,
        module_versions: Mapping[str, str],
        prompt_version: str | None = None,
    ) -> Lookup:
        return self._classify(
            self.records_for(action_type), module_versions, prompt_version
        )

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[HypothesisRecord]:
        return iter(self._by_id.values())


@dataclass
class FileStore(HypothesisStore):
    """
    A store backed by one JSON file per record.

    One file per record rather than a single document, so that a record written
    under a model this process no longer has does not make the whole store
    unreadable: an undecodable file is reported and skipped, and everything else
    still loads.
    """

    directory: Path
    registry: OntologyRegistry = REGISTRY
    _index: dict[str, str] = field(default_factory=dict, repr=False)
    """
    Record id to action IRI, so an action's records can be found without decoding every
    file.
    """

    unreadable: list[str] = field(default_factory=list)
    """
    Files that could not be decoded, named rather than silently ignored.
    """

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._reindex()

    def _reindex(self) -> None:
        self._index.clear()
        self.unreadable.clear()
        for path in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                self._index[data["record_id"]] = data["action_type"]["iri"]
            except Exception:
                self.unreadable.append(path.name)

    def _path(self, record_id: str) -> Path:
        return self.directory / f"{record_id}.json"

    def put(self, record: HypothesisRecord) -> HypothesisRecord:
        action = record.action_type.iri
        for existing in self.records_for(action):
            if (
                existing.record_id != record.record_id
                and existing.status is HypothesisStatus.VALIDATED
                and existing.cache_key != record.cache_key
            ):
                existing.supersede(record)
                self._write(existing)
        self._write(record)
        self._index[record.record_id] = action
        return record

    def _write(self, record: HypothesisRecord) -> None:
        self._path(record.record_id).write_text(json.dumps(encode(record), indent=1))

    def get(self, record_id: str) -> Optional[HypothesisRecord]:
        path = self._path(record_id)
        if not path.exists():
            return None
        try:
            return decode(json.loads(path.read_text()), registry=self.registry)
        except Exception:
            if path.name not in self.unreadable:
                self.unreadable.append(path.name)
            return None

    def records_for(self, action_type: str) -> tuple[HypothesisRecord, ...]:
        out = []
        for record_id, action in self._index.items():
            if action != action_type:
                continue
            record = self.get(record_id)
            if record is not None:
                out.append(record)
        return tuple(out)

    def lookup(
        self,
        action_type: str,
        module_versions: Mapping[str, str],
        prompt_version: str | None = None,
    ) -> Lookup:
        return self._classify(
            self.records_for(action_type), module_versions, prompt_version
        )

    def __len__(self) -> int:
        return len(self._index)


def mark_stale(
    store: HypothesisStore,
    action_types: Iterable[str],
    module_versions: Mapping[str, str],
    prompt_version: str | None = None,
) -> tuple[HypothesisRecord, ...]:
    """
    Report which records the current module versions have left behind.

    Reports rather than mutates. A stale record has not been replaced, and marking it
    before a replacement exists would lose the difference between knowledge that was
    regenerated and knowledge that was merely noticed to be out of date -- which is
    exactly the question asked when a resolution made under an old ontology has to be
    explained.
    """
    out: list[HypothesisRecord] = []
    for action in action_types:
        out.extend(store.lookup(action, module_versions, prompt_version).stale)
    return tuple(out)


__all__ = [
    "FileStore",
    "HypothesisStore",
    "InMemoryStore",
    "Lookup",
    "mark_stale",
]
