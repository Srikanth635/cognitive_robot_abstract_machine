"""
Migrating stored knowledge and recorded answers when the ontology modules change: terms
renamed in Turtle, in records and in recorded answers, and the versions a record names
moved once the migrated record still meets its ontology.

The migrations are tables shipped beside this module, one JSON file each, applied in the
order of their names.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from rdflib import Graph, URIRef
from typing_extensions import Any, Mapping, Sequence

from llm_reasoner.knowledge.acceptance.derived import RangesHold, RestrictionsHold
from llm_reasoner.knowledge.acceptance.rules import AcceptanceSuite, Severity
from llm_reasoner.knowledge.acceptance.shapes import ShapesHold
from llm_reasoner.knowledge.ontologies import (
    ONTOLOGIES_PACKAGE,
    ActionPerspectiveFile,
    PhysicalMeasurementFile,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.knowledge.ontologies.terms import local_name, python_attribute_name
from llm_reasoner.knowledge.ontology_translation.reading import upper_snake_case
from llm_reasoner.knowledge.representation.rdf import GraphReader
from llm_reasoner.knowledge.storage.exceptions import UnknownTermFamily
from llm_reasoner.knowledge.storage.record import RecordField
from llm_reasoner.knowledge.storage.store import StoredFile

# %% the tables


class MigrationKey(StrEnum):
    """
    The keys of a migration table.
    """

    RENAMES = "renames"
    """
    The terms renamed, each as its old and its new IRI.
    """

    MODULE_VERSIONS = "module_versions"
    """
    The modules whose version moved, each with its old and its new version.
    """

    OLD = "old"
    """
    What a term or a version was.
    """

    NEW = "new"
    """
    What it is now.
    """

    MODULE = "module"
    """
    The file name of a module.
    """


class MigrationFile(StrEnum):
    """
    The files migrations keep.
    """

    APPLIED = "applied_migrations.json"
    """
    The names of the migrations a store has been migrated through, kept in the store.
    """

    TABLE_SUFFIX = ".json"
    """
    The suffix of a migration table.
    """


@dataclass(frozen=True)
class IriRename:
    """
    A term of the ontology renamed: a class, a property or an enumeration member.
    """

    old: str
    """
    The term's IRI before the change.
    """

    new: str
    """
    The term's IRI after the change.
    """

    @classmethod
    def from_json(cls, data: Mapping[str, str]) -> IriRename:
        """
        :return: The rename a migration table states.
        """
        return cls(old=data[MigrationKey.OLD], new=data[MigrationKey.NEW])

    def python_names(self) -> dict[str, str]:
        """
        :return: The old Python names the term may be written by in a recorded answer,
            each with its new one: as a class, as a field and as an enumeration member.
        """
        return {
            local_name(self.old): local_name(self.new),
            python_attribute_name(self.old): python_attribute_name(self.new),
            upper_snake_case(local_name(self.old)): upper_snake_case(
                local_name(self.new)
            ),
        }


@dataclass(frozen=True)
class ModuleVersionChange:
    """
    A module whose version moved.
    """

    module: str
    """
    The file name of the module, as records name it.
    """

    old: str
    """
    The version before the change.
    """

    new: str
    """
    The version after the change.
    """

    @classmethod
    def from_json(cls, data: Mapping[str, str]) -> ModuleVersionChange:
        """
        :return: The version change a migration table states.
        """
        return cls(
            module=data[MigrationKey.MODULE],
            old=data[MigrationKey.OLD],
            new=data[MigrationKey.NEW],
        )


def python_path_of(iri: str) -> str:
    """
    :return: The module path and name of the Python class generated for a class IRI of
        one of the families.
    :raises UnknownTermFamily: When no family's namespaces hold the IRI.
    """
    for family in (
        ActionPerspectiveFile,
        PredictiveManipulationOntologyFile,
        PhysicalMeasurementFile,
    ):
        first = next(iter(family))
        if iri.startswith(first.namespace_root):
            module, _, name = iri[len(first.namespace_root) :].partition("#")
            return ".".join(
                [
                    ONTOLOGIES_PACKAGE,
                    first.directory.name,
                    module.replace("-", "_"),
                    name,
                ]
            )
    raise UnknownTermFamily(iri=iri)


@dataclass(frozen=True)
class Migration:
    """
    One change of the ontology modules, as it applies to stored knowledge and to
    recorded answers.
    """

    name: str
    """
    The name of the migration, which orders it among the others.
    """

    renames: tuple[IriRename, ...] = ()
    """
    The terms renamed.
    """

    module_versions: tuple[ModuleVersionChange, ...] = ()
    """
    The modules whose version moved.
    """

    @classmethod
    def from_json(cls, name: str, data: Mapping[str, Any]) -> Migration:
        """
        :return: The migration a table states.
        """
        return cls(
            name=name,
            renames=tuple(
                IriRename.from_json(rename)
                for rename in data.get(MigrationKey.RENAMES, [])
            ),
            module_versions=tuple(
                ModuleVersionChange.from_json(change)
                for change in data.get(MigrationKey.MODULE_VERSIONS, [])
            ),
        )

    @property
    def renamed_nodes(self) -> dict[URIRef, URIRef]:
        """
        Each renamed IRI, with the IRI it is renamed to.
        """
        return {URIRef(rename.old): URIRef(rename.new) for rename in self.renames}

    def touches(self, graph: Graph) -> bool:
        """
        :return: Whether the graph names a renamed IRI.
        """
        renamed = self.renamed_nodes
        return any(node in renamed for statement in graph for node in statement)

    def graph(self, graph: Graph) -> Graph:
        """
        :return: The graph with every renamed IRI renamed, in every position.
        """
        renamed = self.renamed_nodes
        migrated = Graph()
        for prefix, namespace in graph.namespaces():
            migrated.bind(prefix, namespace)
        for statement in graph:
            migrated.add(tuple(renamed.get(node, node) for node in statement))
        return migrated

    def record(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """
        :param data: A record as stored.
        :return: The record with its root type and fixed values following the renamed
            terms; its module versions are moved by :meth:`versions_of`.
        """
        migrated = dict(data)
        for rename in self.renames:
            if migrated[RecordField.ROOT_TYPE] == python_path_of(rename.old):
                migrated[RecordField.ROOT_TYPE] = python_path_of(rename.new)
        migrated[RecordField.FIXED_VALUES] = self.answer(
            migrated[RecordField.FIXED_VALUES]
        )
        return migrated

    def versions_of(self, versions: Mapping[str, str]) -> dict[str, str]:
        """
        :return: The module versions a record names, those this migration moved moved.
        """
        moved = dict(versions)
        for change in self.module_versions:
            if moved.get(change.module) == change.old:
                moved[change.module] = change.new
        return moved

    def answer(self, value: Any) -> Any:
        """
        :param value: A recorded answer, or a part of one.
        :return: The answer with every renamed term renamed where it stands as a whole
            string, a full IRI or a dotted path of Python names; text is left as it is.
        """
        if isinstance(value, dict):
            return {self.answer(key): self.answer(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.answer(item) for item in value]
        if not isinstance(value, str):
            return value
        iris = {rename.old: rename.new for rename in self.renames}
        if value in iris:
            return iris[value]
        names = {
            old: new
            for rename in self.renames
            for old, new in rename.python_names().items()
        }
        return ".".join(names.get(part, part) for part in value.split("."))


def shipped_migrations() -> list[Migration]:
    """
    :return: The migrations shipped beside this module, in the order of their names.
    """
    directory = Path(__file__).parent / "migrations"
    return [
        Migration.from_json(table.stem, json.loads(table.read_text()))
        for table in sorted(directory.glob("*" + MigrationFile.TABLE_SUFFIX))
    ]


# %% migrating a store


@dataclass(frozen=True)
class MigrationReport:
    """
    What migrating a store did to its records.
    """

    migrated: list[str] = field(default_factory=list)
    """
    The records migrated and still meeting their ontology, which now name the moved
    versions.
    """

    stale: list[str] = field(default_factory=list)
    """
    The records that would break their ontology once migrated, left as they were.
    """


@dataclass
class StoreMigration:
    """
    Migrates a store of hypotheses through migrations it has not been migrated through.
    """

    directory: Path
    """
    The store's directory, one subdirectory per record.
    """

    migrations: Sequence[Migration] = field(default_factory=shipped_migrations)
    """
    The migrations, in the order they apply.
    """

    acceptance: AcceptanceSuite = field(
        default_factory=lambda: AcceptanceSuite(
            rules=[RestrictionsHold(), RangesHold(), ShapesHold()]
        )
    )
    """
    The rules a migrated record must still meet.
    """

    @property
    def applied(self) -> list[str]:
        """
        The names of the migrations the store has been migrated through.
        """
        ledger = self.directory / MigrationFile.APPLIED
        return json.loads(ledger.read_text()) if ledger.exists() else []

    def back_up(self, label: str) -> Path:
        """
        :param label: What tells the backup apart, such as the date.
        :return: A copy of the store beside it, named with the label.
        """
        backup = self.directory.with_name(f"{self.directory.name}.backup-{label}")
        shutil.copytree(self.directory, backup)
        return backup

    def run(self) -> MigrationReport:
        """
        Migrate every record through each migration not yet applied, in order.

        :return: Which records were migrated and which were left stale.
        """
        applied = self.applied
        report = MigrationReport()
        for migration in self.migrations:
            if migration.name in applied:
                continue
            for location in sorted(self.directory.iterdir()):
                if (location / StoredFile.RECORD).exists() and (
                    location.name not in report.stale
                ):
                    self.migrate_record(migration, location, report)
            applied.append(migration.name)
        (self.directory / MigrationFile.APPLIED).write_text(
            json.dumps(applied, indent=2)
        )
        return report

    def migrate_record(
        self, migration: Migration, location: Path, report: MigrationReport
    ) -> None:
        """
        Migrate one record, writing it only when the migrated hypothesis still meets its
        ontology, and its Turtle only when a renamed term occurs in it.
        """
        record_file = location / StoredFile.RECORD
        hypothesis_file = location / StoredFile.HYPOTHESIS
        stored = Graph().parse(hypothesis_file, format="turtle")
        graph = migration.graph(stored)
        errors = [
            violation
            for violation in self.acceptance.violations(
                GraphReader().read(graph).fragments
            )
            if violation.severity is Severity.ERROR
        ]
        if errors:
            report.stale.append(location.name)
            return
        record = migration.record(json.loads(record_file.read_text()))
        record[RecordField.MODULE_VERSIONS] = migration.versions_of(
            record[RecordField.MODULE_VERSIONS]
        )
        if migration.touches(stored):
            graph.serialize(hypothesis_file, format="turtle")
        record_file.write_text(json.dumps(record, indent=2))
        if location.name not in report.migrated:
            report.migrated.append(location.name)


# %% migrating recorded files


@dataclass(frozen=True)
class RecordedFileMigration:
    """
    Migrates files kept beside the code, such as recorded answers and Turtle fixtures,
    through one migration.
    """

    migration: Migration
    """
    The migration applied.
    """

    def run(self, paths: Sequence[Path]) -> list[Path]:
        """
        :param paths: JSON files, whose terms are renamed as in a recorded answer, and
            Turtle files, whose IRIs are renamed.
        :return: The files that changed.
        """
        changed = []
        for path in paths:
            if path.suffix == MigrationFile.TABLE_SUFFIX:
                before = json.loads(path.read_text())
                after = self.migration.answer(before)
                if after != before:
                    path.write_text(json.dumps(after, indent=2) + "\n")
                    changed.append(path)
                continue
            stored = Graph().parse(path, format="turtle")
            if self.migration.touches(stored):
                self.migration.graph(stored).serialize(path, format="turtle")
                changed.append(path)
        return changed
