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

from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, URIRef
from rdflib.term import Node
from typing_extensions import Any, Mapping, Sequence

from llm_reasoner.knowledge.acceptance.derived import RangesHold, RestrictionsHold
from llm_reasoner.knowledge.acceptance.rules import AcceptanceSuite, Severity
from llm_reasoner.knowledge.acceptance.shapes import ShapesHold
from llm_reasoner.knowledge.ontologies import (
    HYPOTHESIS_CONCEPT_NAMESPACE,
    ONTOLOGIES_PACKAGE,
    ActionPerspectiveFile,
    PhysicalMeasurementFile,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.knowledge.ontologies.linked import LinkedClassHierarchy
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

    CONCEPT_VALUES = "concept_values"
    """
    The properties whose concepts become anonymous instances of their classes, each
    with the class it ranges over, if any.
    """

    LITERAL_DATATYPES = "literal_datatypes"
    """
    The properties whose literals take a datatype, each with the datatype.
    """

    PROPERTY = "property"
    """
    The IRI of a property.
    """

    RANGE = "range"
    """
    The IRI of the class a property ranges over.
    """

    DATATYPE = "datatype"
    """
    The IRI of an XML Schema datatype.
    """

    PART_OF_FILLS = "part_of_fills"
    """
    The parts of a participant linked to the participant they are part of.
    """

    PART = "part"
    """
    The IRI of the class of a part.
    """

    WHOLE = "whole"
    """
    The IRI of the class of the participant a part is part of.
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
    The version before the change; empty for a module records did not name before,
    which the change adds.
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


@dataclass(frozen=True)
class ConceptValueAsInstance:
    """
    The concepts of a property written as anonymous instances of their classes, where
    they are written as the IRI of the class; a concept of the hypothesis's own becomes
    a class under the property's range.
    """

    property: str
    """
    The IRI of the property.
    """

    range: str = ""
    """
    The IRI of the class the property ranges over, if it states one.
    """

    @classmethod
    def from_json(cls, data: Mapping[str, str]) -> ConceptValueAsInstance:
        """
        :return: The rule a migration table states.
        """
        return cls(
            property=data[MigrationKey.PROPERTY], range=data.get(MigrationKey.RANGE, "")
        )

    def touches(self, graph: Graph) -> bool:
        """
        :return: Whether the graph writes a concept of the property as an IRI, or lacks
            the range of a concept of the hypothesis's own.
        """
        return any(
            isinstance(value, URIRef)
            for value in graph.objects(None, URIRef(self.property))
        ) or any(
            (concept, RDFS.subClassOf, URIRef(self.range)) not in graph
            for concept in self.own_concepts(graph)
            if self.range
        )

    def own_concepts(self, graph: Graph) -> set[Node]:
        """
        :return: The classes of the hypothesis's own the property's anonymous values are
            instances of.
        """
        return {
            concept
            for value in graph.objects(None, URIRef(self.property))
            if isinstance(value, BNode)
            for concept in graph.objects(value, RDF.type)
            if str(concept).startswith(HYPOTHESIS_CONCEPT_NAMESPACE)
        }

    def graph(self, graph: Graph) -> Graph:
        """
        :return: The graph with the property's concepts written as anonymous instances
            of their classes.
        """
        migrated = Graph()
        for prefix, namespace in graph.namespaces():
            migrated.bind(prefix, namespace)
        for subject, predicate, value in graph:
            if predicate != URIRef(self.property) or not isinstance(value, URIRef):
                migrated.add((subject, predicate, value))
                continue
            instance = BNode()
            migrated.add((subject, predicate, instance))
            migrated.add((instance, RDF.type, value))
        for concept in self.own_concepts(migrated):
            migrated.add((concept, RDF.type, OWL.Class))
            if self.range:
                migrated.add((concept, RDFS.subClassOf, URIRef(self.range)))
        return migrated


@dataclass(frozen=True)
class LiteralDatatype:
    """
    The literals of a property given the datatype the property ranges over.
    """

    property: str
    """
    The IRI of the property.
    """

    datatype: str
    """
    The IRI of the datatype.
    """

    @classmethod
    def from_json(cls, data: Mapping[str, str]) -> LiteralDatatype:
        """
        :return: The rule a migration table states.
        """
        return cls(
            property=data[MigrationKey.PROPERTY], datatype=data[MigrationKey.DATATYPE]
        )

    def touches(self, graph: Graph) -> bool:
        """
        :return: Whether a literal of the property has another datatype.
        """
        return any(
            isinstance(value, Literal) and value.datatype != URIRef(self.datatype)
            for value in graph.objects(None, URIRef(self.property))
        )

    def graph(self, graph: Graph) -> Graph:
        """
        :return: The graph with every literal of the property given the datatype, an
            integral number written as an integer.
        """
        migrated = Graph()
        for prefix, namespace in graph.namespaces():
            migrated.bind(prefix, namespace)
        for subject, predicate, value in graph:
            if predicate == URIRef(self.property) and isinstance(value, Literal):
                held = value.toPython()
                if isinstance(held, float) and held.is_integer():
                    held = int(held)
                value = Literal(held, datatype=URIRef(self.datatype))
            migrated.add((subject, predicate, value))
        return migrated


@dataclass(frozen=True)
class PartOfFill:
    """
    The individuals of a part's class linked, where they name none, to the one
    individual of the class of the participant they are part of; nothing is linked where
    there is no such individual or several.
    """

    part: str
    """
    The IRI of the class of the part.
    """

    whole: str
    """
    The IRI of the class of the participant.
    """

    property: str
    """
    The IRI of the property linking a part to its participant.
    """

    @classmethod
    def from_json(cls, data: Mapping[str, str]) -> PartOfFill:
        """
        :return: The rule a migration table states.
        """
        return cls(
            part=data[MigrationKey.PART],
            whole=data[MigrationKey.WHOLE],
            property=data[MigrationKey.PROPERTY],
        )

    def unlinked(self, graph: Graph) -> tuple[list[Node], Node | None]:
        """
        :return: The individuals of the part's class naming no participant, and the
            one individual of the participant's class, ``None`` when there is none or
            several.
        """
        hierarchy = LinkedClassHierarchy.pinned()

        def of_class(general: str) -> list[Node]:
            return sorted(
                {
                    individual
                    for individual, named in graph.subject_objects(RDF.type)
                    if isinstance(named, URIRef)
                    and hierarchy.is_subclass(str(named), general)
                },
                key=str,
            )

        wholes = of_class(self.whole)
        parts = [
            part
            for part in of_class(self.part)
            if graph.value(part, URIRef(self.property)) is None
        ]
        return parts, wholes[0] if len(wholes) == 1 else None

    def touches(self, graph: Graph) -> bool:
        """
        :return: Whether a part naming no participant can be linked to one.
        """
        parts, whole = self.unlinked(graph)
        return bool(parts) and whole is not None

    def graph(self, graph: Graph) -> Graph:
        """
        :return: The graph with every part naming no participant linked to the one
            participant there is.
        """
        migrated = Graph()
        for prefix, namespace in graph.namespaces():
            migrated.bind(prefix, namespace)
        for statement in graph:
            migrated.add(statement)
        parts, whole = self.unlinked(graph)
        if whole is not None:
            for part in parts:
                migrated.add((part, URIRef(self.property), whole))
        return migrated


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

    concept_values: tuple[ConceptValueAsInstance, ...] = ()
    """
    The properties whose concepts become anonymous instances of their classes.
    """

    literal_datatypes: tuple[LiteralDatatype, ...] = ()
    """
    The properties whose literals take a datatype.
    """

    part_of_fills: tuple[PartOfFill, ...] = ()
    """
    The parts of participants linked to the participants they are part of.
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
            concept_values=tuple(
                ConceptValueAsInstance.from_json(rule)
                for rule in data.get(MigrationKey.CONCEPT_VALUES, [])
            ),
            literal_datatypes=tuple(
                LiteralDatatype.from_json(rule)
                for rule in data.get(MigrationKey.LITERAL_DATATYPES, [])
            ),
            part_of_fills=tuple(
                PartOfFill.from_json(rule)
                for rule in data.get(MigrationKey.PART_OF_FILLS, [])
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
        :return: Whether migrating changes the graph: it names a renamed IRI, or a rule
            of its concepts or literals touches it.
        """
        renamed = self.renamed_nodes
        renamed_graph = self.renamed(graph)
        return any(node in renamed for statement in graph for node in statement) or any(
            rule.touches(renamed_graph)
            for rule in (
                *self.concept_values,
                *self.literal_datatypes,
                *self.part_of_fills,
            )
        )

    def renamed(self, graph: Graph) -> Graph:
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

    def graph(self, graph: Graph) -> Graph:
        """
        :return: The graph with every renamed IRI renamed, then its concepts and
            literals rewritten by the migration's rules.
        """
        migrated = self.renamed(graph)
        for rule in (
            *self.concept_values,
            *self.literal_datatypes,
            *self.part_of_fills,
        ):
            migrated = rule.graph(migrated)
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
        :return: The module versions a record names, those this migration moved moved
            and those it adds added.
        """
        moved = dict(versions)
        for change in self.module_versions:
            if moved.get(change.module, "") == change.old:
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
        Migrate every record through the migrations not yet applied, in order, and
        check each record once all of them are applied.

        :return: Which records were migrated and which were left stale.
        """
        applied = self.applied
        pending = [
            migration for migration in self.migrations if migration.name not in applied
        ]
        report = MigrationReport()
        for location in sorted(self.directory.iterdir()):
            if pending and (location / StoredFile.RECORD).exists():
                self.migrate_record(pending, location, report)
        applied.extend(migration.name for migration in pending)
        (self.directory / MigrationFile.APPLIED).write_text(
            json.dumps(applied, indent=2)
        )
        return report

    def migrate_record(
        self, migrations: Sequence[Migration], location: Path, report: MigrationReport
    ) -> None:
        """
        Migrate one record through migrations, writing it only when the migrated
        hypothesis still meets its ontology, and its Turtle only when a migration
        changes it.
        """
        record_file = location / StoredFile.RECORD
        hypothesis_file = location / StoredFile.HYPOTHESIS
        graph = Graph().parse(hypothesis_file, format="turtle")
        record = json.loads(record_file.read_text())
        changed = False
        for migration in migrations:
            changed = changed or migration.touches(graph)
            graph = migration.graph(graph)
            record = migration.record(record)
            record[RecordField.MODULE_VERSIONS] = migration.versions_of(
                record[RecordField.MODULE_VERSIONS]
            )
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
        if changed:
            graph.serialize(hypothesis_file, format="turtle")
        record_file.write_text(json.dumps(record, indent=2))
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
