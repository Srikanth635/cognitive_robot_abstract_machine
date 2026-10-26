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
from graphlib import TopologicalSorter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, URIRef
from rdflib.term import Node
from typing_extensions import Any, Mapping, Sequence

from llm_reasoner.knowledge.acceptance.derived import RangesHold, RestrictionsHold
from llm_reasoner.knowledge.acceptance.restrictions import ClosedWorldMembership
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
from llm_reasoner.knowledge.ontologies.terms import (
    DescriptionEntity,
    IntersectionOf,
    PropertyRestriction,
    local_name,
    python_attribute_name,
)
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

    ROLE_LIST_SPLITS = "role_list_splits"
    """
    The properties listing roles in no order, each with the properties the roles are
    split into.
    """

    TARGETS = "targets"
    """
    The IRIs of the properties a list of roles is split into, the first naming the slot
    an individual is on or about.
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


@dataclass(frozen=True)
class RoleListSplit:
    """
    The roles an individual lists through one property in no order, split into the
    properties its class restricts to slots of given classes. The first target takes
    exactly one role; a later functional target takes one where one fits; any other
    takes every fitting role left. An individual whose roles cannot all be placed, or
    fit a functional target several times, keeps its list.
    """

    property: str
    """
    The IRI of the property listing the roles.
    """

    targets: tuple[str, ...]
    """
    The IRIs of the properties the roles are split into, the first naming the slot the
    individual is on or about.
    """

    membership: ClosedWorldMembership = field(
        default_factory=ClosedWorldMembership, compare=False
    )
    """
    Decides which roles a target's restrictions accept.
    """

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> RoleListSplit:
        """
        :return: The rule a migration table states.
        """
        return cls(
            property=data[MigrationKey.PROPERTY],
            targets=tuple(data[MigrationKey.TARGETS]),
        )

    def placements(self, graph: Graph) -> dict[Node, dict[str, list[Node]]]:
        """
        :return: Each individual listing roles whose roles can be split, with the roles
            each target takes.
        """
        read = GraphReader(vocabulary=self.membership.vocabulary).read(graph)
        by_iri = read.by_iri()
        placements: dict[Node, dict[str, list[Node]]] = {}
        for subject in sorted(
            set(graph.subjects(URIRef(self.property), None)), key=str
        ):
            listed = [
                str(role) for role in graph.objects(subject, URIRef(self.property))
            ]
            if str(subject) not in by_iri or not all(role in by_iri for role in listed):
                continue
            placed = self.placement(
                by_iri[str(subject)], [by_iri[role] for role in listed]
            )
            if placed is not None:
                placements[subject] = {
                    target: [URIRef(read.identifiers[id(role)]) for role in roles]
                    for target, roles in placed.items()
                }
        return placements

    def placement(
        self, individual: DescriptionEntity, roles: list[DescriptionEntity]
    ) -> dict[str, list[DescriptionEntity]] | None:
        """
        :return: The roles each target takes, or ``None`` when the roles cannot all be
            placed or a functional target fits several.
        """
        remaining = list(roles)
        placed: dict[str, list[DescriptionEntity]] = {}
        for position, target in enumerate(self.targets):
            fillers = [
                restriction.filler
                for owner in type(individual).__mro__
                if "term" in vars(owner)
                for restriction in owner.term.restrictions
                if isinstance(restriction, PropertyRestriction)
                and restriction.property_iri == target
                and restriction.filler is not None
            ]
            if not fillers:
                continue
            fitting = self.membership.members(IntersectionOf(tuple(fillers)), remaining)
            if self.membership.vocabulary.is_functional(individual, target):
                if len(fitting) > 1 or (position == 0 and not fitting):
                    return None
            placed[target] = fitting
            fitting_ids = {id(role) for role in fitting}
            remaining = [role for role in remaining if id(role) not in fitting_ids]
        return None if remaining else placed

    def touches(self, graph: Graph) -> bool:
        """
        :return: Whether an individual's list of roles can be split.
        """
        return bool(self.placements(graph))

    def graph(self, graph: Graph) -> Graph:
        """
        :return: The graph with every list of roles that can be split replaced by the
            targets that take them.
        """
        placements = self.placements(graph)
        migrated = Graph()
        for prefix, namespace in graph.namespaces():
            migrated.bind(prefix, namespace)
        for subject, predicate, value in graph:
            if subject not in placements or predicate != URIRef(self.property):
                migrated.add((subject, predicate, value))
        for subject, placed in placements.items():
            for target, roles in placed.items():
                for role in roles:
                    migrated.add((subject, URIRef(target), role))
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

    role_list_splits: tuple[RoleListSplit, ...] = ()
    """
    The lists of roles split into the slots they name.
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
            role_list_splits=tuple(
                RoleListSplit.from_json(rule)
                for rule in data.get(MigrationKey.ROLE_LIST_SPLITS, [])
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
        :return: Whether migrating changes the graph: it names a renamed IRI, or one
            of the migration's rules touches it.
        """
        renamed = self.renamed_nodes
        renamed_graph = self.renamed(graph)
        return any(node in renamed for statement in graph for node in statement) or any(
            rule.touches(renamed_graph)
            for rule in (
                *self.concept_values,
                *self.literal_datatypes,
                *self.part_of_fills,
                *self.role_list_splits,
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
        :return: The graph with every renamed IRI renamed, then rewritten by each of
            the migration's rules.
        """
        migrated = self.renamed(graph)
        for rule in (
            *self.concept_values,
            *self.literal_datatypes,
            *self.part_of_fills,
            *self.role_list_splits,
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

    def still_applies_to(self, versions: Mapping[str, str]) -> bool:
        """
        :param versions: The module versions a record names.
        :return: Whether the record has not been migrated through the migration: it
            names a version the migration moves on; a module the migration adds is not
            counted.
        """
        return any(
            change.old and versions.get(change.module) == change.old
            for change in self.module_versions
        )

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
        through those applied that it missed by being left stale, and check each record
        once all of them are applied, together with the migrated fragments of the
        records it builds on; a record building on a stale record is stale too.

        :return: Which records were migrated and which were left stale.
        """
        applied = self.applied
        pending = [
            migration for migration in self.migrations if migration.name not in applied
        ]
        report = MigrationReport()
        read: dict[str, dict[str, DescriptionEntity]] = {}
        for location in self.records_after_their_bases() if pending else []:
            self.migrate_record(applied, location, report, read)
        applied.extend(migration.name for migration in pending)
        (self.directory / MigrationFile.APPLIED).write_text(
            json.dumps(applied, indent=2)
        )
        return report

    def records_after_their_bases(self) -> list[Path]:
        """
        :return: The directory of every record of the store, each after the records it
            builds on.
        """
        locations = {
            location.name: location
            for location in sorted(self.directory.iterdir())
            if (location / StoredFile.RECORD).exists()
        }
        bases = {
            name: [
                base
                for base in json.loads((location / StoredFile.RECORD).read_text()).get(
                    RecordField.BUILDS_ON, []
                )
                if base in locations
            ]
            for name, location in locations.items()
        }
        return [locations[name] for name in TopologicalSorter(bases).static_order()]

    def migrations_for(
        self, versions: Mapping[str, str], applied: Sequence[str]
    ) -> list[Migration] | None:
        """
        :param versions: The module versions a record names.
        :param applied: The names of the migrations the store has been migrated through.
        :return: The migrations the record takes, in order: the applied ones it missed by
            being left stale, then those not yet applied; ``None`` when the record is
            behind the versions the applied migrations reached even after those it missed.
        """
        missed: list[Migration] = []
        reached: dict[str, str] = {}
        caught_up = dict(versions)
        for migration in self.migrations:
            if migration.name not in applied:
                continue
            reached.update(
                {change.module: change.new for change in migration.module_versions}
            )
            if migration.still_applies_to(caught_up):
                missed.append(migration)
                caught_up = migration.versions_of(caught_up)
        if any(
            module in reached and version != reached[module]
            for module, version in caught_up.items()
        ):
            return None
        return missed + [
            migration for migration in self.migrations if migration.name not in applied
        ]

    def migrate_record(
        self,
        applied: Sequence[str],
        location: Path,
        report: MigrationReport,
        read: dict[str, dict[str, DescriptionEntity]],
    ) -> None:
        """
        Migrate one record through the migrations it takes (:meth:`migrations_for`),
        writing it only when the migrated hypothesis still meets its ontology, and its
        Turtle only when a migration changes it; a record no migrations bring to the
        store's versions is stale.

        :param applied: The names of the migrations the store has been migrated through.
        :param read: The migrated fragments of every record migrated so far, by the IRI
            they are stored under, by record; the record's own are added.
        """
        record_file = location / StoredFile.RECORD
        hypothesis_file = location / StoredFile.HYPOTHESIS
        graph = Graph().parse(hypothesis_file, format="turtle")
        record = json.loads(record_file.read_text())
        bases = record.get(RecordField.BUILDS_ON, [])
        if any(base in report.stale for base in bases):
            report.stale.append(location.name)
            return
        migrations = self.migrations_for(record[RecordField.MODULE_VERSIONS], applied)
        if migrations is None:
            report.stale.append(location.name)
            return
        changed = False
        for migration in migrations:
            changed = changed or migration.touches(graph)
            graph = migration.graph(graph)
            record = migration.record(record)
            record[RecordField.MODULE_VERSIONS] = migration.versions_of(
                record[RecordField.MODULE_VERSIONS]
            )
        hypothesis = GraphReader().read(
            graph,
            known={
                iri: fragment
                for base in bases
                for iri, fragment in read.get(base, {}).items()
            },
        )
        read[location.name] = hypothesis.by_iri()
        errors = [
            violation
            for violation in self.acceptance.violations(hypothesis.fragments)
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
