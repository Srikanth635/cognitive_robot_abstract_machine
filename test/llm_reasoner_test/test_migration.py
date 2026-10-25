"""
Tests of migrating stored knowledge and recorded answers when the ontology modules change:
terms renamed in Turtle, in records and in recorded answers, module versions moved for
records that still meet their ontology, and every migration applied once only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.compare import isomorphic

from llm_reasoner.knowledge.ontologies import PredictiveManipulationOntologyFile
from llm_reasoner.knowledge.ontologies.predictive_manipulation import core, pouring
from llm_reasoner.knowledge.storage.migration import (
    IriRename,
    Migration,
    ModuleVersionChange,
    RecordedFileMigration,
    StoreMigration,
    shipped_migrations,
)
from llm_reasoner.knowledge.storage.record import HypothesisRecord
from llm_reasoner.knowledge.storage.store import HypothesisStore, StoredFile

# %% fixtures

SAMPLE_STORE = (
    Path(__file__).parents[2]
    / "cram_viz"
    / "src"
    / "cram_viz"
    / "resources"
    / "sample_hypotheses"
)
"""
The store of the hand-authored pouring hypothesis cram_viz ships, which meets its
ontology.
"""

SAMPLE_RECORD = "pour_milk_sample"
"""
The record of that hypothesis.
"""

RENAMED_CLASS = IriRename(
    old=pouring.PouringPhaseStructure.term.iri,
    new=pouring.PouringPhaseStructure.term.iri.replace(
        "PouringPhaseStructure", "TransferPhaseStructure"
    ),
)
"""
A class renamed, which changes its IRI, its Python class name and the root type of
records rooted in it.
"""

RENAMED_PROPERTY = IriRename(
    old=PredictiveManipulationOntologyFile.CORE.namespace + "hasUnit",
    new=PredictiveManipulationOntologyFile.CORE.namespace + "measuredInUnit",
)
"""
A property renamed, which changes its IRI and its field name.
"""

RENAMED_MEMBER = IriRename(
    old=core.MeasurementUnit.DEGREE.value,
    new=core.MeasurementUnit.DEGREE.value.replace("Degree", "AngularDegree"),
)
"""
An enumeration member renamed, which changes its IRI and its member name.
"""


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """
    A copy of the sample store.
    """
    copied = tmp_path / "store"
    shutil.copytree(SAMPLE_STORE, copied)
    return copied


def sample_record(directory: Path) -> HypothesisRecord:
    """
    :return: The sample's record as stored in the directory.
    """
    return HypothesisRecord.from_json(
        json.loads((directory / SAMPLE_RECORD / StoredFile.RECORD).read_text())
    )


def pouring_version_moved(old: str) -> ModuleVersionChange:
    """
    :return: The pouring module moved from a version to the next one.
    """
    return ModuleVersionChange(
        module=PredictiveManipulationOntologyFile.POURING.value,
        old=old,
        new=old + ".1",
    )


# %% renaming terms


class TestRenamingTerms:
    def test_a_renamed_iri_is_renamed_in_every_position_of_a_graph(self):
        old, new = URIRef(RENAMED_PROPERTY.old), URIRef(RENAMED_PROPERTY.new)
        graph = Graph()
        graph.add((old, old, old))
        graph.add((URIRef("http://example.org/a"), old, Literal(1.0)))
        migrated = Migration(name="rename", renames=(RENAMED_PROPERTY,)).graph(graph)
        assert set(migrated) == {
            (new, new, new),
            (URIRef("http://example.org/a"), new, Literal(1.0)),
        }

    def test_a_recorded_answer_takes_the_new_class_field_and_member_names(self):
        answer = {
            "fragments": [
                {
                    "id": "range",
                    "type": "PouringPhaseStructure",
                    "has_unit": "DEGREE",
                    "label": "PouringPhaseStructure stays in prose",
                }
            ],
            "PouringPhaseStructure.has_unit": "DEGREE",
        }
        migration = Migration(
            name="rename",
            renames=(RENAMED_CLASS, RENAMED_PROPERTY, RENAMED_MEMBER),
        )
        assert migration.answer(answer) == {
            "fragments": [
                {
                    "id": "range",
                    "type": "TransferPhaseStructure",
                    "measured_in_unit": "ANGULAR_DEGREE",
                    "label": "PouringPhaseStructure stays in prose",
                }
            ],
            "TransferPhaseStructure.measured_in_unit": "ANGULAR_DEGREE",
        }

    def test_a_full_iri_in_an_answer_is_renamed(self):
        migration = Migration(name="rename", renames=(RENAMED_MEMBER,))
        assert migration.answer({"unit": RENAMED_MEMBER.old}) == {
            "unit": RENAMED_MEMBER.new
        }

    def test_a_records_root_type_follows_its_renamed_class(self):
        record = sample_record(SAMPLE_STORE).to_json()
        migrated = Migration(name="rename", renames=(RENAMED_CLASS,)).record(record)
        assert migrated["root_type"] == record["root_type"].replace(
            "PouringPhaseStructure", "TransferPhaseStructure"
        )

    def test_a_migration_without_changes_changes_nothing(self):
        migration = Migration(name="nothing")
        graph = Graph().parse(
            SAMPLE_STORE / SAMPLE_RECORD / StoredFile.HYPOTHESIS, format="turtle"
        )
        record = sample_record(SAMPLE_STORE).to_json()
        assert isomorphic(migration.graph(graph), graph)
        assert migration.record(record) == record


# %% migrating a store


class TestStoreMigration:
    def test_a_record_still_meeting_its_ontology_takes_the_new_versions(self, store):
        old = sample_record(store).module_versions["pouring.ttl"]
        report = StoreMigration(
            directory=store,
            migrations=[
                Migration(name="move", module_versions=(pouring_version_moved(old),))
            ],
        ).run()
        assert report.migrated == [SAMPLE_RECORD]
        assert sample_record(store).module_versions["pouring.ttl"] == old + ".1"

    def test_a_record_breaking_its_ontology_keeps_its_versions_and_is_stale(
        self, store
    ):
        old = sample_record(store).module_versions["pouring.ttl"]
        report = StoreMigration(
            directory=store,
            migrations=[
                Migration(
                    name="break",
                    renames=(RENAMED_PROPERTY,),
                    module_versions=(pouring_version_moved(old),),
                )
            ],
        ).run()
        assert report.stale == [SAMPLE_RECORD]
        assert sample_record(store).module_versions["pouring.ttl"] == old

    def test_each_migration_is_applied_once(self, store):
        old = sample_record(store).module_versions["pouring.ttl"]
        migration = Migration(
            name="move", module_versions=(pouring_version_moved(old),)
        )
        StoreMigration(directory=store, migrations=[migration]).run()
        report = StoreMigration(directory=store, migrations=[migration]).run()
        assert report.migrated == []
        assert sample_record(store).module_versions["pouring.ttl"] == old + ".1"

    def test_a_backup_holds_the_store_as_it_was(self, store):
        old = sample_record(store).module_versions["pouring.ttl"]
        backup = StoreMigration(
            directory=store,
            migrations=[
                Migration(name="move", module_versions=(pouring_version_moved(old),))
            ],
        ).back_up(label="test")
        assert sample_record(backup) == sample_record(SAMPLE_STORE)
        assert HypothesisStore(directory=backup).records()


# %% migrating recorded files


class TestRecordedFileMigration:
    def test_a_recorded_answer_and_a_turtle_fixture_are_rewritten(self, tmp_path):
        answer = tmp_path / "answer.json"
        answer.write_text(json.dumps({"type": "PouringPhaseStructure"}))
        fixture = tmp_path / "fixture.ttl"
        Graph().add(
            (URIRef("http://example.org/a"), URIRef(RENAMED_CLASS.old), Literal(1.0))
        ).serialize(fixture, format="turtle")
        untouched = tmp_path / "untouched.json"
        untouched.write_text(json.dumps({"type": "Elsewhere"}))
        changed = RecordedFileMigration(
            migration=Migration(name="rename", renames=(RENAMED_CLASS,))
        ).run([answer, fixture, untouched])
        assert changed == [answer, fixture]
        assert json.loads(answer.read_text()) == {"type": "TransferPhaseStructure"}
        assert (None, URIRef(RENAMED_CLASS.new), None) in Graph().parse(
            fixture, format="turtle"
        )


# %% the shipped migrations


class TestShippedMigrations:
    def test_shipped_migrations_are_read_in_the_order_of_their_names(self):
        names = [migration.name for migration in shipped_migrations()]
        assert names == sorted(names)
