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
from rdflib import OWL, RDF, RDFS, XSD, BNode, Graph, Literal, Namespace, URIRef
from rdflib.compare import isomorphic

from llm_reasoner.knowledge.ontologies import (
    CRAM_AICOR_NAMESPACE,
    ActionPerspectiveFile,
    HYPOTHESIS_CONCEPT_NAMESPACE,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
from llm_reasoner.knowledge.ontologies.physical_measurements import flow_onset
from llm_reasoner.knowledge.ontologies.predictive_manipulation import (
    core,
    pouring,
    prehension,
)
from llm_reasoner.knowledge.ontologies.terms import ConceptReference, PropertyTerm
from llm_reasoner.knowledge.representation.rdf import GraphReader
from llm_reasoner.knowledge.storage.migration import (
    ConceptValueAsInstance,
    IriRename,
    LiteralDatatype,
    Migration,
    MigrationFile,
    ModuleVersionChange,
    PartOfFill,
    RoleListSplit,
    RecordedFileMigration,
    StoreMigration,
    shipped_migrations,
)
from llm_reasoner.knowledge.storage.record import (
    HypothesisRecord,
    current_module_versions,
)
from llm_reasoner.knowledge.storage.store import HypothesisStore, StoredFile

from .test_action_perspective_knowledge import (
    AnswerCase,
    answer_text,
    core_query,
    frame_query_on,
    service_answering,
)

# %% fixtures

SAMPLE_STORE = (
    Path(__file__).parents[2]
    / "cramera"
    / "src"
    / "cramera"
    / "resolution"
    / "resources"
    / "sample_hypotheses"
)
"""
The store of the hand-authored pouring hypothesis cramera ships, which meets its
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
    old=action_core.MeasurementUnit.DEGREE.value,
    new=action_core.MeasurementUnit.DEGREE.value.replace("Degree", "AngularDegree"),
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
        data=json.loads((directory / SAMPLE_RECORD / StoredFile.RECORD).read_text())
    )


STRUCTURE = URIRef("http://example.org/hypothesis/migration#structure")
"""
The fragment the concept and literal tests write.
"""


def decomposed_task() -> URIRef:
    """
    :return: The property naming the task a structure decomposes.
    """
    return URIRef(
        PropertyTerm.of_field(
            clazz=core.ActionPhaseStructure, field_name="decomposes_task"
        ).iri
    )


def structure_decomposing(task: URIRef) -> Graph:
    """
    :return: A structure decomposing a task written as an individual, as records were
        written before concepts became instances of their classes.
    """
    graph = Graph()
    graph.add((STRUCTURE, RDF.type, OWL.NamedIndividual))
    graph.add((STRUCTURE, RDF.type, URIRef(core.ActionPhaseStructure.term.iri)))
    graph.add((STRUCTURE, decomposed_task(), task))
    return graph


def concepts_as_individuals(graph: Graph) -> Graph:
    """
    :return: The graph with every concept written as the individual its class names,
        as records were written before concepts became instances of their classes.
    """
    individuals = Graph()
    for subject, predicate, value in graph:
        if isinstance(subject, BNode) or (predicate == RDF.type and value == OWL.Class):
            continue
        if predicate == RDFS.subClassOf and str(subject).startswith(
            HYPOTHESIS_CONCEPT_NAMESPACE
        ):
            continue
        if isinstance(value, BNode):
            (value,) = graph.objects(value, RDF.type)
        individuals.add((subject, predicate, value))
    return individuals


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


# %% concepts as instances of their classes


class TestConceptValues:
    def test_a_concept_written_as_an_individual_becomes_an_instance_of_its_class(self):
        pouring_task = URIRef(CRAM_AICOR_NAMESPACE + "Pouring")
        migration = Migration(
            name="concepts",
            concept_values=(ConceptValueAsInstance(property=str(decomposed_task())),),
        )
        (structure,) = (
            GraphReader()
            .read(migration.graph(structure_decomposing(pouring_task)))
            .fragments
        )
        assert structure.decomposes_task == ConceptReference(iri=str(pouring_task))

    def test_a_concept_of_the_hypothesis_becomes_a_class_under_the_range(self):
        decanting = URIRef(HYPOTHESIS_CONCEPT_NAMESPACE + "Decanting")
        task_range = URIRef(CRAM_AICOR_NAMESPACE + "ManipulationActionCategory")
        migrated = Migration(
            name="concepts",
            concept_values=(
                ConceptValueAsInstance(
                    property=str(decomposed_task()), range=str(task_range)
                ),
            ),
        ).graph(structure_decomposing(decanting))
        assert set(migrated.predicate_objects(decanting)) == {
            (RDF.type, OWL.Class),
            (RDFS.subClassOf, task_range),
        }

    def test_a_range_given_later_is_added_to_a_migrated_concept(self):
        decanting = URIRef(HYPOTHESIS_CONCEPT_NAMESPACE + "Decanting")
        task_range = URIRef(CRAM_AICOR_NAMESPACE + "ManipulationActionCategory")
        without_range = Migration(
            name="concepts",
            concept_values=(ConceptValueAsInstance(property=str(decomposed_task())),),
        )
        with_range = Migration(
            name="ranges",
            concept_values=(
                ConceptValueAsInstance(
                    property=str(decomposed_task()), range=str(task_range)
                ),
            ),
        )
        migrated = with_range.graph(
            without_range.graph(structure_decomposing(decanting))
        )
        assert (decanting, RDFS.subClassOf, task_range) in migrated

    def test_migrating_concepts_twice_is_migrating_them_once(self):
        migration = Migration(
            name="concepts",
            concept_values=(ConceptValueAsInstance(property=str(decomposed_task())),),
        )
        once = migration.graph(
            structure_decomposing(URIRef(CRAM_AICOR_NAMESPACE + "Pouring"))
        )
        assert isomorphic(migration.graph(once), once)
        assert not migration.touches(once)

    def test_a_literal_takes_the_datatype_its_property_now_ranges_over(self):
        term = PropertyTerm.of_field(
            clazz=flow_onset.PhysicsSimulationSetup, field_name="run_count"
        )
        graph = Graph()
        graph.add((STRUCTURE, URIRef(term.iri), Literal(3.0)))
        migration = Migration(
            name="datatypes",
            literal_datatypes=(
                LiteralDatatype(property=term.iri, datatype=str(XSD.positiveInteger)),
            ),
        )
        assert migration.touches(graph)
        assert set(migration.graph(graph)) == {
            (STRUCTURE, URIRef(term.iri), Literal(3, datatype=XSD.positiveInteger))
        }


# %% parts linked to their participants


LIP_OF_SOURCE = PartOfFill(
    part=pouring.PouringLip.term.iri,
    whole=action_core.SourceSlot.term.iri,
    property=PropertyTerm.of_field(
        clazz=action_core.PartSlot, field_name="is_part_of_slot"
    ).iri,
)
"""
A pouring lip linked to the source it is part of.
"""


def lip_and_sources(*sources: str) -> Graph:
    """
    :return: A pouring lip naming no source, with the given source containers.
    """
    graph = Graph()
    graph.add((STRUCTURE, RDF.type, URIRef(pouring.PouringLip.term.iri)))
    for source in sources:
        graph.add((URIRef(source), RDF.type, URIRef(pouring.SourceContainer.term.iri)))
    return graph


class TestPartOfFill:
    def test_a_part_is_linked_to_the_one_participant_of_its_whole(self):
        source = "http://example.org/hypothesis/migration#carton"
        migrated = Migration(name="parts", part_of_fills=(LIP_OF_SOURCE,)).graph(
            lip_and_sources(source)
        )
        assert (STRUCTURE, URIRef(LIP_OF_SOURCE.property), URIRef(source)) in migrated

    def test_a_part_is_not_linked_among_several_participants(self):
        graph = lip_and_sources(
            "http://example.org/hypothesis/migration#carton",
            "http://example.org/hypothesis/migration#jug",
        )
        assert not Migration(name="parts", part_of_fills=(LIP_OF_SOURCE,)).touches(
            graph
        )


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

    def test_a_module_the_record_did_not_name_is_added_to_its_versions(self, store):
        added = ModuleVersionChange(
            module=ActionPerspectiveFile.POURING_FRAME.value,
            old="",
            new=current_module_versions([ActionPerspectiveFile.POURING_FRAME])[
                ActionPerspectiveFile.POURING_FRAME.value
            ],
        )
        StoreMigration(
            directory=store,
            migrations=[Migration(name="add", module_versions=(added,))],
        ).run()
        assert sample_record(store).module_versions[added.module] == added.new

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

    def test_a_record_is_read_back_only_once_every_pending_migration_applied(
        self, store
    ):
        hypothesis_file = store / SAMPLE_RECORD / StoredFile.HYPOTHESIS
        concepts_as_individuals(
            Graph().parse(hypothesis_file, format="turtle")
        ).serialize(hypothesis_file, format="turtle")
        old = sample_record(store).module_versions["pouring.ttl"]
        report = StoreMigration(
            directory=store,
            migrations=[
                Migration(name="move", module_versions=(pouring_version_moved(old),)),
                Migration(
                    name="concepts",
                    concept_values=tuple(
                        rule
                        for migration in shipped_migrations()
                        for rule in migration.concept_values
                    ),
                ),
            ],
        ).run()
        assert report.migrated == [SAMPLE_RECORD]

    def test_a_record_left_stale_takes_the_migrations_it_missed(self, store):
        hypothesis_file = store / SAMPLE_RECORD / StoredFile.HYPOTHESIS
        concepts_as_individuals(
            Graph().parse(hypothesis_file, format="turtle")
        ).serialize(hypothesis_file, format="turtle")
        old = sample_record(store).module_versions["pouring.ttl"]
        missed = Migration(
            name="concepts",
            concept_values=tuple(
                rule
                for migration in shipped_migrations()
                for rule in migration.concept_values
            ),
            module_versions=(pouring_version_moved(old),),
        )
        later = Migration(
            name="later", module_versions=(pouring_version_moved(old + ".1"),)
        )
        (store / MigrationFile.APPLIED).write_text(json.dumps([missed.name]))
        report = StoreMigration(directory=store, migrations=[missed, later]).run()
        assert report.migrated == [SAMPLE_RECORD]
        assert (
            sample_record(store).module_versions["pouring.ttl"]
            == later.module_versions[0].new
        )

    def test_a_migrated_record_takes_no_migration_it_was_migrated_through(self, store):
        old = sample_record(store).module_versions["pouring.ttl"]
        through = Migration(
            name="through",
            renames=(RENAMED_PROPERTY,),
            module_versions=(
                ModuleVersionChange(
                    module=PredictiveManipulationOntologyFile.POURING.value,
                    old=old + ".0",
                    new=old,
                ),
            ),
        )
        (store / MigrationFile.APPLIED).write_text(json.dumps([through.name]))
        later = Migration(name="later", module_versions=(pouring_version_moved(old),))
        report = StoreMigration(directory=store, migrations=[through, later]).run()
        assert report.migrated == [SAMPLE_RECORD]

    def test_a_record_its_missed_migrations_leave_behind_is_stale_unread(self, store):
        hypothesis_file = store / SAMPLE_RECORD / StoredFile.HYPOTHESIS
        concepts_as_individuals(
            Graph().parse(hypothesis_file, format="turtle")
        ).serialize(hypothesis_file, format="turtle")
        old = sample_record(store).module_versions["pouring.ttl"]
        through = Migration(
            name="through", module_versions=(pouring_version_moved(old + ".0"),)
        )
        (store / MigrationFile.APPLIED).write_text(json.dumps([through.name]))
        later = Migration(name="later", module_versions=(pouring_version_moved(old),))
        report = StoreMigration(directory=store, migrations=[through, later]).run()
        assert report.stale == [SAMPLE_RECORD]

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


CONDITION_ON_ROLE = (
    PredictiveManipulationOntologyFile.CORE.namespace + "conditionOnRole"
)
"""
The property a condition of the microworld listed its roles with, in no order.
"""

CONDITION_ROLES_SPLIT = RoleListSplit(
    property=CONDITION_ON_ROLE,
    targets=(
        PropertyTerm.of_field(
            clazz=action_core.SlotConditionType, field_name="condition_on_slot"
        ).iri,
        PropertyTerm.of_field(
            clazz=action_core.SlotConditionType, field_name="condition_involves_slot"
        ).iri,
    ),
)
"""
The roles of a condition split into the slot it is on and the slots it involves.
"""


PARAMETER_ON_ROLE = (
    PredictiveManipulationOntologyFile.CORE.namespace + "parameterOnRole"
)
"""
The property a parameter of the microworld listed its roles with, in no order.
"""

PARAMETER_ROLES_SPLIT = RoleListSplit(
    property=PARAMETER_ON_ROLE,
    targets=tuple(
        PropertyTerm.of_field(clazz=action_core.SlotQuantity, field_name=name).iri
        for name in ("about_slot", "relative_to_slot", "quantity_involves_slot")
    ),
)
"""
The roles of a parameter split into the slot it is about, the slot it is measured from
and the slots it involves.
"""

ROLES = Namespace("http://example.org/hypothesis/migration#")
"""
The namespace of the roles a listing names.
"""


def listing_roles(
    property_iri: str, individual_class: str, *roles: tuple[str, str]
) -> Graph:
    """
    :param property_iri: The property listing the roles.
    :param roles: Each role's name and the IRI of its class.
    :return: An individual of a class listing roles of the given classes.
    """
    graph = Graph()
    graph.add((STRUCTURE, RDF.type, URIRef(individual_class)))
    for name, role_class in roles:
        graph.add((ROLES[name], RDF.type, URIRef(role_class)))
        graph.add((STRUCTURE, URIRef(property_iri), ROLES[name]))
    return graph


class TestRoleListSplit:
    def test_roles_are_split_by_the_slots_the_class_names(self):
        migrated = Migration(
            name="split", role_list_splits=(CONDITION_ROLES_SPLIT,)
        ).graph(
            listing_roles(
                CONDITION_ON_ROLE,
                pouring.LipOverReceptacle.term.iri,
                ("lip", pouring.PouringLip.term.iri),
                ("bowl", pouring.TargetReceptacle.term.iri),
            )
        )
        (condition,) = [
            fragment
            for fragment in GraphReader().read(migrated).fragments
            if isinstance(fragment, pouring.LipOverReceptacle)
        ]
        assert isinstance(condition.condition_on_slot, pouring.PouringLip)
        assert [type(slot) for slot in condition.condition_involves_slot] == [
            pouring.TargetReceptacle
        ]
        assert (None, URIRef(CONDITION_ON_ROLE), None) not in migrated

    def test_a_single_role_is_the_slot_the_condition_is_on(self):
        migrated = Migration(
            name="split", role_list_splits=(CONDITION_ROLES_SPLIT,)
        ).graph(
            listing_roles(
                CONDITION_ON_ROLE,
                core.PhaseConditionType.term.iri,
                ("carton", pouring.SourceContainer.term.iri),
            )
        )
        assert (
            STRUCTURE,
            URIRef(CONDITION_ROLES_SPLIT.targets[0]),
            ROLES.carton,
        ) in migrated

    def test_roles_a_class_does_not_tell_apart_are_left_listed(self):
        graph = listing_roles(
            CONDITION_ON_ROLE,
            core.PhaseConditionType.term.iri,
            ("carton", pouring.SourceContainer.term.iri),
            ("bowl", pouring.TargetReceptacle.term.iri),
        )
        assert not Migration(
            name="split", role_list_splits=(CONDITION_ROLES_SPLIT,)
        ).touches(graph)


class TestParameterRoleListSplit:
    def test_an_interaction_is_about_one_role_and_involves_the_other(self):
        migrated = Migration(
            name="split", role_list_splits=(PARAMETER_ROLES_SPLIT,)
        ).graph(
            listing_roles(
                PARAMETER_ON_ROLE,
                prehension.GripForceSetpoint.term.iri,
                ("gripper", prehension.GraspingEffector.term.iri),
                ("carton", pouring.SourceContainer.term.iri),
            )
        )
        about, relative, involves = PARAMETER_ROLES_SPLIT.targets
        assert set(migrated.predicate_objects(STRUCTURE)) - {
            (RDF.type, URIRef(prehension.GripForceSetpoint.term.iri))
        } == {(URIRef(about), ROLES.carton), (URIRef(involves), ROLES.gripper)}

    def test_a_geometric_quantity_is_measured_from_the_other_role(self):
        migrated = Migration(
            name="split", role_list_splits=(PARAMETER_ROLES_SPLIT,)
        ).graph(
            listing_roles(
                PARAMETER_ON_ROLE,
                pouring.LipAlignmentSetpoint.term.iri,
                ("bowl", pouring.TargetReceptacle.term.iri),
                ("lip", pouring.PouringLip.term.iri),
            )
        )
        about, relative, involves = PARAMETER_ROLES_SPLIT.targets
        assert set(migrated.predicate_objects(STRUCTURE)) - {
            (RDF.type, URIRef(pouring.LipAlignmentSetpoint.term.iri))
        } == {(URIRef(about), ROLES.lip), (URIRef(relative), ROLES.bowl)}

    def test_a_role_no_slot_of_the_class_takes_is_left_listed(self):
        graph = listing_roles(
            PARAMETER_ON_ROLE,
            pouring.TransferredQuantitySetpoint.term.iri,
            ("bowl", pouring.TargetReceptacle.term.iri),
        )
        assert not Migration(
            name="split", role_list_splits=(PARAMETER_ROLES_SPLIT,)
        ).touches(graph)


class TestRecordsBuildingOnOthers:
    def test_a_record_is_checked_with_the_fragments_of_the_records_it_builds_on(
        self, tmp_path
    ):
        service = service_answering(
            HypothesisStore(directory=tmp_path),
            answer_text(AnswerCase.CORE),
            answer_text(AnswerCase.FRAME),
        )
        core_view = service.view_for(core_query())
        frame_record = service.view_for(frame_query_on(core_view)).record
        report = StoreMigration(
            directory=tmp_path, migrations=[Migration(name="nothing")]
        ).run()
        assert sorted(report.migrated) == sorted(
            [core_view.record.record_id, frame_record.record_id]
        )


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
