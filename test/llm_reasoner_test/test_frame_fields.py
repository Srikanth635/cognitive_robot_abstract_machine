"""
Tests of which fields of an action designator fill which parts of the action core's
frame: the mapping kept as data, its checks against the code ontology and the core, and
its matching to the fragments of a core hypothesis.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from krrood.entity_query_language.factories import the

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.knowledge.generation.backend import HypothesisBackend
from llm_reasoner.knowledge.hypotheses.service import KnowledgeService
from llm_reasoner.knowledge.ontologies import CRAM_AICOR_NAMESPACE
from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
from llm_reasoner.knowledge.ontologies.described_actions import DescribedActions
from llm_reasoner.knowledge.ontologies.frame_fields import (
    CodeFieldKind,
    CodeOntology,
    DimensionDescription,
    FrameFieldChecks,
    FrameFieldProblemKind,
    SlotDescription,
    UnknownFrameTerm,
    frame_part_from_json,
)
from llm_reasoner.knowledge.ontologies.terms import ConceptReference
from llm_reasoner.knowledge.storage.store import HypothesisStore

from .test_observed_components import RECORD, KnowledgeServingOneRecord

CASES = Path(__file__).parent / "action_perspective_cases"
"""
The scripted answers describing pouring in the action perspectives.
"""

POURING_DESIGNATOR = "coraplex.robot_plans.actions.composite.tool_based.PouringAction"
"""
The identifier of the pouring action designator.
"""

# %% helpers


@pytest.fixture(scope="module")
def code_ontology() -> CodeOntology:
    """
    The code ontology shipped with the package.
    """
    return CodeOntology.shipped()


@pytest.fixture
def core_frame(tmp_path) -> action_core.PouringFrame:
    """
    The frame of the scripted core hypothesis about pouring, read back from the store.
    """
    service = KnowledgeService(
        backend=HypothesisBackend(
            client=ScriptedClient(
                responses=[(CASES / "pouring_core.json").read_text()]
            ),
            store=HypothesisStore(directory=tmp_path),
        )
    )
    return service.view_for(
        the(action_core.PouringFrame)(
            frames_action=ConceptReference(iri=CRAM_AICOR_NAMESPACE + "Pouring")
        )
    ).root


def fragments_filled(described, frame) -> dict:
    """
    :return: The fragments of a frame each mapped field of a described action fills.
    """
    return {
        field_name: part.fragments_in(frame)
        for field_name, part in described.frame_fields.items()
    }


def labelled(fragments) -> list[str]:
    """
    :return: The labels of fragments, which name them in the scripted core.
    """
    return [fragment.label for fragment in fragments]


def scripted_label(fragment_id: str) -> str:
    """
    :return: The label the scripted core gives the fragment of an identifier.
    """
    answer = json.loads((CASES / "pouring_core.json").read_text())
    (label,) = [
        fragment["label"]
        for fragment in answer["fragments"]
        if fragment["id"] == fragment_id
    ]
    return label


# %% the code ontology


class TestCodeOntology:
    def test_a_field_holding_an_object_is_an_object_field(self, code_ontology):
        assert (
            code_ontology.fields_of(POURING_DESIGNATOR)["source_container"]
            is CodeFieldKind.OBJECT
        )

    def test_a_field_holding_a_number_is_a_value_field(self, code_ontology):
        assert (
            code_ontology.fields_of(POURING_DESIGNATOR)["tilt_angle"]
            is CodeFieldKind.VALUE
        )

    def test_a_class_the_ontology_lacks_has_no_fields(self, code_ontology):
        assert code_ontology.fields_of("some.module.Unknown") == {}


# %% reading a mapping


class TestReadingFrameParts:
    def test_a_slot_is_read_by_its_class_name(self):
        assert frame_part_from_json({"slot": "SourceSlot"}) == SlotDescription(
            slot=action_core.SourceSlot
        )

    def test_a_dimension_is_read_by_its_quantity_and_slots(self):
        assert frame_part_from_json(
            {"quantity": "POSITION", "about": "SourceSlot", "relative_to": "GoalSlot"}
        ) == DimensionDescription(
            quantity=action_core.QuantityKind.POSITION,
            about=action_core.SourceSlot,
            relative_to=action_core.GoalSlot,
        )

    def test_a_slot_class_the_core_lacks_is_refused(self):
        with pytest.raises(UnknownFrameTerm):
            frame_part_from_json({"slot": "CupSlot"})

    def test_a_class_of_the_core_that_is_no_slot_is_refused(self):
        with pytest.raises(UnknownFrameTerm):
            frame_part_from_json({"slot": "Objective"})

    def test_a_quantity_kind_the_core_lacks_is_refused(self):
        with pytest.raises(UnknownFrameTerm):
            frame_part_from_json({"quantity": "COLOUR", "about": "SourceSlot"})


# %% matching a mapping to a core frame


class TestMatching:
    def test_a_slot_matches_the_frames_slots_of_its_class(self, core_frame):
        assert labelled(
            SlotDescription(slot=action_core.SourceSlot).fragments_in(core_frame)
        ) == [scripted_label("source")]

    def test_a_dimension_about_a_slot_matches_one_about_a_part_of_it(self, core_frame):
        position = DimensionDescription(
            quantity=action_core.QuantityKind.POSITION,
            about=action_core.SourceSlot,
            relative_to=action_core.GoalSlot,
        )
        assert labelled(position.fragments_in(core_frame)) == [
            scripted_label("lip_position")
        ]

    def test_a_dimension_without_a_reference_slot_matches_by_quantity_and_slot(
        self, core_frame
    ):
        amount = DimensionDescription(
            quantity=action_core.QuantityKind.AMOUNT, about=action_core.SubstanceSlot
        )
        assert labelled(amount.fragments_in(core_frame)) == [scripted_label("amount")]

    def test_a_dimension_the_frame_lacks_matches_nothing(self, core_frame):
        tilt = DimensionDescription(
            quantity=action_core.QuantityKind.ORIENTATION, about=action_core.SourceSlot
        )
        assert tilt.fragments_in(core_frame) == []

    def test_a_dimension_measured_from_another_slot_does_not_match(self, core_frame):
        position_from_agent = DimensionDescription(
            quantity=action_core.QuantityKind.POSITION,
            about=action_core.SourceSlot,
            relative_to=action_core.AgentSlot,
        )
        assert position_from_agent.fragments_in(core_frame) == []


# %% checking a mapping


class TestChecks:
    def test_the_shipped_mappings_follow_every_check(self, code_ontology):
        checks = FrameFieldChecks(code_ontology=code_ontology)
        for described in DescribedActions().entries:
            assert checks.problems(described) == []

    def test_the_pouring_designator_maps_some_fields(self):
        described = DescribedActions().describing(POURING_DESIGNATOR)
        assert described.frame_fields

    def test_a_field_the_designator_lacks_is_refused(self, code_ontology):
        described = DescribedActions().describing(POURING_DESIGNATOR)
        mapped = replace(
            described,
            frame_fields={"spout": SlotDescription(slot=action_core.SourceSlot)},
        )
        assert [
            problem.kind
            for problem in FrameFieldChecks(code_ontology=code_ontology).problems(
                mapped
            )
        ] == [FrameFieldProblemKind.UNKNOWN_FIELD]

    def test_a_number_field_mapped_to_a_slot_is_refused(self, code_ontology):
        described = DescribedActions().describing(POURING_DESIGNATOR)
        mapped = replace(
            described,
            frame_fields={"tilt_angle": SlotDescription(slot=action_core.SourceSlot)},
        )
        assert [
            problem.kind
            for problem in FrameFieldChecks(code_ontology=code_ontology).problems(
                mapped
            )
        ] == [FrameFieldProblemKind.SLOT_FOR_VALUE]

    def test_an_object_field_mapped_to_a_dimension_is_refused(self, code_ontology):
        described = DescribedActions().describing(POURING_DESIGNATOR)
        mapped = replace(
            described,
            frame_fields={
                "source_container": DimensionDescription(
                    quantity=action_core.QuantityKind.POSITION,
                    about=action_core.SourceSlot,
                )
            },
        )
        assert [
            problem.kind
            for problem in FrameFieldChecks(code_ontology=code_ontology).problems(
                mapped
            )
        ] == [FrameFieldProblemKind.DIMENSION_FOR_OBJECT]


# %% the fragments each field fills in a consulted core


@pytest.fixture
def stored_core(tmp_path) -> tuple[KnowledgeService, str]:
    """
    A knowledge service holding the scripted core hypothesis about pouring, and its
    record.
    """
    service = KnowledgeService(
        backend=HypothesisBackend(
            client=ScriptedClient(
                responses=[(CASES / "pouring_core.json").read_text()]
            ),
            store=HypothesisStore(directory=tmp_path),
        )
    )
    return service, service.serve(
        the(action_core.PouringFrame)(
            frames_action=ConceptReference(iri=CRAM_AICOR_NAMESPACE + "Pouring")
        )
    )


class TestFramePartsFilled:
    def test_each_mapped_field_is_named_by_the_core_fragments_it_fills(
        self, stored_core
    ):
        service, core_record = stored_core
        view = service.view(core_record)
        described = DescribedActions().describing(POURING_DESIGNATOR)
        filled = service.frame_parts_filled(core_record, described.frame_fields)
        assert {
            field_name: [service.citation(core_record, name) for name in names]
            for field_name, names in filled.items()
        } == {
            field_name: [view.citation_of(fragment).iri for fragment in fragments]
            for field_name, fragments in fragments_filled(described, view.root).items()
            if fragments
        }

    def test_a_field_filling_nothing_in_the_core_is_left_out(self, stored_core):
        service, core_record = stored_core
        view = service.view(core_record)
        described = DescribedActions().describing(POURING_DESIGNATOR)
        unfilled = {
            field_name
            for field_name, fragments in fragments_filled(described, view.root).items()
            if not fragments
        }
        assert unfilled
        assert unfilled.isdisjoint(
            service.frame_parts_filled(core_record, described.frame_fields)
        )

    def test_a_designator_without_a_mapping_names_nothing(self, stored_core):
        service, core_record = stored_core
        assert service.frame_parts_filled(core_record, {}) == {}

    def test_knowledge_without_frames_names_nothing(self):
        assert (
            KnowledgeServingOneRecord().frame_parts_filled(
                RECORD, DescribedActions().describing(POURING_DESIGNATOR).frame_fields
            )
            == {}
        )
