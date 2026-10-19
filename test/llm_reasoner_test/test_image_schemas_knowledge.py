"""
Tests of knowledge about the image schemas of an action: a hypothesis about pouring in
the image-schemas vocabulary, generated on top of a stored core hypothesis, checked,
stored and read with it.
"""

from __future__ import annotations

import json

import pytest
from krrood.entity_query_language.factories import the

from llm_reasoner.clients.tools import ToolBox
from llm_reasoner.knowledge.admission.hermit import HermitCheck
from llm_reasoner.knowledge.generation.prompts import GenerationSection
from llm_reasoner.knowledge.hypotheses.knowledge_access import KnowledgeToolName
from llm_reasoner.knowledge.ontologies import ActionPerspectiveFile
from llm_reasoner.knowledge.ontologies.action_perspectives import image_schemas
from llm_reasoner.knowledge.storage.record import ConsistencyVerdict
from llm_reasoner.knowledge.storage.store import HypothesisStore
from llm_reasoner.knowledge.vocabulary_proposal.perspective import Perspective

from .prompt_sections import bodies
from .test_action_perspective_knowledge import (
    AnswerCase,
    answer_text as answer_in_cases,
    asked,
    core_query,
    service_answering,
    system_prompt_for,
)


class ImageSchemasCase:
    """
    The scripted answer about the image schemas of pouring, by file.
    """

    SCHEMAS = "pouring_image_schemas.json"


def schemas_query_on(core_view):
    """
    :return: A query for the schema stereotype of a stored core hypothesis's frame.
    """
    return the(image_schemas.SchemaStereotype)(
        schema_stereotype_of=core_view.citation_of(core_view.root)
    )


@pytest.fixture
def stored(tmp_path):
    """
    The scripted core and image schemas of pouring, generated and stored, with the
    service holding both and the views of the two.
    """
    service = service_answering(
        HypothesisStore(directory=tmp_path),
        answer_in_cases(AnswerCase.CORE),
        answer_in_cases(ImageSchemasCase.SCHEMAS),
    )
    core_record = service.serve(core_query())
    schemas_record = service.serve(schemas_query_on(service.view(core_record)))
    return service, core_record, schemas_record


# %% the hypothesis


class TestSchemaHypothesis:
    def test_it_is_about_the_stored_core_frame_itself(self, stored):
        service, core_record, schemas_record = stored
        assert (
            service.view(schemas_record).root.schema_stereotype_of
            is service.view(core_record).root
        )

    def test_its_participants_and_requirements_name_the_core_slots_themselves(
        self, stored
    ):
        service, core_record, schemas_record = stored
        core_slots = {id(slot) for slot in service.view(core_record).root.defines_slot}
        stereotype = service.view(schemas_record).root
        named = set()
        for schema in stereotype.has_schema:
            named |= {id(part.participating_slot) for part in schema.has_participation}
            for requirement in schema.has_requirement:
                named |= {
                    id(requirement.subject_slot),
                    id(requirement.reference_slot),
                }
        assert named <= core_slots

    def test_a_requirement_constrains_dimensions_the_core_leaves_open(self, stored):
        service, core_record, schemas_record = stored
        open_dimensions = {
            id(dimension) for dimension in service.view(core_record).root.leaves_open
        }
        constrained = {
            id(dimension)
            for schema in service.view(schemas_record).root.has_schema
            for requirement in schema.has_requirement
            for dimension in requirement.constrains_dimension
        }
        assert constrained
        assert constrained <= open_dimensions

    def test_the_lip_over_the_opening_is_a_vertically_over_requirement(self, stored):
        service, _, schemas_record = stored
        relations = {
            requirement.required_relation_kind
            for schema in service.view(schemas_record).root.has_schema
            for requirement in schema.has_requirement
        }
        assert image_schemas.RelationKind.VERTICALLY_OVER in relations

    def test_its_record_builds_on_the_core_record(self, stored):
        service, core_record, schemas_record = stored
        assert service.view(schemas_record).record.builds_on == [core_record]


# %% asking for it


class TestGenerationBrief:
    def test_it_is_asked_with_the_description_its_vocabulary_was_proposed_from(self):
        role = bodies(
            system_prompt_for(image_schemas.SchemaStereotype), GenerationSection.ROLE
        )
        assert ActionPerspectiveFile.IMAGE_SCHEMAS.description in role[0]
        assert (
            Perspective.IMAGE_SCHEMAS.description
            == ActionPerspectiveFile.IMAGE_SCHEMAS.description
        )


# %% reading it with the core


class TestReadingWithTheCore:
    def test_the_overview_names_the_perspective_read_with_the_core(self, stored):
        service, core_record, schemas_record = stored
        overview = asked(
            ToolBox(tools=service.tools(core_record, schemas_record)),
            KnowledgeToolName.KNOWLEDGE_OVERVIEW,
            {},
        )
        modules = {perspective["module"] for perspective in overview["perspectives"]}
        assert modules == {ActionPerspectiveFile.IMAGE_SCHEMAS.value}


# %% reasoning


@pytest.mark.slow
class TestAdmission:
    def test_it_is_admitted_by_the_reasoner_with_the_core_it_builds_on(self, tmp_path):
        check = HermitCheck.from_environment()
        if check is None or HermitCheck.shipped_directory() is None:
            pytest.skip("needs the linked ontologies and owlready2's HermiT")
        service = service_answering(
            HypothesisStore(directory=tmp_path),
            answer_in_cases(AnswerCase.CORE),
            answer_in_cases(ImageSchemasCase.SCHEMAS),
            consistency=check,
        )
        core_view = service.view_for(core_query())
        schemas_view = service.view_for(schemas_query_on(core_view))
        assert schemas_view.record.consistency == ConsistencyVerdict.CONSISTENT
