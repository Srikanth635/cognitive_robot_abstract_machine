"""
Tests of knowledge in the action perspective vocabularies: hypotheses about an action's
core frame and about each perspective on it are generated, checked, stored and read like
the Flanagan ones, each under the versions of the modules its vocabulary uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from krrood.entity_query_language.factories import the
from rdflib import Graph, Namespace
from typing_extensions import Iterable, List

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.knowledge.admission.consistency import ConsistencyCheck
from llm_reasoner.knowledge.admission.hermit import HermitCheck
from llm_reasoner.knowledge.generation.backend import HypothesisBackend
from llm_reasoner.knowledge.generation.briefs import (
    PREDICTIVE_MANIPULATION_BRIEF,
    GenerationBrief,
)
from llm_reasoner.knowledge.generation.prompts import (
    GenerationPrompt,
    GenerationSection,
)
from llm_reasoner.knowledge.generation.schema import GenerationSchema
from llm_reasoner.knowledge.hypotheses.service import KnowledgeService
from llm_reasoner.knowledge.ontologies import (
    CRAM_AICOR_NAMESPACE,
    KNOWLEDGE_ONTOLOGY_MODULES,
    OntologyModuleFile,
    ActionPerspectiveFile,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import (
    action_core,
    action_frame,
)
from llm_reasoner.knowledge.ontologies.predictive_manipulation import core, pouring
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary
from llm_reasoner.knowledge.ontologies.terms import ConceptReference
from llm_reasoner.knowledge.storage.record import (
    ConsistencyVerdict,
    current_module_versions,
)
from llm_reasoner.knowledge.storage.store import HypothesisStore
from llm_reasoner.knowledge.vocabulary_proposal.perspective import Perspective

from .prompt_sections import bodies

CASES = Path(__file__).parent / "action_perspective_cases"
"""
Scripted answers of a language model describing pouring in the action perspectives.
"""

POURING = ConceptReference(iri=CRAM_AICOR_NAMESPACE + "Pouring")
"""
CRAM-AICOR's pouring, the action the scripted answers describe.
"""

CORE = Namespace(ActionPerspectiveFile.ACTION_CORE.namespace)
"""
The terms of the action core.
"""

# %% helpers


class AlwaysConsistent(ConsistencyCheck):
    """
    A consistency check that finds every hypothesis consistent.
    """

    def is_consistent(self, graph: Graph) -> bool:
        return True


# %% looking up the generated vocabulary


class TestVocabularyLookup:
    def test_every_registered_module_is_looked_up(self):
        assert set(Vocabulary().modules) == set(KNOWLEDGE_ONTOLOGY_MODULES)

    def test_a_class_of_the_action_core_is_found_by_its_iri(self):
        assert (
            Vocabulary().definition_named(str(CORE.PouringFrame))
            is action_core.PouringFrame
        )

    def test_a_class_of_a_perspective_is_found_by_its_name(self):
        assert (
            Vocabulary().class_named(action_frame.FrameStereotype.__name__)
            is action_frame.FrameStereotype
        )


# %% the modules a hypothesis uses


class TestModulesUsed:
    def test_a_core_hypothesis_uses_only_the_action_core(self):
        assert GenerationSchema.for_root(action_core.PouringFrame).module_files() == {
            ActionPerspectiveFile.ACTION_CORE
        }

    def test_a_frame_hypothesis_uses_the_action_core_and_the_frame(self):
        assert GenerationSchema.for_root(
            action_frame.FrameStereotype
        ).module_files() == {
            ActionPerspectiveFile.ACTION_CORE,
            ActionPerspectiveFile.ACTION_FRAME,
        }

    def test_a_phase_structure_uses_no_action_perspective(self):
        used = GenerationSchema.for_root(pouring.PouringPhaseStructure).module_files()
        assert used and used <= set(PredictiveManipulationOntologyFile)

    def test_a_core_hypothesis_is_offered_no_specialisation_a_perspective_declares(
        self,
    ):
        assert (
            action_frame.AdditionalConsequence.__name__
            not in GenerationSchema.for_root(action_core.PouringFrame).types
        )

    def test_a_core_field_accepts_no_specialisation_a_perspective_declares(self):
        frame = GenerationSchema.for_root(action_core.PouringFrame).get(
            action_core.PouringFrame.__name__
        )
        assert (
            action_frame.AdditionalConsequence.__name__
            not in frame.field_named("brings_about").accepted_types
        )

    def test_a_general_phase_structure_still_uses_every_predictive_manipulation_module(
        self,
    ):
        assert GenerationSchema.for_root(
            core.ActionPhaseStructure
        ).module_files() == set(PredictiveManipulationOntologyFile)


class TestUsableModules:
    def test_a_perspective_is_used_with_the_action_core_only(self):
        assert ActionPerspectiveFile.ACTION_FRAME.usable_modules == {
            ActionPerspectiveFile.ACTION_CORE,
            ActionPerspectiveFile.ACTION_FRAME,
        }

    def test_the_action_core_is_used_alone(self):
        assert ActionPerspectiveFile.ACTION_CORE.usable_modules == {
            ActionPerspectiveFile.ACTION_CORE
        }

    def test_a_predictive_manipulation_module_is_used_with_its_whole_family(self):
        for module_file in PredictiveManipulationOntologyFile:
            assert module_file.usable_modules == set(PredictiveManipulationOntologyFile)


# %% versions a record is kept under


class TestModuleVersions:
    def test_the_versions_are_those_of_the_modules_asked_about(self):
        assert set(current_module_versions([ActionPerspectiveFile.ACTION_CORE])) == {
            ActionPerspectiveFile.ACTION_CORE.value
        }

    def test_without_modules_named_every_registered_module_has_its_version(self):
        assert set(current_module_versions()) == {
            module_file.value for module_file in KNOWLEDGE_ONTOLOGY_MODULES
        }

    def test_module_file_names_are_unique_across_the_families(self):
        names = [module_file.value for module_file in KNOWLEDGE_ONTOLOGY_MODULES]
        assert len(names) == len(set(names))


# %% the reasoner's background


class TestReasonerBackground:
    def test_a_check_covering_a_module_adds_it_to_the_background(self):
        microworld = [
            module_file.path for module_file in PredictiveManipulationOntologyFile
        ]
        check = HermitCheck(background_files=microworld)
        assert list(
            check.covering({ActionPerspectiveFile.ACTION_CORE}).background_files
        ) == [*microworld, ActionPerspectiveFile.ACTION_CORE.path]

    def test_a_check_covering_modules_it_has_is_unchanged(self):
        microworld = [
            module_file.path for module_file in PredictiveManipulationOntologyFile
        ]
        check = HermitCheck(background_files=microworld)
        assert (
            list(
                check.covering(set(PredictiveManipulationOntologyFile)).background_files
            )
            == microworld
        )

    def test_a_check_judging_by_other_means_covers_any_module_as_it_is(self):
        check = AlwaysConsistent()
        assert check.covering({ActionPerspectiveFile.ACTION_CORE}) is check


@dataclass
class CoverageRecorded(ConsistencyCheck):
    """
    A consistency check that finds every hypothesis consistent and keeps the modules it
    was asked to cover.
    """

    covered: List[set[OntologyModuleFile]] = field(default_factory=list)
    """
    The modules of each request to cover, in order.
    """

    def is_consistent(self, graph: Graph) -> bool:
        return True

    def covering(self, module_files: Iterable[OntologyModuleFile]) -> ConsistencyCheck:
        self.covered.append(set(module_files))
        return self


def core_backend(store: HypothesisStore, consistency=None) -> HypothesisBackend:
    """
    :return: A backend whose model answers with the scripted core of pouring, once.
    """
    return HypothesisBackend(
        client=ScriptedClient(responses=[(CASES / "pouring_core.json").read_text()]),
        store=store,
        consistency=consistency,
    )


def core_query():
    """
    :return: A query for the core frame of pouring.
    """
    return the(action_core.PouringFrame)(frames_action=POURING)


# %% a core hypothesis


class TestCoreHypothesis:
    def test_a_core_hypothesis_is_admitted_with_its_slots(self, tmp_path):
        backend = core_backend(HypothesisStore(directory=tmp_path))
        (frame,) = core_query().evaluate(backend=backend)
        assert isinstance(frame, action_core.PouringFrame)
        assert {type(slot) for slot in frame.defines_slot} == {
            action_core.AgentSlot,
            action_core.SourceSlot,
            action_core.SubstanceSlot,
            action_core.GoalSlot,
            action_core.PartSlot,
        }

    def test_its_record_keeps_the_version_of_the_action_core_only(self, tmp_path):
        backend = core_backend(HypothesisStore(directory=tmp_path))
        list(core_query().evaluate(backend=backend))
        assert backend.last_record.module_versions == current_module_versions(
            [ActionPerspectiveFile.ACTION_CORE]
        )

    def test_its_open_world_check_covers_the_action_core(self, tmp_path):
        consistency = CoverageRecorded()
        backend = core_backend(HypothesisStore(directory=tmp_path), consistency)
        list(core_query().evaluate(backend=backend))
        assert consistency.covered == [{ActionPerspectiveFile.ACTION_CORE}]

    def test_it_is_read_back_from_the_store_as_the_core_frame(self, tmp_path):
        service = KnowledgeService(
            backend=core_backend(HypothesisStore(directory=tmp_path))
        )
        view = service.view_for(core_query())
        assert isinstance(view.root, action_core.PouringFrame)
        assert view.root.frames_action == POURING

    def test_it_is_served_from_the_store_without_asking_again(self, tmp_path):
        service = KnowledgeService(
            backend=core_backend(HypothesisStore(directory=tmp_path))
        )
        first = service.serve(core_query())
        assert service.serve(core_query()) == first

    @pytest.mark.slow
    def test_it_is_admitted_by_the_reasoner_against_the_core_and_the_linked_ontologies(
        self, tmp_path
    ):
        check = HermitCheck.from_environment()
        if check is None or HermitCheck.shipped_directory() is None:
            pytest.skip("needs the linked ontologies and owlready2's HermiT")
        backend = core_backend(HypothesisStore(directory=tmp_path), check)
        list(core_query().evaluate(backend=backend))
        assert backend.last_record.consistency == ConsistencyVerdict.CONSISTENT


# %% what the model is asked to describe


def system_prompt_for(root: type) -> str:
    """
    :return: The system prompt of a generation rooted in a class.
    """
    return GenerationPrompt().system(GenerationSchema.for_root(root))


class TestGenerationBrief:
    def test_a_core_hypothesis_is_asked_with_the_cores_description(self):
        (role,) = bodies(
            system_prompt_for(action_core.PouringFrame), GenerationSection.ROLE
        )
        assert ActionPerspectiveFile.ACTION_CORE.description in role

    def test_a_frame_hypothesis_is_asked_with_the_frames_description(self):
        (role,) = bodies(
            system_prompt_for(action_frame.FrameStereotype), GenerationSection.ROLE
        )
        assert ActionPerspectiveFile.ACTION_FRAME.description in role

    def test_an_action_perspective_is_not_asked_about_phases(self):
        (rules,) = bodies(
            system_prompt_for(action_core.PouringFrame), GenerationSection.RULES
        )
        for rule in PREDICTIVE_MANIPULATION_BRIEF.rules:
            assert rule.render() not in rules

    def test_a_phase_structure_is_asked_as_before(self):
        assert GenerationBrief.for_root(pouring.PouringPhaseStructure) is (
            PREDICTIVE_MANIPULATION_BRIEF
        )
        (role,) = bodies(
            system_prompt_for(pouring.PouringPhaseStructure), GenerationSection.ROLE
        )
        assert role == PREDICTIVE_MANIPULATION_BRIEF.role

    def test_knowledge_is_generated_from_the_description_its_vocabulary_was_proposed_from(
        self,
    ):
        assert (
            Perspective.ACTION_FRAME.description
            == ActionPerspectiveFile.ACTION_FRAME.description
        )
