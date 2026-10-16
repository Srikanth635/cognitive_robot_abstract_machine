"""
Tests of knowledge in the action perspective vocabularies: hypotheses about an action's
core frame and about each perspective on it are generated, checked, stored and read like
the Flanagan ones, each under the versions of the modules its vocabulary uses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from krrood.entity_query_language.factories import the
from rdflib import RDF, Graph, Namespace, URIRef
from typing_extensions import Iterable, List

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.clients.tools import ToolBox
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
from llm_reasoner.knowledge.generation.request import HypothesisRequest
from llm_reasoner.knowledge.generation.schema import GenerationSchema
from llm_reasoner.knowledge.hypotheses.knowledge_access import KnowledgeToolName
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
from llm_reasoner.knowledge.ontologies.perspective_roots import PERSPECTIVE_ROOTS
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary
from llm_reasoner.knowledge.ontologies.terms import ConceptReference
from llm_reasoner.knowledge.storage.record import (
    ConsistencyVerdict,
    current_module_versions,
)
from llm_reasoner.knowledge.storage.citation import Citation, StoredRoot
from llm_reasoner.knowledge.storage.store import HypothesisStore
from llm_reasoner.knowledge.vocabulary_proposal.perspective import Perspective

from .prompt_sections import bodies
from .test_generation import GenerationCase
from .test_hypotheses import pouring_query as flanagan_query
from .test_hypotheses import service_answering as flanagan_service_answering

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


# %% a perspective hypothesis on top of a stored core


class AnswerCase:
    """
    The scripted answers about pouring, by file.
    """

    CORE = "pouring_core.json"
    FRAME = "pouring_frame.json"


def answer_text(case: str) -> str:
    """
    :return: A scripted answer as the model writes it.
    """
    return (CASES / case).read_text()


def frame_answer_adding(fragment: dict) -> str:
    """
    :return: The scripted frame answer with one more fragment.
    """
    answer = json.loads(answer_text(AnswerCase.FRAME))
    answer["fragments"].append(fragment)
    return json.dumps(answer)


def service_answering(store: HypothesisStore, *answers: str, consistency=None):
    """
    :return: A knowledge service whose model gives the answers in turn.
    """
    return KnowledgeService(
        backend=HypothesisBackend(
            client=ScriptedClient(responses=list(answers)),
            store=store,
            consistency=consistency,
        )
    )


def frame_query_on(core_view):
    """
    :return: A query for the frame stereotype of a stored core hypothesis's frame.
    """
    return the(action_frame.FrameStereotype)(
        stereotype_of=core_view.citation_of(core_view.root)
    )


CITED_CORE_FRAME = Citation(record_id="core-record", fragment_name="pouring")
"""
A citation of the frame of a stored core hypothesis.
"""


class TestCitedFixedValues:
    def test_a_cited_fragment_is_keyed_by_its_iri(self):
        request = HypothesisRequest.from_match(
            the(action_frame.FrameStereotype)(stereotype_of=CITED_CORE_FRAME), ""
        )
        assert request.written_fixed_values == {"stereotype_of": CITED_CORE_FRAME.iri}

    def test_the_model_is_told_a_cited_fragment_by_its_name(self):
        prompt = GenerationPrompt().initial(
            GenerationSchema.for_root(action_frame.FrameStereotype),
            "pour",
            {"stereotype_of": CITED_CORE_FRAME},
        )
        (fixed,) = bodies(prompt, GenerationSection.FIXED_VALUES)
        assert json.loads(fixed) == {"stereotype_of": CITED_CORE_FRAME.fragment_name}

    def test_a_request_builds_on_the_records_it_cites(self):
        request = HypothesisRequest.from_match(
            the(action_frame.FrameStereotype)(stereotype_of=CITED_CORE_FRAME), ""
        )
        assert request.builds_on == [CITED_CORE_FRAME.record_id]


class TestPerspectiveHypothesis:
    def test_a_frame_hypothesis_is_about_the_stored_core_frame_itself(self, tmp_path):
        service = service_answering(
            HypothesisStore(directory=tmp_path),
            answer_text(AnswerCase.CORE),
            answer_text(AnswerCase.FRAME),
        )
        core_view = service.view_for(core_query())
        frame_view = service.view_for(frame_query_on(core_view))
        assert frame_view.root.stereotype_of is core_view.root

    def test_its_expectations_name_the_core_slots_themselves(self, tmp_path):
        service = service_answering(
            HypothesisStore(directory=tmp_path),
            answer_text(AnswerCase.CORE),
            answer_text(AnswerCase.FRAME),
        )
        core_view = service.view_for(core_query())
        frame_view = service.view_for(frame_query_on(core_view))
        expected_slots = {
            id(expectation.expects_of_slot)
            for expectation in frame_view.root.has_slot_expectation
        }
        assert expected_slots <= {id(slot) for slot in core_view.root.defines_slot}

    def test_the_model_is_shown_the_core_fragments_as_given(self, tmp_path):
        service = service_answering(
            HypothesisStore(directory=tmp_path),
            answer_text(AnswerCase.CORE),
            answer_text(AnswerCase.FRAME),
        )
        core_view = service.view_for(core_query())
        frame_view = service.view_for(frame_query_on(core_view))
        (given,) = bodies(frame_view.record.exchanges[0].user, GenerationSection.GIVEN)
        assert {entry["id"] for entry in json.loads(given)} == {
            core_view.citation_of(fragment).fragment_name
            for fragment in core_view.fragments
        }

    def test_its_record_builds_on_the_core_record(self, tmp_path):
        service = service_answering(
            HypothesisStore(directory=tmp_path),
            answer_text(AnswerCase.CORE),
            answer_text(AnswerCase.FRAME),
        )
        core_record = service.serve(core_query())
        frame_record = service.serve(frame_query_on(service.view(core_record)))
        assert service.view(frame_record).record.builds_on == [core_record]

    def test_its_stored_turtle_refers_to_the_core_fragments_without_restating_them(
        self, tmp_path
    ):
        store = HypothesisStore(directory=tmp_path)
        service = service_answering(
            store, answer_text(AnswerCase.CORE), answer_text(AnswerCase.FRAME)
        )
        core_view = service.view_for(core_query())
        frame_graph = store.get(service.serve(frame_query_on(core_view))).graph
        core_frame = URIRef(core_view.citation_of(core_view.root).iri)
        assert (None, None, core_frame) in frame_graph
        assert (core_frame, RDF.type, None) not in frame_graph

    def test_an_answer_restating_a_core_fragment_is_returned_for_repair(self, tmp_path):
        restating = frame_answer_adding(
            {"id": "second_source", "type": "SourceSlot", "label": "another source"}
        )
        service = service_answering(
            HypothesisStore(directory=tmp_path),
            answer_text(AnswerCase.CORE),
            restating,
            json.dumps({"fragments": [], "remove": ["second_source"]}),
        )
        core_view = service.view_for(core_query())
        exchanges = service.view_for(frame_query_on(core_view)).record.exchanges
        (problems,) = bodies(exchanges[1].user, GenerationSection.PROBLEMS)
        assert "second_source" in problems
        assert len(exchanges) == 2

    def test_an_answer_naming_a_fragment_like_a_given_one_is_returned_for_repair(
        self, tmp_path
    ):
        answer = json.loads(answer_text(AnswerCase.FRAME))
        answer["fragments"][1]["id"] = "goal"
        answer["fragments"][0]["has_slot_expectation"][0] = "goal"
        repair = {
            "fragments": json.loads(answer_text(AnswerCase.FRAME))["fragments"],
            "remove": ["goal"],
        }
        service = service_answering(
            HypothesisStore(directory=tmp_path),
            answer_text(AnswerCase.CORE),
            json.dumps(answer),
            json.dumps(repair),
        )
        core_view = service.view_for(core_query())
        exchanges = service.view_for(frame_query_on(core_view)).record.exchanges
        (problems,) = bodies(exchanges[1].user, GenerationSection.PROBLEMS)
        assert "'goal'" in problems

    def test_it_is_served_from_the_store_without_asking_again(self, tmp_path):
        service = service_answering(
            HypothesisStore(directory=tmp_path),
            answer_text(AnswerCase.CORE),
            answer_text(AnswerCase.FRAME),
        )
        core_view = service.view_for(core_query())
        first = service.serve(frame_query_on(core_view))
        assert service.serve(frame_query_on(core_view)) == first

    @pytest.mark.slow
    def test_it_is_admitted_by_the_reasoner_with_the_core_it_builds_on(self, tmp_path):
        check = HermitCheck.from_environment()
        if check is None or HermitCheck.shipped_directory() is None:
            pytest.skip("needs the linked ontologies and owlready2's HermiT")
        service = service_answering(
            HypothesisStore(directory=tmp_path),
            answer_text(AnswerCase.CORE),
            answer_text(AnswerCase.FRAME),
            consistency=check,
        )
        core_view = service.view_for(core_query())
        frame_view = service.view_for(frame_query_on(core_view))
        assert frame_view.record.consistency == ConsistencyVerdict.CONSISTENT


# %% reading a core and its perspectives as one body of knowledge


@dataclass
class StoredPouring:
    """
    A stored core hypothesis about pouring, a frame hypothesis on top of it, and the
    service holding both.
    """

    service: KnowledgeService
    """
    The knowledge service holding the hypotheses.
    """

    core_record: str
    """
    The record of the core hypothesis.
    """

    frame_record: str
    """
    The record of the frame hypothesis.
    """

    def toolbox(self, *record_ids: str) -> ToolBox:
        """
        :return: The knowledge tools reading the records as one body of knowledge.
        """
        return ToolBox(tools=self.service.tools(*record_ids))

    def iri_of_name(self, name: str) -> str:
        """
        :return: The IRI a fragment name the tools give cites, in whichever of the two
            records holds the fragment.
        """
        return self.service.citation(self.core_record, name) or self.service.citation(
            self.frame_record, name
        )


@pytest.fixture
def stored_pouring(tmp_path) -> StoredPouring:
    """
    The scripted core and frame of pouring, generated and stored.
    """
    service = service_answering(
        HypothesisStore(directory=tmp_path),
        answer_text(AnswerCase.CORE),
        answer_text(AnswerCase.FRAME),
    )
    core_record = service.serve(core_query())
    frame_record = service.serve(frame_query_on(service.view(core_record)))
    return StoredPouring(
        service=service, core_record=core_record, frame_record=frame_record
    )


def asked(toolbox: ToolBox, name: KnowledgeToolName, arguments: dict) -> dict:
    """
    :return: The answer of a knowledge tool, read from JSON.
    """
    return json.loads(toolbox.call(name, arguments).content)


class TestReadingACore:
    def test_the_overview_names_the_action_of_the_core_frame(self, stored_pouring):
        overview = asked(
            stored_pouring.toolbox(stored_pouring.core_record),
            KnowledgeToolName.KNOWLEDGE_OVERVIEW,
            {},
        )
        assert overview["action"] == POURING.iri

    def test_the_overview_gives_the_frames_slots(self, stored_pouring):
        overview = asked(
            stored_pouring.toolbox(stored_pouring.core_record),
            KnowledgeToolName.KNOWLEDGE_OVERVIEW,
            {},
        )
        core_view = stored_pouring.service.view(stored_pouring.core_record)
        assert {
            stored_pouring.iri_of_name(slot["name"])
            for slot in overview["frame"]["slots"]
        } == {core_view.citation_of(slot).iri for slot in core_view.root.defines_slot}

    def test_the_overview_gives_the_frames_parameter_dimensions(self, stored_pouring):
        overview = asked(
            stored_pouring.toolbox(stored_pouring.core_record),
            KnowledgeToolName.KNOWLEDGE_OVERVIEW,
            {},
        )
        core_view = stored_pouring.service.view(stored_pouring.core_record)
        assert {
            stored_pouring.iri_of_name(dimension["name"])
            for dimension in overview["frame"]["parameter_dimensions"]
        } == {
            core_view.citation_of(dimension).iri
            for dimension in core_view.root.leaves_open
        }

    def test_a_core_read_alone_holds_no_perspective(self, stored_pouring):
        overview = asked(
            stored_pouring.toolbox(stored_pouring.core_record),
            KnowledgeToolName.KNOWLEDGE_OVERVIEW,
            {},
        )
        assert overview["perspectives"] == []

    def test_the_overview_names_the_perspectives_read_with_the_core(
        self, stored_pouring
    ):
        overview = asked(
            stored_pouring.toolbox(
                stored_pouring.core_record, stored_pouring.frame_record
            ),
            KnowledgeToolName.KNOWLEDGE_OVERVIEW,
            {},
        )
        frame_view = stored_pouring.service.view(stored_pouring.frame_record)
        (perspective,) = overview["perspectives"]
        assert perspective["module"] == ActionPerspectiveFile.ACTION_FRAME.value
        assert (
            stored_pouring.iri_of_name(perspective["root"]["name"])
            == frame_view.citation_of(frame_view.root).iri
        )

    def test_a_perspective_is_read_with_the_core_it_builds_on(self, stored_pouring):
        overview = asked(
            stored_pouring.toolbox(stored_pouring.frame_record),
            KnowledgeToolName.KNOWLEDGE_OVERVIEW,
            {},
        )
        core_view = stored_pouring.service.view(stored_pouring.core_record)
        assert (
            stored_pouring.iri_of_name(overview["frame"]["frame"]["name"])
            == core_view.citation_of(core_view.root).iri
        )

    def test_a_slot_is_described_with_the_expectations_linking_to_it(
        self, stored_pouring
    ):
        core_view = stored_pouring.service.view(stored_pouring.core_record)
        frame_view = stored_pouring.service.view(stored_pouring.frame_record)
        toolbox = stored_pouring.toolbox(
            stored_pouring.core_record, stored_pouring.frame_record
        )
        source = next(
            slot
            for slot in core_view.root.defines_slot
            if isinstance(slot, action_core.SourceSlot)
        )
        source_name = next(
            slot["name"]
            for slot in asked(toolbox, KnowledgeToolName.KNOWLEDGE_OVERVIEW, {})[
                "frame"
            ]["slots"]
            if stored_pouring.iri_of_name(slot["name"])
            == core_view.citation_of(source).iri
        )
        described = asked(
            toolbox, KnowledgeToolName.DESCRIBE_FRAGMENT, {"name": source_name}
        )
        linking_expectations = {
            stored_pouring.iri_of_name(link["name"])
            for link in described["linked_from"]
            if link["kind"] == action_frame.SlotExpectation.__name__
        }
        assert linking_expectations == {
            frame_view.citation_of(expectation).iri
            for expectation in frame_view.root.has_slot_expectation
            if expectation.expects_of_slot is source
        }

    def test_fragments_of_a_class_are_found_in_every_record_read(self, stored_pouring):
        found = asked(
            stored_pouring.toolbox(
                stored_pouring.core_record, stored_pouring.frame_record
            ),
            KnowledgeToolName.FIND_FRAGMENTS,
            {"kind": action_frame.SlotExpectation.__name__},
        )
        frame_view = stored_pouring.service.view(stored_pouring.frame_record)
        assert {
            stored_pouring.iri_of_name(fragment["name"])
            for fragment in found["fragments"]
        } == {
            frame_view.citation_of(expectation).iri
            for expectation in frame_view.root.has_slot_expectation
        }

    def test_every_tool_answers_in_its_declared_shape_for_a_core_and_a_perspective(
        self, stored_pouring
    ):
        tools = {
            tool.definition.name: tool
            for tool in stored_pouring.service.tools(
                stored_pouring.core_record, stored_pouring.frame_record
            )
        }
        overview = json.loads(
            tools[KnowledgeToolName.KNOWLEDGE_OVERVIEW].run({}).content
        )
        calls = {
            KnowledgeToolName.KNOWLEDGE_OVERVIEW: {},
            KnowledgeToolName.FIND_FRAGMENTS: {
                "kind": action_frame.SlotExpectation.__name__
            },
            KnowledgeToolName.DESCRIBE_FRAGMENT: {
                "name": overview["frame"]["slots"][0]["name"]
            },
        }
        for name, arguments in calls.items():
            Draft202012Validator(tools[name].definition.output_schema).validate(
                json.loads(tools[name].run(arguments).content)
            )


class TestReadingBesideFlanagan:
    def test_a_core_fragment_is_named_apart_from_a_flanagan_fragment_of_its_name(
        self, tmp_path
    ):
        store = HypothesisStore(directory=tmp_path)
        flanagan = flanagan_service_answering(store, GenerationCase.LIFT_AND_POUR)
        flanagan_record = flanagan.serve(flanagan_query())
        flanagan_view = flanagan.view(flanagan_record)
        flanagan_root_name = flanagan_view.citation_of(flanagan_view.root).fragment_name
        core_answer = json.loads(answer_text(AnswerCase.CORE))
        core_answer["fragments"][0]["id"] = flanagan_root_name
        service = service_answering(store, json.dumps(core_answer))
        core_record = service.serve(core_query())
        toolbox = ToolBox(tools=service.tools(flanagan_record, core_record))
        overview = asked(toolbox, KnowledgeToolName.KNOWLEDGE_OVERVIEW, {})
        core_frame_name = overview["frame"]["frame"]["name"]
        assert core_frame_name != flanagan_root_name
        assert (
            asked(
                toolbox, KnowledgeToolName.DESCRIBE_FRAGMENT, {"name": core_frame_name}
            )["kind"]["name"]
            == action_core.PouringFrame.__name__
        )
        assert (
            asked(
                toolbox,
                KnowledgeToolName.DESCRIBE_FRAGMENT,
                {"name": flanagan_root_name},
            )["kind"]["name"]
            == type(flanagan_view.root).__name__
        )

    def test_the_overview_keeps_the_phases_beside_the_frame(self, tmp_path):
        store = HypothesisStore(directory=tmp_path)
        flanagan_record = flanagan_service_answering(
            store, GenerationCase.LIFT_AND_POUR
        ).serve(flanagan_query())
        service = service_answering(store, answer_text(AnswerCase.CORE))
        core_record = service.serve(core_query())
        overview = asked(
            ToolBox(tools=service.tools(flanagan_record, core_record)),
            KnowledgeToolName.KNOWLEDGE_OVERVIEW,
            {},
        )
        flanagan_alone = asked(
            ToolBox(tools=service.tools(flanagan_record)),
            KnowledgeToolName.KNOWLEDGE_OVERVIEW,
            {},
        )
        assert overview["phases"] == flanagan_alone["phases"]
        assert "frame" in overview


# %% naming the root of a stored hypothesis, and the roots of the perspectives


class TestStoredRoot:
    def test_a_query_may_name_the_root_of_a_stored_hypothesis_by_its_record(
        self, stored_pouring
    ):
        frame_record = stored_pouring.service.serve(
            the(action_frame.FrameStereotype)(
                stereotype_of=StoredRoot(record_id=stored_pouring.core_record)
            )
        )
        assert frame_record == stored_pouring.frame_record

    def test_the_root_of_a_stored_hypothesis_is_cited_as_its_view_cites_it(
        self, stored_pouring
    ):
        core_view = stored_pouring.service.view(stored_pouring.core_record)
        assert stored_pouring.service.store.root_citation(
            stored_pouring.core_record
        ) == core_view.citation_of(core_view.root)


class TestPerspectiveRoots:
    def test_every_perspective_on_the_core_has_a_root(self):
        assert set(PERSPECTIVE_ROOTS) == set(ActionPerspectiveFile) - {
            ActionPerspectiveFile.ACTION_CORE
        }

    def test_the_root_of_a_perspective_belongs_to_its_module(self):
        for module_file, root in PERSPECTIVE_ROOTS.items():
            assert Vocabulary().module_file_of(root.root) is module_file

    def test_a_perspective_query_fixes_the_core_frame_it_describes(self):
        cited = StoredRoot(record_id="core-record")
        query = PERSPECTIVE_ROOTS[ActionPerspectiveFile.ACTION_FRAME].query(cited)
        assert query.type is action_frame.FrameStereotype
        assert query.kwargs == {"stereotype_of": cited}


# %% what every condition of the frame must state


def frame_answer_without(fragment_id: str, field_name: str) -> str:
    """
    :return: The scripted frame answer with one field left out of one fragment.
    """
    answer = json.loads(answer_text(AnswerCase.FRAME))
    for fragment in answer["fragments"]:
        if fragment["id"] == fragment_id:
            del fragment[field_name]
    return json.dumps(answer)


def problems_of_first_frame_answer(tmp_path, first_answer: str) -> str:
    """
    :return: The problems the first frame answer was returned for repair with, the
        repair being the scripted frame answer in full.
    """
    service = service_answering(
        HypothesisStore(directory=tmp_path),
        answer_text(AnswerCase.CORE),
        first_answer,
        answer_text(AnswerCase.FRAME),
    )
    core_view = service.view_for(core_query())
    exchanges = service.view_for(frame_query_on(core_view)).record.exchanges
    (problems,) = bodies(exchanges[1].user, GenerationSection.PROBLEMS)
    return problems


class TestConditionsStated:
    def test_a_condition_without_its_polarity_is_returned_for_repair(self, tmp_path):
        problems = problems_of_first_frame_answer(
            tmp_path, frame_answer_without("source_has_lip", "has_polarity")
        )
        assert "'source_has_lip'" in problems

    def test_a_state_condition_without_its_state_is_returned_for_repair(self, tmp_path):
        problems = problems_of_first_frame_answer(
            tmp_path, frame_answer_without("source_held", "condition_has_state")
        )
        assert "'source_held'" in problems
