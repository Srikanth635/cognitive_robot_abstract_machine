"""
Tests of the vocabulary proposal: a language model proposes the vocabulary of one
perspective on an action, which is checked against the rules every perspective follows,
written as an ontology module, and returned to the model for repair when it breaks one.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from typing_extensions import Any, Dict, List

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.knowledge.admission.consistency import ConsistencyCheck
from llm_reasoner.knowledge.ontologies import (
    CRAM_AICOR_NAMESPACE,
    CRAM_AICOR_ONTOLOGY_IRI,
    DUL_NAMESPACE,
    DUL_ONTOLOGY_IRI,
    ActionPerspectiveFile,
)
from llm_reasoner.knowledge.vocabulary_proposal.catalogue import TermCatalogue, TermKind
from llm_reasoner.knowledge.vocabulary_proposal.checks import (
    ProblemKind,
    ProposalChecker,
)
from llm_reasoner.knowledge.vocabulary_proposal.perspective import Perspective
from llm_reasoner.knowledge.vocabulary_proposal.prompts import (
    PROPOSAL_SCHEMA,
    ProposalPrompt,
    ProposalSection,
)
from llm_reasoner.knowledge.vocabulary_proposal.proposal import VocabularyProposal
from llm_reasoner.knowledge.vocabulary_proposal.proposer import (
    ProposalRejected,
    VocabularyProposer,
)
from llm_reasoner.knowledge.vocabulary_proposal.writing import ProposalWriter

from .prompt_sections import bodies

CASES = Path(__file__).parent / "vocabulary_proposal_cases"
"""
Where the answers and linked ontology excerpts of the tests are kept.
"""

LINKED_FILES = [CASES / "cram_aicor_terms.ttl", CASES / "dul_terms.owl"]
"""
Excerpts of CRAM-AICOR and DUL holding the terms the answers use.
"""

ACTION = "aicor:Pouring"
"""
The action the vocabulary is proposed for.
"""

FRAME = "core:PouringFrame"
"""
The action core's frame of that action.
"""

FRAME_PERSPECTIVE = Namespace(Perspective.ACTION_FRAME.namespace)
"""
The terms the frame perspective declares.
"""

CORE = Namespace(ActionPerspectiveFile.ACTION_CORE.namespace)
"""
The terms of the action core.
"""

AICOR = Namespace(CRAM_AICOR_NAMESPACE)
"""
The terms of CRAM-AICOR.
"""

DUL = Namespace(DUL_NAMESPACE)
"""
The terms of DOLCE+DnS Ultralite.
"""

# %% helpers


@dataclass
class RecordedConsistency(ConsistencyCheck):
    """
    A consistency check that answers as told and keeps the graphs it was asked about.
    """

    consistent: bool
    """
    What it answers.
    """

    asked: List[Graph] = field(default_factory=list)
    """
    The graphs it was asked about.
    """

    def is_consistent(self, graph: Graph) -> bool:
        self.asked.append(graph)
        return self.consistent


@pytest.fixture(scope="module")
def catalogue() -> TermCatalogue:
    """
    The terms a proposal may reuse: the action core's, and those of the excerpts.
    """
    return TermCatalogue.from_files(
        core=ActionPerspectiveFile.ACTION_CORE.path, linked=LINKED_FILES
    )


@pytest.fixture
def answer() -> Dict[str, Any]:
    """
    A valid answer, as the model writes it.
    """
    return json.loads((CASES / "valid_proposal.json").read_text())


@pytest.fixture
def concept_naming_property() -> Dict[str, Any]:
    """
    A property whose values name a CRAM-AICOR concept by its IRI, so it has no range.
    """
    return json.loads((CASES / "concept_naming_property.json").read_text())


def checker(catalogue: TermCatalogue, consistency=None) -> ProposalChecker:
    return ProposalChecker(
        catalogue=catalogue,
        perspective=Perspective.ACTION_FRAME,
        consistency=consistency,
    )


def problem_kinds(
    catalogue: TermCatalogue, answer: Dict[str, Any]
) -> List[ProblemKind]:
    return [
        problem.kind
        for problem in checker(catalogue).problems(VocabularyProposal.from_json(answer))
    ]


def written(answer: Dict[str, Any]) -> Graph:
    return ProposalWriter(
        proposal=VocabularyProposal.from_json(answer),
        perspective=Perspective.ACTION_FRAME,
    ).graph()


# %% the terms a proposal may reuse


class TestCatalogue:
    def test_the_cores_terms_are_named_with_its_prefix_by_their_kind(self, catalogue):
        assert catalogue.kind_of("core:SourceSlot") is TermKind.CLASS
        assert catalogue.kind_of("core:definesSlot") is TermKind.PROPERTY
        assert catalogue.kind_of("core:Amount") is TermKind.INDIVIDUAL

    def test_the_linked_terms_are_named_with_their_ontologys_prefix(self, catalogue):
        assert catalogue.kind_of("aicor:Container") is TermKind.CLASS
        assert catalogue.kind_of("aicor:contains") is TermKind.PROPERTY
        assert catalogue.kind_of("dul:Description") is TermKind.CLASS

    def test_a_term_no_ontology_declares_is_unknown(self, catalogue):
        assert catalogue.kind_of("aicor:Unicorn") is None
        assert catalogue.kind_of("pm:SourceContainer") is None

    def test_a_core_term_is_described_with_its_meaning_and_parents(self, catalogue):
        (source,) = [
            entry for entry in catalogue.core_terms() if entry.name == "core:SourceSlot"
        ]
        assert source.to_json()["meaning"].startswith("The slot of where something")
        assert set(source.to_json()["parents"]) == {
            "core:ParticipantSlot",
            "aicor:SourceRole",
        }

    def test_a_linked_term_is_described_with_its_meaning_when_it_has_one(
        self, catalogue
    ):
        (container,) = [
            entry
            for entry in catalogue.linked_terms()
            if entry.name == "aicor:Container"
        ]
        assert container.to_json()["meaning"] == (
            "A physical individual that can contain a substance."
        )


# %% reading an answer


class TestReading:
    def test_an_answer_is_read_into_classes_properties_and_partitions(self, answer):
        proposal = VocabularyProposal.from_json(answer)
        assert [proposed.name for proposed in proposal.classes] == [
            "FillerExpectation",
            "Precondition",
        ]
        assert len(proposal.properties) == len(answer["properties"])
        assert [member.name for member in proposal.value_partitions[0].members] == [
            "HasOpening",
            "HoldsSubstance",
            "StandsUpright",
        ]

    def test_a_property_without_a_range_is_read_without_one(
        self, answer, concept_naming_property
    ):
        answer["properties"].append(concept_naming_property)
        assert VocabularyProposal.from_json(answer).properties[-1].value_class is None


# %% the rules a proposal follows


class TestChecks:
    def test_a_proposal_that_follows_every_rule_has_no_problem(self, catalogue, answer):
        assert problem_kinds(catalogue, answer) == []

    def test_a_name_that_is_not_one_word_in_capitals_is_refused(
        self, catalogue, answer
    ):
        answer["classes"][1]["name"] = "pre condition"
        answer["properties"][2]["domain"] = "pre condition"
        answer["properties"][3]["range"] = "pre condition"
        assert ProblemKind.INVALID_NAME in problem_kinds(catalogue, answer)

    def test_a_property_named_in_capitals_is_refused(self, catalogue, answer):
        answer["properties"][3]["name"] = "HoldsBefore"
        assert ProblemKind.INVALID_NAME in problem_kinds(catalogue, answer)

    def test_a_term_no_ontology_declares_is_refused(self, catalogue, answer):
        answer["classes"][0]["parents"] = ["aicor:Unicorn"]
        assert ProblemKind.UNKNOWN_TERM in problem_kinds(catalogue, answer)

    def test_a_term_of_another_perspective_is_refused(self, catalogue, answer):
        answer["classes"][0]["parents"] = ["pm:SourceContainer"]
        assert ProblemKind.UNKNOWN_TERM in problem_kinds(catalogue, answer)

    def test_a_new_term_the_proposal_does_not_declare_is_refused(
        self, catalogue, answer
    ):
        answer["properties"][3]["range"] = "Postcondition"
        assert ProblemKind.UNKNOWN_TERM in problem_kinds(catalogue, answer)

    def test_a_term_named_like_one_that_exists_is_refused_for_reuse(
        self, catalogue, answer
    ):
        answer["classes"][1]["name"] = "SourceSlot"
        answer["properties"][2]["domain"] = "SourceSlot"
        answer["properties"][3]["range"] = "SourceSlot"
        assert ProblemKind.EXISTING_NAME in problem_kinds(catalogue, answer)

    def test_two_terms_of_one_name_are_refused(self, catalogue, answer):
        answer["classes"].append(copy.deepcopy(answer["classes"][1]))
        assert ProblemKind.DUPLICATE_NAME in problem_kinds(catalogue, answer)

    def test_a_term_without_a_meaning_is_refused(self, catalogue, answer):
        answer["value_partitions"][0]["members"][0]["meaning"] = " "
        assert ProblemKind.MISSING_MEANING in problem_kinds(catalogue, answer)

    def test_a_class_that_specialises_nothing_is_refused(self, catalogue, answer):
        answer["classes"][1]["parents"] = []
        assert ProblemKind.UNGROUNDED_CLASS in problem_kinds(catalogue, answer)

    def test_classes_that_only_specialise_each_other_are_refused(
        self, catalogue, answer
    ):
        answer["classes"][0]["parents"] = ["Precondition"]
        answer["classes"][1]["parents"] = ["FillerExpectation"]
        assert ProblemKind.UNGROUNDED_CLASS in problem_kinds(catalogue, answer)

    def test_a_vocabulary_that_never_refers_to_the_core_is_refused(
        self, catalogue, answer
    ):
        for proposed in answer["classes"]:
            proposed["restrictions"] = []
        answer["properties"] = [
            proposed
            for proposed in answer["properties"]
            if not any(
                end.startswith("core:")
                for end in (proposed["domain"], proposed["range"])
            )
        ]
        assert ProblemKind.NOT_ATTACHED in problem_kinds(catalogue, answer)

    def test_a_restriction_to_a_value_that_is_a_class_is_refused(
        self, catalogue, answer
    ):
        answer["classes"][0]["restrictions"][1] = {
            "property": "hasExpectedFeature",
            "quantifier": "value",
            "filler": "FillerFeature",
        }
        assert ProblemKind.WRONG_KIND in problem_kinds(catalogue, answer)

    def test_a_restriction_to_a_value_that_is_a_member_is_kept(self, catalogue, answer):
        answer["classes"][0]["restrictions"][1] = {
            "property": "hasExpectedFeature",
            "quantifier": "value",
            "filler": "HasOpening",
        }
        assert problem_kinds(catalogue, answer) == []

    def test_a_domain_that_is_a_property_is_refused(self, catalogue, answer):
        answer["properties"][3]["domain"] = "core:definesSlot"
        assert ProblemKind.WRONG_KIND in problem_kinds(catalogue, answer)

    def test_a_property_naming_a_concept_needs_no_range(
        self, catalogue, answer, concept_naming_property
    ):
        answer["properties"].append(concept_naming_property)
        assert problem_kinds(catalogue, answer) == []

    def test_a_vocabulary_the_generator_cannot_write_as_python_is_refused(
        self, catalogue, answer
    ):
        answer["classes"].append(
            {
                "name": "HoldsBefore",
                "meaning": "A class named like the descriptor of a property.",
                "parents": ["dul:Description"],
                "restrictions": [],
                "see_also": [],
            }
        )
        assert ProblemKind.PYTHON_GENERATION in problem_kinds(catalogue, answer)

    def test_a_vocabulary_the_reasoner_finds_inconsistent_is_refused(
        self, catalogue, answer
    ):
        consistency = RecordedConsistency(consistent=False)
        problems = checker(catalogue, consistency).problems(
            VocabularyProposal.from_json(answer)
        )
        assert [problem.kind for problem in problems] == [ProblemKind.INCONSISTENT]
        (asked,) = consistency.asked
        assert (FRAME_PERSPECTIVE.FillerExpectation, RDF.type, OWL.Class) in asked

    def test_the_reasoner_is_given_no_import_to_fetch_since_it_has_the_files(
        self, catalogue, answer
    ):
        consistency = RecordedConsistency(consistent=True)
        checker(catalogue, consistency).problems(VocabularyProposal.from_json(answer))
        (asked,) = consistency.asked
        assert list(asked.triples((None, OWL.imports, None))) == []

    def test_the_reasoner_is_not_asked_about_a_proposal_that_already_broke_a_rule(
        self, catalogue, answer
    ):
        answer["classes"][1]["parents"] = []
        consistency = RecordedConsistency(consistent=True)
        checker(catalogue, consistency).problems(VocabularyProposal.from_json(answer))
        assert consistency.asked == []


# %% writing the module


class TestWriting:
    def test_the_module_imports_the_core_cram_aicor_and_dul(self, answer):
        graph = written(answer)
        (ontology,) = graph.subjects(RDF.type, OWL.Ontology)
        assert str(ontology) == Perspective.ACTION_FRAME.namespace.rstrip("#")
        assert set(graph.objects(ontology, OWL.imports)) == {
            URIRef(ActionPerspectiveFile.ACTION_CORE.namespace.rstrip("#")),
            URIRef(CRAM_AICOR_ONTOLOGY_IRI),
            URIRef(DUL_ONTOLOGY_IRI),
        }

    def test_a_class_is_written_with_its_label_meaning_and_parents(self, answer):
        graph = written(answer)
        expectation = FRAME_PERSPECTIVE.FillerExpectation
        assert (expectation, RDF.type, OWL.Class) in graph
        assert graph.value(expectation, RDFS.label) == Literal(
            "filler expectation", lang="en"
        )
        assert graph.value(expectation, RDFS.comment) == Literal(
            answer["classes"][0]["meaning"], lang="en"
        )
        assert (expectation, RDFS.subClassOf, AICOR.Expectation) in graph

    def test_a_restriction_is_written_on_its_class(self, answer):
        graph = written(answer)
        restrictions = [
            parent
            for parent in graph.objects(
                FRAME_PERSPECTIVE.FillerExpectation, RDFS.subClassOf
            )
            if isinstance(parent, BNode)
        ]
        assert any(
            graph.value(restriction, OWL.onProperty) == FRAME_PERSPECTIVE.expectsOfSlot
            and graph.value(restriction, OWL.someValuesFrom) == CORE.FrameSlot
            for restriction in restrictions
        )

    def test_a_property_is_written_with_its_domain_range_and_whether_functional(
        self, answer
    ):
        graph = written(answer)
        expects = FRAME_PERSPECTIVE.expectsOfSlot
        assert (expects, RDF.type, OWL.ObjectProperty) in graph
        assert (expects, RDF.type, OWL.FunctionalProperty) in graph
        assert graph.value(expects, RDFS.domain) == FRAME_PERSPECTIVE.FillerExpectation
        assert graph.value(expects, RDFS.range) == CORE.FrameSlot
        assert (
            FRAME_PERSPECTIVE.holdsBefore,
            RDF.type,
            OWL.FunctionalProperty,
        ) not in graph

    def test_a_property_without_a_range_is_written_without_one(
        self, answer, concept_naming_property
    ):
        answer["properties"].append(concept_naming_property)
        graph = written(answer)
        expects_state = FRAME_PERSPECTIVE[concept_naming_property["name"]]
        assert (expects_state, RDF.type, OWL.ObjectProperty) in graph
        assert graph.value(expects_state, RDFS.range) is None

    def test_a_value_partition_lists_its_members(self, answer):
        graph = written(answer)
        feature = FRAME_PERSPECTIVE.FillerFeature
        (one_of,) = [
            graph.value(equivalent, OWL.oneOf)
            for equivalent in graph.objects(feature, OWL.equivalentClass)
        ]
        assert list(Collection(graph, one_of)) == [
            FRAME_PERSPECTIVE.HasOpening,
            FRAME_PERSPECTIVE.HoldsSubstance,
            FRAME_PERSPECTIVE.StandsUpright,
        ]
        assert (FRAME_PERSPECTIVE.HasOpening, RDF.type, feature) in graph
        assert (FRAME_PERSPECTIVE.HasOpening, RDF.type, OWL.NamedIndividual) in graph

    def test_a_cited_source_is_written_as_see_also(self, answer):
        graph = written(answer)
        assert graph.value(FRAME_PERSPECTIVE.FillerExpectation, RDFS.seeAlso) == URIRef(
            answer["classes"][0]["see_also"][0]
        )

    def test_the_turtle_reads_back_as_the_graph(self, answer):
        writer = ProposalWriter(
            proposal=VocabularyProposal.from_json(answer),
            perspective=Perspective.ACTION_FRAME,
        )
        assert len(Graph().parse(data=writer.turtle(), format="turtle")) == len(
            writer.graph()
        )


# %% the prompts


class TestPrompts:
    def test_the_system_prompt_states_the_role_the_inputs_and_the_rules(
        self, catalogue
    ):
        system = ProposalPrompt(catalogue=catalogue).system()
        for section in (
            ProposalSection.ROLE,
            ProposalSection.INPUTS,
            ProposalSection.RULES,
        ):
            assert bodies(system, section), section

    def test_the_request_gives_the_perspective_the_action_and_the_terms(
        self, catalogue
    ):
        request = ProposalPrompt(catalogue=catalogue).request(
            Perspective.ACTION_FRAME, ACTION, FRAME
        )
        (perspective,) = bodies(request, ProposalSection.PERSPECTIVE)
        assert perspective == Perspective.ACTION_FRAME.description
        (action,) = bodies(request, ProposalSection.ACTION)
        assert json.loads(action) == {"action": ACTION, "frame": FRAME}
        (core,) = bodies(request, ProposalSection.CORE)
        assert "core:PouringFrame" in {entry["name"] for entry in json.loads(core)}
        (linked,) = bodies(request, ProposalSection.LINKED_TERMS)
        assert "aicor:Container" in {entry["name"] for entry in json.loads(linked)}
        (output,) = bodies(request, ProposalSection.OUTPUT)
        assert json.loads(output) == PROPOSAL_SCHEMA


# %% proposing


def proposer(
    catalogue: TermCatalogue, answers: List[Dict[str, Any]], max_repairs: int = 2
) -> VocabularyProposer:
    return VocabularyProposer(
        client=ScriptedClient(responses=[json.dumps(given) for given in answers]),
        prompt=ProposalPrompt(catalogue=catalogue),
        checker=checker(catalogue),
        max_repairs=max_repairs,
    )


class TestProposing:
    def test_a_proposal_that_follows_every_rule_is_written_as_a_module(
        self, catalogue, answer
    ):
        outcome = proposer(catalogue, [answer]).propose(
            Perspective.ACTION_FRAME, ACTION, FRAME
        )
        assert len(outcome.exchanges) == 1
        assert (
            FRAME_PERSPECTIVE.FillerExpectation,
            RDF.type,
            OWL.Class,
        ) in Graph().parse(data=outcome.turtle, format="turtle")

    def test_the_model_is_asked_for_an_answer_shaped_by_the_proposal_schema(
        self, catalogue, answer
    ):
        given = proposer(catalogue, [answer])
        given.propose(Perspective.ACTION_FRAME, ACTION, FRAME)
        assert given.client.schemas == [PROPOSAL_SCHEMA]

    def test_the_proposal_schema_lets_a_property_name_a_concept_without_a_range(
        self, answer, concept_naming_property
    ):
        answer["properties"].append(concept_naming_property)
        assert list(Draft202012Validator(PROPOSAL_SCHEMA).iter_errors(answer)) == []

    def test_a_proposal_breaking_a_rule_is_returned_with_its_problems_for_repair(
        self, catalogue, answer
    ):
        broken = copy.deepcopy(answer)
        broken["classes"][1]["parents"] = []
        given = proposer(catalogue, [broken, answer])
        outcome = given.propose(Perspective.ACTION_FRAME, ACTION, FRAME)
        assert len(outcome.exchanges) == 2
        _, repair = given.client.calls[1]
        (problems,) = bodies(repair, ProposalSection.PROBLEMS)
        assert [problem["kind"] for problem in json.loads(problems)] == [
            ProblemKind.UNGROUNDED_CLASS
        ]
        (rejected,) = bodies(repair, ProposalSection.REJECTED_ANSWER)
        assert json.loads(rejected) == broken

    def test_a_proposal_never_following_the_rules_is_rejected_with_its_problems(
        self, catalogue, answer
    ):
        broken = copy.deepcopy(answer)
        broken["classes"][1]["parents"] = []
        with pytest.raises(ProposalRejected) as rejected:
            proposer(catalogue, [broken, broken], max_repairs=1).propose(
                Perspective.ACTION_FRAME, ACTION, FRAME
            )
        assert len(rejected.value.exchanges) == 2
        assert [problem.kind for problem in rejected.value.problems] == [
            ProblemKind.UNGROUNDED_CLASS
        ]
