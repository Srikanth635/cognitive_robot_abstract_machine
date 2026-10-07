"""
Tests for turning a natural-language instruction into the knowledge the viewer displays
and queries.

Generation, admission and storage belong to ``llm_reasoner`` and are tested there; what
matters here is how this package asks for a hypothesis and makes it the active
knowledge.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from llm_reasoner.acceptance.rules import AcceptanceSuite
from llm_reasoner.admission.hermit import HermitCheck
from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.generation.backend import HypothesisRejected
from llm_reasoner.hypotheses.view import HypothesisView
from llm_reasoner.ontologies import CRAM_AICOR_NAMESPACE, LinkedOntologyVariable
from llm_reasoner.ontologies.predictive_manipulation import pouring

from cram_viz import hypothesis_knowledge
from cram_viz.hypothesis_knowledge.generation import (
    GENERATION_MODEL,
    HypothesisGenerationService,
    HypothesisSource,
    MissingCredentialsError,
    ModelHypothesisSource,
)

INSTRUCTION = "pour milk from the carton into the bowl"
"""
The instruction the tests generate for.
"""

LINKED_EXCERPTS = [
    Path(__file__).parents[1] / "llm_reasoner_test" / "linked_ontology_cases" / name
    for name in ("cram_aicor_excerpt.ttl", "dul_excerpt.owl")
]
"""
Excerpts of CRAM-AICOR and DUL declaring a few terms.
"""

SCRIPTED_ANSWER = (
    Path(__file__).parents[1]
    / "llm_reasoner_test"
    / "generation_cases"
    / "lift_and_pour.json"
)
"""
A scripted model answer describing a pouring structure.
"""


@dataclass
class RecordingHypothesisSource(HypothesisSource):
    """
    A source answering every instruction with the sample, remembering what it was asked.
    """

    requested_instructions: list[str] = field(default_factory=list)
    """
    Every instruction the source was asked for.
    """

    def view_for(self, instruction: str) -> HypothesisView:
        self.requested_instructions.append(instruction)
        return hypothesis_knowledge.sample_view()


@dataclass
class RejectingHypothesisSource(HypothesisSource):
    """
    A source whose generation is never admitted.
    """

    def view_for(self, instruction: str) -> HypothesisView:
        raise HypothesisRejected(problems=["rejected"], exchanges=[])


@pytest.fixture
def restore_active_knowledge():
    """
    Put the active knowledge back after a test swaps it.
    """
    previous = hypothesis_knowledge.active()
    yield
    hypothesis_knowledge.activate(previous)


def scripted_source(tmp_path: Path) -> ModelHypothesisSource:
    """
    Return a source whose model gives one scripted answer, admitted without rules or an
    open-world check.
    """
    return ModelHypothesisSource(
        store_directory=tmp_path,
        client=ScriptedClient(responses=[SCRIPTED_ANSWER.read_text()]),
        suite=AcceptanceSuite(rules=[]),
        consistency=None,
    )


# %% instruction -> knowledge


class TestHypothesisGenerationService:
    def test_generated_knowledge_becomes_active(self, restore_active_knowledge):
        knowledge = HypothesisGenerationService(
            source=RecordingHypothesisSource()
        ).generate(INSTRUCTION)
        assert hypothesis_knowledge.active() is knowledge
        assert knowledge.generated is True

    def test_instruction_is_passed_to_the_source(self, restore_active_knowledge):
        source = RecordingHypothesisSource()
        HypothesisGenerationService(source=source).generate(INSTRUCTION)
        assert source.requested_instructions == [INSTRUCTION]

    def test_rejected_generation_keeps_the_active_knowledge(
        self, restore_active_knowledge, sample_hypothesis_knowledge
    ):
        with pytest.raises(HypothesisRejected):
            HypothesisGenerationService(source=RejectingHypothesisSource()).generate(
                INSTRUCTION
            )
        assert hypothesis_knowledge.active() is sample_hypothesis_knowledge


# %% generating with a model


class TestModelHypothesisSource:
    def test_answer_becomes_a_pouring_structure(self, tmp_path):
        view = scripted_source(tmp_path).view_for(INSTRUCTION)
        assert type(view.root) is pouring.PouringPhaseStructure
        assert view.record.instruction == INSTRUCTION

    def test_repeated_instruction_is_answered_from_the_store(self, tmp_path):
        source = scripted_source(tmp_path)
        first = source.view_for(INSTRUCTION)
        second = source.view_for(INSTRUCTION)
        assert second.record == first.record

    def test_default_client_uses_the_generation_model(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        source = ModelHypothesisSource(store_directory=tmp_path)
        assert source.client_for_generation().model == GENERATION_MODEL

    def test_open_world_check_is_configured_from_the_environment(
        self, monkeypatch, tmp_path
    ):
        for variable, path in zip(LinkedOntologyVariable, LINKED_EXCERPTS):
            monkeypatch.setenv(variable, str(path))
        source = ModelHypothesisSource(
            store_directory=tmp_path, client=ScriptedClient()
        )
        assert isinstance(source.consistency, HermitCheck)

    def test_linked_terms_are_read_from_the_environment(self, monkeypatch, tmp_path):
        for variable, path in zip(LinkedOntologyVariable, LINKED_EXCERPTS):
            monkeypatch.setenv(variable, str(path))
        source = ModelHypothesisSource(
            store_directory=tmp_path, client=ScriptedClient(), consistency=None
        )
        assert source.linked_terms.declares(CRAM_AICOR_NAMESPACE + "Grasping")


class TestModelHypothesisSourceCredentials:
    def test_a_missing_key_is_reported_as_a_named_configuration_problem(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        source = ModelHypothesisSource(
            store_directory=tmp_path, environment_file=tmp_path / "absent.env"
        )
        with pytest.raises(MissingCredentialsError):
            source.view_for(INSTRUCTION)

    def test_the_key_is_read_from_an_environment_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        environment_file = tmp_path / ".env"
        environment_file.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n")
        ModelHypothesisSource(
            store_directory=tmp_path, environment_file=environment_file
        ).load_credentials()
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-file"

    def test_an_already_set_key_is_not_overwritten_by_the_file(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
        environment_file = tmp_path / ".env"
        environment_file.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n")
        ModelHypothesisSource(
            store_directory=tmp_path, environment_file=environment_file
        ).load_credentials()
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-shell"

    def test_an_injected_client_needs_no_credentials(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        ModelHypothesisSource(
            store_directory=tmp_path,
            client=ScriptedClient(),
            environment_file=tmp_path / "absent.env",
        ).load_credentials()


# %% live generation


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="live generation needs ANTHROPIC_API_KEY",
)
class TestLiveGeneration:
    def test_a_real_generation_produces_a_queryable_structure(
        self, restore_active_knowledge, tmp_path
    ):
        knowledge = HypothesisGenerationService(
            source=ModelHypothesisSource(store_directory=tmp_path)
        ).generate(INSTRUCTION)
        assert knowledge.view.root.defines_phase
        assert knowledge.view.root.defines_role
