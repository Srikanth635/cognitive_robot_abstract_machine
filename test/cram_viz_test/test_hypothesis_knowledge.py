"""
Tests for the checked-in sample hypothesis and for the knowledge the viewer displays and
queries.
"""

import pytest

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from llm_reasoner.knowledge.acceptance.rules import Severity  # noqa: E402
from llm_reasoner.knowledge.acceptance.suite import (  # noqa: E402
    predictive_manipulation_suite,
)
from llm_reasoner.knowledge.admission.rdf import GraphReader  # noqa: E402
from llm_reasoner.knowledge.storage.store import HypothesisStore  # noqa: E402
from llm_reasoner.hypotheses.graph import HypothesisGraphNodeKind  # noqa: E402
from llm_reasoner.knowledge.ontologies.predictive_manipulation import (  # noqa: E402
    pouring,
)

from cram_viz import hypothesis_knowledge, paths  # noqa: E402

# %% sample


class TestSampleHypothesis:
    """
    The sample is a stored hypothesis, read the same way as a generated one.
    """

    def test_every_statement_of_the_sample_is_read(self):
        stored = HypothesisStore(directory=paths.SAMPLE_HYPOTHESES_PATH).get(
            hypothesis_knowledge.SAMPLE_RECORD_ID
        )
        assert GraphReader().read(stored.graph).unread_statements == []

    def test_sample_meets_every_acceptance_rule(self):
        errors = [
            violation
            for violation in predictive_manipulation_suite().violations(
                hypothesis_knowledge.sample_view().fragments
            )
            if violation.severity is Severity.ERROR
        ]
        assert errors == []

    def test_sample_root_is_a_pouring_structure(self):
        assert type(hypothesis_knowledge.sample_view().root) is (
            pouring.PouringPhaseStructure
        )


# %% active knowledge


@pytest.mark.usefixtures("sample_hypothesis_knowledge")
class TestActiveKnowledge:
    def test_projection_root_is_the_active_structure(self):
        knowledge = hypothesis_knowledge.active()
        assert knowledge.projection.root.source is knowledge.view.root
        assert knowledge.projection.root.kind is HypothesisGraphNodeKind.ROOT

    def test_projection_centre_is_the_recorded_instruction(self):
        knowledge = hypothesis_knowledge.active()
        assert knowledge.projection.instruction.label == (
            knowledge.view.record.instruction
        )

    def test_node_identifier_of_a_fragment_is_its_citation(self):
        knowledge = hypothesis_knowledge.active()
        phase = knowledge.view.root.defines_phase[0]
        assert hypothesis_knowledge.node_identifier_for(phase) == (
            knowledge.view.citation_of(phase).iri
        )

    def test_node_identifier_lookup_returns_none_for_an_unrelated_object(self):
        assert hypothesis_knowledge.node_identifier_for(object()) is None

    def test_sample_is_not_reported_as_generated(self):
        assert hypothesis_knowledge.active().generated is False


class TestNothingActiveUntilGenerated:
    def test_no_knowledge_is_active_before_an_instruction_is_generated(self):
        assert hypothesis_knowledge.active() is None

    def test_node_identifier_lookup_is_answerable_with_nothing_active(self):
        assert hypothesis_knowledge.node_identifier_for(object()) is None

    def test_activating_the_sample_makes_it_the_active_knowledge(
        self, sample_hypothesis_knowledge
    ):
        assert hypothesis_knowledge.active() is sample_hypothesis_knowledge
