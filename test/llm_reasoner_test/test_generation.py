"""
Tests of generating hypotheses on the vocabulary generated from the ontology: the
schema, the decoding of answers, and the backend's loop of asking, checking and
repairing, with scripted model answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pytest
from krrood.entity_query_language.factories import the
from rdflib import Graph

from llm_reasoner.acceptance.core import CoreRule, TransitionIsNotALoop
from llm_reasoner.admission.consistency import ConsistencyCheck, OpenWorldCheck
from llm_reasoner.admission.record import ConsistencyVerdict
from llm_reasoner.admission.store import HypothesisStore
from llm_reasoner.acceptance.rules import AcceptanceSuite
from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.generation.backend import HypothesisBackend, HypothesisRejected
from llm_reasoner.generation.schema import FieldKind, GenerationSchema
from llm_reasoner.generation.wire import FragmentDecoder
from llm_reasoner.ontologies import CRAM_AICOR_NAMESPACE
from llm_reasoner.ontologies.predictive_manipulation import core, pouring
from llm_reasoner.ontologies.terms import ConceptReference

POURING = ConceptReference(iri=CRAM_AICOR_NAMESPACE + "Pouring")
"""
The CRAM-AICOR pouring task the scripted structures decompose.
"""


class GenerationCase(StrEnum):
    """
    Scripted answers of a language model.
    """

    LOOPING_STRUCTURE = "looping_structure.json"
    """
    A pouring structure whose only transition leads from a phase back to itself.
    """

    CHAINED_STRUCTURE = "chained_structure.json"
    """
    A pouring structure whose transition leads from tilting to regulating the flow.
    """

    STRUCTURE_OF_ANOTHER_TASK = "structure_of_another_task.json"
    """
    A structure decomposing cutting rather than pouring.
    """

    UNKNOWN_TYPE = "unknown_type.json"
    """
    An answer using a type the vocabulary does not have.
    """

    DANGLING_REFERENCE = "dangling_reference.json"
    """
    An answer referring to an identifier no fragment has.
    """

    REFERENCE_OF_WRONG_TYPE = "reference_of_wrong_type.json"
    """
    An answer linking a field to a fragment of a type the field does not allow.
    """

    GRIP_DRIVEN_LIFTOFF = "grip_driven_liftoff.json"
    """
    A pouring structure claiming lift-off occurs when grip force overcomes the weight.
    """

    LIFT_AND_POUR = "lift_and_pour.json"
    """
    A pouring structure that loads, lifts and tilts a carton, with the forces at lift-
    off, a control point there, a gaze fixation and a grip-load coupling.
    """

    @property
    def text(self) -> str:
        """
        The scripted answer.
        """
        return (Path(__file__).parent / "generation_cases" / self.value).read_text()


# %% helpers


def structure_decoder() -> FragmentDecoder:
    """
    Return a decoder for answers building a phase structure.
    """
    return FragmentDecoder(GenerationSchema.for_root(core.ActionPhaseStructure))


def backend_answering(*cases: GenerationCase) -> HypothesisBackend:
    """
    Return a backend whose model gives the scripted answers in turn, checked only for
    looping transitions.
    """
    return HypothesisBackend(
        client=ScriptedClient(responses=[case.text for case in cases]),
        instruction="pour milk from the carton into the bowl",
        suite=AcceptanceSuite(rules=[TransitionIsNotALoop()]),
    )


def pouring_query() -> object:
    """
    Return a query for one phase structure of the pouring task, all else open.
    """
    return the(core.ActionPhaseStructure)(decomposes_task=POURING)


# %% schema


class TestGenerationSchema:
    """
    The output shape is read from the generated vocabulary.
    """

    def test_types_reachable_from_the_root_are_usable(self):
        schema = GenerationSchema.for_root(core.ActionPhaseStructure)
        assert {"ActionPhase", "TiltToFlowPhase", "PhaseTransition"} <= set(
            schema.types
        )

    def test_fragment_field_accepts_every_specialisation_of_its_type(self):
        schema = GenerationSchema.for_root(core.ActionPhaseStructure)
        defines_phase = schema.get("ActionPhaseStructure").field_named("defines_phase")
        assert (defines_phase.kind, defines_phase.many) == (FieldKind.FRAGMENT, True)
        assert pouring.TiltToFlowPhase.__name__ in defines_phase.accepted_types

    def test_enumeration_field_offers_its_members(self):
        schema = GenerationSchema.for_root(core.ActionPhaseStructure)
        regime = schema.get("ActionPhase").field_named("has_coupling_regime")
        assert regime.options == tuple(member.name for member in core.CouplingRegime)

    def test_inferred_super_property_is_not_asked_for(self):
        schema = GenerationSchema.for_root(core.ActionPhaseStructure)
        assert (
            schema.get("ActionPhaseStructure").field_named("defines_component") is None
        )


# %% decoding


class TestFragmentDecoder:
    """
    Answers are decoded into linked vocabulary instances, or into their problems.
    """

    def test_answer_is_decoded_into_linked_instances(self):
        hypothesis = (
            structure_decoder().decode(GenerationCase.CHAINED_STRUCTURE.text).hypothesis
        )
        (transition,) = hypothesis.root.defines_transition
        assert isinstance(transition.from_phase, pouring.TiltToFlowPhase)
        assert transition.from_phase in hypothesis.root.defines_phase

    def test_prefixed_concept_is_expanded(self):
        hypothesis = (
            structure_decoder().decode(GenerationCase.CHAINED_STRUCTURE.text).hypothesis
        )
        assert hypothesis.root.decomposes_task == POURING

    def test_enumeration_member_is_read_by_name(self):
        hypothesis = (
            structure_decoder().decode(GenerationCase.CHAINED_STRUCTURE.text).hypothesis
        )
        (tilt,) = [
            phase
            for phase in hypothesis.root.defines_phase
            if isinstance(phase, pouring.TiltToFlowPhase)
        ]
        assert tilt.has_coupling_regime == core.CouplingRegime.COUPLED

    def test_answer_identifiers_are_kept(self):
        hypothesis = (
            structure_decoder().decode(GenerationCase.CHAINED_STRUCTURE.text).hypothesis
        )
        assert hypothesis.identifier_of(hypothesis.root) == "pour"

    @pytest.mark.parametrize(
        "case, named",
        [
            (GenerationCase.UNKNOWN_TYPE, "StirringPhase"),
            (GenerationCase.DANGLING_REFERENCE, "missing_phase"),
            (GenerationCase.REFERENCE_OF_WRONG_TYPE, "Subgoal"),
        ],
    )
    def test_malformed_answer_yields_problems_naming_the_fault(self, case, named):
        decoded = structure_decoder().decode(case.text)
        assert decoded.hypothesis is None
        assert any(named in problem for problem in decoded.problems)


# %% backend


class TestHypothesisBackend:
    """
    The backend asks, checks, repairs and yields the admitted hypothesis.
    """

    def test_admitted_answer_is_the_result(self):
        backend = backend_answering(GenerationCase.CHAINED_STRUCTURE)
        (structure,) = pouring_query().evaluate(backend=backend)
        assert isinstance(structure, core.ActionPhaseStructure)
        assert len(backend.log) == 1

    def test_rejected_answer_is_returned_with_the_violation_to_repair(self):
        backend = backend_answering(
            GenerationCase.LOOPING_STRUCTURE, GenerationCase.CHAINED_STRUCTURE
        )
        (structure,) = pouring_query().evaluate(backend=backend)
        repair_prompt = backend.log[1].user
        assert str(CoreRule.TRANSITION_IS_NOT_A_LOOP) in repair_prompt
        assert "'back_to_tilt'" in repair_prompt
        assert len(structure.defines_phase) == 2

    def test_answer_contradicting_a_fixed_value_is_repaired(self):
        backend = backend_answering(
            GenerationCase.STRUCTURE_OF_ANOTHER_TASK, GenerationCase.CHAINED_STRUCTURE
        )
        (structure,) = pouring_query().evaluate(backend=backend)
        assert "decomposes_task = aicor:Pouring" in backend.log[1].user
        assert structure.decomposes_task == POURING

    def test_fixed_values_and_instruction_are_stated(self):
        backend = backend_answering(GenerationCase.CHAINED_STRUCTURE)
        list(pouring_query().evaluate(backend=backend))
        prompt = backend.log[0].user
        assert backend.instruction in prompt
        assert "decomposes_task = aicor:Pouring" in prompt

    def test_no_admissible_answer_within_the_budget_is_refused(self):
        backend = backend_answering(*[GenerationCase.LOOPING_STRUCTURE] * 3)
        with pytest.raises(HypothesisRejected) as refusal:
            list(pouring_query().evaluate(backend=backend))
        assert len(refusal.value.exchanges) == backend.max_repairs + 1


# %% admission


@dataclass
class ConflictWith(ConsistencyCheck):
    """
    A stand-in reasoner that finds a hypothesis inconsistent exactly when it states
    something about a fragment of the given identifier.
    """

    identifier: str
    """
    The identifier of the fragment that contradicts the ontology.
    """

    def is_consistent(self, graph: Graph) -> bool:
        return not any(
            str(subject).endswith("#" + self.identifier) for subject in graph.subjects()
        )


class TestAdmission:
    """
    Admitted hypotheses are checked in the open world, recorded and stored.
    """

    def test_open_world_contradiction_is_returned_with_its_fragments(self):
        backend = backend_answering(
            GenerationCase.LOOPING_STRUCTURE, GenerationCase.CHAINED_STRUCTURE
        )
        backend.suite = AcceptanceSuite(rules=[])
        backend.consistency = ConflictWith(identifier="back_to_tilt")
        list(pouring_query().evaluate(backend=backend))
        repair_prompt = backend.log[1].user
        assert str(OpenWorldCheck.CONSISTENT_WITH_ONTOLOGY) in repair_prompt
        assert "'back_to_tilt'" in repair_prompt

    def test_checked_hypothesis_is_recorded_as_consistent(self):
        backend = backend_answering(GenerationCase.CHAINED_STRUCTURE)
        backend.consistency = ConflictWith(identifier="absent")
        list(pouring_query().evaluate(backend=backend))
        assert backend.last_record.consistency == ConsistencyVerdict.CONSISTENT

    def test_admitted_hypothesis_is_served_again_without_the_model(self, tmp_path):
        backend = backend_answering(GenerationCase.CHAINED_STRUCTURE)
        backend.store = HypothesisStore(directory=tmp_path)
        list(pouring_query().evaluate(backend=backend))
        (served,) = pouring_query().evaluate(backend=backend)
        assert len(backend.client.calls) == 1
        assert len(served.defines_phase) == 2
