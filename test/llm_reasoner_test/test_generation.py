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
from rdflib import Graph, Namespace

from llm_reasoner.knowledge.acceptance.derived import RestrictionsHold, described
from llm_reasoner.knowledge.acceptance.rules import AcceptanceSuite
from llm_reasoner.knowledge.acceptance.shapes import ShapesHold, StructureShapes
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary
from llm_reasoner.knowledge.admission.consistency import (
    ConsistencyCheck,
    OpenWorldCheck,
)
from llm_reasoner.knowledge.storage.record import ConsistencyVerdict, qualified_name
from llm_reasoner.knowledge.storage.store import HypothesisStore
from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.knowledge.generation.backend import HypothesisBackend
from llm_reasoner.knowledge.generation.prompts import (
    GenerationPrompt,
    described_violation,
    field_hint,
    rendered_shape,
    rendered_type,
    written_value,
)
from llm_reasoner.knowledge.generation.schema import (
    FieldKind,
    GenerationSchema,
    type_schema_for,
)
from llm_reasoner.knowledge.generation.assessment import HypothesisAssessor
from llm_reasoner.knowledge.generation.dialogue import (
    GenerationDialogue,
    GenerationOutcome,
    HypothesisRejected,
)
from llm_reasoner.knowledge.generation.recording import HypothesisRecorder
from llm_reasoner.knowledge.generation.request import (
    HypothesisRequest,
    UnsupportedFixedValue,
)
from llm_reasoner.knowledge.generation.wire import DecodingResult, FragmentDecoder
from llm_reasoner.knowledge.representation.hypothesis import DecodedHypothesis
from llm_reasoner.knowledge.ontologies import (
    CRAM_AICOR_NAMESPACE,
    HYPOTHESIS_CONCEPT_NAMESPACE,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.knowledge.ontologies.linked import LinkedOntologyTerms, OntologyFiles
from llm_reasoner.knowledge.ontologies.predictive_manipulation import (
    core,
    gaze,
    pouring,
    prehension,
)
from llm_reasoner.knowledge.ontologies.terms import ConceptReference

LINKED_EXCERPTS = OntologyFiles(
    paths=[
        Path(__file__).parent / "linked_ontology_cases" / "cram_aicor_excerpt.ttl",
        Path(__file__).parent / "linked_ontology_cases" / "dul_excerpt.owl",
    ]
)
"""
Excerpts of CRAM-AICOR and DUL declaring a few terms.
"""

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

    STRINGIFIED_FRAGMENTS = "stringified_fragments.json"
    """
    The chained pouring structure with its fragment list written as one JSON string
    followed by a stray closing brace, as a model answering through a forced tool call
    sometimes writes it.
    """

    UNDECLARED_COMPONENTS = "undeclared_components.json"
    """
    A structure listing only some of its components, one of them never written, with a
    comparison and a continuous prediction whose tolerance is a magnitude range.
    """

    CLASS_ENTAILED_VALUES = "class_entailed_values.json"
    """
    An alignment phase and a tactile prediction stating none of the values their classes
    entail.
    """

    LOOPING_WITH_DANGLING_REFERENCE = "looping_with_dangling_reference.json"
    """
    A structure whose one transition loops back to its phase and whose other leads to a
    phase no fragment is.
    """

    FIELD_OF_ANOTHER_TYPE = "field_of_another_type.json"
    """
    A phase stating the regulation mode, a field of the phase's controller.
    """

    LOOP_REPLACED_PATCH = "loop_replaced_patch.json"
    """
    A repair of the looping structure removing its looping transition and leading from
    tilting to a new flow regulation phase instead.
    """

    LOOP_AND_REFERENCE_REPAIRED_PATCH = "loop_and_reference_repaired_patch.json"
    """
    A repair of the looping structure with a dangling reference, removing the loop and
    leading its other transition to a new phase.
    """

    TASK_CORRECTED_PATCH = "task_corrected_patch.json"
    """
    A repair of the structure of another task, setting the pouring task.
    """

    RELABELLED_AND_EXTENDED_PATCH = "relabelled_and_extended_patch.json"
    """
    A patch of the chained structure relabelling a phase, adding one and removing the
    transition.
    """

    LIFTOFF_DRIVEN_BY_LOAD_PATCH = "liftoff_driven_by_load_patch.json"
    """
    A repair of the grip-driven lift-off making the driving force the load force.
    """

    CONCEPTS_LINKED_AND_LOCAL = "concepts_linked_and_local.json"
    """
    Categorical values naming a CRAM-AICOR term that does not exist, a concept of the
    hypothesis itself, and a misspelt CRAM-AICOR term.
    """

    LIFT_AND_POUR = "lift_and_pour.json"
    """
    A pouring structure that loads, lifts and tilts a carton, with the forces at lift-
    off, a control point there, a gaze fixation and a grip-load coupling.
    """

    GRANULAR_POURING = "granular_pouring.json"
    """
    A pouring structure for a substance flowing as grains, and nothing else.
    """

    LIQUID_POURING = "liquid_pouring.json"
    """
    A pouring structure for a liquid, and nothing else.
    """

    @property
    def text(self) -> str:
        """
        The scripted answer.
        """
        return (Path(__file__).parent / "generation_cases" / self.value).read_text()


# %% helpers

CORE_SHAPES = Namespace(PredictiveManipulationOntologyFile.CORE.shapes_namespace)
"""
The namespace of the core module's structural rules.
"""


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
        suite=AcceptanceSuite(
            rules=[
                ShapesHold(checked_shapes=frozenset({CORE_SHAPES.TransitionIsNotALoop}))
            ]
        ),
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
        to_phase = schema.get("PhaseTransition").field_named("to_phase")
        assert (to_phase.kind, to_phase.many) == (FieldKind.FRAGMENT, False)
        assert pouring.TiltToFlowPhase.__name__ in to_phase.accepted_types

    def test_enumeration_field_offers_its_members(self):
        schema = GenerationSchema.for_root(core.ActionPhaseStructure)
        regime = schema.get("ActionPhase").field_named("has_coupling_regime")
        assert regime.options == tuple(member.name for member in core.CouplingRegime)

    def test_component_field_typed_by_an_earlier_superclass_declares_its_range(self):
        component_fields = Vocabulary().component_fields(core.ActionPhaseStructure)
        assert component_fields["defines_gaze_fixation"] is gaze.GazeFixation

    def test_answer_may_name_fragments_to_remove(self):
        schema = GenerationSchema.for_root(core.ActionPhaseStructure).json_schema()
        assert schema["properties"]["remove"] == {
            "type": "array",
            "items": {"type": "string"},
        }
        assert "remove" not in schema["required"]

    def test_component_lists_of_the_structure_are_not_asked_for(self):
        structure = GenerationSchema.for_root(core.ActionPhaseStructure).get(
            "ActionPhaseStructure"
        )
        assert structure.field_named("defines_phase") is None
        assert structure.field_named("defines_gaze_fixation") is None

    def test_type_is_shown_with_what_it_requires(self):
        rendered = rendered_type(type_schema_for(pouring.PouringPhaseStructure))
        assert all(
            described(restriction, False) in rendered
            for restriction in pouring.PouringPhaseStructure.term.restrictions
        )

    def test_defined_type_is_shown_with_its_definition(self):
        rendered = rendered_type(type_schema_for(core.ForceRateParameter))
        assert described(core.ForceRateParameter.term.definition, False) in rendered

    def test_structural_rules_of_the_generated_types_are_in_the_system_prompt(self):
        system = GenerationPrompt().system(
            GenerationSchema.for_root(pouring.PouringPhaseStructure)
        )
        assert all(
            rendered_shape(shape) in system
            for shape in StructureShapes.shipped().shapes
        )

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

    def test_fragment_list_written_as_a_string_is_read(self):
        hypothesis = (
            structure_decoder()
            .decode(GenerationCase.STRINGIFIED_FRAGMENTS.text)
            .hypothesis
        )
        (transition,) = hypothesis.root.defines_transition
        assert isinstance(transition.from_phase, pouring.TiltToFlowPhase)
        assert hypothesis.identifier_of(hypothesis.root) == "pour"

    def test_every_component_written_is_declared_by_the_structure(self):
        hypothesis = (
            structure_decoder()
            .decode(GenerationCase.UNDECLARED_COMPONENTS.text)
            .hypothesis
        )
        root = hypothesis.root
        assert sorted(map(hypothesis.identifier_of, root.defines_phase)) == [
            "regulate",
            "tilt",
        ]
        assert list(map(hypothesis.identifier_of, root.defines_transition)) == [
            "tilt_to_regulate"
        ]
        assert list(map(hypothesis.identifier_of, root.defines_comparison)) == [
            "mismatch"
        ]
        assert list(
            map(hypothesis.identifier_of, root.defines_continuous_prediction)
        ) == ["flow_rate"]

    def test_value_no_component_field_accepts_is_not_declared(self):
        hypothesis = (
            structure_decoder()
            .decode(GenerationCase.UNDECLARED_COMPONENTS.text)
            .hypothesis
        )
        (tolerance,) = [
            fragment
            for fragment in hypothesis.fragments
            if isinstance(fragment, core.MagnitudeRange)
        ]
        assert not any(
            tolerance is component for component in hypothesis.root.defines_component
        )

    def test_values_the_class_entails_are_filled_in(self):
        decoder = FragmentDecoder(
            GenerationSchema.for_root(pouring.PouringPhaseStructure)
        )
        hypothesis = decoder.decode(
            GenerationCase.CLASS_ENTAILED_VALUES.text
        ).hypothesis
        (alignment,) = [
            fragment
            for fragment in hypothesis.fragments
            if isinstance(fragment, pouring.AlignmentPhase)
        ]
        (contact,) = [
            fragment
            for fragment in hypothesis.fragments
            if isinstance(fragment, core.TactilePrediction)
        ]
        assert alignment.has_coupling_regime is core.CouplingRegime.COUPLED
        assert contact.has_modality is core.SensoryModality.TACTILE

    def test_patch_replaces_adds_and_removes_fragments_by_identifier(self):
        decoder = structure_decoder()
        merged = decoder.merged(
            GenerationCase.CHAINED_STRUCTURE.text,
            GenerationCase.RELABELLED_AND_EXTENDED_PATCH.text,
        )
        hypothesis = decoder.decode(merged).hypothesis
        labels = {
            hypothesis.identifier_of(phase): phase.label
            for phase in hypothesis.root.defines_phase
        }
        assert labels == {"tilt": "", "regulate": "regulate the flow", "stop": ""}
        assert hypothesis.root.defines_transition == []

    def test_undecodable_answer_keeps_what_could_be_linked(self):
        decoded = structure_decoder().decode(GenerationCase.DANGLING_REFERENCE.text)
        assert decoded.hypothesis is None
        (transition,) = decoded.partial.root.defines_transition
        assert decoded.partial.identifier_of(transition) == "step"
        assert transition.from_phase is None

    def test_field_of_another_type_names_the_types_declaring_it(self):
        (problem,) = (
            structure_decoder()
            .decode(GenerationCase.FIELD_OF_ANOTHER_TYPE.text)
            .problems
        )
        assert core.ActionPhaseController.__name__ in problem

    def test_reference_of_wrong_type_repeats_the_fields_meaning(self):
        decoder = structure_decoder()
        description = (
            decoder.schema.get("PhaseTransition").field_named("from_phase").description
        )
        (problem,) = decoder.decode(
            GenerationCase.REFERENCE_OF_WRONG_TYPE.text
        ).problems
        assert description in problem

    def test_linked_concept_the_ontologies_lack_is_a_problem(self):
        decoder = FragmentDecoder(
            GenerationSchema.for_root(core.ActionPhaseStructure),
            linked_terms=LinkedOntologyTerms.from_files(LINKED_EXCERPTS),
        )
        problems = decoder.decode(
            GenerationCase.CONCEPTS_LINKED_AND_LOCAL.text
        ).problems
        assert [problem for problem in problems if "EffectorAtGraspSite" in problem]
        assert [problem for problem in problems if "aicor:Grasping" in problem]
        assert not [
            problem for problem in problems if "SideGraspAtMidHeight" in problem
        ]

    def test_concept_of_the_hypothesis_is_read_into_its_own_namespace(self):
        decoder = FragmentDecoder(
            GenerationSchema.for_root(core.ActionPhaseStructure),
            linked_terms=LinkedOntologyTerms.from_files(LINKED_EXCERPTS),
        )
        partial = decoder.decode(GenerationCase.CONCEPTS_LINKED_AND_LOCAL.text).partial
        (side,) = [
            fragment
            for fragment in partial.fragments
            if partial.identifier_of(fragment) == "side_category"
        ]
        assert side.has_category == ConceptReference(
            iri=HYPOTHESIS_CONCEPT_NAMESPACE + "SideGraspAtMidHeight"
        )

    def test_linked_concepts_are_not_checked_without_the_ontologies(self):
        decoded = structure_decoder().decode(
            GenerationCase.CONCEPTS_LINKED_AND_LOCAL.text
        )
        assert decoded.problems == []

    def test_reference_of_wrong_type_names_the_accepted_types(self):
        decoder = structure_decoder()
        accepted = (
            decoder.schema.get("PhaseTransition")
            .field_named("from_phase")
            .accepted_types
        )
        (problem,) = decoder.decode(
            GenerationCase.REFERENCE_OF_WRONG_TYPE.text
        ).problems
        assert all(type_name in problem for type_name in accepted)

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


class TestHypothesisRequest:
    """
    What a query asks to be generated.
    """

    def test_the_values_a_query_fixes_are_read_and_open_ones_left_out(self):
        query = the(core.ActionPhaseStructure)(decomposes_task=POURING, label=...)
        request = HypothesisRequest.from_match(query, instruction="pour")
        assert request.root_type is core.ActionPhaseStructure
        assert request.fixed_values == {"decomposes_task": POURING}
        assert request.written_fixed_values == {
            "decomposes_task": written_value(POURING)
        }

    def test_a_query_without_an_instruction_is_asked_for_by_its_type(self):
        request = HypothesisRequest.from_match(pouring_query(), instruction="")
        assert request.instruction == core.ActionPhaseStructure.__name__

    def test_a_field_fixed_to_a_nested_match_is_refused(self):
        query = the(core.ActionPhaseStructure)(
            defines_phase=the(core.ActionPhase)(label="lift")
        )
        with pytest.raises(UnsupportedFixedValue):
            HypothesisRequest.from_match(query, instruction="")


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
            GenerationCase.LOOPING_STRUCTURE, GenerationCase.LOOP_REPLACED_PATCH
        )
        (structure,) = pouring_query().evaluate(backend=backend)
        repair_prompt = backend.log[1].user
        assert (
            StructureShapes.shipped().named(CORE_SHAPES.TransitionIsNotALoop).name
            in repair_prompt
        )
        assert "'back_to_tilt'" in repair_prompt
        assert len(structure.defines_phase) == 2

    def test_undecodable_answer_is_also_checked_against_the_rules(self):
        backend = backend_answering(
            GenerationCase.LOOPING_WITH_DANGLING_REFERENCE,
            GenerationCase.LOOP_AND_REFERENCE_REPAIRED_PATCH,
        )
        list(pouring_query().evaluate(backend=backend))
        repair_prompt = backend.log[1].user
        assert "missing_phase" in repair_prompt
        assert (
            StructureShapes.shipped().named(CORE_SHAPES.TransitionIsNotALoop).name
            in repair_prompt
        )

    def test_violation_names_the_fragments_it_concerns(self):
        liftoff = prehension.Liftoff()
        structure = core.ActionPhaseStructure(
            defines_phase=[core.ActionPhase(terminated_by=[liftoff])]
        )
        (violation,) = ShapesHold().violations([structure])
        identifiers = {id(structure): "pour", id(liftoff): "lift_off"}
        described = described_violation(
            violation, lambda fragment: identifiers[id(fragment)]
        )
        assert "'pour'" in described and "'lift_off'" in described

    def test_repair_patches_the_answer(self):
        backend = backend_answering(
            GenerationCase.LOOPING_STRUCTURE, GenerationCase.LOOP_REPLACED_PATCH
        )
        (structure,) = pouring_query().evaluate(backend=backend)
        assert sorted(map(type, structure.defines_phase), key=str) == sorted(
            [pouring.TiltToFlowPhase, pouring.FlowRegulationPhase], key=str
        )
        assert (
            backend.last_outcome.hypothesis.identifier_of(
                structure.defines_transition[0]
            )
            == "tilt_to_regulate"
        )

    def test_repair_builds_on_the_answer_with_fewest_problems(self):
        backend = backend_answering(
            GenerationCase.LOOPING_STRUCTURE,
            GenerationCase.DANGLING_REFERENCE,
            GenerationCase.LOOP_REPLACED_PATCH,
        )
        list(pouring_query().evaluate(backend=backend))
        initial, _, last = [user for _, user in backend.client.calls]
        assert last.startswith(initial)
        assert "missing_phase" not in last
        assert (
            StructureShapes.shipped().named(CORE_SHAPES.TransitionIsNotALoop).name
            in last
        )

    def test_answer_contradicting_a_fixed_value_is_repaired(self):
        backend = backend_answering(
            GenerationCase.STRUCTURE_OF_ANOTHER_TASK,
            GenerationCase.TASK_CORRECTED_PATCH,
        )
        (structure,) = pouring_query().evaluate(backend=backend)
        assert "decomposes_task = aicor:Pouring" in backend.log[1].user
        assert structure.decomposes_task == POURING

    def test_type_catalogue_is_in_the_system_prompt_shared_by_every_attempt(self):
        backend = backend_answering(
            GenerationCase.LOOPING_STRUCTURE, GenerationCase.LOOP_REPLACED_PATCH
        )
        list(pouring_query().evaluate(backend=backend))
        (first_system, first_user), (repair_system, _) = backend.client.calls
        catalogue_entry = rendered_type(type_schema_for(pouring.TiltToFlowPhase))
        assert first_system == repair_system
        assert catalogue_entry in first_system
        assert catalogue_entry not in first_user

    def test_fixed_values_and_instruction_are_stated(self):
        backend = backend_answering(GenerationCase.CHAINED_STRUCTURE)
        list(pouring_query().evaluate(backend=backend))
        prompt = backend.log[0].user
        assert backend.instruction in prompt
        assert "decomposes_task = aicor:Pouring" in prompt

    def test_concept_the_linked_ontologies_lack_is_returned_to_repair(self):
        backend = backend_answering(GenerationCase.CONCEPTS_LINKED_AND_LOCAL)
        backend.linked_terms = LinkedOntologyTerms.from_files(LINKED_EXCERPTS)
        backend.max_repairs = 0
        with pytest.raises(HypothesisRejected) as refusal:
            list(pouring_query().evaluate(backend=backend))
        assert [
            problem
            for problem in refusal.value.problems
            if "aicor:EffectorAtGraspSite" in problem
        ]

    def test_concept_fields_offer_concepts_of_the_hypothesis(self):
        schema = GenerationSchema.for_root(core.ActionPhaseStructure)
        category = schema.get("CategoricalValue").field_named("has_category")
        assert "hypothesis:" in field_hint(category)

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


def loop_checking_dialogue(
    *cases: GenerationCase, max_repairs: int
) -> GenerationDialogue:
    """
    Return a dialogue whose model gives the scripted answers in turn, checked only for
    looping transitions.
    """
    return GenerationDialogue(
        client=ScriptedClient(responses=[case.text for case in cases]),
        assessor=HypothesisAssessor(
            suite=AcceptanceSuite(
                rules=[
                    ShapesHold(
                        checked_shapes=frozenset({CORE_SHAPES.TransitionIsNotALoop})
                    )
                ]
            )
        ),
        max_repairs=max_repairs,
    )


class TestGenerationDialogue:
    """
    Asking the model, and repairing a rejected answer by patch.
    """

    def test_a_patched_answer_is_admitted_with_every_exchange_kept(self):
        dialogue = loop_checking_dialogue(
            GenerationCase.LOOPING_STRUCTURE,
            GenerationCase.LOOP_REPLACED_PATCH,
            max_repairs=1,
        )
        request = HypothesisRequest.from_match(pouring_query(), instruction="pour")
        outcome = dialogue.generate(request)
        assert outcome.exchanges == dialogue.exchanges
        assert [exchange.attempt for exchange in dialogue.exchanges] == [0, 1]

    def test_no_admissible_answer_within_the_budget_is_rejected(self):
        dialogue = loop_checking_dialogue(
            GenerationCase.LOOPING_STRUCTURE, max_repairs=0
        )
        request = HypothesisRequest.from_match(pouring_query(), instruction="pour")
        with pytest.raises(HypothesisRejected) as rejected:
            dialogue.generate(request)
        assert rejected.value.exchanges == dialogue.exchanges


class TestHypothesisAssessor:
    """
    Judging one decoded answer: closed world first, then open world.
    """

    def test_an_answer_meeting_the_request_and_the_rules_is_admitted(self):
        decoded = structure_decoder().decode(GenerationCase.CHAINED_STRUCTURE.text)
        request = HypothesisRequest.from_match(pouring_query(), instruction="pour")
        assessment = HypothesisAssessor(suite=AcceptanceSuite(rules=[])).assess(
            decoded, request
        )
        assert assessment.problems == []
        assert assessment.hypothesis is decoded.hypothesis

    def test_an_answer_breaking_the_variant_its_fixed_feature_defines_is_a_problem(
        self,
    ):
        structure = pouring.PouringPhaseStructure(
            decomposes_task=POURING,
            has_material_regime=pouring.MaterialRegime.GRANULAR,
            defines_parameter=[pouring.DynamicViscosityEstimate()],
        )
        decoded = DecodingResult(
            hypothesis=DecodedHypothesis(root=structure, fragments=[structure])
        )
        request = HypothesisRequest.from_match(
            the(pouring.PouringPhaseStructure)(
                decomposes_task=POURING,
                has_material_regime=pouring.MaterialRegime.GRANULAR,
            ),
            instruction="pour",
        )
        rules = AcceptanceSuite(rules=[RestrictionsHold()])
        (broken_variant,) = [
            violation
            for violation in RestrictionsHold().violations([structure])
            if violation.owner is pouring.GranularPouringStructure
        ]
        problems = HypothesisAssessor(suite=rules).assess(decoded, request).problems
        assert (
            described_violation(broken_variant, decoded.hypothesis.identifier_of)
            in problems
        )

    def test_an_open_world_contradiction_is_a_problem_naming_its_check(self):
        decoded = structure_decoder().decode(GenerationCase.LOOPING_STRUCTURE.text)
        request = HypothesisRequest.from_match(pouring_query(), instruction="pour")
        assessor = HypothesisAssessor(
            suite=AcceptanceSuite(rules=[]),
            consistency=ConflictWith(identifier="back_to_tilt"),
        )
        (problem,) = assessor.assess(decoded, request).problems
        assert str(OpenWorldCheck.CONSISTENT_WITH_ONTOLOGY) in problem


def admitted_chained_structure() -> GenerationOutcome:
    """
    Return an admitted outcome holding the chained pouring structure.
    """
    decoded = structure_decoder().decode(GenerationCase.CHAINED_STRUCTURE.text)
    return GenerationOutcome(hypothesis=decoded.hypothesis, warnings=[], exchanges=[])


class TestHypothesisRecorder:
    """
    Recording an admitted hypothesis, and storing it when there is a store.
    """

    def test_the_record_states_what_was_requested(self):
        request = HypothesisRequest.from_match(pouring_query(), instruction="pour")
        record = HypothesisRecorder(
            client_identity="scripted", consistency_checked=False
        ).record(admitted_chained_structure(), request)
        assert record.instruction == request.instruction
        assert record.root_type == qualified_name(request.root_type)
        assert record.fixed_values == request.written_fixed_values
        assert record.consistency == ConsistencyVerdict.NOT_CHECKED

    def test_a_recorded_hypothesis_is_stored(self, tmp_path):
        store = HypothesisStore(directory=tmp_path)
        request = HypothesisRequest.from_match(pouring_query(), instruction="pour")
        record = HypothesisRecorder(
            client_identity="scripted", consistency_checked=True, store=store
        ).record(admitted_chained_structure(), request)
        assert [stored.record_id for stored in store.records()] == [record.record_id]
        assert record.consistency == ConsistencyVerdict.CONSISTENT


class TestAdmission:
    """
    Admitted hypotheses are checked in the open world, recorded and stored.
    """

    def test_open_world_contradiction_is_returned_with_its_fragments(self):
        backend = backend_answering(
            GenerationCase.LOOPING_STRUCTURE, GenerationCase.LOOP_REPLACED_PATCH
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

    def test_a_hypothesis_is_served_for_the_same_query_whatever_its_instruction(
        self, tmp_path
    ):
        backend = backend_answering(GenerationCase.CHAINED_STRUCTURE)
        backend.store = HypothesisStore(directory=tmp_path)
        list(pouring_query().evaluate(backend=backend))
        backend.instruction = "tip the carton until the bowl is full"
        (served,) = pouring_query().evaluate(backend=backend)
        assert len(backend.client.calls) == 1
        assert len(served.defines_phase) == 2

    def test_admitted_hypothesis_is_served_again_without_the_model(self, tmp_path):
        backend = backend_answering(GenerationCase.CHAINED_STRUCTURE)
        backend.store = HypothesisStore(directory=tmp_path)
        list(pouring_query().evaluate(backend=backend))
        (served,) = pouring_query().evaluate(backend=backend)
        assert len(backend.client.calls) == 1
        assert len(served.defines_phase) == 2
