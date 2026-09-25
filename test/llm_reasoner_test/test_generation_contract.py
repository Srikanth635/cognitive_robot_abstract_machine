from __future__ import annotations

import json

import pytest
from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.knowledge.generation.backend import (
    HypothesisGenerator,
    ResponseUnusable,
)
from llm_reasoner.knowledge.generation.prompts import GenerationPromptBuilder, hint_for
from llm_reasoner.knowledge.generation.schema import (
    ClassSchema,
    FieldKind,
    FieldSchema,
    GenerationSchema,
    derive_schema,
    to_json_schema,
)
from llm_reasoner.knowledge.generation.wire import (
    CanonicalFragmentIdentityAllocator,
    GenerationWireDecoder,
    WireFormatError,
)
from llm_reasoner.knowledge.hypothesis.provenance import Provenance
from llm_reasoner.knowledge.hypothesis.record import HypothesisRecord
from llm_reasoner.knowledge.hypothesis.serialization import dumps, loads
from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ActionPhaseStructure,
    MagnitudeRange,
    PhaseConditionType,
    RoleSlot,
)
from llm_reasoner.knowledge.microworlds.flanagan.pouring import POUR
from llm_reasoner.knowledge.microworlds.flanagan.prehension import Liftoff
from llm_reasoner.knowledge.microworlds.registry import (
    DEFAULT_PROFILE,
    FLANAGAN_CORE,
    FLANAGAN_OBJECT_DYNAMICS,
    FLANAGAN_POURING,
    FLANAGAN_PREHENSION,
    ModuleSelection,
    current_versions,
)
from llm_reasoner.knowledge.ontological.constraints import ConstraintSuite
from llm_reasoner.knowledge.ontological.decorators import ConceptReference

# %% Schema fixtures


def pouring_schema() -> GenerationSchema:
    """
    Return the action-specific schema used by pouring generation.
    """
    return derive_schema(
        modules=(
            FLANAGAN_CORE.name,
            FLANAGAN_OBJECT_DYNAMICS.name,
            FLANAGAN_PREHENSION.name,
            FLANAGAN_POURING.name,
        ),
        root="PouringPhaseStructure",
    )


def fragment_field(schema: GenerationSchema, owner: str, name: str) -> FieldSchema:
    """
    Return one named field from a generated class schema.
    """
    class_schema = schema.get(owner)
    assert class_schema is not None
    return next(field for field in class_schema.fields if field.name == name)


def generator_for(root_class: str) -> HypothesisGenerator:
    """
    Return an offline generator suitable for direct construction tests.
    """
    return HypothesisGenerator(
        client=ScriptedClient(),
        suite=ConstraintSuite(name="construction", constraints=()),
        root_class=root_class,
    )


# %% Canonical schema vocabulary


class TestMostSpecificFragmentChoices:
    """
    Action-specific schemas offer leaf ontology concepts.
    """

    def test_phase_references_offer_specific_phase_types(self) -> None:
        phase_field = fragment_field(
            pouring_schema(),
            "PouringPhaseStructure",
            "defines_phase",
        )

        assert "ReachPhase" in phase_field.options
        assert "FlowRegulationPhase" in phase_field.options
        assert "ActionPhase" not in phase_field.options

    def test_event_references_exclude_intermediate_event_categories(self) -> None:
        event_field = fragment_field(
            pouring_schema(),
            "PouringPhaseStructure",
            "defines_event_type",
        )

        assert "FlowOnset" in event_field.options
        assert "MechanicalEventType" not in event_field.options
        assert "FlowEventType" not in event_field.options


class TestCanonicalConceptChoices:
    """
    Referenced role concepts are closed over the selected ontology.
    """

    def test_role_slot_offers_specific_role_concepts(self) -> None:
        role_field = fragment_field(pouring_schema(), "RoleSlot", "binds_as")

        assert role_field.kind is FieldKind.CONCEPT
        assert "SourceContainer" in role_field.options
        assert "TargetReceptacle" in role_field.options
        assert "ManipulationRole" not in role_field.options

    def test_role_options_are_exposed_to_structured_decoding(self) -> None:
        json_schema = to_json_schema(pouring_schema())
        fragment_shapes = json_schema["properties"]["fragments"]["items"]["anyOf"]
        role_shape = next(
            shape
            for shape in fragment_shapes
            if shape["properties"]["type"]["const"] == "RoleSlot"
        )

        assert role_shape["properties"]["binds_as"]["enum"] == list(
            fragment_field(pouring_schema(), "RoleSlot", "binds_as").options
        )


class TestGenerationPromptContract:
    """
    The prompt distinguishes ontology classification from description.
    """

    def test_prompt_requests_specific_types_and_descriptive_labels(self) -> None:
        prompt = GenerationPromptBuilder().build(
            schema=pouring_schema(),
            action_label="pouring cereal into a bowl",
        )

        assert "Select the most specific available type" in prompt
        assert '"label" is only a human-readable description' in prompt
        assert "Do not encode a new concept only in a label" in prompt


class TestFieldDefaultsSurfacedToTheModel:
    """
    A field's real class-level default is read back from the dataclass and shown in the
    prompt, so the model has a concrete prior for the ordinary case instead of only the
    generic "omit if optional" instruction -- which a plausible-sounding value (a
    threshold crossing "sounds like" a failure) can override even though the field
    defaults away from it.
    """

    def test_a_boolean_fields_default_is_read_from_the_dataclass(self) -> None:
        is_failure_event = fragment_field(
            pouring_schema(), "Liftoff", "is_failure_event"
        )

        assert is_failure_event.has_default is True
        assert is_failure_event.default is False

    def test_a_field_with_no_class_level_default_gets_no_default_note(self) -> None:
        # prop() always sets dataclasses.field(default=None, ...) unless the
        # microworld gives it a real one, so an unset field's raw default is
        # None, not MISSING -- it is None specifically (not absent) that must
        # suppress the note, which is what the hint actually promises the model
        at_event_type = fragment_field(
            pouring_schema(), "SensorimotorControlPoint", "at_event_type"
        )

        assert at_event_type.required
        assert "defaults to" not in hint_for(at_event_type)

    def test_a_collection_fields_default_factory_is_not_reported_as_a_default(
        self,
    ) -> None:
        # default_factory=list fields must not surface "defaults to []" -- that's
        # the ordinary unset state for any collection, not information worth a
        # model's attention the way a non-obvious scalar default is
        defines_phase = fragment_field(
            pouring_schema(), "PouringPhaseStructure", "defines_phase"
        )

        assert defines_phase.collection is True
        assert defines_phase.has_default is False

    def test_the_hint_states_the_default_for_an_optional_scalar_field(self) -> None:
        is_failure_event = fragment_field(
            pouring_schema(), "Liftoff", "is_failure_event"
        )

        assert "defaults to false" in hint_for(is_failure_event)

    def test_the_rendered_prompt_carries_the_default(self) -> None:
        prompt = GenerationPromptBuilder().build(
            schema=pouring_schema(),
            action_label="pouring cereal into a bowl",
        )

        assert "defaults to false" in prompt


class TestPredictiveControlGenerationSchema:
    """
    Generation exposes the complete predictive-control graph to the model.
    """

    def test_structure_declares_each_predictive_fragment_collection(self) -> None:
        structure = pouring_schema().get("PouringPhaseStructure")
        assert structure is not None

        assert {
            "defines_prediction",
            "defines_observation",
            "has_comparison",
            "defines_error",
            "defines_correction",
            "has_model_update",
            "defines_initial_state_estimate",
        } <= {field.name for field in structure.fields}

    def test_control_point_uses_predictions_and_comparisons_instead_of_labels(
        self,
    ) -> None:
        control_point = pouring_schema().get("SensorimotorControlPoint")
        assert control_point is not None

        field_names = {field.name for field in control_point.fields}
        assert {"at_event_type", "has_prediction", "has_comparison"} <= field_names
        assert "predicts_signal" not in field_names
        assert "on_mismatch_invokes" not in field_names

    def test_prediction_exposes_every_operational_comparison_dimension(self) -> None:
        prediction = pouring_schema().get("SensoryEventPrediction")
        assert prediction is not None

        assert {
            "modality",
            "predicts_event",
            "predicts_feature",
            "caused_by_event",
            "expected_timing",
            "expected_on_role",
            "expected_magnitude",
            "tolerance",
            "generated_by",
        } == {field.name for field in prediction.fields}

    def test_correction_requires_trigger_target_direction_effect_and_recovery(
        self,
    ) -> None:
        correction = pouring_schema().get("CorrectiveAction")
        assert correction is not None

        assert {
            "triggered_by_error",
            "targets_setpoint",
            "direction",
            "intended_effect",
            "recovery_mode",
        } == {field.name for field in correction.required_fields}

    def test_comparison_outcome_offers_match_and_mismatch(self) -> None:
        outcome = fragment_field(
            pouring_schema(),
            "PredictionComparison",
            "outcome",
        )

        assert outcome.options == ("MATCH", "MISMATCH")


class TestBackwardCompatibleGenerationFlow:
    """
    Canonical generation retains the established fallback and repair paths.
    """

    def test_core_fallback_still_offers_general_action_phases(self) -> None:
        schema = derive_schema(
            modules=(FLANAGAN_CORE.name,),
            root=ActionPhaseStructure.__name__,
        )

        phase_field = fragment_field(
            schema,
            ActionPhaseStructure.__name__,
            "defines_phase",
        )

        assert phase_field.options == ("ActionPhase",)

    def test_wire_rejection_uses_the_existing_repair_loop(self) -> None:
        task = ConceptReference(iri="soma:PhysicalTask", label="PhysicalTask")
        client = ScriptedClient(
            responses=[
                json.dumps({"fragments": {}}),
                json.dumps(
                    {
                        "fragments": [
                            {
                                "id": "temporary-range",
                                "type": MagnitudeRange.__name__,
                                "lower_bound": 0.2,
                                "upper_bound": 0.5,
                                "unit": "kg",
                            }
                        ]
                    }
                ),
            ]
        )
        generator = HypothesisGenerator(
            client=client,
            suite=ConstraintSuite(name="construction", constraints=()),
            root_class=MagnitudeRange.__name__,
            max_repairs=1,
        )
        selection = ModuleSelection(
            task=task,
            profile=DEFAULT_PROFILE,
            modules=(FLANAGAN_CORE,),
        )

        result = generator.generate(task, selection=selection)

        assert result.admitted is True
        assert result.record.root is not None
        assert result.record.root.local_id == "magnitude-range"
        assert len(client.calls) == 2
        assert 'Expected a non-empty "fragments" array' in client.calls[1][1]


# %% Typed wire validation


def wire_schema() -> GenerationSchema:
    """
    Return a small flat graph schema for wire-contract tests.
    """
    return GenerationSchema(
        root="ReachPhase",
        classes=(
            ClassSchema(
                name="ReachPhase",
                docstring=None,
                fields=(
                    FieldSchema(
                        name="terminated_by",
                        kind=FieldKind.FRAGMENT,
                        type_name="MechanicalEventType",
                        collection=True,
                        required=0,
                        options=("Liftoff",),
                    ),
                ),
            ),
            ClassSchema(name="Liftoff", docstring=None, fields=()),
        ),
    )


class TestTypedFlatWireValidation:
    """
    Wire decoding rejects shapes outside the derived flat contract.
    """

    def test_collection_field_rejects_a_scalar_reference(self) -> None:
        payload = {
            "fragments": [
                {
                    "id": "reach",
                    "type": "ReachPhase",
                    "terminated_by": "liftoff",
                },
                {"id": "liftoff", "type": "Liftoff"},
            ]
        }

        with pytest.raises(WireFormatError, match="must be an array"):
            GenerationWireDecoder(schema=wire_schema()).decode(payload)

    def test_unknown_field_is_rejected_before_construction(self) -> None:
        payload = {
            "fragments": [
                {
                    "id": "reach",
                    "type": "ReachPhase",
                    "invented_relation": "liftoff",
                },
                {"id": "liftoff", "type": "Liftoff"},
            ]
        }

        with pytest.raises(WireFormatError, match="invented_relation"):
            GenerationWireDecoder(schema=wire_schema()).decode(payload)

    def test_valid_graph_is_returned_as_typed_declarations(self) -> None:
        payload = {
            "fragments": [
                {
                    "id": "reach",
                    "type": "ReachPhase",
                    "label": "Reach toward the cereal box",
                    "terminated_by": ["liftoff"],
                },
                {"id": "liftoff", "type": "Liftoff"},
            ]
        }

        declaration = GenerationWireDecoder(schema=wire_schema()).decode(payload)

        assert declaration.root.identifier == "reach"
        assert declaration.root.description == "Reach toward the cereal box"
        assert declaration.root.value_of("terminated_by") == ["liftoff"]


# %% Construction and identity


class TestCanonicalFragmentIdentity:
    """
    Constructed fragments receive ontology-derived local identities.
    """

    def test_allocator_uses_type_name_and_occurrence_number(self) -> None:
        allocator = CanonicalFragmentIdentityAllocator()

        assert allocator.allocate("FlowRegulationPhase") == "flow-regulation-phase"
        assert allocator.allocate("FlowRegulationPhase") == "flow-regulation-phase-2"

    def test_construction_keeps_description_separate_from_identity(self) -> None:
        schema = GenerationSchema(
            root="Liftoff",
            classes=(ClassSchema(name="Liftoff", docstring=None, fields=()),),
        )
        raw_response = json.dumps(
            {
                "fragments": [
                    {
                        "id": "temporary-event-reference",
                        "type": "Liftoff",
                        "label": "Load exceeds the cereal box weight",
                    }
                ]
            }
        )

        root, fragments = generator_for("Liftoff")._build(raw_response, schema)

        assert fragments == [root]
        assert root.local_id == "liftoff"
        assert root.label == "Load exceeds the cereal box weight"


class TestReferenceRangeValidation:
    """
    Fragment references must satisfy their declared ontology range.
    """

    def test_event_cannot_fill_a_phase_condition_reference(self) -> None:
        schema = GenerationSchema(
            root="ConditionRequirement",
            classes=(
                ClassSchema(
                    name="ConditionRequirement",
                    docstring=None,
                    fields=(
                        FieldSchema(
                            name="on_condition",
                            kind=FieldKind.FRAGMENT,
                            type_name=PhaseConditionType.__name__,
                            collection=False,
                            required=1,
                            options=("EffectorObjectContact",),
                        ),
                    ),
                ),
                ClassSchema(name="SpillEvent", docstring=None, fields=()),
            ),
        )
        raw_response = json.dumps(
            {
                "fragments": [
                    {"id": "spill", "type": "SpillEvent"},
                    {
                        "id": "requirement",
                        "type": "ConditionRequirement",
                        "on_condition": "spill",
                    },
                ]
            }
        )

        with pytest.raises(ResponseUnusable, match="PhaseConditionType"):
            generator_for("ConditionRequirement")._build(raw_response, schema)


class TestCanonicalConceptDecoding:
    """
    Bare ontology names are normalized to registered concept IRIs.
    """

    def test_role_name_becomes_a_canonical_concept_reference(self) -> None:
        schema = GenerationSchema(
            root="RoleSlot",
            classes=(
                ClassSchema(
                    name="RoleSlot",
                    docstring=None,
                    fields=(
                        FieldSchema(
                            name="binds_as",
                            kind=FieldKind.CONCEPT,
                            type_name="ConceptReference",
                            collection=False,
                            required=1,
                            options=("SourceContainer",),
                        ),
                    ),
                ),
            ),
        )
        raw_response = json.dumps(
            {
                "fragments": [
                    {
                        "id": "source-role",
                        "type": "RoleSlot",
                        "binds_as": "SourceContainer",
                    }
                ]
            }
        )

        root, _ = generator_for("RoleSlot")._build(raw_response, schema)

        assert isinstance(root, RoleSlot)
        assert root.binds_as == ConceptReference(
            iri=POUR + "SourceContainer",
            label="SourceContainer",
        )


class TestPersistentFragmentIdentity:
    """
    Serialization preserves fragment identity within a record.
    """

    def test_round_trip_preserves_local_identity(self) -> None:
        event = Liftoff(local_id="liftoff-event", label="Liftoff")
        task = ConceptReference(iri="soma:PouringInto", label="PouringInto")
        record = HypothesisRecord.from_root(
            event,
            action_type=task,
            provenance=Provenance(
                generator="test",
                prompt_version="generation/v2",
            ),
        )

        loaded = loads(dumps(record))

        assert loaded.root is not None
        assert loaded.root.local_id == "liftoff-event"


class TestKnowledgeVersioning:
    """
    Canonical generation invalidates hypotheses from the earlier contract.
    """

    def test_generator_and_microworld_versions_are_incremented(self) -> None:
        generator = generator_for("PouringPhaseStructure")

        assert generator.prompt_version == "generation/v2"
        assert current_versions()[FLANAGAN_CORE.name] == "0.3.0"
        assert current_versions()[FLANAGAN_PREHENSION.name] == "0.1.0"
        assert current_versions()[FLANAGAN_POURING.name] == "0.4.0"
        assert current_versions()[FLANAGAN_OBJECT_DYNAMICS.name] == "0.1.0"
