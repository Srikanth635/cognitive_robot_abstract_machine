from __future__ import annotations

from llm_reasoner.knowledge.generation.schema import GenerationSchema, derive_schema
from llm_reasoner.knowledge.microworlds.registry import (
    FLANAGAN_CORE,
    FLANAGAN_OBJECT_DYNAMICS,
    FLANAGAN_POURING,
    FLANAGAN_PREHENSION,
    SOMA,
    select_modules,
    task_reference,
)

# %% schema fixture


def object_dynamics_schema() -> GenerationSchema:
    """
    Return the schema selected for a pouring structure with dynamics.
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


# %% module selection


class TestObjectDynamicsModuleSelection:
    """
    Manipulation profiles select dynamics through capability metadata.
    """

    def test_pouring_selects_object_dynamics_before_the_pouring_extension(
        self,
    ) -> None:
        task = task_reference(
            iri="urn:test:pouring",
            ancestors=(SOMA + "Pouring", SOMA + "PhysicalTask"),
            label="Pouring",
        )

        selection = select_modules(task)

        assert selection.module_names == (
            FLANAGAN_CORE.name,
            FLANAGAN_OBJECT_DYNAMICS.name,
            FLANAGAN_PREHENSION.name,
            FLANAGAN_POURING.name,
        )


# %% generation vocabulary


class TestObjectDynamicsGenerationSchema:
    """
    Generation exposes the complete internal-model composition.
    """

    def test_pouring_root_declares_each_dynamics_fragment_collection(self) -> None:
        structure = object_dynamics_schema().get("PouringPhaseStructure")
        assert structure is not None

        assert {
            "defines_internal_model",
            "defines_desired_state",
            "defines_motor_command",
            "defines_efference_copy",
            "has_internal_model_composition",
            "has_sensorimotor_memory",
            "has_representation_update",
            "defines_dynamics_perturbation",
            "has_dynamics_transfer",
            "has_representation_dissociation",
        } <= {field.name for field in structure.fields}

    def test_representation_requires_estimates_frame_purpose_and_roles(self) -> None:
        representation = object_dynamics_schema().get("ObjectDynamicsModel")
        assert representation is not None

        assert {
            "holds_estimate",
            "coordinate_frame",
            "purpose",
            "represents_role",
            "uses_efference_copy",
        } == {field.name for field in representation.required_fields}

    def test_dynamics_schema_replaces_the_generic_object_representation(self) -> None:
        structure = object_dynamics_schema().get("PouringPhaseStructure")
        assert structure is not None
        internal_models = next(
            field
            for field in structure.fields
            if field.name == "defines_internal_model"
        )

        assert "ObjectDynamicsModel" in internal_models.options
        assert "ObjectRepresentation" not in internal_models.options

    def test_dynamics_root_requires_and_explains_representation_ownership(
        self,
    ) -> None:
        structure = object_dynamics_schema().get("PouringPhaseStructure")
        assert structure is not None
        internal_models = next(
            field
            for field in structure.fields
            if field.name == "defines_internal_model"
        )

        assert internal_models.required == 1
        assert internal_models.docstring is not None
        assert all(
            field_name in internal_models.docstring
            for field_name in (
                "defines_parameter",
                "holds_estimate",
                "defines_internal_model",
                "represents_role",
                "parameter_on_role",
                "purpose",
            )
        )

    def test_composition_requires_each_command_and_prediction_stage(self) -> None:
        composition = object_dynamics_schema().get("InternalModelComposition")
        assert composition is not None

        assert {
            "desired_state",
            "inverse_model",
            "motor_command",
            "efference_copy",
            "dynamics_model",
            "forward_model",
            "predicted_mechanical_consequence",
            "predicted_sensory_consequence",
        } == {field.name for field in composition.required_fields}

    def test_coordinate_frame_offers_every_supported_frame(self) -> None:
        representation = object_dynamics_schema().get("ObjectDynamicsModel")
        assert representation is not None
        frame = next(
            field for field in representation.fields if field.name == "coordinate_frame"
        )

        assert frame.options == (
            "OBJECT_CENTERED",
            "ARM_CENTERED",
            "HAND_CENTERED",
            "GAZE_CENTERED",
        )
