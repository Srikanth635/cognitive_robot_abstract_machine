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


def prehension_schema() -> GenerationSchema:
    """
    Return the schema selected for plain prehensile manipulation.
    """
    return derive_schema(
        modules=(
            FLANAGAN_CORE.name,
            FLANAGAN_OBJECT_DYNAMICS.name,
            FLANAGAN_PREHENSION.name,
        ),
        root="PrehensionPhaseStructure",
    )


# %% module selection


class TestPrehensionModuleSelection:
    """
    Grasping obtains prehension without substance-transfer vocabulary.
    """

    def test_plain_grasping_does_not_select_pouring(self) -> None:
        task = task_reference(
            iri="urn:test:grasping",
            ancestors=(SOMA + "Grasping", SOMA + "PhysicalTask"),
            label="Grasping",
        )

        selection = select_modules(task)

        assert selection.module_names == (
            FLANAGAN_CORE.name,
            FLANAGAN_OBJECT_DYNAMICS.name,
            FLANAGAN_PREHENSION.name,
        )
        assert FLANAGAN_POURING.name not in selection.module_names

    def test_pouring_composes_prehension_before_its_extension(self) -> None:
        task = task_reference(
            iri="urn:test:pouring",
            ancestors=(SOMA + "Pouring", SOMA + "PhysicalTask"),
            label="Pouring",
        )

        assert select_modules(task).module_names == (
            FLANAGAN_CORE.name,
            FLANAGAN_OBJECT_DYNAMICS.name,
            FLANAGAN_PREHENSION.name,
            FLANAGAN_POURING.name,
        )


# %% generation vocabulary


class TestPrehensionGenerationSchema:
    """
    The grasping schema is complete without flow-specific concepts.
    """

    def test_plain_schema_excludes_pouring_vocabulary(self) -> None:
        schema = prehension_schema()

        assert "ForceLoadingPhase" in schema.class_names
        assert "GripLoadCoupling" in schema.class_names
        assert "FlowRegulationPhase" not in schema.class_names
        assert "SourceContainer" not in schema.class_names

    def test_root_declares_coupling_and_anticipatory_adjustments(self) -> None:
        structure = prehension_schema().get("PrehensionPhaseStructure")
        assert structure is not None

        assert {
            "has_force_coupling",
            "has_anticipatory_grip_adjustment",
        } <= {field.name for field in structure.fields}

    def test_coupling_requires_its_functional_inputs(self) -> None:
        coupling = prehension_schema().get("GripLoadCoupling")
        assert coupling is not None

        assert {
            "predicted_load",
            "friction_estimate",
            "safety_margin",
            "grip_force_response",
            "temporal_coupling",
        } == {field.name for field in coupling.required_fields}

    def test_optional_hold_and_withdraw_phases_remain_available(self) -> None:
        schema = prehension_schema()

        assert "HoldPhase" in schema.class_names
        assert "WithdrawPhase" in schema.class_names
