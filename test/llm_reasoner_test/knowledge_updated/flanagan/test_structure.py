from __future__ import annotations

from typing import NamedTuple

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.control import (
    BoundaryControlExemption,
    CorrectiveAction,
    InitialStateEstimate,
    ModelUpdate,
    PhaseTransition,
    PredictionComparison,
    PredictionError,
    SensorimotorControlPoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    PhaseBoundaryEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    InternalModel,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    Parameter,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.phases import ActionPhase
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.prediction import (
    ContinuousSignalPrediction,
    SensoryEventPrediction,
    SensoryObservationType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import RoleSlot
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.structure import (
    ActionPhaseStructure,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE, FLANAGAN
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import PropertyMeta
from llm_reasoner.knowledge_updated.ontological.ranges import RangeResolver
from llm_reasoner.knowledge_updated.ontological.vocabulary import PropertyKind

# %% the specification


class Declares(NamedTuple):
    """
    What one collection of a structure must declare.
    """

    range: type
    minimum: int


DECLARATIONS: dict[str, Declares] = {
    "defines_role": Declares(RoleSlot, 1),
    "defines_phase": Declares(ActionPhase, 1),
    "defines_event_type": Declares(PhaseBoundaryEventType, 1),
    "defines_transition": Declares(PhaseTransition, 1),
    "defines_parameter": Declares(Parameter, 0),
    "defines_internal_model": Declares(InternalModel, 0),
    "defines_prediction": Declares(SensoryEventPrediction, 0),
    "defines_observation": Declares(SensoryObservationType, 0),
    "defines_continuous_prediction": Declares(ContinuousSignalPrediction, 0),
    "defines_comparison": Declares(PredictionComparison, 0),
    "defines_error": Declares(PredictionError, 0),
    "defines_correction": Declares(CorrectiveAction, 0),
    "defines_model_update": Declares(ModelUpdate, 0),
    "defines_initial_state_estimate": Declares(InitialStateEstimate, 0),
    "defines_control_point": Declares(SensorimotorControlPoint, 0),
    "defines_control_point_exemption": Declares(BoundaryControlExemption, 0),
}
"""
Each declaring collection of a structure beside what it holds.
"""


def declared() -> dict[str, PropertyMeta]:
    """
    Return the described properties of an action phase structure.
    """
    return PropertyMeta.declared_on(ActionPhaseStructure)


# %% the concept


class TestStructureConcept:
    """
    A structure is the top-level unit of knowledge: one hypothesis about one kind of task.
    """

    def test_structure_is_a_concrete_core_entity(self) -> None:
        described = FLANAGAN.find_concept(ActionPhaseStructure)

        assert issubclass(ActionPhaseStructure, KnowledgeEntity)
        assert described.module == CORE.name
        assert described.abstract is False

    def test_structure_is_aligned_as_a_workflow(self) -> None:
        assert FLANAGAN.find_concept(ActionPhaseStructure).alignments == (
            subclass_of(DUL, "Workflow"),
        )

    def test_structure_decomposes_exactly_one_task(self) -> None:
        described = declared()["decomposes_task"]

        assert described.kind is PropertyKind.REFERENCE
        assert described.functional is True
        assert described.minimum_count == 1
        assert described.owns is False


# %% declaring collections


class TestDeclaringCollections:
    """
    A structure declares every fragment it is made of, in one collection per kind.
    """

    def test_structure_declares_exactly_the_specified_collections(self) -> None:
        collections = {name for name, meta in declared().items() if meta.owns}

        assert collections == set(DECLARATIONS)

    @pytest.mark.parametrize("name", DECLARATIONS)
    def test_collection_is_a_many_valued_relation_that_owns_what_it_holds(
        self, name: str
    ) -> None:
        described = declared()[name]

        assert described.kind is PropertyKind.RELATION
        assert described.functional is False
        assert described.owns is True

    @pytest.mark.parametrize("name", DECLARATIONS)
    def test_collection_holds_the_specified_kind_and_needs_the_specified_minimum(
        self, name: str
    ) -> None:
        expected = DECLARATIONS[name]

        assert RangeResolver().resolve(ActionPhaseStructure, name).element_types == (
            expected.range,
        )
        assert declared()[name].minimum_count == expected.minimum

    def test_every_declaring_collection_follows_one_naming_convention(self) -> None:
        assert all(name.startswith("defines_") for name in DECLARATIONS)

    def test_structure_declares_the_models_and_predictions_its_phases_use(self) -> None:
        assert {"defines_internal_model", "defines_continuous_prediction"} <= set(
            DECLARATIONS
        )


# %% documentation


class TestStructureIsDocumented:
    """
    Every property of a structure says what it means.
    """

    def test_concept_and_properties_are_documented(self) -> None:
        reader = DocumentationReader()

        assert reader.read_class(ActionPhaseStructure) is not None
        assert reader.find_undocumented_fields(ActionPhaseStructure) == ()
