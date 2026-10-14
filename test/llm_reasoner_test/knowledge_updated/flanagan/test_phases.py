from __future__ import annotations

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    SOMA,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.conditions import (
    ConditionRequirement,
    Subgoal,
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
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import RoleSlot
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    CouplingRegime,
    RegulationMode,
    SensoryModality,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE, FLANAGAN
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import PropertyMeta
from llm_reasoner.knowledge_updated.ontological.ranges import RangeResolver
from llm_reasoner.knowledge_updated.ontological.vocabulary import PropertyKind

# %% fixtures


def declared() -> dict[str, PropertyMeta]:
    """
    Return the described properties of an action phase.
    """
    return PropertyMeta.declared_on(ActionPhase)


def range_of(name: str) -> tuple[type, ...]:
    """
    Return the classes a property of an action phase ranges over.
    """
    return RangeResolver().resolve(ActionPhase, name).element_types


# %% the concept


class TestActionPhaseConcept:
    """
    A phase is a task-level building block of a manipulation, never an occurrence.
    """

    def test_phase_is_a_concrete_core_entity(self) -> None:
        described = FLANAGAN.find_concept(ActionPhase)

        assert issubclass(ActionPhase, KnowledgeEntity)
        assert described.module == CORE.name
        assert described.abstract is False

    def test_phase_is_aligned_as_a_physical_task(self) -> None:
        assert FLANAGAN.find_concept(ActionPhase).alignments == (
            subclass_of(SOMA, "PhysicalTask"),
        )

    def test_kinds_of_phase_share_no_instance(self) -> None:
        assert FLANAGAN.find_concept(ActionPhase).axioms.disjoint_siblings is True


# %% what makes a phase


class TestPhaseIdentity:
    """
    Every phase has one subgoal, one coupling regime and one regulation mode.
    """

    def test_phase_aims_for_exactly_one_subgoal(self) -> None:
        described = declared()["aims_for"]

        assert described.kind is PropertyKind.RELATION
        assert described.functional is True
        assert described.minimum_count == 1
        assert range_of("aims_for") == (Subgoal,)

    @pytest.mark.parametrize(
        ("name", "value_type"),
        [("coupling_regime", CouplingRegime), ("regulation_mode", RegulationMode)],
    )
    def test_phase_has_one_value_of_each_orthogonal_dimension(
        self, name: str, value_type: type
    ) -> None:
        described = declared()[name]

        assert described.kind is PropertyKind.ATTRIBUTE
        assert described.functional is True
        assert described.minimum_count == 1
        assert range_of(name) == (value_type,)


# %% boundaries


class TestPhaseBoundaries:
    """
    A phase begins at optional events and ends at one or more events.
    """

    def test_a_phase_needs_at_least_one_terminating_event(self) -> None:
        described = declared()["terminated_by"]

        assert described.functional is False
        assert described.minimum_count == 1
        assert range_of("terminated_by") == (PhaseBoundaryEventType,)

    def test_the_first_phase_of_a_structure_has_no_initiating_event(self) -> None:
        described = declared()["initiated_by"]

        assert described.minimum_count == 0
        assert range_of("initiated_by") == (PhaseBoundaryEventType,)


# %% participants and demands


class TestPhaseParticipantsAndDemands:
    """
    A phase names who takes part in it and what it demands of the world.
    """

    def test_phase_involves_at_least_one_role(self) -> None:
        described = declared()["phase_involves"]

        assert described.minimum_count == 1
        assert range_of("phase_involves") == (RoleSlot,)

    def test_phase_lists_its_condition_requirements(self) -> None:
        assert declared()["has_condition_requirement"].minimum_count == 0
        assert range_of("has_condition_requirement") == (ConditionRequirement,)

    def test_phase_is_governed_by_internal_models(self) -> None:
        assert declared()["governed_by"].minimum_count == 0
        assert range_of("governed_by") == (InternalModel,)


# %% what is active in a phase


class TestWhatIsActiveInAPhase:
    """
    A phase lists the parameters and predictions active in it, so those do not name
    phases.
    """

    def test_phase_lists_the_parameters_active_in_it(self) -> None:
        described = declared()["active_parameters"]

        assert described.kind is PropertyKind.RELATION
        assert described.functional is False
        assert range_of("active_parameters") == (Parameter,)

    def test_phase_lists_the_continuous_predictions_running_in_it(self) -> None:
        described = declared()["continuous_predictions"]

        assert described.kind is PropertyKind.RELATION
        assert described.functional is False
        assert range_of("continuous_predictions") == (ContinuousSignalPrediction,)

    def test_phase_lists_the_modalities_it_tracks_continuously(self) -> None:
        described = declared()["monitors_modality"]

        assert described.kind is PropertyKind.ATTRIBUTE
        assert described.functional is False
        assert range_of("monitors_modality") == (SensoryModality,)


# %% documentation


class TestActionPhaseIsDocumented:
    """
    Every property of a phase says what it means.
    """

    def test_concept_and_properties_are_documented(self) -> None:
        reader = DocumentationReader()

        assert reader.read_class(ActionPhase) is not None
        assert reader.find_undocumented_fields(ActionPhase) == ()
