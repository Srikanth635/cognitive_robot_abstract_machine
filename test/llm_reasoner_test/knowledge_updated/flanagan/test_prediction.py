from __future__ import annotations

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    MechanicalEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import ForwardModel
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import Setpoint
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.prediction import (
    ContinuousSignalPrediction,
    SensoryEventPrediction,
    SensoryObservationType,
    SignalTrajectory,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.quantities import (
    MagnitudeRange,
    TemporalWindow,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import RoleSlot
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    SensoryModality,
    TrajectoryTrend,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE, FLANAGAN
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import PropertyMeta
from llm_reasoner.knowledge_updated.ontological.ranges import RangeResolver
from llm_reasoner.knowledge_updated.ontological.references import ConceptReference
from llm_reasoner.knowledge_updated.ontological.vocabulary import PropertyKind

# %% fixtures

ALL_PREDICTION_CONCEPTS = (
    SensoryEventPrediction,
    SensoryObservationType,
    SignalTrajectory,
    ContinuousSignalPrediction,
)
"""
Every concept of the prediction vocabulary.
"""


def declared(cls: type) -> dict[str, PropertyMeta]:
    """
    Return the described properties of a class.
    """
    return PropertyMeta.declared_on(cls)


def range_of(cls: type, name: str) -> tuple[type, ...]:
    """
    Return the classes a property of a class ranges over.
    """
    return RangeResolver().resolve(cls, name).element_types


# %% declaration


class TestPredictionConceptsAreDeclared:
    """
    Predictions and observations are description-level core knowledge.
    """

    @pytest.mark.parametrize("cls", ALL_PREDICTION_CONCEPTS)
    def test_concept_is_a_concrete_core_entity(self, cls: type) -> None:
        described = FLANAGAN.find_concept(cls)

        assert issubclass(cls, KnowledgeEntity)
        assert described.module == CORE.name
        assert described.abstract is False

    def test_predictions_are_aligned_as_descriptions_and_observations_as_event_types(
        self,
    ) -> None:
        for cls in (
            SensoryEventPrediction,
            SignalTrajectory,
            ContinuousSignalPrediction,
        ):
            assert FLANAGAN.find_concept(cls).alignments == (
                subclass_of(DUL, "Description"),
            )
        assert FLANAGAN.find_concept(SensoryObservationType).alignments == (
            subclass_of(DUL, "EventType"),
        )

    def test_modality_is_a_value_and_not_a_kind_of_signal_class(self) -> None:
        iri = CORE.namespace.term("SensorySignalType")

        assert FLANAGAN.find_concept_by_iri(iri) is None


# %% predicted and observed events


class TestEventEvidence:
    """
    A prediction and an observation each say which channel, when, and about whom.
    """

    @pytest.mark.parametrize("cls", [SensoryEventPrediction, SensoryObservationType])
    def test_modality_is_a_required_choice_of_channel(self, cls: type) -> None:
        described = declared(cls)["modality"]

        assert described.kind is PropertyKind.ATTRIBUTE
        assert described.functional is True
        assert described.minimum_count == 1
        assert range_of(cls, "modality") == (SensoryModality,)

    @pytest.mark.parametrize(
        ("cls", "timing", "roles"),
        [
            (SensoryEventPrediction, "expected_timing", "expected_on_role"),
            (SensoryObservationType, "observed_at", "observed_on_role"),
        ],
    )
    def test_evidence_has_a_time_window_and_participants(
        self, cls: type, timing: str, roles: str
    ) -> None:
        assert declared(cls)[timing].minimum_count == 1
        assert range_of(cls, timing) == (TemporalWindow,)
        assert declared(cls)[roles].minimum_count == 1
        assert range_of(cls, roles) == (RoleSlot,)

    @pytest.mark.parametrize(
        ("cls", "event", "feature"),
        [
            (SensoryEventPrediction, "predicts_event", "predicts_feature"),
            (SensoryObservationType, "observes_event", "observes_feature"),
        ],
    )
    def test_evidence_is_about_exactly_one_of_an_event_or_a_feature(
        self, cls: type, event: str, feature: str
    ) -> None:
        described = declared(cls)

        assert described[event].exclusive_group == described[feature].exclusive_group
        assert described[event].exclusive_group is not None
        assert range_of(cls, event) == (MechanicalEventType,)
        assert described[feature].kind is PropertyKind.REFERENCE

    def test_only_a_physical_event_can_cause_a_prediction(self) -> None:
        described = declared(SensoryEventPrediction)["caused_by_event"]

        assert described.functional is True
        assert described.minimum_count == 1
        assert range_of(SensoryEventPrediction, "caused_by_event") == (
            MechanicalEventType,
        )

    def test_a_prediction_comes_from_a_forward_model(self) -> None:
        described = declared(SensoryEventPrediction)["generated_by"]

        assert described.minimum_count == 1
        assert range_of(SensoryEventPrediction, "generated_by") == (ForwardModel,)

    @pytest.mark.parametrize(
        ("cls", "name"),
        [
            (SensoryEventPrediction, "expected_magnitude"),
            (SensoryEventPrediction, "tolerance"),
            (SensoryObservationType, "observed_magnitude"),
        ],
    )
    def test_magnitudes_are_optional_ranges(self, cls: type, name: str) -> None:
        assert declared(cls)[name].minimum_count == 0
        assert range_of(cls, name) == (MagnitudeRange,)


# %% continuous prediction


class TestContinuousPrediction:
    """
    A continuous prediction compares a trajectory with an observation through a phase.
    """

    def test_trajectory_states_a_feature_a_trend_and_a_range(self) -> None:
        described = declared(SignalTrajectory)

        assert described["signal_feature"].kind is PropertyKind.REFERENCE
        assert described["trend"].minimum_count == 1
        assert range_of(SignalTrajectory, "trend") == (TrajectoryTrend,)
        assert described["expected_range"].minimum_count == 1
        assert range_of(SignalTrajectory, "expected_range") == (MagnitudeRange,)

    def test_trajectory_feature_is_a_term_reference(self) -> None:
        assert range_of(SignalTrajectory, "signal_feature") == (ConceptReference,)

    def test_continuous_prediction_names_everything_it_is_compared_and_controlled_by(
        self,
    ) -> None:
        described = declared(ContinuousSignalPrediction)

        for name in (
            "modality",
            "predicted_trajectory",
            "tolerance",
            "generated_by",
            "comparison_target",
            "controlled_variable",
        ):
            assert described[name].minimum_count == 1

    def test_the_controlled_variable_is_a_setpoint(self) -> None:
        assert range_of(ContinuousSignalPrediction, "controlled_variable") == (
            Setpoint,
        )

    def test_the_comparison_target_is_an_observation_type(self) -> None:
        assert range_of(ContinuousSignalPrediction, "comparison_target") == (
            SensoryObservationType,
        )

    def test_a_continuous_prediction_does_not_name_the_phases_it_runs_in(self) -> None:
        assert "predicted_for_phase" not in declared(ContinuousSignalPrediction)


# %% documentation


class TestPredictionIsDocumented:
    """
    Every prediction concept and property says what it means.
    """

    @pytest.mark.parametrize("cls", ALL_PREDICTION_CONCEPTS)
    def test_concept_and_properties_are_documented(self, cls: type) -> None:
        reader = DocumentationReader()

        assert reader.read_class(cls) is not None
        assert reader.find_undocumented_fields(cls) == ()
