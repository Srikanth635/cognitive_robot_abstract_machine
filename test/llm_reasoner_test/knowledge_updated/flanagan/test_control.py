from __future__ import annotations

from typing import NamedTuple

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    SOMA,
    related_to,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.conditions import (
    PhaseConditionType,
    Subgoal,
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
    MechanicalEventType,
    PhaseBoundaryEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    EstimateRepresentation,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    Estimate,
    MotorSetpoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.phases import ActionPhase
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.prediction import (
    SensoryEventPrediction,
    SensoryObservationType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.quantities import (
    MagnitudeRange,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ComparisonOutcome,
    CorrectionDirection,
    EstimateUpdateDirection,
    PredictionErrorKind,
    RecoveryMode,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE, FLANAGAN
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import PropertyMeta
from llm_reasoner.knowledge_updated.ontological.ranges import RangeResolver
from llm_reasoner.knowledge_updated.ontological.references import ConceptReference
from llm_reasoner.knowledge_updated.ontological.validation import (
    Check,
    OntologyValidator,
)
from llm_reasoner.knowledge_updated.ontological.vocabulary import (
    PropertyKind,
    Severity,
)

# %% the specification of each concept


class Spec(NamedTuple):
    """
    What one property of a control concept must be.
    """

    kind: PropertyKind
    functional: bool
    minimum: int
    range: type


ATTR, REL, REF = PropertyKind.ATTRIBUTE, PropertyKind.RELATION, PropertyKind.REFERENCE

SPECIFICATION: dict[type, dict[str, Spec]] = {
    PredictionError: {
        "error_kind": Spec(ATTR, True, 1, PredictionErrorKind),
        "discrepancy": Spec(REL, True, 0, MagnitudeRange),
    },
    CorrectiveAction: {
        "triggered_by_error": Spec(REL, False, 1, PredictionError),
        "targets_setpoint": Spec(REL, False, 1, MotorSetpoint),
        "correction_direction": Spec(ATTR, True, 1, CorrectionDirection),
        "intended_effect": Spec(REL, False, 1, PhaseConditionType),
        "recovery_mode": Spec(ATTR, True, 1, RecoveryMode),
    },
    ModelUpdate: {
        "triggered_by_error": Spec(REL, False, 1, PredictionError),
        "updates_estimate": Spec(REL, False, 1, Estimate),
        "updates_representation": Spec(REL, True, 1, EstimateRepresentation),
        "update_direction": Spec(ATTR, True, 1, EstimateUpdateDirection),
    },
    InitialStateEstimate: {
        "for_phase": Spec(REL, True, 1, ActionPhase),
        "based_on_observation": Spec(REL, False, 1, SensoryObservationType),
        "estimates_condition": Spec(REL, False, 1, PhaseConditionType),
    },
    PredictionComparison: {
        "compares_prediction": Spec(REL, True, 1, SensoryEventPrediction),
        "compares_observation": Spec(REL, True, 0, SensoryObservationType),
        "comparison_outcome": Spec(ATTR, True, 1, ComparisonOutcome),
        "confirms_subgoal": Spec(REL, True, 0, Subgoal),
        "supplies_state_estimate": Spec(REL, True, 0, InitialStateEstimate),
        "has_error": Spec(REL, False, 0, PredictionError),
        "invokes_correction": Spec(REL, False, 0, CorrectiveAction),
        "produces_model_update": Spec(REL, False, 0, ModelUpdate),
    },
    SensorimotorControlPoint: {
        "at_event_type": Spec(REL, True, 1, MechanicalEventType),
        "has_prediction": Spec(REL, False, 1, SensoryEventPrediction),
        "has_comparison": Spec(REL, False, 1, PredictionComparison),
    },
    BoundaryControlExemption: {
        "exempts_event": Spec(REL, True, 1, PhaseBoundaryEventType),
        "justification": Spec(REF, True, 1, ConceptReference),
    },
    PhaseTransition: {
        "from_phase": Spec(REL, True, 1, ActionPhase),
        "to_phase": Spec(REL, True, 1, ActionPhase),
        "gated_by": Spec(REL, False, 1, PhaseBoundaryEventType),
        "permitted_by": Spec(REL, False, 0, PredictionComparison),
    },
}
"""
Every control concept beside the properties it must declare.
"""

ALIGNMENTS = {
    PredictionError: subclass_of(DUL, "Situation"),
    PredictionComparison: subclass_of(DUL, "Situation"),
    InitialStateEstimate: subclass_of(DUL, "Situation"),
    ModelUpdate: related_to(DUL, "Situation"),
    CorrectiveAction: subclass_of(SOMA, "PhysicalTask"),
    SensorimotorControlPoint: subclass_of(DUL, "Relation"),
    PhaseTransition: subclass_of(DUL, "Relation"),
}
"""
The foundational term each aligned control concept is placed under.
"""

# %% declaration


class TestControlConceptsAreDeclared:
    """
    The comparison and recovery vocabulary is core, description-level and concrete.
    """

    @pytest.mark.parametrize("cls", SPECIFICATION)
    def test_concept_is_a_concrete_core_entity(self, cls: type) -> None:
        described = FLANAGAN.find_concept(cls)

        assert issubclass(cls, KnowledgeEntity)
        assert described.module == CORE.name
        assert described.abstract is False

    @pytest.mark.parametrize("cls", ALIGNMENTS)
    def test_concept_is_aligned_by_what_its_definition_entails(self, cls: type) -> None:
        assert FLANAGAN.find_concept(cls).alignments == (ALIGNMENTS[cls],)

    def test_exemption_is_an_annotation_of_the_structure_and_is_not_aligned(
        self,
    ) -> None:
        assert FLANAGAN.find_concept(BoundaryControlExemption).alignments == ()


# %% properties


CASES = [
    (cls, name, spec)
    for cls, properties in SPECIFICATION.items()
    for name, spec in properties.items()
]
"""
One case per property of a control concept.
"""


class TestControlProperties:
    """
    Each control concept declares exactly the properties it is specified to have.
    """

    @pytest.mark.parametrize(
        ("cls", "name", "spec"),
        CASES,
        ids=[f"{cls.__name__}.{name}" for cls, name, _ in CASES],
    )
    def test_property_matches_its_specification(
        self, cls: type, name: str, spec: Spec
    ) -> None:
        described = PropertyMeta.declared_on(cls)[name]

        assert described.kind is spec.kind
        assert described.functional is spec.functional
        assert described.minimum_count == spec.minimum
        assert RangeResolver().resolve(cls, name).element_types == (spec.range,)

    @pytest.mark.parametrize("cls", SPECIFICATION)
    def test_no_property_is_declared_beyond_the_specification(self, cls: type) -> None:
        own = {
            declaration.field_name
            for declaration in FLANAGAN.property_declarations(CORE.name)
            if declaration.owner is cls
        }

        assert own == set(SPECIFICATION[cls])


# %% decisions


class TestControlDecisions:
    """
    The structural choices made for the control vocabulary.
    """

    def test_permission_is_stored_on_the_transition_only(self) -> None:
        assert "permits_transition" not in PropertyMeta.declared_on(
            PredictionComparison
        )

    def test_direction_of_a_correction_and_of_an_update_are_different_properties(
        self,
    ) -> None:
        report = OntologyValidator(FLANAGAN).validate(CORE.name)

        assert report.of(Check.PROPERTY_IRI_CONFLICT) == ()

    def test_error_reference_shared_by_correction_and_update_has_one_identifier(
        self,
    ) -> None:
        identifiers = {
            declaration.iri
            for declaration in FLANAGAN.property_declarations(CORE.name)
            if declaration.field_name == "triggered_by_error"
        }

        assert identifiers == {CORE.namespace.property_term("triggered_by_error")}

    def test_control_point_can_only_sit_at_a_physical_event(self) -> None:
        assert not issubclass(PhaseBoundaryEventType, MechanicalEventType)
        assert RangeResolver().resolve(
            SensorimotorControlPoint, "at_event_type"
        ).element_types == (MechanicalEventType,)

    def test_the_core_stays_free_of_errors(self) -> None:
        report = OntologyValidator(FLANAGAN).validate(CORE.name)

        assert [v for v in report.violations if v.severity is Severity.ERROR] == []


# %% documentation


class TestControlIsDocumented:
    """
    Every control concept and property says what it means.
    """

    @pytest.mark.parametrize("cls", SPECIFICATION)
    def test_concept_and_properties_are_documented(self, cls: type) -> None:
        reader = DocumentationReader()

        assert reader.read_class(cls) is not None
        assert reader.find_undocumented_fields(cls) == ()
