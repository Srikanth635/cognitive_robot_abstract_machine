from __future__ import annotations

import re

import llm_reasoner.knowledge_updated.microworlds.flanagan.core  # noqa: F401  declares the core
import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.conditions import (
    ConditionRequirement,
    Subgoal,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.control import (
    CorrectiveAction,
    ModelUpdate,
    PhaseTransition,
    SensorimotorControlPoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    ContactEventType,
    ForceDynamicEventType,
    MechanicalEventType,
    PhaseBoundaryEventType,
    ThresholdCrossingEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    EstimateRepresentation,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    ContactPropertyEstimate,
    DynamicsEstimate,
    Estimate,
    ForceRateSetpoint,
    MotorSetpoint,
    ObjectPropertyEstimate,
    ObjectStateEstimate,
    SafetyMargin,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.phases import ActionPhase
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.prediction import (
    SignalTrajectory,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import (
    Actor,
    Manipulandum,
    RoleSlot,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.structure import (
    ActionPhaseStructure,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE, FLANAGAN
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

# %% fixtures

CLAIM_IDENTIFIER = re.compile(r"FJ-\d{3}")
"""
The shape of an identifier in the scientific contract's claim matrix.
"""

UNCLASSIFIED = {
    PhaseBoundaryEventType,
    ThresholdCrossingEventType,
    Estimate,
    ContactPropertyEstimate,
    ObjectStateEstimate,
    DynamicsEstimate,
    EstimateRepresentation,
    SignalTrajectory,
}
"""
The core concepts the scientific contract gives no basis for.

Nothing invents a status for them, so this set only shrinks, as the contract is extended
to cover them.
"""


def status_of(cls: type) -> EvidenceStatus | None:
    """
    Return the evidential status a core concept records.
    """
    return FLANAGAN.find_concept(cls).status


def claims_of(cls: type) -> tuple[str, ...]:
    """
    Return the source claims a core concept cites.
    """
    return FLANAGAN.find_concept(cls).claims


# %% coverage


class TestCoreProvenance:
    """
    Each core concept says how well the source supports it.
    """

    def test_exactly_the_concepts_without_a_basis_in_the_contract_are_unclassified(
        self,
    ) -> None:
        unclassified = {
            meta.cls for meta in FLANAGAN.concepts(CORE.name) if meta.status is None
        }

        assert unclassified == UNCLASSIFIED

    def test_every_cited_claim_has_the_shape_of_a_claim_identifier(self) -> None:
        cited = [
            claim for meta in FLANAGAN.concepts(CORE.name) for claim in meta.claims
        ]

        assert cited
        assert all(CLAIM_IDENTIFIER.fullmatch(claim) for claim in cited)


# %% what the contract says


class TestRecordedStatuses:
    """
    Statuses follow the classification of the scientific contract.
    """

    def test_a_concept_the_source_reports_directly_is_an_empirical_finding(
        self,
    ) -> None:
        assert status_of(Actor) is EvidenceStatus.EMPIRICAL_FINDING
        assert status_of(ForceRateSetpoint) is EvidenceStatus.EMPIRICAL_FINDING
        assert status_of(SafetyMargin) is EvidenceStatus.EMPIRICAL_FINDING
        assert status_of(ObjectPropertyEstimate) is EvidenceStatus.EMPIRICAL_FINDING

    def test_a_concept_the_source_infers_is_a_supported_interpretation(self) -> None:
        assert status_of(Manipulandum) is EvidenceStatus.SUPPORTED_INTERPRETATION
        assert status_of(MotorSetpoint) is EvidenceStatus.SUPPORTED_INTERPRETATION

    def test_the_sensorimotor_framework_is_an_author_hypothesis(self) -> None:
        for cls in (ActionPhase, Subgoal, SensorimotorControlPoint):
            assert status_of(cls) is EvidenceStatus.AUTHOR_HYPOTHESIS

    def test_a_concept_the_contract_rates_two_ways_takes_the_weaker_status(
        self,
    ) -> None:
        assert status_of(ContactEventType) is EvidenceStatus.AUTHOR_HYPOTHESIS
        assert status_of(MechanicalEventType) is EvidenceStatus.AUTHOR_HYPOTHESIS

    def test_machinery_added_for_application_reasoning_is_an_engineering_extension(
        self,
    ) -> None:
        for cls in (
            RoleSlot,
            ConditionRequirement,
            PhaseTransition,
            ActionPhaseStructure,
            ForceDynamicEventType,
        ):
            assert status_of(cls) is EvidenceStatus.ENGINEERING_EXTENSION

    def test_no_concept_claims_that_the_source_leaves_it_an_open_question(self) -> None:
        assert all(
            meta.status is not EvidenceStatus.OPEN_QUESTION
            for meta in FLANAGAN.concepts(CORE.name)
        )


# %% claims


class TestCitedClaims:
    """
    Concepts cite the claims of the contract that they implement.
    """

    @pytest.mark.parametrize(
        ("cls", "claim"),
        [
            (ActionPhaseStructure, "FJ-001"),
            (SensorimotorControlPoint, "FJ-001"),
            (ActionPhase, "FJ-003"),
            (Subgoal, "FJ-003"),
            (PhaseTransition, "FJ-003"),
            (MechanicalEventType, "FJ-004"),
            (CorrectiveAction, "FJ-005"),
            (ModelUpdate, "FJ-010"),
        ],
    )
    def test_concept_cites_the_claim_it_implements(self, cls: type, claim: str) -> None:
        assert claim in claims_of(cls)

    def test_a_concept_never_cites_a_claim_without_a_status(self) -> None:
        assert all(
            meta.status is not None
            for meta in FLANAGAN.concepts(CORE.name)
            if meta.claims
        )
