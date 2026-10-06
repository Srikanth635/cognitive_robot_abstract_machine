from __future__ import annotations

from llm_reasoner.knowledge.microworlds.flanagan.constraints import SUITE
from llm_reasoner.knowledge.microworlds.flanagan.core import (
    FLAN,
    ActionPhase,
    CategoricalValue,
    ConditionMode,
    ContactEventType,
    ContactPropertyEstimate,
    ContactTransition,
    ControlEventType,
    CouplingRegime,
    DynamicsEstimate,
    ExternalTerminationEventType,
    ForceDynamicEventType,
    MagnitudeRange,
    MeasurementUnit,
    ObjectPropertyEstimate,
    ObjectStateEstimate,
    ParameterValueForm,
    PhysicalDimension,
    RegulationMode,
    RoleSlot,
    Subgoal,
    TactileSignal,
    TaskSetpoint,
)
from llm_reasoner.knowledge.microworlds.flanagan.pouring import (
    POUR,
    CenterOfMassShiftEstimate,
    DynamicViscosityEstimate,
    RemainingSourceQuantityEstimate,
    SloshDynamicsEstimate,
    SourceContainer,
    TargetQuantityReached,
    TransferredQuantitySetpoint,
)
from llm_reasoner.knowledge.microworlds.flanagan.prehension import (
    ExpectedCenterOfMassPosition,
    ExpectedFrictionCoefficient,
    ExpectedMass,
    ExpectedShape,
    Liftoff,
    LiftoffThreshold,
    Touchdown,
    TouchdownThreshold,
)
from llm_reasoner.knowledge.ontological.decorators import REGISTRY, ConceptReference

# %% test data


def make_source_role() -> RoleSlot:
    """
    Make a valid source-container role slot.
    """
    return RoleSlot(
        label="source container",
        binds_as=ConceptReference(
            iri=POUR + "SourceContainer",
            label=SourceContainer.__name__,
        ),
    )


def make_magnitude(
    lower_bound: float,
    upper_bound: float,
    unit: MeasurementUnit,
) -> MagnitudeRange:
    """
    Make a dimensionally consistent magnitude.
    """
    return MagnitudeRange(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        physical_dimension=unit.physical_dimension,
        unit=unit,
    )


# %% value partitions


class TestValuePartitionRegistration:
    """
    Each orthogonal enum owns its ontology value partition.
    """

    def test_condition_mode_has_its_own_partition(self) -> None:
        metadata = REGISTRY.meta_for(ConditionMode)

        assert metadata is not None
        assert metadata.iri == FLAN + "ConditionMode"

    def test_parameter_value_form_has_its_own_partition(self) -> None:
        metadata = REGISTRY.meta_for(ParameterValueForm)

        assert metadata is not None
        assert metadata.iri == FLAN + "ParameterValueForm"


# %% orthogonal event characteristics


class TestLiftoffCharacteristics:
    """
    Liftoff combines contact and force-threshold characteristics.
    """

    def test_liftoff_breaks_contact_and_crosses_its_threshold(self) -> None:
        role = make_source_role()
        threshold = LiftoffThreshold(
            label="liftoff threshold",
            parameter_range=make_magnitude(4.0, 4.0, MeasurementUnit.NEWTON),
            parameter_on_role=[role],
        )
        liftoff = Liftoff(
            label="liftoff",
            contact_transition=ContactTransition.BREAKS,
            crosses_threshold=threshold,
            event_on_role=[role],
        )

        assert isinstance(liftoff, ContactEventType)
        assert isinstance(liftoff, ForceDynamicEventType)
        assert liftoff.contact_transition is ContactTransition.BREAKS
        assert liftoff.crosses_threshold is threshold

    def test_liftoff_without_a_threshold_is_rejected(self) -> None:
        role = make_source_role()
        liftoff = Liftoff(
            label="incomplete liftoff",
            contact_transition=ContactTransition.BREAKS,
            event_on_role=[role],
        )

        violations = SUITE.evaluate((role, liftoff))

        assert any(
            violation.check == "REQ" and "crosses_threshold" in violation.message
            for violation in violations
        )


class TestTouchdownCharacteristics:
    """
    Touchdown combines contact and force-threshold characteristics.
    """

    def test_touchdown_establishes_contact_and_crosses_its_threshold(self) -> None:
        role = make_source_role()
        threshold = TouchdownThreshold(
            label="touchdown threshold",
            parameter_range=make_magnitude(1.0, 2.0, MeasurementUnit.NEWTON),
            parameter_on_role=[role],
        )
        touchdown = Touchdown(
            label="touchdown",
            contact_transition=ContactTransition.ESTABLISHES,
            crosses_threshold=threshold,
            event_on_role=[role],
        )

        assert isinstance(touchdown, ContactEventType)
        assert isinstance(touchdown, ForceDynamicEventType)
        assert touchdown.contact_transition is ContactTransition.ESTABLISHES
        assert touchdown.crosses_threshold is threshold


class TestSetpointAttainmentCharacteristic:
    """
    Setpoint attainment names the setpoint reached by the event.
    """

    def test_target_quantity_event_references_its_task_setpoint(self) -> None:
        role = make_source_role()
        target = TransferredQuantitySetpoint(
            label="target volume",
            parameter_range=make_magnitude(
                200.0,
                400.0,
                MeasurementUnit.MILLILITRE,
            ),
            parameter_on_role=[role],
        )
        reached = TargetQuantityReached(
            label="target reached",
            attains_setpoint=[target],
            event_on_role=[role],
        )

        assert isinstance(reached, ControlEventType)
        assert reached.attains_setpoint == [target]


# %% phase termination and regulation


class TestIndependentPhaseControlDimensions:
    """
    Boundary events and within-phase regulation remain independent.
    """

    def test_hybrid_phase_can_still_end_at_a_discrete_event(self) -> None:
        role = make_source_role()
        target = TransferredQuantitySetpoint(
            label="target volume",
            parameter_range=make_magnitude(
                200.0,
                400.0,
                MeasurementUnit.MILLILITRE,
            ),
            parameter_on_role=[role],
        )
        boundary = TargetQuantityReached(
            label="target reached",
            attains_setpoint=[target],
            event_on_role=[role],
        )
        subgoal = Subgoal(
            label="transfer target quantity",
            condition_on_role=[role],
            attainment_signalled_by=boundary,
        )
        phase = ActionPhase(
            label="regulated transfer",
            aims_for=subgoal,
            coupling_regime=CouplingRegime.COUPLED,
            regulation_mode=RegulationMode.HYBRID,
            terminated_by=[boundary],
            phase_involves=[role],
            monitors_signal=[TactileSignal(label="contact force")],
        )

        assert phase.regulation_mode is RegulationMode.HYBRID
        assert phase.terminated_by == [boundary]


class TestExternalPhaseTermination:
    """
    A phase may end through a typed external termination.
    """

    def test_external_termination_is_a_phase_boundary(self) -> None:
        role = make_source_role()
        termination = ExternalTerminationEventType(
            label="operator stop",
            triggered_by=ConceptReference(
                iri="urn:example:operator-stop",
                label="operator stop request",
            ),
        )
        subgoal = Subgoal(
            label="safely stop",
            condition_on_role=[role],
            attainment_signalled_by=termination,
        )
        phase = ActionPhase(
            label="externally terminated phase",
            aims_for=subgoal,
            coupling_regime=CouplingRegime.UNCOUPLED,
            regulation_mode=RegulationMode.FEEDFORWARD,
            terminated_by=[termination],
            phase_involves=[role],
        )

        assert phase.terminated_by == [termination]
        assert termination.triggered_by.iri == "urn:example:operator-stop"


# %% parameter categories


class TestParameterCategorySeparation:
    """
    Estimates and setpoints use their distinct epistemic categories.
    """

    def test_pouring_parameters_have_their_specific_categories(self) -> None:
        assert issubclass(ExpectedMass, ObjectPropertyEstimate)
        assert issubclass(ExpectedFrictionCoefficient, ContactPropertyEstimate)
        assert issubclass(ExpectedCenterOfMassPosition, ObjectStateEstimate)
        assert issubclass(RemainingSourceQuantityEstimate, ObjectStateEstimate)
        assert issubclass(SloshDynamicsEstimate, DynamicsEstimate)
        assert issubclass(CenterOfMassShiftEstimate, DynamicsEstimate)
        assert issubclass(DynamicViscosityEstimate, ObjectPropertyEstimate)
        assert issubclass(TransferredQuantitySetpoint, TaskSetpoint)


class TestCategoricalParameterValue:
    """
    Categorical estimates do not fabricate numeric ranges.
    """

    def test_shape_uses_a_categorical_value_only(self) -> None:
        role = make_source_role()
        shape_value = CategoricalValue(
            label="rectangular prism",
            category=ConceptReference(
                iri="urn:shape:rectangular-prism",
                label="rectangular prism",
            ),
        )
        shape = ExpectedShape(
            label="expected shape",
            categorical_value=shape_value,
            parameter_on_role=[role],
        )

        violations = SUITE.evaluate((role, shape_value, shape))

        assert shape.parameter_range is None
        assert shape.categorical_value is shape_value
        assert not any(violation.check == "C34" for violation in violations)

    def test_shape_with_only_a_numeric_value_is_rejected(self) -> None:
        role = make_source_role()
        shape = ExpectedShape(
            label="numeric shape",
            parameter_range=make_magnitude(
                1.0,
                1.0,
                MeasurementUnit.DIMENSIONLESS,
            ),
            parameter_on_role=[role],
        )

        violations = SUITE.evaluate((role, shape))

        assert any(violation.check == "C36" for violation in violations)

    def test_parameter_with_numeric_and_categorical_values_is_rejected(self) -> None:
        role = make_source_role()
        shape = ExpectedShape(
            label="contradictory shape",
            parameter_range=make_magnitude(
                1.0,
                1.0,
                MeasurementUnit.DIMENSIONLESS,
            ),
            categorical_value=CategoricalValue(
                label="rectangular prism",
                category=ConceptReference(iri="urn:shape:rectangular-prism"),
            ),
            parameter_on_role=[role],
        )

        violations = SUITE.evaluate((role, shape))

        assert any(violation.check == "C34" for violation in violations)


# %% physical dimensions


class TestPhysicalDimensionValidation:
    """
    Units and parameter meanings remain dimensionally consistent.
    """

    def test_measurement_unit_supplies_an_omitted_dimension(self) -> None:
        magnitude = MagnitudeRange(
            lower_bound=1.0,
            upper_bound=2.0,
            unit=MeasurementUnit.NEWTON,
        )

        assert magnitude.physical_dimension is PhysicalDimension.FORCE

    def test_unit_with_incompatible_physical_dimension_is_rejected(self) -> None:
        magnitude = MagnitudeRange(
            label="invalid magnitude",
            lower_bound=1.0,
            upper_bound=2.0,
            physical_dimension=PhysicalDimension.MASS,
            unit=MeasurementUnit.NEWTON,
        )

        violations = SUITE.evaluate((magnitude,))

        assert any(violation.check == "C33" for violation in violations)

    def test_force_threshold_cannot_use_a_mass_quantity(self) -> None:
        role = make_source_role()
        threshold = LiftoffThreshold(
            label="invalid liftoff threshold",
            parameter_range=make_magnitude(
                0.3,
                0.6,
                MeasurementUnit.KILOGRAM,
            ),
            parameter_on_role=[role],
        )

        violations = SUITE.evaluate((role, threshold))

        assert any(violation.check == "C35" for violation in violations)
