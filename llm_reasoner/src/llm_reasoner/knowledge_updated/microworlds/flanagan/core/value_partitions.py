"""
The closed sets of alternatives the core vocabulary chooses between.

These are orthogonal dimensions of a phase, an event or a comparison, so they are
enumerations and not sibling classes: a phase can be coupled and continuously regulated
at once.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping

from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE

# %% phases


@CORE.value_partition()
class CouplingRegime(StrEnum):
    """
    Whether the effector and the manipulandum form a single kinematic unit.
    """

    UNCOUPLED = "uncoupled"
    """
    The effector moves without being mechanically bound to the object.
    """

    TRANSITIONAL = "transitional"
    """
    Contact or force conditions are changing but the object does not translate.
    """

    COUPLED = "coupled"
    """
    The effector and the object move together as one kinematic unit.
    """


@CORE.value_partition()
class RegulationMode(StrEnum):
    """
    How control commands are regulated within a phase.
    """

    FEEDFORWARD = "feedforward"
    """
    Commands are selected from predictions before sensory correction.
    """

    CONTINUOUS_FEEDBACK = "continuous_feedback"
    """
    Commands are continuously corrected from monitored sensory signals.
    """

    HYBRID = "hybrid"
    """
    Feedforward predictions and continuous sensory correction cooperate.
    """


@CORE.value_partition()
class ConditionMode(StrEnum):
    """
    When, relative to a phase, a condition is demanded.
    """

    AT_ENTRY = "at_entry"
    """
    Must hold for the phase to begin: the classical precondition.
    """

    THROUGHOUT = "throughout"
    """
    Must hold at every instant of the phase.

    An invariant, not a precondition: its violation is a failure event and not a failed
    entry test.
    """

    AT_EXIT = "at_exit"
    """
    Is brought about by successful completion: the classical postcondition.
    """


@CORE.value_partition()
class ConditionPolarity(StrEnum):
    """
    Whether a condition is demanded to hold or to be absent.

    Carried by the requirement and not by the condition, so one condition such as
    "receptacle full" can be required by one phase and forbidden by another.
    """

    MUST_HOLD = "must_hold"
    """
    The condition is demanded to be true.
    """

    MUST_NOT_HOLD = "must_not_hold"
    """
    The condition is demanded to be false.
    """


# %% events


@CORE.value_partition()
class ContactTransition(StrEnum):
    """
    The change in contact carried by a mechanical event.
    """

    ESTABLISHES = "establishes"
    """
    The event creates contact between its participants.
    """

    BREAKS = "breaks"
    """
    The event removes contact between its participants.
    """


# %% sensing


@CORE.value_partition()
class SensoryModality(StrEnum):
    """
    The sensory channel that carries evidence of an event.
    """

    TACTILE = "tactile"
    """
    Fingertip afferent signals, the primary channel for contact events.
    """

    PROPRIOCEPTIVE = "proprioceptive"
    """
    Signals of limb configuration and effort, which complement tactile signals while the
    effector and the object are coupled.
    """

    VISUAL = "visual"
    """
    Exteroceptive visual signals, used to anticipate contact before it occurs.
    """

    AUDITORY = "auditory"
    """
    Acoustic evidence about an event or a continuously changing state.
    """


@CORE.value_partition()
class TrajectoryTrend(StrEnum):
    """
    The direction of change expected across a signal trajectory.
    """

    INCREASING = "increasing"
    """
    The signal magnitude rises over the prediction interval.
    """

    DECREASING = "decreasing"
    """
    The signal magnitude falls over the prediction interval.
    """

    STABLE = "stable"
    """
    The signal remains within a bounded steady range.
    """


# %% comparison and recovery


@CORE.value_partition()
class ComparisonOutcome(StrEnum):
    """
    Whether an observation agrees with its sensory prediction.
    """

    MATCH = "match"
    """
    The observation supports the prediction within its declared tolerances.
    """

    MISMATCH = "mismatch"
    """
    The observation differs from the prediction in a classified way.
    """


@CORE.value_partition()
class PredictionErrorKind(StrEnum):
    """
    The dimension along which an observation violates a prediction.
    """

    ABSENT = "absent"
    """
    The predicted event or feature was not observed.
    """

    UNEXPECTED = "unexpected"
    """
    An event or feature outside the prediction was observed.
    """

    EARLY = "early"
    """
    The predicted event or feature occurred before its timing window.
    """

    LATE = "late"
    """
    The predicted event or feature occurred after its timing window.
    """

    MAGNITUDE_MISMATCH = "magnitude_mismatch"
    """
    The observed magnitude fell outside the prediction and its tolerance.
    """

    DIRECTION_MISMATCH = "direction_mismatch"
    """
    The observed change proceeded in the opposite direction.
    """

    SPATIAL_MISMATCH = "spatial_mismatch"
    """
    The observation involved a different location or participant.
    """


@CORE.value_partition()
class CorrectionDirection(StrEnum):
    """
    How a corrective response changes its target setpoint.
    """

    INCREASE = "increase"
    """
    Raise the target setpoint.
    """

    DECREASE = "decrease"
    """
    Lower the target setpoint.
    """

    HOLD = "hold"
    """
    Maintain the target at its present value.
    """

    REVERSE = "reverse"
    """
    Command a change opposite to the current direction.
    """

    STOP = "stop"
    """
    Bring the commanded change to rest.
    """


@CORE.value_partition()
class RecoveryMode(StrEnum):
    """
    How execution proceeds after a corrective response.
    """

    CONTINUE_PHASE = "continue_phase"
    """
    Apply the correction without restarting the phase.
    """

    REPEAT_PHASE = "repeat_phase"
    """
    Restart the current phase with corrected control parameters.
    """

    REPLAN = "replan"
    """
    Select a new phase sequence from the updated state.
    """

    ABORT = "abort"
    """
    Terminate the manipulation safely.
    """


@CORE.value_partition()
class EstimateUpdateDirection(StrEnum):
    """
    How evidence changes a represented estimate.
    """

    INCREASE = "increase"
    """
    Revise the estimate upward.
    """

    DECREASE = "decrease"
    """
    Revise the estimate downward.
    """

    REVISE = "revise"
    """
    Change an unordered estimate or replace its current value.
    """


# %% quantities


@CORE.value_partition()
class PhysicalDimension(StrEnum):
    """
    A physical dimension used to check quantitative knowledge.
    """

    MASS = "mass"
    """
    Quantity of matter.
    """

    MASS_FLOW_RATE = "mass_flow_rate"
    """
    Mass transferred per unit time.
    """

    FORCE = "force"
    """
    Mechanical force, including weight force.
    """

    FORCE_RATE = "force_rate"
    """
    Rate of force change.
    """

    TORQUE = "torque"
    """
    Moment of force.
    """

    LENGTH = "length"
    """
    Spatial distance or position component.
    """

    ANGLE = "angle"
    """
    Planar or rotational angle.
    """

    ANGULAR_RATE = "angular_rate"
    """
    Rate of angular change.
    """

    VOLUME = "volume"
    """
    Three-dimensional capacity.
    """

    VOLUMETRIC_FLOW_RATE = "volumetric_flow_rate"
    """
    Volume transferred per unit time.
    """

    DYNAMIC_VISCOSITY = "dynamic_viscosity"
    """
    Resistance of a fluid to shear flow.
    """

    FREQUENCY = "frequency"
    """
    Repetitions of a periodic signal per unit time.
    """

    TIME = "time"
    """
    Temporal duration or offset.
    """

    DIMENSIONLESS = "dimensionless"
    """
    A ratio or fraction without a physical unit.
    """


@CORE.value_partition()
class MeasurementUnit(StrEnum):
    """
    A supported measurement unit, which measures exactly one physical dimension.
    """

    KILOGRAM = "kg"
    """
    Kilogram, a unit of mass.
    """

    KILOGRAM_PER_SECOND = "kg/s"
    """
    Kilogram per second, a unit of mass flow rate.
    """

    NEWTON = "N"
    """
    Newton, a unit of force.
    """

    NEWTON_PER_SECOND = "N/s"
    """
    Newton per second, a unit of force rate.
    """

    NEWTON_METRE = "N*m"
    """
    Newton metre, a unit of torque.
    """

    METRE = "m"
    """
    Metre, a unit of length.
    """

    DEGREE = "deg"
    """
    Degree, a unit of angle.
    """

    DEGREE_PER_SECOND = "deg/s"
    """
    Degree per second, a unit of angular rate.
    """

    MILLILITRE = "mL"
    """
    Millilitre, a unit of volume.
    """

    MILLILITRE_PER_SECOND = "mL/s"
    """
    Millilitre per second, a unit of volumetric flow rate.
    """

    CENTIPOISE = "cP"
    """
    Centipoise, a unit of dynamic viscosity.
    """

    HERTZ = "Hz"
    """
    Hertz, a unit of frequency.
    """

    SECOND = "s"
    """
    Second, a unit of time.
    """

    MILLISECOND = "ms"
    """
    Millisecond, a unit of time.
    """

    DIMENSIONLESS = "dimensionless"
    """
    Unit marker for dimensionless ratios.
    """

    FRACTION = "fraction"
    """
    Unit marker for bounded dimensionless fractions.
    """

    @property
    def physical_dimension(self) -> PhysicalDimension:
        """
        The physical dimension this unit measures.
        """
        return UNIT_DIMENSIONS[self]


@CORE.value_partition()
class ParameterValueForm(StrEnum):
    """
    Whether a parameter holds a magnitude or a named category.
    """

    QUANTITATIVE = "quantitative"
    """
    The parameter holds a magnitude range with a unit.
    """

    CATEGORICAL = "categorical"
    """
    The parameter holds a categorical value.
    """


UNIT_DIMENSIONS: Mapping[MeasurementUnit, PhysicalDimension] = {
    MeasurementUnit.KILOGRAM: PhysicalDimension.MASS,
    MeasurementUnit.KILOGRAM_PER_SECOND: PhysicalDimension.MASS_FLOW_RATE,
    MeasurementUnit.NEWTON: PhysicalDimension.FORCE,
    MeasurementUnit.NEWTON_PER_SECOND: PhysicalDimension.FORCE_RATE,
    MeasurementUnit.NEWTON_METRE: PhysicalDimension.TORQUE,
    MeasurementUnit.METRE: PhysicalDimension.LENGTH,
    MeasurementUnit.DEGREE: PhysicalDimension.ANGLE,
    MeasurementUnit.DEGREE_PER_SECOND: PhysicalDimension.ANGULAR_RATE,
    MeasurementUnit.MILLILITRE: PhysicalDimension.VOLUME,
    MeasurementUnit.MILLILITRE_PER_SECOND: PhysicalDimension.VOLUMETRIC_FLOW_RATE,
    MeasurementUnit.CENTIPOISE: PhysicalDimension.DYNAMIC_VISCOSITY,
    MeasurementUnit.HERTZ: PhysicalDimension.FREQUENCY,
    MeasurementUnit.SECOND: PhysicalDimension.TIME,
    MeasurementUnit.MILLISECOND: PhysicalDimension.TIME,
    MeasurementUnit.DIMENSIONLESS: PhysicalDimension.DIMENSIONLESS,
    MeasurementUnit.FRACTION: PhysicalDimension.DIMENSIONLESS,
}
"""
The physical dimension each measurement unit measures.
"""

TIME_UNITS = frozenset(
    unit
    for unit in MeasurementUnit
    if unit.physical_dimension is PhysicalDimension.TIME
)
"""
The units that measure time.
"""
