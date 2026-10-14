"""
A pouring phase structure built on the lift-and-replace structure of prehension.

The pouring phases are inserted between transporting and replacing: the lip is aligned
with the receptacle, the container is tilted until the liquid flows, flow is regulated
against the transferred quantity, stopped, and the container is returned upright.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from llm_reasoner.knowledge_updated.microworlds.flanagan.core.conditions import (
    PhaseConditionType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.control import (
    BoundaryControlExemption,
    PhaseTransition,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    ForwardModel,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.prediction import (
    ContinuousSignalPrediction,
    SensoryObservationType,
    SignalTrajectory,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    CouplingRegime,
    MeasurementUnit,
    ParameterValueForm,
    PhysicalDimension,
    RegulationMode,
    SensoryModality,
    TrajectoryTrend,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.representations import (
    ObjectDynamicsModel,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.events import (
    FlowCessation,
    FlowOnset,
    LipAligned,
    SpillEvent,
    TargetQuantityReached,
    UprightRestored,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.parameters import (
    DynamicViscosityEstimate,
    FlowOnsetThreshold,
    LipAlignmentSetpoint,
    SloshDynamicsEstimate,
    SpillThreshold,
    TiltRateSetpoint,
    TransferredQuantitySetpoint,
    UprightAngleSetpoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.phases import (
    AlignmentPhase,
    FlowRegulationPhase,
    FlowTerminationPhase,
    ReturnUprightPhase,
    TiltToFlowPhase,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.roles import (
    PouredSubstance,
    PouringLip,
    SourceContainer,
    TargetReceptacle,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.structure import (
    PouringPhaseStructure,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.value_partitions import (
    MaterialRegime,
    TransferTerminationReason,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.references import ConceptReference

from ..core_structure import SCOPE, fragments_of, magnitude, role_of, window
from ..prehension.prehension_structure import (
    PrehensionParts,
    build_valid_prehension,
    category,
    make_phase,
)

FEATURE = ConceptReference(iri="urn:test:feature/poured-weight")
"""
A sensory feature, named by a term of some other vocabulary.
"""


def parameter(cls: type, label: str, dimension, unit, low, high, roles):
    """
    Return a quantitative parameter of a dimension.
    """
    return cls(
        label=label,
        value_form=ParameterValueForm.QUANTITATIVE,
        dimension=dimension,
        parameter_range=magnitude(low, high, unit),
        parameter_on_role=roles,
    )


# %% the structure


@dataclass
class PouringParts:
    """
    The structure and every named part of it, so a test can break one thing.
    """

    prehension: PrehensionParts
    structure: PouringPhaseStructure
    viscosity: DynamicViscosityEstimate
    quantity: TransferredQuantitySetpoint
    onset_threshold: FlowOnsetThreshold
    spill_threshold: SpillThreshold
    tilt_rate: TiltRateSetpoint
    aligned: LipAligned
    onset: FlowOnset
    target: TargetQuantityReached
    cessation: FlowCessation
    upright_restored: UprightRestored
    spill: SpillEvent
    align: AlignmentPhase
    tilt: TiltToFlowPhase
    flow: FlowRegulationPhase
    stop: FlowTerminationPhase
    upright: ReturnUprightPhase
    transitions: list[PhaseTransition]

    def fragments(self) -> list[KnowledgeEntity]:
        """
        Return the structure and every fragment reachable from it, nested ones included.
        """
        return fragments_of(self.structure)


def build_valid_pouring() -> PouringParts:
    """
    Build a fresh structure that satisfies every acceptance rule of the core, of object
    dynamics, of prehension and of pouring.
    """
    base = build_valid_prehension()
    old = base.structure
    container = role_of(SourceContainer, "source container")
    receptacle = role_of(TargetReceptacle, "target receptacle")
    substance = role_of(PouredSubstance, "poured liquid")
    lip = role_of(PouringLip, "pouring lip")
    flowing = [substance, lip]

    viscosity = parameter(
        DynamicViscosityEstimate,
        "expected viscosity",
        PhysicalDimension.DYNAMIC_VISCOSITY,
        MeasurementUnit.CENTIPOISE,
        1.0,
        5.0,
        [substance],
    )
    slosh = category(SloshDynamicsEstimate, "expected slosh", "slosh", [substance])
    quantity = parameter(
        TransferredQuantitySetpoint,
        "quantity to pour",
        PhysicalDimension.VOLUME,
        MeasurementUnit.MILLILITRE,
        100.0,
        200.0,
        [receptacle],
    )
    alignment = parameter(
        LipAlignmentSetpoint,
        "lip alignment",
        PhysicalDimension.ANGLE,
        MeasurementUnit.DEGREE,
        0.0,
        5.0,
        [container],
    )
    upright_angle = parameter(
        UprightAngleSetpoint,
        "upright angle",
        PhysicalDimension.ANGLE,
        MeasurementUnit.DEGREE,
        0.0,
        5.0,
        [container],
    )
    tilt_rate = parameter(
        TiltRateSetpoint,
        "tilt rate",
        PhysicalDimension.ANGULAR_RATE,
        MeasurementUnit.DEGREE_PER_SECOND,
        1.0,
        10.0,
        [container],
    )
    onset_threshold = parameter(
        FlowOnsetThreshold,
        "onset angle",
        PhysicalDimension.ANGLE,
        MeasurementUnit.DEGREE,
        30.0,
        60.0,
        [container],
    )
    spill_threshold = parameter(
        SpillThreshold,
        "spill rate",
        PhysicalDimension.VOLUMETRIC_FLOW_RATE,
        MeasurementUnit.MILLILITRE_PER_SECOND,
        50.0,
        100.0,
        [substance],
    )

    aligned = LipAligned(
        label="lip aligned", event_on_role=[container], attains_setpoint=[alignment]
    )
    onset = FlowOnset(
        label="flow onset", event_on_role=flowing, crosses_threshold=onset_threshold
    )
    target = TargetQuantityReached(
        label="target reached", event_on_role=[receptacle], attains_setpoint=[quantity]
    )
    cessation = FlowCessation(label="flow ceased", event_on_role=flowing)
    upright_restored = UprightRestored(
        label="upright", event_on_role=[container], attains_setpoint=[upright_angle]
    )
    spill = SpillEvent(
        label="spill",
        is_failure_event=True,
        event_on_role=[substance, receptacle],
        crosses_threshold=spill_threshold,
    )

    def state(label: str, roles: list) -> PhaseConditionType:
        return PhaseConditionType(label=label, condition_on_role=roles)

    forward_model = next(
        model for model in old.defines_internal_model if isinstance(model, ForwardModel)
    )
    observation = SensoryObservationType(
        label="poured weight observed",
        modality=SensoryModality.TACTILE,
        observes_feature=FEATURE,
        observed_at=window(0.0, 1.0),
        observed_on_role=[container],
    )
    control_law = ContinuousSignalPrediction(
        label="poured weight rises",
        modality=SensoryModality.TACTILE,
        predicted_trajectory=SignalTrajectory(
            signal_feature=FEATURE,
            trend=TrajectoryTrend.INCREASING,
            expected_range=magnitude(0.0, 2.0, MeasurementUnit.NEWTON),
        ),
        tolerance=magnitude(0.0, 0.2, MeasurementUnit.NEWTON),
        generated_by=forward_model,
        comparison_target=observation,
        controlled_variable=tilt_rate,
    )

    transport = next(p for p in old.defines_phase if p.label == "transport")
    replace = base.replace
    transitional, coupled = (
        CouplingRegime.TRANSITIONAL,
        CouplingRegime.COUPLED,
    )
    align = make_phase(
        AlignmentPhase,
        "align",
        state("lip over receptacle", [lip, receptacle]),
        aligned,
        coupled,
        [alignment],
        [container, receptacle],
        transport.terminated_by,
    )
    tilt = make_phase(
        TiltToFlowPhase,
        "tilt",
        state("liquid at the lip", flowing),
        onset,
        coupled,
        [tilt_rate, onset_threshold],
        [container, substance],
        [aligned],
    )
    flow = make_phase(
        FlowRegulationPhase,
        "pour",
        state("quantity poured", [receptacle]),
        target,
        coupled,
        [tilt_rate, quantity, spill_threshold],
        [container, substance],
        [onset],
        regulation=RegulationMode.CONTINUOUS_FEEDBACK,
    )
    flow.monitors_modality = [SensoryModality.TACTILE]
    flow.continuous_predictions = [control_law]
    stop = make_phase(
        FlowTerminationPhase,
        "stop flow",
        state("liquid at rest", flowing),
        cessation,
        coupled,
        [tilt_rate],
        [container, substance],
        [target],
    )
    upright = make_phase(
        ReturnUprightPhase,
        "return upright",
        state("container upright", [container]),
        upright_restored,
        transitional,
        [upright_angle],
        [container],
        [cessation],
    )
    replace.initiated_by = [upright_restored]

    added = [align, tilt, flow, stop, upright]
    chain = [transport, *added, replace]
    kept = [
        t
        for t in old.defines_transition
        if not (t.from_phase is transport and t.to_phase is replace)
    ]
    new_transitions = [
        PhaseTransition(
            label=f"{earlier.label} to {later.label}",
            from_phase=earlier,
            to_phase=later,
            gated_by=[later.initiated_by[0]],
        )
        for earlier, later in zip(chain, chain[1:])
    ]

    dynamics = next(
        model
        for model in old.defines_internal_model
        if isinstance(model, ObjectDynamicsModel)
    )
    dynamics.holds_estimate = [*dynamics.holds_estimate, viscosity, slosh]
    dynamics.represents_role = [*dynamics.represents_role, substance]

    events = [aligned, onset, target, cessation, upright_restored]
    inherited = {
        item.name: old.__dict__[item.name] for item in fields(old) if item.init
    }
    inherited = {
        name: list(value) if isinstance(value, list) else value
        for name, value in inherited.items()
    }
    position = inherited["defines_phase"].index(replace)
    inherited["defines_phase"][position:position] = added
    inherited["defines_transition"] = [*kept, *new_transitions]
    inherited["defines_role"] += [container, receptacle, substance, lip]
    inherited["defines_event_type"] += [*events, spill]
    inherited["defines_parameter"] += [
        viscosity,
        slosh,
        quantity,
        alignment,
        upright_angle,
        tilt_rate,
        onset_threshold,
        spill_threshold,
    ]
    inherited["defines_observation"] = [observation]
    inherited["defines_continuous_prediction"] = [control_law]
    inherited["defines_control_point_exemption"] += [
        BoundaryControlExemption(
            label=f"{event.label} exemption", exempts_event=event, justification=SCOPE
        )
        for event in events
    ]
    structure = PouringPhaseStructure(
        **inherited,
        material_regime=MaterialRegime.LIQUID,
        termination_reason=TransferTerminationReason.TARGET_QUANTITY,
    )
    return PouringParts(
        prehension=base,
        structure=structure,
        viscosity=viscosity,
        quantity=quantity,
        onset_threshold=onset_threshold,
        spill_threshold=spill_threshold,
        tilt_rate=tilt_rate,
        aligned=aligned,
        onset=onset,
        target=target,
        cessation=cessation,
        upright_restored=upright_restored,
        spill=spill,
        align=align,
        tilt=tilt,
        flow=flow,
        stop=stop,
        upright=upright,
        transitions=new_transitions,
    )
