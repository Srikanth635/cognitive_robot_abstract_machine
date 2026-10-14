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
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.control import (
    CorrectiveAction,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    ContactEventType,
    ForceDynamicEventType,
    SetpointAttained,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    ContactPropertyEstimate,
    ForceRateSetpoint,
    MechanicalThreshold,
    MotorSetpoint,
    ObjectPropertyEstimate,
    ObjectStateEstimate,
    SafetyMargin,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.phases import ActionPhase
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.prediction import (
    SensoryEventPrediction,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.quantities import (
    TemporalWindow,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import (
    Manipulandum,
    ManipulationRole,
    RoleSlot,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ContactTransition,
    CorrectionDirection,
    CouplingRegime,
    ParameterValueForm,
    PhysicalDimension,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.representations import (
    LoadForceEstimate,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.structure import (
    ObjectDynamicsPhaseStructure,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import (
    CORE,
    FLANAGAN,
    PREHENSION,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.conditions import (
    EffectorObjectContact,
    EffectorPositioned,
    GraspStability,
    ObjectClearOfSupport,
    ObjectSupportContact,
    SafeContactForce,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.control import (
    AnticipatoryGripAdjustment,
    CorrectiveGripResponse,
    GripLoadCoupling,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.events import (
    ContactBreak,
    ContactEstablishment,
    ForcesUnloaded,
    LiftClearanceReached,
    Liftoff,
    ReachTargetReached,
    SlipEvent,
    StableHoldReached,
    Touchdown,
    TransportTargetReached,
    WithdrawalComplete,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.parameters import (
    ExpectedCenterOfMassPosition,
    ExpectedFrictionCoefficient,
    ExpectedMass,
    ExpectedShape,
    GripForceSetpoint,
    LiftClearanceSetpoint,
    LiftoffThreshold,
    LoadForceRateSetpoint,
    LoadForceSetpoint,
    ReachPoseSetpoint,
    SlipThreshold,
    TouchdownThreshold,
    TransportPoseSetpoint,
    WithdrawalDistanceSetpoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.phases import (
    ContactEstablishmentPhase,
    ContactReleasePhase,
    ForceLoadingPhase,
    HoldPhase,
    LiftPhase,
    ReachPhase,
    ReplacePhase,
    TransportPhase,
    UnloadPhase,
    WithdrawPhase,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.roles import (
    FinalSupport,
    GraspingEffector,
    GraspSite,
    InitialSupport,
    PrehensileManipulandum,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.structure import (
    PrehensionPhaseStructure,
)
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.properties import PropertyMeta
from llm_reasoner.knowledge_updated.ontological.ranges import RangeResolver
from llm_reasoner.knowledge_updated.ontological.vocabulary import (
    ConceptLevel,
    EvidenceStatus,
    PropertyKind,
    Rigidity,
)

# %% the specification

PARENTS: dict[type, type] = {
    GraspingEffector: ManipulationRole,
    GraspSite: ManipulationRole,
    InitialSupport: ManipulationRole,
    FinalSupport: ManipulationRole,
    PrehensileManipulandum: Manipulandum,
    ReachTargetReached: SetpointAttained,
    LiftClearanceReached: SetpointAttained,
    StableHoldReached: SetpointAttained,
    TransportTargetReached: SetpointAttained,
    ForcesUnloaded: SetpointAttained,
    WithdrawalComplete: SetpointAttained,
    ContactEstablishment: ContactEventType,
    ContactBreak: ContactEventType,
    Liftoff: ContactEventType,
    Touchdown: ContactEventType,
    SlipEvent: ForceDynamicEventType,
    EffectorObjectContact: PhaseConditionType,
    ObjectSupportContact: PhaseConditionType,
    GraspStability: PhaseConditionType,
    ObjectClearOfSupport: PhaseConditionType,
    EffectorPositioned: PhaseConditionType,
    SafeContactForce: PhaseConditionType,
    ReachPhase: ActionPhase,
    ContactEstablishmentPhase: ActionPhase,
    ForceLoadingPhase: ActionPhase,
    LiftPhase: ActionPhase,
    HoldPhase: ActionPhase,
    TransportPhase: ActionPhase,
    ReplacePhase: ActionPhase,
    UnloadPhase: ActionPhase,
    ContactReleasePhase: ActionPhase,
    WithdrawPhase: ActionPhase,
    ReachPoseSetpoint: MotorSetpoint,
    GripForceSetpoint: MotorSetpoint,
    LoadForceSetpoint: MotorSetpoint,
    LoadForceRateSetpoint: ForceRateSetpoint,
    LiftClearanceSetpoint: MotorSetpoint,
    TransportPoseSetpoint: MotorSetpoint,
    WithdrawalDistanceSetpoint: MotorSetpoint,
    SlipThreshold: MechanicalThreshold,
    LiftoffThreshold: MechanicalThreshold,
    TouchdownThreshold: MechanicalThreshold,
    ExpectedMass: ObjectPropertyEstimate,
    ExpectedShape: ObjectPropertyEstimate,
    ExpectedFrictionCoefficient: ContactPropertyEstimate,
    ExpectedCenterOfMassPosition: ObjectStateEstimate,
    CorrectiveGripResponse: CorrectiveAction,
    PrehensionPhaseStructure: ObjectDynamicsPhaseStructure,
}
"""
Each concept of the module beside the concept it specialises, except the two that
specialise nothing in the core.
"""

CONCEPTS = (*PARENTS, AnticipatoryGripAdjustment, GripLoadCoupling)
"""
Every concept of the module.
"""

QUANTITATIVE = ParameterValueForm.QUANTITATIVE
CATEGORICAL = ParameterValueForm.CATEGORICAL


def quantity(dimension: PhysicalDimension) -> dict[str, object]:
    """
    Return the values fixed for a quantity of a dimension.
    """
    return {"value_form": QUANTITATIVE, "dimension": dimension}


FIXED: dict[type, dict[str, object]] = {
    ReachPhase: {"coupling_regime": CouplingRegime.UNCOUPLED},
    TransportPhase: {"coupling_regime": CouplingRegime.COUPLED},
    ReplacePhase: {"coupling_regime": CouplingRegime.COUPLED},
    ContactEstablishment: {"contact_transition": ContactTransition.ESTABLISHES},
    ContactBreak: {"contact_transition": ContactTransition.BREAKS},
    Liftoff: {"contact_transition": ContactTransition.BREAKS},
    Touchdown: {"contact_transition": ContactTransition.ESTABLISHES},
    ReachPoseSetpoint: {"value_form": CATEGORICAL},
    TransportPoseSetpoint: {"value_form": CATEGORICAL},
    GripForceSetpoint: quantity(PhysicalDimension.FORCE),
    LoadForceSetpoint: quantity(PhysicalDimension.FORCE),
    LiftClearanceSetpoint: quantity(PhysicalDimension.LENGTH),
    WithdrawalDistanceSetpoint: quantity(PhysicalDimension.LENGTH),
    SlipThreshold: quantity(PhysicalDimension.FORCE),
    LiftoffThreshold: quantity(PhysicalDimension.FORCE),
    TouchdownThreshold: quantity(PhysicalDimension.FORCE),
    ExpectedMass: quantity(PhysicalDimension.MASS),
    ExpectedFrictionCoefficient: quantity(PhysicalDimension.DIMENSIONLESS),
    ExpectedCenterOfMassPosition: quantity(PhysicalDimension.LENGTH),
    ExpectedShape: {"value_form": CATEGORICAL},
}
"""
The values each concept fixes for every instance, and nothing else.
"""


class Holds(NamedTuple):
    """
    What one property of a concept must hold.
    """

    kind: PropertyKind
    range: type
    functional: bool
    minimum: int


def relation(range: type, minimum: int = 0, functional: bool = False) -> Holds:
    """
    Describe a property that relates to other concepts.
    """
    return Holds(PropertyKind.RELATION, range, functional, minimum)


def single(range: type) -> Holds:
    """
    Describe a required link to exactly one other concept.
    """
    return relation(range, 1, functional=True)


PROPERTIES: dict[type, dict[str, Holds]] = {
    ReachTargetReached: {"attains_setpoint": relation(ReachPoseSetpoint, 1)},
    LiftClearanceReached: {"attains_setpoint": relation(LiftClearanceSetpoint, 1)},
    TransportTargetReached: {"attains_setpoint": relation(TransportPoseSetpoint, 1)},
    WithdrawalComplete: {"attains_setpoint": relation(WithdrawalDistanceSetpoint, 1)},
    ForcesUnloaded: {"weight_supported_by": relation(RoleSlot, 1)},
    Liftoff: {"crosses_threshold": single(LiftoffThreshold)},
    Touchdown: {"crosses_threshold": single(TouchdownThreshold)},
    SlipEvent: {"crosses_threshold": single(SlipThreshold)},
    CorrectiveGripResponse: {"targets_setpoint": relation(GripForceSetpoint, 1)},
    AnticipatoryGripAdjustment: {
        "triggered_by_prediction": single(SensoryEventPrediction),
        "targets_grip_force": single(GripForceSetpoint),
        "adjustment_direction": Holds(
            PropertyKind.ATTRIBUTE, CorrectionDirection, True, 1
        ),
        "adjustment_active_in": relation(ReplacePhase, 1),
    },
    GripLoadCoupling: {
        "predicted_load": single(LoadForceEstimate),
        "friction_estimate": single(ExpectedFrictionCoefficient),
        "safety_margin": single(SafetyMargin),
        "grip_force_response": single(GripForceSetpoint),
        "temporal_coupling": single(TemporalWindow),
        "coupling_active_in": relation(ActionPhase),
    },
    PrehensionPhaseStructure: {
        "defines_force_coupling": relation(GripLoadCoupling),
        "defines_anticipatory_grip_adjustment": relation(AnticipatoryGripAdjustment),
    },
}
"""
The properties each concept of the module writes in its own body.
"""

CORE_IDENTIFIED = {
    "attains_setpoint",
    "crosses_threshold",
    "targets_setpoint",
}
"""
The properties that narrow a property of the core and keep its identifier.
"""


def written_by(cls: type) -> set[str]:
    """
    Return the names of the properties a concept writes in its own body.
    """
    return {
        declaration.field_name
        for declaration in FLANAGAN.property_declarations(PREHENSION.name)
        if declaration.owner is cls
    }


# %% taxonomy


class TestPrehensionTaxonomy:
    """
    The concepts specialise the core and are declared in this module.
    """

    @pytest.mark.parametrize("cls", CONCEPTS)
    def test_concept_is_declared_in_this_module(self, cls: type) -> None:
        assert FLANAGAN.find_concept(cls).module == PREHENSION.name

    def test_module_declares_no_other_concepts(self) -> None:
        declared = {meta.cls for meta in FLANAGAN.concepts(PREHENSION.name)}

        assert declared == set(CONCEPTS)

    @pytest.mark.parametrize(("child", "parent"), list(PARENTS.items()))
    def test_concept_specialises_its_parent(self, child: type, parent: type) -> None:
        assert issubclass(child, parent)

    def test_no_concept_is_abstract(self) -> None:
        assert not [
            meta.cls for meta in FLANAGAN.concepts(PREHENSION.name) if meta.abstract
        ]

    def test_liftoff_and_touchdown_are_contact_changes_that_cross_a_threshold(
        self,
    ) -> None:
        for cls in (Liftoff, Touchdown):
            assert issubclass(cls, ContactEventType)
            assert issubclass(cls, ForceDynamicEventType)

    @pytest.mark.parametrize(
        "cls",
        [
            GraspingEffector,
            GraspSite,
            InitialSupport,
            FinalSupport,
            PrehensileManipulandum,
        ],
    )
    def test_roles_are_borne_and_not_essential(self, cls: type) -> None:
        meta = FLANAGAN.find_concept(cls)

        assert meta.rigidity is Rigidity.ANTI_RIGID
        assert meta.level is ConceptLevel.OCCURRENCE

    def test_a_support_before_liftoff_may_be_the_support_after_touchdown(self) -> None:
        axioms = FLANAGAN.find_concept(InitialSupport).axioms

        assert FinalSupport.__name__ in axioms.may_overlap_with


class TestKindsOfEventAreDistinct:
    """
    Events that specialise one kind of event are different events.
    """

    @pytest.mark.parametrize(
        "cls",
        [ContactEventType, ForceDynamicEventType, SetpointAttained],
    )
    def test_the_kinds_of_event_share_no_instance(self, cls: type) -> None:
        assert FLANAGAN.find_concept(cls).axioms.disjoint_siblings

    @pytest.mark.parametrize(
        "cls", [MotorSetpoint, MechanicalThreshold, ObjectPropertyEstimate]
    )
    def test_the_kinds_of_parameter_share_no_instance(self, cls: type) -> None:
        assert FLANAGAN.find_concept(cls).axioms.disjoint_siblings


# %% what a concept fixes


class TestFixedValues:
    """
    A concept states exactly the values its kind determines.
    """

    @pytest.mark.parametrize("cls", list(FIXED))
    def test_concept_fixes_what_its_kind_determines(self, cls: type) -> None:
        fixed = {
            restriction.property_name: restriction.value
            for restriction in FLANAGAN.find_concept(cls).axioms.fixed_values
        }

        assert fixed == FIXED[cls]

    def test_the_rate_of_load_force_inherits_what_a_force_rate_fixes(self) -> None:
        assert not FLANAGAN.find_concept(LoadForceRateSetpoint).axioms.fixed_values


# %% properties


class TestProperties:
    """
    Each concept writes the properties its specification lists, and no others.
    """

    @pytest.mark.parametrize("cls", CONCEPTS)
    def test_concept_declares_exactly_its_properties(self, cls: type) -> None:
        assert written_by(cls) == set(PROPERTIES.get(cls, {}))

    @pytest.mark.parametrize(
        ("cls", "name"),
        [(cls, name) for cls, held in PROPERTIES.items() for name in held],
    )
    def test_property_holds_what_it_is_specified_to(self, cls: type, name: str) -> None:
        expected = PROPERTIES[cls][name]
        declared = PropertyMeta.declared_on(cls)[name]
        resolved = RangeResolver().resolve(cls, name)

        assert declared.kind is expected.kind
        assert declared.functional is expected.functional
        assert declared.minimum_count == expected.minimum
        assert resolved.element_types == (expected.range,)

    @pytest.mark.parametrize(
        ("cls", "name"),
        [
            (cls, name)
            for cls, held in PROPERTIES.items()
            for name in held
            if name in CORE_IDENTIFIED
        ],
    )
    def test_narrowing_a_core_property_keeps_its_identifier(
        self, cls: type, name: str
    ) -> None:
        narrowed = next(
            declaration
            for declaration in FLANAGAN.property_declarations(PREHENSION.name)
            if declaration.owner is cls and declaration.field_name == name
        )

        assert narrowed.iri == CORE.namespace.property_term(name)


class TestStructureCollections:
    """
    A structure owns each prehensile fragment it declares.
    """

    @pytest.mark.parametrize("name", list(PROPERTIES[PrehensionPhaseStructure]))
    def test_collection_owns_the_fragments_it_holds(self, name: str) -> None:
        assert PropertyMeta.declared_on(PrehensionPhaseStructure)[name].owns is True


# %% alignments


class TestAlignments:
    """
    Each concept is placed in an upper ontology only as far as its definition allows.
    """

    @staticmethod
    def alignments(cls: type) -> tuple:
        return FLANAGAN.find_concept(cls).alignments

    def test_effectors_and_supports_are_only_related_to_the_physical_things(
        self,
    ) -> None:
        assert self.alignments(GraspingEffector) == (
            related_to(SOMA, "PrehensileEffector"),
        )
        for cls in (InitialSupport, FinalSupport):
            assert self.alignments(cls) == (related_to(SOMA, "Supporter"),)

    def test_contact_conditions_are_only_related_to_contact_states(self) -> None:
        for cls in (EffectorObjectContact, ObjectSupportContact):
            assert self.alignments(cls) == (related_to(SOMA, "ContactState"),)

    def test_a_grip_load_coupling_is_a_relation(self) -> None:
        assert self.alignments(GripLoadCoupling) == (subclass_of(DUL, "Relation"),)


# %% provenance

EMPIRICAL = EvidenceStatus.EMPIRICAL_FINDING
ENGINEERING = EvidenceStatus.ENGINEERING_EXTENSION

CLASSIFIED: dict[type, EvidenceStatus] = {
    GraspingEffector: EMPIRICAL,
    GraspSite: EMPIRICAL,
    InitialSupport: EMPIRICAL,
    FinalSupport: EMPIRICAL,
    ContactEstablishment: EMPIRICAL,
    ContactBreak: EMPIRICAL,
    Liftoff: EMPIRICAL,
    Touchdown: EMPIRICAL,
    SlipEvent: EMPIRICAL,
    LiftClearanceReached: ENGINEERING,
    StableHoldReached: ENGINEERING,
    WithdrawalComplete: ENGINEERING,
    EffectorObjectContact: EMPIRICAL,
    ObjectSupportContact: EMPIRICAL,
    GraspStability: EMPIRICAL,
    ObjectClearOfSupport: EMPIRICAL,
    EffectorPositioned: ENGINEERING,
    ReachPhase: EMPIRICAL,
    ForceLoadingPhase: EMPIRICAL,
    LiftPhase: EMPIRICAL,
    HoldPhase: ENGINEERING,
    TransportPhase: EMPIRICAL,
    ReplacePhase: EMPIRICAL,
    UnloadPhase: EMPIRICAL,
    ContactReleasePhase: EMPIRICAL,
    GripForceSetpoint: EMPIRICAL,
    LoadForceSetpoint: EMPIRICAL,
    SlipThreshold: EMPIRICAL,
    LiftoffThreshold: EMPIRICAL,
    ExpectedMass: EMPIRICAL,
    ExpectedFrictionCoefficient: EMPIRICAL,
    ExpectedCenterOfMassPosition: EMPIRICAL,
    ExpectedShape: EMPIRICAL,
    AnticipatoryGripAdjustment: EMPIRICAL,
    GripLoadCoupling: EMPIRICAL,
    PrehensionPhaseStructure: ENGINEERING,
}
"""
The status the scientific contract gives each concept it classifies.
"""

UNCLASSIFIED = {
    PrehensileManipulandum,
    ReachTargetReached,
    TransportTargetReached,
    ForcesUnloaded,
    SafeContactForce,
    ContactEstablishmentPhase,
    WithdrawPhase,
    ReachPoseSetpoint,
    LoadForceRateSetpoint,
    LiftClearanceSetpoint,
    TransportPoseSetpoint,
    WithdrawalDistanceSetpoint,
    TouchdownThreshold,
    CorrectiveGripResponse,
}
"""
The concepts the contract gives no basis for, which nothing invents a status for.
"""


class TestProvenance:
    """
    Each concept records how well the scientific contract supports it.
    """

    def test_every_concept_is_classified_or_deliberately_not(self) -> None:
        assert set(CLASSIFIED) | UNCLASSIFIED == set(CONCEPTS)
        assert not set(CLASSIFIED) & UNCLASSIFIED

    @pytest.mark.parametrize("cls", list(CLASSIFIED))
    def test_concept_records_the_status_of_the_contract(self, cls: type) -> None:
        assert FLANAGAN.find_concept(cls).status is CLASSIFIED[cls]

    @pytest.mark.parametrize("cls", sorted(UNCLASSIFIED, key=lambda c: c.__name__))
    def test_concept_without_a_basis_has_no_status(self, cls: type) -> None:
        assert FLANAGAN.find_concept(cls).status is None

    def test_the_claims_the_traceability_names_are_cited(self) -> None:
        assert "FJ-008" in FLANAGAN.find_concept(GripLoadCoupling).claims
        assert "FJ-025" in FLANAGAN.find_concept(GripLoadCoupling).claims
        assert "FJ-009" in FLANAGAN.find_concept(AnticipatoryGripAdjustment).claims
        assert "FJ-012" in FLANAGAN.find_concept(ExpectedFrictionCoefficient).claims


# %% documentation


class TestPrehensionIsDocumented:
    """
    Every concept and property says what it means.
    """

    @pytest.mark.parametrize("cls", CONCEPTS)
    def test_concept_and_properties_are_documented(self, cls: type) -> None:
        reader = DocumentationReader()

        assert reader.read_class(cls) is not None
        assert reader.find_undocumented_fields(cls) == ()
