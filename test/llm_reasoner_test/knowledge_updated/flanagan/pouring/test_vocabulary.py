from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    SOMA,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    ContactEventType,
    MechanicalEventType,
    SetpointAttained,
    ThresholdCrossingEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    DynamicsEstimate,
    MechanicalThreshold,
    MotorSetpoint,
    ObjectPropertyEstimate,
    ObjectStateEstimate,
    TaskSetpoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.phases import ActionPhase
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import (
    ManipulationRole,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    CouplingRegime,
    ParameterValueForm,
    PhysicalDimension,
    RegulationMode,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import (
    CORE,
    FLANAGAN,
    POURING,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.events import (
    FlowCessation,
    FlowEventType,
    FlowOnset,
    LipAligned,
    SourceEmpty,
    SpillEvent,
    TargetQuantityReached,
    UprightRestored,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.parameters import (
    AlignmentErrorEstimate,
    CenterOfMassShiftEstimate,
    CohesiveFlowDynamicsEstimate,
    DynamicViscosityEstimate,
    FlowOnsetThreshold,
    FlowRateSetpoint,
    GranularFlowDynamicsEstimate,
    LipAlignmentSetpoint,
    PourPoseSetpoint,
    RemainingSourceQuantityEstimate,
    SloshDynamicsEstimate,
    SpillThreshold,
    TargetFillStateEstimate,
    TiltAngleSetpoint,
    TiltRateSetpoint,
    TorqueSetpoint,
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
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.roles import (
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

ENUMERATIONS = (MaterialRegime, TransferTerminationReason)
"""
Every value partition of the module.
"""

PARENTS: dict[type, type] = {
    SourceContainer: PrehensileManipulandum,
    TargetReceptacle: ManipulationRole,
    PouredSubstance: ManipulationRole,
    PouringLip: ManipulationRole,
    FlowEventType: MechanicalEventType,
    FlowOnset: FlowEventType,
    FlowCessation: FlowEventType,
    SpillEvent: FlowEventType,
    TargetQuantityReached: SetpointAttained,
    SourceEmpty: SetpointAttained,
    LipAligned: SetpointAttained,
    UprightRestored: SetpointAttained,
    AlignmentPhase: ActionPhase,
    TiltToFlowPhase: ActionPhase,
    FlowRegulationPhase: ActionPhase,
    FlowTerminationPhase: ActionPhase,
    ReturnUprightPhase: ActionPhase,
    DynamicViscosityEstimate: ObjectPropertyEstimate,
    RemainingSourceQuantityEstimate: ObjectStateEstimate,
    TargetFillStateEstimate: ObjectStateEstimate,
    AlignmentErrorEstimate: ObjectStateEstimate,
    SloshDynamicsEstimate: DynamicsEstimate,
    GranularFlowDynamicsEstimate: DynamicsEstimate,
    CohesiveFlowDynamicsEstimate: DynamicsEstimate,
    CenterOfMassShiftEstimate: DynamicsEstimate,
    TiltAngleSetpoint: MotorSetpoint,
    TiltRateSetpoint: MotorSetpoint,
    TorqueSetpoint: MotorSetpoint,
    PourPoseSetpoint: MotorSetpoint,
    LipAlignmentSetpoint: MotorSetpoint,
    UprightAngleSetpoint: MotorSetpoint,
    FlowRateSetpoint: TaskSetpoint,
    TransferredQuantitySetpoint: TaskSetpoint,
    FlowOnsetThreshold: MechanicalThreshold,
    SpillThreshold: MechanicalThreshold,
    PouringPhaseStructure: PrehensionPhaseStructure,
}
"""
Each concept of the module beside the concept it specialises.
"""

CONCEPTS = tuple(PARENTS)
"""
Every concept of the module.
"""

QUANTITATIVE = ParameterValueForm.QUANTITATIVE
CATEGORICAL = ParameterValueForm.CATEGORICAL
MASS_OR_VOLUME = frozenset({PhysicalDimension.MASS, PhysicalDimension.VOLUME})
FLOW_RATES = frozenset(
    {PhysicalDimension.MASS_FLOW_RATE, PhysicalDimension.VOLUMETRIC_FLOW_RATE}
)


def quantity(dimension: PhysicalDimension) -> dict[str, object]:
    """
    Return the values fixed for a quantity of a dimension.
    """
    return {"value_form": QUANTITATIVE, "dimension": dimension}


FIXED: dict[type, dict[str, object]] = {
    FlowRegulationPhase: {"coupling_regime": CouplingRegime.COUPLED},
    DynamicViscosityEstimate: quantity(PhysicalDimension.DYNAMIC_VISCOSITY),
    RemainingSourceQuantityEstimate: {"value_form": QUANTITATIVE},
    TargetFillStateEstimate: quantity(PhysicalDimension.DIMENSIONLESS),
    AlignmentErrorEstimate: quantity(PhysicalDimension.ANGLE),
    SloshDynamicsEstimate: {"value_form": CATEGORICAL},
    GranularFlowDynamicsEstimate: {"value_form": CATEGORICAL},
    CohesiveFlowDynamicsEstimate: {"value_form": CATEGORICAL},
    CenterOfMassShiftEstimate: quantity(PhysicalDimension.LENGTH),
    TiltAngleSetpoint: quantity(PhysicalDimension.ANGLE),
    TiltRateSetpoint: quantity(PhysicalDimension.ANGULAR_RATE),
    TorqueSetpoint: quantity(PhysicalDimension.TORQUE),
    PourPoseSetpoint: {"value_form": CATEGORICAL},
    LipAlignmentSetpoint: quantity(PhysicalDimension.ANGLE),
    UprightAngleSetpoint: quantity(PhysicalDimension.ANGLE),
    FlowRateSetpoint: {"value_form": QUANTITATIVE},
    TransferredQuantitySetpoint: {"value_form": QUANTITATIVE},
    FlowOnsetThreshold: quantity(PhysicalDimension.ANGLE),
    SpillThreshold: {"value_form": QUANTITATIVE},
}
"""
The values each concept fixes for every instance, and nothing else.
"""

ALLOWED: dict[type, dict[str, frozenset]] = {
    FlowRegulationPhase: {
        "regulation_mode": frozenset(
            {RegulationMode.CONTINUOUS_FEEDBACK, RegulationMode.HYBRID}
        )
    },
    RemainingSourceQuantityEstimate: {"dimension": MASS_OR_VOLUME},
    FlowRateSetpoint: {"dimension": FLOW_RATES},
    TransferredQuantitySetpoint: {"dimension": MASS_OR_VOLUME},
    SpillThreshold: {"dimension": FLOW_RATES},
}
"""
The values each concept permits for a property without fixing one.
"""


class Holds(NamedTuple):
    """
    What one property of a concept must hold.
    """

    kind: PropertyKind
    range: type
    functional: bool
    minimum: int


PROPERTIES: dict[type, dict[str, Holds]] = {
    FlowOnset: {
        "crosses_threshold": Holds(PropertyKind.RELATION, FlowOnsetThreshold, True, 1)
    },
    SpillEvent: {
        "crosses_threshold": Holds(PropertyKind.RELATION, SpillThreshold, True, 1)
    },
    TargetQuantityReached: {
        "attains_setpoint": Holds(
            PropertyKind.RELATION, TransferredQuantitySetpoint, False, 1
        )
    },
    LipAligned: {
        "attains_setpoint": Holds(PropertyKind.RELATION, LipAlignmentSetpoint, False, 1)
    },
    UprightRestored: {
        "attains_setpoint": Holds(PropertyKind.RELATION, UprightAngleSetpoint, False, 1)
    },
    PouringPhaseStructure: {
        "material_regime": Holds(PropertyKind.ATTRIBUTE, MaterialRegime, True, 1),
        "termination_reason": Holds(
            PropertyKind.ATTRIBUTE, TransferTerminationReason, True, 1
        ),
    },
}
"""
The properties each concept of the module writes in its own body.
"""

CORE_IDENTIFIED = {"crosses_threshold", "attains_setpoint"}
"""
The properties that narrow a property of the core and keep its identifier.
"""


def written_by(cls: type) -> set[str]:
    """
    Return the names of the properties a concept writes in its own body.
    """
    return {
        declaration.field_name
        for declaration in FLANAGAN.property_declarations(POURING.name)
        if declaration.owner is cls
    }


# %% value partitions


class TestPartitionsAreDeclared:
    """
    Each enumeration is recorded as a closed set of alternatives of this module.
    """

    @pytest.mark.parametrize("enumeration", ENUMERATIONS)
    def test_enumeration_is_declared_in_the_module(
        self, enumeration: type[Enum]
    ) -> None:
        partition = FLANAGAN.find_value_partition(enumeration)

        assert partition is not None
        assert partition.module == POURING.name
        assert partition.iri == POURING.namespace.term(enumeration.__name__)

    def test_module_declares_no_other_partitions(self) -> None:
        declared = {p.enumeration for p in FLANAGAN.value_partitions(POURING.name)}

        assert declared == set(ENUMERATIONS)

    @pytest.mark.parametrize("enumeration", ENUMERATIONS)
    def test_enumeration_and_its_members_are_documented(
        self, enumeration: type[Enum]
    ) -> None:
        reader = DocumentationReader()

        assert reader.read_class(enumeration) is not None
        assert reader.find_undocumented_members(enumeration) == ()


# %% taxonomy


class TestPouringTaxonomy:
    """
    The concepts specialise the core and prehension and are declared in this module.
    """

    def test_module_declares_exactly_the_specified_concepts(self) -> None:
        declared = {meta.cls for meta in FLANAGAN.concepts(POURING.name)}

        assert declared == set(CONCEPTS)

    @pytest.mark.parametrize(("child", "parent"), list(PARENTS.items()))
    def test_concept_specialises_its_parent(self, child: type, parent: type) -> None:
        assert issubclass(child, parent)

    def test_only_the_grouping_concepts_are_abstract(self) -> None:
        abstract = {
            meta.cls for meta in FLANAGAN.concepts(POURING.name) if meta.abstract
        }

        assert abstract == set()

    @pytest.mark.parametrize(
        "cls", [SourceContainer, TargetReceptacle, PouredSubstance, PouringLip]
    )
    def test_roles_are_borne_and_not_essential(self, cls: type) -> None:
        meta = FLANAGAN.find_concept(cls)

        assert meta.rigidity is Rigidity.ANTI_RIGID
        assert meta.level is ConceptLevel.OCCURRENCE

    def test_flow_onset_and_spill_are_flows_that_cross_a_threshold(self) -> None:
        for cls in (FlowOnset, SpillEvent):
            assert issubclass(cls, FlowEventType)
            assert issubclass(cls, ThresholdCrossingEventType)

    def test_a_flow_may_also_cross_a_threshold_by_intent(self) -> None:
        axioms = FLANAGAN.find_concept(FlowEventType).axioms

        assert ThresholdCrossingEventType.__name__ in axioms.may_overlap_with

    def test_a_flow_is_not_a_change_of_contact(self) -> None:
        axioms = FLANAGAN.find_concept(FlowEventType).axioms

        assert ContactEventType.__name__ in axioms.disjoint_with

    @pytest.mark.parametrize(
        "cls",
        [
            FlowEventType,
            ThresholdCrossingEventType,
            ObjectStateEstimate,
            DynamicsEstimate,
            TaskSetpoint,
        ],
    )
    def test_the_kinds_of_a_concept_share_no_instance(self, cls: type) -> None:
        assert FLANAGAN.find_concept(cls).axioms.disjoint_siblings


# %% what a concept fixes or allows


class TestFixedAndAllowedValues:
    """
    A concept states the values its kind determines and permits.
    """

    @pytest.mark.parametrize("cls", list(FIXED))
    def test_concept_fixes_what_its_kind_determines(self, cls: type) -> None:
        fixed = {
            restriction.property_name: restriction.value
            for restriction in FLANAGAN.find_concept(cls).axioms.fixed_values
        }

        assert fixed == FIXED[cls]

    @pytest.mark.parametrize("cls", list(ALLOWED))
    def test_concept_allows_what_its_kind_permits(self, cls: type) -> None:
        allowed = {
            restriction.property_name: restriction.values
            for restriction in FLANAGAN.find_concept(cls).axioms.allowed_values
        }

        assert allowed == ALLOWED[cls]

    def test_flow_regulation_is_continuously_modulated(self) -> None:
        allowed = FLANAGAN.find_concept(FlowRegulationPhase).axioms.allowed_values_of(
            "regulation_mode"
        )

        assert RegulationMode.FEEDFORWARD not in allowed


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
            for declaration in FLANAGAN.property_declarations(POURING.name)
            if declaration.owner is cls and declaration.field_name == name
        )

        assert narrowed.iri == CORE.namespace.property_term(name)

    def test_continuous_predictions_are_those_of_the_core(self) -> None:
        assert "has_continuous_prediction" not in written_by(PouringPhaseStructure)


# %% alignments


class TestAlignments:
    """
    Each role is placed in the upper ontology only as far as its definition allows.
    """

    def test_roles_specialise_the_roles_of_the_task_ontology(self) -> None:
        expected = {
            SourceContainer: subclass_of(SOMA, "Container"),
            TargetReceptacle: subclass_of(SOMA, "Destination"),
            PouredSubstance: subclass_of(SOMA, "PouredObject"),
        }

        for cls, alignment in expected.items():
            assert FLANAGAN.find_concept(cls).alignments == (alignment,)


# %% provenance

UNCLASSIFIED = {
    SourceEmpty,
    RemainingSourceQuantityEstimate,
    AlignmentErrorEstimate,
    GranularFlowDynamicsEstimate,
    CohesiveFlowDynamicsEstimate,
    FlowRateSetpoint,
    PourPoseSetpoint,
    LipAlignmentSetpoint,
    UprightAngleSetpoint,
}
"""
The concepts the contract gives no basis for, which nothing invents a status for.
"""


class TestProvenance:
    """
    Each concept records how well the scientific contract supports it.
    """

    @pytest.mark.parametrize(
        "cls", sorted(set(CONCEPTS) - UNCLASSIFIED, key=lambda c: c.__name__)
    )
    def test_pouring_is_an_engineering_extension(self, cls: type) -> None:
        assert FLANAGAN.find_concept(cls).status is EvidenceStatus.ENGINEERING_EXTENSION

    @pytest.mark.parametrize("cls", sorted(UNCLASSIFIED, key=lambda c: c.__name__))
    def test_concept_without_a_basis_has_no_status(self, cls: type) -> None:
        assert FLANAGAN.find_concept(cls).status is None


# %% documentation


class TestPouringIsDocumented:
    """
    Every concept and property says what it means.
    """

    @pytest.mark.parametrize("cls", CONCEPTS)
    def test_concept_and_properties_are_documented(self, cls: type) -> None:
        reader = DocumentationReader()

        assert reader.read_class(cls) is not None
        assert reader.find_undocumented_fields(cls) == ()
