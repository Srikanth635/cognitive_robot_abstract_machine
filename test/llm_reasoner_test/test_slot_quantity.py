"""
Tests that a kind of quantity about the filler of a slot is one term of the action core:
the choices an action leaves open, the quantities deciding whether a relation of the
image schemas holds, the indicators a failure is noticed by and the parameters of the
predictive manipulation microworld are kinds of it, each adding only what is its own.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
from rdflib import RDFS, Graph, Namespace, URIRef
from typing_extensions import List, Optional, get_type_hints

from llm_reasoner.knowledge.ontologies import (
    DUL_NAMESPACE,
    KNOWLEDGE_ONTOLOGY_MODULES,
    ActionPerspectiveFile,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import (
    action_core,
    failures,
    image_schemas,
)
from llm_reasoner.knowledge.ontologies.predictive_manipulation import (
    core,
    object_dynamics,
    pouring,
    prehension,
)
from llm_reasoner.knowledge.ontologies.terms import PropertyTerm
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary

from .restriction_violations import violated_on
from .test_perspective_shapes import CORE_SHAPES, broken

DUL = Namespace(DUL_NAMESPACE)
"""
The terms of DOLCE+DnS Ultralite.
"""

KINDS_OF_QUANTITY = [
    (action_core.ParameterDimension, set()),
    (image_schemas.DecidingQuantity, {"quantity_aspect"}),
    (failures.Indicator, {"has_indicator_form", "noted_by"}),
]
"""
Each kind of quantity about a slot, with the fields only it has.
"""

Kind = action_core.QuantityKind
"""
The kinds of quantity.
"""

Dimension = action_core.PhysicalDimension
"""
The physical dimensions.
"""

DIMENSIONS_OF_KINDS = [
    (Kind.AMOUNT, {Dimension.MASS, Dimension.VOLUME}),
    (
        Kind.RATE,
        {
            Dimension.MASS_FLOW_RATE,
            Dimension.VOLUMETRIC_FLOW_RATE,
            Dimension.ANGULAR_RATE,
            Dimension.FORCE_RATE,
            Dimension.FREQUENCY,
        },
    ),
    (Kind.POSITION, {Dimension.LENGTH}),
    (Kind.ORIENTATION, {Dimension.ANGLE}),
    (Kind.DISTANCE, {Dimension.LENGTH}),
    (Kind.DURATION, {Dimension.TIME}),
    (Kind.SELECTION, set()),
    (Kind.MAGNITUDE, {Dimension.FORCE, Dimension.TORQUE}),
    (Kind.MATERIAL_PROPERTY, {Dimension.DYNAMIC_VISCOSITY, Dimension.DIMENSIONLESS}),
    (Kind.RATIO, {Dimension.DIMENSIONLESS}),
    (Kind.SHAPE, set()),
]
"""
Each kind of quantity, with the physical dimensions it may be measured in.
"""

PARAMETER_KINDS = [
    (core.ForceRateSetpoint, Kind.RATE),
    (core.SafetyMargin, Kind.RATIO),
    (object_dynamics.LoadForceEstimate, Kind.MAGNITUDE),
    (prehension.CrushThreshold, Kind.MAGNITUDE),
    (prehension.ExpectedCenterOfMassPosition, Kind.POSITION),
    (prehension.ExpectedFrictionCoefficient, Kind.MATERIAL_PROPERTY),
    (prehension.ExpectedMass, Kind.AMOUNT),
    (prehension.ExpectedShape, Kind.SHAPE),
    (prehension.GripForceSetpoint, Kind.MAGNITUDE),
    (prehension.GripToLoadRatioSetpoint, Kind.RATIO),
    (prehension.LiftClearanceSetpoint, Kind.DISTANCE),
    (prehension.LiftoffThreshold, Kind.MAGNITUDE),
    (prehension.LoadForceRateSetpoint, Kind.RATE),
    (prehension.LoadForceSetpoint, Kind.MAGNITUDE),
    (prehension.ReachPoseSetpoint, Kind.SELECTION),
    (prehension.SlipThreshold, Kind.MAGNITUDE),
    (prehension.TouchdownThreshold, Kind.MAGNITUDE),
    (prehension.TransportPoseSetpoint, Kind.SELECTION),
    (prehension.WithdrawalDistanceSetpoint, Kind.DISTANCE),
    (pouring.AlignmentErrorEstimate, Kind.ORIENTATION),
    (pouring.CenterOfMassShiftEstimate, Kind.DISTANCE),
    (pouring.CohesiveFlowDynamicsEstimate, Kind.MATERIAL_PROPERTY),
    (pouring.DynamicViscosityEstimate, Kind.MATERIAL_PROPERTY),
    (pouring.FlowOnsetThreshold, Kind.ORIENTATION),
    (pouring.FlowRateSetpoint, Kind.RATE),
    (pouring.GranularFlowDynamicsEstimate, Kind.MATERIAL_PROPERTY),
    (pouring.LipAlignmentSetpoint, Kind.ORIENTATION),
    (pouring.PourPoseSetpoint, Kind.SELECTION),
    (pouring.RemainingSourceQuantityEstimate, Kind.AMOUNT),
    (pouring.SloshDynamicsEstimate, Kind.MATERIAL_PROPERTY),
    (pouring.SpillThreshold, Kind.RATE),
    (pouring.TargetFillStateEstimate, Kind.RATIO),
    (pouring.TiltAngleSetpoint, Kind.ORIENTATION),
    (pouring.TiltRateSetpoint, Kind.RATE),
    (pouring.TorqueSetpoint, Kind.MAGNITUDE),
    (pouring.TransferredQuantitySetpoint, Kind.AMOUNT),
    (pouring.UprightAngleSetpoint, Kind.ORIENTATION),
]
"""
Each parameter class of the microworld that fixes its kind of quantity, with the kind.
"""

PARAMETER_SLOTS = [
    (
        prehension.GripForceSetpoint,
        (core.Manipulandum,),
        (),
        (prehension.GraspingEffector,),
    ),
    (
        prehension.LoadForceSetpoint,
        (core.Manipulandum,),
        (),
        (prehension.GraspingEffector,),
    ),
    (
        prehension.LoadForceRateSetpoint,
        (core.Manipulandum,),
        (),
        (prehension.GraspingEffector,),
    ),
    (
        prehension.GripToLoadRatioSetpoint,
        (core.Manipulandum,),
        (),
        (prehension.GraspingEffector,),
    ),
    (
        prehension.SlipThreshold,
        (core.Manipulandum,),
        (),
        (prehension.GraspingEffector,),
    ),
    (
        prehension.CrushThreshold,
        (core.Manipulandum,),
        (),
        (prehension.GraspingEffector,),
    ),
    (
        prehension.LiftoffThreshold,
        (core.Manipulandum,),
        (),
        (prehension.GraspingEffector,),
    ),
    (
        prehension.ExpectedFrictionCoefficient,
        (core.Manipulandum, prehension.GraspSite),
        (),
        (prehension.GraspingEffector,),
    ),
    (
        prehension.TouchdownThreshold,
        (core.Manipulandum,),
        (),
        (prehension.FinalSupport,),
    ),
    (
        prehension.LiftClearanceSetpoint,
        (core.Manipulandum,),
        (prehension.InitialSupport,),
        (),
    ),
    (
        prehension.TransportPoseSetpoint,
        (core.Manipulandum,),
        (action_core.ParticipantSlot,),
        (),
    ),
    (
        prehension.ReachPoseSetpoint,
        (prehension.GraspingEffector,),
        (prehension.GraspSite,),
        (),
    ),
    (
        prehension.WithdrawalDistanceSetpoint,
        (prehension.GraspingEffector,),
        (core.Manipulandum,),
        (),
    ),
    (prehension.ExpectedMass, (core.Manipulandum,), (), ()),
    (prehension.ExpectedCenterOfMassPosition, (core.Manipulandum,), (), ()),
    (prehension.ExpectedShape, (core.Manipulandum,), (), ()),
    (
        pouring.LipAlignmentSetpoint,
        (pouring.PouringLip,),
        (pouring.TargetReceptacle,),
        (),
    ),
    (
        pouring.AlignmentErrorEstimate,
        (pouring.PouringLip,),
        (pouring.TargetReceptacle,),
        (),
    ),
    (pouring.TiltAngleSetpoint, (pouring.SourceContainer,), (), ()),
    (pouring.TiltRateSetpoint, (pouring.SourceContainer,), (), ()),
    (pouring.UprightAngleSetpoint, (pouring.SourceContainer,), (), ()),
    (
        pouring.FlowOnsetThreshold,
        (pouring.SourceContainer,),
        (),
        (pouring.PouredSubstance,),
    ),
    (
        pouring.CenterOfMassShiftEstimate,
        (pouring.SourceContainer,),
        (),
        (pouring.PouredSubstance,),
    ),
    (
        pouring.TorqueSetpoint,
        (pouring.SourceContainer,),
        (),
        (prehension.GraspingEffector,),
    ),
    (
        pouring.PourPoseSetpoint,
        (pouring.SourceContainer,),
        (pouring.TargetReceptacle,),
        (),
    ),
    (pouring.FlowRateSetpoint, (pouring.PouredSubstance,), (), ()),
    (
        pouring.TransferredQuantitySetpoint,
        (pouring.PouredSubstance,),
        (),
        (pouring.TargetReceptacle,),
    ),
    (
        pouring.RemainingSourceQuantityEstimate,
        (pouring.PouredSubstance,),
        (),
        (pouring.SourceContainer,),
    ),
    (
        pouring.SpillThreshold,
        (pouring.PouredSubstance,),
        (),
        (pouring.TargetReceptacle,),
    ),
    (
        pouring.SloshDynamicsEstimate,
        (pouring.PouredSubstance,),
        (),
        (pouring.SourceContainer,),
    ),
    (pouring.DynamicViscosityEstimate, (pouring.PouredSubstance,), (), ()),
    (pouring.CohesiveFlowDynamicsEstimate, (pouring.PouredSubstance,), (), ()),
    (pouring.GranularFlowDynamicsEstimate, (pouring.PouredSubstance,), (), ()),
    (
        pouring.TargetFillStateEstimate,
        (pouring.TargetReceptacle,),
        (),
        (pouring.PouredSubstance,),
    ),
]
"""
Each parameter class of the microworld that names its slots, with the classes of the
slot it is about, the slot it may be measured from and the slots it may involve.
"""

ABOUT_SLOT = PropertyTerm.of_field(
    clazz=action_core.SlotQuantity, field_name="about_slot"
)
"""
The property naming the slot a quantity is about.
"""

RELATIVE_TO_SLOT = PropertyTerm.of_field(
    clazz=action_core.SlotQuantity, field_name="relative_to_slot"
)
"""
The property naming the slot a quantity is measured from.
"""

INVOLVES_SLOT = PropertyTerm.of_field(
    clazz=action_core.SlotQuantity, field_name="quantity_involves_slot"
)
"""
The property naming the other slots a quantity is between.
"""

AC = Namespace(ActionPerspectiveFile.ACTION_CORE.namespace)
"""
The terms of the action core.
"""

# %% fixtures


@pytest.fixture(scope="module")
def graph() -> Graph:
    """
    Every module of both families in one graph.
    """
    union = Graph()
    for module in KNOWLEDGE_ONTOLOGY_MODULES:
        union.parse(module.path, format="turtle")
    return union


def field_names(cls: type) -> set[str]:
    """
    :return: The names of the fields of a generated class.
    """
    return {declared.name for declared in fields(cls)}


# %% the term


class TestTheTerm:
    def test_a_quantity_names_its_kind_its_slot_and_the_slot_it_is_measured_from(
        self,
    ):
        hints = get_type_hints(action_core.SlotQuantity)
        assert hints["has_quantity_kind"] == Optional[action_core.QuantityKind]
        assert hints["about_slot"] == Optional[action_core.FrameSlot]
        assert hints["relative_to_slot"] == Optional[action_core.FrameSlot]

    def test_a_quantity_names_the_slots_it_is_between(self):
        hints = get_type_hints(action_core.SlotQuantity)
        assert hints["quantity_involves_slot"] == List[action_core.FrameSlot]

    def test_a_quantity_names_the_dimension_it_is_measured_in(self):
        hints = get_type_hints(action_core.SlotQuantity)
        assert (
            hints["has_physical_dimension"] == Optional[action_core.PhysicalDimension]
        )

    def test_it_is_a_dul_concept(self, graph):
        assert (
            URIRef(action_core.SlotQuantity.term.iri),
            RDFS.subClassOf,
            DUL.Concept,
        ) in graph

    @pytest.mark.parametrize("kind, own_fields", KINDS_OF_QUANTITY)
    def test_every_kind_specialises_it(self, kind, own_fields):
        assert issubclass(kind, action_core.SlotQuantity)

    @pytest.mark.parametrize("kind, own_fields", KINDS_OF_QUANTITY)
    def test_every_kind_adds_only_its_own_fields(self, kind, own_fields):
        assert field_names(kind) - field_names(action_core.SlotQuantity) == own_fields

    @pytest.mark.parametrize("kind, own_fields", KINDS_OF_QUANTITY)
    def test_no_kind_is_a_dul_description(self, graph, kind, own_fields):
        assert (
            URIRef(kind.term.iri),
            RDFS.subClassOf,
            DUL.Description,
        ) not in graph


# %% kinds and dimensions


class TestKindsAndDimensions:
    def test_the_core_owns_the_dimensions_and_the_units(self):
        assert {"PhysicalDimension", "MeasurementUnit"} & set(vars(core)) == set()
        assert (
            action_core.MeasurementUnit.KILOGRAM.measures_dimension
            == action_core.PhysicalDimension.MASS
        )

    @pytest.mark.parametrize("kind, dimensions", DIMENSIONS_OF_KINDS)
    def test_a_kind_names_the_dimensions_it_is_measured_in(
        self, graph, kind, dimensions
    ):
        assert set(graph.objects(URIRef(kind.value), AC.measuredIn)) == {
            URIRef(dimension.value) for dimension in dimensions
        }

    def test_every_kind_is_listed_with_its_dimensions(self):
        assert [kind for kind, _ in DIMENSIONS_OF_KINDS] == list(Kind)

    def test_a_quantity_measured_in_a_dimension_of_its_kind_is_accepted(self):
        quantity = pouring.TiltAngleSetpoint(
            has_quantity_kind=Kind.ORIENTATION, has_physical_dimension=Dimension.ANGLE
        )
        assert broken(quantity, CORE_SHAPES.QuantityMeasuredInItsKind) == []

    def test_a_quantity_measured_in_another_dimension_is_refused(self):
        quantity = pouring.TiltAngleSetpoint(
            has_quantity_kind=Kind.ORIENTATION, has_physical_dimension=Dimension.LENGTH
        )
        assert broken(quantity, CORE_SHAPES.QuantityMeasuredInItsKind)


# %% the parameters of the predictive manipulation microworld


class TestPredictiveManipulationParameters:
    def test_a_parameter_of_the_microworld_is_a_slot_quantity(self):
        assert issubclass(core.Parameter, action_core.SlotQuantity)

    def test_no_parameter_of_the_microworld_lists_roles(self):
        names = {
            name
            for generated in Vocabulary().classes
            if issubclass(generated, core.Parameter)
            for name in get_type_hints(generated)
        }
        assert "parameter_on_role" not in names

    @pytest.mark.parametrize("parameter, kind", PARAMETER_KINDS)
    def test_a_parameter_class_fixes_its_kind(self, parameter, kind):
        assert Vocabulary().entailed_values(parameter)["has_quantity_kind"] == kind

    def test_every_most_specific_parameter_class_fixes_its_kind(self):
        classes = [
            generated
            for generated in Vocabulary().classes
            if issubclass(generated, core.Parameter)
            and generated.term.definition is None
        ]
        most_specific = {
            parameter
            for parameter in classes
            if not any(
                other is not parameter and issubclass(other, parameter)
                for other in classes
            )
        }
        fixing = {parameter for parameter, _ in PARAMETER_KINDS}
        assert most_specific - fixing == set()

    @pytest.mark.parametrize("parameter, about, relative, involved", PARAMETER_SLOTS)
    def test_a_parameter_is_about_the_slot_its_class_names(
        self, parameter, about, relative, involved
    ):
        for kind in about:
            assert violated_on(parameter(about_slot=kind()), ABOUT_SLOT) == []
        assert violated_on(parameter(about_slot=action_core.FrameSlot()), ABOUT_SLOT)

    @pytest.mark.parametrize(
        "parameter, about, relative, involved",
        [case for case in PARAMETER_SLOTS if case[2]],
    )
    def test_a_parameter_is_measured_only_from_the_slot_its_class_names(
        self, parameter, about, relative, involved
    ):
        (relative_kind,) = relative
        assert (
            violated_on(
                parameter(about_slot=about[0](), relative_to_slot=relative_kind()),
                RELATIVE_TO_SLOT,
            )
            == []
        )
        assert violated_on(
            parameter(about_slot=about[0](), relative_to_slot=action_core.FrameSlot()),
            RELATIVE_TO_SLOT,
        )

    @pytest.mark.parametrize(
        "parameter, about, relative, involved",
        [case for case in PARAMETER_SLOTS if case[3]],
    )
    def test_a_parameter_involves_only_slots_its_class_names(
        self, parameter, about, relative, involved
    ):
        assert (
            violated_on(
                parameter(
                    about_slot=about[0](),
                    quantity_involves_slot=[kind() for kind in involved],
                ),
                INVOLVES_SLOT,
            )
            == []
        )
        assert violated_on(
            parameter(
                about_slot=about[0](), quantity_involves_slot=[action_core.FrameSlot()]
            ),
            INVOLVES_SLOT,
        )
