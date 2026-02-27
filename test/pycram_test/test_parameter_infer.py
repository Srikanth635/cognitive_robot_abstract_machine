from dataclasses import dataclass

from typing_extensions import List

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import ApproachDirection, VerticalAlignment, Arms
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.parameter_inference import (
    ParameterInferenceRule,
    ParameterIdentifier,
    T,
    PlanDomain,
    ValueDomainSpecification,
)
from pycram.parameter_rules.default_type_domains import (
    EnumDomainSpecification,
    GraspDomainSpecification,
    load_default_domains,
)
from pycram.robot_plans import (
    PickUpActionDescription,
    MoveTorsoActionDescription,
    ParkArmsActionDescription,
    PlaceActionDescription,
)
from semantic_digital_twin.datastructures.definitions import TorsoState


@dataclass
class DummyRule(ParameterInferenceRule):

    n_th: int

    def _apply(self, domain: List, context: Context) -> List:
        return domain[: self.n_th]


@dataclass
class EffectRule(ParameterInferenceRule):

    def _apply(self, domain: List[T], context: Context) -> List[T]:
        return [Arms.BOTH, Arms.RIGHT]

    def effect(self):
        new_rule = DummyRule(
            self.parameter_type, self.action_description, self.parameter_name, 1
        )
        self.parameter_infeerer.add_rule(new_rule)


def test_infer_enum_domain(immutable_model_world):

    world, robot_view, context = immutable_model_world

    pick_action = PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        ...,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot_view.right_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)

    plan.parameter_infeerer.add_domains(
        EnumDomainSpecification(Arms),
        GraspDomainSpecification(GraspDescription, robot_view.left_arm.manipulator),
    )

    assert plan.parameter_infeerer.get_domain_for_type(Arms) == [
        Arms.LEFT,
        Arms.RIGHT,
        Arms.BOTH,
    ]

    grasp_domain = plan.parameter_infeerer.get_domain_for_type(GraspDescription)

    assert len(grasp_domain) == 12


def test_infer_domain_with_rules(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pick_action = PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        ...,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot_view.right_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)

    plan.parameter_infeerer.add_domains(
        EnumDomainSpecification(Arms),
        GraspDomainSpecification(GraspDescription, robot_view.left_arm.manipulator),
    )

    plan.parameter_infeerer.add_rule(DummyRule(Arms, pick_action, "arm", 1))

    arm_domain = plan.parameter_infeerer.infer_domain_for_parameter(
        ParameterIdentifier(pick_action, "arm")
    )

    assert len(arm_domain) == 1
    assert arm_domain[0] == Arms.LEFT


def test_rule_effect(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pick_action = PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        ...,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot_view.right_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)

    plan.parameter_infeerer.add_domains(
        EnumDomainSpecification(Arms),
        GraspDomainSpecification(GraspDescription, robot_view.left_arm.manipulator),
    )

    arm_domain = plan.parameter_infeerer.infer_domain_for_parameter(
        ParameterIdentifier(pick_action, "arm")
    )

    assert len(arm_domain) == 3
    assert arm_domain == [Arms.LEFT, Arms.RIGHT, Arms.BOTH]

    plan.parameter_infeerer.add_rule(EffectRule(Arms, pick_action, "arm"))

    arm_domain = plan.parameter_infeerer.infer_domain_for_parameter(
        ParameterIdentifier(pick_action, "arm")
    )

    assert len(arm_domain) == 2
    assert arm_domain == [Arms.BOTH, Arms.RIGHT]
    assert len(plan.parameter_infeerer.parameter_rules) == 2

    arm_domain = plan.parameter_infeerer.infer_domain_for_parameter(
        ParameterIdentifier(pick_action, "arm")
    )

    assert len(arm_domain) == 1
    assert arm_domain == [Arms.BOTH]
    assert len(plan.parameter_infeerer.parameter_rules) == 3


def test_empty_domain(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pick_action = PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        ...,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot_view.right_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)

    arm_type_domain = plan.parameter_infeerer.get_domain_for_type(Arms)

    assert len(arm_type_domain) == 0
    assert arm_type_domain == []

    arm_domain = plan.parameter_infeerer.infer_domain_for_parameter(
        ParameterIdentifier(pick_action, "arm")
    )

    assert len(arm_domain) == 0
    assert arm_domain == []


def test_no_body_domain(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pick_action = PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        ...,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot_view.right_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)

    plan.parameter_infeerer.add_domains(
        EnumDomainSpecification(Arms),
        GraspDomainSpecification(GraspDescription, robot_view.right_arm.manipulator),
    )

    body_domain = plan.parameter_infeerer.infer_domain_for_parameter(
        ParameterIdentifier(pick_action, "object_designator")
    )

    assert len(body_domain) == 1
    assert body_domain == [world.get_body_by_name("milk.stl")]


def test_arms_fit_grasp_rule(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pick_action = PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        ...,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot_view.right_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)


def test_domain_creation(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pick_action = PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        ...,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot_view.right_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)

    pd = PlanDomain(plan)

    pd.add_domains(
        EnumDomainSpecification(Arms),
        GraspDomainSpecification(GraspDescription, robot_view.left_arm.manipulator),
    )

    pd.create_plan_domain()

    assert len(pd.domain_specifications) == 2
    designator_domain = pd.designator_domains[pick_action]
    assert len(designator_domain.parameter_domains) == 3
    assert designator_domain.designator == pick_action
    assert type(designator_domain.parameter_domains["arm"]) == EnumDomainSpecification
    assert (
        type(designator_domain.parameter_domains["grasp_description"])
        == GraspDomainSpecification
    )
    assert (
        type(designator_domain.parameter_domains["object_designator"])
        == ValueDomainSpecification
    )

    designator_domain_list = designator_domain.domain()
    assert len(designator_domain_list) == 16
    assert Arms.LEFT in designator_domain_list
    assert world.get_body_by_name("milk.stl") in designator_domain_list


def test_more_complex_plan(immutable_model_world):
    world, robot_view, context = immutable_model_world

    plan = SequentialPlan(
        context,
        MoveTorsoActionDescription(TorsoState.HIGH),
        ParkArmsActionDescription(Arms.BOTH),
        PickUpActionDescription(
            world.get_body_by_name("milk.stl"),
            ...,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.NoAlignment,
                robot_view.right_arm.manipulator,
            ),
        ),
        place := PlaceActionDescription(
            world.get_body_by_name("milk.stl"),
            PoseStamped(),
            ...,
        ),
    )

    pd = PlanDomain(plan)

    pd.add_domains(
        EnumDomainSpecification(Arms),
        EnumDomainSpecification(TorsoState),
        GraspDomainSpecification(GraspDescription, robot_view.right_arm.manipulator),
    )

    pd.create_plan_domain()

    assert len(pd.domain_specifications) == 3
    assert len(pd.designator_domains) == 4

    place_domain = pd.designator_domains[place]

    assert len(place_domain.parameter_domains) == 3
    assert [type(domain) for domain in place_domain.parameter_domains.values()] == [
        ValueDomainSpecification,
        ValueDomainSpecification,
        EnumDomainSpecification,
    ]

    assert len(place_domain.domain()) == 5
