"""
Tests of the structural rules the SHACL shapes of the action perspectives state: each
rule refuses or warns about an answer breaking it, lets one keeping it pass, and is told
to the model generating a hypothesis it applies to.
"""

from __future__ import annotations

import json

import pytest
from rdflib import Namespace

from llm_reasoner.knowledge.acceptance.rules import Severity
from llm_reasoner.knowledge.acceptance.shapes import ShapesHold, StructureShapes
from llm_reasoner.knowledge.generation.prompts import (
    GenerationPrompt,
    GenerationSection,
    structure_rule_entry,
)
from llm_reasoner.knowledge.generation.schema import GenerationSchema
from llm_reasoner.knowledge.ontologies import ActionPerspectiveFile
from llm_reasoner.knowledge.ontologies.action_perspectives import (
    action_core,
    action_frame,
    failures,
    image_schemas,
)

from .test_generation import bodies

CORE_SHAPES = Namespace(ActionPerspectiveFile.ACTION_CORE.shapes_namespace)
FRAME_SHAPES = Namespace(ActionPerspectiveFile.ACTION_FRAME.shapes_namespace)
SCHEMA_SHAPES = Namespace(ActionPerspectiveFile.IMAGE_SCHEMAS.shapes_namespace)
FAILURE_SHAPES = Namespace(ActionPerspectiveFile.FAILURES.shapes_namespace)

# %% helpers


class Slots:
    """
    The slots of a pouring frame the fragments under test refer to.
    """

    def __init__(self):
        self.agent = action_core.AgentSlot()
        self.source = action_core.SourceSlot()
        self.goal = action_core.GoalSlot()
        self.lip = action_core.PartSlot(is_part_of_slot=self.source)
        self.opening = action_core.PartSlot(is_part_of_slot=self.goal)


@pytest.fixture
def slots() -> Slots:
    return Slots()


def broken(fragment, shape) -> list:
    """
    :return: The violations of one shape the fragment's check reports.
    """
    return [
        violation
        for violation in ShapesHold().violations([fragment])
        if violation.shape == shape
    ]


# %% loading


class TestLoading:
    @pytest.mark.parametrize(
        "namespace", [CORE_SHAPES, FRAME_SHAPES, SCHEMA_SHAPES, FAILURE_SHAPES]
    )
    def test_the_shapes_of_every_perspective_module_are_shipped(self, namespace):
        assert any(
            str(shape.iri).startswith(str(namespace))
            for shape in StructureShapes.shipped().shapes
        )

    @pytest.mark.parametrize(
        "root, namespace",
        [
            (action_frame.FrameStereotype, FRAME_SHAPES),
            (image_schemas.SchemaStereotype, SCHEMA_SHAPES),
            (failures.FailureStereotype, FAILURE_SHAPES),
        ],
    )
    def test_a_perspectives_rules_are_told_to_the_model(self, root, namespace):
        system = GenerationPrompt().system(GenerationSchema.for_root(root))
        (rules,) = bodies(system, GenerationSection.STRUCTURE_RULES)
        told = json.loads(rules)
        own = [
            shape
            for shape in StructureShapes.shipped().shapes
            if str(shape.iri).startswith(str(namespace))
        ]
        assert own
        assert all(structure_rule_entry(shape) in told for shape in own)


# %% the action core


class TestCoreShapes:
    def test_a_quantity_is_not_measured_from_its_own_slot(self, slots):
        dimension = action_core.ParameterDimension(
            about_slot=slots.source, relative_to_slot=slots.source
        )
        assert broken(dimension, CORE_SHAPES.QuantityNotFromItsOwnSlot)

    def test_a_quantity_measured_from_another_slot_passes(self, slots):
        dimension = action_core.ParameterDimension(
            about_slot=slots.source, relative_to_slot=slots.goal
        )
        assert not broken(dimension, CORE_SHAPES.QuantityNotFromItsOwnSlot)

    def test_a_condition_does_not_involve_its_own_slot(self, slots):
        condition = action_core.SlotCondition(
            condition_on_slot=slots.source, condition_involves_slot=[slots.source]
        )
        assert broken(condition, CORE_SHAPES.ConditionNotInvolvingItsOwnSlot)

    def test_a_condition_involving_another_slot_passes(self, slots):
        condition = action_core.SlotCondition(
            condition_on_slot=slots.source, condition_involves_slot=[slots.agent]
        )
        assert not broken(condition, CORE_SHAPES.ConditionNotInvolvingItsOwnSlot)


# %% the action frame


class TestFrameShapes:
    def test_a_required_part_is_a_part_of_the_conditions_slot(self, slots):
        condition = action_frame.FeatureCondition(
            condition_on_slot=slots.goal, condition_requires_part=[slots.lip]
        )
        (violation,) = broken(condition, FRAME_SHAPES.FeaturePartOfConditionSlot)
        assert violation.involved == (slots.lip,)

    def test_a_part_of_the_conditions_slot_passes(self, slots):
        condition = action_frame.FeatureCondition(
            condition_on_slot=slots.source, condition_requires_part=[slots.lip]
        )
        assert not broken(condition, FRAME_SHAPES.FeaturePartOfConditionSlot)

    def test_an_expectations_conditions_are_about_its_slot(self, slots):
        elsewhere = action_core.SlotCondition(condition_on_slot=slots.source)
        expectation = action_frame.SlotExpectation(
            expects_of_slot=slots.goal, expects_filler_to_satisfy=[elsewhere]
        )
        (violation,) = broken(expectation, FRAME_SHAPES.ExpectationConditionsOnItsSlot)
        assert violation.involved == (elsewhere,)

    def test_a_condition_on_a_part_of_the_expected_slot_passes(self, slots):
        expectation = action_frame.SlotExpectation(
            expects_of_slot=slots.source,
            expects_filler_to_satisfy=[
                action_core.SlotCondition(condition_on_slot=slots.lip)
            ],
        )
        assert not broken(expectation, FRAME_SHAPES.ExpectationConditionsOnItsSlot)

    def test_a_second_expectation_of_a_slot_is_a_warning(self, slots):
        stereotype = action_frame.FrameStereotype(
            has_slot_expectation=[
                action_frame.SlotExpectation(expects_of_slot=slots.source),
                action_frame.SlotExpectation(expects_of_slot=slots.source),
            ]
        )
        violations = broken(stereotype, FRAME_SHAPES.OneExpectationPerSlot)
        assert violations
        assert {violation.severity for violation in violations} == {Severity.WARNING}

    def test_one_expectation_per_slot_passes(self, slots):
        stereotype = action_frame.FrameStereotype(
            has_slot_expectation=[
                action_frame.SlotExpectation(expects_of_slot=slots.source),
                action_frame.SlotExpectation(expects_of_slot=slots.goal),
            ]
        )
        assert not broken(stereotype, FRAME_SHAPES.OneExpectationPerSlot)


# %% the image schemas


class TestImageSchemaShapes:
    def test_a_requirements_slots_take_part_in_its_schema(self, slots):
        requirement = image_schemas.SchemaRequirement(
            subject_slot=slots.lip, reference_slot=slots.opening
        )
        schema = image_schemas.SourcePathGoalSchema(
            has_participation=[
                image_schemas.SchemaParticipation(participating_slot=slots.lip)
            ],
            has_requirement=[requirement],
        )
        (violation,) = broken(schema, SCHEMA_SHAPES.RequirementSlotsTakePart)
        assert violation.involved == (requirement,)

    def test_a_requirement_on_a_part_of_a_participant_passes(self, slots):
        schema = image_schemas.SourcePathGoalSchema(
            has_participation=[
                image_schemas.SchemaParticipation(participating_slot=slots.source),
                image_schemas.SchemaParticipation(participating_slot=slots.opening),
            ],
            has_requirement=[
                image_schemas.SchemaRequirement(
                    subject_slot=slots.lip, reference_slot=slots.opening
                )
            ],
        )
        assert not broken(schema, SCHEMA_SHAPES.RequirementSlotsTakePart)

    def test_the_subject_is_not_the_reference(self, slots):
        requirement = image_schemas.SchemaRequirement(
            subject_slot=slots.lip, reference_slot=slots.lip
        )
        assert broken(requirement, SCHEMA_SHAPES.SubjectIsNotTheReference)

    def test_a_deciding_quantity_is_about_the_requirements_slots(self, slots):
        unrelated = image_schemas.DecidingQuantity(about_slot=slots.agent)
        requirement = image_schemas.SchemaRequirement(
            subject_slot=slots.lip, reference_slot=slots.opening, decided_by=[unrelated]
        )
        (violation,) = broken(requirement, SCHEMA_SHAPES.QuantityAboutRequirementSlots)
        assert violation.involved == (unrelated,)

    def test_a_quantity_about_the_whole_of_a_requirement_slot_passes(self, slots):
        requirement = image_schemas.SchemaRequirement(
            subject_slot=slots.lip,
            reference_slot=slots.opening,
            decided_by=[image_schemas.DecidingQuantity(about_slot=slots.goal)],
        )
        assert not broken(requirement, SCHEMA_SHAPES.QuantityAboutRequirementSlots)


# %% the failures


def failure_corrected_by(change, **failure_fields) -> failures.Failure:
    """
    :return: A failure put right by one correction made of the change.
    """
    return failures.Failure(
        has_correction=[failures.Correction(has_change=[change])], **failure_fields
    )


class TestFailureShapes:
    def test_a_change_corrects_only_a_condition_of_its_failure(self, slots):
        own = action_core.SlotCondition(condition_on_slot=slots.lip)
        other = action_core.SlotCondition(condition_on_slot=slots.goal)
        change = failures.ParameterChange(corrects_condition=[other])
        failure = failure_corrected_by(change, has_failed_condition=[own])
        (violation,) = broken(
            failure, FAILURE_SHAPES.ChangeCorrectsItsFailuresCondition
        )
        assert violation.involved == (change,)

    def test_a_change_correcting_its_failures_condition_passes(self, slots):
        own = action_core.SlotCondition(condition_on_slot=slots.lip)
        failure = failure_corrected_by(
            failures.ParameterChange(corrects_condition=[own]),
            has_failed_condition=[own],
        )
        assert not broken(failure, FAILURE_SHAPES.ChangeCorrectsItsFailuresCondition)

    def test_a_change_towards_a_target_names_the_target(self):
        change = failures.ParameterChange(
            has_direction=failures.ParameterChangeDirection.TOWARDS_TARGET
        )
        assert broken(change, FAILURE_SHAPES.TargetNamedOnlyForTargetDirections)

    def test_only_a_change_towards_or_away_names_a_target(self, slots):
        change = failures.ParameterChange(
            has_direction=failures.ParameterChangeDirection.INCREASES,
            change_towards_slot=[slots.goal],
        )
        assert broken(change, FAILURE_SHAPES.TargetNamedOnlyForTargetDirections)

    def test_a_change_away_from_a_named_target_passes(self, slots):
        change = failures.ParameterChange(
            has_direction=failures.ParameterChangeDirection.AWAY_FROM_TARGET,
            change_towards_slot=[slots.goal],
        )
        assert not broken(change, FAILURE_SHAPES.TargetNamedOnlyForTargetDirections)

    def test_a_change_of_a_dimension_the_failure_does_not_come_from_is_a_warning(
        self,
    ):
        cause = action_core.ParameterDimension()
        moved = action_core.ParameterDimension()
        failure = failure_corrected_by(
            failures.ParameterChange(changes_dimension=moved),
            caused_by_dimension=[cause],
        )
        violations = broken(failure, FAILURE_SHAPES.ChangeMovesACausingDimension)
        assert violations
        assert {violation.severity for violation in violations} == {Severity.WARNING}

    def test_a_change_of_a_causing_dimension_passes(self):
        cause = action_core.ParameterDimension()
        failure = failure_corrected_by(
            failures.ParameterChange(changes_dimension=cause),
            caused_by_dimension=[cause],
        )
        assert not broken(failure, FAILURE_SHAPES.ChangeMovesACausingDimension)

    def test_a_condition_on_a_slot_the_failure_does_not_concern_is_a_warning(
        self, slots
    ):
        failure = failures.Failure(
            failure_concerns_slot=[slots.source],
            has_failed_condition=[
                action_core.SlotCondition(condition_on_slot=slots.goal)
            ],
        )
        violations = broken(failure, FAILURE_SHAPES.ObservedOnConcernedSlots)
        assert violations
        assert {violation.severity for violation in violations} == {Severity.WARNING}

    def test_an_indicator_on_a_part_of_a_concerned_slot_passes(self, slots):
        failure = failures.Failure(
            failure_concerns_slot=[slots.source],
            has_failed_condition=[
                action_core.SlotCondition(condition_on_slot=slots.lip)
            ],
            has_indicator=[failures.Indicator(about_slot=slots.lip)],
        )
        assert not broken(failure, FAILURE_SHAPES.ObservedOnConcernedSlots)
