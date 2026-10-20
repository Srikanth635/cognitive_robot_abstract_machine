"""
For the frame of an action, a knowledge base can state each Failure: the slots (or parts of slots) it concerns, the observable condition that goes wrong, the Indicator by which it is noticed, the core parameter dimensions it comes from, and a Correction made of ParameterChanges. A ParameterChange names the dimension, the direction, the slot it moves towards, the condition it removes and the condition the new choice must meet. A Correction is marked as applicable during the action or only by retrying.

Generated from ``failures.ttl`` by ``python -m llm_reasoner.knowledge.ontology_translation``. Regenerate rather than edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import ClassVar, List, Optional

from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
from llm_reasoner.knowledge.ontologies.properties import ObjectPropertyDescriptor
from llm_reasoner.knowledge.ontologies.terms import (
    ClassTerm,
    ComplementOf,
    ConceptReference,
    DescriptionEntity,
    EnumeratedClass,
    IntersectionOf,
    NamedClass,
    PropertyRestriction,
    PropertyTerm,
    Quantifier,
    UnionOf,
)

# %% value partitions


class CorrectionTiming(StrEnum):
    """
    When a correction can be applied.
    """

    DURING_ACTION = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#DuringAction"
    """
    The correction can be applied while the action goes on.
    """

    RETRY_ONLY = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#RetryOnly"
    """
    The correction can only be applied by trying the action again.
    """


class IndicatorForm(StrEnum):
    """
    The form of the quantity by which a failure is noticed.
    """

    MEASURED_VALUE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#MeasuredValue"
    """
    A value of a quantity at some time, such as a distance between two parts.
    """

    ONSET_VALUE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#OnsetValue"
    """
    The value of a quantity at which a process starts.
    """

    HAPPENING = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#Happening"
    """
    Whether something happens or finishes.
    """


class Noticer(StrEnum):
    """
    Who notices a failure.
    """

    PERSON_OBSERVER = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#PersonObserver"
    """
    A person looking at the action notices it.
    """

    AGENT_SENSING = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#AgentSensing"
    """
    The agent performing the action notices it through its own sensing or motion feedback.
    """


class ParameterChangeDirection(StrEnum):
    """
    The direction in which a correction moves a parameter choice: larger or smaller, as the action core's change direction says, or towards, away from or to another alternative.
    """

    INCREASES = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Increases"
    """
    Becomes larger.
    """

    DECREASES = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Decreases"
    """
    Becomes smaller, or grows more slowly.
    """

    TOWARDS_TARGET = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#TowardsTarget"
    """
    The chosen position or orientation is moved towards a slot or part of one.
    """

    AWAY_FROM_TARGET = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#AwayFromTarget"
    """
    The chosen position or orientation is moved away from a slot or part of one.
    """

    TO_ALTERNATIVE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#ToAlternative"
    """
    Another of the available alternatives is chosen.
    """


# %% classes


@dataclass(eq=False)
class Correction(DescriptionEntity):
    """
    How a failure is put right: one or more parameter changes, possibly to several dimensions, together with whether it can be applied during the action or only by trying again.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#Correction",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#Repair",),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#correctableWhen",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#CorrectionTiming"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#hasChange",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#ParameterChange"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    correctable_when: Optional[CorrectionTiming] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#correctableWhen",
            functional=True,
        ).as_dict(),
    )
    """
    Whether the correction can be applied during the action or only by trying the action again.
    """

    has_change: List[ParameterChange] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#hasChange",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#ParameterChange",
        ).as_dict(),
    )
    """
    A parameter change that is part of the correction.
    """


@dataclass(eq=False)
class Failure(DescriptionEntity):
    """
    A way an action framed by a core:ActionFrame can go wrong, tied to the slots it concerns, the condition that goes wrong, how it is noticed, the parameter dimensions it comes from and how it is corrected.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#Failure",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#FailureDescription",),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#causedByDimension",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParameterDimension"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#failureConcernsSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#hasFailedCondition",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#ObservableCondition"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#hasIndicator",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#Indicator"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    caused_by_dimension: List[action_core.ParameterDimension] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#causedByDimension",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParameterDimension",
        ).as_dict(),
    )
    """
    A parameter dimension of the core whose choice the failure comes from.
    """

    failure_concerns_slot: List[action_core.FrameSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#failureConcernsSlot",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    A slot, or part of a slot, that the failure concerns; for a failure of a relation between slots, each of the slots related.
    """

    has_correction: List[Correction] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#hasCorrection",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#Correction",
        ).as_dict(),
    )
    """
    A correction that puts the failure right.
    """

    has_failed_condition: List[ObservableCondition] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#hasFailedCondition",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#ObservableCondition",
        ).as_dict(),
    )
    """
    The observable condition of the world that holds when the failure occurs.
    """

    has_indicator: List[Indicator] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#hasIndicator",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#Indicator",
        ).as_dict(),
    )
    """
    A quantity or occurrence by which the failure is noticed.
    """


@dataclass(eq=False)
class FailureStereotype(DescriptionEntity):
    """
    The ways an action framed by an action frame normally goes wrong, each with how it is noticed, what causes it and how it is corrected.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#FailureStereotype",
        linked_terms=(
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#failureStereotypeOf",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ActionFrame"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    failure_stereotype_of: Optional[action_core.ActionFrame] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#failureStereotypeOf",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ActionFrame",
        ).as_dict(),
    )
    """
    The action frame of the action core whose failures the stereotype holds.
    """

    has_failure: List[Failure] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#hasFailure",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#Failure",
        ).as_dict(),
    )
    """
    A way the action of the frame can go wrong.
    """


@dataclass(eq=False)
class Indicator(DescriptionEntity):
    """
    The quantity or occurrence by which a failure is noticed, such as how far one part ends from another, the value at which a process starts, or whether a motion finishes.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#Indicator",
        linked_terms=(
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasQuantityKind",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#QuantityKind"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#measuredOnSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_quantity_kind: Optional[action_core.QuantityKind] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasQuantityKind",
            functional=True,
        ).as_dict(),
    )
    """
    What kind of quantity something about an action is about, such as the quantity a choice sets, a condition constrains or a change changes.
    """

    has_indicator_form: Optional[IndicatorForm] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#hasIndicatorForm",
            functional=True,
        ).as_dict(),
    )
    """
    Whether the indicator is a measured value, the value at which something starts, or whether something happens.
    """

    measured_on_slot: List[action_core.FrameSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#measuredOnSlot",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    A slot, or part of a slot, between or on which the indicator's quantity is measured.
    """

    noted_by: List[Noticer] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#notedBy",
            functional=False,
        ).as_dict(),
    )
    """
    Who can notice the failure through the indicator.
    """


@dataclass(eq=False)
class ObservableCondition(DescriptionEntity):
    """
    A state of the world that a person or an agent can check, such as a state type holding or not holding of a slot's filler, stated as what the filler has or affords rather than as the class it belongs to.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#ObservableCondition",
        linked_terms=(
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#failedConditionOnSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    failed_condition_on_slot: List[action_core.FrameSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#failedConditionOnSlot",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    The slot whose filler the condition is about.
    """

    failed_condition_presence: Optional[action_core.ConditionPolarity] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#failedConditionPresence",
            functional=True,
        ).as_dict(),
    )
    """
    Whether the named state holds or does not hold when the failure is noticed.
    """

    failed_condition_state_type: List[ConceptReference] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#failedConditionStateType",
            functional=False,
        ).as_dict(),
    )
    """
    The state type the condition is about, named by the IRI of a CRAM-AICOR state type such as aicor:FlowingState or aicor:ContainedState.
    """


@dataclass(eq=False)
class ParameterChange(DescriptionEntity):
    """
    A change to one of the core's parameter dimensions that corrects a failure, in a direction and for the reason that it removes a failed condition.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#ParameterChange",
        linked_terms=(
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#changesDimension",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParameterDimension"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#correctsCondition",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#ObservableCondition"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#hasDirection",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#ParameterChangeDirection"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    change_towards_slot: List[action_core.FrameSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#changeTowardsSlot",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    The slot, or part of a slot, that the change moves the choice towards or away from.
    """

    changes_dimension: Optional[action_core.ParameterDimension] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#changesDimension",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParameterDimension",
        ).as_dict(),
    )
    """
    The parameter dimension of the core whose choice the change alters.
    """

    corrects_condition: List[ObservableCondition] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#correctsCondition",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#ObservableCondition",
        ).as_dict(),
    )
    """
    The failed condition whose holding is the reason for the change.
    """

    has_direction: Optional[ParameterChangeDirection] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#hasDirection",
            functional=True,
        ).as_dict(),
    )
    """
    The direction in which the change moves the choice.
    """

    requires_of_choice: List[ObservableCondition] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#requiresOfChoice",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#ObservableCondition",
        ).as_dict(),
    )
    """
    A condition a world can check that the newly chosen value must meet, such as being reachable by the agent.
    """


# %% property descriptors


@dataclass
class CausedByDimension(ObjectPropertyDescriptor):
    """
    A parameter dimension of the core whose choice the failure comes from.
    """


@dataclass
class ChangeTowardsSlot(ObjectPropertyDescriptor):
    """
    The slot, or part of a slot, that the change moves the choice towards or away from.
    """


@dataclass
class ChangesDimension(ObjectPropertyDescriptor):
    """
    The parameter dimension of the core whose choice the change alters.
    """


@dataclass
class CorrectsCondition(ObjectPropertyDescriptor):
    """
    The failed condition whose holding is the reason for the change.
    """


@dataclass
class FailedConditionOnSlot(ObjectPropertyDescriptor):
    """
    The slot whose filler the condition is about.
    """


@dataclass
class FailureConcernsSlot(ObjectPropertyDescriptor):
    """
    A slot, or part of a slot, that the failure concerns; for a failure of a relation between slots, each of the slots related.
    """


@dataclass
class FailureStereotypeOf(ObjectPropertyDescriptor):
    """
    The action frame of the action core whose failures the stereotype holds.
    """


@dataclass
class HasChange(ObjectPropertyDescriptor):
    """
    A parameter change that is part of the correction.
    """


@dataclass
class HasCorrection(ObjectPropertyDescriptor):
    """
    A correction that puts the failure right.
    """


@dataclass
class HasFailedCondition(ObjectPropertyDescriptor):
    """
    The observable condition of the world that holds when the failure occurs.
    """


@dataclass
class HasFailure(ObjectPropertyDescriptor):
    """
    A way the action of the frame can go wrong.
    """


@dataclass
class HasIndicator(ObjectPropertyDescriptor):
    """
    A quantity or occurrence by which the failure is noticed.
    """


@dataclass
class MeasuredOnSlot(ObjectPropertyDescriptor):
    """
    A slot, or part of a slot, between or on which the indicator's quantity is measured.
    """


@dataclass
class RequiresOfChoice(ObjectPropertyDescriptor):
    """
    A condition a world can check that the newly chosen value must meet, such as being reachable by the agent.
    """


# %% descriptor attachments

Failure.caused_by_dimension = CausedByDimension(Failure, "caused_by_dimension")
ParameterChange.change_towards_slot = ChangeTowardsSlot(
    ParameterChange, "change_towards_slot"
)
ParameterChange.changes_dimension = ChangesDimension(
    ParameterChange, "changes_dimension"
)
ParameterChange.corrects_condition = CorrectsCondition(
    ParameterChange, "corrects_condition"
)
ObservableCondition.failed_condition_on_slot = FailedConditionOnSlot(
    ObservableCondition, "failed_condition_on_slot"
)
Failure.failure_concerns_slot = FailureConcernsSlot(Failure, "failure_concerns_slot")
FailureStereotype.failure_stereotype_of = FailureStereotypeOf(
    FailureStereotype, "failure_stereotype_of"
)
Correction.has_change = HasChange(Correction, "has_change")
Failure.has_correction = HasCorrection(Failure, "has_correction")
Failure.has_failed_condition = HasFailedCondition(Failure, "has_failed_condition")
FailureStereotype.has_failure = HasFailure(FailureStereotype, "has_failure")
Failure.has_indicator = HasIndicator(Failure, "has_indicator")
Indicator.measured_on_slot = MeasuredOnSlot(Indicator, "measured_on_slot")
ParameterChange.requires_of_choice = RequiresOfChoice(
    ParameterChange, "requires_of_choice"
)
