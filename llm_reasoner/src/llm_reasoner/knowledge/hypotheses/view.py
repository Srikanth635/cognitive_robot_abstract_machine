"""
Questions about one stored hypothesis that cross several of its fragments, such as which
phase follows another or which corrections a control point at an event can invoke.

Questions a single fragment answers through its own fields are left to the fragment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import cached_property

from typing_extensions import Optional

from krrood.utils import resolve_class_from_full_name

from llm_reasoner.knowledge.storage.citation import Citation
from llm_reasoner.knowledge.representation.rdf import GraphReader
from llm_reasoner.knowledge.storage.record import HypothesisRecord
from llm_reasoner.knowledge.storage.store import StoredHypothesis
from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.ontologies.predictive_manipulation import (
    core,
    gaze,
    prehension,
)
from llm_reasoner.knowledge.ontologies.terms import DescriptionEntity


@dataclass
class UnknownCitation(LLMReasonerError):
    """
    A citation names a fragment its stored hypothesis does not have.
    """

    citation: Citation
    """
    The citation.
    """

    def __post_init__(self):
        super().__init__(f"the stored hypothesis has no fragment {self.citation.iri}")


@dataclass
class ForceDynamics:
    """
    The forces an event changes: the opposition it resolves, and the forces it starts
    and ends.
    """

    event: core.MechanicalEventType
    """
    The event.
    """

    driving_force: Optional[core.ForceComponent] = None
    """
    The force that overcomes the other at the event.
    """

    resisting_force: Optional[core.ForceComponent] = None
    """
    The force that is overcome at the event.
    """

    started_forces: list[core.ForceComponent] = field(default_factory=list)
    """
    The forces that begin to act at the event.
    """

    ended_forces: list[core.ForceComponent] = field(default_factory=list)
    """
    The forces that stop acting at the event.
    """


@dataclass
class HypothesisView:
    """
    One stored hypothesis, as instances of the generated vocabulary.
    """

    record: HypothesisRecord
    """
    The provenance of the hypothesis.
    """

    fragments: list[DescriptionEntity]
    """
    Every fragment of the hypothesis.
    """

    identifiers: dict[int, str]
    """
    The IRI each fragment is stored under, keyed by the fragment's identity.
    """

    @classmethod
    def of(cls, stored: StoredHypothesis) -> HypothesisView:
        """
        :return: The view of a stored hypothesis.
        """
        read = GraphReader().read(stored.graph)
        return cls(
            record=stored.record,
            fragments=read.fragments,
            identifiers=read.identifiers,
        )

    @cached_property
    def root(self) -> core.ActionPhaseStructure:
        """
        The fragment the hypothesis was generated for.
        """
        root_type = resolve_class_from_full_name(self.record.root_type)
        (root,) = [
            fragment for fragment in self.fragments if isinstance(fragment, root_type)
        ]
        return root

    # %% citations

    def citation_of(self, fragment: DescriptionEntity) -> Citation:
        """
        :return: The citation of a fragment of this hypothesis.
        """
        return Citation.from_iri(self.identifiers[id(fragment)])

    def fragment_cited(self, citation: Citation) -> DescriptionEntity:
        """
        :return: The fragment a citation names.
        :raises UnknownCitation: If this hypothesis has no such fragment.
        """
        cited = [
            fragment
            for fragment in self.fragments
            if self.identifiers[id(fragment)] == citation.iri
        ]
        if not cited:
            raise UnknownCitation(citation=citation)
        return cited[0]

    # %% phases

    def phases(self) -> list[core.ActionPhase]:
        """
        :return: The phases in the order the transitions reach them from the phases no
            transition leads into; phases no transition reaches come last, in the order
            the structure declares them.
        """
        declared = self.root.defines_phase
        entered = {
            id(transition.to_phase) for transition in self.root.defines_transition
        }
        queue = deque(phase for phase in declared if id(phase) not in entered)
        ordered: list[core.ActionPhase] = []
        while queue:
            phase = queue.popleft()
            if any(phase is seen for seen in ordered):
                continue
            ordered.append(phase)
            queue.extend(self.next_phases(phase))
        return ordered + [
            phase for phase in declared if not any(phase is seen for seen in ordered)
        ]

    def next_phases(self, phase: core.ActionPhase) -> list[core.ActionPhase]:
        """
        :return: The phases a transition leads into from the given phase.
        """
        return [
            transition.to_phase
            for transition in self.root.defines_transition
            if transition.from_phase is phase and transition.to_phase is not None
        ]

    def requirements_of(
        self, phase: core.ActionPhase, mode: core.ConditionMode
    ) -> list[core.ConditionRequirement]:
        """
        :return: The requirements on the phase that hold in the given mode, such as on
            entry or throughout.
        """
        return [
            requirement
            for requirement in phase.has_condition_requirement
            if requirement.has_condition_mode is mode
        ]

    def world_assumptions(self) -> list[core.ConditionRequirement]:
        """
        :return: The requirements of any phase that are assumed of the world rather
            than brought about by the action.
        """
        assumptions: list[core.ConditionRequirement] = []
        for phase in self.root.defines_phase:
            assumptions.extend(
                requirement
                for requirement in phase.has_condition_requirement
                if requirement.is_world_assumption
                and not any(requirement is seen for seen in assumptions)
            )
        return assumptions

    def gaze_fixations_for(self, phase: core.ActionPhase) -> list[gaze.GazeFixation]:
        """
        :return: The gaze fixations serving the phase.
        """
        return [
            fixation
            for fixation in self.root.defines_gaze_fixation
            if isinstance(fixation, gaze.GazeFixation)
            and fixation.serves_phase is phase
        ]

    def force_couplings_in(
        self, phase: core.ActionPhase
    ) -> list[prehension.GripLoadCoupling]:
        """
        :return: The grip-load couplings active in the phase; none when the structure
            is not one of prehension.
        """
        if not isinstance(self.root, prehension.PrehensionPhaseStructure):
            return []
        return [
            coupling
            for coupling in self.root.defines_force_coupling
            if any(active is phase for active in coupling.coupling_active_in)
        ]

    # %% events

    def failure_events(self) -> list[core.FailureEventType]:
        """
        :return: The failures the structure defines.
        """
        return [
            event
            for event in self.root.defines_event_type
            if isinstance(event, core.FailureEventType)
        ]

    def corrections_at(
        self, event: core.MechanicalEventType
    ) -> list[core.CorrectiveAction]:
        """
        :return: The corrections the comparisons at the event's control points invoke.
        """
        corrections: list[core.CorrectiveAction] = []
        for point in self.root.defines_control_point:
            if point.at_event_type is not event:
                continue
            for comparison in point.has_comparison:
                corrections.extend(comparison.invokes_correction)
        return corrections

    def force_dynamics(self) -> list[ForceDynamics]:
        """
        :return: The forces each mechanical event of the structure changes, for the
            events that change any.
        """
        return [
            ForceDynamics(
                event=event,
                driving_force=(
                    event.resolves_opposition.has_driving_force
                    if event.resolves_opposition is not None
                    else None
                ),
                resisting_force=(
                    event.resolves_opposition.has_resisting_force
                    if event.resolves_opposition is not None
                    else None
                ),
                started_forces=list(event.starts_force),
                ended_forces=list(event.ends_force),
            )
            for event in self.root.defines_event_type
            if isinstance(event, core.MechanicalEventType)
            and (
                event.resolves_opposition is not None
                or event.starts_force
                or event.ends_force
            )
        ]
