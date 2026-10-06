"""
What the resolver may ask of stereotypical knowledge.

GUIDANCE, NOT SOLUTION
----------------------
The resolver never receives a hypothesis. It asks questions and receives cited
answers, which it is free to weigh, use partially, or reject outright. That is a
structural commitment rather than a convention: handing over the structure would
make following it the path of least resistance, and a stereotype is exactly the
kind of knowledge that is usually right and sometimes badly wrong.

EVERY ANSWER CARRIES ITS SOURCE
-------------------------------
Each fragment comes back wrapped with the record and fragment it came from, so a
resolution can state which stereotypical assumptions it leaned on and which
world facts overrode which defaults. A conclusion that cannot say what it relied
on cannot be argued with later.

ASSUMPTIONS ARE THE ONE ANSWER THAT IS NOT OPTIONAL
---------------------------------------------------
Everything else here is structure the resolver may take or leave. An assumption
is a claim about the world that must hold before the hypothesis applies at all,
so it is the one thing to check rather than weigh, and to report when it cannot
be checked.

ON NAVIGATING BACKWARDS
-----------------------
Going from a fragment to the structure that declares it could have been solved by
declaring inverse properties on the model -- the original ontology had seven of
them for exactly this. It is done here with an index built once per view instead.
The navigation is only ever needed while answering a question, an index cannot
fall out of step with the forward links the way a second mutable field can, and
the model stays free of fields that exist only to serve one caller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, Mapping, Optional, Sequence, TypeVar

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.hypothesis.record import Assumption, HypothesisRecord
from llm_reasoner.knowledge.hypothesis.status import Defeater
from llm_reasoner.knowledge.hypothesis.store import HypothesisStore, Lookup
from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ActionPhase,
    ActionPhaseStructure,
    ContinuousSignalPrediction,
    CorrectiveAction,
    MechanicalEventType,
    ModelUpdate,
    PredictionComparison,
    PredictionError,
    SensorimotorControlPoint,
)
from llm_reasoner.knowledge.microworlds.flanagan.gaze import GazeTarget
from llm_reasoner.knowledge.microworlds.flanagan.object_dynamics import (
    DynamicsModel,
    LimbDynamicsModel,
    ObjectDynamicsModel,
    ObjectDynamicsPhaseStructure,
    Representation,
    RepresentationPurpose,
)
from llm_reasoner.knowledge.microworlds.flanagan.prehension import (
    AnticipatoryGripAdjustment,
    GripLoadCoupling,
    PrehensionPhaseStructure,
)
from llm_reasoner.knowledge.microworlds.registry import select_modules
from llm_reasoner.knowledge.ontological.constraints import resolve
from llm_reasoner.knowledge.ontological.decorators import ConceptReference

T = TypeVar("T")

_CITATION = re.compile(r"\[(?P<record>[^\[\]/]+)/(?P<fragment>[^\[\]/]+)]")
"""
Matches the bracketed part of a citation.

Extracted by pattern rather than by splitting, because a citation leads with a human-
readable label that contains spaces and, often enough, a slash -- so splitting the whole
string finds the wrong boundary and silently resolves nothing.
"""


@dataclass(frozen=True)
class Cited(Generic[T]):
    """
    One answer, with the source it came from.

    Wrapping rather than returning bare fragments is deliberate. The resolver would
    otherwise have no way to attribute a decision, and an unattributed stereotype is
    indistinguishable from an observation -- which is precisely the confusion the
    type/token separation exists to prevent.
    """

    value: T
    record_id: str
    fragment_id: str
    label: str = ""
    degraded: bool = False
    """
    Whether the hypothesis was generated with no action-specific vocabulary.

    Carried on every answer because it qualifies how much weight the answer deserves,
    and the resolver has no other way to know.
    """

    @property
    def citation(self) -> str:
        return (
            f"{self.label or type(self.value).__name__} "
            f"[{self.record_id}/{self.fragment_id}]"
        )

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.citation


@dataclass
class HypothesisView:
    """
    A read-only window onto one hypothesis, answering fixed questions.

    Fixed rather than open-ended on purpose. An arbitrary query interface would let the
    resolver reach into the structure and, in effect, hold it; a closed set of questions
    keeps the coupling to what the resolver actually needs and makes each question's
    meaning something this module can be responsible for.
    """

    record: HypothesisRecord
    _owner_of: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index()

    def _index(self) -> None:
        """
        Map each fragment to the fragment that declares it.

        Built once, from the forward links only, so it cannot disagree with them.
        """
        for fragment in self.record.scope.fragments:
            for name in (
                "defines_phase",
                "defines_role",
                "defines_event_type",
                "defines_parameter",
                "defines_prediction",
                "defines_observation",
                "has_comparison",
                "defines_error",
                "defines_correction",
                "has_model_update",
                "defines_initial_state_estimate",
                "has_transition",
                "has_control_point",
                "has_condition_requirement",
                "defines_internal_model",
                "defines_desired_state",
                "defines_motor_command",
                "defines_efference_copy",
                "has_internal_model_composition",
                "has_sensorimotor_memory",
                "has_representation_update",
                "defines_dynamics_perturbation",
                "has_dynamics_transfer",
                "has_representation_dissociation",
                "has_force_coupling",
                "has_anticipatory_grip_adjustment",
            ):
                for owned in vars(fragment).get(name, ()) or ():
                    identity = getattr(owned, "local_id", None)
                    if identity:
                        self._owner_of[identity] = fragment

    # -- wrapping ----------------------------------------------------------

    def _cite(self, fragment: Any) -> Cited[Any]:
        return Cited(
            value=fragment,
            record_id=self.record.record_id,
            fragment_id=getattr(fragment, "local_id", ""),
            label=getattr(fragment, "label", "") or type(fragment).__name__,
            degraded=self.record.provenance.degraded,
        )

    def _cite_all(self, fragments: Iterable[Any]) -> list[Cited[Any]]:
        return [self._cite(f) for f in fragments if f is not None]

    # -- what the resolver asks -------------------------------------------

    def assumptions(self) -> tuple[Assumption, ...]:
        """
        What this hypothesis presupposes rather than establishes.

        The one answer to act on rather than weigh. Each is a claim about the world to
        verify before applying the hypothesis, and to report when it cannot be verified
        -- silently proceeding on an unchecked assumption is how a stereotype becomes a
        wrong answer that looks like a right one.
        """
        return self.record.assumptions

    def domain(self, class_type: type[T]) -> list[T]:
        """
        Return raw fragments of a type, including registered subclasses.

        The returned objects form a suitable explicit domain for an EQL query. Cited
        question methods remain the preferred interface when provenance must travel with
        each answer.
        """
        ordered: list[T] = []
        seen_identities: set[int] = set()
        if self.root is not None:
            for value in vars(self.root).values():
                candidates = (
                    value
                    if isinstance(value, (list, tuple, set, frozenset))
                    else (value,)
                )
                for candidate in candidates:
                    if isinstance(candidate, class_type):
                        ordered.append(candidate)
                        seen_identities.add(id(candidate))
        for candidate in self.record.scope.of_type(class_type):
            if id(candidate) not in seen_identities:
                ordered.append(candidate)
        return ordered

    def roles(self) -> list[Cited[Any]]:
        """
        The participant slots that must be bound to world entities.

        Resolution consists of binding these. Each slot names the role class it becomes
        once bound, so the resolver knows what kind of thing it is looking for without
        this package knowing what exists.
        """
        return self._cite_all(getattr(self.root, "defines_role", ()))

    def phases(self) -> list[Cited[Any]]:
        """
        The phases, in execution order where the transitions determine one.

        Ordered by walking the transition graph from the entry phase rather than by
        declaration order, because declaration order is an artifact of generation and
        carries no claim about sequence.
        """
        phases = list(getattr(self.root, "defines_phase", ()) or ())
        transitions = list(getattr(self.root, "has_transition", ()) or ())
        if not transitions:
            return self._cite_all(phases)

        successor: dict[int, Any] = {}
        entered: set[int] = set()
        for transition in transitions:
            source, target = transition.from_phase, transition.to_phase
            if source is not None and target is not None:
                successor.setdefault(id(source), target)
                entered.add(id(target))
        entry = next((p for p in phases if id(p) not in entered), None)
        if entry is None:
            return self._cite_all(phases)

        ordered, seen = [], set()
        current = entry
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            ordered.append(current)
            current = successor.get(id(current))
        ordered += [p for p in phases if id(p) not in seen]
        return self._cite_all(ordered)

    def subgoal_of(self, phase: Any) -> Optional[Cited[Any]]:
        """
        What a phase exists to achieve.
        """
        goal = getattr(self._unwrap(phase), "aims_for", None)
        return self._cite(goal) if goal is not None else None

    def parameters_for(self, phase: Any) -> list[Cited[Any]]:
        """
        The quantities a phase needs, as ranges rather than measurements.

        A range is the honest generic answer. Narrowing it is the resolver's job, from
        the world or from the request; a hypothesis that offered a point value would be
        claiming an observation it never made.
        """
        target = self._unwrap(phase)
        return self._cite_all(
            parameter
            for parameter in getattr(self.root, "defines_parameter", ()) or ()
            if any(
                p is target for p in getattr(parameter, "parameter_in_phase", ()) or ()
            )
        )

    def preconditions_of(self, phase: Any) -> list[Cited[Any]]:
        """
        What must hold for a phase to begin.
        """
        return self._requirements(phase, mode_names=("AT_ENTRY",))

    def invariants_of(self, phase: Any) -> list[Cited[Any]]:
        """
        What must hold at every instant of a phase.

        Different in kind from a precondition: an invariant is a guard checked
        throughout, and its violation is a failure event rather than a failed
        entry test. A resolver that treats the two alike will test grasp
        stability once and never again.
        """
        return self._requirements(phase, mode_names=("THROUGHOUT",))

    def outcomes_of(self, phase: Any) -> list[Cited[Any]]:
        """
        What a phase brings about.
        """
        return self._requirements(phase, mode_names=("AT_EXIT",))

    def _requirements(self, phase: Any, mode_names: Sequence[str]) -> list[Cited[Any]]:
        target = self._unwrap(phase)
        wanted = set(mode_names)
        out: list[Any] = []
        for requirement in getattr(target, "has_condition_requirement", ()) or ():
            mode = getattr(requirement, "mode", None)
            if mode is not None and getattr(mode, "name", str(mode)) in wanted:
                out.append(requirement)
        return self._cite_all(out)

    def boundaries_of(self, phase: Any) -> list[Cited[Any]]:
        """
        The events that end a phase.

        What the controller waits for. Distinct from what it guards against -- see
        :meth:`failure_modes`.
        """
        return self._cite_all(getattr(self._unwrap(phase), "terminated_by", ()) or ())

    def monitored_signals(self, phase: Any) -> list[Cited[Any]]:
        """
        What is tracked continuously during a phase.

        Empty for a phase that runs to a discrete boundary; non-empty is the signal that
        the phase has a control law rather than a stopping condition.
        """
        return self._cite_all(getattr(self._unwrap(phase), "monitors_signal", ()) or ())

    def failure_modes(self, phase: Any = None) -> list[Cited[Any]]:
        """
        What can go wrong, for one phase or for the whole action.

        Answered from the ``is_failure_event`` flag rather than by matching names, so it
        stays correct for a microworld that calls its failures something else.
        """
        events = (
            getattr(self._unwrap(phase), "terminated_by", ()) or ()
            if phase is not None
            else getattr(self.root, "defines_event_type", ()) or ()
        )
        candidates = list(events)
        if phase is None:
            for condition in self.record.scope.fragments:
                violation = getattr(condition, "violation_signalled_by", None)
                if violation is not None and violation not in candidates:
                    candidates.append(violation)
        return self._cite_all(
            e for e in candidates if getattr(e, "is_failure_event", False)
        )

    def expected_events_at(
        self,
        control_point: SensorimotorControlPoint | Cited[SensorimotorControlPoint],
    ) -> list[Cited[MechanicalEventType]]:
        """
        Return the mechanical events predicted at a control point.
        """
        target = self._unwrap(control_point)
        return self._cite_all(
            prediction.predicts_event
            for prediction in target.has_prediction
            if prediction.predicts_event is not None
        )

    def comparison_at(
        self,
        control_point: SensorimotorControlPoint | Cited[SensorimotorControlPoint],
    ) -> list[Cited[PredictionComparison]]:
        """
        Return the possible comparison outcomes at a control point.
        """
        target = self._unwrap(control_point)
        return self._cite_all(target.has_comparison)

    def prediction_errors_for(
        self,
        event: MechanicalEventType | Cited[MechanicalEventType],
    ) -> list[Cited[PredictionError]]:
        """
        Return discrepancies defined for predictions of an event.
        """
        if not isinstance(self.root, ActionPhaseStructure):
            return []
        target = self._unwrap(event)
        errors = [
            error
            for comparison in self.root.has_comparison
            if comparison.compares_prediction is not None
            and comparison.compares_prediction.predicts_event is target
            for error in comparison.has_error
        ]
        return self._cite_all(errors)

    def corrections_for(
        self,
        error: PredictionError | Cited[PredictionError],
    ) -> list[Cited[CorrectiveAction]]:
        """
        Return corrective actions triggered by a prediction error.
        """
        if not isinstance(self.root, ActionPhaseStructure):
            return []
        target = self._unwrap(error)
        return self._cite_all(
            correction
            for correction in self.root.defines_correction
            if target in correction.triggered_by_error
        )

    def model_updates_for(
        self,
        error: PredictionError | Cited[PredictionError],
    ) -> list[Cited[ModelUpdate]]:
        """
        Return representation updates triggered by a prediction error.
        """
        if not isinstance(self.root, ActionPhaseStructure):
            return []
        target = self._unwrap(error)
        return self._cite_all(
            update
            for update in self.root.has_model_update
            if target in update.triggered_by_error
        )

    def next_phase_after(
        self,
        comparison: PredictionComparison | Cited[PredictionComparison],
    ) -> list[Cited[ActionPhase]]:
        """
        Return phases released by a successful prediction comparison.
        """
        target = self._unwrap(comparison)
        return self._cite_all(
            transition.to_phase
            for transition in target.permits_transition
            if transition.to_phase is not None
        )

    def dynamics_model_for(
        self,
        phase: ActionPhase | Cited[ActionPhase],
    ) -> list[Cited[DynamicsModel]]:
        """
        Return represented dynamics governing a phase.
        """
        target = self._unwrap(phase)
        return self._cite_all(
            model for model in target.governed_by if isinstance(model, DynamicsModel)
        )

    def force_coupling_for(
        self,
        phase: ActionPhase | Cited[ActionPhase],
    ) -> list[Cited[GripLoadCoupling]]:
        """
        Return grip-load couplings active during a phase.
        """
        return self.force_couplings(phase)

    def continuous_predictions_for(
        self,
        phase: ActionPhase | Cited[ActionPhase],
    ) -> list[Cited[ContinuousSignalPrediction]]:
        """
        Return continuous sensory predictions active during a phase.
        """
        target = self._unwrap(phase)
        return self._cite_all(
            prediction
            for prediction in self.domain(ContinuousSignalPrediction)
            if target in prediction.predicted_for_phase
        )

    def gaze_target_for(
        self,
        phase: ActionPhase | Cited[ActionPhase],
    ) -> list[Cited[GazeTarget]]:
        """
        Return gaze targets that anticipate contact during a phase.
        """
        target = self._unwrap(phase)
        return self._cite_all(
            gaze_target
            for gaze_target in self.domain(GazeTarget)
            if target in gaze_target.target_for_phase
        )

    def corrective_for(self, event: Any) -> list[Cited[Any]]:
        """
        What to do when an expected signal does not arrive.

        The framework's own answer to a prediction mismatch, which is otherwise the
        point at which a controller has the least to go on.
        """
        target = self._unwrap(event)
        corrections: list[Any] = []
        for control_point in self.root.has_control_point:
            if control_point.at_event_type is not target:
                continue
            for comparison in control_point.has_comparison:
                corrections.extend(comparison.invokes_correction)
        return self._cite_all(corrections)

    def object_dynamics(self, phase: Any = None) -> list[Cited[ObjectDynamicsModel]]:
        """
        Return object-dynamics representations, optionally for one phase.
        """
        if not isinstance(self.root, ObjectDynamicsPhaseStructure):
            return []
        candidates = (
            self._unwrap(phase).governed_by
            if phase is not None
            else self.root.defines_internal_model
        )
        return [
            self._cite(model)
            for model in candidates
            if isinstance(model, ObjectDynamicsModel)
        ]

    def limb_dynamics(self, phase: Any = None) -> list[Cited[LimbDynamicsModel]]:
        """
        Return limb-dynamics representations, optionally for one phase.
        """
        if not isinstance(self.root, ObjectDynamicsPhaseStructure):
            return []
        candidates = (
            self._unwrap(phase).governed_by
            if phase is not None
            else self.root.defines_internal_model
        )
        return [
            self._cite(model)
            for model in candidates
            if isinstance(model, LimbDynamicsModel)
        ]

    def representation_for(
        self,
        estimate: Any,
        purpose: RepresentationPurpose,
    ) -> list[Cited[Representation]]:
        """
        Return representations holding an estimate for one purpose.
        """
        if not isinstance(self.root, ObjectDynamicsPhaseStructure):
            return []
        target = self._unwrap(estimate)
        representations = [
            model
            for model in self.root.defines_internal_model
            if isinstance(model, Representation) and model.purpose is purpose
        ]
        if isinstance(target, type):
            return self._cite_all(
                representation
                for representation in representations
                if any(
                    isinstance(value, target) for value in representation.holds_estimate
                )
            )
        return self._cite_all(
            representation
            for representation in representations
            if target in representation.holds_estimate
        )

    def force_couplings(
        self,
        phase: Any = None,
    ) -> list[Cited[GripLoadCoupling]]:
        """
        Return grip-load relations, optionally active in one phase.
        """
        if not isinstance(self.root, PrehensionPhaseStructure):
            return []
        couplings = self.root.has_force_coupling
        if phase is None:
            return [self._cite(coupling) for coupling in couplings]
        target = self._unwrap(phase)
        return [
            self._cite(coupling)
            for coupling in couplings
            if target in coupling.coupling_active_in
        ]

    def anticipatory_grip_adjustments(
        self,
        phase: Any = None,
    ) -> list[Cited[AnticipatoryGripAdjustment]]:
        """
        Return prediction-triggered grip changes, optionally for one phase.
        """
        if not isinstance(self.root, PrehensionPhaseStructure):
            return []
        adjustments = self.root.has_anticipatory_grip_adjustment
        if phase is None:
            return [self._cite(adjustment) for adjustment in adjustments]
        target = self._unwrap(phase)
        return [
            self._cite(adjustment)
            for adjustment in adjustments
            if target in adjustment.adjustment_active_in
        ]

    def declares(self, fragment: Any) -> Optional[Cited[Any]]:
        """
        The fragment that declares this one.

        The backwards navigation, served from the index rather than by scanning.
        """
        identity = getattr(self._unwrap(fragment), "local_id", None)
        owner = self._owner_of.get(identity) if identity else None
        return self._cite(owner) if owner is not None else None

    def follow(self, fragment: Any, path: str) -> list[Cited[Any]]:
        """
        Follow a dotted path from a fragment.

        The one open-ended question, for a resolver that needs something the fixed set
        does not cover. Kept read-only and citing, so an unanticipated need does not
        become a reason to hand over the structure.
        """
        return self._cite_all(resolve(self._unwrap(fragment), path))

    # -- feedback ----------------------------------------------------------

    def reject(
        self, reason: str, fragment: Any = None, source: str = "resolver"
    ) -> Defeater:
        """
        Record that this hypothesis did not survive contact with a situation.

        The other half of "guidance, not solution": a resolver that may reject needs
        somewhere for the rejection to go, or the same wrong stereotype is served again
        tomorrow. A defeater is retained on the record rather than deleting it, because
        a hypothesis that fails in one situation may hold in others, and the pattern of
        where it fails is the evidence that would eventually justify an exception rule.
        """
        identity = (
            getattr(self._unwrap(fragment), "local_id", None) if fragment else None
        )
        defeater = Defeater(reason=reason, source=source, fragment_id=identity)
        self.record.refute(defeater)
        return defeater

    # -- helpers -----------------------------------------------------------

    @property
    def root(self) -> Any:
        return self.record.root

    @property
    def degraded(self) -> bool:
        return self.record.provenance.degraded

    @staticmethod
    def _unwrap(fragment: Any) -> Any:
        """
        Accept either a fragment or an answer that wrapped one.
        """
        return fragment.value if isinstance(fragment, Cited) else fragment


@dataclass
class KnowledgeService:
    """
    The single entry point from resolution into stereotypical knowledge.

    Holds the store and, optionally, a way to generate what is missing. The
    resolver depends on this and on nothing else in ``knowledge``, which is what
    keeps the two halves separable: everything below is free to change so long as
    these questions still have answers.
    """

    store: HypothesisStore
    module_versions: Optional[Mapping[str, str]] = None
    """
    The module versions a served record must have been generated under.

    When omitted, each action is judged against the versions of the modules that action
    selects, so one service can serve actions whose selections differ.
    """

    prompt_version: str | None = None
    """
    The generation prompt version whose records may be served.

    ``None`` preserves compatibility for read-only consumers that do not enforce a
    generation contract.
    """

    generator: Optional[Any] = None
    """
    A ``HypothesisGenerator``, when the service may produce what it lacks.

    Optional because a resolver in a deployed system may be entitled to read knowledge
    but not to mint it, and because generation costs a large prompt that a latency-
    sensitive caller may prefer to refuse.
    """

    last_generation_failure: Optional[Exception] = field(default=None, init=False)
    """
    Why the most recent generation attempt produced nothing.

    Retained rather than raised, so that a caller which only wanted knowledge if it
    happened to be available is not interrupted, while one that cares can still find
    out. Cleared on the next successful generation.
    """

    def view_for(
        self,
        task: ConceptReference,
        generate_if_missing: bool = True,
        raise_on_failure: bool = False,
    ) -> Optional[HypothesisView]:
        """
        A view onto the best hypothesis for an action, or ``None``.

        ``None`` is a real answer, not a failure: no stereotype for this action
        means the resolver proceeds on the world alone, which is what it did
        before any of this existed.

        A generation failure therefore also returns ``None`` by default rather
        than propagating. Raising would make the caller handle two unrelated
        outcomes -- "there is no stereotype" and "one could not be produced" --
        at every call site, when only the second is ever actionable and only for
        callers positioned to act on it. Those callers pass
        ``raise_on_failure``; the rest read ``last_generation_failure`` if they
        care, and are otherwise undisturbed.

        The distinction still matters, which is why the failure is kept rather
        than discarded: no stereotype is a fact about the domain, while a
        failure to generate one is usually a fact about the prompt, the schema
        or the client.
        """
        lookup = self.explain(task)
        if lookup.record is not None:
            return HypothesisView(record=lookup.record)
        if not (generate_if_missing and self.generator is not None):
            return None
        try:
            result = self.generator.generate(task)
        except LLMReasonerError as failure:
            self.last_generation_failure = failure
            if raise_on_failure:
                raise
            return None
        self.last_generation_failure = None
        self.store.put(result.record)
        return HypothesisView(record=result.record) if result.admitted else None

    def explain(self, task: ConceptReference) -> Lookup:
        """
        Why there is or is not a hypothesis to serve.

        Distinguishes never generated from generated-and-stale from generated-and-
        refuted, which call for different responses: regenerate, regenerate from the
        stale one, or do not bother.
        """
        return self.store.lookup(
            task.iri,
            self._module_versions_for(task),
            self.prompt_version,
        )

    def _module_versions_for(self, task: ConceptReference) -> Mapping[str, str]:
        """
        Return the module versions a record for this action must match.
        """
        if self.module_versions is not None:
            return self.module_versions
        return select_modules(task).module_versions

    def cited(self, citation: str) -> Optional[Any]:
        """
        Resolve a citation back to the fragment it names.

        Works for superseded and refuted records too, because that is exactly when an
        old decision has to be explained.
        """
        match = _CITATION.search(citation)
        if match is None:
            return None
        record = self.store.get(match.group("record"))
        if record is None:
            return None
        return record.scope.by_id(match.group("fragment"))


__all__ = ["Cited", "HypothesisView", "KnowledgeService"]
