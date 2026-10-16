"""
What is recorded of each run of the comparison and how the arms are summarised: the
values chosen, what they rest on and whether that is the action frame, the knowledge
tools read, the pours tried in the simulation and the cost.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from statistics import mean

from llm_reasoner.clients.usage import TokenPrices
from llm_reasoner.knowledge.access import (
    ActionPerspectiveFile,
    Citation,
    KnowledgeToolName,
)
from llm_reasoner.resolver.grounds import GroundKey
from typing_extensions import Any, List, Optional, Sequence

from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.world_description.world_entity import Body

from cram_viz.resolution.resolver import ResolvedDesignator
from cram_viz.resolution.trial import OutcomeKey, TriedPour
from experiments.llm_reasoner_kitchen.cases import name_of
from experiments.llm_reasoner_kitchen.resolution import Resolution

TOOL_CALLED = "ToolCalled"
"""
The kind of the event a run logs when the model calls a tool.
"""


class EventKey(StrEnum):
    """
    The keys of a logged event read here.
    """

    EVENT = "event"
    TOOL = "tool"


class Measurement(StrEnum):
    """
    Where a run is measured.
    """

    SIMULATED_POUR = "simulated_pour"
    """
    cram_viz's pour, each answer poured in the simulation.
    """

    KITCHEN = "kitchen"
    """
    The kitchen's knowledge cases, without an outcome.
    """


class ComparisonArm(StrEnum):
    """
    The two ways the resolver is run, which differ only in the perspectives consulted.
    """

    WITHOUT_FRAME = "without_frame"
    """
    The resolver as it is by default: the phase structure only.
    """

    WITH_FRAME = "with_frame"
    """
    The phase structure, the action core and the action frame on it.
    """

    @property
    def perspectives(self) -> tuple[ActionPerspectiveFile, ...]:
        """
        The perspectives on the action core the arm consults.
        """
        if self is ComparisonArm.WITHOUT_FRAME:
            return ()
        return (ActionPerspectiveFile.ACTION_FRAME,)


# %% a run


@dataclass(frozen=True)
class GroundRecord:
    """
    One thing an answer rests on.
    """

    source: GroundKey
    """
    The source it belongs to: the knowledge or the world.
    """

    name: str
    """
    Its name, or the IRI citing it.
    """

    cites_perspective: bool
    """
    Whether it is a fragment of the action core or of a perspective on it.
    """


@dataclass(frozen=True)
class TrialRecord:
    """
    One pour tried in the simulation.
    """

    lands: bool
    """
    Whether the pitcher's mouth ended over the target's opening.
    """

    distance: Optional[float]
    """
    How far the mouth ended from the opening's centre, in metres, when it was measured.
    """

    values: dict[str, Any]
    """
    The values the pour was tried with, by field name.
    """

    @classmethod
    def of(cls, tried: TriedPour) -> TrialRecord:
        """
        :return: The record of a pour tried in the simulation.
        """
        return cls(
            lands=bool(tried.outcome.get(OutcomeKey.LANDS_IN_OPENING)),
            distance=tried.outcome.get(OutcomeKey.DISTANCE_FROM_OPENING_CENTRE),
            values=dict(tried.values),
        )


@dataclass(frozen=True)
class RunRecord:
    """
    One run of one case in one arm.
    """

    measurement: Measurement
    """
    Where the run was measured.
    """

    case: str
    """
    The case run, such as ``pour_geometry_open`` or ``K2``.
    """

    arm: ComparisonArm
    """
    The arm the run belongs to.
    """

    run: int
    """
    The run's number within its case and arm.
    """

    values: dict[str, str]
    """
    The values the resolver chose, by field name; the values the query gave are left
    out.
    """

    grounds: List[GroundRecord]
    """
    What the answer rests on.
    """

    knowledge_tools: List[str]
    """
    The knowledge tools the model called, in order.
    """

    cost: float
    """
    What the resolver's conversations cost, in dollars; generating knowledge is left
    out.
    """

    trials: List[TrialRecord] = field(default_factory=list)
    """
    The pours tried in the simulation, in order; none in the kitchen.
    """

    meets: Optional[bool] = None
    """
    Whether the actions met what the case expects; ``None`` when the case expects
    nothing checked here.
    """

    @property
    def first_try_success(self) -> Optional[bool]:
        """
        Whether the first pour tried landed in the opening; ``None`` when none was
        tried.
        """
        if not self.trials:
            return None
        return self.trials[0].lands

    @property
    def rests_on_a_perspective(self) -> bool:
        """
        Whether the answer rests on a fragment of the action core or a perspective.
        """
        return any(ground.cites_perspective for ground in self.grounds)

    def to_json(self) -> dict[str, Any]:
        """
        :return: The record as JSON data, with its first-try success.
        """
        return {**asdict(self), "first_try_success": self.first_try_success}

    @classmethod
    def of_designator(
        cls,
        resolved: ResolvedDesignator,
        events: Path,
        case: str,
        arm: ComparisonArm,
        run: int,
    ) -> RunRecord:
        """
        :param resolved: What cram_viz resolved, with the pours it tried.
        :param events: The events the run logged.
        :return: The record of a run of cram_viz's pour.
        """
        (action,) = resolved.actions
        perspective_records = set(resolved.perspective_record_ids)
        return cls(
            measurement=Measurement.SIMULATED_POUR,
            case=case,
            arm=arm,
            run=run,
            values={slot.name: slot.value for slot in action.slots if not slot.given},
            grounds=[
                GroundRecord(
                    source=ground.source,
                    name=ground.identifier or ground.name,
                    cites_perspective=cites_one_of(
                        ground.source, ground.identifier, perspective_records
                    ),
                )
                for ground in action.grounds
            ],
            knowledge_tools=knowledge_tools_in(events),
            cost=resolved.cost,
            trials=[TrialRecord.of(tried) for tried in resolved.trials],
        )

    @classmethod
    def of_kitchen(
        cls,
        resolution: Resolution,
        case: str,
        arm: ComparisonArm,
        run: int,
        prices: TokenPrices,
        meets: Optional[bool] = None,
    ) -> RunRecord:
        """
        :param resolution: What the kitchen's resolver resolved, with its backend.
        :param prices: What the model charges.
        :param meets: Whether the actions met what the case expects, when it expects
            something.
        :return: The record of a run of a kitchen case.
        """
        backend = resolution.backend
        consultation = backend.consultation
        perspective_records = (
            set() if consultation is None else set(consultation.perspective_record_ids)
        )
        (justification,) = backend.justifications
        return cls(
            measurement=Measurement.KITCHEN,
            case=case,
            arm=arm,
            run=run,
            values=chosen_values(justification.instance),
            grounds=[
                GroundRecord(
                    source=ground.source,
                    name=ground.identifier,
                    cites_perspective=cites_one_of(
                        ground.source, ground.identifier, perspective_records
                    ),
                )
                for ground in justification.grounds
            ],
            knowledge_tools=[
                call.name
                for conversation in resolution.conversations
                for call in conversation.calls
                if call.name in set(KnowledgeToolName)
            ],
            cost=resolution.cost(prices),
            meets=meets,
        )


def cites_one_of(source: GroundKey, identifier: str, records: set[str]) -> bool:
    """
    :return: Whether a ground cites a fragment of one of the given stored hypotheses.
    """
    return (
        source is GroundKey.KNOWLEDGE
        and bool(identifier)
        and Citation.from_iri(identifier).record_id in records
    )


def knowledge_tools_in(events: Path) -> List[str]:
    """
    :return: The knowledge tools the model called in a run, in order, from its events.
    """
    logged = [json.loads(line) for line in events.read_text().splitlines() if line]
    return [
        event[EventKey.TOOL]
        for event in logged
        if event.get(EventKey.EVENT) == TOOL_CALLED
        and event.get(EventKey.TOOL) in set(KnowledgeToolName)
    ]


def chosen_values(action: Any) -> dict[str, str]:
    """
    :return: What an action holds, by field name, as text: its plain values, its
        enumeration members by name and the objects it acts on by
        :func:`written_object`; anything else is left out.
    """
    return {
        name: written
        for name, value in vars(action).items()
        if not name.startswith("_")
        for written in [written_value(value)]
        if written is not None
    }


def written_value(value: Any) -> Optional[str]:
    """
    :return: A value an action holds, as text; ``None`` for a value neither plain, an
        enumeration member nor an object of the world.
    """
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (bool, int, float, str)):
        return str(value)
    if isinstance(value, (Body, HasRootBody)):
        return written_object(value)
    return None


def written_object(chosen: Body | HasRootBody) -> str:
    """
    :return: An object of the world as text: a body by its name, an annotated object by
        its kind and its body's name.
    """
    if isinstance(chosen, Body):
        return name_of(chosen)
    return f"{type(chosen).__name__} ({name_of(chosen.root)})"


# %% summarising the arms


@dataclass(frozen=True)
class ArmSummary:
    """
    The runs of one case in one arm, summarised.
    """

    measurement: Measurement
    """
    Where the runs were measured.
    """

    case: str
    """
    The case run.
    """

    arm: ComparisonArm
    """
    The arm.
    """

    runs: int
    """
    How many runs there were.
    """

    first_try_successes: int
    """
    How many runs' first pour landed in the opening.
    """

    mean_trials: float
    """
    How many pours a run tried, on average; none in the kitchen.
    """

    runs_resting_on_a_perspective: int
    """
    How many runs' answers rest on the action core or a perspective.
    """

    mean_cost: float
    """
    What a run cost, on average, in dollars.
    """

    @classmethod
    def of(cls, runs: Sequence[RunRecord]) -> ArmSummary:
        """
        :param runs: The runs of one case in one arm.
        :return: Their summary.
        """
        first = runs[0]
        return cls(
            measurement=first.measurement,
            case=first.case,
            arm=first.arm,
            runs=len(runs),
            first_try_successes=sum(run.first_try_success is True for run in runs),
            mean_trials=mean(len(run.trials) for run in runs),
            runs_resting_on_a_perspective=sum(
                run.rests_on_a_perspective for run in runs
            ),
            mean_cost=mean(run.cost for run in runs),
        )


def summarise(records: Sequence[RunRecord]) -> List[ArmSummary]:
    """
    :return: One summary per measurement, case and arm, in the order the records first
        give them.
    """
    groups: dict[tuple[Measurement, str, ComparisonArm], List[RunRecord]] = {}
    for record in records:
        groups.setdefault((record.measurement, record.case, record.arm), []).append(
            record
        )
    return [ArmSummary.of(group) for group in groups.values()]
