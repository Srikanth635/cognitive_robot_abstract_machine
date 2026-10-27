"""
Resolutions for the tests: one as the resolver reports it, and a resolver replaying the
models' recorded answers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.knowledge.access import (
    AcceptanceSuite,
    HypothesisBackend,
    HypothesisStore,
    KnowledgeService,
)
from llm_reasoner.resolver.grounds import GroundKey
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.world import World
from typing_extensions import Any, List, Optional, Sequence

from cramera.resolution.resolver.designators import DesignatorQuery, PouringDesignator
from cramera.resolution.resolver.resolver import (
    DesignatorResolver,
    ResolvedAction,
    ResolvedDesignator,
    ResolvedSlot,
    ShownGround,
)
from cramera.resolution.resolver.runs import ResolutionRuns
from cramera.resolution.resolver.trial import OutcomeKey


class HandSide(Enum):
    """
    The hand an answer picks, standing in for an arm where a test reads only that the
    answer names one.
    """

    LEFT = "left"
    RIGHT = "right"


def arm_in(world: World, name: str) -> Arm:
    """
    :param world: A world holding a robot.
    :param name: The name of one of the robot's arms.
    :return: The arm of that name.
    """
    (arm,) = (
        arm
        for arm in world.get_semantic_annotations_by_type(Arm)
        if arm.name.name == name
    )
    return arm


def objects_given() -> ResolvedDesignator:
    """
    :return: Pouring from the given pouring cup into the given bowl, the arm and the
        tilt chosen, resting on a knowledge fragment and the bowl.
    """
    return ResolvedDesignator(
        query=DesignatorQuery.shipped(PouringDesignator.OBJECTS_GIVEN).source.text,
        designator="PouringAction",
        actions=[
            ResolvedAction(
                slots=[
                    ResolvedSlot(
                        "source_container",
                        "PouringCup (Static_MilkPitcher.stl)",
                        given=True,
                        serialized="Static_MilkPitcher.stl",
                    ),
                    ResolvedSlot(
                        "target_container",
                        "bowl.stl",
                        given=True,
                        serialized="bowl.stl",
                    ),
                    ResolvedSlot(
                        "arm", "PR2RightArm", given=False, serialized="PR2RightArm"
                    ),
                    ResolvedSlot("tilt_angle", "1.85", given=False, serialized=1.85),
                ],
                why="The default tilt lets the substance flow.",
                grounds=[
                    ShownGround(GroundKey.KNOWLEDGE, "ph_tilt"),
                    ShownGround(GroundKey.WORLD, "Body#1 (bowl.stl)"),
                ],
            )
        ],
        situation={},
        record_id=None,
        cost=0.05,
        run="a-run",
    )


CASES = Path(__file__).parent / "resolution_cases"
"""
The recorded answers the resolutions replay.
"""


def recorded(name: str) -> dict:
    """
    :return: A recording of the models' answers to one resolution.
    """
    return json.loads((CASES / f"{name}.json").read_text())


def resolver_replaying(
    name: str,
    store: HypothesisStore,
    runs: Path,
    times: int = 1,
    trial_outcomes: Optional[Sequence[Any]] = None,
) -> DesignatorResolver:
    """
    :param name: The recording to replay.
    :param store: The knowledge store every run starts from.
    :param runs: Where the runs are kept.
    :param times: How many runs the recording is replayed for.
    :param trial_outcomes: What the pours tried in the simulation bring about, in
        turn; every pour lands when none are given.
    :return: A resolver whose models replay the recording in each run, its pours tried
        by scripted trials rather than the simulation.
    """
    recording = recorded(name)
    return DesignatorResolver(
        simulate=ScriptedTrials(
            outcomes=(
                list(trial_outcomes)
                if trial_outcomes is not None
                else [LANDS] * len(recording["turns"]) * times
            )
        ),
        client=ScriptedClient(
            tool_turns=[json.dumps(turn["answer"]) for turn in recording["turns"]]
            * times
        ),
        baseline=KnowledgeService(
            backend=HypothesisBackend(
                client=ScriptedClient(
                    responses=[
                        json.dumps(answer) for answer in recording["generations"]
                    ]
                    * times
                ),
                suite=AcceptanceSuite(rules=[]),
                store=store,
            )
        ),
        runs=ResolutionRuns(directory=runs),
    )


# %% trials of a pour in the simulation


@dataclass
class FinishedTrial:
    """
    How a trial process finished, as the check reads it.
    """

    returncode: int
    """
    The process's exit status.
    """

    stderr: str = ""
    """
    What the process wrote to its error stream.
    """


@dataclass
class ScriptedTrials:
    """
    Runs no process: each call writes the next scripted outcome where the trial would,
    or finishes as a failed simulation when the outcome is ``None``.
    """

    outcomes: List[Any]
    """
    The outcomes to report in turn.
    """

    commands: List[List[str]] = field(default_factory=list)
    """
    The commands asked for, in order.
    """

    def __call__(self, command: List[str], **settings: Any) -> FinishedTrial:
        self.commands.append(command)
        outcome = self.outcomes.pop(0)
        if outcome is None:
            return FinishedTrial(
                returncode=1, stderr="Traceback\nMotionMadeNoProgress: out of reach\n"
            )
        Path(command[-1]).write_text(json.dumps(outcome))
        return FinishedTrial(returncode=0)


LANDS = {OutcomeKey.LANDS_IN_OPENING: True}
"""
An outcome of a pour landing in the target's opening.
"""

MISSES = {OutcomeKey.LANDS_IN_OPENING: False}
"""
An outcome of a pour missing the target's opening.
"""
