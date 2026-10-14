"""
Trying a resolved pouring in the simulation before it is accepted: the pour performed
from the scene's initial world in a process of its own, measured by where the source
container's mouth ends against the target's opening, and a pour that misses refused with
that outcome, so the resolver reads it and answers again.

Run as ``python -m cram_viz.resolution.trial <bundle> <values> <outcome file>`` it
performs one trial and writes its outcome to the file.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from time import perf_counter
from enum import StrEnum
from operator import attrgetter
from pathlib import Path

import numpy as np
from llm_reasoner.observation.events import new_call
from llm_reasoner.observation.sinks import EventSink, NullSink
from llm_reasoner.resolver.exceptions import OutcomeFellShort
from llm_reasoner.resolver.outcomes import OutcomeCheck
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any, Callable, Dict, List, Sequence

from cram_viz import configure_console_logging
from cram_viz.events import TrialFinished, TrialStarted
from cram_viz.resolution.execution import PouringExecution
from cram_viz.resolution.scene_world import SceneWorldBuilder
from cram_viz.resolution.slot_data import SlotData

logger = logging.getLogger(__spec__.name)
"""
This module's logger, named for the module even when ``python -m`` runs it as
``__main__``.
"""

TRIAL_MODULE = __spec__.name
"""
The module a trial runs as in a process of its own.
"""

OUTCOME_FILE = "outcome.json"
"""
The name of the file a trial writes its outcome to.
"""

COORDINATE_DIGITS = 3
"""
The decimals a length in an outcome is stated with: millimetres.
"""

OUTCOME_UNITS = (
    "metres; 'ahead' along the robot base's heading, 'right' to its right, as seen "
    "after the pour"
)
"""
How an outcome's lengths read.
"""

# %% where the mouth ends


class OutcomeKey(StrEnum):
    """
    The keys of a trial's outcome.
    """

    LANDS_IN_OPENING = "lands_in_opening"
    UNITS = "units"
    MOUTH_POSITION = "mouth_position"
    OPENING_CENTRE = "opening_centre"
    OPENING_RADIUS = "opening_radius"
    DISTANCE_FROM_OPENING_CENTRE = "mouth_distance_from_opening_centre"
    AHEAD_OF_OPENING_CENTRE = "mouth_ahead_of_opening_centre"
    RIGHT_OF_OPENING_CENTRE = "mouth_right_of_opening_centre"
    ABOVE_OPENING = "mouth_above_opening"
    MOTION_FAILED = "motion_failed"


def top_centre(body: Body) -> np.ndarray:
    """
    :return: The world position of the centre of the top of the body's bounding box,
        taken in the body's own frame: a container's opening.
    """
    box = body.visual.as_bounding_box_collection_in_frame(body).bounding_box()
    centre = np.array(
        [0.5 * (box.min_x + box.max_x), 0.5 * (box.min_y + box.max_y), box.max_z, 1.0]
    )
    return (body.global_transform.to_np() @ centre)[:3]


@dataclass(frozen=True)
class PourOutcome:
    """
    Where the source container's mouth ends after a pour, against the target's opening
    and as seen from the robot's base.
    """

    mouth: np.ndarray
    """
    The world position of the source container's mouth.
    """

    opening_centre: np.ndarray
    """
    The world position of the centre of the target's opening.
    """

    opening_radius: float
    """
    Half the smaller width of the target's opening.
    """

    robot_yaw: float
    """
    The heading of the robot's base, in radians.
    """

    @classmethod
    def measure(
        cls, source: HasRootBody, target: Body, robot_base: Body
    ) -> PourOutcome:
        """
        :return: Where the source's mouth is now, against the target's opening.
        """
        box = target.visual.as_bounding_box_collection_in_frame(target).bounding_box()
        return cls(
            mouth=top_centre(source.root),
            opening_centre=top_centre(target),
            opening_radius=0.5 * min(box.max_x - box.min_x, box.max_y - box.min_y),
            robot_yaw=float(robot_base.global_pose.to_quaternion().to_rpy()[2]),
        )

    @property
    def ahead(self) -> float:
        """
        :return: How far the mouth ends ahead of the opening's centre, along the
            robot's heading.
        """
        offset = self.mouth[:2] - self.opening_centre[:2]
        return float(offset @ [np.cos(self.robot_yaw), np.sin(self.robot_yaw)])

    @property
    def right(self) -> float:
        """
        :return: How far the mouth ends to the robot's right of the opening's centre.
        """
        offset = self.mouth[:2] - self.opening_centre[:2]
        return float(offset @ [np.sin(self.robot_yaw), -np.cos(self.robot_yaw)])

    @property
    def distance(self) -> float:
        """
        :return: How far the mouth ends from the opening's centre, across the floor.
        """
        return float(np.linalg.norm(self.mouth[:2] - self.opening_centre[:2]))

    @property
    def above(self) -> float:
        """
        :return: How high the mouth ends above the opening.
        """
        return float(self.mouth[2] - self.opening_centre[2])

    @property
    def lands_in_opening(self) -> bool:
        """
        :return: Whether the mouth ends above the opening, inside its rim.
        """
        return bool(self.distance <= self.opening_radius and self.above > 0)

    def as_data(self) -> Dict[str, Any]:
        """
        :return: The outcome as data the resolver reads.
        """
        return {
            OutcomeKey.LANDS_IN_OPENING: self.lands_in_opening,
            OutcomeKey.UNITS: OUTCOME_UNITS,
            OutcomeKey.MOUTH_POSITION: self.rounded(self.mouth),
            OutcomeKey.OPENING_CENTRE: self.rounded(self.opening_centre),
            OutcomeKey.OPENING_RADIUS: self.rounded([self.opening_radius])[0],
            OutcomeKey.DISTANCE_FROM_OPENING_CENTRE: self.rounded([self.distance])[0],
            OutcomeKey.AHEAD_OF_OPENING_CENTRE: self.rounded([self.ahead])[0],
            OutcomeKey.RIGHT_OF_OPENING_CENTRE: self.rounded([self.right])[0],
            OutcomeKey.ABOVE_OPENING: self.rounded([self.above])[0],
        }

    @staticmethod
    def rounded(values: Sequence[float]) -> List[float]:
        """
        :return: The lengths rounded to :data:`COORDINATE_DIGITS` decimals.
        """
        return [round(float(value), COORDINATE_DIGITS) for value in values]


# %% one trial


class PouringTrial:
    """
    Performs a resolved pouring in a world and measures where it ends.
    """

    @staticmethod
    def outcome_in(world: World, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        :param world: A world the robot can act in, its robot annotated.
        :param values: The resolved slots' values as data, by field name.
        :return: The outcome of performing the pouring there.
        """
        execution = PouringExecution(world=world, values=values)
        action = execution.action()
        execution.perform()
        return PourOutcome.measure(
            action.source_container, action.target_container, execution.robot.root
        ).as_data()


def main() -> None:
    """
    Perform one trial from a scene's initial world and write its outcome to a file.
    """
    configure_console_logging()
    bundle, values, outcome_file = sys.argv[1:4]
    outcome = PouringTrial.outcome_in(
        SceneWorldBuilder.robot_world(Path(bundle)), json.loads(values)
    )
    Path(outcome_file).write_text(json.dumps(outcome))
    logger.info("trial outcome: %s", outcome)


# %% the check


@dataclass(frozen=True)
class TriedPour:
    """
    One pour tried in the simulation.
    """

    values: Dict[str, Any]
    """
    The resolved slots' values tried, as data, by field name.
    """

    outcome: Dict[str, Any]
    """
    What the pour brought about.
    """


@dataclass
class SimulatedPouring(OutcomeCheck):
    """
    Tries each answer's pouring in the simulation, from the scene's initial world in a
    process of its own, and refuses a pour that misses the target's opening with its
    outcome, until the last allowed trial, which is accepted whatever it brings about.
    """

    bundle: Path
    """
    The scene's bundle directory, from which every trial's world is built anew.
    """

    slots: List[str]
    """
    The slots of the pouring tried, by field name.
    """

    allowed_trials: int
    """
    How many pours are tried before the last one is accepted.
    """

    run: Callable[..., Any] = subprocess.run
    """
    Runs a trial's process.
    """

    trials: List[TriedPour] = field(default_factory=list)
    """
    The pours tried, in order.
    """

    observer: EventSink = field(default_factory=NullSink)
    """
    Where each trial is reported to; nowhere by default.
    """

    def check(self, instances: Sequence[Any]) -> None:
        instance, *_ = instances
        values = {
            name: SlotData.from_value(attrgetter(name)(instance)) for name in self.slots
        }
        outcome = self.outcome_of(values)
        self.trials.append(TriedPour(values=values, outcome=outcome))
        if outcome[OutcomeKey.LANDS_IN_OPENING]:
            return
        if len(self.trials) >= self.allowed_trials:
            return
        raise OutcomeFellShort(outcome=outcome)

    def outcome_of(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        :return: The outcome of performing the pouring with the values, or of the
            simulation failing.
        """
        trial = new_call()
        self.observer.emit(TrialStarted(trial=trial, values=values))
        started = perf_counter()
        outcome = self.trial_outcome(values)
        self.observer.emit(
            TrialFinished(
                trial=trial, outcome=outcome, seconds=perf_counter() - started
            )
        )
        return outcome

    def trial_outcome(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        :return: The outcome of the process performing the pouring with the values.
        """
        with tempfile.TemporaryDirectory() as directory:
            outcome_file = Path(directory) / OUTCOME_FILE
            finished = self.run(
                [
                    sys.executable,
                    "-m",
                    TRIAL_MODULE,
                    str(self.bundle),
                    json.dumps(values),
                    str(outcome_file),
                ],
                capture_output=True,
                text=True,
            )
            if finished.returncode != 0:
                return {
                    OutcomeKey.LANDS_IN_OPENING: False,
                    OutcomeKey.MOTION_FAILED: finished.stderr.strip().splitlines()[-1],
                }
            return json.loads(outcome_file.read_text())


if __name__ == "__main__":
    main()
