"""
Statecharts of plans, built but never run, whose life cycles a test sets itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from coraplex.plans.context_extensions import RobotAccess
from coraplex.robot_plans.actions.base import SequentialAction
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues
from cramph.node import EndedByOwner, StatechartNode
from cramph.state_history import StateHistoryItem
from cramph.statechart import Statechart
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

from cramera.live.bridge import Bridge
from cramera.live.visualization import BridgeStatechartObserver

# %% nodes of a plan


@dataclass(eq=False, repr=False, kw_only=True)
class MotionStep(EndedByOwner, MotionStatechartNode):
    """
    A motion of a plan, as far as publishing the plan reads it: its kind and life cycle.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class ActingOnABody(SequentialAction):
    """
    An action parameterized by the body it acts on, performed as one motion.
    """

    target_body: Body
    """
    The body the action acts on.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [MotionStep(name="moving")]


@dataclass(eq=False, repr=False, kw_only=True)
class ActingOnAnAnnotation(SequentialAction):
    """
    An action parameterized by a semantic annotation, performed as one motion.
    """

    target_annotation: SemanticAnnotation
    """
    The annotation the action acts on.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [MotionStep(name="moving")]


# %% statecharts of plans


@dataclass(kw_only=True)
class CompiledExecutor:
    """
    An executor that compiled a statechart, as far as a visualization added to it reads.
    """

    context: StatechartContext
    """
    The context the statechart is performed in.
    """

    statechart: Statechart
    """
    The compiled statechart.
    """


@pytest.fixture()
def plan_statechart(pr2_world_copy: World) -> Statechart:
    """
    :return: An empty statechart performed by the PR2, to which actions can be added
        without running it.
    """
    [robot] = pr2_world_copy.get_semantic_annotations_by_type(PR2)
    return Statechart(
        context=StatechartContext.with_extensions(
            pr2_world_copy, [RobotAccess(robot=robot)]
        )
    )


@dataclass(kw_only=True)
class FollowedStatechart:
    """
    A statechart a bridge follows, whose ticks a test records.
    """

    statechart: Statechart
    """
    The statechart, whose state history records the ticks.
    """

    bridge: Bridge
    """
    The bridge the plan and its statechart are published to.
    """

    observer: BridgeStatechartObserver
    """
    Publishes every tick the statechart's state history records.
    """

    def record(self, *, life_cycle: LifeCycleValues) -> None:
        """
        Record a tick at which every node of the statechart is in `life_cycle`.

        :param life_cycle: The life cycle every node is in at that tick.
        """
        for node in self.statechart.nodes:
            self.statechart.life_cycle_state[node] = life_cycle
        self.statechart.state_history.append(
            next_item=StateHistoryItem(
                tick_count=len(self.statechart.state_history),
                life_cycle_state=self.statechart.life_cycle_state,
                observation_state=self.statechart.observation_state,
            )
        )


@dataclass(kw_only=True)
class PlanExecution(FollowedStatechart):
    """
    A plan of an action acting on a body, whose statechart a bridge follows.
    """

    root: Sequence
    """
    The node the action is a step of.
    """

    action: ActingOnABody
    """
    The action of the plan.
    """

    motion: MotionStep
    """
    The motion the action is performed as.
    """


@pytest.fixture()
def plan_execution(plan_statechart: Statechart) -> PlanExecution:
    """
    :return: A plan acting on the milk, followed by a bridge that publishes the milk,
        before any tick was recorded.
    """
    milk = Body(name=PrefixedName("milk.stl", prefix="world"))
    action = ActingOnABody(target_body=milk)
    root = Sequence(nodes=[action])
    plan_statechart.add_node(root)
    bridge = Bridge()
    bridge.publish_bodies({"milk.stl": milk})
    observer = BridgeStatechartObserver(bridge=bridge, statechart=plan_statechart)
    observer.follow()
    [motion] = action.children
    return PlanExecution(
        statechart=plan_statechart,
        root=root,
        action=action,
        motion=motion,
        bridge=bridge,
        observer=observer,
    )
