"""
Tests of the live bridge's snapshots of a plan and of the statechart it runs as.

The bridge runs against mimics of statechart nodes and statecharts: it reads only their
life cycles, children, parents, transitions and fields, so no simulation is needed.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field

import pytest
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor, PlanExecutor
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from cramph.data_types import LifeCycleValues
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from typing_extensions import Any, Dict, List

from cram_viz.live import hooks
from cram_viz.live.bridge import BRIDGE, LIFE_CYCLE_TO_STATUS, Bridge, PlanNodeKey

# %% mimics of a plan


@dataclass(kw_only=True)
class PlanNodeMimic:
    """
    A node of a plan, as far as the bridge reads it.
    """

    life_cycle_state: LifeCycleValues = LifeCycleValues.NOT_STARTED
    """
    Where the node is in its life cycle.
    """

    children: List[PlanNodeMimic] = field(default_factory=list)
    """
    The nodes directly below this one.
    """

    arm: Any = None
    """
    The arm the node moves, if it moves one.
    """

    obj: Any = None
    """
    The object the node acts on, if it acts on one.
    """


@dataclass(eq=False, kw_only=True)
class ArmOfTheWorld(SemanticAnnotation):
    """
    An arm as a world annotates it, named after its class as an unnamed annotation is.
    """


@dataclass(eq=False, kw_only=True)
class OtherArmOfTheWorld(SemanticAnnotation):
    """
    A second arm as a world annotates it.
    """


@dataclass(kw_only=True)
class StatechartMimic:
    """
    The statechart of a plan, as far as the bridge reads its plan nodes.
    """

    top_level_nodes: List[PlanNodeMimic]
    """
    The nodes no other node is above.
    """


@dataclass(kw_only=True)
class SmallPlan:
    """
    A plan of a node holding an action holding a motion, bound to a bridge.
    """

    bridge: Bridge
    """
    The bridge bound to the plan.
    """

    top: PlanNodeMimic
    """
    The node at the top of the plan.
    """

    action: PlanNodeMimic
    """
    The action below the top node.
    """

    motion: PlanNodeMimic
    """
    The motion below the action.
    """


@pytest.fixture()
def small_plan() -> SmallPlan:
    """
    :return: A bridge bound to a plan whose action moves the left arm and whose motion
        acts on the milk, a body the bridge publishes.
    """
    motion = PlanNodeMimic(obj=types.SimpleNamespace(name="world/milk.stl"))
    action = PlanNodeMimic(arm="LEFT", children=[motion])
    top = PlanNodeMimic(children=[action])
    bridge = Bridge()
    bridge._bodies = {"milk.stl": object(), "__base__": object()}
    bridge._plan = StatechartMimic(top_level_nodes=[top])
    return SmallPlan(bridge=bridge, top=top, action=action, motion=motion)


def published(bridge: Bridge) -> List[Dict[str, Any]]:
    """
    :return: The nodes of the bridge's newest plan snapshot, top first, each followed by
        the nodes below it.
    """
    return bridge.get_plan()["nodes"]


# %% the plan snapshot


class TestPlanSnapshot:
    @pytest.mark.parametrize("life_cycle", list(LifeCycleValues))
    def test_a_node_is_published_with_the_status_its_life_cycle_names(
        self, small_plan: SmallPlan, life_cycle: LifeCycleValues
    ) -> None:
        small_plan.action.life_cycle_state = life_cycle
        small_plan.bridge.snapshot_plan()
        _, action, _ = published(small_plan.bridge)
        assert action[PlanNodeKey.STATUS] == LIFE_CYCLE_TO_STATUS[life_cycle]

    def test_every_node_is_published_under_the_node_above_it(
        self, small_plan: SmallPlan
    ) -> None:
        small_plan.bridge.snapshot_plan()
        top, action, motion = published(small_plan.bridge)
        assert top[PlanNodeKey.PARENT] is None
        assert action[PlanNodeKey.PARENT] == top[PlanNodeKey.ID]
        assert motion[PlanNodeKey.PARENT] == action[PlanNodeKey.ID]

    def test_the_arm_and_the_object_a_node_acts_on_are_published(
        self, small_plan: SmallPlan
    ) -> None:
        small_plan.bridge.snapshot_plan()
        _, action, motion = published(small_plan.bridge)
        assert action["arm"] == small_plan.action.arm
        assert motion["target"] == "milk.stl"

    def test_an_arm_of_the_world_is_published_by_its_name(
        self, small_plan: SmallPlan
    ) -> None:
        arm = ArmOfTheWorld()
        small_plan.action.arm = arm
        small_plan.bridge.snapshot_plan()
        _, action, _ = published(small_plan.bridge)
        assert action["arm"] == arm.name.name

    def test_several_arms_are_published_by_their_names(
        self, small_plan: SmallPlan
    ) -> None:
        arms = [ArmOfTheWorld(), OtherArmOfTheWorld()]
        small_plan.action.arm = arms
        small_plan.bridge.snapshot_plan()
        _, action, _ = published(small_plan.bridge)
        assert action["arm"] == ", ".join(arm.name.name for arm in arms)

    def test_the_signature_stays_while_only_statuses_change(
        self, small_plan: SmallPlan
    ) -> None:
        small_plan.bridge.snapshot_plan()
        signature = small_plan.bridge.get_plan()["sig"]
        small_plan.motion.life_cycle_state = LifeCycleValues.SUCCEEDED
        small_plan.bridge.snapshot_plan()
        assert small_plan.bridge.get_plan()["sig"] == signature


# %% the plan hook


class TestPlanHook:
    """
    A plan an executor performs is published as it starts and as it ended.
    """

    @staticmethod
    def publishing(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
        """
        Install the plan hook and keep what the bridge publishes.

        :return: The statecharts the bridge snapshots, which fills as they are published.
        """
        snapshots: List[Any] = []
        monkeypatch.setattr(BRIDGE, "_plan", None)
        monkeypatch.setattr(
            BRIDGE, "snapshot_plan", lambda: snapshots.append(BRIDGE._plan)
        )
        hooks.install_plan_hooks()
        return snapshots

    @staticmethod
    def executor_of(statechart: Any) -> KinematicallySimulatedPlanExecutor:
        """
        :return: An executor holding the statechart, built without a world.
        """
        executor = object.__new__(KinematicallySimulatedPlanExecutor)
        executor.statechart = statechart
        return executor

    def test_the_statechart_of_a_plan_executed_is_published_as_it_starts_and_ends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        statechart = object()
        monkeypatch.setattr(PlanExecutor, "execute", lambda self: None)
        snapshots = self.publishing(monkeypatch)

        self.executor_of(statechart).execute()

        assert snapshots == [statechart, statechart]

    def test_a_plan_that_fails_is_published_as_it_ended(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        statechart = object()

        def failing(self) -> None:
            raise RuntimeError("the motion failed")

        monkeypatch.setattr(PlanExecutor, "execute", failing)
        snapshots = self.publishing(monkeypatch)

        with pytest.raises(RuntimeError):
            self.executor_of(statechart).execute()

        assert snapshots == [statechart, statechart]


# %% mimics of a statechart


class _SNode:
    def __init__(self, index, name, parent=None):
        self.index = index
        self.name = name
        self.parent_node = parent


class _Transition:
    def __init__(self, kind):
        self.kind = types.SimpleNamespace(name=kind)


class _RxGraph:
    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges

    def edge_index_map(self):
        return {i: e for i, e in enumerate(self._edges)}

    def get_node_data(self, i):
        return self._nodes[i]


class NavigationUnderWay(NavigateAction):
    """
    A navigation running at the top of a statechart, read without the statechart.
    """

    @property
    def life_cycle_state(self) -> LifeCycleValues:
        return LifeCycleValues.RUNNING

    @property
    def parent_node(self) -> None:
        return None


def make_chart(life=(1, 1, 0), obs=(0.5, 0.5, 0.0)):
    chart = types.SimpleNamespace()
    goal = _SNode(0, "Goal")
    chart.nodes = [goal, _SNode(1, "MoveJoints", goal), _SNode(2, "JointGoalReached")]
    chart.rx_graph = _RxGraph(
        chart.nodes, [(0, 1, _Transition("START")), (1, 2, _Transition("END"))]
    )
    chart.life_cycle_state = types.SimpleNamespace(data=list(life))
    chart.observation_state = types.SimpleNamespace(data=list(obs))
    return chart


class TestChartSnapshot:
    def test_structure_and_states(self):
        bridge = Bridge()
        bridge.observe_chart(make_chart())
        chart = bridge.get_chart()
        assert [n["life"] for n in chart["nodes"]] == [
            "RUNNING",
            "RUNNING",
            "NOT_STARTED",
        ]
        assert [n["obs"] for n in chart["nodes"]] == ["UNKNOWN", "UNKNOWN", "FALSE"]
        assert chart["nodes"][1]["parent"] == "s0"
        assert chart["edges"] == [
            {"from": "s0", "to": "s1", "kind": "START"},
            {"from": "s1", "to": "s2", "kind": "END"},
        ]

    def test_the_title_is_the_action_running(self):
        action = object.__new__(NavigationUnderWay)
        action.index = 3
        action.name = NavigationUnderWay.__name__
        chart = make_chart(life=(1, 1, 0, 1), obs=(0.5, 0.5, 0.0, 0.5))
        chart.nodes.append(action)
        bridge = Bridge()
        bridge.observe_chart(chart)
        assert bridge.get_chart()["title"] == NavigationUnderWay.__name__

    def test_no_title_while_no_action_runs(self):
        bridge = Bridge()
        bridge.observe_chart(make_chart())
        assert bridge.get_chart()["title"] == ""

    def test_lifecycle_update_keeps_signature(self):
        bridge = Bridge()
        chart = make_chart()
        bridge.observe_chart(chart)
        sig = bridge.get_chart()["sig"]
        chart.life_cycle_state.data = [3, 3, 3]
        bridge.observe_chart(chart)
        assert bridge.get_chart()["sig"] == sig
        assert [n["life"] for n in bridge.get_chart()["nodes"]] == ["DONE"] * 3

    def test_new_chart_replaces_structure(self):
        bridge = Bridge()
        bridge.observe_chart(make_chart())
        sig = bridge.get_chart()["sig"]
        other = types.SimpleNamespace(
            nodes=[_SNode(0, "OtherGoal")],
            rx_graph=_RxGraph([_SNode(0, "OtherGoal")], []),
            life_cycle_state=types.SimpleNamespace(data=[1]),
            observation_state=types.SimpleNamespace(data=[1.0]),
        )
        bridge.observe_chart(other)
        chart = bridge.get_chart()
        assert chart["sig"] != sig
        assert len(chart["nodes"]) == 1
        assert chart["nodes"][0]["obs"] == "TRUE"

    def test_trinary_observation_thresholds(self):
        bridge = Bridge()
        bridge.observe_chart(make_chart(obs=(0.0, 0.5, 1.0)))
        assert [n["obs"] for n in bridge.get_chart()["nodes"]] == [
            "FALSE",
            "UNKNOWN",
            "TRUE",
        ]
