import pytest

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    EndMotion,
    CancelMotion,
)
from giskardpy.motion_statechart.monitors.monitors import TrueMonitor
from giskardpy.motion_statechart.monitors.payload_monitors import Print
from giskardpy.motion_statechart.motion_statechart_graph import MotionStatechart
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World


def test_motion_statechart():
    msg = MotionStatechart(World())

    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    node2 = TrueMonitor(name=PrefixedName("muh2"), motion_statechart=msg)
    node3 = TrueMonitor(name=PrefixedName("muh3"), motion_statechart=msg)
    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)

    node1.start_condition = cas.trinary_logic_or(node3, node2)
    end.start_condition = node1
    assert len(msg.nodes) == 4
    assert len(msg.edges) == 3

    msg.compile()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node3] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node3] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node3] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node3] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.RUNNING
    assert not msg.is_end_motion()
    msg.tick()
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node3] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryTrue
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.RUNNING
    assert msg.is_end_motion()


def test_duplicate_name():
    msg = MotionStatechart(World())

    with pytest.raises(ValueError):
        cas.Symbol(name=PrefixedName("muh"))
        MotionStatechartNode(name=PrefixedName("muh"), motion_statechart=msg)
        MotionStatechartNode(name=PrefixedName("muh"), motion_statechart=msg)


def test_print():
    msg = MotionStatechart(World())
    print_node1 = Print(name=PrefixedName("cow"), message="muh", motion_statechart=msg)
    print_node2 = Print(name=PrefixedName("cow2"), message="muh", motion_statechart=msg)

    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    node1.start_condition = print_node1
    print_node2.start_condition = node1
    end = EndMotion(name=PrefixedName("done"), motion_statechart=msg)
    end.start_condition = print_node2
    assert len(msg.nodes) == 4
    assert len(msg.edges) == 3

    msg.compile()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[print_node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[print_node2] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[print_node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[print_node2] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node1] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[print_node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[print_node2] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryUnknown
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[print_node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[print_node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.NOT_STARTED
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryUnknown

    assert msg.life_cycle_state[print_node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[print_node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.RUNNING
    assert not msg.is_end_motion()

    msg.tick()
    assert msg.observation_state[print_node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[node1] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[print_node2] == msg.observation_state.TrinaryTrue
    assert msg.observation_state[end] == msg.observation_state.TrinaryTrue

    assert msg.life_cycle_state[print_node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[print_node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[end] == msg.life_cycle_state.RUNNING
    assert msg.is_end_motion()


def test_cancel_motion():
    msg = MotionStatechart(World())
    node1 = TrueMonitor(name=PrefixedName("muh"), motion_statechart=msg)
    cancel = CancelMotion(
        name=PrefixedName("done"), motion_statechart=msg, exception=Exception("test")
    )
    cancel.start_condition = node1

    msg.compile()
    msg.tick()  # first tick, cancel motion node1 turns true
    msg.tick()  # second tick, cancel goes into running
    with pytest.raises(Exception):
        msg.tick()  # third tick, cancel goes true and triggers
