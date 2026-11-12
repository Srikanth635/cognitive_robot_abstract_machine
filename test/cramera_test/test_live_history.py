"""
Native state history publication and recording boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from typing_extensions import TYPE_CHECKING

from cramph.data_types import LifeCycleValues
from cramph.statechart import Statechart

from cramera.live import visualization as visualization_module
from cramera.live.recording import Recording
from cramera.live.visualization import (
    BridgeStatechartObserver,
    LiveVisualization,
    WorldStateSync,
)

from .dataset.plan_statecharts import (
    CompiledExecutor,
    FollowedStatechart,
    MotionStep,
    PlanExecution,
    plan_execution,
    plan_statechart,
)
from .test_live_visualization import ServerRecorder, world

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


# %% history publication


class TestStatechartHistoryPublication:
    """
    State history subscriptions publish statechart changes for the plan's lifetime.
    """

    @pytest.mark.parametrize(
        "outcome",
        [
            LifeCycleValues.SUCCEEDED,
            LifeCycleValues.FAILED,
            LifeCycleValues.INTERRUPTED,
        ],
    )
    def test_native_history_owns_pause_completion_and_reset(
        self, plan_execution: PlanExecution, outcome: LifeCycleValues
    ) -> None:
        """
        Publish the native life cycle throughout a complete attempt.

        :param plan_execution: The plan, its statechart and the observer to exercise.
        :param outcome: The terminal outcome recorded before resetting the plan.
        """
        for state in (
            LifeCycleValues.RUNNING,
            LifeCycleValues.PAUSED,
            outcome,
            LifeCycleValues.NOT_STARTED,
        ):
            plan_execution.record(life_cycle=state)
            assert plan_execution.motion.life_cycle_state is state
            assert [node.status for node in plan_execution.bridge.plan_state.nodes] == [
                plan_execution.root.life_cycle_state,
                plan_execution.action.life_cycle_state,
                state,
            ]

    def test_following_publishes_the_statechart(
        self, plan_execution: PlanExecution
    ) -> None:
        """
        The viewer sees the statechart before the first tick.
        """
        assert [node.name for node in plan_execution.bridge.chart_state.nodes] == [
            node.name for node in plan_execution.statechart.nodes
        ]

    def test_native_history_changes_publish_the_chart_and_plan(
        self, plan_execution: PlanExecution
    ) -> None:
        """
        A recorded native state change refreshes both execution views.

        :param plan_execution: The plan, its statechart and the observer.
        """
        plan_execution.record(life_cycle=LifeCycleValues.RUNNING)

        assert plan_execution.bridge.chart_state.nodes[0].life_cycle == (
            LifeCycleValues.RUNNING.name
        )
        assert plan_execution.bridge.plan_state.nodes[0].status == (
            LifeCycleValues.RUNNING
        )

    def test_following_twice_subscribes_to_the_history_once(
        self, plan_execution: PlanExecution
    ) -> None:
        """
        A followed statechart does not accumulate one subscription per follow.
        """
        plan_execution.observer.follow()

        assert plan_execution.statechart.state_history.observers == [
            plan_execution.observer
        ]

    def test_native_reset_clears_motion_and_parent_progress(
        self, plan_execution: PlanExecution
    ) -> None:
        """
        A reset statechart restores every plan entry to its unstarted state.

        :param plan_execution: The plan, its statechart and the observer.
        """
        plan_execution.record(life_cycle=LifeCycleValues.RUNNING)
        plan_execution.record(life_cycle=LifeCycleValues.SUCCEEDED)

        plan_execution.record(life_cycle=LifeCycleValues.NOT_STARTED)

        assert [node.status for node in plan_execution.bridge.plan_state.nodes] == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
        ]

    def test_stopping_removes_history_subscriptions(
        self, plan_execution: PlanExecution
    ) -> None:
        """
        Completed plans no longer alter the published state.
        """
        plan_execution.record(life_cycle=LifeCycleValues.RUNNING)
        published = plan_execution.bridge.chart_state
        history = plan_execution.statechart.state_history
        assert history.observers == [plan_execution.observer]

        plan_execution.observer.stop()
        plan_execution.observer.stop()
        plan_execution.record(life_cycle=LifeCycleValues.SUCCEEDED)

        assert history.observers == []
        assert plan_execution.bridge.chart_state == published

    def test_visualization_stop_removes_history_subscriptions(
        self,
        world: World,
        plan_execution: PlanExecution,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Stopping a viewer also detaches histories of unfinished plans.
        """
        monkeypatch.setattr(
            visualization_module, "serve", lambda bridge, port: ServerRecorder()
        )
        monkeypatch.setattr(
            visualization_module.RosMarkerListener,
            "start_if_available",
            lambda bridge: None,
        )
        monkeypatch.setattr(
            visualization_module, "finalize_recording", lambda bridge, recording: None
        )
        plan_execution.observer.stop()
        statechart = plan_execution.statechart
        visualization = LiveVisualization(bridge=plan_execution.bridge).start(world)
        visualization.after_compile(
            CompiledExecutor(context=statechart.context, statechart=statechart)
        )
        assert len(statechart.state_history.observers) == 1

        visualization.stop()
        visualization.stop()

        assert statechart.state_history.observers == []


# %% recording alignment


class TestHistoryRecordingAlignment:
    """
    Recorded poses retain the chart state of their own control cycle.
    """

    def test_next_history_change_does_not_overwrite_previous_world_frame(
        self, world: World, plan_execution: PlanExecution
    ) -> None:
        """
        History is published before the corresponding world frame is appended.
        """
        bridge = plan_execution.bridge
        bridge.attach(world)
        bridge.recording = Recording()
        bridge.recording.start()
        world_sync = WorldStateSync(_world=world, bridge=bridge)
        plan_execution.record(life_cycle=LifeCycleValues.RUNNING)
        world_sync.on_state_change()

        plan_execution.record(life_cycle=LifeCycleValues.SUCCEEDED)
        world_sync.on_state_change()

        frames = bridge.recording.stop()
        assert [frame.statechart.nodes[0].life_cycle for frame in frames] == [
            LifeCycleValues.RUNNING.name,
            LifeCycleValues.SUCCEEDED.name,
        ]

    def test_plan_end_flushes_a_chart_change_without_another_world_update(
        self, world: World, plan_execution: PlanExecution
    ) -> None:
        """
        A final chart-only controller update completes the last captured pose.
        """
        bridge = plan_execution.bridge
        bridge.attach(world)
        bridge.recording = Recording()
        bridge.recording.start()
        world_sync = WorldStateSync(_world=world, bridge=bridge)
        plan_execution.record(life_cycle=LifeCycleValues.RUNNING)
        world_sync.on_state_change()

        plan_execution.record(life_cycle=LifeCycleValues.SUCCEEDED)
        plan_execution.observer.stop()

        frames = bridge.recording.stop()
        assert len(frames) == 1
        assert (
            frames[0].statechart.nodes[0].life_cycle == LifeCycleValues.SUCCEEDED.name
        )

    def test_a_single_motion_plan_flushes_its_final_chart_before_unsubscribing(
        self, world: World, plan_execution: PlanExecution
    ) -> None:
        """
        A plan that is a single motion retains its terminal chart-only update.
        """
        plan_execution.observer.stop()
        bridge = plan_execution.bridge
        statechart = Statechart(context=plan_execution.statechart.context)
        statechart.add_node(MotionStep(name="moving"))
        single_motion = FollowedStatechart(
            statechart=statechart,
            bridge=bridge,
            observer=BridgeStatechartObserver(bridge=bridge, statechart=statechart),
        )
        single_motion.observer.follow()
        bridge.attach(world)
        bridge.recording = Recording()
        bridge.recording.start()
        world_sync = WorldStateSync(_world=world, bridge=bridge)
        single_motion.record(life_cycle=LifeCycleValues.RUNNING)
        world_sync.on_state_change()

        single_motion.record(life_cycle=LifeCycleValues.SUCCEEDED)
        single_motion.observer.stop()

        frames = bridge.recording.stop()
        assert len(frames) == 1
        assert (
            frames[0].statechart.nodes[0].life_cycle == LifeCycleValues.SUCCEEDED.name
        )
        assert not any(
            observer is single_motion.observer
            for observer in statechart.state_history.observers
        )
