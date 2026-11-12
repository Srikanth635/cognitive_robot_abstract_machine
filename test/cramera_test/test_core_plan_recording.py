"""
Native plan status publication and plan inspection after recording.
"""

from __future__ import annotations

from cramph.composites import Sequence
from cramph.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from cramph.statechart import Statechart

from cramera.live.bridge import Bridge
from cramera.live.chart_structure import ObservationName
from cramera.live.recording_bundle import write_recording_bundle
from cramera.live.recording_storage import trim_recording_bundle
from cramera.live.frame_range import FrameRange
from cramera.live.recording import RecordedFrame, Recording
from cramera.knowledge.recorded_statecharts import RecordedStatecharts, STATECHART_FILE
from cramera.generated_json import GeneratedJson
from cramera import paths

from .dataset.plan_statecharts import (
    MotionStep,
    PlanExecution,
    plan_execution,
    plan_statechart,
)
from .test_live_bridge import nodes_by_kind
from .test_live_bundle import attached_bridge
from .test_recording_bundle import frame_with_milk
from .test_live_recording import statechart, snapshot


# %% native status publication


class TestNativePlanStatus:
    """
    Native lifecycle values retain their meaning in the viewer.
    """

    def test_an_unstarted_parent_keeps_its_state_with_a_running_child(
        self, plan_statechart: Statechart
    ) -> None:
        """
        A child's execution does not replace its parent's current lifecycle.

        :param plan_statechart: An empty statechart to build the plan in.
        """
        bridge = Bridge()
        child = MotionStep(name="moving")
        root = Sequence(nodes=[child])
        plan_statechart.add_node(root)
        plan_statechart.life_cycle_state[child] = LifeCycleValues.RUNNING
        bridge.begin_plan(statechart=plan_statechart)
        assert (
            nodes_by_kind(bridge)[Sequence.__name__]["status"]
            == root.life_cycle_state.name
        )

    def test_a_paused_motion_publishes_a_paused_status(
        self, plan_statechart: Statechart
    ) -> None:
        """
        A paused native motion is serialized with its native lifecycle name.

        :param plan_statechart: An empty statechart to build the plan in.
        """
        bridge = Bridge()
        motion = MotionStep(name="moving")
        plan_statechart.add_node(motion)
        plan_statechart.life_cycle_state[motion] = LifeCycleValues.PAUSED
        bridge.begin_plan(statechart=plan_statechart)
        assert (
            nodes_by_kind(bridge)[MotionStep.__name__]["status"]
            == motion.life_cycle_state.name
        )


# %% plan persistence


class TestMotionHistoryRecording:
    """
    Plan completion retains a final chart observation on the last recorded pose.
    """

    def test_last_world_frame_keeps_the_final_observation(
        self, plan_execution: PlanExecution
    ) -> None:
        """
        The final chart-only observation preserves the last world pose.
        """
        bridge = plan_execution.bridge
        bridge.recording = Recording()
        bridge.recording.start()
        statechart_of_plan = plan_execution.statechart
        last_node = statechart_of_plan.nodes[-1]
        statechart_of_plan.observation_state[last_node] = ObservationStateValues.FALSE
        plan_execution.record(life_cycle=LifeCycleValues.RUNNING)
        bridge.recording.append(
            snapshot(frames={"joint": 0.5}), statechart=bridge.executing_statechart()
        )
        first_world_frame = bridge.recording.frames_in(FrameRange(0, 0))[0]
        statechart_of_plan.observation_state[last_node] = ObservationStateValues.TRUE

        plan_execution.record(life_cycle=LifeCycleValues.SUCCEEDED)
        plan_execution.observer.stop()

        [recorded] = bridge.recording.stop()
        assert recorded.statechart == bridge.executing_statechart()
        assert recorded.frames == first_world_frame.frames
        assert recorded.statechart.nodes[-1].observation == ObservationName.TRUE
        assert (
            first_world_frame.statechart.nodes[-1].observation == ObservationName.FALSE
        )

    def test_a_finalized_recording_does_not_change_on_later_history_updates(
        self, plan_execution: PlanExecution
    ) -> None:
        """
        History changes and plan completion leave saved captures unchanged.
        """
        bridge = plan_execution.bridge
        bridge.recording = Recording()
        bridge.recording.start()
        plan_execution.record(life_cycle=LifeCycleValues.RUNNING)
        bridge.recording.append(snapshot(), statechart=bridge.executing_statechart())
        original = bridge.recording.stop()
        statechart_of_plan = plan_execution.statechart
        statechart_of_plan.observation_state[statechart_of_plan.nodes[-1]] = (
            ObservationStateValues.TRUE
        )

        plan_execution.record(life_cycle=LifeCycleValues.SUCCEEDED)
        plan_execution.observer.stop()

        assert bridge.recording.stop() == original


class TestRecordedPlan:
    """
    A replay keeps the completed run's plan available for inspection.
    """

    def test_recording_contains_the_published_plan(
        self, tmp_path, plan_statechart: Statechart
    ) -> None:
        """
        The saved hierarchy retains native plan labels and completion states.

        :param tmp_path: Temporary directory for the exported recording.
        :param plan_statechart: An empty statechart to build the plan in.
        """
        bridge = attached_bridge()
        child = MotionStep(name="moving")
        root = Sequence(nodes=[child])
        plan_statechart.add_node(root)
        plan_statechart.life_cycle_state[root] = LifeCycleValues.SUCCEEDED
        plan_statechart.life_cycle_state[child] = LifeCycleValues.SUCCEEDED
        bridge.begin_plan(statechart=plan_statechart)

        scene = write_recording_bundle(
            bridge, [frame_with_milk()], 20.0, tmp_path / "recording", "finished_run"
        )

        [recorded_root] = scene["planTrees"]
        assert recorded_root["label"] == type(root).__name__
        assert recorded_root["status"] == LifeCycleValues.SUCCEEDED.name
        [recorded_child] = recorded_root["children"]
        assert recorded_child["label"] == type(child).__name__
        assert recorded_child["children"] == []

    def test_trim_keeps_the_statecharts_of_the_selected_frames(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        bridge = attached_bridge()
        frames = [
            RecordedFrame(
                frames={}, base=None, objects={}, statechart=statechart(status)
            )
            for status in (
                LifeCycleValues.NOT_STARTED.name,
                LifeCycleValues.RUNNING.name,
                LifeCycleValues.SUCCEEDED.name,
            )
        ]
        bundle = paths.local_scenes_directory() / paths.RECORDING_SCENE_NAME
        write_recording_bundle(bridge, frames, 20.0, bundle, paths.RECORDING_SCENE_NAME)
        original = RecordedStatecharts.of_payload(
            GeneratedJson(bundle / STATECHART_FILE).read()
        )

        trim_recording_bundle(FrameRange(first=1, last=2))

        trimmed = RecordedStatecharts.of_payload(
            GeneratedJson(bundle / STATECHART_FILE).read()
        )
        assert trimmed.moment_of_frame == original.moment_of_frame[1:3]
