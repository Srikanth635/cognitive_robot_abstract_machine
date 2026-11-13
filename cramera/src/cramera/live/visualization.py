"""
Publish native CRAM world and plan state to the browser viewer.

World callbacks publish geometry and poses. The state history of the statechart a plan
is performed as publishes execution progress.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from contextlib import ExitStack
from functools import partial

from typing_extensions import Any, Optional, TYPE_CHECKING

from cramph.state_history import StateHistoryObserver
from cramph.visualization import WorldVisualization
from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
    StateChangeCallback,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import MeshFileStorage

from cramera.live.bridge import Bridge
from cramera.live.http import DEFAULT_PORT, serve
from cramera.live.recording import Recording
from cramera.live.recording_bundle import finalize_recording
from cramera.live.ros_markers import RosMarkerListener
from cramera.logging_setup import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from cramph.executor import Executor
    from cramph.state_history import StateHistory
    from cramph.statechart import Statechart

# %% world synchronization


@dataclass(eq=False)
class WorldStateSync(StateChangeCallback):
    """
    Publishes a world snapshot to the bridge whenever the world's state changes.
    """

    bridge: Bridge = field(kw_only=True)
    """
    The bridge the snapshots are published to.
    """

    def on_state_change(self, **kwargs: Any) -> None:
        """
        Publish and record the world's updated state.

        :param kwargs: Metadata provided by the native callback dispatcher.
        """
        self.bridge.snapshot()
        if self.bridge.recording is not None:
            self.bridge.recording.append(
                self.bridge.state,
                self.bridge.running_step(),
                self.bridge.executing_statechart(),
            )


@dataclass(eq=False)
class WorldModelSync(ModelChangeCallback):
    """
    Refreshes the bridge's body and geometry catalogs when the world model changes.
    """

    bridge: Bridge = field(kw_only=True)
    """
    The bridge whose catalogs are refreshed.
    """

    def on_model_change(self, **kwargs: Any) -> None:
        """
        Refresh the published model after a structural change.

        :param kwargs: Metadata provided by the native callback dispatcher.
        """
        self.bridge.observe_model_change()


# %% plan synchronization


@dataclass
class BridgeStatechartObserver(StateHistoryObserver):
    """
    Publishes the plan's statechart every time its state history records a change.
    """

    bridge: Bridge = field(kw_only=True)
    """
    The bridge the plan's execution is published to.
    """

    statechart: Statechart = field(kw_only=True)
    """
    The statechart of the plan being performed.
    """

    def on_state_change(self, *, history: StateHistory) -> None:
        self.bridge.observe_chart(self.statechart)
        self.bridge.snapshot_plan()

    def follow(self) -> None:
        """
        Publish the statechart's tree and start publishing its every change.
        """
        self.bridge.begin_plan(statechart=self.statechart)
        self.bridge.observe_chart(self.statechart)
        self.statechart.state_history.add_observer(observer=self)

    def stop(self) -> None:
        """
        Publish the statechart as it ended, keep it with the recording, and stop
        publishing its changes.
        """
        self.statechart.state_history.remove_observer(observer=self)
        self.bridge.observe_chart(self.statechart)
        self.bridge.snapshot_plan()
        if self.bridge.recording is not None:
            self.bridge.recording.update_statechart(self.bridge.executing_statechart())


def _finalize_recording_at_exit(
    bridge: Bridge, recording: Optional[Recording] = None
) -> None:
    """
    Best-effort safety net: write the current recording to disk if the process is about
    to exit without the viewer ever sending ``/recording/stop``.

    A demo run directly (rather than through ``cramera-live``, which stays up for
    inspection after the demo finishes) has no long-lived process left for the browser
    to ask, so the recording would otherwise be lost the moment the script's main body
    returns.

    :param bridge: The bridge whose geometry belongs to the recording.
    :param recording: The session's capture, or the bridge's current capture.
    """
    if recording is None:
        recording = bridge.recording
    if recording is None:
        return
    try:
        finalize_recording(bridge, recording)
    except Exception:
        # boundary guard: the interpreter is tearing down and the world may be in a
        # partial state; losing the recording is better than a traceback on every exit
        logger.exception("could not finalize the live recording at exit")


# %% the backend


@dataclass
class LiveVisualization(WorldVisualization):
    """
    Serves a world to the cramera browser viewer, and the plans the executors it is
    added to perform in it.

    Unlike :meth:`WorldVisualization.after_run`, a run ending leaves the world served,
    so what it did can be inspected; :meth:`stop` ends serving.
    """

    port: int = DEFAULT_PORT
    """
    Port of the bridge's HTTP endpoints.
    """

    bridge: Bridge = field(default_factory=Bridge)
    """
    The bridge translating between the world and the viewer.
    """

    state_sync: Optional[WorldStateSync] = field(init=False, default=None)
    """
    The callback publishing state changes, while started.
    """

    model_sync: Optional[WorldModelSync] = field(init=False, default=None)
    """
    The callback refreshing the published model after structural changes, while started.
    """

    _statechart_observer: Optional[BridgeStatechartObserver] = field(
        init=False, default=None, repr=False
    )
    """
    Publishes the statechart of the plan being performed, while one is.
    """

    @property
    def is_rendering(self) -> bool:
        return self.state_sync is not None

    def _start(self, *, world: World, cleanup: ExitStack) -> None:
        if self.bridge.recording is not None:
            finalize_recording(self.bridge, self.bridge.recording)
        query_attachment = self.bridge.attach(world)
        cleanup.callback(self.bridge.release_world_queries, query_attachment)
        recording = Recording()
        self.bridge.recording = recording
        cleanup.callback(self._finalize_recording, recording)
        recording.start()
        MeshFileStorage()
        exit_callback = partial(_finalize_recording_at_exit, self.bridge, recording)
        atexit.register(exit_callback)
        cleanup.callback(atexit.unregister, exit_callback)
        self.bridge.snapshot()
        self.state_sync = WorldStateSync(_world=world, bridge=self.bridge)
        cleanup.callback(self._stop_state_sync)
        self.model_sync = WorldModelSync(_world=world, bridge=self.bridge)
        cleanup.callback(self._stop_model_sync)
        marker_listener = RosMarkerListener.start_if_available(self.bridge)
        self.bridge.marker_listener = marker_listener
        if marker_listener is not None:
            cleanup.callback(self._stop_marker_listener, marker_listener)
        if self.bridge.live_server is None:
            self.bridge.live_server = serve(self.bridge, self.port)
            cleanup.callback(self._stop_server)
        cleanup.callback(self._stop_following_statechart)

    def after_compile(self, *, executor: Executor) -> None:
        """
        Start serving the executor's world, as :meth:`WorldVisualization.after_compile`
        does, and publish the plan the executor compiled as it is performed.
        """
        super().after_compile(executor=executor)
        self._stop_following_statechart()
        self._statechart_observer = BridgeStatechartObserver(
            bridge=self.bridge, statechart=executor.statechart
        )
        self._statechart_observer.follow()

    def after_run(self, *, executor: Executor) -> None:
        """
        Publish the plan as it ended and keep serving the world, see the class.
        """
        self._stop_following_statechart()

    def _stop_following_statechart(self) -> None:
        """
        Stop publishing the statechart of the plan performed last, if one was.
        """
        if self._statechart_observer is None:
            return
        self._statechart_observer.stop()
        self._statechart_observer = None

    def _stop_state_sync(self) -> None:
        """
        Stop publishing the world's state changes.
        """
        self.state_sync.stop()
        self.state_sync = None

    def _stop_model_sync(self) -> None:
        """
        Stop refreshing the published model after structural changes.
        """
        self.model_sync.stop()
        self.model_sync = None

    def _stop_marker_listener(self, marker_listener: RosMarkerListener) -> None:
        """
        Stop the ROS marker subscription feeding the debug overlay.

        :param marker_listener: The subscription to stop.
        """
        marker_listener.stop()
        self.bridge.marker_listener = None

    def _stop_server(self) -> None:
        """
        Stop serving the bridge's HTTP endpoints.
        """
        self.bridge.live_server.shutdown()
        self.bridge.live_server.server_close()
        self.bridge.live_server = None

    def _finalize_recording(self, recording: Recording) -> None:
        """
        Write the session's recording to disk and detach it from the bridge.

        :param recording: The session's capture.
        """
        finalize_recording(self.bridge, recording)
        self.bridge.recording = None
