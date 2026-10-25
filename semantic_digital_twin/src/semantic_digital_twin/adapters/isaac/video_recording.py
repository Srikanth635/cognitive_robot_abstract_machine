"""
Recording videos of a world simulated under Isaac Sim, from cameras on its stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from semantic_digital_twin.adapters.isaac.simulation import IsaacSim
from semantic_digital_twin.adapters.mujoco_video_recording import (
    RecordedVideo,
    VideoResolution,
)
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.exceptions import (
    InvalidVideoRecordingRateError,
    VideoRecordingAlreadyStartedError,
    VideoRecordingNotStartedError,
)

# %% hearing of the world's changes


@dataclass(eq=False)
class _FrameCaptureCallback(StateChangeCallback):
    """
    Tells an :class:`IsaacVideoRecorder` every time its simulation's world changes.
    """

    recorder: IsaacVideoRecorder = field(kw_only=True)
    """
    The recorder to tell.
    """

    def on_state_change(self, **kwargs: object) -> None:
        self.recorder._on_world_state_change()


# %% the recorder


@dataclass
class IsaacVideoRecorder:
    """
    Records a video from each of some cameras on the stage of a running
    :class:`IsaacSim`, with a frame for every ``1 / frames_per_second`` of simulated time,
    so the videos play at the simulation's own speed.

    Frames are taken when the world changes: a simulation stepped by a controller's ticks
    gives every frame its own moment, while one stepped over several frames' time at once
    repeats the picture its world was left in.
    """

    simulation: IsaacSim
    """
    The simulation to record, started stepped.
    """

    cameras: list[str]
    """
    The paths of the camera prims to record from.
    """

    frames_per_second: int = 30
    """
    How many frames each second of simulated time gets, and the rate the videos play at.
    """

    resolution: VideoResolution = field(
        default_factory=lambda: VideoResolution(width=640, height=480)
    )
    """
    The size of every frame.
    """

    _frame_capture_callback: _FrameCaptureCallback | None = field(
        init=False, default=None, repr=False
    )
    """
    What tells the recorder of the world's changes, while recording.
    """

    _frames: dict[str, list[np.ndarray]] = field(
        init=False, default_factory=dict, repr=False
    )
    """
    The frames taken since :meth:`start`, by camera.
    """

    _frames_taken: int = field(init=False, default=0, repr=False)
    """
    How many frames of simulated time have been taken, counted from the simulation's
    start.
    """

    def __post_init__(self) -> None:
        if self.frames_per_second <= 0:
            raise InvalidVideoRecordingRateError(
                field_name="frames_per_second", value=self.frames_per_second
            )

    def start(self) -> None:
        """
        Take a frame from every camera now, and from then on one for every ``1 /
        frames_per_second`` of simulated time.

        :raises VideoRecordingAlreadyStartedError: If the recorder is recording.
        """
        if self._frame_capture_callback is not None:
            raise VideoRecordingAlreadyStartedError(world=self.simulation.world)
        self._frames = {camera: [] for camera in self.cameras}
        self._frames_taken = self._frames_due() - 1
        self._frame_capture_callback = _FrameCaptureCallback(
            _world=self.simulation.world, recorder=self
        )
        self._on_world_state_change()

    def stop(self) -> dict[str, RecordedVideo]:
        """
        Stop taking frames.

        :return: The video of each camera, by the path of its prim.
        :raises VideoRecordingNotStartedError: If the recorder is not recording.
        """
        if self._frame_capture_callback is None:
            raise VideoRecordingNotStartedError(world=self.simulation.world)
        self._frame_capture_callback.stop()
        self._frame_capture_callback = None
        videos = {
            camera: RecordedVideo(
                frames=frames, frames_per_second=self.frames_per_second
            )
            for camera, frames in self._frames.items()
        }
        self._frames = {}
        return videos

    @property
    def captured_frame_count(self) -> int:
        """
        :return: How many frames each camera has had taken since :meth:`start`.
        """
        return min((len(frames) for frames in self._frames.values()), default=0)

    def _on_world_state_change(self) -> None:
        """
        Take every frame that has become due, from every camera.
        """
        simulator = self.simulation.simulator
        while self._frames_taken < self._frames_due():
            for camera, frames in self._frames.items():
                frames.append(
                    simulator.capture_rgb(
                        camera,
                        height=self.resolution.height,
                        width=self.resolution.width,
                    )
                )
            self._frames_taken += 1

    def _frames_due(self) -> int:
        """
        :return: How many frames the simulated time so far is worth, counting the one at
            its start.
        """
        return (
            math.floor(
                self.simulation.simulator.current_simulation_time
                * self.frames_per_second
                + 1e-9
            )
            + 1
        )
