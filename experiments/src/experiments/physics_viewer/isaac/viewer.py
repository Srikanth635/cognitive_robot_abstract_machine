"""
Stepping a scene under Isaac Sim's PhysX at a chosen speed, in a window or without one,
recording it from several cameras if asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import omni.usd
from isaacsim.core.rendering_manager import ViewportManager
from pxr import Sdf
from semantic_digital_twin.adapters.isaac.video_recording import IsaacVideoRecorder
from semantic_digital_twin.adapters.multi_sim import MultiSim
from semantic_digital_twin.adapters.mujoco_video_recording import VideoResolution
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World

from experiments.physics_viewer.isaac.cameras import SceneCamera, SceneLighting
from experiments.physics_viewer.isaac.paced_simulation import RealTimeIsaacSim
from experiments.physics_viewer.isaac.player import (
    PlayerCamera,
    RecordingFile,
    RecordingPlayer,
)
from experiments.physics_viewer.overlay import ActionOverlay
from experiments.physics_viewer.viewer import SteppedSceneViewer

# %% the viewer


@dataclass
class IsaacPhysicsViewer(SteppedSceneViewer):
    """
    Puts a scene under Isaac Sim's PhysX, lit, with every :class:`SceneCamera` put up;
    with a window it shows one of them, and with a directory to record into it writes a
    video of each and a page that plays them.
    """

    frame: timedelta = timedelta(milliseconds=20)
    """
    How much simulated time one frame shows: a whole number of Isaac Sim's physics
    steps, which a simulation is only advanced by.
    """

    camera: SceneCamera = SceneCamera.BEHIND_RIGHT
    """
    The camera the window shows first; its camera menu switches between them.
    """

    record_directory: Path | None = None
    """
    Where a video of each camera and the page that plays them are written; ``None``
    records nothing.
    """

    recording_frames_per_second: int = 25
    """
    How many frames each second of simulated time gets in the videos.
    """

    recording_resolution: VideoResolution = field(
        default_factory=lambda: VideoResolution(width=1280, height=720)
    )
    """
    The size of every frame of the videos.
    """

    lighting: SceneLighting = field(default_factory=SceneLighting)
    """
    The lights the scene is watched under.
    """

    _cameras: dict[SceneCamera, Sdf.Path] = field(init=False, default_factory=dict)
    """
    The prim each camera is written as, once the simulation has started.
    """

    _recorder: IsaacVideoRecorder | None = field(init=False, default=None)
    """
    What records the cameras, while recording.
    """

    def _simulation_of(self, world: World) -> RealTimeIsaacSim:
        return RealTimeIsaacSim(world=world, speed=self.speed, headless=self.headless)

    def _start_showing(self, simulation: MultiSim) -> ActionOverlay | None:
        """
        :meth:`SteppedSceneViewer._start_showing`; nothing writes into Isaac Sim's
        window.
        """
        stage = omni.usd.get_context().get_stage()
        self.lighting.add_to(stage)
        [robot] = simulation.world.get_semantic_annotations_by_type(AbstractRobot)
        self._cameras = {camera: camera.mount(stage, robot) for camera in SceneCamera}
        if not self.headless:
            ViewportManager.set_camera(str(self._cameras[self.camera]))
        if self.record_directory is not None:
            self._recorder = IsaacVideoRecorder(
                simulation=simulation,
                cameras=[str(path) for path in self._cameras.values()],
                frames_per_second=self.recording_frames_per_second,
                resolution=self.recording_resolution,
            )
            self._recorder.start()
        return None

    def _finish_showing(self) -> None:
        if self._recorder is None:
            return
        videos = self._recorder.stop()
        self._recorder = None
        for camera, path in self._cameras.items():
            videos[str(path)].write(self.record_directory / self._video_name(camera))
        RecordingPlayer(
            title=self._title(),
            cameras=[
                PlayerCamera(label=camera.label, video=self._video_name(camera))
                for camera in self._cameras
            ],
        ).write(self.record_directory)

    @staticmethod
    def _video_name(camera: SceneCamera) -> str:
        """
        :param camera: A camera.
        :return: The name of the file its video is written to.
        """
        return f"{camera.name.lower()}{RecordingFile.VIDEO_SUFFIX}"

    def _title(self) -> str:
        """
        :return: What the page that plays the videos calls the run.
        """
        if self.action is None:
            return self.scene.name.lower()
        return f"{self.action.name.lower()} in {self.scene.name.lower()}"
