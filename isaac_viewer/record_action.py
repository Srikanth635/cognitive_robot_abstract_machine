"""
``python record_action.py SCENE ACTION DIRECTORY``: record the robot of a physics viewer
scene performing an action under Isaac Sim's PhysX, headless, from every camera of
:class:`scene_cameras.SceneCamera`.

Each camera's video is written to DIRECTORY as an MP4 file, next to ``player.html``, a page
that plays them, switching between the cameras and speeding playback up. Run it through
``run.sh`` next to this file, with ``record_action.py`` as the script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

from physics_simulators.isaac_application import IsaacApplication
from typing_extensions import Any, Sequence

# %% Isaac Sim's application, which runs before any of its modules is imported

IsaacApplication(headless=True).start()

import imageio.v2 as imageio
import numpy as np
import omni.replicator.core as replicator
from experiments.physics_viewer.actions import PhysicsAction
from experiments.physics_viewer.scenes import PhysicsScene
from imageio.core.format import Format
from isaacsim.core.rendering_manager import RenderingManager
from omni.replicator.core.scripts.annotators import Annotator
from pxr import Sdf
from scene_cameras import SceneCamera
from watched_run import WatchedRun

from semantic_digital_twin.adapters.isaac.simulation import IsaacSim
from semantic_digital_twin.world import World

# %% the command line


def parse_arguments(arguments: Sequence[str]) -> IsaacRecordingRun:
    """
    :param arguments: The command line after the program.
    :return: The run the command line asks for.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene", help="a physics viewer scene, such as pr2_at_the_counter"
    )
    parser.add_argument(
        "action", help="an action the scene's robot performs, such as pick_up_milk"
    )
    parser.add_argument(
        "directory", type=Path, help="the directory the videos and the player go into"
    )
    parser.add_argument("--fps", type=int, default=25, help="video frames per second")
    parser.add_argument("--width", type=int, default=1280, help="video width in pixels")
    parser.add_argument(
        "--height", type=int, default=720, help="video height in pixels"
    )
    parsed = parser.parse_args(list(arguments))
    return IsaacRecordingRun(
        scene=PhysicsScene.from_command_line(parsed.scene),
        action=PhysicsAction.from_command_line(parsed.action),
        directory=parsed.directory,
        frames_per_second=parsed.fps,
        width=parsed.width,
        height=parsed.height,
    )


# %% files a recording writes


class RecordingFile(StrEnum):
    """
    The files a recording writes or reads, by what they hold.
    """

    PLAYER = "player.html"
    """
    The page that plays the videos.
    """

    PLAYER_TEMPLATE = "recording_player.html"
    """
    The page the player is made from, next to this script.
    """

    VIDEO_SUFFIX = ".mp4"
    """
    The ending of every video's file name.
    """


class PlayerTemplate(StrEnum):
    """
    The places in :attr:`RecordingFile.PLAYER_TEMPLATE` filled in for a recording.
    """

    RECORDING = "__RECORDING__"
    """
    Where the description of the recording goes, as JSON.
    """


class VideoCodec(StrEnum):
    """
    How the videos are compressed, by FFmpeg's names.
    """

    H264 = "libx264"
    """
    H.264, which every browser plays.
    """


class ReplicatorAnnotator(StrEnum):
    """
    What Isaac Sim's Replicator reads off a camera.
    """

    COLOUR = "rgb"
    """
    The picture, in red, green, blue and alpha.
    """


# %% one camera's video


@dataclass
class CameraVideo:
    """
    The video of one camera, as it is recorded.
    """

    camera: SceneCamera
    """
    The camera.
    """

    annotator: Annotator
    """
    What reads the camera's picture each time the stage is drawn.
    """

    writer: Format.Writer
    """
    Where the camera's frames are written.
    """

    def write_frame(self) -> None:
        """
        Write the picture the camera shows now as the video's next frame, if it has one.
        """
        picture = self.annotator.get_data()
        if picture is None or picture.size == 0:
            return
        self.writer.append_data(np.asarray(picture)[:, :, :3])


# %% the recording simulation


class RecordingIsaacSim(IsaacSim):
    """
    An :class:`IsaacSim` that, once recording, writes a frame of every camera's video for
    every :attr:`frame_period` of simulated time.
    """

    def __init__(self, frames_per_second: int, **kwargs: Any) -> None:
        """
        :param frames_per_second: How many frames each second of simulated time gets.
        :param kwargs: The settings of :class:`IsaacSim`.
        """
        super().__init__(**kwargs)
        self.frame_period = 1.0 / frames_per_second
        self._videos: list[CameraVideo] = []
        self._frames_written = 0

    def start_recording(self, videos: list[CameraVideo]) -> None:
        """
        :param videos: The videos to write frames of from now on.
        """
        self._videos = videos
        self._frames_written = self._frames_due()

    def stop_recording(self) -> None:
        """
        Finish every video.
        """
        for video in self._videos:
            video.writer.close()
        self._videos = []

    def step_simulation(self, duration: timedelta) -> None:
        """
        :meth:`IsaacSim.step_simulation`, then write every frame that has become due.
        """
        super().step_simulation(duration)
        while self._videos and self._frames_written < self._frames_due():
            RenderingManager.render()
            for video in self._videos:
                video.write_frame()
            self._frames_written += 1

    def _frames_due(self) -> int:
        """
        :return: How many frames the simulated time so far is worth.
        """
        return math.floor(
            self.simulator.current_simulation_time / self.frame_period + 1e-9
        )


# %% the description the player reads


@dataclass
class PlayerCamera:
    """
    One camera as the player offers it.
    """

    label: str
    """
    What the player calls the camera.
    """

    video: str
    """
    The camera's video file, next to the player.
    """


@dataclass
class PlayerRecording:
    """
    A recording as the player describes it.
    """

    title: str
    """
    What the player calls the recording.
    """

    cameras: list[PlayerCamera]
    """
    The cameras, in the order the player offers them.
    """


# %% recording an action


@dataclass
class IsaacRecordingRun(WatchedRun):
    """
    A run recorded headless from every camera, with a page that plays the videos.
    """

    directory: Path = field(kw_only=True)
    """
    The directory the videos and the player go into.
    """

    frames_per_second: int = 25
    """
    How many frames each second of simulated time gets.
    """

    width: int = 1280
    """
    The width of every video, in pixels.
    """

    height: int = 720
    """
    The height of every video, in pixels.
    """

    after_the_action: timedelta = field(default=timedelta(seconds=1.5))
    """
    The simulated time recorded after the robot has acted.
    """

    frames_to_warm_up: int = 20
    """
    How many times the stage is drawn before recording, so every camera has a picture.
    """

    video_quality: int = 8
    """
    The quality the videos are compressed with, from 0 to 10.
    """

    _simulation: RecordingIsaacSim | None = field(init=False, default=None)
    """
    The simulation being recorded, once the run has made it.
    """

    def _simulation_of(self, world: World) -> IsaacSim:
        self._simulation = RecordingIsaacSim(
            frames_per_second=self.frames_per_second, world=world
        )
        return self._simulation

    def _watch_with(self, cameras: dict[SceneCamera, Sdf.Path]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        videos = [
            CameraVideo(
                camera=camera,
                annotator=self._annotator_of(path),
                writer=imageio.get_writer(
                    str(self._video_path(camera)),
                    fps=self.frames_per_second,
                    codec=VideoCodec.H264,
                    quality=self.video_quality,
                ),
            )
            for camera, path in cameras.items()
        ]
        for _ in range(self.frames_to_warm_up):
            RenderingManager.render()
        self._simulation.start_recording(videos)

    def _before_the_action(self) -> None:
        pass

    def _after_the_action(self, simulation: IsaacSim) -> None:
        simulation.step_simulation(self.after_the_action)
        self._simulation.stop_recording()
        self._write_player()

    def _annotator_of(self, camera: Sdf.Path) -> Annotator:
        """
        :param camera: A camera's prim.
        :return: What reads the camera's picture, at the videos' size.
        """
        render_product = replicator.create.render_product(
            str(camera), (self.width, self.height)
        )
        annotator = replicator.AnnotatorRegistry.get_annotator(
            ReplicatorAnnotator.COLOUR
        )
        annotator.attach([render_product])
        return annotator

    def _video_path(self, camera: SceneCamera) -> Path:
        """
        :param camera: A camera.
        :return: The file its video is written to.
        """
        return self.directory / f"{camera.name.lower()}{RecordingFile.VIDEO_SUFFIX}"

    def _write_player(self) -> None:
        """
        Write the page that plays the videos.
        """
        recording = PlayerRecording(
            title=f"{self.action.name.lower()} in {self.scene.name.lower()}",
            cameras=[
                PlayerCamera(label=camera.label, video=self._video_path(camera).name)
                for camera in SceneCamera
            ],
        )
        template = (Path(__file__).parent / RecordingFile.PLAYER_TEMPLATE).read_text()
        (self.directory / RecordingFile.PLAYER).write_text(
            template.replace(PlayerTemplate.RECORDING, json.dumps(asdict(recording)))
        )


# %% the program


def main(arguments: Sequence[str] | None = None) -> None:
    """
    Record the run the command line asks for.

    :param arguments: The command line after the program, or ``None`` to read it as
        given.
    """
    parse_arguments(sys.argv[1:] if arguments is None else arguments).run()
    IsaacApplication().start().close()


if __name__ == "__main__":
    main()
