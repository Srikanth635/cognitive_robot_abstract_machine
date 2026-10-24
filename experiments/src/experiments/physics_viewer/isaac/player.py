"""
The page that plays the videos of a recorded run, one camera at a time or all side by
side, at a chosen speed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

# %% files


class RecordingFile(StrEnum):
    """
    The files a recording writes or reads, by what they hold.
    """

    PLAYER = "player.html"
    """
    The page that plays the videos, written next to them; the page it is made from has
    the same name, next to this module.
    """

    VIDEO_SUFFIX = ".mp4"
    """
    The ending of every video's file name.
    """


class PlayerTemplate(StrEnum):
    """
    The places in the page the player is made from that are filled in for a recording.
    """

    RECORDING = "__RECORDING__"
    """
    Where the description of the recording goes, as JSON.
    """


# %% the description the page reads


@dataclass
class PlayerCamera:
    """
    One camera as the page offers it.
    """

    label: str
    """
    What the page calls the camera.
    """

    video: str
    """
    The camera's video file, next to the page.
    """


@dataclass
class RecordingPlayer:
    """
    The page that plays the videos of a recording.
    """

    title: str
    """
    What the page calls the recording.
    """

    cameras: list[PlayerCamera]
    """
    The cameras, in the order the page offers them.
    """

    def write(self, directory: Path) -> Path:
        """
        Write the page next to the videos.

        :param directory: The directory the videos are in.
        :return: The page's file.
        """
        template = (Path(__file__).parent / RecordingFile.PLAYER).read_text()
        page = directory / RecordingFile.PLAYER
        page.write_text(
            template.replace(PlayerTemplate.RECORDING, json.dumps(asdict(self)))
        )
        return page
