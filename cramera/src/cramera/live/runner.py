"""
Starting the live viewer: as a library call or as the demo wrapper CLI.
"""

from __future__ import annotations

import logging
import runpy
import signal
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from cramph.executor import Executor
from typing_extensions import TYPE_CHECKING, Iterator

from cramera.live.http import DEFAULT_PORT
from cramera.live.visualization import LiveVisualization
from cramera.logging_setup import get_logger

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

logger = get_logger(__name__)


# %% serving a world


def start(world: World, port: int = DEFAULT_PORT) -> LiveVisualization:
    """
    Serve a world to the browser viewer.

    Add the visualization to the extensions of the executor that performs a plan in the
    world, so that the viewer follows the plan as well.

    :param world: The world to serve.
    :param port: Port of the bridge's HTTP endpoints.
    :return: The started visualization.
    """
    return LiveVisualization(port=port).start(world=world)


# %% running a demo


@dataclass(kw_only=True)
class LiveDemo:
    """
    A demo run unchanged, every executor it creates serving its world and its plan to
    the browser viewer.
    """

    path: Path
    """
    The demo's script.
    """

    visualization: LiveVisualization = field(default_factory=LiveVisualization)
    """
    Serves the world and the plans of the demo's executors.
    """

    def run(self) -> None:
        """
        Run the demo as if it were started directly, then keep the viewer up for
        inspection until Ctrl-C.
        """
        sys.path.insert(0, str(self.path.parent))
        logger.info("running demo: %s", self.path)
        with self.extending_every_executor():
            runpy.run_path(str(self.path), run_name="__main__")
            logger.info(
                "demo finished — bridge stays up for inspection (Ctrl-C to quit)"
            )
            try:
                signal.pause()
            except KeyboardInterrupt:
                pass

    @contextmanager
    def extending_every_executor(self) -> Iterator[None]:
        """
        Add :attr:`visualization` to every executor created while the context is open,
        and stop the visualization when it closes, however the demo ended.
        """
        create_executor = Executor.__post_init__
        visualization = self.visualization

        def create_visualized_executor(executor: Executor) -> None:
            """
            Create the executor as usual, then add :attr:`visualization` to it.
            """
            create_executor(executor)
            if not any(extension is visualization for extension in executor.extensions):
                executor.extensions.append(visualization)

        Executor.__post_init__ = create_visualized_executor
        try:
            yield
        finally:
            Executor.__post_init__ = create_executor
            visualization.stop()


def main() -> None:
    """
    ``cramera-live path/to/demo.py`` — run a demo with the browser viewer.

    The demo's directory is put on ``sys.path`` so it can import its local helper
    modules, exactly as if it were started directly.
    """
    logging.basicConfig(level=logging.INFO, force=True)
    if len(sys.argv) < 2:
        sys.exit("usage: cramera-live path/to/demo.py")
    LiveDemo(path=Path(sys.argv[1]).resolve()).run()


if __name__ == "__main__":
    main()
