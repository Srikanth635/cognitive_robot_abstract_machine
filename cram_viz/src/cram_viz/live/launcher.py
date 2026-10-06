"""
Launching a demo under the live bridge from the viewer's "Execute" button,
instead of a separate ``cram-viz-live path/to/demo.py`` terminal command.

The launched process is the exact same entry point that command runs
(``python -m cram_viz.live``, i.e. :func:`cram_viz.live.runner.main`), so the
viewer's own frontend needs no new wiring to pick it up: the robot scene panel
already polls the live bridge's ``/info`` endpoint every few seconds and shows
its "Live" button the moment one answers.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing_extensions import Any, Callable, Optional

from cram_viz import paths


@dataclass
class LiveDemoLauncher:
    """
    Starts a demo script under the live bridge as a background process.

    One instance is meant to be reused across requests (see
    :func:`cram_viz.server._live_demo_launcher`), so a later launch can stop a
    demo process it previously started before starting the next one.
    """

    spawn: Callable[..., Any] = subprocess.Popen
    """Process constructor; substituted with a stub in tests so nothing here
    ever starts a real subprocess."""

    termination_timeout_seconds: float = 5.0
    """How long a previous demo process is given to exit after being asked to,
    before it is killed outright, so restarting never hangs waiting on it."""

    _process: Optional[Any] = field(default=None, init=False, repr=False)

    def launch(self, demo: Path) -> None:
        """
        Launch ``demo`` in the background, its output redirected to a log file
        under :func:`~cram_viz.paths.data_dir`.

        A demo process previously started by this launcher and still alive is
        stopped first. This includes one whose demo logic already finished: the
        live bridge (:func:`cram_viz.live.runner.main`) keeps its process alive
        indefinitely after finishing, waiting to be interrupted, so a launch
        request always restarts the simulation rather than being refused.

        :raises FileNotFoundError: ``demo`` does not exist.
        """
        if not demo.is_file():
            raise FileNotFoundError(str(demo))
        self._stop_running_process()

        log_directory = paths.data_dir()
        log_directory.mkdir(parents=True, exist_ok=True)
        with (log_directory / "live_demo.log").open("w") as log_file:
            self._process = self.spawn(
                [sys.executable, "-m", "cram_viz.live", str(demo)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

    def _stop_running_process(self) -> None:
        """Terminate this launcher's previous process, killing it if it hangs."""
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=self.termination_timeout_seconds)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=self.termination_timeout_seconds)
