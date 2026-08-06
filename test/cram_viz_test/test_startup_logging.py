"""
The console entry points must report that they are serving.

A silent ``cram-viz`` is indistinguishable from a hung one: the server binds its port
and answers requests, but the terminal shows nothing, so the command looks like it
never finished starting.
"""

import logging
import re
import subprocess
import sys
import time

import pytest

BANNER = "cram_viz running at"

#: generous: the entry point imports krrood and builds the knowledge base first
STARTUP_TIMEOUT = 180.0


# %% running the entry point out of process
def read_banner(log_path, process, timeout):
    """
    The entry point's startup banner, waited for until the timeout elapses.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log_path.is_file():
            for line in log_path.read_text(errors="replace").splitlines():
                if BANNER in line:
                    return line
        if process.poll() is not None:
            raise AssertionError(
                "the entry point exited (code %s) without printing its banner:\n%s"
                % (process.returncode, log_path.read_text(errors="replace"))
            )
        time.sleep(0.5)
    raise AssertionError(
        "no banner within %.0fs — the terminal stays blank while the server runs:\n%s"
        % (timeout, log_path.read_text(errors="replace"))
    )


@pytest.fixture()
def server_banner(fixture_scene, tmp_path):
    """
    The startup banner of ``python -m cram_viz.server`` on an ephemeral port.
    """
    log_path = tmp_path / "cram-viz.log"
    with log_path.open("wb") as sink:
        process = subprocess.Popen(
            [sys.executable, "-m", "cram_viz.server", "0"],
            stdout=sink,
            stderr=subprocess.STDOUT,
        )
    try:
        yield read_banner(log_path, process, STARTUP_TIMEOUT)
    finally:
        process.terminate()
        process.wait(timeout=60)


class TestEntryPointReportsItIsServing:
    """
    The banner is the only signal the user gets that startup finished.
    """

    def test_banner_names_the_url(self, server_banner):
        assert "http://localhost:" in server_banner

    def test_banner_reports_the_port_that_was_actually_bound(self, server_banner):
        port = int(re.search(r"localhost:(\d+)", server_banner).group(1))
        assert port != 0


class TestPackageImportLeavesLoggingToTheEntryPoint:
    """
    Configuring logging on import silently disables every later configuration attempt,
    because :func:`logging.basicConfig` is a no-op once the root logger has a handler.
    """

    def test_importing_the_package_installs_no_root_handler(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import logging, cram_viz; "
                "print(len(logging.getLogger().handlers))",
            ],
            capture_output=True,
            text=True,
            timeout=STARTUP_TIMEOUT,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "0"

    def test_configured_level_survives_a_preexisting_handler(self):
        from cram_viz import configure_console_logging

        root = logging.getLogger()
        original_handlers, original_level = list(root.handlers), root.level
        root.addHandler(logging.NullHandler())
        try:
            configure_console_logging()
            assert logging.getLogger("cram_viz.server").isEnabledFor(logging.INFO)
        finally:
            root.handlers[:] = original_handlers
            root.setLevel(original_level)
