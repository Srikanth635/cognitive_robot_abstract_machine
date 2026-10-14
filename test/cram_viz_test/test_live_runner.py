"""
Tests of what the live bridge runs, told by the command line: a demo script, or a
resolution run performed from the environment it started from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cram_viz import paths
from cram_viz.live.launcher import LiveArgument
from cram_viz.live.runner import DemoScript, LiveTarget, ResolvedRun


class TestLiveTarget:
    def test_a_path_alone_is_a_demo_script(self, tmp_path):
        assert LiveTarget.from_arguments([str(tmp_path / "demo.py")]) == DemoScript(
            path=tmp_path / "demo.py"
        )

    def test_a_run_argument_names_a_resolution_run_whose_neem_is_kept_in_the_shared_database(
        self, tmp_path
    ):
        assert LiveTarget.from_arguments(
            [LiveArgument.RUN, str(tmp_path / "runs" / "a-run")]
        ) == ResolvedRun(
            directory=tmp_path / "runs" / "a-run",
            neem_database=paths.neem_database_path(),
        )

    def test_no_argument_is_refused(self):
        with pytest.raises(SystemExit):
            LiveTarget.from_arguments([])
