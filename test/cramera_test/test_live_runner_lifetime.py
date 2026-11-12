"""
The live command releases its visualization on every exit.
"""

from unittest.mock import Mock

import pytest

from cramera.live import runner
from cramera.live.visualization import LiveVisualization


# %% command lifetime


@pytest.mark.parametrize("failed_demo", [False, True])
def test_runner_closes_its_visualization_scope(monkeypatch, failed_demo):
    """
    Normal completion and demo errors both release the visualization.
    """
    cleanup = Mock()
    monkeypatch.setattr(LiveVisualization, "stop", cleanup)
    monkeypatch.setattr(runner.sys, "argv", ["cramera-live", "/demo.py"])
    run_demo = Mock(side_effect=RuntimeError if failed_demo else None)
    monkeypatch.setattr(runner.runpy, "run_path", run_demo)
    monkeypatch.setattr(runner.signal, "pause", Mock())

    if failed_demo:
        with pytest.raises(RuntimeError):
            runner.main()
    else:
        runner.main()

    cleanup.assert_called_once_with()
