"""
Unit tests for the live runner: the library entry point and the demo wrapper CLI.
"""

from __future__ import annotations

import runpy
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path

from cramph.executor import Executor
from semantic_digital_twin.world import World
from typing_extensions import Any, Dict, List, Optional

from cramera.live import runner
from cramera.live.runner import LiveDemo
from cramera.live.visualization import LiveVisualization


@dataclass
class StartedVisualization:
    """
    A visualization recorder standing in for ``LiveVisualization``.
    """

    port: int
    world: Optional[World] = None
    started: bool = False

    def start(self, world: World):
        self.world = world
        self.started = True
        return self


@dataclass(kw_only=True)
class ExtendedExecutor:
    """
    An executor as far as the demo runner extends it: the extensions it runs with.
    """

    extensions: List[Any] = field(default_factory=list)
    """
    The extensions the executor runs with.
    """


class TestStart:
    def test_start_serves_the_given_world(self, monkeypatch):
        monkeypatch.setattr(runner, "LiveVisualization", StartedVisualization)
        world = World()

        result = runner.start(world, port=1234)

        assert isinstance(result, StartedVisualization)
        assert result.world is world
        assert result.port == 1234
        assert result.started is True


class TestMain:
    def test_main_runs_the_demo_from_its_own_directory(self, monkeypatch):
        run_calls: List[Dict[str, Any]] = []
        monkeypatch.setattr(sys, "argv", ["cramera-live", "/demos/demo.py"])
        monkeypatch.setattr(
            runpy,
            "run_path",
            lambda path, run_name: run_calls.append(
                {"path": path, "run_name": run_name}
            ),
        )
        monkeypatch.setattr(signal, "pause", lambda: None)

        runner.main()

        assert run_calls == [{"path": "/demos/demo.py", "run_name": "__main__"}]
        assert "/demos" in sys.path


class TestEveryExecutorIsVisualized:
    def test_an_executor_the_demo_creates_runs_with_the_visualization(
        self, monkeypatch
    ):
        monkeypatch.setattr(Executor, "__post_init__", lambda executor: None)
        created: List[ExtendedExecutor] = []

        def demo(path: str, run_name: str) -> None:
            executor = ExtendedExecutor()
            Executor.__post_init__(executor)
            created.append(executor)

        monkeypatch.setattr(runpy, "run_path", demo)
        monkeypatch.setattr(signal, "pause", lambda: None)
        live_demo = LiveDemo(path=Path("/demos/demo.py"))

        live_demo.run()

        [executor] = created
        assert executor.extensions == [live_demo.visualization]

    def test_executors_created_after_the_demo_are_not_visualized(self, monkeypatch):
        monkeypatch.setattr(Executor, "__post_init__", lambda executor: None)
        monkeypatch.setattr(runpy, "run_path", lambda path, run_name: None)
        monkeypatch.setattr(signal, "pause", lambda: None)
        LiveDemo(path=Path("/demos/demo.py")).run()
        executor = ExtendedExecutor()

        Executor.__post_init__(executor)

        assert executor.extensions == []
