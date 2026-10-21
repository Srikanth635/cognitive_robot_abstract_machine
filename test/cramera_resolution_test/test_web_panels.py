"""
Checks of the add-on's panels: that every asset the page is given exists, that the panel
calls the routes the server defines, and the browser tests of the panel run with node.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from cramera_resolution.paths import AddOnDirectory
from cramera_resolution.server import AddOnScript, AddOnStylesheet, Route

# %% what the page is given


class TestAssetsOfThePage:
    @pytest.mark.parametrize("asset", [*AddOnScript, *AddOnStylesheet])
    def test_every_asset_the_page_is_given_is_kept(self, asset):
        assert (AddOnDirectory.WEB / asset).is_file()

    def test_every_script_kept_is_given_to_the_page(self):
        kept = {
            f"addon/{path.name}" for path in (AddOnDirectory.WEB / "addon").glob("*.js")
        }
        assert kept == {str(script) for script in AddOnScript}

    @pytest.mark.parametrize(
        "route",
        [Route.DESIGNATORS, Route.RUNS, Route.RUN, Route.RESOLVE, Route.EXECUTE],
    )
    def test_the_panel_calls_the_routes_the_server_defines(self, route):
        panel = (AddOnDirectory.WEB / AddOnScript.DESIGNATOR_PANEL).read_text()
        assert f"'{route}'" in panel


# %% the browser tests


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
class TestPanelsInTheBrowser:
    @staticmethod
    def run_node(name: str) -> None:
        """
        Run one node test file, failing with its output.

        :param name: The file, under the ``js`` directory beside this one.
        """
        result = subprocess.run(
            ["node", "--test", str(Path(__file__).parent / "js" / name)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_the_markup_of_a_resolution(self):
        self.run_node("test_resolution_view.js")

    def test_the_designator_panel(self):
        self.run_node("test_designator_panel.js")

    def test_the_panel_tabs(self):
        self.run_node("test_panel_tabs.js")

    def test_the_knowledge_tabs(self):
        self.run_node("test_knowledge_tabs.js")

    def test_the_designator_completion(self):
        self.run_node("test_designator_completion.js")

    def test_the_hypothesis_graph(self):
        self.run_node("test_hypothesis_graph.js")

    def test_the_ontology_graph(self):
        self.run_node("test_ontology_graph.js")

    def test_the_designator_view(self):
        self.run_node("test_designator_view.js")
