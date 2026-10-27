"""
Checks of the resolution panels: that the page loads every one of them, that the panel
calls the routes the server defines, and the browser tests of the panels run with node.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from cramera import paths

from cramera.resolution.routes import ResolutionRoute

PANELS = paths.WEB_ROOT / "panels" / "resolution"

# %% what the page is given


class TestPanelsOfThePage:
    @pytest.fixture
    def page(self) -> str:
        return (paths.WEB_ROOT / "index.html").read_text(encoding="utf-8")

    def test_every_script_kept_is_given_to_the_page(self, page):
        given = set(re.findall(r'<script src="panels/resolution/([^"]+)"', page))
        assert given == {path.name for path in PANELS.glob("*.js")}

    def test_the_knowledge_views_of_the_graph_panel_are_given_to_the_page(self, page):
        assert '<script src="panels/graph/knowledge_views.js"></script>' in page

    def test_the_tab_frame_and_its_groups_are_given_to_the_page(self, page):
        assert '<script src="core/panel-tabs.js"></script>' in page
        assert '<script src="panel-groups.js"></script>' in page

    def test_every_stylesheet_kept_is_given_to_the_page(self, page):
        given = set(
            re.findall(r'<link rel="stylesheet" href="panels/resolution/([^"]+)"', page)
        )
        assert given == {path.name for path in PANELS.glob("*.css")}

    def test_the_scripts_come_before_the_script_that_boots_the_panels(self, page):
        boot = page.index("Panels.boot()")
        assert all(
            page.index(f"panels/resolution/{path.name}") < boot
            for path in PANELS.glob("*.js")
        )

    def test_the_layout_places_the_designators_between_the_eql_panel_and_the_graph(
        self,
    ):
        config = (paths.WEB_ROOT / "config.js").read_text()
        assert re.search(
            r"right:\s*\['eql',\s*'resolution-designators',\s*'graph'\]", config
        )

    @pytest.mark.parametrize(
        "route",
        [
            ResolutionRoute.DESIGNATORS,
            ResolutionRoute.RUNS,
            ResolutionRoute.RUN,
            ResolutionRoute.RESOLVE,
            ResolutionRoute.EXECUTE,
        ],
    )
    def test_the_panel_calls_the_routes_the_server_defines(self, route):
        panel = (PANELS / "panel.js").read_text()
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

    def test_the_knowledge_views(self):
        self.run_node("test_knowledge_views.js")

    def test_the_designator_completion(self):
        self.run_node("test_designator_completion.js")

    def test_the_hypothesis_graph(self):
        self.run_node("test_hypothesis_graph.js")

    def test_the_ontology_graph(self):
        self.run_node("test_ontology_graph.js")

    def test_the_designator_view(self):
        self.run_node("test_designator_view.js")
