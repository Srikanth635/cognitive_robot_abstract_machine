"""
Tests of where the resolution add-on links to the llm_reasoner viewer.
"""

from __future__ import annotations

from urllib.parse import urlparse

from llm_reasoner_viz import server as viewer_server

from cramera.resolution import paths

# %% the viewer's address


class TestViewerUrl:
    def test_a_run_is_shown_by_the_viewer_at_its_default_address(self, monkeypatch):
        monkeypatch.delenv(paths.EnvironmentVariable.VIEWER_URL, raising=False)
        assert paths.viewer_url("a-run") == f"{paths.ViewerAddress.DEFAULT}/?run=a-run"

    def test_by_default_the_viewer_is_at_the_port_it_listens_on(self):
        assert urlparse(paths.ViewerAddress.DEFAULT).port == viewer_server.DEFAULT_PORT

    def test_the_address_of_the_viewer_can_be_set(self, monkeypatch):
        monkeypatch.setenv(
            paths.EnvironmentVariable.VIEWER_URL, "http://viewer.example:9000"
        )
        assert paths.viewer_url("a-run") == "http://viewer.example:9000/?run=a-run"

    def test_a_trailing_slash_in_the_address_makes_no_difference(self, monkeypatch):
        monkeypatch.setenv(
            paths.EnvironmentVariable.VIEWER_URL, "http://viewer.example:9000/"
        )
        assert paths.viewer_url("a-run") == "http://viewer.example:9000/?run=a-run"
