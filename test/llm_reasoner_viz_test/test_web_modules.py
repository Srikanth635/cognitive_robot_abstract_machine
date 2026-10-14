"""
The modules of the page that hold its logic, tested apart from any browser: each node
test file in ``js`` is run once.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from llm_reasoner_viz.server import WEB_ROOT

JAVASCRIPT_TESTS = Path(__file__).parent / "js"
"""
Where the node tests of the page's modules are.
"""

NODE_TIMEOUT_SECONDS = 120
"""
How long one test file may take.
"""

SCRIPT_OF_THE_PAGE = re.compile(r"<script[^>]*>")
"""
A script the page loads, as its opening tag.
"""

IMPORT_OF_A_MODULE = re.compile(r"from\s+'(\./[^']+)'")
"""
A module of the page importing another by its path.
"""


class TestPageFiles:
    def test_the_page_loads_its_script_as_a_module(self):
        page = (WEB_ROOT / "index.html").read_text()
        scripts = SCRIPT_OF_THE_PAGE.findall(page)
        assert scripts
        assert all('type="module"' in script for script in scripts)

    @pytest.mark.parametrize(
        "module", sorted(WEB_ROOT.glob("*.mjs")), ids=lambda module: module.name
    )
    def test_a_module_imports_only_modules_that_exist(self, module: Path):
        imported = IMPORT_OF_A_MODULE.findall(module.read_text())
        assert all((WEB_ROOT / reference).is_file() for reference in imported)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
class TestPageModules:
    @pytest.mark.parametrize(
        "test_file",
        sorted(JAVASCRIPT_TESTS.glob("test_*.mjs")),
        ids=lambda test_file: test_file.stem,
    )
    def test_the_tests_of_a_module_pass(self, test_file: Path):
        result = subprocess.run(
            ["node", "--test", str(test_file)],
            capture_output=True,
            check=False,
            text=True,
            timeout=NODE_TIMEOUT_SECONDS,
        )
        assert result.returncode == 0, result.stdout + result.stderr
