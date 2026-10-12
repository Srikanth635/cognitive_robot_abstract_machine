"""
cram_viz reaches llm_reasoner's knowledge component only through its public module.
"""

from __future__ import annotations

import ast
from pathlib import Path

import cram_viz
from llm_reasoner.knowledge import access

KNOWLEDGE_PACKAGE = "llm_reasoner.knowledge"
"""
The package of the knowledge component.
"""

CRAM_VIZ_DIRECTORY = Path(cram_viz.__file__).parent
"""
Where cram_viz's modules are.
"""


def knowledge_modules_imported_by(module: Path) -> list:
    """
    :return: The knowledge component's modules a module imports.
    """
    tree = ast.parse(module.read_text())
    return [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith(KNOWLEDGE_PACKAGE)
    ]


def test_cram_viz_reaches_the_knowledge_component_only_through_its_access_module():
    imported = {
        module
        for path in CRAM_VIZ_DIRECTORY.rglob("*.py")
        for module in knowledge_modules_imported_by(path)
    }
    assert imported == {access.__name__}
