"""
Cramera reaches llm_reasoner's knowledge component only through its public module.
"""

from __future__ import annotations

import ast
from pathlib import Path

import cramera
from llm_reasoner.knowledge import access

KNOWLEDGE_PACKAGE = "llm_reasoner.knowledge"
"""
The package of the knowledge component.
"""

CRAMERA_DIRECTORY = Path(cramera.__file__).parent
"""
Where cramera's modules are.
"""

# %% imports of the knowledge component


def knowledge_modules_imported_by(module: Path) -> list[str]:
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


def test_cramera_reaches_the_knowledge_component_only_through_its_access_module():
    imported = {
        module
        for path in CRAMERA_DIRECTORY.rglob("*.py")
        for module in knowledge_modules_imported_by(path)
    }
    assert imported == {access.__name__}
