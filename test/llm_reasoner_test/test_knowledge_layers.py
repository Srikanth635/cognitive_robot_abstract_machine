"""
The knowledge component's packages form layers: each imports only the packages below it,
and the resolver reaches the component only through its public module.
"""

from __future__ import annotations

import ast
from pathlib import Path

import llm_reasoner.knowledge
import llm_reasoner.resolver
from llm_reasoner.knowledge import access

KNOWLEDGE_DIRECTORY = Path(llm_reasoner.knowledge.__file__).parent
"""
Where the knowledge component's packages are.
"""

RESOLVER_DIRECTORY = Path(llm_reasoner.resolver.__file__).parent
"""
Where the resolver's modules are.
"""

LAYERS = [
    "ontologies",
    "ontology_translation",
    "representation",
    "acceptance",
    "admission",
    "storage",
    "generation",
    "hypotheses",
]
"""
The knowledge component's packages, lowest layer first.
"""


def knowledge_packages_imported_by(module: Path) -> set:
    """
    :return: The knowledge component's packages a module imports.
    """
    tree = ast.parse(module.read_text())
    return {
        node.module.split(".")[2]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith(f"{llm_reasoner.knowledge.__name__}.")
    }


def knowledge_modules_imported_by(module: Path) -> set:
    """
    :return: The knowledge component's modules a module imports.
    """
    tree = ast.parse(module.read_text())
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith(llm_reasoner.knowledge.__name__)
    }


def test_every_knowledge_package_imports_only_the_packages_below_it():
    upward = [
        f"{module.relative_to(KNOWLEDGE_DIRECTORY)} imports {imported}"
        for package in LAYERS
        for module in (KNOWLEDGE_DIRECTORY / package).rglob("*.py")
        for imported in knowledge_packages_imported_by(module)
        if imported in LAYERS and LAYERS.index(imported) > LAYERS.index(package)
    ]
    assert upward == []


def test_every_knowledge_package_has_its_layer():
    packages = {
        path.name
        for path in KNOWLEDGE_DIRECTORY.iterdir()
        if (path / "__init__.py").exists()
    }
    assert packages == set(LAYERS)


def test_the_resolver_reaches_the_knowledge_component_only_through_its_access_module():
    imported = {
        module
        for path in RESOLVER_DIRECTORY.rglob("*.py")
        for module in knowledge_modules_imported_by(path)
    }
    assert imported == {access.__name__}
