"""
Source-module dependencies remain acyclic, including type-only imports.
"""

from __future__ import annotations

import ast
from graphlib import TopologicalSorter
from pathlib import Path

# %% package import graph


def test_package_import_graph_has_no_cycles() -> None:
    """
    Every internal module admits an ordering before its dependents.
    """
    source_root = Path(__file__).parents[2] / "repository_knowledge/src"
    modules = {
        ".".join(source.relative_to(source_root).with_suffix("").parts): source
        for source in (source_root / "repository_knowledge").rglob("*.py")
    }
    dependencies: dict[str, set[str]] = {}
    for module_name, source in modules.items():
        imported_modules: set[str] = set()
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported_modules.update(
                    alias.name for alias in node.names if alias.name in modules
                )
            elif isinstance(node, ast.ImportFrom) and node.module in modules:
                imported_modules.add(node.module)
        dependencies[module_name] = imported_modules
    assert set(TopologicalSorter(dependencies).static_order()) == set(modules)
