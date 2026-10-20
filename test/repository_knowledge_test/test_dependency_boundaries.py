"""
Import direction protects source models from changing infrastructure.
"""

from __future__ import annotations

import ast
from pathlib import Path

# %% stable dependency direction


def test_inner_layers_do_not_import_concrete_adapters_or_clients() -> None:
    """
    Models, contracts and operations retain the intended dependency direction.
    """
    source_root = (
        Path(__file__).parents[2] / "repository_knowledge/src/repository_knowledge"
    )
    inner_sources = [
        source
        for source in source_root.glob("*.py")
        if source.name not in ("composition.py", "__main__.py")
    ]
    inner_sources.extend((source_root / "operations").glob("*.py"))
    forbidden = (
        "repository_knowledge.adapters",
        "repository_knowledge.interfaces",
        "repository_knowledge.composition",
        "cramera",
        "coraplex",
        "giskardpy",
        "semantic_digital_twin",
        "sqlite3",
        "subprocess",
        "argparse",
        "yaml",
    )
    violations = []
    for source in inner_sources:
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
            else:
                continue
            violations.extend(
                (source.name, imported)
                for imported in imports
                if any(
                    imported == prefix or imported.startswith(prefix + ".")
                    for prefix in forbidden
                )
            )
    assert violations == []


def test_interface_uses_operations_without_storage_imports() -> None:
    """
    The command interface cannot bypass application contracts.
    """
    source = (
        Path(__file__).parents[2]
        / "repository_knowledge/src/repository_knowledge/interfaces/cli.py"
    )
    imports = [
        node.module
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
    ]
    assert [
        module
        for module in imports
        if module and module.startswith("repository_knowledge.adapters")
    ] == []
