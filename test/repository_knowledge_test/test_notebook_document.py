"""
The distributed notebook remains clean and executes without provider credentials.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.model import SourceScope

from .support import CommittedProject, FixtureSymbol

# %% saved notebook


def test_notebook_has_no_saved_execution_or_outputs() -> None:
    """
    The portable document ships without executed cells or provider output.
    """
    document = nbformat.read(
        Path(__file__).parents[2]
        / "repository_knowledge/notebooks/knowledge_playground.ipynb",
        as_version=4,
    )
    nbformat.validate(document)
    for cell in document.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []


# %% offline execution


def test_notebook_executes_real_mcp_inspection_without_an_llm(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    Run all cells against fixture evidence with the configured Python kernel.
    """
    document = nbformat.read(
        Path(__file__).parents[2]
        / "repository_knowledge/notebooks/knowledge_playground.ipynb",
        as_version=4,
    )
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CRAM_NOTEBOOK_REPOSITORY": str(committed_project.root),
            "CRAM_NOTEBOOK_DATABASE": str(indexed_storage.database_path),
            "CRAM_NOTEBOOK_SCOPE": SourceScope.COMMITTED,
            "CRAM_NOTEBOOK_SYMBOL": FixtureSymbol.CHILD,
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }
    )
    executed = NotebookClient(
        document,
        km=manager,
        timeout=90,
        resources={"metadata": {"path": str(committed_project.root)}},
    ).execute(env=environment)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code_cells] == list(
        range(1, len(code_cells) + 1)
    )
    output = "".join(
        item.text
        for cell in code_cells
        for item in cell.outputs
        if item.output_type == "stream"
    )
    assert committed_project.commit_identifier in output
    assert FixtureSymbol.CHILD in output
