# %% Imports
import ast
import json
from pathlib import Path


# %% full-kitchen pouring notebook
def test_full_kitchen_pouring_notebook_has_executable_workflow() -> None:
    """
    Ensure the pouring demo retains its complete executable workflow.
    """
    notebook_path = (
        Path(__file__).parents[2]
        / "coraplex"
        / "demos"
        / "coraplex_gcs_demo"
        / "graph_of_convex_sets_full_kitchen_pouring.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }

    assert list(code_cells) == [
        "imports",
        "world-setup",
        "ros-node",
        "ros-visualization",
        "task-context",
        "route-processing",
        "full-kitchen-gcs",
        "gcs-plot",
        "navigation-playback",
        "task-execution",
        "result",
    ]

    parsed_cells = {
        cell_id: ast.parse(source, filename=f"{notebook_path.name}:{cell_id}")
        for cell_id, source in code_cells.items()
    }
    imported_names = {
        alias.name
        for parsed_cell in parsed_cells.values()
        for node in ast.walk(parsed_cell)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    called_names = {
        node.func.id
        for parsed_cell in parsed_cells.values()
        for node in ast.walk(parsed_cell)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert {
        "GraphOfConvexSets",
        "NavigateAction",
        "PickUpAction",
        "PouringAction",
    } <= imported_names
    assert {"setup_world", "sequential"} <= called_names
    assert "PouringAction" in called_names
    assert "task.perform" in code_cells["task-execution"]
    assert "pouring_tilt_radians" in code_cells["result"]
