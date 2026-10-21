# %% Imports
import ast
import json
from pathlib import Path

# %% Same-counter pouring notebook
NOTEBOOK_PATH = (
    Path(__file__).parents[2]
    / "coraplex"
    / "demos"
    / "coraplex_gcs_demo"
    / "graph_of_convex_sets_kitchen_counter_pouring.ipynb"
)


def test_counter_pouring_notebook_has_one_short_navigation_leg() -> None:
    """
    Ensure the robot approaches once and pours without further navigation.
    """
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
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
        "short-gcs-route",
        "gcs-plot",
        "navigation-playback",
        "task-execution",
        "result",
    ]
    for cell_id, source in code_cells.items():
        ast.parse(source, filename=f"{NOTEBOOK_PATH.name}:{cell_id}")

    world_source = code_cells["world-setup"]
    route_source = code_cells["short-gcs-route"]
    task_source = code_cells["task-execution"]
    result_source = code_cells["result"]

    assert "pitcher_height = island_height" in world_source
    assert "bowl_height = island_height" in world_source
    assert "table_height" not in world_source
    assert "pickup_route = create_navigation_route(" in route_source
    assert "pouring_route" not in route_source
    assert "short_route_length" in route_source
    assert "*pickup_navigation_steps" in task_source
    assert "*pouring_navigation_steps" not in task_source
    assert "PickUpAction(" in task_source
    assert "PouringAction(" in task_source
    assert "task = sequential(" in task_source
    assert "assert short_route_length <" in result_source
    assert "assert final_base_error <" in result_source


def test_counter_pouring_starts_tf_before_markers() -> None:
    """
    Ensure the same-counter example avoids stale MarkerArray timestamps.
    """
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    infrastructure_source = "".join(
        next(cell for cell in notebook["cells"] if cell.get("id") == "imports")[
            "source"
        ]
    )
    parsed_infrastructure = ast.parse(infrastructure_source)
    visualization_helper = next(
        node
        for node in parsed_infrastructure.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "start_visualization_publisher"
    )
    constructor_calls = sorted(
        (node.lineno, node.func.id)
        for node in ast.walk(visualization_helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"TFPublisher", "VizMarkerPublisher"}
    )

    assert [name for _, name in constructor_calls] == [
        "TFPublisher",
        "VizMarkerPublisher",
    ]
