# %% Imports
import ast
from pathlib import Path

from kitchen_pouring.cram_viz_simple_pouring import execute_simple_pouring


# %% Browser pouring demo
def test_browser_pouring_demo_avoids_threaded_reachability_execution() -> None:
    """
    Ensure the browser demo retains its single-threaded pouring workflow.
    """
    demo_path = (
        Path(__file__).parents[2] / "kitchen_pouring" / "cram_viz_simple_pouring.py"
    )
    parsed_demo = ast.parse(
        demo_path.read_text(encoding="utf-8"), filename=demo_path.name
    )

    imported_names = {
        alias.name
        for node in ast.walk(parsed_demo)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(parsed_demo)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    context_calls = [
        node
        for node in ast.walk(parsed_demo)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Context"
    ]

    assert {"PickUpAction", "PouringAction"} <= imported_names
    assert {"setup_world", "sequential"} <= called_names
    assert "GraphOfConvexSets" not in imported_names
    assert "NavigateAction" not in imported_names
    assert len(context_calls) == 1
    assert {
        keyword.arg: keyword.value.value
        for keyword in context_calls[0].keywords
        if isinstance(keyword.value, ast.Constant)
    }["evaluate_conditions"] is False


def test_browser_pouring_demo_uses_one_deferred_sequential_task() -> None:
    """
    Ensure pickup and late-grounded pouring share one top-level sequence.
    """
    demo_path = (
        Path(__file__).parents[2] / "kitchen_pouring" / "cram_viz_simple_pouring.py"
    )
    parsed_demo = ast.parse(
        demo_path.read_text(encoding="utf-8"), filename=demo_path.name
    )
    execute_function = next(
        node
        for node in parsed_demo.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_simple_pouring"
    )
    sequential_calls = [
        node
        for node in ast.walk(execute_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sequential"
    ]
    perform_calls = [
        node
        for node in ast.walk(execute_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "perform"
    ]

    assert len(sequential_calls) == 1
    assert len(perform_calls) == 1
    task_children = sequential_calls[0].args[0]
    assert isinstance(task_children, ast.List)
    assert len(task_children.elts) == 4
    deferred_pouring = task_children.elts[-1]
    assert isinstance(deferred_pouring, ast.Call)
    assert isinstance(deferred_pouring.func, ast.Call)
    assert isinstance(deferred_pouring.func.func, ast.Name)
    assert deferred_pouring.func.func.id == "a"
    assert isinstance(deferred_pouring.func.args[0], ast.Name)
    assert deferred_pouring.func.args[0].id == "PouringAction"


def test_browser_pouring_demo_places_pitcher_mouth_over_bowl() -> None:
    """
    Ensure the executed pour finishes with the pitcher mouth over the bowl.
    """
    alignment = execute_simple_pouring()

    assert alignment.horizontal_error <= 0.04
    assert 0.0 <= alignment.vertical_clearance <= 0.15
