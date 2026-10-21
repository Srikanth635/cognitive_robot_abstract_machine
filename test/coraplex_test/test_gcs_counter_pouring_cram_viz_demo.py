# %% Imports
import ast
import importlib.util
from pathlib import Path

import pytest

# %% Live same-counter pouring demo
DEMO_PATH = (
    Path(__file__).parents[2]
    / "coraplex"
    / "demos"
    / "coraplex_gcs_demo"
    / "cram_viz"
    / "kitchen_counter_pouring.py"
)


def sequential_call_assigned_to(
    execute_function: ast.FunctionDef, variable_name: str
) -> ast.Call:
    """
    The ``sequential([...])`` call assigned to ``variable_name`` inside
    ``execute_function``.
    """
    assignment = next(
        node
        for node in ast.walk(execute_function)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == variable_name
    )
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == "sequential"
    return assignment.value


def test_live_counter_pouring_uses_one_gcs_navigation_leg() -> None:
    """
    Ensure the live plan approaches once, then picks up and pours -- and,
    separately, holds, rights and places the cereal box back afterwards.

    The revert is its own ``sequential(...)``/``.plan.perform()`` pair rather than
    more elements appended to the pour's: reading world state (attachment,
    alignment) from *inside* a still-running plan sees it mid-update, before
    attachment/detachment settles, so the pour's own outcome must be read only
    after its plan has fully returned -- see
    ``test_live_counter_pouring_executes_the_plan_for_browser_statuses``.
    """
    parsed_demo = ast.parse(
        DEMO_PATH.read_text(encoding="utf-8"), filename=DEMO_PATH.name
    )
    imported_names = {
        alias.name
        for node in ast.walk(parsed_demo)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    execute_function = next(
        node
        for node in parsed_demo.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_counter_pouring"
    )

    assert {
        "GraphOfConvexSets",
        "NavigateAction",
        "PickUpAction",
        "PouringAction",
        "PlaceAction",
    } <= imported_names

    pour_children = sequential_call_assigned_to(execute_function, "pour_task").args[0]
    assert isinstance(pour_children, ast.List)
    assert sum(isinstance(child, ast.Starred) for child in pour_children.elts) == 1
    deferred_pouring = pour_children.elts[-1]
    assert isinstance(deferred_pouring, ast.Call)
    assert isinstance(deferred_pouring.func, ast.Call)
    assert isinstance(deferred_pouring.func.func, ast.Name)
    assert deferred_pouring.func.func.id == "a"
    assert isinstance(deferred_pouring.func.args[0], ast.Name)
    assert deferred_pouring.func.args[0].id == "PouringAction"

    revert_children = sequential_call_assigned_to(execute_function, "revert_task").args[
        0
    ]
    assert isinstance(revert_children, ast.List)
    placing = revert_children.elts[-1]
    assert isinstance(placing, ast.Call)
    assert isinstance(placing.func, ast.Name)
    assert placing.func.id == "PlaceAction"


def test_live_counter_pouring_executes_the_plan_for_browser_statuses() -> None:
    """
    Ensure CRAM-Viz can capture both plan trees before execution begins: the pour
    and, separately, the hold/right/place-back revert performed afterwards.
    """
    parsed_demo = ast.parse(
        DEMO_PATH.read_text(encoding="utf-8"), filename=DEMO_PATH.name
    )
    execute_function = next(
        node
        for node in parsed_demo.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_counter_pouring"
    )

    def plan_perform_calls(variable_name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(execute_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "perform"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "plan"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == variable_name
        ]

    assert len(plan_perform_calls("pour_task")) == 1
    assert len(plan_perform_calls("revert_task")) == 1


def test_live_counter_pouring_uses_cereal_and_original_bowl_assets() -> None:
    """
    Ensure the compact demo pours cereal into the original target bowl.
    """
    source = DEMO_PATH.read_text(encoding="utf-8")

    assert 'world.get_body_by_name("breakfast_cereal.stl")' in source
    assert 'object_directory / "bowl.stl"' in source
    assert "Bowl(root=bowl)" in source
    assert '"Static_MilkPitcher.stl"' not in source
    assert '"jeroen_cup.stl"' not in source


def test_live_counter_pouring_executes_with_short_motion_and_alignment() -> None:
    """
    Ensure the live-compatible script completes the compact pouring task.
    """
    specification = importlib.util.spec_from_file_location(
        "gcs_counter_pouring_live_demo", DEMO_PATH
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    result = module.execute_counter_pouring()

    assert result.route_length == pytest.approx(0.3, abs=0.01)
    assert result.navigation_updates == 4
    assert result.final_base_error <= 1e-6
    assert result.source_name == "breakfast_cereal.stl"
    assert result.target_name == "bowl.stl"
    assert result.alignment.horizontal_error <= 0.04
    assert 0.0 <= result.alignment.vertical_clearance <= 0.15
