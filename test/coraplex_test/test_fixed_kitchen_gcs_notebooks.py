# %% Imports
import ast
import json
from pathlib import Path


# %% Fixed-kitchen notebook presentation
class TestFixedKitchenNotebookPresentation:
    notebook_names = (
        "graph_of_convex_sets_kitchen_pickup.ipynb",
        "graph_of_convex_sets_full_kitchen_transport.ipynb",
        "graph_of_convex_sets_full_kitchen_pouring.ipynb",
    )
    """
    Fixed-kitchen notebooks that expose the same infrastructure boundary.
    """

    required_helpers = {
        "setup_fixed_kitchen",
        "create_navigation_graph",
        "create_navigation_route",
        "find_reachable_route",
        "create_navigation_playback",
        "plot_navigation_routes",
        "start_ros_node",
        "start_visualization_publisher",
    }
    """
    Infrastructure operations hidden behind the collapsed helper cell.
    """

    expected_code_cell_ids = {
        "graph_of_convex_sets_kitchen_pickup.ipynb": [
            "imports",
            "world-setup",
            "ros-node",
            "ros-visualization",
            "task-context",
            "gcs-path",
            "gcs-plot",
            "slow-navigation",
            "task-execution",
            "result",
        ],
        "graph_of_convex_sets_full_kitchen_transport.ipynb": [
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
        ],
        "graph_of_convex_sets_full_kitchen_pouring.ipynb": [
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
        ],
    }
    """
    Ordered code-cell identifiers that form each notebook's visible workflow.
    """

    def test_boilerplate_is_collapsed_behind_named_helpers(self) -> None:
        """
        Ensure every fixed-kitchen notebook foregrounds its experiment.
        """
        notebook_directory = (
            Path(__file__).parents[2] / "coraplex" / "demos" / "coraplex_gcs_demo"
        )

        for notebook_name in self.notebook_names:
            notebook = json.loads(
                (notebook_directory / notebook_name).read_text(encoding="utf-8")
            )
            code_cells = {
                cell["id"]: cell
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            }
            infrastructure = code_cells["imports"]
            parsed_infrastructure = ast.parse(
                "".join(infrastructure["source"]), filename=notebook_name
            )
            helper_names = {
                node.name
                for node in parsed_infrastructure.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            visible_source = "\n".join(
                "".join(cell["source"])
                for cell_id, cell in code_cells.items()
                if cell_id != "imports"
            )

            assert infrastructure["metadata"]["jupyter"]["source_hidden"] is True
            assert self.required_helpers <= helper_names, notebook_name
            assert "setup_fixed_kitchen(" in visible_source, notebook_name
            assert "create_navigation_graph(" in visible_source, notebook_name
            if notebook_name != "graph_of_convex_sets_kitchen_pickup.ipynb":
                assert "find_reachable_route(" in visible_source, notebook_name
            assert "create_navigation_playback(" in visible_source, notebook_name
            assert "plot_navigation_routes(" in visible_source, notebook_name
            assert "rclpy.create_node" not in visible_source, notebook_name

    def test_ros_startup_and_visualization_are_separate(self) -> None:
        """
        Ensure ROS startup and visualization remain distinct notebook steps.
        """
        notebook_directory = (
            Path(__file__).parents[2] / "coraplex" / "demos" / "coraplex_gcs_demo"
        )

        for notebook_name in self.notebook_names:
            notebook = json.loads(
                (notebook_directory / notebook_name).read_text(encoding="utf-8")
            )
            code_cells = {
                cell["id"]: "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            }

            assert "start_ros_node(" in code_cells["ros-node"], notebook_name
            assert "Context(" in code_cells["ros-node"], notebook_name
            assert "start_visualization_publisher(" not in code_cells["ros-node"]
            assert (
                "start_visualization_publisher(" in code_cells["ros-visualization"]
            ), notebook_name
            assert "start_ros_node(" not in code_cells["ros-visualization"]
            assert "start_ros_node(" not in code_cells["world-setup"]

    def test_tf_starts_before_marker_publication(self) -> None:
        """
        Ensure the first marker can be transformed at its publication time.
        """
        notebook_directory = (
            Path(__file__).parents[2] / "coraplex" / "demos" / "coraplex_gcs_demo"
        )

        for notebook_name in self.notebook_names:
            notebook = json.loads(
                (notebook_directory / notebook_name).read_text(encoding="utf-8")
            )
            infrastructure_source = "".join(
                next(cell for cell in notebook["cells"] if cell.get("id") == "imports")[
                    "source"
                ]
            )
            parsed_infrastructure = ast.parse(
                infrastructure_source, filename=notebook_name
            )
            visualization_helper = next(
                node
                for node in parsed_infrastructure.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "start_visualization_publisher"
            )
            constructor_calls = sorted(
                (
                    node.lineno,
                    node.func.id,
                )
                for node in ast.walk(visualization_helper)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"TFPublisher", "VizMarkerPublisher"}
            )
            helper_source = ast.get_source_segment(
                infrastructure_source, visualization_helper
            )

            assert [name for _, name in constructor_calls] == [
                "TFPublisher",
                "VizMarkerPublisher",
            ], notebook_name
            assert ".with_tf_publisher(" not in helper_source, notebook_name

    def test_action_composition_remains_visible(self) -> None:
        """
        Ensure infrastructure extraction does not hide the demonstrated task.
        """
        notebook_directory = (
            Path(__file__).parents[2] / "coraplex" / "demos" / "coraplex_gcs_demo"
        )

        for notebook_name in self.notebook_names:
            notebook = json.loads(
                (notebook_directory / notebook_name).read_text(encoding="utf-8")
            )
            task_cell = next(
                cell for cell in notebook["cells"] if cell.get("id") == "task-execution"
            )
            task_source = "".join(task_cell["source"])

            assert "task = sequential(" in task_source, notebook_name
            assert "task.perform()" in task_source, notebook_name

    def test_code_cells_form_a_named_workflow(self) -> None:
        """
        Ensure generated and empty cells do not interrupt the demonstration.
        """
        notebook_directory = (
            Path(__file__).parents[2] / "coraplex" / "demos" / "coraplex_gcs_demo"
        )

        for notebook_name, expected_ids in self.expected_code_cell_ids.items():
            notebook = json.loads(
                (notebook_directory / notebook_name).read_text(encoding="utf-8")
            )
            code_cell_ids = [
                cell["id"] for cell in notebook["cells"] if cell["cell_type"] == "code"
            ]

            assert code_cell_ids == expected_ids, notebook_name

    def test_pickup_retains_its_known_navigation_target(self) -> None:
        """
        Ensure pickup does not replace its fixed route with reachability search.
        """
        notebook_path = (
            Path(__file__).parents[2]
            / "coraplex"
            / "demos"
            / "coraplex_gcs_demo"
            / "graph_of_convex_sets_kitchen_pickup.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = {
            cell["id"]: "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        }

        assert "Point3(1.65, 2.0, 0.2" in code_cells["gcs-path"]
        assert "minimum_waypoint_count=3" in code_cells["gcs-path"]
        assert "find_reachable_route(" not in code_cells["gcs-path"]
        assert "yaw=pickup_yaw" in code_cells["gcs-path"]
