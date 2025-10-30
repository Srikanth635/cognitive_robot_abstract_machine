from __future__ import annotations

from dataclasses import field, dataclass
from typing import List, Union

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import GoalInitalizationException
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.monitors.monitors import Monitor
from giskardpy.motion_statechart.tasks.task import Task
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
)


@dataclass
class Goal(MotionStatechartNode):
    tasks: List[Task] = field(default_factory=list, init=False)
    monitors: List[Monitor] = field(default_factory=list, init=False)
    goals: List[Goal] = field(default_factory=list, init=False)

    def has_tasks(self) -> bool:
        return len(self.tasks) > 0

    def arrange_in_sequence(self, nodes: List[MotionStatechartNode]) -> None:
        first_node = nodes[0]
        first_node.end_condition = first_node
        for node in nodes[1:]:
            node.start_condition = first_node
            node.end_condition = node
            first_node = node

    def connect_start_condition_to_all_tasks(self, condition: str) -> None:
        for task in self.tasks:
            if task.start_condition == "True":
                task.start_condition = condition
            else:
                task.start_condition = f"{task.start_condition} and {condition}"

    def connect_pause_condition_to_all_tasks(self, condition: str) -> None:
        for task in self.tasks:
            if task.pause_condition == "False":
                task.pause_condition = condition
            else:
                task.pause_condition = f"{task.pause_condition} or {condition}"

    def connect_end_condition_to_all_tasks(self, condition: str) -> None:
        for task in self.tasks:
            if task.end_condition == "False":
                task.end_condition = condition
            elif not condition == "False":
                task.end_condition = f"{task.end_condition} and {condition}"

    def connect_monitors_to_all_tasks(
        self, start_condition: str, pause_condition: str, end_condition: str
    ):
        self.connect_start_condition_to_all_tasks(start_condition)
        self.connect_pause_condition_to_all_tasks(pause_condition)
        self.connect_end_condition_to_all_tasks(end_condition)

    @property
    def ref_str(self) -> str:
        """
        A string referring to self on the god_map. Used with symbol manager.
        """
        return f"god_map.motion_statechart_manager.goal_state.get_node('{self.name}')"

    def __add__(self, other: str) -> str:
        if isinstance(other, str):
            return self.ref_str + other
        raise NotImplementedError("Goal can only be added with a string.")

    def _task_sanity_check(self):
        if not self.has_tasks():
            raise GoalInitalizationException(f"Goal {str(self)} has no tasks.")

    def add_constraints_of_goal(self, goal: Goal):
        for task in goal.tasks:
            if not [t for t in self.tasks if t.name == task.name]:
                self.tasks.append(task)
            else:
                raise GoalInitalizationException(
                    f"Constraint with name {task.name} already exists."
                )

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

    def add_monitor(self, monitor: Monitor) -> None:
        self.monitors.append(monitor)

    def add_goal(self, goal: Goal) -> None:
        self.goals.append(goal)
