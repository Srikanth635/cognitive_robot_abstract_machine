from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pycram.exceptions import PlanNodeIsNone
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from pycram.plans.plan import Plan
    from pycram.plans.plan_node import PlanNode


@dataclass
class Designator:
    """
    Abstract base class for designators.
    Designators are objects that can be executed and are managed by a plan node.
    """

    plan_node: PlanNode = field(kw_only=True, default=None, repr=False)
    """
    The plan node that manages the designator.
    """

    @property
    def plan(self) -> Plan:
        if self.plan_node is None:
            raise PlanNodeIsNone(self)
        return self.plan_node.plan

    @property
    def robot(self) -> AbstractRobot:
        if self.plan_node is None:
            raise PlanNodeIsNone(self)
        return self.plan.robot

    @property
    def world(self) -> World:
        if self.plan_node is None:
            raise PlanNodeIsNone(self)
        return self.plan_node.plan.world
