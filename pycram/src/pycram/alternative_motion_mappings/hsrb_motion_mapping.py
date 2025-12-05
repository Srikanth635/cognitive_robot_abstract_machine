from dataclasses import dataclass

from typing_extensions import ClassVar

from krrood.entity_query_language.symbolic import Literal
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.hsrb import HSRB
from ..datastructures.enums import ExecutionType
from ..robot_plans import MoveMotion

from ..robot_plans.motions.base import AlternativeMotionMapping


class HSRBMoveMotionMapping(MoveMotion, AlternativeMotionMapping[HSRB]):

    execution_type = ExecutionType.REAL

    def perform(self):
        pass
