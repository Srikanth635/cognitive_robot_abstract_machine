from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Optional

from typing_extensions import TYPE_CHECKING, TypeVar, ClassVar, List, Any

from giskardpy.executor import Executor
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import Task
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.ormatic.dao import HasGeneric

# from giskardpy.motion_statechart.tasks.task import Task
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world import World
from ...datastructures.enums import ExecutionType
from ...designator import DesignatorDescription
from ...process_module import ProcessModuleManager

import logging

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=AbstractRobot)


@dataclass
class AlternativeMotionMapping(HasGeneric[T], ABC):
    execution_type: ClassVar[ExecutionType]

    @property
    def motion_chart(self) -> Task:
        return None

    def perform(self):
        pass

    @staticmethod
    def check_for_alternative(
        robot_view: AbstractRobot, motion: BaseMotion
    ) -> Optional[BaseMotion]:
        for alternative in AlternativeMotionMapping.__subclasses__():
            if (
                issubclass(alternative, motion.__class__)
                and alternative.original_class == robot_view.__class__
                and ProcessModuleManager.execution_type == alternative.execution_type
            ):
                return alternative
        return None


@dataclass
class BaseMotion(DesignatorDescription):

    @abstractmethod
    def perform(self):
        """
        Passes this designator to the process module for execution. Will be overwritten by each motion.
        """
        pass

    @property
    @abstractmethod
    def motion_chart(self) -> Task:
        pass

    def get_alternative_motion(self) -> Optional[AlternativeMotionMapping]:
        return AlternativeMotionMapping.check_for_alternative(self.robot_view, self)


@dataclass
class MotionExecutor:
    motions: List[BaseMotion]
    """
    The motions to execute
    """

    world: World
    """
    The world in which the motions should be executed.
    """

    motion_state_chart: MotionStatechart = field(init=False)
    """
    Giskard's motion state chart that is created from the motions.
    """

    execution_sequence: List[MotionStatechart, BaseMotion] = field(init=False)
    """
    Sequence of motion state charts and alternative motion mappings that should be executed.
    """

    ros_node: Any = field(kw_only=True, default=None)
    """
    ROS node that should be used for communication. Only relevant for real execution.
    """

    def construct_execution_sequence(self):
        current_nodes = []
        for motion in self.motions:
            if motion.motion_chart and isinstance(motion, AlternativeMotionMapping):
                current_nodes.append(motion.motion_chart)
            elif hasattr(motion, "perform") and isinstance(
                motion, AlternativeMotionMapping
            ):
                msc = MotionStatechart()
                self.execution_sequence.append(
                    msc.add_node(Sequence(nodes=current_nodes))
                )
                current_nodes = []
                self.execution_sequence.append(motion)
            else:
                current_nodes.append(motion.motion_chart)
        msc = MotionStatechart()
        msc.add_node(Sequence(nodes=current_nodes))
        self.execution_sequence.append(msc)

    def execute(self):
        """
        Executes the constructed motion state chart in the given world.
        """
        # If there are no motions to construct an msc, return
        if len(self.execution_sequence) == 0:
            logger.warning("No motions to execute.")
            return
        if ProcessModuleManager.execution_type == ExecutionType.SIMULATED:
            logger.debug(f"Executing {self.motions} motions in simulation")
            giskard_executor = Executor(
                self.world,
                controller_config=QPControllerConfig(
                    control_dt=0.02, mpc_dt=0.02, prediction_horizon=4, verbose=False
                ),
            )

            def executor(motion_statechart: MotionStatechart):
                giskard_executor.compile(motion_statechart)
                try:
                    giskard_executor.tick_until_end(timeout=2000)
                except TimeoutError as e:
                    failed_nodes = [
                        node if node.life_cycle_state != LifeCycleValues.DONE else None
                        for node in self.motion_state_chart.nodes
                    ]
                    logger.error(f"Failed Nodes: {failed_nodes}")
                    raise e

        elif ProcessModuleManager.execution_type == ExecutionType.REAL:
            from giskardpy_ros.python_interface.python_interface import GiskardWrapper

            giskard = GiskardWrapper(self.ros_node)
            executor = giskard.execute

        for motion in self.execution_sequence:
            if isinstance(motion, MotionStatechart):
                executor(motion)
            else:
                motion.perform()

    def _execute_for_simulation(self):
        """
        Creates an executor and executes the motion state chart until it is done.
        """
        logger.debug(f"Executing {self.motions} motions in simulation")
        executor = Executor(
            self.world,
            controller_config=QPControllerConfig(
                control_dt=0.02, mpc_dt=0.02, prediction_horizon=4, verbose=False
            ),
        )

        for motion in self.execution_sequence:
            if isinstance(motion, MotionStatechart):

                executor.compile(self.motion_state_chart)
                try:
                    executor.tick_until_end(timeout=2000)
                except TimeoutError as e:
                    failed_nodes = [
                        node if node.life_cycle_state != LifeCycleValues.DONE else None
                        for node in self.motion_state_chart.nodes
                    ]
                    logger.error(f"Failed Nodes: {failed_nodes}")
                    raise e
            else:
                motion.perform()

    def _execute_for_real(self):
        from giskardpy_ros.python_interface.python_interface import GiskardWrapper

        giskard = GiskardWrapper(self.ros_node)
        giskard.execute(self.motion_state_chart)
