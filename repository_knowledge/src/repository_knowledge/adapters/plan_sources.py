"""
Coraplex source selections for generating simulation plans.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from repository_knowledge.assistance import ContextReference, ContextRole

# %% repository declaration identities


class PlanningSymbol(StrEnum):
    """
    The coraplex contracts and reference scenario selected for planning context.
    """

    ACTION = "coraplex.robot_plans.actions.base.ActionDescription"
    ACTION_PLAN = "coraplex.robot_plans.actions.base.ActionDescription.action_plan"
    CONTEXT = "coraplex.datastructures.dataclasses.Context"
    PLAN = "coraplex.plans.plan.Plan"
    PERFORM_PLAN = "coraplex.plans.plan.Plan.perform"
    SEQUENTIAL = "coraplex.plans.factories.sequential"
    MAKE_NODE = "coraplex.plans.factories.make_node"
    DEFERRED_EXECUTION = "coraplex.plans.executables.UnderspecifiedExecutable"
    CANDIDATE_LIMIT = "coraplex.plans.underspecified.UnderspecifiedNode.candidate_limit"
    DEMONSTRATION = "coraplex.demonstrations.RobotDemonstration"
    RUN_DEMONSTRATION = "coraplex.demonstrations.RobotDemonstration.run"
    SIMULATED_DEMONSTRATION = (
        "coraplex.demos.coraplex_bullet_world_demo.demo.BulletWorldDemonstration"
    )
    SIMULATION_WORLD = "coraplex.demos.coraplex_bullet_world_demo.demo.BulletWorldDemonstration.build_simulated_world"
    SCENE_POPULATION = "coraplex.demos.coraplex_bullet_world_demo.demo.BulletWorldDemonstration.populate_scene"
    PLAN_CONTEXT = "coraplex.demos.coraplex_bullet_world_demo.demo.BulletWorldDemonstration.build_context"
    DEMONSTRATION_PLAN = "coraplex.demos.coraplex_bullet_world_demo.demo.BulletWorldDemonstration.build_plan"
    SIMULATION_ENTRY = "coraplex.demos.coraplex_bullet_world_demo.demo.main"


# %% static reference selection


@dataclass
class CoraplexPlanningSources:
    """
    Select source contracts and an existing simulation scenario without importing them.
    """

    def select(self) -> list[ContextReference]:
        """
        Return declaration identities whose relevance must be checked for the task.
        """
        return [
            ContextReference(ContextRole.INTERFACE, PlanningSymbol.ACTION.value),
            ContextReference(ContextRole.INTERFACE, PlanningSymbol.ACTION_PLAN.value),
            ContextReference(ContextRole.INTERFACE, PlanningSymbol.CONTEXT.value),
            ContextReference(ContextRole.INTERFACE, PlanningSymbol.PLAN.value),
            ContextReference(ContextRole.INTERFACE, PlanningSymbol.PERFORM_PLAN.value),
            ContextReference(ContextRole.INTERFACE, PlanningSymbol.SEQUENTIAL.value),
            ContextReference(ContextRole.INTERFACE, PlanningSymbol.MAKE_NODE.value),
            ContextReference(
                ContextRole.INTERFACE, PlanningSymbol.DEFERRED_EXECUTION.value
            ),
            ContextReference(
                ContextRole.INTERFACE, PlanningSymbol.CANDIDATE_LIMIT.value
            ),
            ContextReference(ContextRole.INTERFACE, PlanningSymbol.DEMONSTRATION.value),
            ContextReference(
                ContextRole.EXAMPLE, PlanningSymbol.RUN_DEMONSTRATION.value
            ),
            ContextReference(
                ContextRole.EXAMPLE, PlanningSymbol.SIMULATED_DEMONSTRATION.value
            ),
            ContextReference(
                ContextRole.EXAMPLE, PlanningSymbol.SIMULATION_WORLD.value
            ),
            ContextReference(
                ContextRole.EXAMPLE, PlanningSymbol.SCENE_POPULATION.value
            ),
            ContextReference(ContextRole.EXAMPLE, PlanningSymbol.PLAN_CONTEXT.value),
            ContextReference(
                ContextRole.EXAMPLE, PlanningSymbol.DEMONSTRATION_PLAN.value
            ),
            ContextReference(
                ContextRole.EXAMPLE, PlanningSymbol.SIMULATION_ENTRY.value
            ),
        ]
