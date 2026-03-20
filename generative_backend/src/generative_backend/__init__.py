"""Generative backend: LLM-driven NL → PartialDesignator → FullAction pipeline."""

from .pipeline.action_pipeline import ActionPipeline
from .pipeline.action_dispatcher import ActionDispatcher, ActionHandler, WorldContext
from .pipeline.clarification import (
    ArmCapacityError,
    ArmCapacityRequest,
    ClarificationNeededError,
    ClarificationRequest,
)
from .pipeline.entity_grounder import EntityGrounder, GroundingResult, ground_entity
from .planning.motion_precondition_planner import (
    ExecutionState,
    MotionPreconditionPlanner,
    PreconditionProvider,
    PreconditionResult,
)
from .execution_loop import ExecutionLoop, ExecutionResult
from .recovery_handler import RecoveryHandler, RecoveryAttemptResult
from .task_decomposer import DecomposedPlan, TaskDecomposer
from .world_setup import load_pr2_apartment_world

__all__ = [
    # Pipeline
    "ActionPipeline",
    "ActionDispatcher",
    "ActionHandler",
    "WorldContext",
    "ArmCapacityError",
    "ArmCapacityRequest",
    "ClarificationNeededError",
    "ClarificationRequest",
    "EntityGrounder",
    "GroundingResult",
    "ground_entity",
    # Planning
    "ExecutionState",
    "MotionPreconditionPlanner",
    "PreconditionProvider",
    "PreconditionResult",
    # Orchestration
    "ExecutionLoop",
    "ExecutionResult",
    "RecoveryHandler",
    "RecoveryAttemptResult",
    "DecomposedPlan",
    "TaskDecomposer",
    # World setup
    "load_pr2_apartment_world",
]