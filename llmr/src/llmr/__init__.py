# ── Monkey-patch: fix ActionNode.collect_motions ─────────────────────────────
# Bug in pycram/plans/plan_node.py: collect_motions() uses self.descendants (BFS
# over all levels), so nested ActionNode motions (e.g. ReachAction's MoveTCP calls)
# get collected into the parent PickUpAction's MSC and re-executed by Giskard,
# causing the arm to back away from the object after grasping ("unlift" artefact).
# Fix: bounded BFS — descend into wrapper nodes but stop at nested ActionNodes,
# since those already executed their own MSC during perform().
from collections import deque as _deque
from pycram.plans.plan_node import ActionNode as _ActionNode, MotionNode as _MotionNode


def _collect_motions_direct_only(self):
    motions = []
    queue = _deque(self.children)
    while queue:
        node = queue.popleft()
        if isinstance(node, _MotionNode):
            motions.append(node.motion.motion_chart)
        elif not isinstance(node, _ActionNode):
            queue.extend(node.children)
        # ActionNode boundary — skip; it owns its motions
    return motions


_ActionNode.collect_motions = _collect_motions_direct_only
# ─────────────────────────────────────────────────────────────────────────────

from llmr.pipeline.action_pipeline import ActionPipeline
from llmr.pipeline.action_dispatcher import ActionDispatcher, ActionHandler, WorldContext
from llmr.pipeline.clarification import (
    ArmCapacityError,
    ArmCapacityRequest,
    ClarificationNeededError,
    ClarificationRequest,
)
from llmr.pipeline.entity_grounder import EntityGrounder, GroundingResult, ground_entity
from llmr.planning.motion_precondition_planner import (
    ExecutionState,
    MotionPreconditionPlanner,
    PreconditionProvider,
    PreconditionResult,
)
from llmr.execution_loop import ExecutionLoop, ExecutionResult
from llmr.recovery_handler import RecoveryHandler, RecoveryAttemptResult
from llmr.task_decomposer import DecomposedPlan, TaskDecomposer
from llmr.world_setup import load_pr2_apartment_world

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
