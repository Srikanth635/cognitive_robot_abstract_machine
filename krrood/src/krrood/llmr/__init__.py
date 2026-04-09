"""krrood.llmr — Generic LLM-driven NL-to-action translation framework.

No sdt, pycram, or robot-specific imports anywhere in this package.

Caller integration pattern
--------------------------
1. Register action handlers::

       from krrood.llmr.pipeline.dispatcher import ActionDispatcher, ActionHandler
       ActionDispatcher.register("pick_up", MyPickUpHandler)

2. Build the pipeline::

       from krrood.llmr.pipeline.action_pipeline import ActionPipeline
       from semantic_digital_twin.world_description.world_entity import Body  # caller side

       pipeline = ActionPipeline(
           groundable_type=Body,           # passed in — never imported by llmr
           action_types={"pick_up": "..."},
       )

3. Run instructions::

       from krrood.llmr.execution_loop import ExecutionLoop
       loop = ExecutionLoop(pipeline=pipeline, executor=my_executor)
       results = loop.run(["pick up the cup"])
"""

from krrood.llmr.execution_loop import ExecutionLoop, ExecutionResult, PlanResult
from krrood.llmr.pipeline.action_pipeline import ActionPipeline
from krrood.llmr.pipeline.clarification import ClarificationNeededError, ClarificationRequest
from krrood.llmr.pipeline.dispatcher import ActionDispatcher, ActionHandler, ActionSpec
from krrood.llmr.pipeline.entity_grounder import EntityGrounder, GroundingResult
from krrood.llmr.recovery_handler import RecoveryHandler
from krrood.llmr.task_decomposer import DecomposedPlan, TaskDecomposer
from krrood.llmr.workflows.schemas.common import ActionSlotSchema, EntityDescriptionSchema
from krrood.llmr.workflows.schemas.recovery import RecoverySchema

__all__ = [
    # Core orchestration
    "ExecutionLoop",
    "ExecutionResult",
    "PlanResult",
    # Pipeline
    "ActionPipeline",
    # Dispatcher
    "ActionDispatcher",
    "ActionHandler",
    "ActionSpec",
    # Grounding
    "EntityGrounder",
    "GroundingResult",
    # Clarification
    "ClarificationNeededError",
    "ClarificationRequest",
    # Recovery
    "RecoveryHandler",
    # Decomposition
    "TaskDecomposer",
    "DecomposedPlan",
    # Schemas
    "ActionSlotSchema",
    "EntityDescriptionSchema",
    "RecoverySchema",
]
