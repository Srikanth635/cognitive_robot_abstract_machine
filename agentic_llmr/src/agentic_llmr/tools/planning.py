"""Planning Tools — PyCRAM action schema, documentation, and simulation."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Type
from pydantic import BaseModel, Field
from agentic_llmr.core.interfaces import AgenticTool
from agentic_llmr.tools.artifacts import (
    AvailableActionsArtifact,
    ActionDocArtifact,
    SimulationResultArtifact,
)
from agentic_llmr.platform.actions import discover_action_classes, build_action_documentation

logger = logging.getLogger(__name__)


class ListActionsInput(BaseModel):
    """Input schema for ListAvailableActionsTool (no parameters required)."""

    pass


class ListAvailableActionsTool(AgenticTool):
    """Return all registered PyCRAM action class names."""

    name: str = "list_available_actions"
    description: str = "Get a list of all available PyCRAM action classes."
    args_schema: Type[BaseModel] = ListActionsInput

    def _query(self) -> AvailableActionsArtifact:
        logger.debug("[PyCRAM Tool] Listing available actions dynamically...")
        actions = discover_action_classes()
        return AvailableActionsArtifact(action_names=list(actions.keys()))

    def _format(self, artifact: AvailableActionsArtifact) -> str:
        return "\n".join(artifact.action_names)


class ActionDocInput(BaseModel):
    """Input schema for GetActionDocumentationTool."""

    action_name: str = Field(description="The exact name of the PyCRAM action class (e.g., 'PickUpAction').")


class GetActionDocumentationTool(AgenticTool):
    """Return the full parameter schema for a named PyCRAM action class."""

    name: str = "get_action_documentation"
    description: str = "Retrieve the documentation, required parameters, and constraints for a specific PyCRAM action class."
    args_schema: Type[BaseModel] = ActionDocInput

    def _query(self, action_name: str) -> ActionDocArtifact:
        logger.debug(f"[PyCRAM Tool] Fetching documentation for: {action_name}")
        actions = discover_action_classes()
        action_cls = actions.get(action_name)
        if action_cls is None:
            raise ValueError(
                f"Action '{action_name}' not found. "
                "Use list_available_actions to see valid classes."
            )
        doc = build_action_documentation(action_cls)
        return ActionDocArtifact(action_name=action_name, documentation=doc)

    def _format(self, artifact: ActionDocArtifact) -> str:
        return artifact.documentation


# ---------------------------------------------------------------------------
# Tool: SimulateAction
# ---------------------------------------------------------------------------

from agentic_llmr.platform.world import get_active_world


class SimulateActionInput(BaseModel):
    """Input schema for SimulateActionTool."""

    action_type: str = Field(description="The type of action (e.g., 'PickUp', 'Place').")
    parameters: Dict[str, Any] = Field(description="Dictionary of parameters for the action.")


class SimulateActionTool(AgenticTool):
    """Execute a proposed action in the physics engine and report success or the exact error."""

    name: str = "simulate_action"
    description: str = "Simulate an action with the given parameters in the physics engine to verify success before executing in the real world."
    args_schema: Type[BaseModel] = SimulateActionInput

    def _query(self, action_type: str, parameters: Dict[str, Any]) -> SimulationResultArtifact:
        logger.debug(f"[Planning Tool] Simulating {action_type} with parameters: {parameters}")
        actions = discover_action_classes()
        action_cls = actions.get(action_type)
        if not action_cls:
            return SimulationResultArtifact(
                action_type=action_type,
                success=False,
                message=f"Action '{action_type}' is not a valid PyCRAM action class.",
            )
        try:
            from agentic_llmr.platform.type_bridge import hydrate_action_kwargs
            hydrated_kwargs = hydrate_action_kwargs(action_cls, parameters)
        except Exception as e:
            logger.exception("[Planning Tool] Parameter hydration failed for %s", action_type)
            return SimulationResultArtifact(
                action_type=action_type,
                success=False,
                message=f"Parameter hydration error: {e}",
            )
        try:
            world, robot_view = get_active_world()
            action_instance = action_cls(**hydrated_kwargs)
            with world.modify_world():
                action_instance.execute()
            return SimulationResultArtifact(
                action_type=action_type,
                success=True,
                message=f"{action_type} executed perfectly in the physics model without errors.",
            )
        except Exception as e:
            logger.exception("[Planning Tool] Simulation of %s threw an error", action_type)
            return SimulationResultArtifact(
                action_type=action_type,
                success=False,
                message=f"Execution threw an error: {e}",
            )

    def _format(self, artifact: SimulationResultArtifact) -> str:
        prefix = "Simulation Success" if artifact.success else "Simulation Failed"
        return f"{prefix}: {artifact.message}"
