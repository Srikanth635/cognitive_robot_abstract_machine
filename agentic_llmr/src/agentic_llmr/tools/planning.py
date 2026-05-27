"""Planning Tools — PyCRAM action schema, documentation, and simulation."""

from typing import Any, Dict, List, Type
from pydantic import BaseModel, Field
from agentic_llmr.core.interfaces import AgenticTool
from agentic_llmr.platform.actions import discover_action_classes, build_action_documentation

class ListActionsInput(BaseModel):
    pass

class ListAvailableActionsTool(AgenticTool):
    name: str = "list_available_actions"
    description: str = "Get a list of all available PyCRAM action classes."
    args_schema: Type[BaseModel] = ListActionsInput

    def _run(self) -> List[str]:
        try:
            print("[PyCRAM Tool] Listing available actions dynamically...")
            actions = discover_action_classes()
            return list(actions.keys())
        except Exception as e:
            return [self._handle_error(e)]

class ActionDocInput(BaseModel):
    action_name: str = Field(description="The exact name of the PyCRAM action class (e.g., 'PickUpAction').")

class GetActionDocumentationTool(AgenticTool):
    name: str = "get_action_documentation"
    description: str = "Retrieve the documentation, required parameters, and constraints for a specific PyCRAM action class."
    args_schema: Type[BaseModel] = ActionDocInput

    def _run(self, action_name: str) -> str:
        try:
            print(f"[PyCRAM Tool] Fetching documentation for: {action_name}")
            
            actions = discover_action_classes()
            action_cls = actions.get(action_name)
            
            if action_cls is None:
                return f"Error: Action '{action_name}' not found. Please use list_available_actions to see valid classes."
            
            return build_action_documentation(action_cls)
                
        except Exception as e:
            return self._handle_error(e)


# ---------------------------------------------------------------------------
# Tool: SimulateAction (merged from execution.py)
# ---------------------------------------------------------------------------

from agentic_llmr.platform.world import get_active_world

class SimulateActionInput(BaseModel):
    action_type: str = Field(description="The type of action (e.g., 'PickUp', 'Place').")
    parameters: Dict[str, Any] = Field(description="Dictionary of parameters for the action.")

class SimulateActionTool(AgenticTool):
    name: str = "simulate_action"
    description: str = "Simulate an action with the given parameters in the physics engine to verify success before executing in the real world."
    args_schema: Type[BaseModel] = SimulateActionInput

    def _run(self, action_type: str, parameters: Dict[str, Any]) -> str:
        try:
            print(f"[Planning Tool] Simulating {action_type} with parameters: {parameters}")
            actions = discover_action_classes()
            action_cls = actions.get(action_type)
            if not action_cls:
                return f"Simulation Failed: Action '{action_type}' is not a valid PyCRAM action class."
            try:
                from agentic_llmr.platform.type_bridge import hydrate_action_kwargs
                hydrated_kwargs = hydrate_action_kwargs(action_cls, parameters)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"Simulation Failed: Parameter hydration error: {e}"
            try:
                world, robot_view = get_active_world()
                action_instance = action_cls(**hydrated_kwargs)
                with world.modify_world():
                    action_instance.execute()
                return f"Simulation Success: {action_type} executed perfectly in the physics model without errors."
            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"Simulation Failed: Execution threw an error: {e}"
        except Exception as e:
            return self._handle_error(e)
