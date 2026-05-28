"""Shared state schema for the LangGraph multi-agent supervisor graph."""
from __future__ import annotations

from typing import Annotated, Any, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


def _merge_dicts(a: dict, b: dict) -> dict:
    """Deep-merge two dicts. Nested dicts are merged recursively; b wins on scalar conflicts."""
    result = dict(a)
    for key, val in b.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _merge_dicts(result[key], val)
        else:
            result[key] = val
    return result


class RobotAgentState(TypedDict):
    """Shared state passed between every node in the supervisor graph."""

    messages: Annotated[list[BaseMessage], add_messages]

    instruction: str
    template_context: str

    # scene_perception writes here; kinematics reads it to skip redundant queries
    scene_facts: Annotated[Dict[str, Any], _merge_dicts]

    # kinematics writes here; planning reads it for arm selection
    kinematic_facts: Annotated[Dict[str, Any], _merge_dicts]

    # planning writes here; orchestrator reads for final designator
    action_schema: Dict[str, Any]


class RoutingDecision(BaseModel):
    """Structured output the supervisor LLM returns to route between agents."""

    next_agent: str = Field(
        description=(
            "Which specialist to call next. "
            "One of: 'scene_perception', 'kinematics', 'planning', 'FINISH'."
        )
    )
    reasoning: str = Field(description="One sentence explaining the routing decision.")
