"""Shared state schema for the LangGraph multi-agent supervisor graph."""
from __future__ import annotations

from typing import Annotated, Any, Dict, Literal
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

    # Set once by the planner node at the start of a run.
    query_kind: str   # classification of the goal (see QueryPlan.query_kind)
    playbook: str     # advisory decomposition the supervisor uses as guidance

    # The scoped sub-task the supervisor hands the next specialist (its slice of
    # the original instruction). Specialists read this instead of the full query.
    current_task: str

    # scene_perception writes here; kinematics reads it to skip redundant queries
    scene_facts: Annotated[Dict[str, Any], _merge_dicts]

    # kinematics writes here; planning reads it for arm selection
    kinematic_facts: Annotated[Dict[str, Any], _merge_dicts]

    # planning writes here; orchestrator reads for final designator
    action_schema: Dict[str, Any]

    # Written once by the composer node — the final synthesized answer.
    final_response: str


class QueryPlan(BaseModel):
    """Structured output the planner LLM returns once at the start of a run."""

    query_kind: Literal[
        "scene_query", "robot_introspection", "reachability",
        "manipulation", "feasibility", "other",
    ] = Field(
        description="The category of the goal, used to anchor decomposition and final framing."
    )
    plan: str = Field(
        description=(
            "A concise, ADVISORY decomposition: the ordered sub-goals and which specialist "
            "(scene / kinematics / planning) handles each. The supervisor follows this as "
            "guidance but may deviate as real facts arrive. Never invent coordinates here."
        )
    )
    reasoning: str = Field(description="One sentence explaining the classification.")


class RoutingDecision(BaseModel):
    """Structured output the supervisor LLM returns to route between agents."""

    next_agent: Literal["scene_perception", "kinematics", "planning", "FINISH"] = Field(
        description=(
            "Which specialist to call next. "
            "Must be exactly one of: 'scene_perception', 'kinematics', 'planning', 'FINISH'."
        )
    )
    task: str = Field(
        description=(
            "A self-contained instruction for the chosen specialist, covering ONLY the "
            "portion of the original query within that specialist's domain. Embed any "
            "concrete facts already gathered (e.g. resolved world-frame coordinates) the "
            "specialist will need so it never re-derives them. REQUIRED whenever you route "
            "to a specialist — an empty task forces the specialist to redo the entire query "
            "from scratch. Use the literal string 'done' when next_agent is 'FINISH'."
        ),
    )
    reasoning: str = Field(description="One sentence explaining the routing decision.")
