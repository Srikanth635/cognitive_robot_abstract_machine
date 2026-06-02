"""Planning Agent — specialist for action schema discovery and physics simulation."""

import logging
from typing import Any, Dict, TYPE_CHECKING
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from agentic_llmr.core.state import _merge_dicts
from agentic_llmr.core.interfaces import (
    ExecuteToolsNode, make_prepare_query, make_call_model, tools_condition,
)

logger = logging.getLogger(__name__)

from agentic_llmr.tools.scratchpad import WriteScratchpadTool, ReadScratchpadTool
from agentic_llmr.tools.planning import ListAvailableActionsTool, GetActionDocumentationTool, SimulateActionTool

_SCRATCHPAD = "pycram_scratchpad.md"

SYSTEM_PROMPT = """You are the PyCRAM Planning specialist. You turn a physical command into
an executable action, working in two modes:
1. Schema discovery — identify the correct PyCRAM action class and describe its parameters.
2. Simulation — execute a proposed action in the physics engine to validate it.

Reason first about which mode the task needs, then choose your tools accordingly. The tool
list below describes capabilities, not a fixed order.

━━━ SCRATCHPAD USAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You have a personal scratchpad (write_scratchpad / read_scratchpad) to document your
work. Use it to:
1. Write the query you received and what you need to accomplish (schema discovery or simulation).
2. Log the action class name and parameter schema once discovered.
3. Document simulation results (success, errors, parameter values used).

━━━ TOOLS AVAILABLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tools available:
- list_available_actions  : lists all registered PyCRAM action classes.
- get_action_documentation: returns the full parameter spec for a named action class.
- simulate_action         : executes an action with given parameters in the physics model.

━━━ HOW TO WORK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- When you are unsure which action class applies, discover it before describing it; when you
  know it, go straight to its schema.
- For a schema task: report the action class name and every parameter (name, type, meaning).
  Describe the schema — do not invent parameter values that were not provided.
- For a simulation task: run the action as specified and report the outcome honestly —
  success, or the exact error. Do not silently retry or paper over a failure.
- GROUND IN GIVEN VALUES. Object poses, the chosen arm, and other concrete facts come from the
  task; use them as given rather than assuming.
"""


# ---------------------------------------------------------------------------
# Subgraph state
# ---------------------------------------------------------------------------

class PlanningAgentState(TypedDict):
    """State for the planning subgraph. Shared keys are passed from/to parent."""
    messages:         Annotated[list[BaseMessage], add_messages]
    kinematic_facts:  Annotated[Dict[str, Any], _merge_dicts]   # READ from parent
    action_schema:    Dict[str, Any]                             # WRITE to parent
    instruction:      str
    current_task:     str
    template_context: str


# ---------------------------------------------------------------------------
# PlanningAgent
# ---------------------------------------------------------------------------

class PlanningAgent:
    """Action schema and simulation specialist with a custom LangGraph subgraph."""

    def __init__(self, llm: "BaseChatModel", scratchpad_path: str = _SCRATCHPAD):
        self.llm = llm
        self.scratchpad_path = scratchpad_path
        self.tools = [
            WriteScratchpadTool(scratchpad_path=scratchpad_path),
            ReadScratchpadTool(scratchpad_path=scratchpad_path),
            ListAvailableActionsTool(),
            GetActionDocumentationTool(),
            SimulateActionTool(),
        ]

        llm_with_tools = llm.bind_tools(self.tools)

        builder = StateGraph(PlanningAgentState)
        builder.add_node("prepare_query", make_prepare_query([
            ("kinematic_facts", "Known kinematic facts (do NOT re-query these)"),
        ]))
        builder.add_node("call_model", make_call_model(llm_with_tools, SYSTEM_PROMPT))
        builder.add_node("execute_tools", ExecuteToolsNode(self.tools, "action_schema", merge=False))

        builder.add_edge(START, "prepare_query")
        builder.add_edge("prepare_query", "call_model")
        builder.add_conditional_edges("call_model", tools_condition,
                                      {"execute_tools": "execute_tools", END: END})
        builder.add_edge("execute_tools", "call_model")

        self.subgraph = builder.compile()
        logger.debug("[PlanningAgent] Subgraph compiled with %d tools.", len(self.tools))
