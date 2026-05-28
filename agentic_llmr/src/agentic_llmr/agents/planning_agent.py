"""Planning Agent — specialist for action schema discovery and physics simulation."""

import logging
from typing import Any, Literal
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from agentic_llmr.core.state import RobotAgentState

logger = logging.getLogger(__name__)
from agentic_llmr.tools.scratchpad import WriteScratchpadTool, ReadScratchpadTool
from agentic_llmr.tools.planning import ListAvailableActionsTool, GetActionDocumentationTool
from agentic_llmr.tools.planning import SimulateActionTool

_SCRATCHPAD = "pycram_scratchpad.md"

SYSTEM_PROMPT = """You are the PyCRAM Agent — a specialist in action schemas and physics simulation.

You have two responsibilities:
1. Schema discovery: identify the correct PyCRAM action class and its required parameters.
2. Simulation: execute a proposed action in the physics engine to validate it.

━━━ SCRATCHPAD USAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You have a personal scratchpad (write_scratchpad / read_scratchpad) to document your
work. Use it to:
1. Write the query you received and what you need to accomplish (schema discovery or simulation).
2. Log the action class name and parameter schema once discovered.
3. Document simulation results (success, errors, parameter values used).

Your scratchpad is private to the orchestrator. Update it as you work.

━━━ TOOLS AVAILABLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tools available:
- list_available_actions  : lists all registered PyCRAM action classes.
- get_action_documentation: returns the full parameter spec for a named action class.
- simulate_action         : executes an action with given parameters in the physics model.

Rules:
- For schema queries: call list_available_actions first if unsure of the class name.
  Return the action class name and a clear description of every parameter (name, type, meaning).
  Do not guess parameter values — only describe the schema.
- For simulation requests: run the action exactly as instructed. Report success or the exact
  error message. Do not retry unless explicitly asked.
"""


class PlanningAgent:
    """Action schema and simulation specialist."""

    def __init__(self, llm: Any):
        self.llm = llm
        self.tools = [
            WriteScratchpadTool(scratchpad_path=_SCRATCHPAD),
            ReadScratchpadTool(scratchpad_path=_SCRATCHPAD),
            ListAvailableActionsTool(),
            GetActionDocumentationTool(),
            SimulateActionTool(),
        ]
        self._agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
            name="planning",
        )


def _build_query(state: RobotAgentState) -> str:
    """Construct the task string from the original instruction and template context."""
    ctx = state.get("template_context", "")
    return f"Instruction: {state['instruction']}\nContext: {ctx}" if ctx else state["instruction"]


def planning_node(
    state: RobotAgentState,
    agent: "PlanningAgent",
) -> Command[Literal["supervisor"]]:
    """LangGraph node: invoke the PyCRAM planning react agent."""
    base_query = _build_query(state)
    kin_ctx = ""
    if state.get("kinematic_facts"):
        facts_str = "\n".join(f"  {k}: {v}" for k, v in state["kinematic_facts"].items())
        kin_ctx = f"\n\nKnown kinematic facts (do NOT re-query these):\n{facts_str}"
    enriched = base_query + kin_ctx
    logger.debug("  ► [PyCRAM Agent] %s", enriched[:120])
    result = agent._agent.invoke({"messages": [HumanMessage(content=enriched)]})
    last_msg = result["messages"][-1]
    logger.debug("  ◄ [PyCRAM Agent] Done.")
    return Command(
        goto="supervisor",
        update={"messages": [last_msg]},
    )
