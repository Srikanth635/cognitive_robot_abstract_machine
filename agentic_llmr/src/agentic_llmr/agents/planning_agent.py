"""Planning Agent — specialist for action schema discovery and physics simulation."""

from typing import Any
from langgraph.prebuilt import create_react_agent

from agentic_llmr.core.interfaces import SubAgentTool
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


class PlanningAgentTool(SubAgentTool):
    """Tool wrapper that lets the orchestrator call the PyCRAM sub-agent."""
    name: str = "query_action_schema"
    description: str = (
        "Delegate to the PyCRAM specialist for action schema discovery or physics simulation. "
        "Use for: (a) finding the correct PyCRAM action class and its parameter schema, or "
        "(b) simulating a fully-specified action to validate it (only when explicitly requested). "
        "Provide a natural language query for schema discovery, or a JSON action description for simulation."
    )

    def _run(self, query: str) -> str:
        print(f"\n  ► [PyCRAM Agent] {query[:120]}")
        result = super()._run(query)
        print(f"  ◄ [PyCRAM Agent] Done.\n")
        return result


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
        )

    def as_tool(self) -> PlanningAgentTool:
        return PlanningAgentTool(agent=self._agent)
