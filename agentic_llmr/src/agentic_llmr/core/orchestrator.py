"""Orchestrator — top-level ReAct agent that decomposes instructions and delegates to specialist sub-agents."""

import logging
import json
import re
from typing import Any
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agentic_llmr.core.interfaces import BaseCognitiveAgent
from agentic_llmr.core.trace import TraceCollector
from agentic_llmr.tools.scratchpad import WriteScratchpadTool, ReadScratchpadTool
from agentic_llmr.agents import SceneQueryAgent, KinematicsAgent, PlanningAgent

logger = logging.getLogger(__name__)

_SCRATCHPAD = "orchestrator_scratchpad.md"

SYSTEM_PROMPT = """You are the Orchestrator Agent for a cognitive robot system.

━━━ OUTPUT CONTRACT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every response is exactly one of two things:

  ANSWER      — plain text. Used when the instruction asks about the world,
                asks for a recommendation, or asks the agent to reason about
                what should be done. No JSON, no action schemas.

  DESIGNATOR  — a fully-resolved JSON action designator with every parameter
                filled in. Used when the instruction is a direct command for
                the robot to act ("Pick up X", "Place X on Y", "Navigate to X").
                The designator is handed to the user for execution — never
                simulated here.

Never produce a partial designator. Never simulate. Never ask the user for
missing values — resolve everything from the tools before outputting.

━━━ DECIDING WHICH OUTPUT TO PRODUCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ask: is the instruction a direct command for the robot to perform a physical action?

  YES → produce a DESIGNATOR.  Follow the resolution pipeline below.
  NO  → produce an ANSWER.     Gather whatever facts are needed from
        query_scene_perception and query_kinematics, then reason over
        them and return plain text. Do NOT call query_action_schema.

━━━ RESOLUTION PIPELINE (DESIGNATOR path only) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 — Schema discovery
  Call query_action_schema to identify the action class and the exact set of
  parameters it requires. Use this only to learn the schema — do not simulate.

Step 2 — Scene fact gathering
  Call query_scene_perception to fill in object poses, surface positions,
  spatial relations, dimensions, and any other world-state parameters the
  schema requires.

Step 3 — Kinematic resolution
  Call query_kinematics with the real world-frame poses retrieved in Step 2.
  a. Call check_kinematic_reachability to determine which arms can reach
     the target. Never pass (0, 0, 0) — always use actual coordinates.
  b. If both arms are Reachable, call compare_arm_suitability.
     Build the nearby_obstacles list by asking query_scene_perception for
     the nearest objects to the target (get_nearest_objects). Pass their
     exact body_name strings — not descriptions or distances.
     Pass [] if no nearby objects were found.
     Use the arm recommended by the suitability score.
  c. If the target is Unreachable for all arms, compute a floor-level
     navigation goal (offset target xy by 0.6 m toward the robot, z = 0)
     and prepend a NavigateAction to the designator.

Step 4 — Assemble and output the designator
  Fill every schema parameter from the facts gathered in Steps 2–3.
  Output the completed JSON and nothing else after it.

━━━ SUB-AGENT ROLES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  query_scene_perception — scene facts: poses, surfaces, sizes, orientations,
                           spatial relations, containment, accessibility
  query_kinematics       — arm reachability, arm selection, grasp poses,
                           robot and gripper state
  query_action_schema    — action class schemas (DESIGNATOR path only,
                           schema discovery only — never simulation)

━━━ DESIGNATOR FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Single action:
```json
{"action_type": "PickUpAction", "parameters": { ... }}
```
Multi-step (navigate then act):
```json
[
  {"action_type": "NavigateAction", "parameters": { ... }},
  {"action_type": "PickUpAction",   "parameters": { ... }}
]
```
"""


class ReActAgent(BaseCognitiveAgent):
    """Orchestrator that delegates perception, kinematics, schema, and execution to sub-agents."""

    def __init__(self, llm: Any):
        self.llm = llm

        # Build specialist sub-agents
        self._scene_query_agent     = SceneQueryAgent(llm)
        self._kinematics_agent = KinematicsAgent(llm)
        self._planning_agent  = PlanningAgent(llm)

        # Orchestrator only holds sub-agent tools + its own scratchpad
        self.tools = [
            WriteScratchpadTool(scratchpad_path=_SCRATCHPAD),
            ReadScratchpadTool(scratchpad_path=_SCRATCHPAD),
            self._scene_query_agent.as_tool(),
            self._kinematics_agent.as_tool(),
            self._planning_agent.as_tool(),
        ]

        self.agent_executor = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
        )

    def resolve_action(self, instruction: str, template_context: str = "") -> str:
        prompt = (
            f"Instruction: {instruction}\n"
            f"Context: {template_context}\n\n"
            "Please resolve this into executable parameters."
        )
        inputs = {"messages": [HumanMessage(content=prompt)]}

        collector = TraceCollector()

        print("\n--- [ORCHESTRATOR STARTED] ---")
        final_message = None
        for s in self.agent_executor.stream(
            inputs, stream_mode="values", config={"callbacks": [collector]}
        ):
            message = s["messages"][-1]
            if hasattr(message, "content") and message.content:
                print(f"[Orchestrator]:\n{message.content}\n")
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    print(f"[Delegate] → {tc['name']}({list(tc['args'].keys())})")
            final_message = message

        self.last_trace = collector
        print("--- [ORCHESTRATOR FINISHED] ---\n")
        return final_message.content if final_message else ""

    def parse_and_hydrate_action(self, agent_response: str) -> Any:
        """Parse the JSON from the agent's final response and return PyCRAM Action instance(s)."""
        from agentic_llmr.resolution.deserializer import hydrate_action_kwargs
        from agentic_llmr.integrations.pycram_adapter import discover_action_classes

        json_match = re.search(r'```json\s*(.*?)\s*```', agent_response, re.DOTALL)
        if not json_match:
            try:
                payload = json.loads(agent_response)
            except json.JSONDecodeError:
                raise ValueError("Could not find a valid JSON block in the agent's response.")
        else:
            payload = json.loads(json_match.group(1))

        actions = discover_action_classes()

        # Support both single action dict and list of actions
        items = payload if isinstance(payload, list) else [payload]
        instances = []
        for item in items:
            action_type = item.get("action_type")
            parameters  = item.get("parameters", {})
            if not action_type:
                raise ValueError("JSON item missing 'action_type'.")
            action_cls = actions.get(action_type)
            if not action_cls:
                raise ValueError(f"Action class '{action_type}' not found in PyCRAM.")
            hydrated_kwargs = hydrate_action_kwargs(action_cls, parameters)
            instances.append(action_cls(**hydrated_kwargs))

        return instances if len(instances) > 1 else instances[0]
