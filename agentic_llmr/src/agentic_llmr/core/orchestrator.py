"""Orchestrator — LangGraph StateGraph supervisor that routes between specialist sub-agents."""

import logging
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START

from agentic_llmr.core.interfaces import BaseCognitiveAgent
from agentic_llmr.core.trace import TraceCollector
from agentic_llmr.core.state import RobotAgentState
from agentic_llmr.core.supervisor import make_supervisor_node
from agentic_llmr.agents import (
    SceneQueryAgent, scene_query_node,
    KinematicsAgent, kinematics_node,
    PlanningAgent, planning_node,
)

logger = logging.getLogger(__name__)


class ReActAgent(BaseCognitiveAgent):
    """Orchestrator — LangGraph StateGraph with supervisor routing across specialist sub-agents."""

    def __init__(self, llm: Any):
        self.llm = llm

        self._scene_agent = SceneQueryAgent(llm)
        self._kinematics_agent = KinematicsAgent(llm)
        self._planning_agent = PlanningAgent(llm)

        builder = StateGraph(RobotAgentState)

        builder.add_node("supervisor", make_supervisor_node(llm))
        builder.add_node(
            "scene_perception",
            lambda s: scene_query_node(s, self._scene_agent),
        )
        builder.add_node(
            "kinematics",
            lambda s: kinematics_node(s, self._kinematics_agent),
        )
        builder.add_node(
            "planning",
            lambda s: planning_node(s, self._planning_agent),
        )

        builder.add_edge(START, "supervisor")

        self.agent_executor = builder.compile()

    def resolve_action(self, instruction: str, template_context: str = "") -> str:
        """Run the supervisor graph and return the final message content.

        Initialises shared state with the instruction and empty fact dicts, then
        invokes the compiled StateGraph. Returns the last assistant message as a string.
        """
        initial_state: RobotAgentState = {
            "messages": [HumanMessage(content=(
                f"Instruction: {instruction}\nContext: {template_context}"
            ))],
            "instruction": instruction,
            "template_context": template_context,
            "scene_facts": {},
            "kinematic_facts": {},
            "action_schema": {},
        }

        collector = TraceCollector()
        config = {"callbacks": [collector]}

        logger.debug("[ORCHESTRATOR STARTED]")
        result = self.agent_executor.invoke(initial_state, config=config)
        logger.debug("[ORCHESTRATOR FINISHED]")

        self.last_trace = collector
        final_messages = result.get("messages", [])
        return final_messages[-1].content if final_messages else ""

    def parse_and_hydrate_action(self, agent_response: str) -> Any:
        """Parse the JSON from the agent's final response and return PyCRAM Action instance(s)."""
        from agentic_llmr.platform.type_bridge import hydrate_action_kwargs
        from agentic_llmr.platform.actions import discover_action_classes

        json_match = re.search(r'```json\s*(.*?)\s*```', agent_response, re.DOTALL)
        if not json_match:
            try:
                payload = json.loads(agent_response)
            except json.JSONDecodeError:
                raise ValueError("Could not find a valid JSON block in the agent's response.")
        else:
            payload = json.loads(json_match.group(1))

        actions = discover_action_classes()

        items = payload if isinstance(payload, list) else [payload]
        instances = []
        for item in items:
            action_type = item.get("action_type")
            parameters = item.get("parameters", {})
            if not action_type:
                raise ValueError("JSON item missing 'action_type'.")
            action_cls = actions.get(action_type)
            if not action_cls:
                raise ValueError(f"Action class '{action_type}' not found in PyCRAM.")
            hydrated_kwargs = hydrate_action_kwargs(action_cls, parameters)
            instances.append(action_cls(**hydrated_kwargs))

        return instances if len(instances) > 1 else instances[0]
