"""Orchestrator — LangGraph StateGraph supervisor that routes between specialist sub-agents."""

import logging
import json
import os
import re
import uuid
from typing import Any, TYPE_CHECKING

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from agentic_llmr.core.interfaces import BaseCognitiveAgent
from agentic_llmr.core.trace import TraceCollector
from agentic_llmr.core.state import RobotAgentState
from agentic_llmr.core.supervisor import make_supervisor_node
from agentic_llmr.core.planner import make_planner_node
from agentic_llmr.core.composer import make_composer_node
from agentic_llmr.agents import SceneQueryAgent, KinematicsAgent, PlanningAgent

logger = logging.getLogger(__name__)

# Maximum supervisor↔specialist super-steps before a run is aborted. Guards
# against a mis-routing supervisor or a non-terminating specialist loop.
_RECURSION_LIMIT = 40


def extract_json_payload(text: str) -> Any:
    """Extract and parse a JSON payload from an agent response.

    Accepts a fenced ```json block or a bare JSON document. Raises ValueError
    when neither yields valid JSON. Shared by the orchestrator and the backend
    so both paths parse identically.
    """
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    candidate = json_match.group(1) if json_match else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("Could not find a valid JSON block in the agent's response.") from exc


class ReActAgent(BaseCognitiveAgent):
    """Orchestrator — LangGraph StateGraph with supervisor routing across specialist sub-agents."""

    def __init__(self, llm: "BaseChatModel"):
        self.llm = llm

        # Per-instance scratchpad directory so concurrent agents/sessions never
        # clobber each other's notes. Reset per query in resolve_action().
        self._scratchpad_dir = os.path.join(
            os.getcwd(), ".agentic_scratchpads", uuid.uuid4().hex[:8]
        )
        os.makedirs(self._scratchpad_dir, exist_ok=True)

        self._scene_agent = SceneQueryAgent(
            llm, scratchpad_path=os.path.join(self._scratchpad_dir, "sdt_scratchpad.md"))
        self._kinematics_agent = KinematicsAgent(
            llm, scratchpad_path=os.path.join(self._scratchpad_dir, "giskard_scratchpad.md"))
        self._planning_agent = PlanningAgent(
            llm, scratchpad_path=os.path.join(self._scratchpad_dir, "pycram_scratchpad.md"))

        builder = StateGraph(RobotAgentState)

        builder.add_node("planner",          make_planner_node(llm))
        builder.add_node("supervisor",       make_supervisor_node(llm))
        builder.add_node("scene_perception", self._scene_node)
        builder.add_node("kinematics",       self._kinematics_node)
        builder.add_node("planning",         self._planning_node)
        builder.add_node("composer",         make_composer_node(llm))

        # START → planner (classify + plan, once) → supervisor (dispatch loop).
        builder.add_edge(START, "planner")
        builder.add_edge("planner", "supervisor")
        # Workers return to the supervisor; the supervisor routes via Command, and
        # routes to the composer on FINISH. The composer frames the answer, then ends.
        builder.add_edge("scene_perception", "supervisor")
        builder.add_edge("kinematics",       "supervisor")
        builder.add_edge("planning",         "supervisor")
        builder.add_edge("composer",         END)

        self.agent_executor = builder.compile()
        logger.debug("[Orchestrator] StateGraph compiled. Nodes: %s",
                     list(self.agent_executor.get_graph().nodes.keys()))

    # ── Specialist node wrappers ────────────────────────────────────────────
    #
    # Each specialist runs in an ISOLATED message context: the subgraph is invoked
    # with a fresh (empty) message list plus its scoped task and the facts it needs.
    # Only a single summary line returns to the supervisor's shared `messages`
    # channel — never the specialist's verbose internal tool transcript. This keeps
    # the supervisor's context bounded (one line per routing step) instead of
    # accumulating every sub-agent's full message history, which otherwise overflows
    # the model context window on multi-step queries.

    @staticmethod
    def _summary_message(result: dict, label: str) -> AIMessage:
        msgs = result.get("messages", [])
        content = msgs[-1].content if msgs else ""
        return AIMessage(content=f"[{label}] {content}")

    def _scene_node(self, state: RobotAgentState, config=None) -> dict:
        result = self._scene_agent.subgraph.invoke({
            "messages":         [],
            "instruction":      state.get("instruction", ""),
            "current_task":     state.get("current_task", ""),
            "template_context": state.get("template_context", ""),
            "scene_facts":      state.get("scene_facts", {}),
        }, config)
        return {
            "messages":    [self._summary_message(result, "scene_perception")],
            "scene_facts": result.get("scene_facts", {}),
        }

    def _kinematics_node(self, state: RobotAgentState, config=None) -> dict:
        result = self._kinematics_agent.subgraph.invoke({
            "messages":         [],
            "instruction":      state.get("instruction", ""),
            "current_task":     state.get("current_task", ""),
            "template_context": state.get("template_context", ""),
            "scene_facts":      state.get("scene_facts", {}),
            "kinematic_facts":  state.get("kinematic_facts", {}),
        }, config)
        return {
            "messages":        [self._summary_message(result, "kinematics")],
            "kinematic_facts": result.get("kinematic_facts", {}),
        }

    def _planning_node(self, state: RobotAgentState, config=None) -> dict:
        result = self._planning_agent.subgraph.invoke({
            "messages":         [],
            "instruction":      state.get("instruction", ""),
            "current_task":     state.get("current_task", ""),
            "template_context": state.get("template_context", ""),
            "kinematic_facts":  state.get("kinematic_facts", {}),
            "action_schema":    state.get("action_schema", {}),
        }, config)
        return {
            "messages":      [self._summary_message(result, "planning")],
            "action_schema": result.get("action_schema", {}),
        }

    def _reset_scratchpads(self) -> None:
        """Clear every specialist's scratchpad so each top-level query starts fresh."""
        for agent in (self._scene_agent, self._kinematics_agent, self._planning_agent):
            path = getattr(agent, "scratchpad_path", None)
            if not path:
                continue
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception as exc:
                logger.debug("Could not reset scratchpad %s: %s", path, exc)

    def resolve_action(self, instruction: str, template_context: str = "") -> str:
        """Run the supervisor graph and return the final message content."""
        self._reset_scratchpads()
        initial_state: RobotAgentState = {
            "messages": [HumanMessage(content=(
                f"Instruction: {instruction}\nContext: {template_context}"
            ))],
            "instruction":      instruction,
            "template_context": template_context,
            "query_kind":       "",
            "playbook":         "",
            "current_task":     "",
            "scene_facts":      {},
            "kinematic_facts":  {},
            "action_schema":    {},
            "final_response":   "",
        }

        collector = TraceCollector()
        config = {"callbacks": [collector], "recursion_limit": _RECURSION_LIMIT}

        logger.debug("[ORCHESTRATOR STARTED]")
        try:
            result = self.agent_executor.invoke(initial_state, config=config)
        except GraphRecursionError:
            logger.warning(
                "[ORCHESTRATOR] Recursion limit (%d) hit before the goal was resolved.",
                _RECURSION_LIMIT,
            )
            self.last_trace = collector
            return (
                "I could not complete this request within the allowed number of reasoning "
                "steps. The query may be ambiguous or require an action the robot cannot "
                "currently perform. Please refine or simplify the request."
            )
        logger.debug("[ORCHESTRATOR FINISHED]")

        self.last_trace = collector

        # The composer is the authoritative final answer; fall back to the last
        # message (stripped of its "[specialist]" prefix) only if it is absent.
        final_response = result.get("final_response", "")
        if final_response:
            return final_response
        final_messages = result.get("messages", [])
        if not final_messages:
            return ""
        return re.sub(r"^\[[^\]]+\]\s*", "", final_messages[-1].content)

    def parse_and_hydrate_action(self, agent_response: str) -> Any:
        """Parse the JSON from the agent's final response and return PyCRAM Action instance(s)."""
        from agentic_llmr.platform.type_bridge import hydrate_action_kwargs
        from agentic_llmr.platform.actions import discover_action_classes

        payload = extract_json_payload(agent_response)
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
