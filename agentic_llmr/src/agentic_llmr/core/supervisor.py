"""Supervisor routing node for the LangGraph multi-agent graph."""
from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import SystemMessage
from langgraph.types import Command

from agentic_llmr.core.state import RobotAgentState, RoutingDecision

logger = logging.getLogger(__name__)

_SUPERVISOR_SYSTEM = """You are the supervisor for a cognitive robot system.
Your only job is to decide which specialist to call next, or FINISH when the
original query is fully answered.

━━━ STEP 1 — CLASSIFY THE QUERY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Read the FIRST HumanMessage (the original instruction). Classify it as:

  ANSWER     — a question asking for information about the scene or robot state.
               The response is natural language. No JSON designator is produced.

  DESIGNATOR — a direct command for the robot to physically act on the world.
               The response must be a fully-resolved JSON action designator.

━━━ STEP 2 — IDENTIFY WHAT IS NEEDED ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before routing, determine what information the query actually requires.

  scene_perception: object locations, types, sizes, orientations, colors, spatial
    relations, surface contents, container contents, furniture joint states,
    placement spots, collision status, support relationships, accessibility.

  kinematics: whether an arm can reach a target pose, which arm to prefer,
    valid grasp approach directions, current gripper/joint/end-effector state.

  planning: the correct PyCRAM action class and its parameter schema,
    and physics simulation of a proposed action.

Only call a specialist if the query genuinely requires its expertise.

━━━ STEP 3 — ROUTE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For ANSWER queries:
  Call only the specialist(s) whose domain directly covers the information needed.
  FINISH as soon as the answer is present in the messages.

For DESIGNATOR queries:
  Specialists must be called in dependency order — scene facts are required before
  kinematics (reachability needs real poses), and kinematic results are required
  before planning (arm selection must be known to build the action schema).
  FINISH only after a complete JSON designator appears in the messages.

━━━ RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- After every specialist response, re-read the original instruction and ask:
  "Is this query now fully answered?" If yes, FINISH immediately.
- Gathering scene data does NOT imply that kinematic planning is now needed.
  Only proceed to kinematics if the original query itself requires arm decisions.
- Never call planning for an ANSWER query — it serves action schema discovery only.
- Never invoke a specialist whose response is already present in the messages
  for the same information. If a specialist has already answered, do not re-call it.
"""


def make_supervisor_node(llm: Any):
    """Return a supervisor node function bound to the given LLM."""
    routing_llm = llm.with_structured_output(RoutingDecision)

    def supervisor(
        state: RobotAgentState,
    ) -> Command[Literal["scene_perception", "kinematics", "planning", "__end__"]]:
        """Route to the next specialist or finish based on conversation state."""
        messages = [SystemMessage(content=_SUPERVISOR_SYSTEM)] + list(state["messages"])
        decision: RoutingDecision = routing_llm.invoke(messages)
        logger.debug("[Supervisor] → %s (%s)", decision.next_agent, decision.reasoning)

        if decision.next_agent == "FINISH":
            return Command(goto="__end__")
        return Command(goto=decision.next_agent)

    return supervisor
