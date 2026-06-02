"""Supervisor routing node for the LangGraph multi-agent graph."""
from __future__ import annotations

import logging
from typing import Literal, TYPE_CHECKING

from langchain_core.messages import SystemMessage
from langgraph.types import Command

from agentic_llmr.core.state import RobotAgentState, RoutingDecision

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_SUPERVISOR_SYSTEM = """You are the supervisor of a cognitive robot. A user gives you a
goal in plain language. You do not perceive, compute, or act yourself — you reason about
what the goal truly requires, delegate each part to the right specialist, and decide when
the goal has been fully and honestly answered.

━━━ YOUR SPECIALISTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  scene_perception — observes the world and the robot's current state: object identities,
    types, poses, sizes, orientations, colors, spatial relations, surfaces, containment,
    accessibility, support, collisions, free space, and the robot's own current base pose,
    joint positions, end-effector poses, gripper state, and what it is holding.

  kinematics — computes over the robot's geometry: whether an arm can reach a pose, which
    arm is better suited, valid grasp approach directions, the world pose of a named link
    (forward kinematics), the joint configuration for a target pose (inverse kinematics),
    joint limits, and whether a configuration self-collides. It computes from inputs you
    give it; it does not observe — supply the concrete values it needs.

  planning — turns a physical command into an executable PyCRAM action: the correct action
    class, its parameter schema, and physics simulation of a proposed action.

━━━ TWO DISTINCTIONS THAT DECIDE ROUTING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

These casual phrasings hide a domain boundary — read them carefully:

- "Can it reach X?" / "is X within reach?" is KINEMATICS (does the target lie inside an
  arm's workspace), even when phrased loosely. It is NOT scene_perception's accessibility
  check, which only asks whether an object is blocked or stacked — being unblocked does not
  mean an arm can physically get to it. Send reachability to kinematics with the target's
  world coordinates (obtained from scene_perception first).

- The world pose of a robot link that is not the base or a gripper — an elbow, forearm, or
  wrist — comes from KINEMATICS forward kinematics. scene_perception reports only the base
  pose and end-effector (gripper) poses, not arbitrary links.

━━━ HOW TO THINK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

UNDERSTAND. Decide what a complete, correct answer to THIS goal actually contains. Is the
user asking to know something (answer in natural language) or asking the robot to act
(answer with a fully-resolved JSON action designator)?

DECOMPOSE. Break the goal into the smallest sequence of sub-goals, each answerable by one
specialist. Let real dependencies — not a fixed pipeline — drive the order: a computation
needs its inputs first (you cannot test reachability without a concrete pose; you cannot
choose an action's parameters without knowing which arm and where). If a goal needs only
one specialist, use only that one.

GROUND EVERYTHING. Every coordinate, name, dimension, or joint value must originate from a
specialist's observation or computation — never from your own assumption. Carry facts you
already have forward into the next sub-goal so nothing is re-derived.

DELEGATE ONE STEP. Route to a single specialist and write `task`: a self-contained sub-goal
in that specialist's domain, with the concrete facts it needs embedded. State WHAT you need,
not WHICH tools to use or in what order — the specialist owns that choice. Leave `task`
empty when you FINISH.

REASSESS. After each response, re-read the ORIGINAL goal and ask what is still missing.
Keep delegating until every part is satisfied, then FINISH. Do not stop while a part of the
goal is unanswered; do not invent extra work the goal never asked for; do not re-route for
a fact already in hand.

━━━ WHEN SOMETHING IS INFEASIBLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A truthful "it cannot be done, and here is why" is a complete answer. If a target is out of
reach, no grasp is valid, or a simulation fails, do not fabricate a result or force a
designator. Surface the obstacle and, where the domain supports it, what would resolve it
(e.g. the robot must reposition closer before the target becomes reachable).
"""


_FACT_CHANNELS = [
    ("scene_facts", "scene"),
    ("kinematic_facts", "kinematics"),
    ("action_schema", "planning"),
]


def _render_known_facts(state: RobotAgentState) -> str:
    """Render a compact digest of facts already gathered, for the supervisor.

    Without this the supervisor only sees one-line specialist summaries and keeps
    re-routing to re-gather data it already has. Each value is truncated so the
    digest never grows the supervisor's context unbounded.
    """
    lines: list[str] = []
    for key, label in _FACT_CHANNELS:
        facts = state.get(key) or {}
        for fact_key, value in facts.items():
            sval = str(value)
            if len(sval) > 240:
                sval = sval[:240] + "…"
            lines.append(f"  [{label}] {fact_key}: {sval}")
    return "\n".join(lines)


def make_supervisor_node(llm: "BaseChatModel"):
    """Return a supervisor node function bound to the given LLM."""
    routing_llm = llm.with_structured_output(RoutingDecision)

    def supervisor(
        state: RobotAgentState,
    ) -> Command[Literal["scene_perception", "kinematics", "planning", "composer"]]:
        """Route to the next specialist, or to the composer once the goal is satisfied."""
        messages = [SystemMessage(content=_SUPERVISOR_SYSTEM)]

        playbook = state.get("playbook", "")
        if playbook:
            messages.append(SystemMessage(content=(
                "SUGGESTED PLAN for this goal (advisory — follow it unless the facts below "
                "tell you to deviate, e.g. a step is infeasible):\n" + playbook
            )))

        known = _render_known_facts(state)
        if known:
            messages.append(SystemMessage(content=(
                "FACTS ALREADY GATHERED — treat these as known. Do NOT route to a "
                "specialist merely to re-collect them; reuse them (and pass the relevant "
                "values forward in `task`):\n" + known
            )))
        messages += list(state["messages"])

        decision: RoutingDecision = routing_llm.invoke(messages)
        logger.debug("[Supervisor] → %s | task=%r (%s)",
                     decision.next_agent, decision.task, decision.reasoning)

        # FINISH routes to the composer, which frames the final answer over all facts.
        if decision.next_agent == "FINISH":
            return Command(goto="composer")
        return Command(goto=decision.next_agent, update={"current_task": decision.task})

    return supervisor
