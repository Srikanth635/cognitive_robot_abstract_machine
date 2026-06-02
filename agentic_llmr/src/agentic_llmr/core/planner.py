"""Planner node — one-shot goal classification and advisory decomposition.

Runs once at the start of a run. It classifies the user's goal and produces an
ADVISORY plan (which specialists, in what order) that the supervisor uses as
guidance. The plan is never binding: the supervisor adapts it to the facts that
actually come back, so this anchors decomposition without hardcoding a workflow.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import SystemMessage, HumanMessage

from agentic_llmr.core.state import RobotAgentState, QueryPlan

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM = """You are the planner for a cognitive robot. Given the user's goal,
classify it and produce a short ADVISORY decomposition: the ordered sub-goals and which
specialist handles each. The supervisor uses your plan as guidance and may deviate as real
facts arrive — so keep it grounded and never invent coordinates or values.

━━━ SPECIALISTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  scene       — observes the world and the robot's current state: object poses, sizes,
                relations, surfaces, accessibility, and the robot's own base pose, joint
                positions, end-effector poses, gripper state, held objects.
  kinematics  — computes over the robot's geometry: reachability, which arm suits a target,
                forward kinematics of a named link, inverse kinematics, joint limits,
                self-collision of a configuration. It is given concrete values; it does not observe.
  planning    — turns a physical command into an executable PyCRAM action (class + parameters,
                and physics simulation).

━━━ CANONICAL PATTERNS (adapt — use only the steps the goal needs) ━━━━━━━━

  scene_query        — pure information about the world or robot state.
                       Plan: scene only.
  robot_introspection— about the robot's own geometry/config: where a link is (FK), joint
                       limits, whether a posture self-collides.
                       Plan: kinematics (route scene first ONLY if current joint positions
                       are an input to the computation).
  reachability       — can an arm reach a target / which arm is better.
                       Plan: scene (resolve the target's world pose) → kinematics
                       (reachability; then arm suitability if both arms can reach).
  manipulation       — perform, or produce the action designator for, a physical action
                       (pick up, place).
                       Plan: scene (object pose, accessibility) → kinematics (reachability →
                       IK → self-collision → grasp directions) → planning (action class +
                       parameters). Stop early and report if a step proves infeasible.
  feasibility        — can the robot do X, and if not, why.
                       Plan: scene (resolve target) → kinematics (reachability / limits);
                       report honestly with the blocker and what would change it.

Reachability ("can it reach X") is a KINEMATICS question, distinct from scene accessibility
(blocked/stacked). The world pose of a link that is not the base or a gripper (elbow,
forearm, wrist) is KINEMATICS forward kinematics, not a scene query.
"""


def make_planner_node(llm: "BaseChatModel"):
    """Return a planner node function bound to the given LLM."""
    planning_llm = llm.with_structured_output(QueryPlan)

    def planner(state: RobotAgentState) -> dict:
        goal = state.get("instruction", "")
        ctx = state.get("template_context", "")
        content = f"Goal: {goal}"
        if ctx:
            content += f"\nContext: {ctx}"

        decision: QueryPlan = planning_llm.invoke([
            SystemMessage(content=_PLANNER_SYSTEM),
            HumanMessage(content=content),
        ])
        logger.debug("[Planner] kind=%s | plan=%s", decision.query_kind, decision.plan)
        return {"query_kind": decision.query_kind, "playbook": decision.plan}

    return planner
