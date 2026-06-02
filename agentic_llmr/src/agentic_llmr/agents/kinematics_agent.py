"""Kinematics Agent — specialist for arm reachability, grasp planning, and robot state."""

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
from agentic_llmr.tools.kinematics import (
    CheckReachabilityTool, GetGraspPosesTool, CompareArmSuitabilityTool,
    ComputeForwardKinematicsTool, SolveInverseKinematicsTool,
    GetJointLimitsDetailedTool, CheckSelfCollisionAtConfigTool,
)

_SCRATCHPAD = "giskard_scratchpad.md"

SYSTEM_PROMPT = """You are the Giskard Kinematics Agent — a specialist in robot arm
reachability, grasp planning, and kinematic computation (FK, IK, joint limits, self-collision).

You COMPUTE over the robot's geometry. You do not read the scene or current robot state
yourself — any object poses and current robot state you need are supplied to you in the
task and in the "Known scene facts" section. Use those values directly; never invent
coordinates. Return a concise, actionable summary.

━━━ SCRATCHPAD USAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You have a personal scratchpad (write_scratchpad / read_scratchpad) to document your
reasoning as you work — your plan, intermediate results, and conclusions.

━━━ TOOL CAPABILITIES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each tool does one thing. You decide which tools to call, and in what order, to satisfy
the task — these are capability descriptions, not a prescribed sequence.

- check_kinematic_reachability : binary per-arm reachability for a target 3D world-frame pose.
- compare_arm_suitability      : ranks both arms by a weighted score (lateral alignment,
                                  joint-limit margin, obstacle clearance). Takes a target pose
                                  and a list of nearby obstacle body names ([] if none).
- get_grasp_poses              : valid approach directions (TOP/FRONT/LEFT/RIGHT/BACK) for an
                                  object. Requires the exact body_name (e.g. 'milk.stl').
- compute_forward_kinematics   : world-frame pose of any named link in the kinematic tree.
- solve_inverse_kinematics     : the actual joint configuration that places an arm's tool frame
                                  at a target world-frame pose (distinct from the binary
                                  reachability check, which returns only yes/no).
- get_joint_limits_detailed    : position, velocity, and acceleration limits per joint for an arm —
                                  and the source of the exact joint names other tools require.
- check_self_collision_at_config: sets the robot to a given joint configuration (a dict of joint
                                  name → radians), checks for robot body-to-body (self) collisions,
                                  then restores state. Use the joint names from get_joint_limits_detailed.

━━━ HOW TO WORK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Reason first about what the task is really asking, then choose the tools that answer it.
  A reachability question, a "where is this link" question, and a "what configuration"
  question call for different tools — pick what fits.
- GROUND IN GIVEN VALUES. Target poses, body names, and current state come from the task and
  the "Known scene facts" section. Use those exact values; never invent coordinates.
- REPORT HONESTLY. If a target is unreachable for every arm, or IK does not converge, say so
  plainly and report what the computation actually returned (e.g. per-arm distances). Do not
  fabricate a configuration or a fallback the task did not ask for.
"""


# ---------------------------------------------------------------------------
# Subgraph state
# ---------------------------------------------------------------------------

class KinematicsAgentState(TypedDict):
    """State for the kinematics subgraph. Shared keys are passed from/to parent."""
    messages:         Annotated[list[BaseMessage], add_messages]
    scene_facts:      Annotated[Dict[str, Any], _merge_dicts]   # READ from parent
    kinematic_facts:  Annotated[Dict[str, Any], _merge_dicts]   # WRITE to parent
    instruction:      str
    current_task:     str
    template_context: str


# ---------------------------------------------------------------------------
# KinematicsAgent
# ---------------------------------------------------------------------------

class KinematicsAgent:
    """Kinematics specialist with a custom LangGraph subgraph."""

    def __init__(self, llm: "BaseChatModel", scratchpad_path: str = _SCRATCHPAD):
        self.llm = llm
        self.scratchpad_path = scratchpad_path
        self.tools = [
            WriteScratchpadTool(scratchpad_path=scratchpad_path),
            ReadScratchpadTool(scratchpad_path=scratchpad_path),
            CheckReachabilityTool(),
            CompareArmSuitabilityTool(),
            GetGraspPosesTool(),
            ComputeForwardKinematicsTool(),
            SolveInverseKinematicsTool(),
            GetJointLimitsDetailedTool(),
            CheckSelfCollisionAtConfigTool(),
        ]

        llm_with_tools = llm.bind_tools(self.tools)

        builder = StateGraph(KinematicsAgentState)
        builder.add_node("prepare_query", make_prepare_query([
            ("scene_facts", "Known scene facts (do NOT re-query these)"),
        ]))
        builder.add_node("call_model", make_call_model(llm_with_tools, SYSTEM_PROMPT))
        builder.add_node("execute_tools", ExecuteToolsNode(self.tools, "kinematic_facts"))

        builder.add_edge(START, "prepare_query")
        builder.add_edge("prepare_query", "call_model")
        builder.add_conditional_edges("call_model", tools_condition,
                                      {"execute_tools": "execute_tools", END: END})
        builder.add_edge("execute_tools", "call_model")

        self.subgraph = builder.compile()
        logger.debug("[KinematicsAgent] Subgraph compiled with %d tools.", len(self.tools))
