"""Kinematics Agent — specialist for arm reachability, grasp planning, and robot state."""

import json
import logging
from typing import Any, Dict, Literal
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from agentic_llmr.core.state import RobotAgentState

logger = logging.getLogger(__name__)
from agentic_llmr.tools.scratchpad import WriteScratchpadTool, ReadScratchpadTool
from agentic_llmr.tools.kinematics import CheckReachabilityTool, GetGraspPosesTool, CompareArmSuitabilityTool
from agentic_llmr.tools.scene_query import GetRobotPoseTool, GetEndEffectorPoseTool, GetGripperStateTool, GetJointStatesTool

_SCRATCHPAD = "giskard_scratchpad.md"

SYSTEM_PROMPT = """You are the Giskard Kinematics Agent — a specialist in robot arm reachability, grasp planning, and robot state.

Your job is to answer queries about arm reachability, which arm to use, grasp approach directions,
and the current physical state of the robot (base pose, end-effector poses, gripper opening, arm joint positions).
Return a concise, actionable summary.

━━━ SCRATCHPAD USAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You have a personal scratchpad (write_scratchpad / read_scratchpad) to document your
reasoning as you work. Use it to:
1. Write your plan at the start: what pose you need to check, which arms to test, etc.
2. Log reachability results as you call check_kinematic_reachability — this helps you
   make the arm selection decision and avoid redundant checks.
3. Document the arm suitability analysis if you call compare_arm_suitability.

Your scratchpad is private to the orchestrator. Update it as you work.

━━━ TOOL GUIDE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The query you receive includes a "Known scene facts" section with pre-resolved object
poses and body names from the scene agent. Use those values directly — do not re-query
the scene for data that is already provided.

Tool guide:
- check_kinematic_reachability : checks each arm's reach for a target 3D pose.
                                  Arm reach and shoulder position are derived from the robot model — no hardcoded values.
                                  Always call this before committing to a pick or place position.
                                  Always pass real world-frame coordinates from scene facts — never pass (0, 0, 0).
- compare_arm_suitability      : ranks both arms by a weighted score across three criteria:
                                  lateral alignment (y-offset), joint limit margin (pan + elevation angles),
                                  and obstacle clearance (segment-to-point distance from arm path).
                                  Call this AFTER check_kinematic_reachability confirms both arms are Reachable.
                                  Pass nearby_obstacles as a list of exact body name strings
                                  (e.g. ['cereal.stl', 'bowl.stl']) — NOT [x,y,z] poses.
                                  The tool resolves positions and filters robot parts automatically.
                                  Pass [] when no nearby objects are available.
- get_grasp_poses              : valid approach directions (TOP/FRONT/LEFT/RIGHT/BACK) for an object.
                                  Requires the exact body_name (e.g., 'milk.stl') — not a semantic class name.
                                  Use the body_name from the scene facts provided in the query.
- get_robot_pose               : world-frame position and orientation of the robot base link.
- get_end_effector_pose        : current world-frame pose of the left or right gripper tool frame.
- get_gripper_state            : current opening width and open/closed status of a gripper.
- get_joint_states             : current position and limits of all active robot arm joints.
                                  Use this to inspect arm configuration (e.g., parked vs. extended).

Rules:
- Target poses are provided in the query via "Known scene facts". Use those values directly
  with check_kinematic_reachability and get_grasp_poses. Never pass (0, 0, 0).
- check_kinematic_reachability returns "LEFT: Reachable/Unreachable | RIGHT: Reachable/Unreachable".
  Use whichever arm is Reachable.
- If both arms are Reachable, call compare_arm_suitability to pick the better one using principled
  scoring (joint limits, lateral alignment, obstacle clearance). Do NOT choose arbitrarily.
- If a target pose is Unreachable for all arms, compute the floor-level navigation goal:
    nav_xy = offset the target (x,y) by 0.6m toward the robot's current position
    nav_z  = 0.0  (the robot drives on the floor — never suggest z > 0 as a nav target)
"""


class KinematicsAgent:
    """Kinematics specialist. Holds reachability, grasp, and robot-state tools."""

    def __init__(self, llm: Any):
        self.llm = llm
        self.tools = [
            WriteScratchpadTool(scratchpad_path=_SCRATCHPAD),
            ReadScratchpadTool(scratchpad_path=_SCRATCHPAD),
            CheckReachabilityTool(),
            CompareArmSuitabilityTool(),
            GetGraspPosesTool(),
            GetRobotPoseTool(),
            GetEndEffectorPoseTool(),
            GetGripperStateTool(),
            GetJointStatesTool(),
        ]
        self._agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
            name="kinematics",
        )


def _extract_kinematic_facts(messages: list) -> Dict[str, Any]:
    """Parse ToolMessage results from the kinematics agent into a structured fact dict."""
    tool_calls_by_id: Dict[str, tuple] = {}
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_by_id[tc["id"]] = (tc["name"], tc["args"])

    facts: Dict[str, Any] = {}
    for msg in messages:
        call_id = getattr(msg, "tool_call_id", None)
        if call_id is None or call_id not in tool_calls_by_id:
            continue
        tool_name, tool_args = tool_calls_by_id[call_id]
        try:
            content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
        except (json.JSONDecodeError, TypeError):
            content = msg.content

        if tool_name == "check_kinematic_reachability":
            facts["reachability"] = content
        elif tool_name == "compare_arm_suitability":
            facts["preferred_arm"] = content
        elif tool_name == "get_grasp_poses":
            body = tool_args.get("body_name", "object")
            facts[f"grasp_poses:{body}"] = content

    return facts


def _build_query(state: RobotAgentState) -> str:
    """Construct the task string from the original instruction and template context."""
    ctx = state.get("template_context", "")
    return f"Instruction: {state['instruction']}\nContext: {ctx}" if ctx else state["instruction"]


def kinematics_node(
    state: RobotAgentState,
    agent: "KinematicsAgent",
) -> Command[Literal["supervisor"]]:
    """LangGraph node: invoke the kinematics react agent, injecting known scene facts."""
    base_query = _build_query(state)
    scene_ctx = ""
    if state.get("scene_facts"):
        facts_str = "\n".join(f"  {k}: {v}" for k, v in state["scene_facts"].items())
        scene_ctx = f"\n\nKnown scene facts (do NOT re-query these):\n{facts_str}"
    enriched = base_query + scene_ctx
    logger.debug("  ► [Giskard Agent] %s", enriched[:120])
    result = agent._agent.invoke({"messages": [HumanMessage(content=enriched)]})
    last_msg = result["messages"][-1]
    kinematic_facts = _extract_kinematic_facts(result["messages"])
    logger.debug("  ◄ [Giskard Agent] Done. Extracted %d kinematic fact entries.", len(kinematic_facts))
    return Command(
        goto="supervisor",
        update={"messages": [last_msg], "kinematic_facts": kinematic_facts},
    )
