"""Kinematics Agent — specialist for arm reachability, grasp planning, and robot state."""

from typing import Any
from langgraph.prebuilt import create_react_agent

from agentic_llmr.core.interfaces import SubAgentTool
from agentic_llmr.tools.scratchpad import WriteScratchpadTool, ReadScratchpadTool
from agentic_llmr.tools.kinematics import CheckReachabilityTool, GetGraspPosesTool, CompareArmSuitabilityTool
from agentic_llmr.tools.scene_query import (
    FindObjectsByTypeTool, GetObjectPoseTool,
    GetRobotPoseTool, GetEndEffectorPoseTool, GetGripperStateTool, GetJointStatesTool,
)

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

━━━ TOOL GUIDE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tool guide:
- find_objects_by_type         : resolve a semantic class name (e.g. 'Milk', 'Cereal') to exact
                                  body_name(s) and world-frame positions. Call this first whenever
                                  a query gives you a semantic name rather than an exact body_name.
- get_object_pose              : get the exact 6-DoF world-frame pose of a known body_name.
                                  Call this after find_objects_by_type to get the precise position
                                  needed by check_kinematic_reachability and get_grasp_poses.
- check_kinematic_reachability : checks each arm's reach for a target 3D pose.
                                  Arm reach and shoulder position are derived from the robot model — no hardcoded values.
                                  Always call this before committing to a pick or place position.
                                  Always pass real world-frame coordinates — never pass (0, 0, 0).
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
                                  Call after get_object_orientation is known.
- get_robot_pose               : world-frame position and orientation of the robot base link.
- get_end_effector_pose        : current world-frame pose of the left or right gripper tool frame.
- get_gripper_state            : current opening width and open/closed status of a gripper.
- get_joint_states             : current position and limits of all active robot arm joints.
                                  Use this to inspect arm configuration (e.g., parked vs. extended).
                                  For joints of non-robot objects (drawers, doors), use get_articulated_object_joints via the SDT agent.

Rules:
- If a query gives you a semantic name (e.g. 'cereal box', 'Milk'), ALWAYS resolve it yourself:
    1. call find_objects_by_type to get the body_name and approximate position
    2. call get_object_pose to get the precise world-frame pose
    3. pass that pose to check_kinematic_reachability or get_grasp_poses
  Never pass (0, 0, 0) or a placeholder pose to any tool.
- check_kinematic_reachability returns "LEFT: Reachable/Unreachable | RIGHT: Reachable/Unreachable".
  Use whichever arm is Reachable.
- If both arms are Reachable, call compare_arm_suitability to pick the better one using principled
  scoring (joint limits, lateral alignment, obstacle clearance). Do NOT choose arbitrarily.
- If a target pose is Unreachable for all arms, compute the floor-level navigation goal:
    nav_xy = offset the target (x,y) by 0.6m toward the robot's current position
    nav_z  = 0.0  (the robot drives on the floor — never suggest z > 0 as a nav target)
"""


class KinematicsAgentTool(SubAgentTool):
    """Tool wrapper that lets the orchestrator call the Giskard sub-agent."""
    name: str = "query_kinematics"
    description: str = (
        "Delegate arm planning queries to the Giskard kinematics specialist. "
        "Use this ONLY when planning a pick, place, or grasp action: to check reachability of a target pose, "
        "decide which arm to use, get grasp approach directions, compute a navigation goal when unreachable, "
        "or read the robot's current arm/gripper state. "
        "Do NOT call this for general scene questions — use query_scene_perception instead."
    )

    def _run(self, query: str) -> str:
        print(f"\n  ► [Giskard Agent] {query}")
        result = super()._run(query)
        print(f"  ◄ [Giskard Agent] Done.\n")
        return result


class KinematicsAgent:
    """Kinematics specialist. Holds reachability, grasp, and robot-state tools."""

    def __init__(self, llm: Any):
        self.llm = llm
        self.tools = [
            WriteScratchpadTool(scratchpad_path=_SCRATCHPAD),
            ReadScratchpadTool(scratchpad_path=_SCRATCHPAD),
            FindObjectsByTypeTool(),
            GetObjectPoseTool(),
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
        )

    def as_tool(self) -> KinematicsAgentTool:
        return KinematicsAgentTool(agent=self._agent)
