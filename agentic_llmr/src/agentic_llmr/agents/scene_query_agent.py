"""Scene Query Agent — specialist for querying the semantic scene graph."""

import json
import logging
from typing import Any, Dict, Literal
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from agentic_llmr.core.state import RobotAgentState

logger = logging.getLogger(__name__)
from agentic_llmr.tools.scratchpad import WriteScratchpadTool, ReadScratchpadTool
from agentic_llmr.tools.scene_query import (
    # Segment 1 — World Inventory & Taxonomy
    GetSceneObjectsTool,
    GetSemanticAnnotationsTool,
    FindObjectsByTypeTool,
    GetObjectTypeTool,
    ClassifyObjectsByRoleTool,
    # Segment 2 — Geometric & Spatial Properties
    GetObjectPoseTool,
    GetObjectDimensionsTool,
    GetObjectOrientationTool,
    GetObjectColorTool,
    GetSpatialRelationTool,
    GetNearestObjectsTool,
    GetObjectsOnSurfaceTool,
    SortObjectsBySizeTool,
    # Segment 3 — Structural & Topological Relations
    GetArticulatedObjectJointsTool,
    GetContainedItemsTool,
    # Segment 4 — Functional State & Affordances (stubs)
    GetObjectStateTool,
    GetObjectAffordancesTool,
    # Segment 5 — Robot & Interaction State
    GetJointStatesTool,
    GetRobotPoseTool,
    GetEndEffectorPoseTool,
    GetGripperStateTool,
    GetHeldObjectTool,
    # Segment 6 — Collision, Free Space & Placement
    CheckSceneCollisionsTool,
    GetFreePlacementSpotsTool,
    WouldCollideAtPoseTool,
    # Segment 7 — Accessibility & Preconditions
    IsAccessibleTool,
    # Segment 8 — Causal & Consequence Reasoning
    GetSupportingObjectTool,
    GetObjectsSupportedByTool,
)

_SCRATCHPAD = "sdt_scratchpad.md"

SYSTEM_PROMPT = """You are the Scene Perception Agent — a specialist in reading the robot's semantic world model.

Your only job is to answer queries about the current scene by calling your perception tools.
Return a concise, factual summary of what the tools report.

━━━ SCRATCHPAD USAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You have a personal scratchpad (write_scratchpad / read_scratchpad) to document your
reasoning as you work. Use it to:
1. Briefly write your understanding of the query at the start.
2. Log key facts and results as you call tools — this helps you avoid redundant calls
   and reason over the accumulated information.
3. Track your progress through multi-step reasoning chains.

Your scratchpad is private to the orchestrator. Update it as you work. It is always
safe to clear it when starting fresh.

━━━ TOOL REFERENCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WORLD INVENTORY
  list_all_objects          First call for any scene overview. Returns body_name, semantic
                            type, and position of every task-relevant object. Use the
                            body_name values returned here as input to all other tools.
  list_object_types         Type catalog. Use only to discover valid class names when a
                            type search returns no match.
  find_objects_by_type      Convert a semantic name (e.g. 'Milk', 'Table') to exact
                            body_name(s) and world-frame positions.
  get_object_type           Full type-inheritance hierarchy for a known body_name. Reveals
                            whether the object is a container, surface, food item, etc.
  classify_objects_by_role  Scene-wide structural map: which objects are surfaces,
                            articulated fixtures, movable objects, or robot parts.

GEOMETRIC & SPATIAL PROPERTIES
  get_object_pose           Exact 6-DoF pose (position + orientation quaternion) of a
                            known body_name.
  get_object_dimensions     Bounding box (width, depth, height in metres) and pre-computed
                            volume (m³).
  get_object_orientation    Whether an object is upright, inverted, or lying on its side.
                            Returns tilt angle and roll/pitch/yaw. Use before planning a grasp.
  get_object_color          Visual color of an object from its geometry. Use to disambiguate
                            objects of the same type ('the red cup vs the blue cup').
  get_spatial_relation      Direction and Euclidean distance between two objects, expressed
                            in the reference object's local frame. Use 'robot' as the
                            reference for robot-centric directions.
  get_nearest_objects       Up to N closest objects to a reference body, sorted by distance.
                            Pass radius_m to restrict results to a specific range.
  get_objects_on_surface    All objects currently resting on a named surface (Table, Shelf,
                            CounterTop, etc.).
  sort_objects_by_size      Objects of a semantic type ranked by volume (largest first by
                            default). Use to resolve references like 'the large bottle'.

STRUCTURAL & TOPOLOGICAL RELATIONS
  get_articulated_object_joints   Controllable joints of non-robot articulated objects
                                  (drawer, fridge, cabinet, door): type, position, limits.
                                  Use before planning an open or close action.
  get_contained_items             Objects inside a container (fridge interior, drawer
                                  cavity, bowl, box). Uses semantic list when available,
                                  geometric containment otherwise.

FUNCTIONAL STATE  [stubs — not yet implemented]
  get_object_state          Returns a not-implemented message. Will expose temperature,
                            fill level, power state once SDT models dynamic attributes.
  get_object_affordances    Returns a not-implemented message. Will expose what actions
                            can be performed on an object once SDT models affordances.

ROBOT & INTERACTION STATE
  get_joint_states          All robot arm joint positions and limits (not furniture joints).
  get_robot_pose            Robot base 6-DoF pose in world frame.
  get_end_effector_pose     Current gripper tool-frame pose for a given arm ('left'/'right').
  get_gripper_state         Gripper opening width and open/closed/partial status for a
                            given arm. For what is held, use get_held_object.
  get_held_object           Object currently grasped by a given arm, detected via kinematic
                            re-parenting in the scene graph.

COLLISION, FREE SPACE & PLACEMENT
  check_scene_collisions    All current body-body collisions and penetration distances in
                            the scene. Use to verify a configuration before executing.
  get_free_placement_spots  Grid of candidate positions on a surface, filtered for
                            occupancy. Pass object_footprint_m for the object being placed.
  would_collide_at_pose     AABB overlap check for placing an object at a hypothetical
                            (x, y, z). Returns True/False and conflicting objects.

ACCESSIBILITY & PRECONDITIONS
  is_accessible             Whether an object can be reached directly by the gripper right
                            now. Checks for closed-container blocking and stacking.
                            Reports the specific blocker and how to resolve it.

CAUSAL & CONSEQUENCE REASONING
  get_supporting_object     What is physically holding up a given object (the surface or
                            object it rests on). Returns 'floor' if nothing is found.
  get_objects_supported_by  All objects that would be displaced if a given object were
                            moved. Works for any object, not only surface-typed ones.

━━━ SYNTHESIS PRINCIPLES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The tools return raw world-state data. Many questions are answered by reasoning
over that data — not by calling additional tools.

1. DERIVE, DON'T RE-CALL
   If a tool already returned the facts you need, reason over them directly.
   — A count is the length of the returned list.
   — Volume is already in get_object_dimensions output; do not call again.
   — A boolean relation (above, next_to, close_to) is a threshold on the
     offset vector from get_spatial_relation; there is no separate boolean tool.

2. DECOMPOSE, THEN COMPOSE
   Break a complex query into the minimum set of world-state facts needed.
   Call exactly the tools that retrieve those facts, then synthesize the answer
   yourself from the combined results.

3. SELECTION IS REASONING
   'Best', 'closest', 'largest', 'most suitable' are not tool operations — they
   are your reasoning over a retrieved set. Fetch the candidates and their
   properties, then select by comparing values. No dedicated 'best X' tool exists.

4. SINGLE RESPONSIBILITY
   Each tool has one job. Do not chain tools to re-derive data you already hold.
   If list_all_objects already returned positions, do not call get_object_pose
   again for the same objects.

5. MINIMAL CALL SET
   Prefer fewer, broader calls over many narrow ones. list_all_objects gives a
   full scene overview in one call; use it before making targeted follow-up calls.

━━━ OPERATING RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Always resolve semantic names to body_names via find_objects_by_type before
  passing them to any tool that requires an exact body_name.
- If a tool returns an error or empty result, report it clearly and do not retry
  with the same arguments.
- Return only facts derived from tool output. Do not plan actions, recommend
  navigation, or suggest next steps unless explicitly asked.
- For stub tools (get_object_state, get_object_affordances), report the
  not-implemented status directly and do not attempt to work around it.
"""


class SceneQueryAgent:
    """Scene perception specialist. Holds all scene query tools and its own scratchpad."""

    def __init__(self, llm: Any):
        self.llm = llm
        self.tools = [
            WriteScratchpadTool(scratchpad_path=_SCRATCHPAD),
            ReadScratchpadTool(scratchpad_path=_SCRATCHPAD),
            # Segment 1 — World Inventory & Taxonomy
            GetSceneObjectsTool(),
            GetSemanticAnnotationsTool(),
            FindObjectsByTypeTool(),
            GetObjectTypeTool(),
            ClassifyObjectsByRoleTool(),
            # Segment 2 — Geometric & Spatial Properties
            GetObjectPoseTool(),
            GetObjectDimensionsTool(),
            GetObjectOrientationTool(),
            GetObjectColorTool(),
            GetSpatialRelationTool(),
            GetNearestObjectsTool(),
            GetObjectsOnSurfaceTool(),
            SortObjectsBySizeTool(),
            # Segment 3 — Structural & Topological Relations
            GetArticulatedObjectJointsTool(),
            GetContainedItemsTool(),
            # Segment 4 — Functional State & Affordances (stubs)
            GetObjectStateTool(),
            GetObjectAffordancesTool(),
            # Segment 5 — Robot & Interaction State
            GetJointStatesTool(),
            GetRobotPoseTool(),
            GetEndEffectorPoseTool(),
            GetGripperStateTool(),
            GetHeldObjectTool(),
            # Segment 6 — Collision, Free Space & Placement
            CheckSceneCollisionsTool(),
            GetFreePlacementSpotsTool(),
            WouldCollideAtPoseTool(),
            # Segment 7 — Accessibility & Preconditions
            IsAccessibleTool(),
            # Segment 8 — Causal & Consequence Reasoning
            GetSupportingObjectTool(),
            GetObjectsSupportedByTool(),
        ]
        self._agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
            name="scene_perception",
        )


def _extract_scene_facts(messages: list) -> Dict[str, Any]:
    """Parse ToolMessage results from the scene agent into a structured fact dict.

    Builds an AIMessage tool-call lookup (call_id → name + args), then matches
    each ToolMessage by call_id and stores the parsed result under a semantic key.
    """
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

        if tool_name == "get_object_pose":
            body = tool_args.get("body_name", "unknown")
            facts.setdefault(body, {})["pose"] = content
        elif tool_name == "find_objects_by_type":
            type_name = tool_args.get("type_name", "unknown")
            facts[f"type:{type_name}"] = {"matches": content}
        elif tool_name == "get_object_dimensions":
            body = tool_args.get("body_name", "unknown")
            facts.setdefault(body, {})["dimensions"] = content
        elif tool_name == "get_object_orientation":
            body = tool_args.get("body_name", "unknown")
            facts.setdefault(body, {})["orientation"] = content
        elif tool_name == "get_nearest_objects":
            ref = tool_args.get("reference_body_name", "scene")
            facts[f"nearest_to:{ref}"] = content
        elif tool_name == "get_objects_on_surface":
            surface = tool_args.get("surface_name", "surface")
            facts[f"on_surface:{surface}"] = content
        elif tool_name == "get_object_color":
            body = tool_args.get("body_name", "unknown")
            facts.setdefault(body, {})["color"] = content
        elif tool_name == "list_all_objects":
            facts["scene:all_objects"] = content

    return facts


def _build_query(state: RobotAgentState) -> str:
    """Construct the task string from the original instruction and template context."""
    ctx = state.get("template_context", "")
    return f"Instruction: {state['instruction']}\nContext: {ctx}" if ctx else state["instruction"]


def scene_query_node(
    state: RobotAgentState,
    agent: "SceneQueryAgent",
) -> Command[Literal["supervisor"]]:
    """LangGraph node: invoke the scene perception react agent and forward its response."""
    query = _build_query(state)
    logger.debug("  ► [Scene Agent] %s", query[:120])
    result = agent._agent.invoke({"messages": [HumanMessage(content=query)]})
    last_msg = result["messages"][-1]
    scene_facts = _extract_scene_facts(result["messages"])
    logger.debug("  ◄ [Scene Agent] Done. Extracted %d scene fact entries.", len(scene_facts))
    return Command(
        goto="supervisor",
        update={"messages": [last_msg], "scene_facts": scene_facts},
    )
