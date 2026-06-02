"""Scene Query Agent — specialist for querying the semantic scene graph."""

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

SYSTEM_PROMPT = """You are the Scene Perception specialist. You answer questions about the
world and the robot's current state by observing them through your tools, then reasoning
over what you observe. Return a concise, factual summary grounded entirely in tool results.

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
  list_all_objects          Body_name, semantic type, and position of every task-relevant
                            object — a broad starting point for an overview. The body_name
                            values it returns are the keys other tools expect.
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
  get_object_state          Returns a not-implemented message.
  get_object_affordances    Returns a not-implemented message.

ROBOT & INTERACTION STATE
  get_joint_states          Current robot arm joint positions (not furniture joints; no limits).
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

━━━ HOW TO WORK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The tool list above describes capabilities, not an order. Decide for yourself which tools
the question needs and call them as you see fit. Let these principles guide that reasoning:

- GROUND IN OBSERVATION. Every fact you report must come from a tool result. Never guess a
  pose, type, dimension, or relationship.
- RESOLVE NAMES FIRST. Tools that act on a specific object need its exact body_name. When the
  query refers to an object by semantic type ('the milk', 'a cup'), resolve it to a
  body_name before asking about its properties.
- REASON, DON'T RE-CALL. Once a tool returns the data, derive the answer yourself.
  'Closest', 'largest', 'which surface', 'is it upright' are conclusions you reach over the
  facts you already hold — not extra tool calls. Prefer the fewest, broadest calls that
  gather what you need.
- REPORT HONESTLY. If a tool errors or returns nothing, say so plainly and do not retry the
  same call. Report only what your observations support; do not plan actions or recommend
  next steps unless the query asks for them.
"""


# ---------------------------------------------------------------------------
# Subgraph state
# ---------------------------------------------------------------------------

class SceneAgentState(TypedDict):
    """State for the scene perception subgraph. Keys matching RobotAgentState are shared."""
    messages:         Annotated[list[BaseMessage], add_messages]
    scene_facts:      Annotated[Dict[str, Any], _merge_dicts]
    instruction:      str
    current_task:     str
    template_context: str


# ---------------------------------------------------------------------------
# SceneQueryAgent
# ---------------------------------------------------------------------------

class SceneQueryAgent:
    """Scene perception specialist with a custom LangGraph subgraph."""

    def __init__(self, llm: "BaseChatModel", scratchpad_path: str = _SCRATCHPAD):
        self.llm = llm
        self.scratchpad_path = scratchpad_path
        self.tools = [
            WriteScratchpadTool(scratchpad_path=scratchpad_path),
            ReadScratchpadTool(scratchpad_path=scratchpad_path),
            GetSceneObjectsTool(),
            GetSemanticAnnotationsTool(),
            FindObjectsByTypeTool(),
            GetObjectTypeTool(),
            ClassifyObjectsByRoleTool(),
            GetObjectPoseTool(),
            GetObjectDimensionsTool(),
            GetObjectOrientationTool(),
            GetObjectColorTool(),
            GetSpatialRelationTool(),
            GetNearestObjectsTool(),
            GetObjectsOnSurfaceTool(),
            SortObjectsBySizeTool(),
            GetArticulatedObjectJointsTool(),
            GetContainedItemsTool(),
            GetObjectStateTool(),
            GetObjectAffordancesTool(),
            GetJointStatesTool(),
            GetRobotPoseTool(),
            GetEndEffectorPoseTool(),
            GetGripperStateTool(),
            GetHeldObjectTool(),
            CheckSceneCollisionsTool(),
            GetFreePlacementSpotsTool(),
            WouldCollideAtPoseTool(),
            IsAccessibleTool(),
            GetSupportingObjectTool(),
            GetObjectsSupportedByTool(),
        ]

        llm_with_tools = llm.bind_tools(self.tools)

        builder = StateGraph(SceneAgentState)
        builder.add_node("prepare_query", make_prepare_query([
            ("scene_facts", "Already known (do NOT re-query)"),
        ]))
        builder.add_node("call_model", make_call_model(llm_with_tools, SYSTEM_PROMPT))
        builder.add_node("execute_tools", ExecuteToolsNode(self.tools, "scene_facts"))

        builder.add_edge(START, "prepare_query")
        builder.add_edge("prepare_query", "call_model")
        builder.add_conditional_edges("call_model", tools_condition,
                                      {"execute_tools": "execute_tools", END: END})
        builder.add_edge("execute_tools", "call_model")

        self.subgraph = builder.compile()
        logger.debug("[SceneQueryAgent] Subgraph compiled with %d tools.", len(self.tools))
