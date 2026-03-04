"""PyCRAM Simulation Bridge
===========================

Connects the llmr CRAM serializer to a **live**
``semantic_digital_twin.world.World`` object (used by PyCRAM as its
simulation backend) so that body lookup, plan construction, and action
execution all work with live world state.

Architecture overview
---------------------

::

    ┌────────────────────────────┐
    │  LangGraph LLM Workflow    │
    │  (enhanced_ad_graph.py)    │
    └────────────┬───────────────┘
                 │  CRAM string
                 ▼
    ┌────────────────────────────┐
    │  CRAMToPyCRAMSerializer    │  (cram_to_pycram.py)
    │  parse() → CRAMActionPlan  │
    └────────────┬───────────────┘
                 │  CRAMActionPlan
                 ▼
    ┌────────────────────────────┐       ┌──────────────────┐
    │  SimulationBridge          │◄──────│  World  (live)   │
    │  to_partial_designator()   │       │  world.bodies    │
    └────────────┬───────────────┘       └──────────────────┘
                 │  PartialDesignator
                 ▼
    ┌────────────────────────────┐
    │  SequentialPlan + Context  │  (language.py / plan.py)
    │  .perform()                │
    └────────────────────────────┘

Key concepts
------------

* **World** lives in ``semantic_digital_twin.world.World``.  It is *not*
  a global singleton — PyCRAM passes it around via ``Context``.
* **Context** (``pycram.datastructures.dataclasses.Context``) carries
  ``(world, robot, super_plan, ros_node)`` and is required by every
  ``ActionDescription.execute()`` call.
* **PartialDesignator** is the lazily-evaluated handle returned by every
  ``SomeAction.description(...)`` class-method.  It becomes executable once
  it is added to a plan that carries a ``Context``.
* **BodyResolver** is ``Callable[[CRAMEntityInfo], Optional[Body]]``.
  ``SimulationBridge`` builds one automatically from the injected ``World``.

Minimal usage
-------------
::

    from semantic_digital_twin.world import World
    from semantic_digital_twin.robots.abstract_robot import AbstractRobot
    from llmr.serializers.simulation_bridge import SimulationBridge

    # -- Obtain references to your running simulation objects ------------
    world: World = ...          # your running World instance
    robot: AbstractRobot = ...  # robot description inside that world

    # -- Create the bridge once ------------------------------------------
    bridge = SimulationBridge(world, robot)

    # -- Execute a CRAM plan string from the LLM --------------------------
    cram_string = (
        "(an action (type PickingUp) "
        "(object (:tag cup (an object (type Artifact)))) "
        "(source (a location (on (:tag table (an object (type Surface)))))))"
    )
    bridge.execute(cram_string)

    # -- Or just obtain the PartialDesignator (no execution) --------------
    partial = bridge.to_partial_designator(cram_string, arm=Arms.RIGHT)

Advanced: multi-step CRAM plans
--------------------------------
::

    from pycram.language import SequentialPlan
    context = bridge.context   # re-use the Context object

    step1 = bridge.to_partial_designator(cram_pick)
    step2 = bridge.to_partial_designator(cram_place)
    SequentialPlan(context, step1, step2).perform()

World introspection helpers
----------------------------
::

    bridge.list_bodies()                    # all Body names in world
    bridge.find_body("cup")                 # Body by exact name
    bridge.snapshot()                       # {name: pose} dict
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _body_name(body: Any) -> str:
    """Return the string name of a ``Body`` object tolerantly."""
    raw = getattr(body, "name", None)
    if raw is None:
        return ""
    # PrefixedName objects stringify to 'prefix:local'; use .name or str()
    if hasattr(raw, "name"):
        return str(raw.name)
    return str(raw)


def _body_pose(body: Any) -> Optional[Any]:
    """Return the global pose of *body*, or ``None`` on failure."""
    try:
        return body.global_pose
    except Exception:
        pass
    try:
        return body.pose
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SimulationBridge
# ─────────────────────────────────────────────────────────────────────────────

class SimulationBridge:
    """Connects CRAM serialization to a live PyCRAM / SemanticDigitalTwin world.

    Parameters
    ----------
    world:
        A running ``semantic_digital_twin.world.World`` instance.  All body
        lookups are performed against ``world.bodies``.
    robot:
        The ``AbstractRobot`` (or subclass) that will execute actions.
    ros_node:
        Optional ROS2 node.  Forwarded into the ``Context`` if provided.
    """

    def __init__(self, world: Any, robot: Any, ros_node: Any = None) -> None:
        self._world = world
        self._robot = robot
        self._ros_node = ros_node

        # Lazy-import heavy PyCRAM / serializer objects so the module remains
        # importable even without a full PyCRAM installation.
        self._serializer = self._make_serializer()
        self._context = self._make_context()

    # ── Public properties ──────────────────────────────────────────────────

    @property
    def world(self) -> Any:
        """The injected ``World`` object."""
        return self._world

    @property
    def robot(self) -> Any:
        """The injected ``AbstractRobot`` object."""
        return self._robot

    @property
    def context(self) -> Any:
        """The ``Context`` dataclass used to execute plans.

        Re-create it after swapping ``world`` or ``robot`` via
        :meth:`update_world` / :meth:`update_robot`.
        """
        return self._context

    # ── World introspection ────────────────────────────────────────────────

    def list_bodies(self) -> List[str]:
        """Return the names of all bodies currently in the world."""
        return [_body_name(b) for b in getattr(self._world, "bodies", [])]

    def find_body(self, name: str) -> Optional[Any]:
        """Find a ``Body`` by exact name (case-sensitive).

        Parameters
        ----------
        name:
            The object name to look up (e.g. ``"cup"``, ``"table"``).

        Returns
        -------
        Body or None
        """
        name_lower = name.lower()
        for body in getattr(self._world, "bodies", []):
            bname = _body_name(body)
            if bname == name or bname.lower() == name_lower:
                return body
        return None

    def snapshot(self) -> Dict[str, Any]:
        """Return ``{body_name: pose}`` for every body in the world.

        Useful for debugging: print what objects the bridge can see before
        triggering a CRAM plan.
        """
        result: Dict[str, Any] = {}
        for body in getattr(self._world, "bodies", []):
            bname = _body_name(body)
            result[bname] = _body_pose(body)
        return result

    # ── Runtime world / robot update ───────────────────────────────────────

    def update_world(self, world: Any) -> None:
        """Swap the injected world and rebuild the internal context.

        Call this if your simulation resets (e.g. new episode) and produces a
        fresh ``World`` object.
        """
        self._world = world
        self._context = self._make_context()
        logger.info("SimulationBridge: world updated to %r", world)

    def update_robot(self, robot: Any) -> None:
        """Swap the robot reference and rebuild the internal context."""
        self._robot = robot
        self._context = self._make_context()
        logger.info("SimulationBridge: robot updated to %r", robot)

    # ── Body resolver ──────────────────────────────────────────────────────

    def make_resolver(self) -> Any:
        """Return a fresh ``BodyResolver`` bound to the current world.

        The resolver tries:
        1. Exact name match against ``world.bodies``.
        2. Case-insensitive name match.
        3. Substring match on ``entity.semantic_type`` inside body names.

        Returns
        -------
        BodyResolver
            ``Callable[[CRAMEntityInfo], Optional[Body]]``
        """
        from .body_resolver import make_world_body_resolver
        return make_world_body_resolver(self._world)

    # ── Serialization helpers ─────────────────────────────────────────────

    def parse(self, cram_string: str) -> Any:
        """Parse a CRAM string → ``CRAMActionPlan`` (no PyCRAM needed).

        Parameters
        ----------
        cram_string:
            The raw LISP-style CRAM plan from the LLM workflow.

        Returns
        -------
        CRAMActionPlan
        """
        return self._serializer.parse(cram_string)

    def to_partial_designator(
        self,
        cram_string: str,
        arm: Any = None,
        grasp_description: Any = None,
        approach_from: Any = None,
    ) -> Any:
        """Parse *cram_string* and return a PyCRAM ``PartialDesignator``.

        The designator is **not yet executed** — attach it to a
        ``SequentialPlan`` or call :meth:`execute` for immediate execution.

        Parameters
        ----------
        cram_string:
            LISP-style CRAM plan string from the LLM.
        arm:
            Optional ``Arms`` enum value (e.g. ``Arms.RIGHT``).  If ``None``
            the mapper picks a reasonable default (usually ``Arms.RIGHT``).
        grasp_description:
            Optional ``GraspDescription`` / ``Grasp`` value overriding the
            mapper default.

        Returns
        -------
        PartialDesignator
        """
        resolver = self.make_resolver()
        plan = self._serializer.parse(cram_string)

        # Resolve the effective arm (default to RIGHT when not specified)
        effective_arm = arm
        if effective_arm is None:
            try:
                from pycram.datastructures.enums import Arms
                effective_arm = Arms.RIGHT
            except ImportError:
                pass

        # Auto-compute grasp description from the object body when not provided
        if grasp_description is None and plan.object is not None:
            try:
                from pycram.datastructures.grasp import GraspDescription
                from pycram.datastructures.pose import PoseStamped
                from pycram.view_manager import ViewManager

                object_body = resolver(plan.object)
                if object_body is not None and effective_arm is not None:
                    arm_view = ViewManager.get_arm_view(effective_arm, self._robot)
                    manipulator = arm_view.manipulator
                    object_pose = PoseStamped.from_spatial_type(object_body.global_pose)
                    grasp_descs = GraspDescription.calculate_grasp_descriptions(
                        manipulator, object_pose
                    )
                    if grasp_descs:
                        grasp_description = grasp_descs[0]
                        logger.info(
                            "Auto-computed grasp: approach=%s vertical=%s",
                            grasp_description.approach_direction.name,
                            grasp_description.vertical_alignment.name,
                        )
            except Exception as exc:
                logger.warning("Could not auto-compute grasp description: %s", exc)

        kwargs: Dict[str, Any] = {}
        if effective_arm is not None:
            kwargs["arm"] = effective_arm
        if grasp_description is not None:
            kwargs["grasp_description"] = grasp_description
        if approach_from is not None:
            kwargs["approach_from"] = approach_from

        partial = self._serializer.to_partial_designator(
            plan, body_resolver=resolver, **kwargs
        )
        logger.info(
            "to_partial_designator: action=%s → %r",
            plan.action_type, partial,
        )
        return partial

    def execute(
        self,
        cram_string: str,
        arm: Any = None,
        grasp_description: Any = None,
    ) -> Any:
        """Parse, resolve bodies, build a plan, and **execute** it.

        This is the one-shot convenience method for robot execution.

        Parameters
        ----------
        cram_string:
            LISP-style CRAM plan string from the LLM.
        arm:
            Optional ``Arms`` override.
        grasp_description:
            Optional grasp override.

        Returns
        -------
        Any
            Whatever the underlying PyCRAM action returns (often ``None``
            for purely actuator-driving actions).

        Raises
        ------
        RuntimeError
            If PyCRAM's ``SequentialPlan`` cannot be imported.
        pycram.failures.PlanFailure
            If the action fails during execution.
        """
        partial = self.to_partial_designator(
            cram_string, arm=arm, grasp_description=grasp_description
        )

        try:
            from pycram.language import SequentialPlan
        except ImportError as exc:
            raise RuntimeError(
                "PyCRAM must be installed to execute plans. "
                "Use to_partial_designator() to obtain the designator and "
                "integrate it into your own plan structure."
            ) from exc

        seq_plan = SequentialPlan(self._context, partial)
        logger.info("Executing plan via SequentialPlan …")
        result = seq_plan.perform()
        logger.info("Plan execution finished, result=%r", result)
        return result

    def execute_batch(
        self,
        cram_strings: List[str],
        arm: Any = None,
    ) -> List[Any]:
        """Execute a list of CRAM plan strings **sequentially**.

        Each string is parsed, resolved, and executed in order.  A failure
        in step *i* propagates immediately (no skip-ahead).

        For placement-type actions (e.g. ``Placing``, ``Transport``), a
        ``NavigateAction`` is automatically injected before the action so the
        robot drives to an IK-reachable base pose near the target — mirroring
        the behaviour of ``TransportAction.execute()`` internally.

        Parameters
        ----------
        cram_strings:
            Ordered list of CRAM plan strings (e.g. from the LLM output).
        arm:
            Arm override applied to every step.

        Returns
        -------
        list
            Results returned by each action (in order).
        """
        if not cram_strings:
            return []

        try:
            from pycram.language import SequentialPlan
        except ImportError as exc:
            raise RuntimeError(
                "PyCRAM must be installed to execute plans."
            ) from exc

        # Placement-type action names that require the robot to navigate to the
        # goal location before the action can succeed (same as TransportAction).
        _PLACEMENT_TYPES = {
            "placing", "place", "placeaction",
            "putobject", "transport", "transportaction",
            "pickandplace", "moveandplace", "moveandplaceaction",
        }

        # Resolve the effective arm once (used both for grasp and CostmapLocation)
        effective_arm = arm
        if effective_arm is None:
            try:
                from pycram.datastructures.enums import Arms
                effective_arm = Arms.RIGHT
            except ImportError:
                pass

        resolver = self.make_resolver()

        # Build all designators first, then run them in ONE SequentialPlan so
        # that multi-step actions (e.g. PlaceAction) can look back in the plan
        # tree and find prior steps (e.g. the preceding PickUpAction).
        partials = []
        for i, cram_str in enumerate(cram_strings):
            logger.info("execute_batch: building step %d/%d", i + 1, len(cram_strings))

            # Parse the CRAM string to detect placement actions
            plan = self._serializer.parse(cram_str)
            action_norm = (
                plan.action_type.lower().replace("_", "").replace(" ", "")
                if plan.action_type else ""
            )

            placement_nav_pose = None  # will be set for PlaceAction steps
            if action_norm in _PLACEMENT_TYPES and plan.goal is not None:
                # Auto-inject a NavigateAction to a collision-free base pose near
                # the placement target, resolved before PickUp (milk not yet attached).
                goal_body = resolver(plan.goal)
                if goal_body is not None:
                    placement_nav_pose = self._resolve_placement_nav_pose(goal_body, effective_arm)
                    if placement_nav_pose is not None:
                        from pycram.robot_plans import NavigateActionDescription
                        partials.append(NavigateActionDescription(placement_nav_pose))
                        logger.info(
                            "execute_batch: injected NavigateAction to (%.2f, %.2f) "
                            "before %s",
                            placement_nav_pose.pose.position.x,
                            placement_nav_pose.pose.position.y,
                            plan.action_type,
                        )
                    else:
                        logger.warning(
                            "execute_batch: could not resolve placement nav pose for %r; "
                            "PlaceAction may fail if robot is too far from target.",
                            goal_body,
                        )

            # For PlaceAction, pass nav_pose as approach_from so the placement
            # target is the near edge of the surface (reachable by the arm)
            # rather than the table centre which may be out of reach.
            partials.append(
                self.to_partial_designator(cram_str, arm=arm, approach_from=placement_nav_pose)
            )

        logger.info("execute_batch: executing %d steps in one SequentialPlan", len(partials))
        seq_plan = SequentialPlan(self._context, *partials)
        result = seq_plan.perform()
        # Return a list with one entry per step for API compatibility
        return [result] * len(cram_strings)

    # ── Private helpers ────────────────────────────────────────────────────

    _COSTMAP_NAV_TIMEOUT = 25.0  # seconds before falling back to direct pose

    def _resolve_placement_nav_pose(self, goal_body: Any, arm: Any) -> Any:
        """Return a navigation PoseStamped near *goal_body*, or ``None`` on failure.

        Tries CostmapLocation (IK-validated) in a background thread with a
        timeout.  If that times out or raises, falls back to a ring of candidate
        poses around the target and returns the first collision-free one.
        """
        import threading
        import types as _types
        import math

        try:
            from pycram.datastructures.pose import PoseStamped
            from pycram.designators.location_designator import CostmapLocation
        except ImportError:
            return None

        goal_ps = PoseStamped.from_spatial_type(goal_body.global_pose)

        # ── Attempt 1: CostmapLocation in a thread with timeout ───────────
        result_holder: list = [None]
        error_holder: list = [None]

        stub = _types.SimpleNamespace(
            plan=_types.SimpleNamespace(world=self._world, robot=self._robot)
        )

        def _worker():
            try:
                nav_loc = CostmapLocation(
                    target=goal_ps,
                    reachable_for=self._robot,
                    reachable_arm=arm,
                )
                nav_loc.plan_node = stub
                result_holder[0] = next(iter(nav_loc))
                nav_loc.plan_node = None
            except Exception as exc:
                error_holder[0] = exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(self._COSTMAP_NAV_TIMEOUT)

        if result_holder[0] is not None:
            return result_holder[0].pose

        if error_holder[0] is not None:
            logger.debug(
                "_resolve_placement_nav_pose: CostmapLocation raised %s", error_holder[0]
            )
        else:
            logger.warning(
                "_resolve_placement_nav_pose: CostmapLocation timed out after %.0fs "
                "for %r — falling back to ring search",
                self._COSTMAP_NAV_TIMEOUT, goal_body,
            )

        # ── Attempt 2: ring of candidate poses at multiple radii ─────────────
        # Tables are surrounded by chairs at ~0.7m, so we try several radii
        # and fall back to an unchecked pose if all collision tests fail.
        try:
            from copy import deepcopy as _deepcopy

            target_x = goal_ps.pose.position.x
            target_y = goal_ps.pose.position.y
            steps = 16
            radii = [0.75, 1.0, 1.25, 1.5]  # metres; expand past surrounding furniture

            # Deep-copy once (robot not yet holding object at build time)
            test_world = _deepcopy(self._world)
            test_robot = self._robot.__class__.from_world(test_world)

            try:
                from semantic_digital_twin.collision_checking.collision_checker import (
                    collision_check,
                )
                have_collision_check = True
            except ImportError:
                have_collision_check = False
                logger.debug("_resolve_placement_nav_pose: collision_check unavailable")

            first_candidate = None  # keep best-effort pose in case all are blocked

            for radius in radii:
                for step in range(steps):
                    angle = 2.0 * math.pi * step / steps
                    cx = target_x + radius * math.cos(angle)
                    cy = target_y + radius * math.sin(angle)
                    # Orientation: face the target
                    yaw = math.atan2(target_y - cy, target_x - cx)
                    qz = math.sin(yaw / 2.0)
                    qw = math.cos(yaw / 2.0)
                    candidate = PoseStamped.from_list(
                        [cx, cy, 0.0], [0.0, 0.0, qz, qw], self._world.root
                    )
                    if first_candidate is None:
                        first_candidate = candidate  # save for last-resort

                    if not have_collision_check:
                        return candidate  # no check possible — use first candidate

                    test_robot.root.parent_connection.origin = (
                        candidate.to_spatial_type()
                    )
                    collisions = collision_check(robot=test_robot, world=test_world)
                    if not collisions:
                        logger.info(
                            "_resolve_placement_nav_pose: ring fallback succeeded "
                            "at r=%.2f angle=%.0f° (%.2f, %.2f)",
                            radius, math.degrees(angle), cx, cy,
                        )
                        return candidate

            # Last resort: return the first candidate (1.5m, 0°) without collision check.
            # NavigateAction's own path planner will handle any minor obstacles.
            if first_candidate is not None:
                logger.warning(
                    "_resolve_placement_nav_pose: all ring candidates in collision; "
                    "using unchecked fallback pose (%.2f, %.2f) for %r",
                    first_candidate.pose.position.x,
                    first_candidate.pose.position.y,
                    goal_body,
                )
                return first_candidate

        except Exception as exc:
            logger.debug("_resolve_placement_nav_pose: ring fallback failed: %s", exc)

        logger.warning(
            "_resolve_placement_nav_pose: all strategies failed for %r", goal_body
        )
        return None

    def _make_serializer(self) -> Any:
        """Instantiate ``CRAMToPyCRAMSerializer`` from the sibling module."""
        from .cram_to_pycram import CRAMToPyCRAMSerializer
        return CRAMToPyCRAMSerializer()

    def _make_context(self) -> Any:
        """Build a PyCRAM ``Context`` from the injected world + robot.

        Returns ``None`` if ``pycram`` is not installed (parse-only mode).
        """
        try:
            from pycram.datastructures.dataclasses import Context
            ctx = Context(
                world=self._world,
                robot=self._robot,
                ros_node=self._ros_node,
            )
            logger.debug("SimulationBridge: Context built %r", ctx)
            return ctx
        except ImportError:
            logger.warning(
                "pycram not installed — Context not created. "
                "execute() will fail; parse-only mode is available."
            )
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def make_bridge(world: Any, robot: Any, ros_node: Any = None) -> SimulationBridge:
    """Convenience factory — same as ``SimulationBridge(world, robot, ros_node)``.

    Example
    -------
    ::

        bridge = make_bridge(world, robot)
        bridge.execute(cram_string)
    """
    return SimulationBridge(world, robot, ros_node)
