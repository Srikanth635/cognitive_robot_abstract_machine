"""World setup: build and reset the PR2 apartment simulation world.

Encapsulates all world-construction boilerplate so notebooks can load the full
environment in a single call::

    from generative_backend.world_setup import load_pr2_apartment_world

    world, pr2, context = load_pr2_apartment_world()

The first call is slow (parses URDFs and merges objects).  Subsequent calls are
fast because the assembled base world is cached at module level and each call
returns a fresh ``deepcopy``.

API
---
``load_pr2_apartment_world(reload=False)``
    Returns ``(world, pr2, context)`` — a mutable copy ready for simulation.
    Pass ``reload=True`` to force a full rebuild (e.g. after changing assets).

``get_base_world()``
    Returns the cached (immutable) base world directly.  Useful when you need
    to manage ``deepcopy`` yourself.
"""

from __future__ import annotations

import os
import pathlib
from copy import deepcopy
from typing import Optional, Tuple, Type

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.adapters.package_resolver import PathResolver
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    DiffDrive,
    Connection6DoF,
)
from semantic_digital_twin.world_description.world_entity import Body

from pycram.datastructures.dataclasses import Context

# ── Path resolution ────────────────────────────────────────────────────────────
# This file lives at:  <repo>/cognitive_robot_abstract_machine/generative_backend/src/generative_backend/world_setup.py
# parents[2]  →  <repo>/cognitive_robot_abstract_machine/generative_backend/
# parents[3]  →  <repo>/cognitive_robot_abstract_machine/
_HERE = pathlib.Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]       # generative_backend/
_REPO_ROOT = _HERE.parents[3]          # cognitive_robot_abstract_machine/

_PYCRAM_RESOURCES = _REPO_ROOT / "pycram" / "resources"
_APARTMENT_URDF = str(_PYCRAM_RESOURCES / "worlds" / "apartment.urdf")
_MILK_STL = str(_PYCRAM_RESOURCES / "objects" / "milk.stl")
_CEREAL_STL = str(_PYCRAM_RESOURCES / "objects" / "breakfast_cereal.stl")
_PR2_URDF = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"

# ── Module-level base world cache ──────────────────────────────────────────────
_base_world_cache: Optional[World] = None


# ── Internal builders ──────────────────────────────────────────────────────────


def _build_apartment_world() -> World:
    """Parse the apartment URDF and place milk + cereal objects inside it."""
    apartment_world = URDFParser.from_file(_APARTMENT_URDF).parse()

    milk_world = STLParser(_MILK_STL).parse()
    cereal_world = STLParser(_CEREAL_STL).parse()

    apartment_world.merge_world_at_pose(
        milk_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 2, 1.05, reference_frame=apartment_world.root
        ),
    )
    apartment_world.merge_world_at_pose(
        cereal_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 1.8, 1.05, reference_frame=apartment_world.root
        ),
    )

    milk_view = Milk(
        root=apartment_world.get_body_by_name("milk.stl"), _world=apartment_world
    )
    with apartment_world.modify_world():
        apartment_world.add_semantic_annotation(milk_view)

    return apartment_world


def _build_urdf_world(
    urdf_path: str,
    robot_annotation: Optional[Type[AbstractRobot]],
    drive_type: Type,
    starting_pose: Optional[HomogeneousTransformationMatrix] = None,
    path_resolver: Optional[PathResolver] = None,
) -> World:
    """Parse a URDF and wire it into the map → odom_combined → robot tree."""
    world = URDFParser.from_file(file_path=urdf_path, path_resolver=path_resolver).parse()

    if robot_annotation is not None:
        robot_annotation.from_world(world)

    with world.modify_world():
        map_body = Body(name=PrefixedName("map"))
        odom_body = Body(name=PrefixedName("odom_combined"))
        world.add_connection(Connection6DoF.create_with_dofs(world, map_body, odom_body))
        drive_conn = drive_type.create_with_dofs(
            parent=odom_body, child=world.root, world=world
        )
        world.add_connection(drive_conn)
        drive_conn.has_hardware_interface = True
        if starting_pose is not None:
            drive_conn.origin = starting_pose

    return world


def _build_pr2_world() -> World:
    return _build_urdf_world(_PR2_URDF, PR2, OmniDrive)


def _build_pr2_apartment_world() -> World:
    """Merge the PR2 robot world with the apartment + objects."""
    pr2_world = _build_pr2_world()
    apartment_world = _build_apartment_world()

    merged = deepcopy(pr2_world)
    PR2.from_world(merged)
    merged.merge_world(deepcopy(apartment_world))
    merged.get_body_by_name("base_footprint").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(1.3, 2, 0)
    )
    return merged


def _make_mutable_copy(base_world: World) -> Tuple[World, PR2, Context]:
    """Return a fresh mutable deepcopy of *base_world* with PR2 + Context attached."""
    world = deepcopy(base_world)
    pr2 = PR2.from_world(world)
    return world, pr2, Context(world, pr2)


# ── Public API ─────────────────────────────────────────────────────────────────


def get_base_world(*, reload: bool = False) -> World:
    """Return the cached immutable PR2 apartment base world.

    Builds it on first call (slow); subsequent calls return the cached instance.

    :param reload: Force a full rebuild even if the cache is populated.
    :return: The assembled base ``World`` (do not modify — deepcopy it first).
    """
    global _base_world_cache
    if _base_world_cache is None or reload:
        _base_world_cache = _build_pr2_apartment_world()
    return _base_world_cache


def load_pr2_apartment_world(*, reload: bool = False) -> Tuple[World, PR2, Context]:
    """Load a mutable PR2 apartment world ready for simulation.

    Returns a fresh ``deepcopy`` of the cached base world so each call is
    independent.  The first call is slow (URDF parsing + object placement);
    subsequent calls are fast (deepcopy only).

    Usage::

        world, pr2, context = load_pr2_apartment_world()

    :param reload: Pass ``True`` to force a full rebuild of the base world
        (e.g. after changing asset files on disk).
    :return: ``(world, pr2, context)`` — a mutable simulation-ready tuple.
    """
    base = get_base_world(reload=reload)
    return _make_mutable_copy(base)
