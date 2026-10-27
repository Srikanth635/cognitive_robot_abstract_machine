"""
Fixtures for the cramera.resolution tests.
"""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import coraplex
import pytest
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.package_resolver import PackageUriResolver
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    PouringCup,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World


@pytest.fixture(autouse=True)
def isolated_cramera_directories(monkeypatch, tmp_path):
    """
    No scene bundles and an empty data directory, so the viewer reads nothing of the
    machine it runs on.
    """
    monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path / "scenes"))
    monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))


@pytest.fixture()
def sample_hypothesis_knowledge():
    """
    Activate the hand-authored pouring sample as the knowledge the views show.

    Nothing is active until an instruction is generated, so any test exercising the
    projection or its overlays has to say which structure it means.
    """
    from cramera.resolution import hypothesis_knowledge

    previous = hypothesis_knowledge.active()
    knowledge = hypothesis_knowledge.activate_sample()
    yield knowledge
    hypothesis_knowledge.activate(previous)


# %% the pouring scene

POURING_SCENE = Path(__file__).parent / "scenes" / "pouring"
"""
The pouring scene bundle the tests build worlds from, its meshes copied in.
"""

OBJECT_MESHES = Path(coraplex.__file__).parents[2] / "resources" / "objects"
"""
The repository's object meshes the fixture scene's objects are made of.
"""

MESH_SOURCES = {
    "bowl.stl": "bowl.stl",
    "Static_MilkPitcher.stl": "jeroen_cup.stl",
    "mystery.stl": "spoon.stl",
}
"""
The repository mesh each fixture object's mesh is copied from.
"""


@pytest.fixture
def pouring_bundle(tmp_path) -> Path:
    """
    The pouring scene bundle, its meshes copied from the repository's objects.
    """
    bundle = tmp_path / "pouring"
    shutil.copytree(POURING_SCENE, bundle)
    meshes = bundle / "meshes" / "objects"
    meshes.mkdir(parents=True)
    for name, source in MESH_SOURCES.items():
        shutil.copy(OBJECT_MESHES / source, meshes / name)
    return bundle


@pytest.fixture
def pouring_bundle_with_arms(pouring_bundle) -> Path:
    """
    The pouring scene bundle with a PR2 for its robot, so the arm a pouring takes is one
    of the arms the scene holds.
    """
    scene_file = pouring_bundle / "scene.json"
    scene = json.loads(scene_file.read_text())
    robot_model, *_ = scene["models"]
    robot_model["urdf"] = PackageUriResolver().resolve(PR2.get_ros_file_path())
    scene["robot"]["name"] = "pr2"
    scene_file.write_text(json.dumps(scene))
    return pouring_bundle


# %% the PR2 kitchen the simple_pouring scene records

KITCHEN = Path(__file__).parent / "execution" / "pouring_kitchen.json"
"""
Where the simple_pouring scene's recording starts the robot and places its objects.
"""


@pytest.fixture()
def pouring_kitchen(pouring_kitchen_template) -> World:
    """
    :return: A copy of the PR2 in the apartment, at the start the simple_pouring scene
        records, with the pitcher and the bowl where the scene places them.
    """
    return deepcopy(pouring_kitchen_template)


@pytest.fixture(scope="session")
def pouring_kitchen_template(pr2_apartment_world) -> World:
    """
    :return: The PR2 in the apartment, at the start the simple_pouring scene records,
        with the pitcher and the bowl where the scene places them; to be copied, not
        changed.
    """
    world = deepcopy(pr2_apartment_world)
    kitchen = json.loads(KITCHEN.read_text())
    for entry in kitchen["objects"]:
        world.merge_world_at_pose(
            STLParser(str(OBJECT_MESHES / entry["mesh"])).parse(),
            HomogeneousTransformationMatrix.from_xyz_rpy(
                *entry["position"], reference_frame=world.root
            ),
        )
    with world.modify_world():
        world.add_semantic_annotations(
            [
                PouringCup(root=world.get_body_by_name("Static_MilkPitcher.stl")),
                Bowl(root=world.get_body_by_name("bowl.stl")),
            ]
        )
        world.get_body_by_name("base_footprint").parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                **kitchen["robot_start"], reference_frame=world.root
            )
        )
    return world
