"""
Tests of the world a scene bundle stands for: its models and objects as a
semantic_digital_twin world, each object annotated as what it is, so the resolver can
read the scene the viewer shows.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from krrood.symbol_graph.symbol_graph import SymbolGraph
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    PouringCup,
)

from cram_viz.resolution.scene_world import (
    SceneKey,
    SceneObjectAnnotations,
    SceneWorldBuilder,
    SceneWorlds,
)

# %% helpers


def scene_objects(bundle: Path) -> dict:
    """
    :return: The scene's objects, by identifier, as the bundle states them.
    """
    scene = json.loads((bundle / "scene.json").read_text())
    return {entry[SceneKey.IDENTIFIER]: entry for entry in scene[SceneKey.OBJECTS]}


def body_of(world, bundle: Path, identifier: str):
    """
    :return: The body of the scene object with the given identifier.
    """
    mesh = scene_objects(bundle)[identifier][SceneKey.MESH]
    return world.get_body_by_name(Path(mesh).name)


class TestSceneWorld:
    def test_the_world_holds_the_scene_models(self, pouring_bundle):
        world = SceneWorldBuilder().build(pouring_bundle)
        assert world.get_body_by_name("gripper_link") is not None

    def test_an_object_stands_where_the_scene_spawns_it(self, pouring_bundle):
        world = SceneWorldBuilder().build(pouring_bundle)
        bowl = body_of(world, pouring_bundle, "bowl")
        position = world.compute_forward_kinematics_np(world.root, bowl)[:3, 3]
        expected = scene_objects(pouring_bundle)["bowl"][SceneKey.SPAWN][:3]
        assert np.allclose(position, expected)

    def test_an_object_is_annotated_as_its_mapping_names(self, pouring_bundle):
        world = SceneWorldBuilder().build(pouring_bundle)
        pitcher = body_of(world, pouring_bundle, "Static_MilkPitcher")
        bowl = body_of(world, pouring_bundle, "bowl")
        annotated = {
            (type(annotation), id(annotation.root))
            for annotation in world.semantic_annotations
        }
        assert {(PouringCup, id(pitcher)), (Bowl, id(bowl))} <= annotated

    def test_an_object_without_a_mapping_is_not_annotated(self, pouring_bundle):
        world = SceneWorldBuilder().build(pouring_bundle)
        mystery = body_of(world, pouring_bundle, "mystery")
        assert all(
            annotation.root is not mystery for annotation in world.semantic_annotations
        )

    def test_the_mapping_names_an_annotation_class_for_each_entry(self):
        annotations = SceneObjectAnnotations.shipped()
        assert annotations.annotation_for("Static_MilkPitcher") is PouringCup
        assert annotations.annotation_for("no_such_object") is None


class TestSceneWorlds:
    def test_a_scene_world_is_built_once(self, pouring_bundle):
        worlds = SceneWorlds()
        first = worlds.world_for("pouring", pouring_bundle)
        assert worlds.world_for("pouring", pouring_bundle) is first

    def test_another_scene_replaces_the_world_kept(self, pouring_bundle, tmp_path):
        other = tmp_path / "other"
        shutil.copytree(pouring_bundle, other)
        worlds = SceneWorlds()
        first = worlds.world_for("pouring", pouring_bundle)
        second = worlds.world_for("other", other)
        assert second is not first
        assert worlds.world is second

    def test_a_replaced_world_leaves_none_of_its_objects_in_the_symbol_graph(
        self, pouring_bundle, tmp_path
    ):
        other = tmp_path / "other"
        shutil.copytree(pouring_bundle, other)
        worlds = SceneWorlds()
        worlds.world_for("pouring", pouring_bundle)
        second = worlds.world_for("other", other)
        assert list(SymbolGraph().get_instances_of_type(PouringCup)) == [
            annotation
            for annotation in second.semantic_annotations
            if isinstance(annotation, PouringCup)
        ]
