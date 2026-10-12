"""
The world a scene bundle stands for: its models and its objects as a
semantic_digital_twin world, each object annotated as what it is, so llm_reasoner's
resolver reads the scene the viewer shows.

A bundle's objects carry no semantic type, so which annotation each one gets is kept as
data, by the object's identifier.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from krrood.symbol_graph.symbol_graph import SymbolGraph
from krrood.utils import resolve_class_from_full_name
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.package_resolver import (
    CompositePathResolver,
    FileUriResolver,
    PackageUriResolver,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from typing_extensions import Any, Mapping, Optional, Sequence

OBJECT_ANNOTATIONS_FILE = Path(__file__).parent / "object_annotations.json"
"""
Which annotation class each scene object is, by the object's identifier.
"""

SCENE_FILE = "scene.json"
"""
The file of a bundle describing the scene.
"""


class SceneKey(StrEnum):
    """
    The keys of a scene description the world is built from.
    """

    MODELS = "models"
    URDF = "urdf"
    OBJECTS = "objects"
    IDENTIFIER = "id"
    MESH = "mesh"
    SPAWN = "spawn"


@dataclass(frozen=True)
class SceneObjectAnnotations:
    """
    The annotation class of each scene object that has one, by the object's identifier.
    """

    by_object: Mapping[str, type[HasRootBody]]
    """
    The annotation class, by the identifier of the object it annotates.
    """

    @classmethod
    def shipped(cls) -> SceneObjectAnnotations:
        """
        :return: The annotations kept as data beside this module.
        """
        stored = json.loads(OBJECT_ANNOTATIONS_FILE.read_text())
        return cls(
            by_object={
                identifier: resolve_class_from_full_name(class_name)
                for identifier, class_name in stored.items()
            }
        )

    def annotation_for(self, identifier: str) -> Optional[type[HasRootBody]]:
        """
        :return: The annotation class of the object with the given identifier, ``None``
            when it has none.
        """
        return self.by_object.get(identifier)


@dataclass
class SceneWorldBuilder:
    """
    Builds the world a scene bundle stands for.
    """

    annotations: SceneObjectAnnotations = field(
        default_factory=SceneObjectAnnotations.shipped
    )
    """
    Which annotation each scene object gets.
    """

    def build(self, bundle: Path) -> World:
        """
        :param bundle: The directory of a scene bundle.
        :return: The bundle's models merged into one world, with every object at its
            spawn pose and annotated as what it is.
        """
        scene = json.loads((bundle / SCENE_FILE).read_text())
        models = [self.model(bundle, model) for model in scene[SceneKey.MODELS]]
        world = models[0]
        for model in models[1:]:
            world.merge_world(model)
        objects = scene.get(SceneKey.OBJECTS) or []
        for entry in objects:
            world.merge_world_at_pose(
                STLParser(str(bundle / entry[SceneKey.MESH])).parse(),
                self.spawn_pose(entry[SceneKey.SPAWN], world),
            )
        self.annotate(world, objects)
        return world

    @staticmethod
    def model(bundle: Path, model: Mapping[str, Any]) -> World:
        """
        :return: One model of the bundle, its mesh paths read relative to the bundle.
        """
        return URDFParser.from_file(
            str(bundle / model[SceneKey.URDF]),
            path_resolver=CompositePathResolver(
                resolvers=[
                    FileUriResolver(base_directory=str(bundle)),
                    PackageUriResolver(),
                ]
            ),
        ).parse()

    @staticmethod
    def spawn_pose(
        spawn: Sequence[float], world: World
    ) -> HomogeneousTransformationMatrix:
        """
        :param spawn: A position, optionally followed by an orientation quaternion.
        :return: The pose relative to the world's root.
        """
        return HomogeneousTransformationMatrix.from_xyz_quaternion(
            *spawn, reference_frame=world.root
        )

    def annotate(self, world: World, objects: Sequence[Mapping[str, Any]]) -> None:
        """
        Annotate every object whose identifier has an annotation class.
        """
        annotations = [
            annotation(root=world.get_body_by_name(Path(entry[SceneKey.MESH]).name))
            for entry in objects
            for annotation in [
                self.annotations.annotation_for(entry[SceneKey.IDENTIFIER])
            ]
            if annotation is not None
        ]
        with world.modify_world():
            world.add_semantic_annotations(annotations)


@dataclass
class SceneWorlds:
    """
    Keeps the world of the scene last asked for, and only it, so the resolver never
    reads two scenes at once.

    .. note::
        krrood's symbol graph holds a world's individuals weakly and keeps their entries
        until it is pruned, and a world refers to itself, so a replaced world leaves the
        graph only once it is collected and the graph pruned.
    """

    builder: SceneWorldBuilder = field(default_factory=SceneWorldBuilder)
    """
    Builds a scene's world.
    """

    scene: Optional[str] = None
    """
    The name of the scene whose world is kept.
    """

    world: Optional[World] = None
    """
    The world kept.
    """

    def world_for(self, scene: str, bundle: Path) -> World:
        """
        :param scene: The scene's name.
        :param bundle: The scene's bundle directory.
        :return: The scene's world, built the first time it is asked for after another
            scene's.
        """
        if self.scene != scene or self.world is None:
            self.replace_with(scene, bundle)
        return self.world

    def initial_world_for(self, scene: str, bundle: Path) -> World:
        """
        :param scene: The scene's name.
        :param bundle: The scene's bundle directory.
        :return: The scene's world as its bundle sets it out, built anew, so nothing done
            in the world kept carries over.
        """
        self.replace_with(scene, bundle)
        return self.world

    def replace_with(self, scene: str, bundle: Path) -> None:
        """
        Keep a newly built world of the scene in place of the world kept, which is
        removed from the symbol graph first.

        :param scene: The scene's name.
        :param bundle: The scene's bundle directory.
        """
        self.world = None
        gc.collect()
        SymbolGraph().remove_dead_instances()
        self.world = self.builder.build(bundle)
        self.scene = scene
