"""
The world a scene bundle stands for: its models and its objects as a
semantic_digital_twin world, each object annotated as what it is, the robot on a drive
over its surroundings where the recording starts, so llm_reasoner's resolver reads the
scene the viewer shows and coraplex can act in it.

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
from semantic_digital_twin.world_description.connections import OmniDrive
from typing_extensions import Any, Mapping, Optional, Sequence

OBJECT_ANNOTATIONS_FILE = Path(__file__).parent / "object_annotations.json"
"""
Which annotation class each scene object is, by the object's identifier.
"""

ROBOT_ANNOTATIONS_FILE = Path(__file__).parent / "robot_annotations.json"
"""
Which robot class each scene robot is, by the robot's name.
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
    ROBOT = "robot"
    NAME = "name"
    TRAJECTORY = "trajectory"
    OBJECTS = "objects"
    IDENTIFIER = "id"
    MESH = "mesh"
    SPAWN = "spawn"


class TrajectoryKey(StrEnum):
    """
    The keys of a scene's recorded trajectory.
    """

    BASE = "base"


@dataclass
class UnknownRobot(LookupError):
    """
    No robot class is kept for the scene's robot.
    """

    name: str
    """
    The robot's name in the scene.
    """

    def __post_init__(self):
        super().__init__(f"no robot class is kept for the scene's robot '{self.name}'")


@dataclass(frozen=True)
class AnnotationClasses:
    """
    The annotation class of each scene object or robot that has one, by its name in the
    scene.
    """

    by_name: Mapping[str, type]
    """
    The annotation class, by the name of what it annotates.
    """

    @classmethod
    def read(cls, path: Path) -> AnnotationClasses:
        """
        :return: The annotation classes kept as data in the file.
        """
        stored = json.loads(path.read_text())
        return cls(
            by_name={
                name: resolve_class_from_full_name(class_name)
                for name, class_name in stored.items()
            }
        )

    @classmethod
    def of_objects(cls) -> AnnotationClasses:
        """
        :return: The annotation class of each scene object, by its identifier.
        """
        return cls.read(OBJECT_ANNOTATIONS_FILE)

    @classmethod
    def of_robots(cls) -> AnnotationClasses:
        """
        :return: The robot class of each scene robot, by its name.
        """
        return cls.read(ROBOT_ANNOTATIONS_FILE)

    def annotation_for(self, name: str) -> Optional[type]:
        """
        :return: The annotation class of what has the given name, ``None`` when it has
            none.
        """
        return self.by_name.get(name)


@dataclass
class SceneWorldBuilder:
    """
    Builds the world a scene bundle stands for.
    """

    annotations: AnnotationClasses = field(default_factory=AnnotationClasses.of_objects)
    """
    Which annotation each scene object gets.
    """

    robot_annotations: AnnotationClasses = field(
        default_factory=AnnotationClasses.of_robots
    )
    """
    Which robot class the scene's robot is.
    """

    def build(self, bundle: Path) -> World:
        """
        :param bundle: The directory of a scene bundle.
        :return: The bundle's models in one world, the robot on a drive over its
            surroundings where the recording starts, with every object at its spawn pose
            and annotated as what it is.
        """
        scene = json.loads((bundle / SCENE_FILE).read_text())
        world = self.models_world(bundle, scene)
        objects = scene.get(SceneKey.OBJECTS) or []
        for entry in objects:
            world.merge_world_at_pose(
                STLParser(str(bundle / entry[SceneKey.MESH])).parse(),
                self.spawn_pose(entry[SceneKey.SPAWN], world),
            )
        self.annotate(world, objects)
        return world

    @classmethod
    def robot_world(cls, bundle: Path) -> World:
        """
        :return: The scene's world built anew, its robot annotated as what it is, so
            coraplex can act in it.
        """
        builder = cls()
        world = builder.build(bundle)
        builder.annotate_robot(world, bundle)
        return world

    def annotate_robot(self, world: World, bundle: Path) -> Any:
        """
        Annotate the scene's robot as the robot class kept for its name.

        :param world: The scene's world.
        :param bundle: The scene's bundle directory.
        :return: The robot's annotation.
        :raises UnknownRobot: When no robot class is kept for the robot.
        """
        name = self.robot_name(bundle)
        robot_class = self.robot_annotations.annotation_for(name)
        if robot_class is None:
            raise UnknownRobot(name=name)
        return robot_class.from_world(world)

    def annotate_known_robot(self, world: World, bundle: Path) -> Optional[Any]:
        """
        Annotate the scene's robot when a robot class is kept for its name, so the world
        holds the robot as what it is; a robot with none kept is left as it is.

        :param world: The scene's world.
        :param bundle: The scene's bundle directory.
        :return: The robot's annotation, ``None`` when no robot class is kept for it.
        """
        if self.robot_annotations.annotation_for(self.robot_name(bundle)) is None:
            return None
        return self.annotate_robot(world, bundle)

    @staticmethod
    def robot_name(bundle: Path) -> str:
        """
        :return: The name of the scene's robot.
        """
        scene = json.loads((bundle / SCENE_FILE).read_text())
        return scene[SceneKey.ROBOT][SceneKey.NAME]

    def models_world(self, bundle: Path, scene: Mapping[str, Any]) -> World:
        """
        :return: The scene's models in one world: the surroundings merged, the robot on
            a drive over them at its recorded starting pose; a scene of a robot alone is
            the robot's world.
        """
        models = scene[SceneKey.MODELS]
        surroundings = [
            self.model(bundle, model)
            for model in models
            if not model.get(SceneKey.ROBOT)
        ]
        robots = [
            self.model(bundle, model) for model in models if model.get(SceneKey.ROBOT)
        ]
        if not surroundings:
            return self.merged(robots)
        world = self.merged(surroundings)
        for robot in robots:
            self.drive(world, robot, self.starting_base_pose(bundle, scene, world))
        return world

    @staticmethod
    def merged(worlds: Sequence[World]) -> World:
        """
        :return: The worlds merged into the first.
        """
        world, *others = worlds
        for other in others:
            world.merge_world(other)
        return world

    @staticmethod
    def drive(
        world: World, robot: World, pose: HomogeneousTransformationMatrix
    ) -> None:
        """
        Merge a robot into the world on a drive over its root, standing at the pose.
        """
        with world.modify_world():
            drive = OmniDrive.create_with_dofs(
                parent=world.root, child=robot.root, world=world
            )
            world.merge_world(robot, drive)
            drive.origin = pose

    @staticmethod
    def starting_base_pose(
        bundle: Path, scene: Mapping[str, Any], world: World
    ) -> HomogeneousTransformationMatrix:
        """
        :return: The robot's base pose in the first frame of the scene's recording.
        """
        trajectory = json.loads((bundle / scene[SceneKey.TRAJECTORY]).read_text())
        return HomogeneousTransformationMatrix.from_xyz_quaternion(
            *trajectory[TrajectoryKey.BASE][0], reference_frame=world.root
        )

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
    reads two scenes at once. The scene's robot is annotated as what it is when a robot
    class is kept for it, so the world holds the robot's parts and what its model says
    about them.

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
        world = self.builder.build(bundle)
        self.builder.annotate_known_robot(world, bundle)
        self.world = world
        self.scene = scene
