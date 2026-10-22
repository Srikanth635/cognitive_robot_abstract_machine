"""
The worlds the physics viewer puts under MuJoCo physics: scenes described in MJCF, and
robots standing on the floor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path

import mujoco
import semantic_digital_twin
from krrood.patterns.path_enumeration import PathEnumeration
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import ContactCategories, MujocoGeom
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import StaticJointState
from experiments.physics_viewer.placement import Position, ViewerCamera
from experiments.tool_based_actions.simple_demo.demo_world import (
    BOWL_COLOR,
    CUP_COLOR,
    parse_object,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Milk,
    PouringCup,
)
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import JointDynamics
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.contact import (
    ContactFriction,
    ContactImpedance,
    ContactParameters,
    ContactStiffness,
)
from semantic_digital_twin.world_description.geometry import Box, Color, Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Iterable

# %% where the scenes are described


class MjcfFile(PathEnumeration):
    """
    The MJCF files of semantic_digital_twin's resources the scenes are read from.
    """

    DIRECTORY = Path(semantic_digital_twin.__file__).parents[2] / "resources" / "mjcf"
    """
    The directory holding the files.
    """

    FLOOR = DIRECTORY / "floor.xml"
    """
    Nothing but the ground.
    """

    CUBE = DIRECTORY / "cube_on_the_ground.xml"
    """
    A cube dropped onto the ground.
    """

    CUPS = DIRECTORY / "jeroen_cups.xml"
    """
    A set of cups.
    """


class UrdfFile(PathEnumeration):
    """
    The URDF files of semantic_digital_twin's resources the scenes are read from.
    """

    DIRECTORY = Path(semantic_digital_twin.__file__).parents[2] / "resources" / "urdf"
    """
    The directory holding the files.
    """

    APARTMENT = DIRECTORY / "apartment.urdf"
    """
    An apartment with its walls, floor and furniture.
    """


# %% the colors of a URDF


@dataclass(frozen=True)
class UrdfMaterialColors:
    """
    Gives the meshes of a world the colors the URDF it was read from declares.

    The URDF parser gives a material's color to a box, a sphere or a cylinder and leaves a
    mesh white, so the meshes of a robot or an apartment all look white.
    """

    file: str
    """
    The URDF file, or the ROS path of one.
    """

    def apply(self, world: World) -> None:
        """
        Color the meshes of the bodies whose links the URDF gives a material.

        :param world: The world the URDF was read into.
        """
        parsed = URDFParser.from_file(self.file).parsed
        named = {
            material.name: material.color.rgba
            for material in parsed.materials
            if material.color
        }
        for link in parsed.links:
            body = world.get_body_by_name(link.name)
            for visual, shape in zip(link.visuals, body.visual):
                rgba = self.rgba_of(visual.material, named)
                if rgba is not None and isinstance(shape, Mesh):
                    shape.color = Color(*rgba)

    @staticmethod
    def rgba_of(material, named: dict[str, list[float]]) -> list[float] | None:
        """
        :param material: The material of a visual, if it has one.
        :param named: The colors of the URDF's own materials, by their names.
        :return: The color the material declares itself or by its name, if any.
        """
        if material is None:
            return None
        return material.color.rgba if material.color else named.get(material.name)


# %% furniture that rests


@dataclass(frozen=True)
class RestingFurniture:
    """
    Makes the furniture of a scene stay where it is in physics.

    A drawer's collision shape overlaps the cabinet it slides in, which throws every drawer
    and door open the moment the simulation starts, so the shapes of the furniture are put
    in a category of their own, which collides with the default category and not with
    itself. Its joints get the friction that keeps a door from swinging under its own weight.
    """

    category: ContactCategories = ContactCategories(2)
    """
    The category of the furniture's collision shapes.
    """

    dynamics: JointDynamics = field(
        default_factory=lambda: JointDynamics(dry_friction=1.0, damping=0.5)
    )
    """
    The friction of the furniture's joints.
    """

    def apply(self, world: World, bodies: Iterable[Body]) -> None:
        """
        Make the bodies rest.

        :param world: The world the bodies are in.
        :param bodies: The bodies of the furniture.
        """
        furniture = list(bodies)
        for body in furniture:
            for shape in body.collision:
                shape.add_simulator_property(
                    MujocoGeom(
                        contact_type=self.category,
                        contact_affinity=ContactCategories.DEFAULT,
                    )
                )
        for connection in world.connections:
            if (
                isinstance(connection, ActiveConnection1DOF)
                and connection.child in furniture
            ):
                connection.dynamics = self.dynamics


# %% how a scene is simulated


@dataclass(frozen=True)
class SimulationOptions:
    """
    How MuJoCo resolves the contacts of a scene; the defaults are MuJoCo's own.
    """

    cone: mujoco.mjtCone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    """
    The friction cone: the elliptic one is the exact one, where the pyramid only
    approximates it and lets a squeezed object creep between the pads of a gripper.
    """

    impedance_ratio: float = 1.0
    """
    How much harder sliding is than pressing in; at 1 a held object slides out of a firm
    grip under its own weight.
    """

    no_slip_iterations: int = 0
    """
    The passes spent removing the slip soft contacts leave, so a held object stays where
    the pads hold it.
    """

    def keywords(self) -> dict[str, object]:
        """
        :return: What the simulation is built with.
        """
        return {
            "cone": self.cone,
            "impratio": self.impedance_ratio,
            "noslip_iterations": self.no_slip_iterations,
        }


# %% how a scene's world is made


class SceneWorld(ABC):
    """
    How the world of a scene is made.
    """

    @abstractmethod
    def build(self) -> World:
        """
        :return: A new world of the scene.
        """

    def initial_camera(self) -> ViewerCamera | None:
        """
        :return: Where the window's camera is first put, or ``None`` to leave it where
            MuJoCo puts it.
        """
        return None

    def simulation_options(self) -> SimulationOptions:
        """
        :return: How the scene's contacts are resolved.
        """
        return SimulationOptions()


@dataclass(frozen=True)
class MjcfScene(SceneWorld):
    """
    A scene described in an MJCF file.
    """

    file: Path
    """
    The file describing the scene.
    """

    def build(self) -> World:
        return MJCFParser(str(self.file)).parse()


@dataclass(frozen=True)
class UrdfScene(SceneWorld):
    """
    A scene described in a URDF file.
    """

    file: Path
    """
    The file describing the scene.
    """

    def build(self) -> World:
        world = URDFParser.from_file(str(self.file)).parse()
        UrdfMaterialColors(str(self.file)).apply(world)
        RestingFurniture().apply(world, world.bodies)
        return world


@dataclass(frozen=True)
class RobotInScene(SceneWorld):
    """
    A robot spawned into another scene, at a position on the floor, with its arms parked.
    """

    scene: SceneWorld
    """
    The scene the robot is spawned into.
    """

    robot: type[AbstractRobot]
    """
    The robot to spawn.
    """

    position: Position = Position()
    """
    Where the robot stands.
    """

    view: ViewerCamera | None = None
    """
    Where the window's camera is first put.
    """

    def initial_camera(self) -> ViewerCamera | None:
        return self.view

    def build(self) -> World:
        world = self.scene.build()
        robot = RobotSpecification(
            self.robot,
            world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(
                self.position.x, self.position.y, self.position.z
            ),
        ).spawn(world)
        UrdfMaterialColors(self.robot.get_ros_file_path()).apply(world)
        for arm in robot.all_arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        return world


class SceneObject(ABC):
    """
    An object lying in a scene.
    """

    @abstractmethod
    def add_to(self, world: World) -> None:
        """
        Put the object into a world, free to move under physics.

        :param world: The world.
        """


@dataclass(frozen=True)
class PlacedObject(SceneObject):
    """
    An object from a mesh file, lying in a scene.
    """

    mesh: str
    """
    The name of the object's mesh file in the demos' resources, which is also the name of
    its body.
    """

    position: Position
    """
    Where the object lies, the height being where it comes to rest.
    """

    color: Color | None = None
    """
    The color to dye the object with; ``None`` keeps the mesh's own.
    """

    annotation: type[SemanticAnnotation] | None = None
    """
    What the object is, for the actions that look for it.
    """

    def add_to(self, world: World) -> None:
        world.merge_world_at_pose(
            parse_object(self.mesh, self.color),
            HomogeneousTransformationMatrix.from_xyz_rpy(
                self.position.x,
                self.position.y,
                self.position.z,
                reference_frame=world.root,
            ),
        )
        if self.annotation is not None:
            with world.modify_world():
                world.add_semantic_annotation(
                    self.annotation(root=world.get_body_by_name(self.mesh))
                )


@dataclass(frozen=True)
class CubeOnTheTable(SceneObject):
    """
    A cube standing on the table a robot is bolted to, which anything may grasp.
    """

    edge: float
    """
    The length of the cube's edges, in meters.
    """

    x: float
    """
    How far in front of the robot the cube stands.
    """

    y: float
    """
    Where the cube stands sideways.
    """

    color: Color
    """
    The color of the cube.
    """

    table: str = "table"
    """
    The name of the body whose origin is the table's work surface.
    """

    resting_clearance: float = 0.001
    """
    The gap left between the cube and the table, so it rests rather than intersects.
    """

    name: str = "cube"
    """
    The name of the cube's body.
    """

    def add_to(self, world: World) -> None:
        top = float(world.get_body_by_name(self.table).global_pose.z)
        scale = Scale(self.edge, self.edge, self.edge)
        cube = Body(
            name=PrefixedName(self.name),
            collision=ShapeCollection([Box(scale=scale, color=self.color)]),
            visual=ShapeCollection([Box(scale=scale, color=self.color)]),
        )
        with world.modify_world():
            world.add_kinematic_structure_entity(cube)
            world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=world,
                    parent=world.root,
                    child=cube,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        self.x, self.y, top + self.edge / 2 + self.resting_clearance
                    ),
                )
            )
            world.add_semantic_annotation(HasGraspCandidates(root=cube))


@dataclass(frozen=True)
class ObjectsInScene(SceneWorld):
    """
    A scene with objects lying in it, each free to move under physics.
    """

    scene: SceneWorld
    """
    The scene the objects lie in.
    """

    objects: tuple[SceneObject, ...]
    """
    The objects.
    """

    def build(self) -> World:
        world = self.scene.build()
        for placed in self.objects:
            placed.add_to(world)
        return world

    def initial_camera(self) -> ViewerCamera | None:
        return self.scene.initial_camera()

    def simulation_options(self) -> SimulationOptions:
        return self.scene.simulation_options()


@dataclass(frozen=True)
class SolidSurfaces(SceneWorld):
    """
    A scene whose bodies are as solid as what a gripper holds: nothing may sink into them,
    where MuJoCo's soft defaults let a firmly held object sink several millimetres into
    the pads.
    """

    scene: SceneWorld
    """
    The scene.
    """

    options: SimulationOptions = SimulationOptions(
        cone=mujoco.mjtCone.mjCONE_ELLIPTIC, impedance_ratio=10.0, no_slip_iterations=10
    )
    """
    How the contacts are resolved: the friction cone, and the passes against slip, that keep
    a held object where the pads hold it.
    """

    time_constant: timedelta = timedelta(milliseconds=2)
    """
    The time a contact takes to resolve a penetration: two physics steps, as stiff as a
    simulation stepping every millisecond stays stable at.
    """

    def build(self) -> World:
        world = self.scene.build()
        ContactParameters(
            friction=ContactFriction(),
            stiffness=ContactStiffness(time_constant=self.time_constant),
            impedance=ContactImpedance(minimum=0.99, maximum=0.999, width=0.001),
        ).apply_to(world.bodies_with_collision)
        return world

    def initial_camera(self) -> ViewerCamera | None:
        return self.scene.initial_camera()

    def simulation_options(self) -> SimulationOptions:
        return self.options


class ObjectColor(Enum):
    """
    The colors of the objects whose meshes carry none, as the demos dye theirs.
    """

    MILK = Color(0.75, 0.85, 1.0)
    """
    A pale blue, for the milk carton.
    """

    CEREAL = Color(0.95, 0.75, 0.2)
    """
    A yellow, for the cereal box.
    """


# %% the scenes


class PhysicsScene(Enum):
    """
    The scenes the physics viewer offers, each by the way its world is made.
    """

    FLOOR = MjcfScene(MjcfFile.FLOOR)
    """
    Nothing but the ground.
    """

    CUBE = MjcfScene(MjcfFile.CUBE)
    """
    A cube dropped onto the ground.
    """

    CUPS = MjcfScene(MjcfFile.CUPS)
    """
    A set of cups.
    """

    APARTMENT = UrdfScene(UrdfFile.APARTMENT)
    """
    An apartment.
    """

    PR2_IN_APARTMENT = RobotInScene(
        UrdfScene(UrdfFile.APARTMENT), PR2, Position(1.5, 2.5)
    )
    """
    The PR2 in the apartment, where the demos have it stand.
    """

    PR2_AT_THE_COUNTER = RobotInScene(
        ObjectsInScene(
            UrdfScene(UrdfFile.APARTMENT),
            (
                PlacedObject(
                    "milk.stl",
                    Position(2.37, 2.0, 1.0345),
                    ObjectColor.MILK.value,
                    Milk,
                ),
                PlacedObject(
                    "breakfast_cereal.stl",
                    Position(2.37, 1.8, 1.0487),
                    ObjectColor.CEREAL.value,
                ),
                PlacedObject("bowl.stl", Position(2.4, 2.4, 0.979), BOWL_COLOR, Bowl),
                PlacedObject(
                    "jeroen_cup.stl", Position(2.4, 2.6, 0.947), CUP_COLOR, PouringCup
                ),
            ),
        ),
        PR2,
        Position(1.1, 2.2),
        ViewerCamera(
            Position(2.2, 2.2, 1.0), distance=2.5, azimuth=65.0, elevation=-25.0
        ),
    )
    """
    The PR2 at the kitchen counter, with a milk carton, a cereal box, a bowl and a cup on it.
    """

    TRACY_AT_THE_TABLE = SolidSurfaces(
        ObjectsInScene(
            RobotInScene(
                MjcfScene(MjcfFile.FLOOR),
                Tracy,
                view=ViewerCamera(
                    Position(0.45, 0.05, 1.0),
                    distance=1.6,
                    azimuth=120.0,
                    elevation=-25.0,
                ),
            ),
            (CubeOnTheTable(edge=0.05, x=0.55, y=0.1, color=Color(0.8, 0.3, 0.1)),),
        )
    )
    """
    Tracy at its table, with a cube on it.
    """

    PR2 = RobotInScene(MjcfScene(MjcfFile.FLOOR), PR2)
    """
    The PR2 on the floor.
    """

    TRACY = RobotInScene(MjcfScene(MjcfFile.FLOOR), Tracy)
    """
    The Tracy robot on the floor.
    """

    @classmethod
    def from_command_line(cls, name: str) -> PhysicsScene:
        """
        :param name: The scene's name in lower case, as typed.
        :return: The scene.
        :raises KeyError: When no scene has the name.
        """
        return cls[name.upper()]
