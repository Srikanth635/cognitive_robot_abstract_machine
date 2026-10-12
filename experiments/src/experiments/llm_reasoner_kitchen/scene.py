"""
The breakfast scene the cases are resolved in: the apartment with a bowl, a pouring cup,
bread and a cutting knife on the counter, each annotated as what it is, and optionally
the furniture the apartment's rule-based annotator finds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import semantic_digital_twin.reasoning.world_rdr.rules  # noqa: F401 (loaded by a running CRAM process)
from semantic_digital_twin.reasoning.world_rdr import world_rdr
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Bread,
    Cereal,
    CuttingKnife,
    PouringCup,
)
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from typing_extensions import Tuple

from experiments.tool_based_actions.simple_demo.demo_world import parse_object


class SceneBody(StrEnum):
    """
    The bodies of the scene the cases name, each by its name: an object's body is named
    after the mesh file it is made from.
    """

    BOWL = "bowl.stl"
    CUP = "jeroen_cup.stl"
    BREAD = "bread.stl"
    KNIFE = "big-knife.stl"
    CEREAL = "breakfast_cereal.stl"
    MILK = "milk.stl"
    SPOON = "spoon.stl"
    SINK = "sink"


@dataclass(frozen=True)
class PlacedObject:
    """
    An object added to the scene, annotated as what it is.
    """

    mesh: SceneBody
    """
    The mesh the object is made from, which names its body.
    """

    position: Tuple[float, float, float]
    """
    Where the object stands relative to the world's root, in metres.
    """

    annotation: type[HasRootBody]
    """
    What the object is, annotated on its body.
    """


@dataclass(frozen=True)
class AnnotatedObject:
    """
    An object the apartment already holds, annotated as what it is.
    """

    mesh: SceneBody
    """
    The mesh the object is made from, which names its body.
    """

    annotation: type[HasRootBody]
    """
    What the object is, annotated on its body.
    """


BREAKFAST_ITEMS = (
    PlacedObject(SceneBody.BOWL, (2.4, 2.2, 1.0), Bowl),
    PlacedObject(SceneBody.CUP, (2.4, 1.6, 1.0), PouringCup),
    PlacedObject(SceneBody.BREAD, (2.45, 1.3, 1.0), Bread),
    PlacedObject(SceneBody.KNIFE, (2.45, 1.1, 0.98), CuttingKnife),
)
"""
The items added to the kitchen counter, where the live checks placed them.
"""

APARTMENT_ANNOTATIONS = (AnnotatedObject(SceneBody.CEREAL, Cereal),)
"""
The apartment's own objects annotated beyond what the apartment annotates itself.
"""


@dataclass(frozen=True)
class BreakfastScene:
    """
    Sets out the breakfast scene in a world holding the PR2 in the apartment.
    """

    items: Tuple[PlacedObject, ...] = field(default=BREAKFAST_ITEMS)
    """
    The items added to the counter.
    """

    annotated: Tuple[AnnotatedObject, ...] = field(default=APARTMENT_ANNOTATIONS)
    """
    The apartment's own objects annotated further.
    """

    furnished: bool = False
    """
    Whether the furniture the apartment's rule-based annotator finds is annotated too,
    such as the sink.
    """

    def set_out(self, world: World) -> World:
        """
        :param world: A world holding the PR2 in the apartment.
        :return: The same world, with the scene set out in it.
        """
        for item in self.items:
            world.merge_world_at_pose(
                parse_object(item.mesh),
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    *item.position, reference_frame=world.root
                ),
            )
        with world.modify_world():
            world.add_semantic_annotations(
                [
                    placed.annotation(root=world.get_body_by_name(placed.mesh))
                    for placed in (*self.items, *self.annotated)
                ]
            )
        if self.furnished:
            self.annotate_furniture(world)
        return world

    @staticmethod
    def annotate_furniture(world: World) -> None:
        """
        Annotate what the apartment's rule-based annotator finds, leaving out the bodies
        that are annotated already.
        """
        annotated_roots = {
            id(annotation.root)
            for annotation in world.semantic_annotations
            if isinstance(annotation, HasRootBody)
        }
        found = world_rdr.classify(world)["semantic_annotations"]
        with world.modify_world():
            for annotation in found:
                if (
                    not isinstance(annotation, HasRootBody)
                    or id(annotation.root) not in annotated_roots
                ):
                    world.add_semantic_annotation_recursively(annotation)
