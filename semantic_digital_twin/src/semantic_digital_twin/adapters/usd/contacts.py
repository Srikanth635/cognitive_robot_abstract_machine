"""
Keeping apart, on the stage, the shapes the MuJoCo simulation keeps from colliding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pxr import Sdf, UsdPhysics

from semantic_digital_twin.adapters.multi_sim import ContactCategories, MujocoGeom
from semantic_digital_twin.adapters.usd.stage import UsdStagePath, WorldStage
from semantic_digital_twin.world_description.geometry import Shape

# %% contact categories


@dataclass(frozen=True)
class ShapeContactCategories:
    """
    The contact categories a shape offers and accepts, as the MuJoCo simulation has
    them.
    """

    offered: ContactCategories
    """
    The categories the shape offers when it presses into another.
    """

    accepted: ContactCategories
    """
    The categories the shape accepts pressing into it.
    """

    @classmethod
    def of(cls, shape: Shape) -> ShapeContactCategories:
        """
        :param shape: A shape.
        :return: The categories the shape is declared in, or MuJoCo's default ones.
        """
        geom = shape.get_simulator_property_of_type(MujocoGeom) or MujocoGeom()
        return cls(offered=geom.contact_type, accepted=geom.contact_affinity)

    def collides_with(self, other: ShapeContactCategories) -> bool:
        """
        :param other: The categories of another shape.
        :return: Whether a shape of these categories collides with one of the other's.
        """
        return bool(self.offered & other.accepted) or bool(
            other.offered & self.accepted
        )


# %% the writer


@dataclass
class ContactGroupWriter:
    """
    Writes the collision groups that keep shapes of contact categories that do not
    collide apart.
    """

    world_stage: WorldStage
    """
    The stage being written.
    """

    shapes_by_categories: dict[ShapeContactCategories, list[Sdf.Path]] = field(
        default_factory=dict
    )
    """
    The collision shapes written so far, by the contact categories they are in.
    """

    def add(self, shape: Shape, path: Sdf.Path) -> None:
        """
        :param shape: A collision shape written onto the stage.
        :param path: The prim it was written as.
        """
        self.shapes_by_categories.setdefault(
            ShapeContactCategories.of(shape), []
        ).append(path)

    def write(self) -> None:
        """
        Keep shapes of contact categories that do not collide apart, each category's
        shapes a collision group that filters out the groups it does not collide with;
        where every category collides with every other, nothing is written.
        """
        categories = list(self.shapes_by_categories)
        if all(a.collides_with(b) for a in categories for b in categories):
            return
        groups = {}
        for category in categories:
            path = Sdf.Path(UsdStagePath.COLLISION_GROUPS.value).AppendChild(
                self.world_stage.prim_name(
                    f"contact_categories_{int(category.offered)}_{int(category.accepted)}"
                )
            )
            group = UsdPhysics.CollisionGroup.Define(self.world_stage.stage, path)
            group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(
                self.shapes_by_categories[category]
            )
            groups[category] = group
        for category, group in groups.items():
            group.CreateFilteredGroupsRel().SetTargets(
                [
                    groups[other].GetPath()
                    for other in categories
                    if not category.collides_with(other)
                ]
            )
