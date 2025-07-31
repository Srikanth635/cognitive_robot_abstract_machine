from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from probabilistic_model.probabilistic_circuit.rx.helper import uniform_measure_of_event
from typing_extensions import List

from semantic_world.geometry import BoundingBox, BoundingBoxCollection
from semantic_world.prefixed_name import PrefixedName
from semantic_world.spatial_types import Point3
from semantic_world.variables import SpatialVariables
from semantic_world.connections import ActiveConnection
from semantic_world.world import View, Body


@dataclass
class ControlledConnections(View):
    connections: List[ActiveConnection] = field(default_factory=list)

    def __hash__(self):
        return hash(tuple(self.connections))

    def compute_chain_reduced_to_controlled_joints(self, root: Body, tip: Body) -> Tuple[Body, Body]:
        """
        1. Compute the kinematic chain of bodies between root and tip.
        2. Remove all entries from link_a downward until one is connected with a connection from this view.
        2. Remove all entries from link_b upward until one is connected with a connection from this view.
        :param root: start of the chain
        :param tip: end of the chain
        :return: start and end link of the reduced chain
        """
        downward_chain, upward_chain = self._world.compute_split_chain_of_connections(root=root, tip=tip)
        chain = downward_chain + upward_chain
        for i, connection in enumerate(chain):
            if connection in self.connections:
                new_root = connection
                break
        else:
            raise KeyError(f'no controlled connection in chain between {root} and {tip}')
        for i, connection in enumerate(reversed(chain)):
            if connection in self.connections:
                new_tip = connection
                break
        else:
            raise KeyError(f'no controlled connection in chain between {root} and {tip}')

        if new_root in upward_chain:
            new_root_body = new_root.parent
        else:  # if new_root is in the downward chain, we need to "flip" it by returning its child
            new_root_body = new_root.child
        if new_tip in upward_chain:
            new_tip_body = new_tip.child
        else:  # if new_root is in the downward chain, we need to "flip" it by returning its parent
            new_tip_body = new_tip.parent
        return new_root_body, new_tip_body


@dataclass(unsafe_hash=True)
class Handle(View):
    body: Body

    def __post_init__(self):
        self.name = PrefixedName(str(self.body.name), self.__class__.__name__)


@dataclass(unsafe_hash=True)
class Container(View):
    body: Body

    def __post_init__(self):
        self.name = PrefixedName(str(self.body.name), self.__class__.__name__)


@dataclass(unsafe_hash=True)
class Door(View):  # Door has a Footprint
    """
    Door in a body that has a Handle and can open towards or away from the user.
    """
    handle: Handle
    body: Body

    def __post_init__(self):
        self.name = PrefixedName(str(self.body.name), self.__class__.__name__)

@dataclass(unsafe_hash=True)
class Fridge(View):
    body: Body
    door: Door

    def __post_init__(self):
        self.name = PrefixedName(str(self.body.name), self.__class__.__name__)

@dataclass(unsafe_hash=True)
class Table(View):
    """
    A view that represents a table.
    """

    top: Body
    """
    The body that represents the table's top surface.
    """

    def points_on_table(self, amount: int = 100) -> List[Point3]:
        """
        Get points that are on the table.

        :amount: The number of points to return.
        :returns: A list of points that are on the table.
        """
        area_of_table = BoundingBoxCollection.from_shapes(self.top.collision)
        event = area_of_table.event
        p = uniform_measure_of_event(event)
        p = p.marginal(SpatialVariables.xy)
        samples = p.sample(amount)
        z_coordinate = np.full((amount, 1), max([b.max_z for b in area_of_table]) + 0.01)
        samples = np.concatenate((samples, z_coordinate), axis=1)
        return [Point3(*s, reference_frame=self.top) for s in samples]

    def __post_init__(self):
        self.name = self.top.name

################################


@dataclass(unsafe_hash=True)
class Components(View):
    ...


@dataclass(unsafe_hash=True)
class Furniture(View):
    ...


#################### subclasses von Components


@dataclass(unsafe_hash=True)
class Door(Components):
    body: Body
    handle: Handle

    def __post_init__(self):
        self.name = PrefixedName(str(self.body.name), self.__class__.__name__)


@dataclass(unsafe_hash=True)
class Drawer(Components):
    container: Container
    handle: Handle

    def __post_init__(self):
        self.name = self.container.name


############################### subclasses to Furniture
@dataclass
class Cupboard(Furniture):
    ...


############################### subclasses to Cupboard
@dataclass(unsafe_hash=True)
class Cabinet(Cupboard):
    container: Container
    drawers: list[Drawer] = field(default_factory=list, hash=False)

    def __post_init__(self):
        self.name = self.container.name


@dataclass
class Wardrobe(Cupboard):
    doors: List[Door] = field(default_factory=list)
