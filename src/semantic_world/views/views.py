from dataclasses import dataclass, field

from typing_extensions import List

from semantic_world import PrefixedName
from semantic_world.world import View, Body
from semantic_world.world_entity import EnvironmentView


@dataclass(unsafe_hash=True)
class Handle(View):
    body: Body

    def __post_init__(self):
        self.name = self.body.name


@dataclass(unsafe_hash=True)
class Container(View):
    body: Body

    def __post_init__(self):
        self.name = self.body.name

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
        self.name = self.body.name

@dataclass(unsafe_hash=True)
class Fridge(View):
    body: Body
    door: Door

    def __post_init__(self):
        self.name = self.body.name


@dataclass(unsafe_hash=True)
class Kitchen(EnvironmentView):
    """
    Represents a view of a kitchen.
    """
    fridges: List[Fridge] = field(default_factory=list)

    def __post_init__(self):
        if self.name is None:
            self.name = PrefixedName('kitchen', prefix='environment')


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

@dataclass
class MultiBodyView(View):
    """
    A Generic View for multiple bodies.
    """
    bodies: List[Body] = field(default_factory=list, hash=False)
    views: List[View] = field(default_factory=list, hash=False)

    def add_body(self, body: Body):
        self.bodies.append(body)
        body._views.append(self)

    def add_view(self, view: View):
        self.views.append(view)
        view._views.append(self)