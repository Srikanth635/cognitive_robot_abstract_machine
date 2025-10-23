import builtins
import os
from enum import Enum

from ormatic.ormatic import ORMatic
from ormatic.utils import classes_of_module, recursive_subclasses

import semantic_digital_twin.world_description.degree_of_freedom
import semantic_digital_twin.robots.abstract_robot
import semantic_digital_twin.semantic_annotations.semantic_annotations
import semantic_digital_twin.world_description.world_entity
from semantic_digital_twin.world import (
    ResetStateContextManager,
    WorldModelUpdateContextManager,
)
from semantic_digital_twin.spatial_computations.forward_kinematics import (
    ForwardKinematicsVisitor,
)
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    HasUpdateState,
)
from semantic_digital_twin.orm.model import *
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

# ----------------------------------------------------------------------------------------------------------------------
# This script generates the ORM classes for the semantic_digital_twin package.
# Dataclasses can be mapped automatically to the ORM model
# using the ORMatic library, they just have to be registered in the classes list.
# Classes that are self_mapped and explicitly_mapped are already mapped in the model.py file. Look there for more
# information on how to map them.
# ----------------------------------------------------------------------------------------------------------------------

# create of classes that should be mapped
classes = set(recursive_subclasses(AlternativeMapping))
classes |= set(classes_of_module(semantic_digital_twin.world_description.geometry))
classes |= set(
    classes_of_module(semantic_digital_twin.world_description.shape_collection)
)
classes |= set(classes_of_module(semantic_digital_twin.world))
classes |= set(classes_of_module(semantic_digital_twin.datastructures.prefixed_name))
classes |= set(classes_of_module(semantic_digital_twin.world_description.world_entity))
classes |= set(classes_of_module(semantic_digital_twin.world_description.connections))
classes |= set(
    classes_of_module(semantic_digital_twin.semantic_annotations.semantic_annotations)
)
classes |= set(
    classes_of_module(semantic_digital_twin.world_description.degree_of_freedom)
)
classes |= set(classes_of_module(semantic_digital_twin.robots.abstract_robot))
# classes |= set(recursive_subclasses(ViewFactory))

# remove classes that should not be mapped
classes -= {
    ResetStateContextManager,
    WorldModelUpdateContextManager,
    HasUpdateState,
    World,
    ForwardKinematicsVisitor,
    DegreeOfFreedom,
}
classes -= set(recursive_subclasses(Enum))
classes -= set(recursive_subclasses(Exception))


def generate_orm():
    """
    Generate the ORM classes for the pycram package.
    """
    # Create an ORMatic object with the classes to be mapped
    ormatic = ORMatic(list(classes), type_mappings={trimesh.Trimesh: TrimeshType})

    # Generate the ORM classes
    ormatic.make_all_tables()

    path = os.path.abspath(
        os.path.join(os.getcwd(), "..", "src", "semantic_digital_twin", "orm")
    )
    with builtins.open(os.path.join(path, "ormatic_interface.py"), "w") as f:
        ormatic.to_sqlalchemy_file(f)


if __name__ == "__main__":
    generate_orm()
