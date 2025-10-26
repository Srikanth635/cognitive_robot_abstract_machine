# ----------------------------------------------------------------------------------------------------------------------
# This script generates the ORM classes for the semantic_digital_twin package.
# Dataclasses can be mapped automatically to the ORM model
# using the ORMatic library, they just have to be registered in the classes list.
# Classes that are self_mapped and explicitly_mapped are already mapped in the model.py file. Look there for more
# information on how to map them.
# ----------------------------------------------------------------------------------------------------------------------

import os
from dataclasses import is_dataclass

import krrood.entity_query_language.orm.model
import krrood.entity_query_language.symbol_graph
from krrood.class_diagrams import ClassDiagram
from krrood.entity_query_language.predicate import Predicate, HasTypes, HasType, Symbol
from krrood.ormatic.dao import AlternativeMapping
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import classes_of_module, recursive_subclasses

import semantic_digital_twin.robots.abstract_robot
import semantic_digital_twin.semantic_annotations.semantic_annotations
import semantic_digital_twin.world_description.degree_of_freedom
import semantic_digital_twin.world_description.world_entity
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.orm.model import *
from semantic_digital_twin.spatial_computations.forward_kinematics import (
    ForwardKinematicsVisitor,
)
from semantic_digital_twin.world import (
    ResetStateContextManager,
    WorldModelUpdateContextManager,
)
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    HasUpdateState,
)

# build the symbol graph
Predicate.build_symbol_graph()
symbol_graph = Predicate.symbol_graph

# collect all KRROOD classes
all_classes = {c.clazz for c in symbol_graph._type_graph.wrapped_classes}
all_classes |= {am.original_class() for am in recursive_subclasses(AlternativeMapping)}
all_classes |= set(classes_of_module(krrood.entity_query_language.symbol_graph))
all_classes |= {Symbol}

# remove classes that don't need persistence
all_classes -= {HasType, HasTypes}


# collect all semantic digital twin classes that should be mapped
all_classes |= set(classes_of_module(semantic_digital_twin.world_description.geometry))
all_classes |= set(
    classes_of_module(semantic_digital_twin.world_description.shape_collection)
)
all_classes |= set(classes_of_module(semantic_digital_twin.world))
all_classes |= set(
    classes_of_module(semantic_digital_twin.datastructures.prefixed_name)
)
all_classes |= set(
    classes_of_module(semantic_digital_twin.world_description.world_entity)
)
all_classes |= set(
    classes_of_module(semantic_digital_twin.world_description.connections)
)
all_classes |= set(
    classes_of_module(semantic_digital_twin.semantic_annotations.semantic_annotations)
)
all_classes |= set(
    classes_of_module(semantic_digital_twin.world_description.degree_of_freedom)
)
all_classes |= set(classes_of_module(semantic_digital_twin.robots.abstract_robot))


# remove classes that should not be mapped
all_classes -= {
    ResetStateContextManager,
    WorldModelUpdateContextManager,
    HasUpdateState,
    World,
    ForwardKinematicsVisitor,
    DegreeOfFreedom,
}

# remove classes that are not dataclasses
all_classes = {c for c in all_classes if is_dataclass(c)}


def generate_orm():
    """
    Generate the ORM classes for the pycram package.
    """
    class_diagram = ClassDiagram(
        list(sorted(all_classes, key=lambda c: c.__name__, reverse=True))
    )

    instance = ORMatic(
        class_dependency_graph=class_diagram,
        type_mappings={trimesh.Trimesh: TrimeshType},
        alternative_mappings=recursive_subclasses(AlternativeMapping),
    )

    instance.make_all_tables()

    path = os.path.abspath(
        os.path.join(os.getcwd(), "..", "src", "semantic_digital_twin", "orm")
    )
    with open(os.path.join(path, "ormatic_interface.py"), "w") as f:
        instance.to_sqlalchemy_file(f)


if __name__ == "__main__":
    generate_orm()
