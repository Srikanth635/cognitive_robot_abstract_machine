---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Semantic Annotations (SemanticAnnotations) (Exercise)

This exercise introduces custom SemanticAnnotations and querying them with the Entity Query Language (EQL).

You will:
- Define a custom Bottle SemanticAnnotation
- Annotate two bodies with cylinders as collision as Bottle and query them using EQL

## 0. Setup

```{code-cell} ipython3
:tags: [remove-input]
from dataclasses import dataclass

from entity_query_language import entity, an, let, symbolic_mode

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import TransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation, Body
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.spatial_computations.raytracer import RayTracer

world = World()
```

## 1. Define a Bottle semantic_annotation and annotate two bodies
Your goal:
- Define a custom dataclass Bottle(SemanticAnnotation) that declares a Body to be a Bottle
- Create one body and give it a Cylinder as collision and visual geometry
- Add the body to the world and register a Bottle semantic_annotation for it

Hints:
- Use PrefixedName(str(self.body.name), self.__class__.__name__) to set a default name in __post_init__ when no name is provided.
- The Cylinder should be created with its origin referencing the corresponding body: TransformationMatrix()

```{code-cell} ipython3
:tags: [exercise]
# TODO: Define a Bottle semantic_annotation, create two bodies with Cylinder collisions, connect them under `root`,
#       and register Bottle semantic_annotations for them in `world`.

@dataclass
class Bottle(SemanticAnnotation):
    # Define the semantic_annotation interface here
    ...

# Create and annotate two bodies as Bottles
cylinder = ...
bottle_body = ...
```

```{code-cell} ipython3
:tags: [example-solution]
@dataclass
class Bottle(SemanticAnnotation):
    """Declares that a Body is a Bottle."""
    body: Body

    def __post_init__(self):
        if self.name is None:
            self.name = PrefixedName(str(self.body.name), self.__class__.__name__)

cylinder = Cylinder(width=0.06, height=0.25, origin=TransformationMatrix())
shape_collection = ShapeCollection([cylinder])
bottle_body = Body(name=PrefixedName("bottle1"),
                     collision=shape_collection,
                     visual=shape_collection)
with world.modify_world():
    world.add_body(bottle_body)
    world.add_semantic_annotation(Bottle(body=bottle_body))
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
assert len(world.semantic_annotations) == 1, "There should be exactly one semantic_annotation in the world."
assert world.get_semantic_annotations_by_type(Bottle) != [], "There should be a Bottle semantic_annotation in the world."
assert world.get_semantic_annotations_by_type(Bottle)[0].body == bottle_body, "The Bottle semantic_annotation should reference the correct body."
rt = RayTracer(world); rt.update_scene(); rt.scene.show("jupyter")
```

## 2. Query for bottles with EQL
Your goal:
- Build an EQL query that returns all Bottle semantic_annotations in the world
- Store the query in a variable named `bottles_query`

```{code-cell} ipython3
:tags: [exercise]
# TODO: create an EQL query for Bottle semantic_annotations and store the result of the query in `query_result`.

query_result: list[Bottle] = ...
```

```{code-cell} ipython3
:tags: [example-solution]
with symbolic_mode():
    bottles_query = an(entity(let(Bottle, world.semantic_annotations)))
    
query_result = list(bottles_query.evaluate())
print(query_result)
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
assert query_result is not ..., "The query result should be stored in a variable."
assert len(query_result) == 1, "There should be exactly one Bottle returned by the query."
```
