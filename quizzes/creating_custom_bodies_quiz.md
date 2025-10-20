---
jupytext:
    formats: md:myst
    text_representation:
        extension: .md
        format_name: myst
kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

(creating-custom-bodies-quiz)=
# Creating Custom Bodies — Self Assessment

This page provides a self-check quiz for the tutorial: [](creating-custom-bodies).  
Source: [Jupyter quiz](https://pypi.org/project/jupyterquiz/#description). $ $

% NOTE: The lone `$ $` above ensures some math is rendered before the quiz,
% which fixes a known math-rendering quirk inside the quiz widget.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz

questions = [
    {
      "question": "What is the purpose of the `PrefixedName` data structure?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "To generate unique names for entities with an optional prefix.", "correct": True },
        { "answer": "To store rotation matrices.", "correct": False },
        { "answer": "To define collision geometry.", "correct": False },
        { "answer": "To manage ray-tracing parameters.", "correct": False }
      ],
      "explanation": "`PrefixedName` combines a name with an optional prefix to keep entity identifiers unique."
    },
    {
      "question": "Which two shape collections can a body have?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "Visual and collision", "correct": True },
        { "answer": "Static and dynamic", "correct": False },
        { "answer": "Physical and logical", "correct": False },
        { "answer": "Primary and secondary", "correct": False }
      ],
      "explanation": "Visual is for rendering; collision is for physics/collision computations."
    },
    {
      "question": "Which shapes are supported?",
      "type": "multiple_select",
      "answers": [
        { "answer": "Box", "correct": True },
        { "answer": "Sphere", "correct": True },
        { "answer": "Cylinder", "correct": True },
        { "answer": "FileMesh/TriangleMesh", "correct": True },
        { "answer": "Cone", "correct": False }
      ],
      "explanation": "Box, Sphere, Cylinder, and FileMesh (TriangleMesh) are supported; Cone is not listed."
    },
    {
      "question": "How should you add a body to the world?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "with world.modify_world(): world.add_body(body)", "correct": True },
        { "answer": "world.add_body(body)", "correct": False },
        { "answer": "world.create_body(body)", "correct": False },
        { "answer": "world.append(body)", "correct": False }
      ],
      "explanation": "All changes to the world must happen within `with world.modify_world():`."
    },
    {
      "question": "What does the `RayTracer` do in the example?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "It traces rays for visualization or computation and can render the scene.", "correct": True },
        { "answer": "It loads STL meshes.", "correct": False },
        { "answer": "It defines coordinate frames.", "correct": False },
        { "answer": "It manages world validation.", "correct": False }
      ],
      "explanation": "After `rt.update_scene()`, `rt.scene.show('jupyter')` visualizes the scene."
    },
    {
      "question": "What issue can occur when creating multiple unconnected bodies?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "World validation fails.", "correct": True },
        { "answer": "Rendering always crashes.", "correct": False },
        { "answer": "Meshes are auto-merged.", "correct": False },
        { "answer": "Textures are dropped.", "correct": False }
      ],
      "explanation": "Unconnected bodies trigger validation problems; see the world-structure-manipulation tutorial."
    },
    {
      "question": "Which method constructs a transform from position and quaternion?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "TransformationMatrix.from_xyz_quaternion()", "correct": True },
        { "answer": "TransformationMatrix.from_point_rotation_matrix()", "correct": False },
        { "answer": "RotationMatrix.from_axis_angle()", "correct": False },
        { "answer": "Point3.from_iterable()", "correct": False }
      ],
      "explanation": "`from_xyz_quaternion()` builds a transform from position and quaternion components."
    },
    {
      "question": "What does `get_semantic_world_directory_root()` help with?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "Locating the Semantic World resource root (e.g., STL files).", "correct": True },
        { "answer": "Creating output folders.", "correct": False },
        { "answer": "Validating shapes.", "correct": False },
        { "answer": "Spawning default entities.", "correct": False }
      ],
      "explanation": "It’s used to find packaged resources like `resources/stl/milk.stl`."
    },
    {
      "question": "What does the `Scale` class specify for a `Box`?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "Dimensions along x, y, z.", "correct": True },
        { "answer": "Rotation about axes.", "correct": False },
        { "answer": "Texture mapping parameters.", "correct": False },
        { "answer": "Color intensity.", "correct": False }
      ],
      "explanation": "Scale sets the box size on each axis."
    },
    {
      "question": "How do you visualize the scene in Jupyter in the example?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "rt.scene.show('jupyter')", "correct": True },
        { "answer": "world.show()", "correct": False },
        { "answer": "body.display()", "correct": False },
        { "answer": "Scene.visualize()", "correct": False }
      ],
      "explanation": "Call after `rt.update_scene()` to render."
    }
  ]

display_quiz(questions)
```