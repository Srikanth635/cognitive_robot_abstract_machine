---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Creating Custom Bodies — Self Assessment

This self-assessment quiz tests your understanding of the tutorial: [Creating Custom Bodies](creating-custom-bodies).

Use it to check your comprehension of **Semantic World**, **body creation**, and **geometry handling** concepts.

---

```{jupyterquiz}
{
    "questions": [
        {
            "question": "What is the purpose of the `PrefixedName` data structure in Semantic World?",
            "type": "multiple_choice",
            "answers": [
                {"answer": "To generate unique names for entities, with an optional prefix.", "correct": true},
                {"answer": "To store numerical transformation data.", "correct": false},
                {"answer": "To represent 3D geometry information.", "correct": false},
                {"answer": "To manage collision detection parameters.", "correct": false}
            ],
            "explanation": "`PrefixedName` ensures that each entity has a unique identifier, combining a name and an optional prefix."
        },
        {
            "question": "What are the two main types of shape collections a body can have?",
            "type": "multiple_choice",
            "answers": [
                {"answer": "Visual and collision", "correct": true},
                {"answer": "Static and dynamic", "correct": false},
                {"answer": "Render and physical", "correct": false},
                {"answer": "Rigid and soft", "correct": false}
            ],
            "explanation": "Each body has a visual ShapeCollection for rendering, and a collision ShapeCollection for physics computations."
        },
        {
            "question": "Which shapes are supported in Semantic World’s ShapeCollection?",
            "type": "multiple_select",
            "answers": [
                {"answer": "Box", "correct": true},
                {"answer": "Sphere", "correct": true},
                {"answer": "Cylinder", "correct": true},
                {"answer": "FileMesh/TriangleMesh", "correct": true},
                {"answer": "Cone", "correct": false}
            ],
            "explanation": "Supported shapes include Box, Sphere, Cylinder, and FileMesh (TriangleMesh). Cone is not listed as supported."
        },
        {
            "question": "What is the correct way to add a body to the world?",
            "type": "multiple_choice",
            "answers": [
                {"answer": "world.add_body(body)", "correct": false},
                {"answer": "with world.modify_world(): world.add_body(body)", "correct": true},
                {"answer": "world.create_body(body)", "correct": false},
                {"answer": "world.append(body)", "correct": false}
            ],
            "explanation": "Any modifications to the world must occur inside a `with world.modify_world()` context manager."
        },
        {
            "question": "What does the `RayTracer` class do in this example?",
            "type": "multiple_choice",
            "answers": [
                {"answer": "It traces rays through the world to visualize or compute collisions.", "correct": true},
                {"answer": "It creates transformation matrices for objects.", "correct": false},
                {"answer": "It loads external mesh files for the world.", "correct": false},
                {"answer": "It defines spatial coordinate systems.", "correct": false}
            ],
            "explanation": "`RayTracer` updates and renders the scene, enabling visualization or ray-based computations in the world."
        },
        {
            "question": "What problem might occur if you create multiple unconnected bodies in the world?",
            "type": "multiple_choice",
            "answers": [
                {"answer": "The world validation process will fail.", "correct": true},
                {"answer": "The program will hang indefinitely.", "correct": false},
                {"answer": "The rendering engine will crash.", "correct": false},
                {"answer": "The objects will automatically merge.", "correct": false}
            ],
            "explanation": "Creating multiple unconnected bodies violates world validation rules. Use the world-structure-manipulation tutorial to learn how to connect them."
        },
        {
            "question": "Which transformation method can create a transformation from position and quaternion?",
            "type": "multiple_choice",
            "answers": [
                {"answer": "TransformationMatrix.from_xyz_quaternion()", "correct": true},
                {"answer": "TransformationMatrix.from_point_rotation_matrix()", "correct": false},
                {"answer": "RotationMatrix.from_axis_angle()", "correct": false},
                {"answer": "Point3.from_iterable()", "correct": false}
            ],
            "explanation": "`from_xyz_quaternion()` constructs a TransformationMatrix using position and quaternion data."
        },
        {
            "question": "What is the purpose of `get_semantic_world_directory_root()` in the example?",
            "type": "multiple_choice",
            "answers": [
                {"answer": "It returns the root directory of the Semantic World package, used to locate resource files like meshes.", "correct": true},
                {"answer": "It creates the output folder for the simulation results.", "correct": false},
                {"answer": "It loads all default world entities automatically.", "correct": false},
                {"answer": "It validates all shape definitions in the world.", "correct": false}
            ],
            "explanation": "`get_semantic_world_directory_root()` helps locate the resource directory for file-based meshes (like STL models)."
        },
        {
            "question": "What is the role of the `Scale` class in defining a Box shape?",
            "type": "multiple_choice",
            "answers": [
                {"answer": "It defines the box dimensions along x, y, and z axes.", "correct": true},
                {"answer": "It specifies the box’s rotation.", "correct": false},
                {"answer": "It manages texture mapping on the box.", "correct": false},
                {"answer": "It adjusts the box’s color intensity.", "correct": false}
            ],
            "explanation": "`Scale` determines how large the box will be in each spatial direction."
        },
        {
            "question": "Which of the following correctly visualizes the scene in Jupyter?",
            "type": "multiple_choice",
            "answers": [
                {"answer": "rt.scene.show('jupyter')", "correct": true},
                {"answer": "world.show()", "correct": false},
                {"answer": "body.display()", "correct": false},
                {"answer": "Scene.visualize()", "correct": false}
            ],
            "explanation": "After updating the scene with `rt.update_scene()`, use `rt.scene.show('jupyter')` to render it interactively in a notebook."
        }
    ]
}
