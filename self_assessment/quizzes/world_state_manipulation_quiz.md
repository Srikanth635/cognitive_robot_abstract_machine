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

(world-state-manipulation-quiz)=
# World State Manipulation — Self Assessment

This page provides a self-check quiz for the tutorial: [](world-state-manipulation).  
Source: Jupyter quiz. $ $

% NOTE: The lone `$ $` above ensures some math is rendered before the quiz,
% which fixes a known math-rendering quirk inside the quiz widget.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz

questions = [
    {
      "question": "Which class aggregates the positions/velocities/accelerations/jerks of all DoFs?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "WorldState", "correct": True},
        {"answer": "RayTracer", "correct": False},
        {"answer": "URDFParser", "correct": False},
        {"answer": "ShapeCollection", "correct": False}
      ],
      "explanation": "semantic_world.world_description.world_state.WorldState stores DoF states."
    },
    {
      "question": "How is a free 6DoF connection created in the example?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Connection6DoF(parent, child, _world=world)", "correct": True},
        {"answer": "RevoluteConnection(parent, child)", "correct": False},
        {"answer": "FixedConnection(parent, child)", "correct": False},
        {"answer": "AddBodyConnection(parent, child)", "correct": False}
      ],
      "explanation": "A passive 6DoF connection is created to allow free motion."
    },
    {
      "question": "What context is required when changing the world (e.g., adding bodies or connections)?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "with world.modify_world():", "correct": True},
        {"answer": "with world.lock():", "correct": False},
        {"answer": "world.begin(); ...; world.end();", "correct": False},
        {"answer": "No special context is required", "correct": False}
      ],
      "explanation": "Structural changes must be grouped in world.modify_world()."
    },
    {
      "question": "How can you set the pose of a free connection in the example?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Assign a TransformationMatrix via parent_T_connection_expression", "correct": True},
        {"answer": "Call set_pose(xyzrpy)", "correct": False},
        {"answer": "Modify the child's visual color", "correct": False},
        {"answer": "Use WorldState.set_pose(connection)", "correct": False}
      ],
      "explanation": "The pose is specified by setting parent_T_connection_expression to a TransformationMatrix."
    },
    {
      "question": "How is a specific Drawer view retrieved using EQL in the example?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "the(entity(let(type_=Drawer, domain=world.views)))", "correct": True},
        {"answer": "world.get_view_by_type(Drawer)", "correct": False},
        {"answer": "select Drawer from views", "correct": False},
        {"answer": "find_view('Drawer')", "correct": False}
      ],
      "explanation": "EQL's the(entity(...)) is used within symbolic_mode()."
    }
]

display_quiz(questions)
```
