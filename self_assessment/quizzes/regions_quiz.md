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

(regions-quiz)=
# Regions — Self Assessment

This page provides a self-check quiz for the tutorial: [](regions).  
Source: Jupyter quiz. $ $

% NOTE: The lone `$ $` above ensures some math is rendered before the quiz,
% which fixes a known math-rendering quirk inside the quiz widget.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz

questions = [
    {
      "question": "What does a Region represent in the semantic world?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "A semantic area in the kinematic tree (e.g., table surface)", "correct": True},
        {"answer": "A physical body used for collision only", "correct": False},
        {"answer": "A database table", "correct": False},
        {"answer": "A ROS2 node", "correct": False}
      ],
      "explanation": "Regions are semantic areas that live in the same kinematic tree."
    },
    {
      "question": "How is a Region connected in the world?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Like bodies, via connections (e.g., FixedConnection)", "correct": True},
        {"answer": "It floats without connections", "correct": False},
        {"answer": "It is attached via textures", "correct": False},
        {"answer": "It uses special region-only connectors", "correct": False}
      ],
      "explanation": "Regions are connected just like bodies are."
    },
    {
      "question": "Which attribute specifies the geometric area of a Region?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "area", "correct": True},
        {"answer": "collision", "correct": False},
        {"answer": "visual", "correct": False},
        {"answer": "volume", "correct": False}
      ],
      "explanation": "The tutorial sets table_surface.area = [surface]."
    },
    {
      "question": "What code block is used when adding the Region to the world?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "with world.modify_world():", "correct": True},
        {"answer": "with world.transaction():", "correct": False},
        {"answer": "with world.write_lock():", "correct": False},
        {"answer": "No block is required", "correct": False}
      ],
      "explanation": "Structural updates are grouped in world.modify_world()."
    },
    {
      "question": "If the supporting table moves, what happens to the attached Region?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "It moves accordingly because it is connected", "correct": True},
        {"answer": "It remains fixed at the world origin", "correct": False},
        {"answer": "It randomly jitters", "correct": False},
        {"answer": "It scales with the table's bounding box", "correct": False}
      ],
      "explanation": "The Region follows its connection to the table top."
    }
]

display_quiz(questions)
```
