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

(semantic-annotations-quiz)=
# Semantic Annotations Quiz

This page provides a self-check quiz for the tutorial: [](semantic_annotations).  
Source: Jupyter quiz. $ $

% NOTE: The lone `$ $` above ensures some math is rendered before the quiz,
% which fixes a known math-rendering quirk inside the quiz widget.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz

questions = [
    {
      "question": "What is a SemanticAnnotation in the semantic world?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "A semantic annotation attached to world entities", "correct": True},
        {"answer": "A mesh loader for STL files", "correct": False},
        {"answer": "A plotting backend", "correct": False},
        {"answer": "A physics engine", "correct": False}
      ],
      "explanation": "SemanticAnnotations encode semantic meaning (e.g., Apple, Container) on top of kinematic entities."
    },
    {
      "question": "How does the Apple semantic_annotaiton example set a default name?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "In __post_init__, if name is None it uses PrefixedName(body name, class name)", "correct": True},
        {"answer": "It auto-generates a UUID", "correct": False},
        {"answer": "It inherits the world's name", "correct": False},
        {"answer": "It uses getattr to fetch a label", "correct": False}
      ],
      "explanation": "Apple(SemanticAnnotation) sets name in __post_init__ using PrefixedName if none is provided."
    },
    {
      "question": "Which library is used to query for semantic_annotaitons like apples?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Entity Query Language (EQL)", "correct": True},
        {"answer": "SQLAlchemy", "correct": False},
        {"answer": "NetworkX", "correct": False},
        {"answer": "NumPy", "correct": False}
      ],
      "explanation": "EQL constructs queries such as an(entity(let(Apple, world.semantic_annotaitons)))."
    },
    {
      "question": "What does the FruitBox semantic_annotaiton group together in the example?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "A Container and a list of Apple semantic_annotaitons", "correct": True},
        {"answer": "Two Body instances into a single mesh", "correct": False},
        {"answer": "A URDF file and a parser", "correct": False},
        {"answer": "A World and a RayTracer", "correct": False}
      ],
      "explanation": "FruitBox(SemanticAnnotation) aggregates a Container semantic_annotaiton and several Apple semantic_annotaitons."
    },
    {
      "question": "Which factory is used to create a hollow container in the tutorial?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "ContainerFactory", "correct": True},
        {"answer": "DrawerFactory", "correct": False},
        {"answer": "HandleFactory", "correct": False},
        {"answer": "MeshFactory", "correct": False}
      ],
      "explanation": "ContainerFactory is used to create a hollowed out container body and semantic_annotaiton."
    }
]

display_quiz(questions)
```
