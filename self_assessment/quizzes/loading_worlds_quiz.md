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

(loading-worlds-quiz)=
# Loading Worlds — Self Assessment

This page provides a self-check quiz for the tutorial: [](loading-worlds).  
Source: Jupyter quiz. $ $

% NOTE: The lone `$ $` above ensures some math is rendered before the quiz,
% which fixes a known math-rendering quirk inside the quiz widget.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz

questions = [
    {
      "question": "Which class is used to load a world from a URDF file?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "URDFParser", "correct": True},
        {"answer": "WorldLoader", "correct": False},
        {"answer": "RayTracer", "correct": False},
        {"answer": "SceneParser", "correct": False}
      ],
      "explanation": "URDFParser.from_file(...).parse() loads a world from a URDF file."
    },
    {
      "question": "Which helper locates packaged resources (e.g., URDFs) in this repository?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "get_semantic_world_directory_root()", "correct": True},
        {"answer": "find_repo_root()", "correct": False},
        {"answer": "locate_resources()", "correct": False},
        {"answer": "get_data_dir()", "correct": False}
      ],
      "explanation": "The example composes paths via get_semantic_world_directory_root(os.getcwd())."
    },
    {
      "question": "Loaded worlds from files typically contain which information?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Kinematic structure only (no semantic annotations)", "correct": True},
        {"answer": "Full semantic annotations (views)", "correct": False},
        {"answer": "Robot motion plans", "correct": False},
        {"answer": "Texturing information only", "correct": False}
      ],
      "explanation": "The tutorial notes that such worlds are purely kinematic; no semantic annotations are included."
    },
    {
      "question": "Which file formats are mentioned as supported in the tutorial?",
      "type": "multiple_select",
      "answers": [
        {"answer": "URDF", "correct": True},
        {"answer": "MJCF", "correct": True},
        {"answer": "STL", "correct": True},
        {"answer": "FBX", "correct": False}
      ],
      "explanation": "URDF, MJCF and STL are mentioned."
    },
    {
      "question": "What is the correct sequence to parse a URDF into a World?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "URDFParser.from_file(path).parse()", "correct": True},
        {"answer": "World.parse(path)", "correct": False},
        {"answer": "URDFParser(path).world()", "correct": False},
        {"answer": "parse_urdf_to_world(path)", "correct": False}
      ],
      "explanation": "from_file(...).parse() is used."
    }
]

display_quiz(questions)
```
