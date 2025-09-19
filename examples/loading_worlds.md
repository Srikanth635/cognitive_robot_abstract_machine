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

(loading-worlds)=
# Loading worlds from files

This tutorial shows how to load a world description from a file into a `World` object using the `URDFParser`.

First, we need to compose the path to your world file.

```{code-cell} ipython2
import logging
import os

from semantic_world.utils import get_semantic_world_directory_root

logging.disable(logging.CRITICAL)
apartment = os.path.join(get_semantic_world_directory_root(os.getcwd()), "resources", "urdf", "apartment.urdf")

```

Next we need to initialize a parser that reads this file. There are many parsers available.

```{code-cell} ipython2
from semantic_world.adapters.urdf import URDFParser

parser = URDFParser.from_file(apartment)
world = parser.parse()

from semantic_world.world import visualize_current_world_snapshot
visualize_current_world_snapshot(world)
```

This constructs a world you can visualize, interact and annotate. Be aware that worlds loaded from files have no semantic annotations and serve as purely kinematic models.
Supported file formates are:
- urdf
- mjcf
- stl
