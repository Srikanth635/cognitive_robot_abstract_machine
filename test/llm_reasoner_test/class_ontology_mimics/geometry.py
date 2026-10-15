"""
Where things are, which designators refer to but whose own fields the ontology leaves
out.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Pose:
    x: float

    y: float
