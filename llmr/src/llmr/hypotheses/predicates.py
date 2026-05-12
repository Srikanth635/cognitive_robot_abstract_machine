"""Manipulation predicates for pick-and-place and navigation tasks.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from krrood.entity_query_language.predicate import symbolic_function


@symbolic_function
def is_near_by(body1: Any, body2: Any, threshold: float = 0.1) -> bool:
    """True if the Euclidean distance between the two bodies' origins is below threshold.

    Covers:
    - Navigation: robot base near target object / destination
    - Approach phase: end-effector near target object

    :param body1: Any object with ``global_transform.to_np()`` → (4,4) matrix.
    :param body2: Any object with ``global_transform.to_np()`` → (4,4) matrix.
    :param threshold: Maximum allowed distance in metres.
    """
    pos1 = np.asarray(body1.global_transform.to_np())[:3, 3]
    pos2 = np.asarray(body2.global_transform.to_np())[:3, 3]
    return float(np.linalg.norm(pos1 - pos2)) < threshold


@symbolic_function
def is_pregrasp_aligned(
    effector: Any,
    obj: Any,
    dist_threshold: float = 0.1,
    angle_threshold: float = 0.2,
) -> bool:
    """True if the effector is within distance and pointing toward the object.

    Checks two conditions:
    1. Distance between effector origin and object origin < dist_threshold.
    2. Angle between the effector's approach axis (local +Z) and the
       effector-to-object direction vector < angle_threshold (radians).

    Covers:
    - Pre-grasp pose verification before the Grasp phase begins.

    :param effector: End-effector body with ``global_transform.to_np()`` → (4,4).
    :param obj: Target object with ``global_transform.to_np()`` → (4,4).
    :param dist_threshold: Maximum allowed distance in metres (default 0.1 m).
    :param angle_threshold: Maximum allowed angle in radians (default ~11.5°).
    """
    effector_tf = np.asarray(effector.global_transform.to_np())
    obj_tf = np.asarray(obj.global_transform.to_np())

    effector_pos = effector_tf[:3, 3]
    obj_pos = obj_tf[:3, 3]

    to_object = obj_pos - effector_pos
    dist = float(np.linalg.norm(to_object))

    if dist >= dist_threshold:
        return False

    approach_axis = effector_tf[:3, 2]
    to_object_unit = to_object / (dist + 1e-9)
    cos_angle = float(np.dot(approach_axis, to_object_unit))
    angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    return angle < angle_threshold
