from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING

import numpy as np
from line_profiler import profile

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import VariableParameters

if TYPE_CHECKING:
    import scipy.sparse as sp


@dataclass
class QPDataInequalityOnly:
    """
    min_x 0.5 x^T H x + g^T x
    s.t.  lbA <= Ax <= ubA
    """

    quadratic_weights: np.ndarray
    linear_weights: np.ndarray

    neq_matrix: sp.csc_matrix
    neq_lower_bounds: np.ndarray
    neq_upper_bounds: np.ndarray

    def filter_zero_weight_constraints(self): ...
