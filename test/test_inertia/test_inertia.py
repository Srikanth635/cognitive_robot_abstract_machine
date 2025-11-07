import numpy as np
import pytest
from numpy.testing import assert_allclose

from semantic_digital_twin.world_description.inertia_types import (
    MomentsOfInertia,
    ProductsOfInertia,
    InertiaTensor,
    PrincipalMoments,
    PrincipalAxes,
    NPVector3,
)

class TestComponentsAndAssembly:
    def test_moments_as_matrix_properties(self):
        m = MomentsOfInertia.from_values(2.0, 3.0, 5.0)
        M = m.as_matrix().data
        assert_allclose(M, np.diag([2.0, 3.0, 5.0]), atol=1e-12)
        assert np.allclose(M, M.T)

    def test_products_as_matrix_properties(self):
        p = ProductsOfInertia.from_values(ixy=0.1, ixz=-0.2, iyz=0.3)
        P = p.as_matrix().data
        # symmetric, zero diagonal, correct off-diagonals
        assert np.allclose(P, P.T)
        assert_allclose(np.diag(P), [0.0, 0.0, 0.0], atol=1e-12)
        assert_allclose([P[0,1], P[0,2], P[1,2]], [0.1, -0.2, 0.3], atol=1e-12)

    def test_from_moments_products_roundtrip(self):
        m = MomentsOfInertia.from_values(2.0, 3.0, 5.0)
        p = ProductsOfInertia.from_values(ixy=0.1, ixz=-0.2, iyz=0.3)
        I = InertiaTensor.from_moments_products(m, p)
        # should equal diagonal + off-diagonal
        expected = np.array([
            [2.0,  0.1, -0.2],
            [0.1,  3.0,  0.3],
            [-0.2, 0.3,  5.0],
        ])
        assert_allclose(I.data, expected, atol=1e-12)

        # decompose to scalar values and rebuild moments/products to reconstruct exactly
        m11, m22, m33, m12, m13, m23 = I.to_inertia_values()
        m2 = MomentsOfInertia.from_values(m11, m22, m33)
        p2 = ProductsOfInertia.from_values(m12, m13, m23)
        I2 = InertiaTensor.from_moments_products(m2, p2)
        assert_allclose(I2.data, I.data, atol=1e-12)

    def test_from_inertia_values_and_to_inertia_values(self):
        # simple sanity: set the 6 unique values and get them back intact
        I = InertiaTensor.from_moments_products(
            MomentsOfInertia.from_values(4.0, 6.0, 7.0),
            ProductsOfInertia.from_values(0.4, -0.7, 0.9),
        )
        vals = I.to_inertia_values()
        assert_allclose(vals, (4.0, 6.0, 7.0, 0.4, -0.7, 0.9), atol=1e-12)

    def test_from_principal_reconstructs_specific_rotation(self):
        # rotation that permutes axes (det=+1), fully deterministic
        R = np.array([[0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])
        m = PrincipalMoments.from_values(2.5, 3.0, 6.0)
        axes = PrincipalAxes(data=R)
        I = InertiaTensor.from_principals(axes, m)

        # Back to principals: eigenvalues must match, axes orthonormal & right-handed
        m2, R2 = I.to_principals()
        assert_allclose(np.sort(m2.data), np.sort(m.data), atol=1e-12)
        assert_allclose(R2.data.T @ R2.data, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R2.data), 1.0, atol=1e-12)

    def test_to_principals_with_zero_products(self):
        # Purely diagonal tensor: principal axes = identity; principal moments = diagonal
        I = InertiaTensor.from_moments_products(
            MomentsOfInertia.from_values(5.0, 2.0, 3.0),
            ProductsOfInertia.from_values(0.0, 0.0, 0.0),
        )
        m, R = I.to_principals()
        assert_allclose(m.data, [5.0, 2.0, 3.0], atol=1e-12)
        assert_allclose(R.data, np.eye(3), atol=1e-12)

    def test_principal_axes_validation_non_orthonormal(self):
        with pytest.raises(AssertionError):
            _ = PrincipalAxes(data=np.array([[1.0, 0.0, 0.0],
                                             [0.0, 0.9, 0.0],
                                             [0.0, 0.0, 1.0]]))

    def test_principal_axes_validation_left_handed(self):
        with pytest.raises(AssertionError):
            _ = PrincipalAxes(data=np.diag([1.0, 1.0, -1.0]))

    def test_principal_moments_nonnegative(self):
        with pytest.raises(AssertionError):
            _ = PrincipalMoments.from_values(1.0, -0.1, 2.0)

    def test_npvector3_from_to_and_as_matrix(self):
        v = NPVector3.from_values(1.0, 2.0, 3.0)
        assert v.to_values() == (1.0, 2.0, 3.0)
        M = v.as_matrix().data
        assert_allclose(M, np.diag([1.0, 2.0, 3.0]), atol=1e-12)

    def test_quaternion_roundtrip(self):
        q = 0.70710678, 0. , 0.70710678, 0.
        R = PrincipalAxes.from_quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        q2 = R.to_quaternion()
        assert_allclose(q, q2, atol=1e-12)